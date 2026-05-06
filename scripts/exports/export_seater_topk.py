"""Export SEATER top-k predictions to JSONL for unified_eval.py.

Notes:
- SEATER uses 0 as padding token, so items in its TSV inputs are shifted by +1.
- This exporter converts predicted item ids back to canonical 0-based ids by subtracting 1.

Output JSONL schema:
  {"user_id": <int>, "predicted_items": [<int>, ...]}
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main() -> None:
    ap = argparse.ArgumentParser(description="Export SEATER predictions to JSONL")
    ap.add_argument(
        "--seater_code_dir",
        default="baselines/ref01",
        help="Path to SEATER repo root (contains main.py)",
    )
    ap.add_argument(
        "--dataset_name",
        required=True,
        help="Dataset name in SEATER config, e.g. SETRec_beauty",
    )
    ap.add_argument(
        "--ckpt",
        required=True,
        help="Path to SEATER checkpoint (best.pth)",
    )
    ap.add_argument("--vocab", type=int, default=8, help="Tree branch number")
    ap.add_argument("--gpu_id", type=int, default=0, help="GPU id within CUDA_VISIBLE_DEVICES")
    ap.add_argument("--use_cpu", action="store_true", help="Force CPU")
    ap.add_argument("--k", type=int, default=20, help="Top-k to export")
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pred_out", required=True)

    args = ap.parse_args()

    set_seed(args.seed)

    seater_code_dir = Path(args.seater_code_dir).resolve()
    sys.path.insert(0, str(seater_code_dir))

    import yaml  # noqa: E402

    from utils.Context import DatasetManager  # noqa: E402
    from utils import data as data_mod  # noqa: E402
    from model import SEATER  # noqa: E402

    # Minimal flags object for DatasetManager
    class _Flags:
        pass

    flags = _Flags()
    flags.dataset_name = args.dataset_name
    flags.batch_size = 1
    flags.test_batch_size = args.batch_size
    flags.num_workers = args.num_workers

    dm = DatasetManager(flags)

    cfg_path = seater_code_dir / "config" / args.dataset_name / "SEATER.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"SEATER.yaml not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)

    model_config = dm.set_dataset_related_hyparam(model_config)

    dm.num_neg = int(model_config.get("rk_num_neg", 0))

    # Match trainer behavior: append vocab subtree dir
    dm.tree_data_par_path = os.path.join(dm.tree_data_par_path, f"{args.vocab}_branch_tree")

    test_dataset = data_mod.SEATER_Dataset(dm, mode="test")

    # Reset decoder index config (trainer does this after building dataloader)
    model_config["decoder_index"]["vocab_size"] = int(args.vocab)
    model_config["decoder_index"]["tree_nodes_num"] = int(test_dataset.tree_nodes_num)
    model_config["decoder_index"]["max_len"] = int(test_dataset.max_len)

    device = torch.device("cpu")
    if not args.use_cpu:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.gpu_id}")

    model = SEATER(model_config).to(device)
    model._init_prefix_mask(test_dataset.prefix_allowed_token, device)

    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state)
    model.eval()

    test_loader = data_mod.get_dataloader(
        data_set=test_dataset,
        bs=int(args.batch_size),
        num_workers=int(args.num_workers),
        prefetch_factor=(int(args.batch_size) // int(args.num_workers) + 1) if int(args.num_workers) != 0 else None,
        shuffle=False,
    )

    pred_out = Path(args.pred_out)
    pred_out.parent.mkdir(parents=True, exist_ok=True)

    expected_users = len(test_dataset)
    max_item_canonical = int(dm.item_ID_num) - 2

    seen = set()
    written = 0

    with torch.no_grad(), pred_out.open("w", encoding="utf-8") as f:
        for batch in test_loader:
            uid, seq = batch
            uid = uid.to(device)
            seq = seq.to(device)

            _, pred_shifted = model.predict_step((uid, seq), topK=int(args.k))
            pred_shifted = pred_shifted.detach().cpu().numpy()
            uid_np = uid.detach().cpu().numpy()

            for u, preds in zip(uid_np.tolist(), pred_shifted.tolist()):
                u = int(u)
                if u in seen:
                    raise RuntimeError(f"duplicate user_id in test loader: {u}")
                seen.add(u)

                preds_canon = [int(x) - 1 for x in preds]

                if len(preds_canon) < 10:
                    raise RuntimeError(f"predicted_items too short for user_id={u}: {len(preds_canon)}")
                if min(preds_canon) < 0:
                    raise RuntimeError(f"negative item id after shift for user_id={u}: {min(preds_canon)}")
                if max(preds_canon) > max_item_canonical:
                    raise RuntimeError(
                        f"item id out of range after shift for user_id={u}: max={max(preds_canon)} expected<= {max_item_canonical}"
                    )

                obj = {"user_id": u, "predicted_items": preds_canon}
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written += 1

    if written != expected_users:
        raise RuntimeError(f"coverage mismatch: wrote={written} expected_users={expected_users}")

    print(f"[export_seater] wrote users={written} pred_out={pred_out}")


if __name__ == "__main__":
    main()
