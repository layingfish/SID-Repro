"""Export SETRec (T5) top-k predictions to JSONL for unified_eval.py.

Key properties:
- Exports canonical 0-based item ids.
- Covers ALL users with non-empty test items.
- Selects beta on VALIDATION split (valData) to avoid test leakage.

Expected inputs:
- --ckpt_dir: finetune_t5.py output_dir containing adapter.pth and saved T5 model.
- --data_path: processed dataset dir containing *dict.npy, warm/cold_item.npy, SASRec_item_embed.pkl, and <domain>.emb-*.npy.

Output JSONL schema:
  {"user_id": <int>, "predicted_items": [<int>, ...]}
"""

import argparse
import json
import pickle
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


def parse_beta_grid(s: str) -> list[float]:
    s = (s or "").strip()
    if not s:
        return []
    out: list[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def flatten_topk_list(x):
    if isinstance(x, list) and len(x) == 1 and isinstance(x[0], list):
        return x[0]
    return x


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SETRec(T5) predictions to JSONL")
    parser.add_argument(
        "--setrec_code_dir",
        default="baselines/reference/code",
        help="Path to SETRec code directory",
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help="Dataset dir containing *dict.npy and item embedding files",
    )
    parser.add_argument(
        "--ckpt_dir",
        required=True,
        help="finetune_t5.py output_dir containing adapter.pth",
    )
    parser.add_argument(
        "--cache_dir",
        default="models/hf/transformers",
        help="HF cache dir (must contain t5-small)",
    )
    parser.add_argument("--prompt_template_name", default="template")
    parser.add_argument("--seed", type=int, default=42)

    # identifier / tokenizer args
    parser.add_argument("--n_cf", type=int, default=1)
    parser.add_argument("--n_sem", type=int, required=True)
    parser.add_argument("--n_query", type=int, default=0, help="0 => n_sem + 1")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--dropout_prob", type=float, default=0.0)
    parser.add_argument("--bn", type=int, default=0)
    parser.add_argument("--loss_type", type=str, default="mse")
    parser.add_argument("--sem_encoder", type=str, default="t5")

    # export args
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument(
        "--beta_grid",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated beta candidates (used on validation)",
    )
    parser.add_argument(
        "--max_val_users",
        type=int,
        default=0,
        help="0 => use all validation users; else sample this many users for beta selection",
    )
    parser.add_argument("--pred_out", required=True, help="Output JSONL path")
    parser.add_argument("--beta_out", default=None, help="Optional path to save selected beta JSON")
    args = parser.parse_args()

    set_seed(args.seed)

    setrec_code_dir = Path(args.setrec_code_dir)
    sys.path.insert(0, str(setrec_code_dir))

    from utils.prompter import Prompter
    from model_t5 import T54Rec
    from utils.data_utils import SequentialDataset
    from utils.eval_utils import computeTopNAccuracy

    data_path_raw = Path(args.data_path)
    prefix = data_path_raw.name or data_path_raw.parent.name
    # Keep symlink name as prefix; avoid Path.resolve() changing suffix like *_k5.
    data_path = str(data_path_raw.expanduser().absolute())
    if not data_path.endswith("/"):
        data_path = data_path + "/"

    ckpt_dir = Path(args.ckpt_dir).resolve()
    adapter_path = ckpt_dir / "adapter.pth"
    if not adapter_path.exists():
        raise FileNotFoundError(f"adapter.pth not found: {adapter_path}")

    # prefix derived from args.data_path (symlink-safe)
    item_embed_path = Path(data_path) / "SASRec_item_embed.pkl"
    feat_path = Path(data_path) / f"{prefix}.emb-{args.sem_encoder}-tdcb.npy"

    if not item_embed_path.exists():
        raise FileNotFoundError(f"SASRec_item_embed.pkl not found: {item_embed_path}")
    if not feat_path.exists():
        raise FileNotFoundError(f"semantic embedding not found: {feat_path}")

    item_embed = pickle.load(open(item_embed_path, "rb"))
    item_feature = torch.FloatTensor(np.load(str(feat_path), allow_pickle=True))

    n_query = args.n_query if args.n_query and args.n_query > 0 else (args.n_sem + 1)

    prompter = Prompter(args.prompt_template_name)
    dataset = SequentialDataset(data_path, 50, n_query, args.n_sem)

    # build model from checkpoint dir (contains saved T5 model + tokenizer)
    device_map = "auto"
    tokenizer_args = {
        "in_dim": item_feature.shape[-1],
        "layers": args.layers,
        "dropout_prob": args.dropout_prob,
        "bn": bool(args.bn),
        "loss_type": args.loss_type,
        "item_feature": item_feature,
    }

    model = T54Rec(
        base_model=str(ckpt_dir),
        input_embeds=item_embed,
        cache_dir=args.cache_dir,
        device_map=device_map,
        input_dim=64,
        instruction_text=prompter.generate_prompt(),
        user_embeds=None,
        m_item=dataset.m_item,
        n_query=n_query,
        n_cf=args.n_cf,
        n_sem=args.n_sem,
        alpha=args.alpha,
        inference=True,
        **tokenizer_args,
    )

    state = torch.load(str(adapter_path), map_location="cpu")
    model.tokenizer.load_state_dict(state["tokenizer"])
    model.input_proj.load_state_dict(state["input_proj"])
    model.input_embeds.load_state_dict(state["input_embeds"])

    model = model.cuda()
    model.eval()

    # Precompute item representations once
    idx_tensor = torch.arange(model.m_item, dtype=torch.long).cuda()
    model.all_cf = model.input_proj(model.input_embeds[0](idx_tensor + 1)).unsqueeze(0)
    if model.n_sem:
        model.recon_all = model.tokenize_all()

    # Select beta on validation split
    beta_grid = parse_beta_grid(args.beta_grid)
    if not beta_grid:
        beta_grid = [0.0] if args.n_sem == 0 else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    val_items = list(dataset.valData.items())
    if args.max_val_users and args.max_val_users > 0:
        random.shuffle(val_items)
        val_items = val_items[: args.max_val_users]

    best_beta = beta_grid[0]
    best_recall5 = -1.0

    topk_eval = [5, 10]

    with torch.no_grad():
        for beta in beta_grid if model.n_sem else [0.0]:
            if model.n_sem:
                w = torch.tensor([1.0 - beta] + [beta] * args.n_sem, dtype=torch.float, requires_grad=False)
            else:
                w = torch.tensor([1.0], dtype=torch.float, requires_grad=False)
            model.beta = torch.nn.Parameter(w)

            gold_list = []
            pred_list = []

            for uid, (train_seq, val_item) in val_items:
                if len(train_seq) == 0:
                    continue
                seq = [int(x) + 1 for x in train_seq]
                inputs = torch.LongTensor(seq).cuda().unsqueeze(0)
                inputs_mask = torch.ones((inputs.shape[0], inputs.shape[1] * n_query)).cuda()
                _, ratings, _, _, _ = model.predict(inputs, inputs_mask, inference=True)
                _, pred = torch.topk(ratings, k=topk_eval[-1])
                pred = flatten_topk_list(pred.cpu().tolist())

                gold_list.append([int(val_item)])
                pred_list.append([int(x) for x in pred])

            test_results = computeTopNAccuracy(gold_list, pred_list, topk_eval)
            recall5 = float(test_results[1][0])
            if recall5 > best_recall5:
                best_recall5 = recall5
                best_beta = beta

    print(f"[export_setrec] selected beta={best_beta} (val Recall@5={best_recall5})")

    # Export test predictions for ALL test users
    pred_out = Path(args.pred_out)
    pred_out.parent.mkdir(parents=True, exist_ok=True)

    if model.n_sem:
        w = torch.tensor([1.0 - best_beta] + [best_beta] * args.n_sem, dtype=torch.float, requires_grad=False)
    else:
        w = torch.tensor([1.0], dtype=torch.float, requires_grad=False)
    model.beta = torch.nn.Parameter(w)

    # refresh cached sem (cheap) for safety
    if model.n_sem:
        model.recon_all = model.tokenize_all()

    written = 0
    with torch.no_grad(), pred_out.open("w", encoding="utf-8") as f:
        for uid, (seq_shifted, gt_items) in dataset.testData.items():
            if len(gt_items) == 0:
                continue
            seq = [int(x) for x in seq_shifted]
            inputs = torch.LongTensor(seq).cuda().unsqueeze(0)
            inputs_mask = torch.ones((inputs.shape[0], inputs.shape[1] * n_query)).cuda()
            _, ratings, _, _, _ = model.predict(inputs, inputs_mask, inference=True)
            _, pred = torch.topk(ratings, k=args.k)
            pred = flatten_topk_list(pred.cpu().tolist())

            obj = {
                "user_id": int(uid),
                "predicted_items": [int(x) for x in pred],
            }
            f.write(json.dumps(obj) + "\n")
            written += 1

    print(f"[export_setrec] wrote users={written} pred_out={pred_out}")

    if args.beta_out:
        beta_out = Path(args.beta_out)
        beta_out.parent.mkdir(parents=True, exist_ok=True)
        beta_out.write_text(json.dumps({"beta": best_beta, "val_recall5": best_recall5}, indent=2), encoding="utf-8")
        print(f"[export_setrec] wrote beta_out={beta_out}")


if __name__ == "__main__":
    main()
