"""DIN pretraining on SETRec splits for EAGER (paper-aligned behavior embeddings).

This script:
- Reads SETRec-style splits: training_dict.npy / validation_dict.npy / testing_dict.npy
- Generates EAGER-compatible instance files under <output_dir>/processed/
- Trains DIN to obtain collaborative/behavior item embeddings (dim=96)
- Saves DIN checkpoints under --save_dir (default: <data_root>/<domain>/)

Notes:
- We only need the item embedding weights; evaluation inside the original repo is extremely expensive.
- The default feature_groups sum to (seq_len-1)=19, matching the repo DIN implementation.
"""

import argparse
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from lib.generate_train_and_test_data import _gen_train_sample
from lib.generate_training_batches import Train_instance
from lib.DIN_trainer import DINTrain


def build_sample_from_user_seqs(user_seqs: dict[int, list[int]]) -> dict[str, np.ndarray]:
    user_ids: list[int] = []
    item_ids: list[int] = []
    timestamps: list[int] = []

    for user_id, seq in user_seqs.items():
        for t, item_id in enumerate(seq):
            user_ids.append(int(user_id))
            item_ids.append(int(item_id))
            timestamps.append(int(t))

    user_ids_arr = np.asarray(user_ids, dtype=np.int64)
    item_ids_arr = np.asarray(item_ids, dtype=np.int64)
    timestamps_arr = np.asarray(timestamps, dtype=np.int64)

    zeros = np.zeros_like(user_ids_arr)
    return {
        "USERID": user_ids_arr,
        "ITEMID": item_ids_arr,
        "CATID": zeros,
        "BEHAV": zeros,
        "TS": timestamps_arr,
    }


def _pad_left(seq: list[int], length: int, pad_value: int = -1) -> list[int]:
    if length <= 0:
        return []
    if len(seq) >= length:
        return seq[-length:]
    return [pad_value] * (length - len(seq)) + seq


def write_instances_file(
    out_path: Path,
    user_histories: dict[int, list[int]],
    user_labels: dict[int, list[int]],
    history_len: int,
    include_only_nonempty_labels: bool,
) -> list[int]:
    """Write EAGER instance file.

    Format per line (compatible with Train_instance.read_*_instances_file):
      user_id|h1,h2,...,h_{history_len}|label1,label2,...

    Returns the ordered list of user_ids written.
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)
    users: list[int] = []

    with out_path.open("w", encoding="utf-8") as f:
        for user_id in sorted(user_histories.keys()):
            labels = user_labels.get(user_id, [])
            if include_only_nonempty_labels and (not labels):
                continue
            history = [int(x) for x in user_histories[user_id]]
            history_fixed = _pad_left(history, history_len, pad_value=-1)
            label_str = ",".join(str(int(x)) for x in labels)
            hist_str = ",".join(str(x) for x in history_fixed)
            f.write(f"{int(user_id)}|{hist_str}|{label_str}\n")
            users.append(int(user_id))

    return users


def parse_int_list(csv: str) -> list[int]:
    parts = [p.strip() for p in csv.split(",") if p.strip()]
    return [int(p) for p in parts]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        choices=["beauty", "toys", "sports", "steam", "amazon23_vg", "microlens_50k", "yelp"],
    )
    parser.add_argument("--data_root", type=str, default="data/setrec_data")
    parser.add_argument("--output_dir", type=str, default="runs/eager_setrec_din")
    parser.add_argument("--seed", type=int, default=2024)

    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--min_seq_len", type=int, default=5)
    parser.add_argument("--train_sample_seg_cnt", type=int, default=10)
    parser.add_argument("--parall", type=int, default=8)
    parser.add_argument("--force_regen", action="store_true")

    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--total_batch_num", type=int, default=29000)
    parser.add_argument("--log_every", type=int, default=100)

    parser.add_argument("--emb_dim", type=int, default=96)
    parser.add_argument("--sample_negative_num", type=int, default=60)
    parser.add_argument("--sum_pooling", action="store_true")
    parser.add_argument(
        "--feature_groups",
        type=str,
        default="5,4,2,2,1,1,1,1,1,1",
        help="Comma-separated groups; must sum to seq_len-1 (default: 19)",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="Where to save DIN_MODEL_*.pt (default: <data_root>/<domain>/)",
    )
    parser.add_argument(
        "--save_steps",
        type=str,
        default="29000",
        help="Comma-separated steps to save checkpoints as DIN_MODEL_<step>.pt",
    )
    parser.add_argument(
        "--promote_step",
        type=int,
        default=29000,
        help="If DIN_MODEL_<promote_step>.pt exists, also copy it to DIN_MODEL.pt",
    )
    parser.add_argument(
        "--force_train",
        action="store_true",
        help="Retrain even if target checkpoint already exists",
    )

    args = parser.parse_args()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    domain_dir = Path(args.data_root) / args.domain
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    processed_dir = out_root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    save_dir = Path(args.save_dir) if args.save_dir else domain_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    save_steps = set(parse_int_list(args.save_steps))
    promote_path = save_dir / f"DIN_MODEL_{int(args.promote_step)}.pt"
    if promote_path.exists() and (not args.force_train):
        dst = save_dir / "DIN_MODEL.pt"
        if (not dst.exists()) or (dst.stat().st_mtime < promote_path.stat().st_mtime):
            shutil.copyfile(str(promote_path), str(dst))
        print(f"[DIN][SETRec] checkpoint exists; skip training: {promote_path}")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DIN pretraining")
    torch.cuda.set_device(0)
    device = "cuda"

    adapter_version = "setrec_v4_full_multitarget_no_inner_holdout_20260303"
    adapter_version_file = processed_dir / "setrec_adapter_version.txt"
    cached_version = adapter_version_file.read_text().strip() if adapter_version_file.exists() else ""
    version_ok = cached_version == adapter_version

    train_instances_prefix = str(processed_dir / "train_instances")
    val_instances_file = str(processed_dir / "validation_instances")
    test_instances_file = str(processed_dir / "test_instances")

    val_users_file = processed_dir / "validation_users.npy"
    test_users_file = processed_dir / "test_users.npy"

    his_maxtix_file = str(processed_dir / "his_maxtix.pt")
    labels_file = str(processed_dir / "labels.pt")

    train_dict: dict[int, list[int]] = np.load(domain_dir / "training_dict.npy", allow_pickle=True).item()
    val_dict: dict[int, list[int]] = np.load(domain_dir / "validation_dict.npy", allow_pickle=True).item()
    test_dict: dict[int, list[int]] = np.load(domain_dir / "testing_dict.npy", allow_pickle=True).item()

    sem_emb_path = domain_dir / f"{args.domain}.emb-t5-tdcb.npy"
    sem_emb = np.load(sem_emb_path, mmap_mode="r")
    item_num = int(sem_emb.shape[0])

    need_train_parts = [Path(f"{train_instances_prefix}_{i}") for i in range(args.train_sample_seg_cnt)]
    need_aux = [Path(val_instances_file), Path(test_instances_file), val_users_file, test_users_file]
    if (not version_ok) or args.force_regen or (not all(p.exists() for p in need_train_parts)) or (not all(p.exists() for p in need_aux)):
        print("[DIN][SETRec] generating instances (FULL multi-target protocol)...")

        train_sample = build_sample_from_user_seqs(train_dict)
        _gen_train_sample(
            train_sample,
            train_instances_prefix,
            train_sample_seg_cnt=args.train_sample_seg_cnt,
            parall=args.parall,
            seq_len=args.seq_len,
            min_seq_len=args.min_seq_len,
            holdout_last_n=0,
        )

        history_len = args.seq_len - 1

        val_histories = {u: train_dict[u] for u in train_dict}
        val_labels = {u: val_dict.get(u, []) for u in train_dict}
        val_users = write_instances_file(
            Path(val_instances_file),
            val_histories,
            val_labels,
            history_len=history_len,
            include_only_nonempty_labels=True,
        )
        np.save(val_users_file, np.asarray(val_users, dtype=np.int64))

        test_histories = {u: train_dict[u] + val_dict.get(u, []) for u in train_dict}
        test_labels = {u: test_dict.get(u, []) for u in train_dict}
        test_users = write_instances_file(
            Path(test_instances_file),
            test_histories,
            test_labels,
            history_len=history_len,
            include_only_nonempty_labels=True,
        )
        np.save(test_users_file, np.asarray(test_users, dtype=np.int64))

        adapter_version_file.write_text(adapter_version)

    feature_groups = parse_int_list(args.feature_groups)
    if sum(feature_groups) != (args.seq_len - 1):
        raise ValueError(
            f"feature_groups must sum to seq_len-1={args.seq_len - 1}; got sum={sum(feature_groups)} groups={feature_groups}"
        )

    train_instances = Train_instance(parall=args.parall)
    training_data, training_labels = train_instances.get_training_data(
        train_instances_prefix,
        args.train_sample_seg_cnt,
        item_num,
        his_maxtix_file,
        labels_file,
    )

    print(
        "[DIN][SETRec] users",
        len(train_dict),
        "items",
        item_num,
        "train_records",
        int(training_data.shape[0]),
        "CUDA_VISIBLE_DEVICES=",
        os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )

    optimizer = lambda params: torch.optim.Adam(params, lr=1e-3, amsgrad=True)

    train_model = DINTrain(
        item_num=item_num,
        sample_negative_num=args.sample_negative_num,
        emb_dim=args.emb_dim,
        device=device,
        sum_pooling=args.sum_pooling,
        feature_groups=feature_groups,
        optimizer=optimizer,
    )

    losses: list[float] = []

    def save_ckpt(step: int) -> Path:
        out = save_dir / f"DIN_MODEL_{step}.pt"
        torch.save(train_model.DINModel, str(out))
        return out

    for batch_x, batch_y in train_instances.generate_training_records(
        training_data,
        training_labels,
        batch_size=args.train_batch_size,
    ):
        loss = train_model.update_DIN(batch_x, batch_y)
        losses.append(float(loss.detach().cpu()))

        step = int(train_model.batch_num)
        if (args.log_every > 0) and (step % args.log_every == 0):
            window = losses[-args.log_every :]
            mean_loss = float(sum(window) / max(len(window), 1))
            print(f"[DIN][SETRec] step={step} mean_loss={mean_loss:.6f}")

        if step in save_steps:
            out = save_ckpt(step)
            print(f"[DIN][SETRec] saved {out}")

        if step >= int(args.total_batch_num):
            break

    last_path = save_dir / "DIN_MODEL_LAST.pt"
    torch.save(train_model.DINModel, str(last_path))
    print(f"[DIN][SETRec] saved {last_path}")

    if promote_path.exists():
        dst = save_dir / "DIN_MODEL.pt"
        shutil.copyfile(str(promote_path), str(dst))
        print(f"[DIN][SETRec] promoted {promote_path} -> {dst}")
    else:
        print(f"[DIN][SETRec] promote_step not saved yet: {promote_path}")


if __name__ == "__main__":
    main()
