#!/usr/bin/env python3
"""Capacity and teacher-forcing diagnostics for SID length scaling.

This script explains why a certain SID length works best by separating:

1. Tokenizer capacity: how much item ambiguity remains in each true prefix
   bucket, independent of the decoder.
2. Decoder learnability: how hard each SID position is under teacher forcing,
   using the already trained decoder checkpoints.

The diagnostics are inference-only and do not retrain tokenizers or decoders.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


DEFAULT_METHODS = ("rqvae", "rqkmeans", "opq")
DEFAULT_LENGTHS = (2, 3, 4, 6, 8, 12, 16)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _load_npy_obj(path: Path):
    arr = np.load(path, allow_pickle=True)
    try:
        return arr.item()
    except Exception:
        return arr.tolist()


def _coerce_items(items) -> list[int]:
    if isinstance(items, np.ndarray):
        values = items.tolist()
    elif isinstance(items, (list, tuple, set)):
        values = list(items)
    elif items is None:
        values = []
    else:
        values = [items]
    return [int(x) for x in values if int(x) >= 0]


def _load_warm_gt(data_dir: Path, dataset: str) -> dict[int, list[int]]:
    test = _load_npy_obj(data_dir / dataset / "testing_dict.npy")
    warm_raw = _load_npy_obj(data_dir / dataset / "warm_item.npy")
    warm_items = set(int(x) for x in (warm_raw.keys() if hasattr(warm_raw, "keys") else warm_raw))
    gt: dict[int, list[int]] = {}
    for uid, items in test.items():
        kept = [item for item in _coerce_items(items) if item in warm_items]
        if kept:
            gt[int(uid)] = kept
    return gt


def _load_seen_items(data_dir: Path, dataset: str) -> dict[int, set[int]]:
    raw = _load_npy_obj(data_dir / dataset / "training_dict.npy")
    return {int(uid): set(_coerce_items(items)) for uid, items in raw.items()}


def _item_popularity(data_dir: Path, dataset: str, n_items: int) -> np.ndarray:
    pop = np.zeros(n_items, dtype=np.int64)
    raw = _load_npy_obj(data_dir / dataset / "training_dict.npy")
    for items in raw.values():
        for item in _coerce_items(items):
            if 0 <= item < n_items:
                pop[item] += 1
    return pop


def _sort_items(items: Iterable[int], popularity: np.ndarray) -> list[int]:
    return sorted((int(x) for x in items), key=lambda x: (-int(popularity[x]), x))


def _build_groups(codes: np.ndarray, k: int, popularity: np.ndarray) -> dict[tuple[int, ...], list[int]]:
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for item_id, code in enumerate(codes):
        groups[tuple(int(x) for x in code[:k])].append(int(item_id))
    return {key: _sort_items(items, popularity) for key, items in groups.items()}


def _gini_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    sorted_counts = np.sort(counts)
    n = sorted_counts.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(idx * sorted_counts)) / (n * total) - (n + 1.0) / n)


def _usage_rows(
    method: str,
    length: int,
    codes: np.ndarray,
    per_pos_sizes: list[int],
    semantic_len: int,
) -> list[dict]:
    rows: list[dict] = []
    n_items = int(codes.shape[0])
    for pos in range(codes.shape[1]):
        vocab_size = int(per_pos_sizes[pos]) if pos < len(per_pos_sizes) else int(codes[:, pos].max()) + 1
        vocab_size = max(vocab_size, int(codes[:, pos].max()) + 1)
        counts = np.bincount(codes[:, pos], minlength=vocab_size).astype(np.float64)
        used = int(np.count_nonzero(counts))
        probs = counts[counts > 0] / max(float(counts.sum()), 1.0)
        entropy = float(-np.sum(probs * np.log(probs))) if probs.size else 0.0
        effective = float(math.exp(entropy)) if entropy > 0 else 0.0
        rows.append(
            {
                "method": method,
                "length": length,
                "position": pos + 1,
                "is_suffix": pos >= semantic_len,
                "vocab_size": vocab_size,
                "used_tokens": used,
                "usage_rate": used / vocab_size if vocab_size else 0.0,
                "entropy": entropy,
                "normalized_entropy": entropy / math.log(vocab_size) if vocab_size > 1 else 0.0,
                "effective_tokens": effective,
                "effective_usage_rate": effective / vocab_size if vocab_size else 0.0,
                "gini": _gini_from_counts(counts),
                "top1_mass": float(np.max(counts) / n_items) if n_items else 0.0,
            }
        )
    return rows


def _capacity_rows(
    *,
    method: str,
    length: int,
    codes: np.ndarray,
    semantic_len: int,
    popularity: np.ndarray,
    gt: dict[int, list[int]],
    seen_by_user: dict[int, set[int]],
    top_ks: tuple[int, ...],
) -> list[dict]:
    rows: list[dict] = []
    n_items = int(codes.shape[0])
    prefix_lengths = list(range(1, semantic_len + 1))
    if codes.shape[1] != semantic_len:
        prefix_lengths.append(int(codes.shape[1]))

    for k in prefix_lengths:
        groups = _build_groups(codes, k, popularity)
        group_sizes = np.array([len(v) for v in groups.values()], dtype=np.float64)
        user_recalls = {top_k: [] for top_k in top_ks}
        ranks: list[int] = []
        bucket_sizes: list[int] = []
        missing = 0

        for uid, items in gt.items():
            seen = seen_by_user.get(int(uid), set())
            per_user_hits = {top_k: 0 for top_k in top_ks}
            denom = 0
            for item in items:
                item = int(item)
                if item < 0 or item >= n_items:
                    continue
                key = tuple(int(x) for x in codes[item, :k])
                bucket = [x for x in groups.get(key, []) if x not in seen]
                bucket_sizes.append(len(bucket))
                denom += 1
                try:
                    rank = bucket.index(item) + 1
                    ranks.append(rank)
                except ValueError:
                    missing += 1
                    rank = 10**9
                for top_k in top_ks:
                    if rank <= top_k:
                        per_user_hits[top_k] += 1
            if denom:
                for top_k in top_ks:
                    user_recalls[top_k].append(per_user_hits[top_k] / denom)

        row = {
            "method": method,
            "length": length,
            "prefix_len": k,
            "prefix_type": "full_code" if k == codes.shape[1] and k != semantic_len else "semantic_prefix",
            "num_groups": len(groups),
            "collision_rate": (n_items - len(groups)) / max(n_items, 1),
            "mean_group_size": float(np.mean(group_sizes)) if group_sizes.size else 0.0,
            "median_group_size": float(np.median(group_sizes)) if group_sizes.size else 0.0,
            "p90_group_size": float(np.percentile(group_sizes, 90)) if group_sizes.size else 0.0,
            "max_group_size": int(np.max(group_sizes)) if group_sizes.size else 0,
            "mean_gt_bucket_size": float(np.mean(bucket_sizes)) if bucket_sizes else 0.0,
            "median_gt_bucket_size": float(np.median(bucket_sizes)) if bucket_sizes else 0.0,
            "p90_gt_bucket_size": float(np.percentile(bucket_sizes, 90)) if bucket_sizes else 0.0,
            "mean_gt_rank_in_bucket": float(np.mean(ranks)) if ranks else 0.0,
            "median_gt_rank_in_bucket": float(np.median(ranks)) if ranks else 0.0,
            "missing_gt_items": missing,
        }
        for top_k in top_ks:
            vals = user_recalls[top_k]
            row[f"oracle_bucket_recall@{top_k}"] = float(np.mean(vals)) if vals else 0.0
        rows.append(row)
    return rows


def _maybe_strip_module_prefix(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict
    if all(isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    if all(isinstance(k, str) and k.startswith("_orig_mod.") for k in state_dict.keys()):
        return {k[len("_orig_mod."):]: v for k, v in state_dict.items()}
    return state_dict


def _infer_num_user_tokens(*, hf_model_path: str, sid_tokens: int, ckpt_vocab_size: int):
    from transformers import AutoTokenizer

    base_tokenizer = AutoTokenizer.from_pretrained(
        hf_model_path,
        local_files_only=("/" in hf_model_path),
        use_fast=True,
    )
    base_len = len(base_tokenizer)
    for candidate in range(0, 100000):
        new_vocab = base_len + sid_tokens + candidate
        if new_vocab == ckpt_vocab_size:
            return candidate
        if new_vocab > ckpt_vocab_size:
            return max(0, candidate - 1)
    return 0


def _filter_eval_dataset(eval_dataset, required_users: set[int], max_users: int) -> None:
    seq_user_ids = eval_dataset.sequence_data.get("rawUserId")
    if seq_user_ids is None:
        seq_user_ids = eval_dataset.sequence_data.get("userId")
    if seq_user_ids is None:
        raise KeyError("SeqData.sequence_data contains neither rawUserId nor userId")
    if isinstance(seq_user_ids, torch.Tensor):
        uid0 = seq_user_ids.detach().to(torch.long).cpu().reshape(-1) - 1
        keep_mask = torch.tensor([int(x) in required_users for x in uid0.tolist()], dtype=torch.bool)
    else:
        keep_mask = torch.tensor([int(x) - 1 in required_users for x in list(seq_user_ids)], dtype=torch.bool)
    keep_idx = keep_mask.nonzero(as_tuple=False).reshape(-1)
    n_before = len(eval_dataset)
    if max_users and max_users > 0:
        keep_idx = keep_idx[:max_users]
        keep_mask2 = torch.zeros_like(keep_mask)
        keep_mask2[keep_idx] = True
        keep_mask = keep_mask2
    for key, value in list(eval_dataset.sequence_data.items()):
        if isinstance(value, torch.Tensor) and value.shape[0] == n_before:
            eval_dataset.sequence_data[key] = value.index_select(0, keep_idx.to(value.device))
        elif isinstance(value, np.ndarray) and value.shape[0] == n_before:
            eval_dataset.sequence_data[key] = value[keep_mask.numpy()]
        elif isinstance(value, list) and len(value) == n_before:
            mask_list = keep_mask.tolist()
            eval_dataset.sequence_data[key] = [x for x, keep in zip(value, mask_list) if keep]


def _teacher_forcing_rows(
    *,
    method: str,
    length: int,
    cached_ids_path: Path,
    tiger_config_path: Path,
    decoder_ckpt: Path,
    hf_model_path: str,
    dataset_folder: str,
    export_domain: str,
    gt_users: set[int],
    batch_size: int,
    gpu: int,
    max_users: int,
) -> tuple[list[dict], dict]:
    root = Path("/data/xqp_data/RecSys26")
    tiger_root = root / "third_party" / "RQ_VAE_Recommender"
    scaling_root = root / "scripts" / "scaling"
    scripts_root = root / "scripts"
    for p in (str(tiger_root), str(scaling_root), str(scripts_root)):
        if p not in sys.path:
            sys.path.insert(0, p)

    from manifest_utils import patch_t5_semid_for_manifest, resolve_sid_layout

    tiger_config = _load_json(tiger_config_path)
    sid_layout = resolve_sid_layout(tiger_config=tiger_config, cached_ids_path=cached_ids_path)
    per_pos_sizes = [int(x) for x in sid_layout["per_pos_sizes"]]
    sem_id_dim = int(sid_layout["sem_id_dim"])
    semantic_len = int(tiger_config.get("semantic_code_length", sem_id_dim))
    codebook_size = int(sid_layout["max_codebook_size"])
    sid_tokens = int(sid_layout["n_sid_tokens"])
    duplicate_policy = sid_layout.get("duplicate_policy", tiger_config.get("duplicate_policy", "first"))

    patch_t5_semid_for_manifest(per_pos_sizes=per_pos_sizes, duplicate_policy=duplicate_policy)

    from cached_id_tokenizer import CachedIdTokenizer
    from data.processed import RecDataset, SeqData
    from data.utils import batch_to
    from modules.tokenizer.t5_semid import build_t5_tokenizer_and_model, make_t5_inputs

    ckpt = torch.load(str(decoder_ckpt), map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    ckpt_vocab_size = int(state["encoder.embed_tokens.weight"].shape[0])
    num_user_tokens = _infer_num_user_tokens(
        hf_model_path=hf_model_path,
        sid_tokens=sid_tokens,
        ckpt_vocab_size=ckpt_vocab_size,
    )

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() and gpu >= 0 else "cpu")
    hf_tokenizer, model, sid_token_ids, uid_token_ids = build_t5_tokenizer_and_model(
        hf_model_path=hf_model_path,
        codebook_size=codebook_size,
        sem_id_dim=sem_id_dim,
        num_user_tokens=num_user_tokens,
        local_files_only=("/" in hf_model_path),
    )
    model.load_state_dict(_maybe_strip_module_prefix(state))
    del ckpt, state
    model = model.to(device)
    model.eval()

    eval_dataset = SeqData(
        root=dataset_folder,
        dataset=RecDataset.AMAZON,
        is_train=False,
        subsample=False,
        split=export_domain,
    )
    _filter_eval_dataset(eval_dataset, gt_users, max_users=max_users)
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    semid_tokenizer = CachedIdTokenizer(
        cached_ids_path=str(cached_ids_path.resolve()),
        codebook_size=codebook_size,
        per_pos_sizes=per_pos_sizes,
    ).to(device)
    pad_token_id = int(hf_tokenizer.pad_token_id)

    nll_sum = torch.zeros(sem_id_dim, dtype=torch.float64)
    acc_sum = torch.zeros(sem_id_dim, dtype=torch.float64)
    top5_sum = torch.zeros(sem_id_dim, dtype=torch.float64)
    counts = torch.zeros(sem_id_dim, dtype=torch.float64)
    prefix_exact_sum = torch.zeros(sem_id_dim, dtype=torch.float64)
    seq_count = 0.0

    desc = f"TF {method} L{length}"
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            data = batch_to(batch, device)
            tokenized = semid_tokenizer(data)
            input_ids, attention_mask, labels = make_t5_inputs(
                tokenized,
                sid_token_ids=sid_token_ids,
                uid_token_ids=uid_token_ids,
                pad_token_id=pad_token_id,
            )
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            labels = labels.to(torch.long)
            losses = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="none",
            ).reshape(labels.shape)
            pred = logits.argmax(dim=-1)
            top5 = logits.topk(k=min(5, logits.shape[-1]), dim=-1).indices
            valid = labels >= 0
            correct = (pred == labels) & valid
            top5_correct = (top5 == labels.unsqueeze(-1)).any(dim=-1) & valid

            nll_sum += (losses * valid).detach().sum(dim=0).cpu().double()
            acc_sum += correct.detach().sum(dim=0).cpu().double()
            top5_sum += top5_correct.detach().sum(dim=0).cpu().double()
            counts += valid.detach().sum(dim=0).cpu().double()

            prefix_correct = torch.cumprod(correct.to(torch.long), dim=1).to(torch.float64)
            prefix_exact_sum += prefix_correct.detach().sum(dim=0).cpu().double()
            seq_count += float(labels.shape[0])

    rows: list[dict] = []
    for pos in range(sem_id_dim):
        c = max(float(counts[pos].item()), 1.0)
        nll = float(nll_sum[pos].item() / c)
        rows.append(
            {
                "method": method,
                "length": length,
                "position": pos + 1,
                "is_suffix": pos >= semantic_len,
                "count": int(counts[pos].item()),
                "nll": nll,
                "ppl": float(math.exp(min(nll, 50.0))),
                "token_acc": float(acc_sum[pos].item() / c),
                "token_top5_acc": float(top5_sum[pos].item() / c),
                "prefix_exact_acc": float(prefix_exact_sum[pos].item() / max(seq_count, 1.0)),
            }
        )

    summary = {
        "method": method,
        "length": length,
        "num_eval_sequences": int(seq_count),
        "semantic_len": semantic_len,
        "sem_id_dim": sem_id_dim,
        "num_user_tokens": int(num_user_tokens),
        "mean_semantic_nll": float(np.mean([r["nll"] for r in rows if not r["is_suffix"]])),
        "mean_semantic_acc": float(np.mean([r["token_acc"] for r in rows if not r["is_suffix"]])),
        "last_semantic_nll": float(rows[semantic_len - 1]["nll"]),
        "last_semantic_acc": float(rows[semantic_len - 1]["token_acc"]),
        "full_prefix_exact_acc": float(rows[-1]["prefix_exact_acc"]),
    }
    if sem_id_dim > semantic_len:
        suffix_rows = [r for r in rows if r["is_suffix"]]
        summary["suffix_mean_nll"] = float(np.mean([r["nll"] for r in suffix_rows]))
        summary["suffix_mean_acc"] = float(np.mean([r["token_acc"] for r in suffix_rows]))
    return rows, summary


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study_root", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    ap.add_argument("--lengths", default=",".join(str(x) for x in DEFAULT_LENGTHS))
    ap.add_argument("--data_dir", type=Path, default=Path("/data/xqp_data/RecSys26/third_party/SETRec/data"))
    ap.add_argument("--eval_dataset", default="amazon23_vg")
    ap.add_argument("--dataset_folder", default="/data/xqp_data/RecSys26/datasets/rqvae_recommender/setrec")
    ap.add_argument("--export_domain", default="amazon23_vg_tiger_strict")
    ap.add_argument("--hf_model_path", default="/data/xqp_data/RecSys26/models/hf/transformers/models--t5-small/snapshots/df1b051c49625cf57a3d0d8d3863ed4d13564fe4")
    ap.add_argument("--top_ks", default="5,10")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max_users", type=int, default=0)
    ap.add_argument("--skip_teacher", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    top_ks = tuple(int(x) for x in args.top_ks.split(",") if x.strip())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gt = _load_warm_gt(args.data_dir, args.eval_dataset)
    seen_by_user = _load_seen_items(args.data_dir, args.eval_dataset)
    gt_users = set(gt.keys())

    usage_rows: list[dict] = []
    capacity_rows: list[dict] = []
    teacher_rows: list[dict] = []
    teacher_summary_rows: list[dict] = []
    manifest: dict[str, dict] = {}

    for method in methods:
        manifest[method] = {}
        for length in lengths:
            tokenizer_dir = args.study_root / "tokenizer_suffix" / method / f"L{length}"
            decoder_dir = args.study_root / "decoder_mainalign_t5small" / method / f"L{length}"
            cached_ids_path = tokenizer_dir / "cached_ids.npy"
            tiger_config_path = tokenizer_dir / "tiger_config.json"
            decoder_ckpt = decoder_dir / "decoder" / "checkpoint_4999.pt"
            metrics_path = decoder_dir / "metrics_full_warm_only.json"
            if not cached_ids_path.exists() or not tiger_config_path.exists():
                print(f"[skip] missing tokenizer {method} L{length}", flush=True)
                continue

            tiger_config = _load_json(tiger_config_path)
            codes = np.load(cached_ids_path).astype(np.int64)
            semantic_len = int(tiger_config.get("semantic_code_length", codes.shape[1]))
            per_pos_sizes = tiger_config.get("per_pos_sizes") or tiger_config.get("codebook_size_per_level")
            if per_pos_sizes is None:
                per_pos_sizes = [int(tiger_config["codebook_size"])] * codes.shape[1]
            per_pos_sizes = [int(x) for x in per_pos_sizes]
            popularity = _item_popularity(args.data_dir, args.eval_dataset, codes.shape[0])

            usage_rows.extend(_usage_rows(method, length, codes, per_pos_sizes, semantic_len))
            capacity_rows.extend(
                _capacity_rows(
                    method=method,
                    length=length,
                    codes=codes,
                    semantic_len=semantic_len,
                    popularity=popularity,
                    gt=gt,
                    seen_by_user=seen_by_user,
                    top_ks=top_ks,
                )
            )

            full_metrics = _load_json(metrics_path) if metrics_path.exists() else {}
            manifest[method][f"L{length}"] = {
                "tokenizer_dir": str(tokenizer_dir),
                "decoder_dir": str(decoder_dir),
                "semantic_len": semantic_len,
                "sem_id_dim": int(codes.shape[1]),
                "full_metrics": {k: full_metrics.get(k) for k in ("Recall@5", "NDCG@5", "Recall@10", "NDCG@10")},
            }

            if not args.skip_teacher:
                if not decoder_ckpt.exists():
                    print(f"[skip] missing decoder {method} L{length}", flush=True)
                    continue
                rows, summary = _teacher_forcing_rows(
                    method=method,
                    length=length,
                    cached_ids_path=cached_ids_path,
                    tiger_config_path=tiger_config_path,
                    decoder_ckpt=decoder_ckpt,
                    hf_model_path=args.hf_model_path,
                    dataset_folder=args.dataset_folder,
                    export_domain=args.export_domain,
                    gt_users=gt_users,
                    batch_size=args.batch_size,
                    gpu=args.gpu,
                    max_users=args.max_users,
                )
                teacher_rows.extend(rows)
                summary.update({f"full_{k}": v for k, v in manifest[method][f"L{length}"]["full_metrics"].items()})
                teacher_summary_rows.append(summary)

            print(f"[done] {method} L{length}", flush=True)

    _write_csv(
        args.output_dir / "token_usage_by_position.csv",
        usage_rows,
        [
            "method",
            "length",
            "position",
            "is_suffix",
            "vocab_size",
            "used_tokens",
            "usage_rate",
            "entropy",
            "normalized_entropy",
            "effective_tokens",
            "effective_usage_rate",
            "gini",
            "top1_mass",
        ],
    )
    capacity_fields = [
        "method",
        "length",
        "prefix_len",
        "prefix_type",
        "num_groups",
        "collision_rate",
        "mean_group_size",
        "median_group_size",
        "p90_group_size",
        "max_group_size",
        "mean_gt_bucket_size",
        "median_gt_bucket_size",
        "p90_gt_bucket_size",
        "mean_gt_rank_in_bucket",
        "median_gt_rank_in_bucket",
        "missing_gt_items",
    ] + [f"oracle_bucket_recall@{k}" for k in top_ks]
    _write_csv(args.output_dir / "capacity_by_prefix.csv", capacity_rows, capacity_fields)

    if not args.skip_teacher:
        _write_csv(
            args.output_dir / "teacher_forcing_by_position.csv",
            teacher_rows,
            [
                "method",
                "length",
                "position",
                "is_suffix",
                "count",
                "nll",
                "ppl",
                "token_acc",
                "token_top5_acc",
                "prefix_exact_acc",
            ],
        )
        _write_csv(
            args.output_dir / "teacher_forcing_summary.csv",
            teacher_summary_rows,
            [
                "method",
                "length",
                "num_eval_sequences",
                "semantic_len",
                "sem_id_dim",
                "num_user_tokens",
                "mean_semantic_nll",
                "mean_semantic_acc",
                "last_semantic_nll",
                "last_semantic_acc",
                "suffix_mean_nll",
                "suffix_mean_acc",
                "full_prefix_exact_acc",
                "full_Recall@5",
                "full_NDCG@5",
                "full_Recall@10",
                "full_NDCG@10",
            ],
        )

    _write_json(
        args.output_dir / "summary.json",
        {
            "study_root": str(args.study_root),
            "eval_dataset": args.eval_dataset,
            "methods": methods,
            "lengths": lengths,
            "top_ks": top_ks,
            "skip_teacher": bool(args.skip_teacher),
            "max_users": int(args.max_users),
            "runs": manifest,
        },
    )
    print(f"[all done] output_dir={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
