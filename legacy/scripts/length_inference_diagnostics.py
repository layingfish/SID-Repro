#!/usr/bin/env python3
"""Inference-only diagnostics for semantic-ID length scaling.

This script does two post-hoc analyses on already trained/exported decoders:

1. Token usage / codebook collapse by code position.
2. Prefix and masked-layer projection curves from existing `pred_topk.jsonl`.

The prefix/mask analysis is intentionally inference-only: it never retrains a
decoder and never re-runs beam search. It projects each predicted item into a
coarser SID bucket, expands the bucket with a deterministic tie-breaker, and
then evaluates the resulting ranked list with the same warm-only protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from unified_eval import compute_metrics, load_ground_truth, load_predictions


DEFAULT_METHODS = ("rqvae", "rqkmeans", "opq")
DEFAULT_LENGTHS = (2, 3, 4, 6, 8, 12, 16)


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_codes(tokenizer_dir: Path) -> tuple[np.ndarray, dict]:
    config = read_json(tokenizer_dir / "tiger_config.json")
    raw_path = tokenizer_dir / "raw_codes.npy"
    if raw_path.exists():
        codes = np.load(raw_path).astype(np.int64)
    else:
        cached = np.load(tokenizer_dir / "cached_ids.npy").astype(np.int64)
        semantic_len = int(config.get("semantic_code_length", cached.shape[1]))
        codes = cached[:, :semantic_len]
    return codes, config


def resolve_vocab_sizes(config: dict, codes: np.ndarray) -> list[int]:
    semantic_len = codes.shape[1]
    per_pos_sizes = config.get("per_pos_sizes")
    if isinstance(per_pos_sizes, list) and len(per_pos_sizes) >= semantic_len:
        return [int(x) for x in per_pos_sizes[:semantic_len]]
    codebook_size = int(config.get("codebook_size", 0) or 0)
    if codebook_size > 0:
        return [codebook_size] * semantic_len
    return [int(codes[:, i].max()) + 1 for i in range(semantic_len)]


def gini_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    sorted_counts = np.sort(counts)
    n = sorted_counts.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(idx * sorted_counts)) / (n * total) - (n + 1.0) / n)


def token_usage_rows(method: str, length: int, codes: np.ndarray, vocab_sizes: list[int]) -> list[dict]:
    rows: list[dict] = []
    n_items = int(codes.shape[0])
    for layer in range(codes.shape[1]):
        vocab_size = int(vocab_sizes[layer])
        if vocab_size <= int(codes[:, layer].max()):
            vocab_size = int(codes[:, layer].max()) + 1
        counts = np.bincount(codes[:, layer], minlength=vocab_size).astype(np.float64)
        used = int(np.count_nonzero(counts))
        probs = counts[counts > 0] / counts.sum()
        entropy = float(-np.sum(probs * np.log(probs))) if probs.size else 0.0
        norm_entropy = float(entropy / math.log(vocab_size)) if vocab_size > 1 else 0.0
        effective_num = float(math.exp(entropy)) if entropy > 0 else 0.0
        sorted_desc = np.sort(counts)[::-1]
        top1_mass = float(sorted_desc[:1].sum() / n_items) if n_items else 0.0
        top5_mass = float(sorted_desc[:5].sum() / n_items) if n_items else 0.0
        rows.append(
            {
                "method": method,
                "length": length,
                "layer": layer + 1,
                "n_items": n_items,
                "vocab_size": vocab_size,
                "used_tokens": used,
                "dead_tokens": vocab_size - used,
                "usage_rate": used / vocab_size if vocab_size else 0.0,
                "entropy": entropy,
                "normalized_entropy": norm_entropy,
                "effective_tokens": effective_num,
                "effective_usage_rate": effective_num / vocab_size if vocab_size else 0.0,
                "gini": gini_from_counts(counts),
                "top1_mass": top1_mass,
                "top5_mass": top5_mass,
            }
        )
    return rows


def load_warm_ground_truth(data_dir: Path, dataset: str, mode: str) -> dict[int, list[int]]:
    gt = load_ground_truth(str(data_dir), dataset, mode)
    warm_path = data_dir / dataset / "warm_item.npy"
    warm_raw = np.load(warm_path, allow_pickle=True)
    warm_obj = warm_raw.item()
    warm_ids = set(warm_obj.keys()) if hasattr(warm_obj, "keys") else set(int(x) for x in warm_obj)
    gt = {int(uid): [int(i) for i in items if int(i) in warm_ids] for uid, items in gt.items()}
    return {uid: items for uid, items in gt.items() if items}


def load_seen_items(data_dir: Path, dataset: str) -> dict[int, set[int]]:
    train_path = data_dir / dataset / "training_dict.npy"
    if not train_path.exists():
        return {}
    raw = np.load(train_path, allow_pickle=True).item()
    seen: dict[int, set[int]] = {}
    for uid, items in raw.items():
        if isinstance(items, np.ndarray):
            items = items.tolist()
        if items is None:
            items = []
        if not isinstance(items, list):
            items = [items]
        seen[int(uid)] = {int(x) for x in items}
    return seen


def item_popularity(data_dir: Path, dataset: str, n_items: int) -> np.ndarray:
    train_path = data_dir / dataset / "training_dict.npy"
    pop = np.zeros(n_items, dtype=np.int64)
    if not train_path.exists():
        return pop
    raw = np.load(train_path, allow_pickle=True).item()
    for items in raw.values():
        if isinstance(items, np.ndarray):
            items = items.tolist()
        if items is None:
            continue
        if not isinstance(items, list):
            items = [items]
        for item in items:
            item = int(item)
            if 0 <= item < n_items:
                pop[item] += 1
    return pop


def sort_groups(groups: dict[tuple[int, ...], list[int]], popularity: np.ndarray) -> dict[tuple[int, ...], list[int]]:
    sorted_groups: dict[tuple[int, ...], list[int]] = {}
    for key, items in groups.items():
        sorted_groups[key] = sorted(items, key=lambda x: (-int(popularity[x]), int(x)))
    return sorted_groups


def build_prefix_groups(codes: np.ndarray, k: int, popularity: np.ndarray) -> dict[tuple[int, ...], list[int]]:
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for item_id, code in enumerate(codes):
        groups[tuple(int(x) for x in code[:k])].append(int(item_id))
    return sort_groups(groups, popularity)


def build_mask_groups(codes: np.ndarray, layer: int, popularity: np.ndarray) -> dict[tuple[int, ...], list[int]]:
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for item_id, code in enumerate(codes):
        key = tuple(int(x) for i, x in enumerate(code) if i != layer)
        groups[key].append(int(item_id))
    return sort_groups(groups, popularity)


def expand_ranked_list(
    original_items: Iterable[int],
    codes: np.ndarray,
    groups: dict[tuple[int, ...], list[int]],
    *,
    prefix_k: int | None,
    mask_layer: int | None,
    seen_items: set[int],
    top_n: int,
) -> list[int]:
    output: list[int] = []
    output_seen: set[int] = set()
    for pred in original_items:
        pred = int(pred)
        if pred < 0 or pred >= codes.shape[0]:
            continue
        code = codes[pred]
        if prefix_k is not None:
            key = tuple(int(x) for x in code[:prefix_k])
        elif mask_layer is not None:
            key = tuple(int(x) for i, x in enumerate(code) if i != mask_layer)
        else:
            raise ValueError("prefix_k or mask_layer must be set")
        for item in groups.get(key, []):
            if item in seen_items or item in output_seen:
                continue
            output_seen.add(item)
            output.append(item)
            if len(output) >= top_n:
                return output
    return output


def fill_to_topn(
    ranked: list[int],
    fallback_items: Iterable[int],
    *,
    seen_items: set[int],
    top_n: int,
) -> list[int]:
    out = list(ranked)
    used = set(out)
    for item in fallback_items:
        item = int(item)
        if item in seen_items or item in used:
            continue
        used.add(item)
        out.append(item)
        if len(out) >= top_n:
            break
    return out


def evaluate_projected(
    predictions: dict[int, list[int]],
    ground_truth: dict[int, list[int]],
    codes: np.ndarray,
    popularity: np.ndarray,
    seen_by_user: dict[int, set[int]],
    *,
    top_n_list: list[int],
    prefix_k: int | None = None,
    mask_layer: int | None = None,
) -> tuple[dict, dict]:
    if prefix_k is not None:
        groups = build_prefix_groups(codes, prefix_k, popularity)
    elif mask_layer is not None:
        groups = build_mask_groups(codes, mask_layer, popularity)
    else:
        raise ValueError("prefix_k or mask_layer must be set")

    max_top_n = max(top_n_list)
    global_fallback = list(np.argsort(-popularity, kind="stable"))
    projected: dict[int, list[int]] = {}
    candidate_sizes: list[int] = []
    for uid in ground_truth.keys():
        original = predictions.get(uid, [])
        seen = seen_by_user.get(uid, set())
        ranked = expand_ranked_list(
            original,
            codes,
            groups,
            prefix_k=prefix_k,
            mask_layer=mask_layer,
            seen_items=seen,
            top_n=max_top_n,
        )
        ranked = fill_to_topn(ranked, global_fallback, seen_items=seen, top_n=max_top_n)
        projected[uid] = ranked
        for pred in original[:max_top_n]:
            if pred < 0 or pred >= codes.shape[0]:
                continue
            code = codes[int(pred)]
            if prefix_k is not None:
                key = tuple(int(x) for x in code[:prefix_k])
            else:
                key = tuple(int(x) for i, x in enumerate(code) if i != mask_layer)
            candidate_sizes.append(len(groups.get(key, [])))

    metrics = compute_metrics(ground_truth, projected, top_n_list)
    aux = {
        "num_groups": len(groups),
        "mean_group_size_seen_by_predictions": float(np.mean(candidate_sizes)) if candidate_sizes else 0.0,
        "median_group_size_seen_by_predictions": float(np.median(candidate_sizes)) if candidate_sizes else 0.0,
        "max_group_size": max((len(v) for v in groups.values()), default=0),
    }
    return metrics, aux


def round_metric_dict(metrics: dict) -> dict:
    rounded = {}
    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            rounded[key] = round(float(value), 6)
        elif isinstance(value, (int, np.integer)):
            rounded[key] = int(value)
        else:
            rounded[key] = value
    return rounded


def summarize_token_usage(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], int(row["length"]))].append(row)
    summary: list[dict] = []
    for (method, length), group_rows in sorted(grouped.items()):
        last = max(group_rows, key=lambda r: int(r["layer"]))
        summary.append(
            {
                "method": method,
                "length": length,
                "mean_usage_rate": float(np.mean([r["usage_rate"] for r in group_rows])),
                "min_usage_rate": float(np.min([r["usage_rate"] for r in group_rows])),
                "last_usage_rate": last["usage_rate"],
                "mean_normalized_entropy": float(np.mean([r["normalized_entropy"] for r in group_rows])),
                "min_normalized_entropy": float(np.min([r["normalized_entropy"] for r in group_rows])),
                "last_normalized_entropy": last["normalized_entropy"],
                "max_gini": float(np.max([r["gini"] for r in group_rows])),
                "last_gini": last["gini"],
                "max_top1_mass": float(np.max([r["top1_mass"] for r in group_rows])),
                "last_top1_mass": last["top1_mass"],
                "max_dead_tokens": int(np.max([r["dead_tokens"] for r in group_rows])),
                "last_dead_tokens": int(last["dead_tokens"]),
            }
        )
    return summary


def baseline_row(method: str, length: int, baseline_metrics: dict, evaluated_metrics: dict) -> dict:
    row = {
        "method": method,
        "length": length,
        "analysis": "baseline_full_decoder",
        "k_or_layer": "full",
        "num_groups": "",
        "mean_group_size_seen_by_predictions": "",
        "median_group_size_seen_by_predictions": "",
        "max_group_size": "",
    }
    for key in ("Recall@5", "NDCG@5", "Recall@10", "NDCG@10", "HR@10", "MRR@10"):
        row[key] = float(baseline_metrics.get(key, evaluated_metrics.get(key, 0.0)))
    return row


def projected_row(method: str, length: int, analysis: str, k_or_layer: int, metrics: dict, aux: dict, baseline: dict) -> dict:
    row = {
        "method": method,
        "length": length,
        "analysis": analysis,
        "k_or_layer": k_or_layer,
        "num_groups": aux["num_groups"],
        "mean_group_size_seen_by_predictions": aux["mean_group_size_seen_by_predictions"],
        "median_group_size_seen_by_predictions": aux["median_group_size_seen_by_predictions"],
        "max_group_size": aux["max_group_size"],
    }
    for key in ("Recall@5", "NDCG@5", "Recall@10", "NDCG@10", "HR@10", "MRR@10"):
        row[key] = float(metrics.get(key, 0.0))
        row[f"drop_vs_baseline_{key}"] = float(baseline.get(key, 0.0)) - float(metrics.get(key, 0.0))
    return row


def markdown_table(rows: list[dict], columns: list[str], float_digits: int = 4) -> str:
    def fmt(v: object) -> str:
        if isinstance(v, float):
            return f"{v:.{float_digits}f}"
        return str(v)

    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)


def write_summary_md(
    path: Path,
    token_summary: list[dict],
    prefix_rows: list[dict],
    mask_rows: list[dict],
) -> None:
    perf_rows = [r for r in prefix_rows if r["analysis"] == "baseline_full_decoder"]
    best_prefix = []
    for method in DEFAULT_METHODS:
        for length in DEFAULT_LENGTHS:
            rows = [
                r for r in prefix_rows
                if r["method"] == method and int(r["length"]) == length and r["analysis"] == "prefix_projection"
            ]
            if rows:
                best = max(rows, key=lambda r: float(r["Recall@10"]))
                best_prefix.append(best)

    worst_mask = []
    for method in DEFAULT_METHODS:
        for length in DEFAULT_LENGTHS:
            rows = [
                r for r in mask_rows
                if r["method"] == method and int(r["length"]) == length and r["analysis"] == "mask_layer_projection"
            ]
            if rows:
                worst = max(rows, key=lambda r: float(r["drop_vs_baseline_Recall@10"]))
                worst_mask.append(worst)

    with path.open("w") as f:
        f.write("# ID Length Inference-Only Diagnostics\n\n")
        f.write("Scope: Amazon23-VG clean suffix length study, existing t5-small main-table-aligned decoder exports only. No decoder retraining and no beam-search regeneration.\n\n")
        f.write("Protocol: warm-only full-test evaluation. Prefix/mask curves are post-hoc bucket projection analyses using the existing full decoder prediction list and a global training-popularity tie-breaker inside each SID bucket.\n\n")
        f.write("## Baseline Full Decoder\n\n")
        f.write(markdown_table(perf_rows, ["method", "length", "Recall@5", "NDCG@5", "Recall@10", "NDCG@10"], 4))
        f.write("\n\n## Token Usage / Collapse Summary\n\n")
        f.write(markdown_table(
            token_summary,
            [
                "method",
                "length",
                "mean_usage_rate",
                "min_usage_rate",
                "last_usage_rate",
                "mean_normalized_entropy",
                "min_normalized_entropy",
                "last_normalized_entropy",
                "max_gini",
                "last_gini",
                "max_dead_tokens",
            ],
            4,
        ))
        f.write("\n\n## Best Prefix Projection per Length\n\n")
        f.write(markdown_table(
            best_prefix,
            [
                "method",
                "length",
                "k_or_layer",
                "Recall@10",
                "NDCG@10",
                "drop_vs_baseline_Recall@10",
                "mean_group_size_seen_by_predictions",
            ],
            4,
        ))
        f.write("\n\n## Most Damaging Layer Mask per Length\n\n")
        f.write(markdown_table(
            worst_mask,
            [
                "method",
                "length",
                "k_or_layer",
                "Recall@10",
                "NDCG@10",
                "drop_vs_baseline_Recall@10",
                "mean_group_size_seen_by_predictions",
            ],
            4,
        ))
        f.write("\n\n## Output Files\n\n")
        f.write("- `token_usage_by_layer.csv`\n")
        f.write("- `token_usage_summary.csv`\n")
        f.write("- `prefix_info_gain_metrics.csv`\n")
        f.write("- `mask_layer_drop_metrics.csv`\n")
        f.write("- `summary.json`\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_root", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=Path("/data/xqp_data/RecSys26/third_party/SETRec/data"))
    parser.add_argument("--dataset", default="amazon23_vg")
    parser.add_argument("--mode", default="full", choices=["full", "loo"])
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--lengths", default=",".join(str(x) for x in DEFAULT_LENGTHS))
    parser.add_argument("--top_n", default="5,10")
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    top_n_list = [int(x) for x in args.top_n.split(",") if x.strip()]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gt = load_warm_ground_truth(args.data_dir, args.dataset, args.mode)
    seen_by_user = load_seen_items(args.data_dir, args.dataset)

    token_rows: list[dict] = []
    prefix_rows: list[dict] = []
    mask_rows: list[dict] = []
    summary: dict[str, dict] = {}

    for method in methods:
        summary[method] = {}
        for length in lengths:
            tokenizer_dir = args.study_root / "tokenizer_suffix" / method / f"L{length}"
            decoder_dir = args.study_root / "decoder_mainalign_t5small" / method / f"L{length}"
            pred_path = decoder_dir / "pred_topk.jsonl"
            metrics_path = decoder_dir / "metrics_full_warm_only.json"
            if not tokenizer_dir.exists() or not pred_path.exists():
                continue

            codes, config = load_codes(tokenizer_dir)
            vocab_sizes = resolve_vocab_sizes(config, codes)
            token_rows.extend(token_usage_rows(method, length, codes, vocab_sizes))

            popularity = item_popularity(args.data_dir, args.dataset, codes.shape[0])
            predictions = load_predictions(str(pred_path))
            evaluated_baseline = compute_metrics(gt, predictions, top_n_list)
            baseline_metrics = read_json(metrics_path) if metrics_path.exists() else evaluated_baseline
            baseline = {k: float(baseline_metrics.get(k, evaluated_baseline.get(k, 0.0))) for k in evaluated_baseline.keys() if k.startswith(("Recall", "NDCG", "HR", "MRR"))}
            prefix_rows.append(baseline_row(method, length, baseline_metrics, evaluated_baseline))

            method_summary = {
                "baseline": round_metric_dict(evaluated_baseline),
                "tokenizer_dir": str(tokenizer_dir),
                "decoder_dir": str(decoder_dir),
                "prefix": {},
                "mask": {},
            }

            for k in range(1, codes.shape[1] + 1):
                metrics, aux = evaluate_projected(
                    predictions,
                    gt,
                    codes,
                    popularity,
                    seen_by_user,
                    top_n_list=top_n_list,
                    prefix_k=k,
                )
                row = projected_row(method, length, "prefix_projection", k, metrics, aux, baseline)
                prefix_rows.append(row)
                method_summary["prefix"][str(k)] = {"metrics": round_metric_dict(metrics), "aux": round_metric_dict(aux)}

            for layer in range(codes.shape[1]):
                metrics, aux = evaluate_projected(
                    predictions,
                    gt,
                    codes,
                    popularity,
                    seen_by_user,
                    top_n_list=top_n_list,
                    mask_layer=layer,
                )
                row = projected_row(method, length, "mask_layer_projection", layer + 1, metrics, aux, baseline)
                mask_rows.append(row)
                method_summary["mask"][str(layer + 1)] = {"metrics": round_metric_dict(metrics), "aux": round_metric_dict(aux)}

            summary[method][f"L{length}"] = method_summary
            print(f"[done] {method} L{length}")

    token_summary = summarize_token_usage(token_rows)

    token_fields = [
        "method",
        "length",
        "layer",
        "n_items",
        "vocab_size",
        "used_tokens",
        "dead_tokens",
        "usage_rate",
        "entropy",
        "normalized_entropy",
        "effective_tokens",
        "effective_usage_rate",
        "gini",
        "top1_mass",
        "top5_mass",
    ]
    summary_fields = [
        "method",
        "length",
        "mean_usage_rate",
        "min_usage_rate",
        "last_usage_rate",
        "mean_normalized_entropy",
        "min_normalized_entropy",
        "last_normalized_entropy",
        "max_gini",
        "last_gini",
        "max_top1_mass",
        "last_top1_mass",
        "max_dead_tokens",
        "last_dead_tokens",
    ]
    projection_fields = [
        "method",
        "length",
        "analysis",
        "k_or_layer",
        "num_groups",
        "mean_group_size_seen_by_predictions",
        "median_group_size_seen_by_predictions",
        "max_group_size",
        "Recall@5",
        "NDCG@5",
        "Recall@10",
        "NDCG@10",
        "HR@10",
        "MRR@10",
        "drop_vs_baseline_Recall@5",
        "drop_vs_baseline_NDCG@5",
        "drop_vs_baseline_Recall@10",
        "drop_vs_baseline_NDCG@10",
        "drop_vs_baseline_HR@10",
        "drop_vs_baseline_MRR@10",
    ]

    write_csv(args.output_dir / "token_usage_by_layer.csv", token_rows, token_fields)
    write_csv(args.output_dir / "token_usage_summary.csv", token_summary, summary_fields)
    write_csv(args.output_dir / "prefix_info_gain_metrics.csv", prefix_rows, projection_fields)
    write_csv(args.output_dir / "mask_layer_drop_metrics.csv", mask_rows, projection_fields)
    write_json(args.output_dir / "summary.json", summary)
    write_summary_md(args.output_dir / "ID_Length_Inference_Diagnostics_20260424.md", token_summary, prefix_rows, mask_rows)


if __name__ == "__main__":
    main()
