from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _gini(values: list[int]) -> float:
    if not values:
        return 0.0
    arr = np.sort(np.asarray(values, dtype=np.float64))
    n = arr.size
    total = arr.sum()
    if total <= 0:
        return 0.0
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((np.sum((2 * index - n - 1) * arr)) / (n * total))


def _entropy(values: list[int]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    total = arr.sum()
    if total <= 0:
        return 0.0, 0.0, 0.0
    probs = arr / total
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    max_entropy = float(math.log(len(values))) if len(values) > 1 else 0.0
    normalized = float(entropy / max_entropy) if max_entropy > 0 else 0.0
    perplexity = float(math.exp(entropy))
    return entropy, normalized, perplexity


def build_level_counts(codes: np.ndarray) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for level in range(codes.shape[1]):
        ctr = Counter(map(int, codes[:, level].tolist()))
        counts[f"level_{level}"] = {str(k): int(v) for k, v in sorted(ctr.items())}
    return counts


def build_full_code_counts(codes: np.ndarray) -> dict[str, int]:
    ctr = Counter(" ".join(map(str, row.tolist())) for row in codes)
    return {k: int(v) for k, v in ctr.items()}


def build_prefix_counts(codes: np.ndarray) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    n_levels = codes.shape[1]
    for prefix_len in range(1, n_levels + 1):
        ctr = Counter(" ".join(map(str, row[:prefix_len].tolist())) for row in codes)
        counts[f"prefix_len_{prefix_len}"] = {k: int(v) for k, v in ctr.items()}
    return counts


def build_transition_counts(codes: np.ndarray) -> dict[str, dict[str, dict[str, int]]]:
    transitions: dict[str, dict[str, dict[str, int]]] = {}
    n_levels = codes.shape[1]
    for prefix_len in range(1, n_levels):
        mapping: dict[str, Counter[int]] = {}
        for row in codes:
            prefix = " ".join(map(str, row[:prefix_len].tolist()))
            nxt = int(row[prefix_len])
            if prefix not in mapping:
                mapping[prefix] = Counter()
            mapping[prefix][nxt] += 1
        transitions[f"prefix_len_{prefix_len}"] = {
            prefix: {str(k): int(v) for k, v in sorted(counter.items())}
            for prefix, counter in mapping.items()
        }
    return transitions


def summarize_counts(
    level_counts: dict[str, dict[str, int]],
    full_code_counts: dict[str, int],
    prefix_counts: dict[str, dict[str, int]],
    transition_counts: dict[str, dict[str, dict[str, int]]],
    codebook_size_per_level: list[int] | None = None,
    semantic_levels: int | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    level_metrics: dict[str, Any] = {}
    used_codes_per_level: dict[str, int] = {}
    usage_rate_per_level: dict[str, float] = {}
    dead_code_count_per_level: dict[str, int] = {}
    effective_usage_rate_per_level: dict[str, float] = {}

    for idx, (level_name, raw_counts) in enumerate(sorted(level_counts.items())):
        counts = list(raw_counts.values())
        used_codes = len(counts)
        total_codes = (
            int(codebook_size_per_level[idx])
            if codebook_size_per_level is not None and idx < len(codebook_size_per_level)
            else used_codes
        )
        usage_rate = float(used_codes / total_codes) if total_codes > 0 else 0.0
        dead_codes = max(0, total_codes - used_codes)
        entropy, normalized_entropy, perplexity = _entropy(counts)
        effective_usage_rate = float(perplexity / total_codes) if total_codes > 0 else 0.0
        top1_share = float(max(counts) / sum(counts)) if counts else 0.0
        top10_share = float(sum(sorted(counts, reverse=True)[:10]) / sum(counts)) if counts else 0.0
        gini = _gini(counts)

        used_codes_per_level[level_name] = used_codes
        usage_rate_per_level[level_name] = round(usage_rate, 6)
        dead_code_count_per_level[level_name] = dead_codes
        effective_usage_rate_per_level[level_name] = round(effective_usage_rate, 6)
        level_metrics[level_name] = {
            "used_codes": used_codes,
            "usage_rate": round(usage_rate, 6),
            "dead_code_count": dead_codes,
            "top1_share": round(top1_share, 6),
            "top10_share": round(top10_share, 6),
            "entropy": round(entropy, 6),
            "normalized_entropy": round(normalized_entropy, 6),
            "perplexity": round(perplexity, 6),
            "effective_usage_rate": round(effective_usage_rate, 6),
            "gini": round(gini, 6),
        }

    prefix_metrics: dict[str, Any] = {}
    prefix_uniqueness_rates: list[float] = []
    n_items = int(sum(full_code_counts.values()))
    for prefix_name, raw_counts in sorted(prefix_counts.items()):
        counts = list(raw_counts.values())
        unique_prefixes = len(counts)
        entropy, normalized_entropy, perplexity = _entropy(counts)
        uniqueness_rate = float(unique_prefixes / n_items) if n_items > 0 else 0.0
        prefix_uniqueness_rates.append(uniqueness_rate)
        prefix_metrics[prefix_name] = {
            "unique_prefixes": unique_prefixes,
            "prefix_uniqueness_rate": round(uniqueness_rate, 6),
            "top1_share": round(float(max(counts) / sum(counts)) if counts else 0.0, 6),
            "top10_share": round(float(sum(sorted(counts, reverse=True)[:10]) / sum(counts)) if counts else 0.0, 6),
            "entropy": round(entropy, 6),
            "normalized_entropy": round(normalized_entropy, 6),
            "perplexity": round(perplexity, 6),
            "gini": round(_gini(counts), 6),
        }

    conditional_metrics: dict[str, Any] = {}
    for prefix_name, mapping in sorted(transition_counts.items()):
        branch_sizes: list[int] = []
        weighted_entropy = 0.0
        weighted_total = 0
        for _, child_counts_raw in mapping.items():
            child_counts = list(child_counts_raw.values())
            branch_sizes.append(len(child_counts))
            ent, _, _ = _entropy(child_counts)
            prefix_total = sum(child_counts)
            weighted_entropy += ent * prefix_total
            weighted_total += prefix_total
        conditional_metrics[prefix_name] = {
            "mean_branching": round(float(np.mean(branch_sizes)) if branch_sizes else 0.0, 6),
            "median_branching": round(float(np.median(branch_sizes)) if branch_sizes else 0.0, 6),
            "max_branching": int(max(branch_sizes)) if branch_sizes else 0,
            "weighted_conditional_entropy": round(float(weighted_entropy / weighted_total) if weighted_total > 0 else 0.0, 6),
        }

    full_counts = list(full_code_counts.values())
    n_unique_full_codes = len(full_counts)
    total_items = n_items
    collision_rate = float((total_items - n_unique_full_codes) / total_items) if total_items > 0 else 0.0
    max_collision = int(max(full_counts)) if full_counts else 0
    singleton_ratio = float(sum(1 for v in full_counts if v == 1) / n_unique_full_codes) if n_unique_full_codes > 0 else 0.0

    first_level = level_metrics.get("level_0", {})

    summary["first_level_used_codes"] = first_level.get("used_codes", 0)
    summary["first_level_usage_rate"] = first_level.get("usage_rate", 0.0)
    summary["first_level_top1_share"] = first_level.get("top1_share", 0.0)
    summary["first_level_top10_share"] = first_level.get("top10_share", 0.0)
    summary["first_level_entropy"] = first_level.get("entropy", 0.0)
    summary["first_level_normalized_entropy"] = first_level.get("normalized_entropy", 0.0)
    summary["first_level_perplexity"] = first_level.get("perplexity", 0.0)
    summary["first_level_effective_usage_rate"] = first_level.get("effective_usage_rate", 0.0)
    summary["first_level_gini"] = first_level.get("gini", 0.0)
    summary["per_level_metrics"] = level_metrics
    summary["per_level_used_codes"] = used_codes_per_level
    summary["per_level_usage_rate"] = usage_rate_per_level
    summary["per_level_effective_usage_rate"] = effective_usage_rate_per_level
    summary["dead_code_count_per_level"] = dead_code_count_per_level
    summary["mean_codebook_usage"] = round(float(np.mean(list(usage_rate_per_level.values()))) if usage_rate_per_level else 0.0, 6)
    summary["mean_effective_codebook_usage"] = round(float(np.mean(list(effective_usage_rate_per_level.values()))) if effective_usage_rate_per_level else 0.0, 6)
    summary["n_unique_full_codes"] = n_unique_full_codes
    summary["collision_rate"] = round(collision_rate, 6)
    summary["max_collision"] = max_collision
    summary["singleton_ratio"] = round(singleton_ratio, 6)
    summary["n_items"] = total_items

    if semantic_levels is None:
        semantic_levels = len(level_metrics)
    semantic_level_names = [f"level_{i}" for i in range(max(0, semantic_levels))]
    summary["semantic_levels"] = semantic_levels
    summary["hierarchical_effective_usage"] = round(
        float(np.mean([effective_usage_rate_per_level.get(name, 0.0) for name in semantic_level_names]))
        if semantic_level_names
        else 0.0,
        6,
    )
    summary["prefix_metrics"] = prefix_metrics
    summary["conditional_metrics"] = conditional_metrics
    prefix_len_keys = [f"prefix_len_{i}" for i in range(1, max(0, semantic_levels) + 1)]
    summary["prefix_auc"] = round(
        float(np.mean([prefix_metrics.get(k, {}).get("prefix_uniqueness_rate", 0.0) for k in prefix_len_keys]))
        if prefix_len_keys
        else 0.0,
        6,
    )
    return summary


def export_tokenizer_metrics(
    output_dir: str | Path,
    codes: np.ndarray,
    meta: dict[str, Any],
    codebook_size_per_level: list[int] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    codes = np.asarray(codes, dtype=np.int64)
    level_counts = build_level_counts(codes)
    full_code_counts = build_full_code_counts(codes)
    prefix_counts = build_prefix_counts(codes)
    transition_counts = build_transition_counts(codes)
    semantic_levels = codes.shape[1] - 1 if meta.get("dedup_suffix", False) and codes.shape[1] > 1 else codes.shape[1]
    summary = summarize_counts(
        level_counts,
        full_code_counts,
        prefix_counts,
        transition_counts,
        codebook_size_per_level,
        semantic_levels=semantic_levels,
    )

    (output_dir / "tokenizer_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "tokenizer_level_counts.json").write_text(
        json.dumps(level_counts, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "tokenizer_full_code_counts.json").write_text(
        json.dumps(full_code_counts, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "tokenizer_prefix_counts.json").write_text(
        json.dumps(prefix_counts, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "tokenizer_transition_counts.json").write_text(
        json.dumps(transition_counts, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "tokenizer_metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return summary
