from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


METHOD_NAMES = {
    "rqvae": "RQ-VAE",
    "rqkmeans": "RQ-Kmeans",
    "opq": "OPQ",
}


def _gini(counts: np.ndarray) -> float:
    values = np.asarray(counts, dtype=np.float64)
    if values.size == 0 or values.sum() <= 0:
        return 0.0
    values = np.sort(values)
    n = values.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float(np.sum((2 * idx - n - 1) * values) / (n * np.sum(values)))


def _level_rows(method_key: str, length_dir: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    cfg = json.loads((length_dir / "tiger_config.json").read_text())
    codes = np.load(length_dir / "cached_ids.npy")
    semantic_length = int(cfg.get("semantic_code_length", int(length_dir.name[1:])))
    per_pos_sizes = cfg.get("per_pos_sizes")
    if not per_pos_sizes:
        per_pos_sizes = [int(codes[:, j].max()) + 1 for j in range(codes.shape[1])]

    rows: list[dict[str, object]] = []
    entropies: list[float] = []
    effective_usages: list[float] = []
    ginis: list[float] = []
    note = (
        f"excluded {codes.shape[1] - semantic_length} suffix col(s)"
        if codes.shape[1] > semantic_length
        else "no suffix col"
    )

    for level_idx in range(semantic_length):
        codebook_size = (
            int(per_pos_sizes[level_idx])
            if level_idx < len(per_pos_sizes)
            else int(codes[:, level_idx].max()) + 1
        )
        values = codes[:, level_idx].astype(np.int64)
        counts = np.bincount(values, minlength=codebook_size)[:codebook_size]
        total = int(counts.sum())
        nonzero = counts[counts > 0]
        probs = nonzero / total if total > 0 else np.array([], dtype=np.float64)
        entropy = float(-(probs * np.log(probs)).sum()) if probs.size else 0.0
        effective_usage = float(math.exp(entropy) / codebook_size) if codebook_size > 0 else 0.0
        gini = _gini(counts)
        used_tokens = int((counts > 0).sum())
        usage_rate = float(used_tokens / codebook_size) if codebook_size > 0 else 0.0
        top1_share = float(counts.max() / total) if total > 0 else 0.0
        top10_share = float(np.sort(counts)[-10:].sum() / total) if total > 0 else 0.0

        rows.append(
            {
                "method": METHOD_NAMES.get(method_key, method_key),
                "method_key": method_key,
                "L": semantic_length,
                "level": level_idx + 1,
                "codebook_size": codebook_size,
                "used_tokens": used_tokens,
                "usage_rate": usage_rate,
                "entropy": entropy,
                "effective_usage": effective_usage,
                "gini": gini,
                "top1_share": top1_share,
                "top10_share": top10_share,
                "n_items": total,
                "note": note,
            }
        )
        entropies.append(entropy)
        effective_usages.append(effective_usage)
        ginis.append(gini)

    summary = {
        "method": METHOD_NAMES.get(method_key, method_key),
        "L": semantic_length,
        "levels": semantic_length,
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "min_entropy": float(np.min(entropies)) if entropies else 0.0,
        "mean_effective_usage": float(np.mean(effective_usages)) if effective_usages else 0.0,
        "min_effective_usage": float(np.min(effective_usages)) if effective_usages else 0.0,
        "mean_gini": float(np.mean(ginis)) if ginis else 0.0,
        "max_gini": float(np.max(ginis)) if ginis else 0.0,
        "note": note,
    }
    return rows, summary


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, object]], summaries: list[dict[str, object]], csv_path: Path, summary_csv: Path) -> None:
    lines: list[str] = []
    lines.append("# 全层级 Code Usage / Codebook Collapse 分析")
    lines.append("")
    lines.append("## 统计口径")
    lines.append("")
    lines.append("- 数据：Amazon23-VG `code_length_study/20260421_clean_suffix_mainalign_t5small` 的 tokenizer suffix clean 版本。")
    lines.append("- 方法：`RQ-VAE`、`RQ-Kmeans`、`OPQ`，长度集合为 `L=2,3,4,6,8,12,16`。")
    lines.append("- 层级：只统计 `semantic_code_length` 对应的有效 semantic levels；`cached_ids.npy` 里的 `dedup suffix` 列不计入下表。")
    lines.append("- 指标：`Entropy` 越高表示分布越均匀；`Effective usage = exp(Entropy) / codebook_size`；`Gini` 越高表示 token 使用越集中。")
    lines.append("")
    lines.append("## 长度级摘要")
    lines.append("")
    lines.append("| Method | L | Mean Entropy | Min Entropy | Mean Eff. Usage | Min Eff. Usage | Mean Gini | Max Gini |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summaries:
        lines.append(
            f"| {row['method']} | {row['L']} | {row['mean_entropy']:.4f} | {row['min_entropy']:.4f} | "
            f"{row['mean_effective_usage']:.4f} | {row['min_effective_usage']:.4f} | "
            f"{row['mean_gini']:.4f} | {row['max_gini']:.4f} |"
        )
    lines.append("")
    lines.append("## 全 Level 明细")
    for method_name in ["RQ-VAE", "RQ-Kmeans", "OPQ"]:
        lines.append("")
        lines.append(f"### {method_name}")
        lines.append("")
        lines.append("| L | Level | Codebook Size | Used Tokens | Usage Rate | Entropy | Effective Usage | Gini | Top-1 Share | Top-10 Share |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in rows:
            if row["method"] != method_name:
                continue
            lines.append(
                f"| {row['L']} | {row['level']} | {row['codebook_size']} | {row['used_tokens']} | "
                f"{row['usage_rate']:.4f} | {row['entropy']:.4f} | {row['effective_usage']:.4f} | "
                f"{row['gini']:.4f} | {row['top1_share']:.4f} | {row['top10_share']:.4f} |"
            )
    lines.append("")
    lines.append("## 直接观察")
    lines.append("")
    lines.append("- `RQ-VAE` 的 collapse 是全层级现象，不只是 level-1：随着 L 增大，后层 `Effective usage` 快速接近 0，`Gini` 接近 1。")
    lines.append("- `RQ-Kmeans` 的前缀层在不同 L 下高度一致，这是由同一长码前缀截断/复用带来的；后层仍保持较高有效利用率，但随层数增加逐步下降。")
    lines.append("- `OPQ` 的各层有效利用率整体更稳定，长码下没有出现 RQ-VAE 那种极端 collapse，但 L 过长时 decoder 性能仍可能受到序列长度和后层边际信息不足影响。")
    lines.append("")
    lines.append(f"完整 CSV：`{csv_path}`")
    lines.append(f"摘要 CSV：`{summary_csv}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--methods", nargs="+", default=["rqvae", "rqkmeans", "opq"])
    args = parser.parse_args()

    tokenizer_root = args.root / "tokenizer_suffix"
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for method_key in args.methods:
        method_dir = tokenizer_root / method_key
        for length_dir in sorted(method_dir.glob("L*"), key=lambda p: int(p.name[1:])):
            level_rows, summary = _level_rows(method_key, length_dir)
            rows.extend(level_rows)
            summaries.append(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "all_level_code_usage.csv"
    summary_csv = args.output_dir / "all_level_code_usage_summary.csv"
    md_path = args.output_dir / "all_level_code_usage.md"
    _write_csv(
        csv_path,
        rows,
        [
            "method",
            "method_key",
            "L",
            "level",
            "codebook_size",
            "used_tokens",
            "usage_rate",
            "entropy",
            "effective_usage",
            "gini",
            "top1_share",
            "top10_share",
            "n_items",
            "note",
        ],
    )
    _write_csv(
        summary_csv,
        summaries,
        [
            "method",
            "L",
            "levels",
            "mean_entropy",
            "min_entropy",
            "mean_effective_usage",
            "min_effective_usage",
            "mean_gini",
            "max_gini",
            "note",
        ],
    )
    _write_markdown(md_path, rows, summaries, csv_path, summary_csv)
    print(md_path)
    print(csv_path)
    print(summary_csv)
    print(f"rows={len(rows)} summaries={len(summaries)}")


if __name__ == "__main__":
    main()
