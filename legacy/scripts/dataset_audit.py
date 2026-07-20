#!/usr/bin/env python
"""Dataset audit for RecSys26 (SETRec dataset).

Goals:
- Produce reproducibility-critical checks for the processed dataset files under
  /data/xqp_data/RecSys26/third_party/SETRec/data/{beauty,toys,sports,steam}
- Record hashes, sizes, and basic statistics for train/val/test dicts.

This script is intended to run inside conda env `recsys26`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# Stable time format for markdown reports (avoid quote issues inside f-strings)
TIME_FMT = "%Y-%m-%d %H:%M:%S"


@dataclass
class SplitStats:
    users_total: int
    users_nonempty: int
    interactions_total: int
    per_user_len: Dict[str, float]
    uid_min: int | None
    uid_max: int | None
    uid_contiguous_0_based: bool
    item_min: int | None
    item_max: int | None
    unique_items: int
    item_nonneg: bool


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _to_list(v: Any) -> List[int]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer, int)):
        return [int(v)]
    # fallback: try list()
    try:
        return list(v)
    except TypeError:
        return [v]


def load_dict_npy(path: Path) -> Dict[int, List[int]]:
    obj = np.load(path, allow_pickle=True).item()
    out: Dict[int, List[int]] = {}
    for k, v in obj.items():
        # keys can be numpy scalars
        uid = int(k)
        items = [int(x) for x in _to_list(v)]
        out[uid] = items
    return out


def summarize_lengths(lengths: List[int]) -> Dict[str, float]:
    if not lengths:
        return {"min": 0.0, "mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}

    arr = np.array(lengths, dtype=np.int64)
    return {
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def compute_split_stats(d: Dict[int, List[int]]) -> SplitStats:
    uids = sorted(d.keys())
    users_total = len(uids)
    nonempty_uids = [u for u, items in d.items() if len(items) > 0]
    users_nonempty = len(nonempty_uids)

    lengths = [len(d[u]) for u in nonempty_uids]
    interactions_total = int(sum(lengths))

    uid_min = int(uids[0]) if uids else None
    uid_max = int(uids[-1]) if uids else None
    uid_contiguous_0_based = bool(uids) and uid_min == 0 and uid_max == users_total - 1 and len(set(uids)) == users_total

    all_items: List[int] = []
    item_nonneg = True
    for items in d.values():
        for it in items:
            if int(it) < 0:
                item_nonneg = False
            all_items.append(int(it))

    if all_items:
        item_min = int(min(all_items))
        item_max = int(max(all_items))
        unique_items = int(len(set(all_items)))
    else:
        item_min = None
        item_max = None
        unique_items = 0

    return SplitStats(
        users_total=users_total,
        users_nonempty=users_nonempty,
        interactions_total=interactions_total,
        per_user_len=summarize_lengths(lengths),
        uid_min=uid_min,
        uid_max=uid_max,
        uid_contiguous_0_based=uid_contiguous_0_based,
        item_min=item_min,
        item_max=item_max,
        unique_items=unique_items,
        item_nonneg=item_nonneg,
    )


def format_bytes(n: int) -> str:
    # simple, stable formatting
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}{unit}"
        n /= 1024
    return f"{n}B"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit SETRec processed dataset files")
    parser.add_argument(
        "--data_dir",
        default="/data/xqp_data/RecSys26/third_party/SETRec/data",
        help="SETRec data root directory",
    )
    parser.add_argument(
        "--datasets",
        default="beauty,toys,sports,steam",
        help="Comma-separated datasets",
    )
    parser.add_argument("--output", default=None, help="Write markdown report to this path")
    parser.add_argument("--json", default=None, help="Write machine-readable json to this path")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    data_root = Path(args.data_dir)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]

    md_lines: List[str] = []
    md_lines.append("# SETRec 数据集 Audit（RecSys26）")
    md_lines.append("")
    md_lines.append(f"生成时间：{now.strftime(TIME_FMT)}（UTC）")
    md_lines.append(f"数据根目录：`{data_root}`")
    md_lines.append("")

    audit_json: Dict[str, Any] = {
        "generated_utc": now.isoformat(),
        "data_root": str(data_root),
        "datasets": {},
    }

    key_files = [
        "training_dict.npy",
        "validation_dict.npy",
        "testing_dict.npy",
        "testing_warm_dict.npy",
        "testing_cold_dict.npy",
        "warm_item.npy",
        "cold_item.npy",
        "SASRec_item_embed.pkl",
        "combine_tdcb_maps.npy",
    ]

    for d in datasets:
        ddir = data_root / d
        md_lines.append("---")
        md_lines.append("")
        md_lines.append(f"## {d}")
        md_lines.append("")

        if not ddir.exists():
            md_lines.append(f"- [FAIL] 目录不存在：`{ddir}`")
            continue

        # file hashes & sizes
        md_lines.append("### 文件清单（哈希/大小）")
        md_lines.append("")
        md_lines.append("| file | exists | size | sha256 | mtime_utc |")
        md_lines.append("|---|---:|---:|---|---|")

        file_rows = []
        for fn in key_files:
            p = ddir / fn
            if p.exists():
                st = p.stat()
                mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                try:
                    h = sha256_file(p)
                except Exception as e:
                    h = f"ERROR:{type(e).__name__}"
                md_lines.append(f"| `{fn}` | 1 | {format_bytes(st.st_size)} | `{h}` | {mtime} |")
                file_rows.append({"file": fn, "exists": True, "size_bytes": st.st_size, "sha256": h, "mtime_utc": mtime})
            else:
                md_lines.append(f"| `{fn}` | 0 | - | - | - |")
                file_rows.append({"file": fn, "exists": False})

        # domain embedding file (pattern: <d>.emb-*.npy)
        emb_files = sorted([p.name for p in ddir.glob(f"{d}.emb-*.npy")])
        for fn in emb_files:
            p = ddir / fn
            st = p.stat()
            mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            h = sha256_file(p)
            md_lines.append(f"| `{fn}` | 1 | {format_bytes(st.st_size)} | `{h}` | {mtime} |")
            file_rows.append({"file": fn, "exists": True, "size_bytes": st.st_size, "sha256": h, "mtime_utc": mtime})

        # split stats
        md_lines.append("")
        md_lines.append("### Split 统计")
        md_lines.append("")

        stats_obj: Dict[str, Any] = {"files": file_rows}

        split_paths = {
            "train": ddir / "training_dict.npy",
            "val": ddir / "validation_dict.npy",
            "test": ddir / "testing_dict.npy",
        }

        split_stats = {}
        for split, sp in split_paths.items():
            if not sp.exists():
                md_lines.append(f"- [FAIL] 缺失 `{split}` 文件：`{sp.name}`")
                continue
            di = load_dict_npy(sp)
            st = compute_split_stats(di)
            split_stats[split] = st

        for split in ("train", "val", "test"):
            st = split_stats.get(split)
            if st is None:
                continue
            md_lines.append(f"#### {split}")
            md_lines.append("")
            md_lines.append("- users_total: {}".format(st.users_total))
            md_lines.append("- users_nonempty: {}".format(st.users_nonempty))
            md_lines.append("- interactions_total: {}".format(st.interactions_total))
            md_lines.append("- uid_range: {}..{} (contiguous_0_based={})".format(st.uid_min, st.uid_max, st.uid_contiguous_0_based))
            md_lines.append("- item_range: {}..{} (unique_items={}, item_nonneg={})".format(st.item_min, st.item_max, st.unique_items, st.item_nonneg))
            md_lines.append(
                "- per_user_len(test_items_per_user for test split): min={min:.0f}, mean={mean:.2f}, median={median:.0f}, p95={p95:.0f}, p99={p99:.0f}, max={max:.0f}".format(
                    **st.per_user_len
                )
                if split == "test"
                else "- per_user_len: min={min:.0f}, mean={mean:.2f}, median={median:.0f}, p95={p95:.0f}, p99={p99:.0f}, max={max:.0f}".format(
                    **st.per_user_len
                )
            )
            md_lines.append("")

        # warm/cold item sanity
        warm_path = ddir / "warm_item.npy"
        cold_path = ddir / "cold_item.npy"
        if warm_path.exists() and cold_path.exists() and split_stats.get("train") is not None:
            warm_items = [int(x) for x in np.load(warm_path, allow_pickle=True).tolist()]
            cold_items = [int(x) for x in np.load(cold_path, allow_pickle=True).tolist()]
            warm_set = set(warm_items)
            cold_set = set(cold_items)
            overlap = len(warm_set & cold_set)
            union = len(warm_set | cold_set)
            md_lines.append("### Warm/Cold item 校验")
            md_lines.append("")
            md_lines.append(f"- warm_items: {len(warm_set)}")
            md_lines.append(f"- cold_items: {len(cold_set)}")
            md_lines.append(f"- warm∩cold overlap: {overlap}")
            md_lines.append(f"- warm∪cold union: {union}")

            # range check
            item_max = split_stats["train"].item_max
            if item_max is not None:
                bad = [x for x in (warm_items + cold_items) if x < 0 or x > item_max]
                md_lines.append(f"- range_check (0..{item_max}): bad_count={len(bad)}")
            md_lines.append("")

            stats_obj["warm_cold"] = {
                "warm_unique": len(warm_set),
                "cold_unique": len(cold_set),
                "overlap": overlap,
                "union": union,
            }

        stats_obj["split_stats"] = {k: asdict(v) for k, v in split_stats.items()}
        audit_json["datasets"][d] = stats_obj

    md = "\n".join(md_lines) + "\n"

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"[OK] wrote markdown: {out}")
    else:
        print(md)

    if args.json:
        jout = Path(args.json)
        jout.parent.mkdir(parents=True, exist_ok=True)
        jout.write_text(json.dumps(audit_json, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[OK] wrote json: {jout}")


if __name__ == "__main__":
    main()
