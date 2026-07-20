#!/usr/bin/env python3
"""Build a decoder-friendly SEATER compat table.

The native SEATER tree exported by `itemID_2_tree_indexID.npy` contains a
constant root at position 0 and a globally unique item leaf at the final
position.  Feeding both into the TIGER decoder makes the final SID token an
88866-way item classifier on MicroLens-1M.  This script keeps the semantic tree
prefix, drops the root and global leaf, and appends a small deterministic suffix
only to disambiguate items sharing the same prefix.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_cached_ids", required=True, type=Path)
    parser.add_argument("--input_config", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--drop_root", action="store_true", default=True)
    parser.add_argument("--drop_leaf", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cached = np.load(args.input_cached_ids).astype(np.int64)
    config = json.loads(args.input_config.read_text())

    if cached.ndim != 2:
        raise ValueError(f"cached_ids must be 2D, got shape={cached.shape}")
    if cached.shape[1] < 3:
        raise ValueError(f"SEATER path is too short to repair: shape={cached.shape}")

    start = 1 if args.drop_root else 0
    end = cached.shape[1] - 1 if args.drop_leaf else cached.shape[1]
    prefix = cached[:, start:end]
    if prefix.shape[1] <= 0:
        raise ValueError(f"Empty SEATER prefix after slicing start={start}, end={end}")

    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for item_id, row in enumerate(prefix.tolist()):
        groups[tuple(int(x) for x in row)].append(item_id)

    suffix = np.zeros((cached.shape[0], 1), dtype=np.int64)
    max_collision = 1
    for items in groups.values():
        max_collision = max(max_collision, len(items))
        for local_idx, item_id in enumerate(items):
            suffix[item_id, 0] = local_idx

    repaired = np.concatenate([prefix, suffix], axis=1)
    unique_repaired = len({tuple(row) for row in repaired.tolist()})
    if unique_repaired != repaired.shape[0]:
        raise ValueError(
            f"Repaired SEATER IDs are still colliding: unique={unique_repaired}, n={repaired.shape[0]}"
        )

    per_pos_sizes = [int(repaired[:, pos].max()) + 1 for pos in range(repaired.shape[1])]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "cached_ids.npy", repaired)

    out_config = {
        "n_items": int(repaired.shape[0]),
        "sem_id_dim": int(repaired.shape[1]),
        "codebook_size": int(max(per_pos_sizes)),
        "per_pos_sizes": per_pos_sizes,
        "id_type": "I03-seater-prefix-dedup",
        "id_name": "SEATER Tree Prefix + Local Dedup",
        "source_cached_ids": str(args.input_cached_ids),
        "source_config": str(args.input_config),
        "drop_root": bool(args.drop_root),
        "drop_leaf": bool(args.drop_leaf),
        "dedup_suffix": True,
        "duplicate_policy": "unique",
    }
    (args.output_dir / "tiger_config.json").write_text(
        json.dumps(out_config, indent=2),
        encoding="utf-8",
    )

    stats = {
        "input_shape": list(cached.shape),
        "output_shape": list(repaired.shape),
        "prefix_unique": int(len(groups)),
        "max_prefix_collision": int(max_collision),
        "mean_prefix_collision": float(cached.shape[0] / max(len(groups), 1)),
        "per_pos_sizes": per_pos_sizes,
        "source_per_pos_sizes": config.get("per_pos_sizes"),
    }
    (args.output_dir / "dedup_stats.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
