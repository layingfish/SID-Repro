#!/usr/bin/env python3
"""Build a decoder-friendly SEATER local-branch compat table.

Native SEATER tree paths use global node ids.  Those ids make later tree levels
look like very large codebooks to an autoregressive decoder.  This converter
keeps the same tree path semantics but re-encodes each edge as a local branch
index under its parent.  It drops the constant root and the globally unique item
leaf, then appends a small deterministic suffix for items sharing the same
semantic prefix.
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


def _make_local_branch_codes(prefix: np.ndarray) -> tuple[np.ndarray, list[int], list[dict[int, dict[int, int]]]]:
    """Convert global node ids in a tree prefix to local child ranks."""
    local = np.zeros_like(prefix, dtype=np.int64)
    per_pos_sizes: list[int] = []
    branch_maps: list[dict[int, dict[int, int]]] = []

    parent_ids = np.full(prefix.shape[0], -1, dtype=np.int64)
    for pos in range(prefix.shape[1]):
        children_by_parent: dict[int, set[int]] = defaultdict(set)
        for parent, child in zip(parent_ids.tolist(), prefix[:, pos].tolist()):
            children_by_parent[int(parent)].add(int(child))

        pos_maps: dict[int, dict[int, int]] = {}
        max_branch = 0
        for parent, children in children_by_parent.items():
            ordered = sorted(children)
            mapping = {child: rank for rank, child in enumerate(ordered)}
            pos_maps[parent] = mapping
            max_branch = max(max_branch, len(mapping))

        for row_idx, (parent, child) in enumerate(zip(parent_ids.tolist(), prefix[:, pos].tolist())):
            local[row_idx, pos] = pos_maps[int(parent)][int(child)]

        per_pos_sizes.append(int(max_branch))
        branch_maps.append(pos_maps)
        parent_ids = prefix[:, pos]

    return local, per_pos_sizes, branch_maps


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
    global_prefix = cached[:, start:end]
    if global_prefix.shape[1] <= 0:
        raise ValueError(f"Empty SEATER prefix after slicing start={start}, end={end}")

    local_prefix, branch_sizes, branch_maps = _make_local_branch_codes(global_prefix)

    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for item_id, row in enumerate(local_prefix.tolist()):
        groups[tuple(int(x) for x in row)].append(item_id)

    suffix = np.zeros((cached.shape[0], 1), dtype=np.int64)
    max_collision = 1
    for items in groups.values():
        max_collision = max(max_collision, len(items))
        for local_idx, item_id in enumerate(sorted(items)):
            suffix[item_id, 0] = local_idx

    repaired = np.concatenate([local_prefix, suffix], axis=1)
    unique_repaired = len({tuple(row) for row in repaired.tolist()})
    if unique_repaired != repaired.shape[0]:
        raise ValueError(
            f"Repaired SEATER IDs are still colliding: unique={unique_repaired}, n={repaired.shape[0]}"
        )

    per_pos_sizes = branch_sizes + [int(max_collision)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "cached_ids.npy", repaired)

    serializable_branch_maps = [
        {str(parent): {str(child): int(rank) for child, rank in child_map.items()}
         for parent, child_map in pos_map.items()}
        for pos_map in branch_maps
    ]
    (args.output_dir / "branch_maps.json").write_text(
        json.dumps(serializable_branch_maps, indent=2),
        encoding="utf-8",
    )

    out_config = {
        "n_items": int(repaired.shape[0]),
        "sem_id_dim": int(repaired.shape[1]),
        "codebook_size": int(max(per_pos_sizes)),
        "per_pos_sizes": per_pos_sizes,
        "id_type": "I03-seater-local-branch-dedup",
        "id_name": "SEATER Local Branch Path + Local Dedup",
        "source_cached_ids": str(args.input_cached_ids),
        "source_config": str(args.input_config),
        "drop_root": bool(args.drop_root),
        "drop_leaf": bool(args.drop_leaf),
        "dedup_suffix": True,
        "duplicate_policy": "unique",
        "encoding": "local_branch_rank",
    }
    (args.output_dir / "tiger_config.json").write_text(
        json.dumps(out_config, indent=2),
        encoding="utf-8",
    )

    stats = {
        "input_shape": list(cached.shape),
        "global_prefix_shape": list(global_prefix.shape),
        "output_shape": list(repaired.shape),
        "prefix_unique": int(len(groups)),
        "max_prefix_collision": int(max_collision),
        "mean_prefix_collision": float(cached.shape[0] / max(len(groups), 1)),
        "per_pos_sizes": per_pos_sizes,
        "branch_sizes": branch_sizes,
        "source_per_pos_sizes": config.get("per_pos_sizes"),
        "max_local_code_per_pos": [int(local_prefix[:, pos].max()) for pos in range(local_prefix.shape[1])],
    }
    (args.output_dir / "dedup_stats.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
