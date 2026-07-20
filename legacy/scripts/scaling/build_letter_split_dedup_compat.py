#!/usr/bin/env python3
"""Build a LETTER compat table with small-codebook dedup suffix digits.

The raw LETTER index may contain many-to-one collisions.  A single dedup suffix
with size equal to the largest collision group turns that final position into a
large item-classification token.  This converter instead represents the local
collision rank using multiple fixed-size suffix digits, keeping every decoder
position within a small codebook.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--letter_index_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--suffix_codebook_size", type=int, default=256)
    parser.add_argument("--id_name", default="LETTER Learned Code + Split Dedup Suffix")
    return parser.parse_args()


def _rank_to_digits(rank: int, base: int, width: int) -> list[int]:
    digits = [0] * width
    value = int(rank)
    for pos in range(width - 1, -1, -1):
        digits[pos] = value % base
        value //= base
    if value:
        raise ValueError(f"rank={rank} does not fit in width={width}, base={base}")
    return digits


def main() -> None:
    args = parse_args()
    if args.suffix_codebook_size <= 1:
        raise ValueError("--suffix_codebook_size must be > 1")

    with args.letter_index_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    item_ids = sorted(int(k) for k in raw.keys())
    if item_ids != list(range(len(item_ids))):
        raise ValueError("LETTER index item ids must be contiguous 0-based ids")

    seqs = [tuple(raw[str(i)]) for i in item_ids]
    target_len = len(seqs[0])
    if any(len(s) != target_len for s in seqs):
        raise ValueError("LETTER index contains variable-length token sequences")

    per_pos_maps: list[dict[str, int]] = []
    for pos in range(target_len):
        toks = sorted({seq[pos] for seq in seqs})
        per_pos_maps.append({tok: idx for idx, tok in enumerate(toks)})

    base_codes = np.zeros((len(seqs), target_len), dtype=np.int64)
    for item_id, seq in enumerate(seqs):
        for pos, tok in enumerate(seq):
            base_codes[item_id, pos] = per_pos_maps[pos][tok]

    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for item_id, code in enumerate(base_codes):
        groups[tuple(int(x) for x in code.tolist())].append(item_id)

    max_group_size = max((len(members) for members in groups.values()), default=1)
    suffix_width = max(1, math.ceil(math.log(max_group_size, args.suffix_codebook_size)))
    suffix_codes = np.zeros((len(seqs), suffix_width), dtype=np.int64)
    for members in groups.values():
        for rank, item_id in enumerate(sorted(members)):
            suffix_codes[item_id] = _rank_to_digits(rank, args.suffix_codebook_size, suffix_width)

    cached_ids = np.concatenate([base_codes, suffix_codes], axis=1)
    n_unique_before = len(groups)
    n_unique_after = len({tuple(int(x) for x in row.tolist()) for row in cached_ids})
    if n_unique_after != len(seqs):
        raise RuntimeError(f"split dedup failed: unique={n_unique_after}, items={len(seqs)}")

    base_per_pos_sizes = [len(m) for m in per_pos_maps]
    per_pos_sizes = base_per_pos_sizes + [args.suffix_codebook_size] * suffix_width
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "cached_ids.npy", cached_ids)

    tiger_config = {
        "n_items": len(seqs),
        "sem_id_dim": int(cached_ids.shape[1]),
        "codebook_size": int(max(per_pos_sizes)),
        "per_pos_sizes": per_pos_sizes,
        "id_type": "I05-letter-split-dedup",
        "id_name": args.id_name,
        "source": str(args.letter_index_path),
        "duplicate_policy": "unique",
        "dedup_suffix": True,
        "dedup_suffix_width": int(suffix_width),
        "dedup_suffix_codebook_size": int(args.suffix_codebook_size),
        "base_sem_id_dim": target_len,
        "base_unique_codes": n_unique_before,
        "base_collision_rate": float((len(seqs) - n_unique_before) / max(len(seqs), 1)),
        "max_collision_group_size": int(max_group_size),
    }
    (output_dir / "tiger_config.json").write_text(
        json.dumps(tiger_config, indent=2),
        encoding="utf-8",
    )

    stats = {
        "n_items": len(seqs),
        "base_unique_codes": n_unique_before,
        "base_collision_rate": tiger_config["base_collision_rate"],
        "dedup_unique_codes": n_unique_after,
        "dedup_collision_rate": 0.0,
        "target_len_before": target_len,
        "target_len_after": int(cached_ids.shape[1]),
        "base_per_pos_sizes": base_per_pos_sizes,
        "per_pos_sizes": per_pos_sizes,
        "max_collision_group_size": int(max_group_size),
        "suffix_codebook_size": int(args.suffix_codebook_size),
        "suffix_width": int(suffix_width),
    }
    (output_dir / "dedup_stats.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(stats, indent=2))
    print(f"[OK] saved {output_dir}")


if __name__ == "__main__":
    main()
