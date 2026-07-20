#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build TIGER-compatible unique cached_ids from a LETTER index with collisions."
    )
    parser.add_argument("--letter_index_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--id_name", default="LETTER Learned Code + Dedup Suffix")
    args = parser.parse_args()

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

    dedup = np.zeros((len(seqs),), dtype=np.int64)
    max_group_size = 0
    for members in groups.values():
        members = sorted(members)
        max_group_size = max(max_group_size, len(members))
        for rank, item_id in enumerate(members):
            dedup[item_id] = rank

    cached_ids = np.concatenate([base_codes, dedup[:, None]], axis=1)
    n_unique_before = len(groups)
    n_unique_after = len({tuple(int(x) for x in row.tolist()) for row in cached_ids})
    if n_unique_after != len(seqs):
        raise RuntimeError(f"dedup failed: unique={n_unique_after}, items={len(seqs)}")

    per_pos_sizes = [len(m) for m in per_pos_maps] + [max_group_size]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "cached_ids.npy", cached_ids)

    tiger_config = {
        "n_items": len(seqs),
        "sem_id_dim": target_len + 1,
        "codebook_size": max(per_pos_sizes),
        "per_pos_sizes": per_pos_sizes,
        "id_type": "I05-letter-dedup",
        "id_name": args.id_name,
        "source": str(args.letter_index_path),
        "duplicate_policy": "unique",
        "dedup_suffix": True,
        "base_sem_id_dim": target_len,
        "base_unique_codes": n_unique_before,
        "base_collision_rate": float((len(seqs) - n_unique_before) / max(len(seqs), 1)),
        "max_collision_group_size": int(max_group_size),
    }
    (output_dir / "tiger_config.json").write_text(
        json.dumps(tiger_config, indent=2), encoding="utf-8"
    )

    stats = {
        "n_items": len(seqs),
        "base_unique_codes": n_unique_before,
        "base_collision_rate": tiger_config["base_collision_rate"],
        "dedup_unique_codes": n_unique_after,
        "dedup_collision_rate": 0.0,
        "target_len_before": target_len,
        "target_len_after": target_len + 1,
        "per_pos_sizes": per_pos_sizes,
        "max_collision_group_size": int(max_group_size),
    }
    (output_dir / "dedup_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    print(json.dumps(stats, indent=2))
    print(f"[OK] saved {output_dir}")


if __name__ == "__main__":
    main()
