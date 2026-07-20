#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair tiger_config.json using the actual cached_ids.npy statistics.")
    parser.add_argument("--cached_ids", required=True, type=Path)
    parser.add_argument("--tiger_config", required=True, type=Path)
    parser.add_argument("--source", default="", help="Optional source string override")
    args = parser.parse_args()

    ids = np.load(args.cached_ids).astype(np.int64)
    n_items, sem_id_dim = ids.shape
    per_pos_sizes = [len(set(ids[:, pos].tolist())) for pos in range(sem_id_dim)]
    codebook_size = max(per_pos_sizes) if per_pos_sizes else 0
    code_counter = Counter(tuple(row) for row in ids.tolist())
    n_unique_codes = len(code_counter)
    max_collision = int(code_counter.most_common(1)[0][1]) if code_counter else 0
    collision_rate = float((n_items - n_unique_codes) / max(n_items, 1))

    cfg = {}
    if args.tiger_config.exists():
        cfg = json.loads(args.tiger_config.read_text())

    cfg.update(
        {
            "n_items": n_items,
            "sem_id_dim": sem_id_dim,
            "codebook_size": codebook_size,
            "per_pos_sizes": per_pos_sizes,
            "n_unique_codes": n_unique_codes,
            "collision_rate": collision_rate,
            "max_collision": max_collision,
        }
    )
    if args.source:
        cfg["source"] = args.source

    args.tiger_config.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(cfg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
