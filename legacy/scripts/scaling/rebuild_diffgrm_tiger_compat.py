#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from tokenizer_metrics_utils import export_tokenizer_metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild DiffGRM item codes into TIGER-compatible cached_ids.")
    ap.add_argument("--item_tokens_npy", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--drop_first_row", action="store_true", default=True)
    ap.add_argument(
        "--mode",
        choices=["dedup_suffix", "raw_last", "raw_first", "raw4_last"],
        default="dedup_suffix",
        help="dedup_suffix: append a suffix digit to make codes unique; raw_last/raw4_last: keep raw PQ code and decode collisions with last-item-wins; raw_first: keep raw PQ code and decode collisions with first-item-wins.",
    )
    args = ap.parse_args()

    raw = np.asarray(np.load(args.item_tokens_npy), dtype=np.int64)
    if args.drop_first_row:
        raw = raw[1:]

    n_items, n_cols = raw.shape

    # Compact each PQ position independently into contiguous 0..K-1 code ids.
    compact_cols = []
    per_pos_sizes = []
    for j in range(n_cols):
        uniq = sorted(int(x) for x in np.unique(raw[:, j]))
        mapping = {tok: idx for idx, tok in enumerate(uniq)}
        compact_cols.append(np.asarray([mapping[int(x)] for x in raw[:, j]], dtype=np.int64))
        per_pos_sizes.append(len(uniq))

    compact = np.stack(compact_cols, axis=1)

    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for idx, row in enumerate(compact.tolist()):
        groups[tuple(row)].append(idx)

    max_collision = max(len(v) for v in groups.values())

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "raw_codes.npy", compact)

    if args.mode == "dedup_suffix":
        suffix = np.zeros((n_items,), dtype=np.int64)
        for rows in groups.values():
            for rank, item_idx in enumerate(rows):
                suffix[item_idx] = rank
        cached_ids = np.concatenate([compact, suffix[:, None]], axis=1)
        per_pos_sizes_out = per_pos_sizes + [int(max_collision)]
        id_name = "DiffGRM PQ Code + dedup suffix"
        duplicate_policy = "unique"
    else:
        cached_ids = compact
        per_pos_sizes_out = per_pos_sizes
        id_name = "DiffGRM raw PQ Code (last-item-wins decode)" if args.mode in {"raw_last", "raw4_last"} else "DiffGRM raw PQ Code (first-item-wins decode)"
        duplicate_policy = "last" if args.mode in {"raw_last", "raw4_last"} else "first"

    code_counter = Counter(map(tuple, cached_ids.tolist()))
    n_unique = len(code_counter)
    collision_rate = float((n_items - n_unique) / max(n_items, 1))

    np.save(out / "cached_ids.npy", cached_ids)

    tiger_config = {
        "n_items": int(n_items),
        "sem_id_dim": int(cached_ids.shape[1]),
        "codebook_size": int(max(max(per_pos_sizes_out), max_collision)),
        "per_pos_sizes": per_pos_sizes_out,
        "id_type": f"I06-diffgrm-mainalign-{args.mode}",
        "id_name": id_name,
        "source": str(args.item_tokens_npy),
        "n_unique_codes": int(n_unique),
        "collision_rate": float(collision_rate),
        "max_collision": int(max_collision),
        "dedup_suffix": bool(args.mode == "dedup_suffix"),
        "duplicate_policy": duplicate_policy,
        "semantic_code_length": int(n_cols),
        "decoder_code_length": int(cached_ids.shape[1]),
    }
    (out / "tiger_config.json").write_text(json.dumps(tiger_config, indent=2), encoding="utf-8")

    metrics_meta = {
        "method_family": "OPQ",
        "paper_variant": "DiffGRM-official",
        "implementation_variant": "official-diffgrm-rebuild",
        "embedding_source": str(args.item_tokens_npy),
        "embedding_dim": 0,
        "n_items": int(n_items),
        "n_levels": int(n_cols),
        "codebook_size_per_level": per_pos_sizes,
        "balanced_assignment": False,
        "dedup_suffix": bool(args.mode == "dedup_suffix"),
        "tokenizer_ckpt_rule": "explicit",
        "duplicate_policy": duplicate_policy,
        "semantic_code_length": int(n_cols),
    }
    metrics_codebook_sizes = per_pos_sizes if args.mode != "dedup_suffix" else per_pos_sizes + [int(max_collision)]
    export_tokenizer_metrics(
        output_dir=out,
        codes=cached_ids,
        meta=metrics_meta,
        codebook_size_per_level=metrics_codebook_sizes,
    )

    print(json.dumps({
        "output_dir": str(out),
        "mode": args.mode,
        "cached_ids_shape": list(cached_ids.shape),
        "per_pos_sizes": per_pos_sizes_out,
        "codebook_size": tiger_config["codebook_size"],
        "collision_rate_after_rebuild": collision_rate,
        "max_collision_after_rebuild": max_collision,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
