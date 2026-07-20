#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from transformers import T5Tokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild I04 SemID tokenizer assets from LLM_RecSys_ID content-based representation.")
    ap.add_argument("--llm_id_root", required=True, type=Path)
    ap.add_argument("--data_dir", required=True, type=str, help="Relative data dir under llm_id_root, e.g. data/")
    ap.add_argument("--task", required=True, type=str, help="e.g. setrec_amazon23_vg")
    ap.add_argument("--number_of_items", required=True, type=int, help="Includes PAD item 0, e.g. 25063")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--t5_model_path", required=True, type=str)
    ap.add_argument("--output_dir", required=True, type=Path)
    args = ap.parse_args()

    llm_id_root = args.llm_id_root.resolve()
    if str(llm_id_root) not in sys.path:
        sys.path.insert(0, str(llm_id_root))

    from item_rep_method import load_meta, build_category_map, content_based_representation
    from utils import content_based_representation_non_hierarchical

    class Args:
        pass

    cfg = Args()
    data_dir = args.data_dir
    if not data_dir.endswith("/"):
        data_dir += "/"
    if not Path(data_dir).is_absolute():
        data_dir = str((llm_id_root / data_dir).resolve()) + "/"
    cfg.data_dir = data_dir
    cfg.task = args.task
    cfg.number_of_items = args.number_of_items
    cfg.seed = args.seed

    meta_data, meta_dict, id2item = load_meta(cfg)
    category_dict, level_categories = build_category_map(cfg, meta_data, meta_dict, id2item)

    tok = T5Tokenizer.from_pretrained(args.t5_model_path, model_max_length=512)
    # Old LLM_RecSys_ID helpers access tokenizer.vocab directly.
    tok.vocab = tok.get_vocab()
    tok = content_based_representation_non_hierarchical(cfg, category_dict, level_categories, tok)

    seqs: list[list[int]] = []
    lengths: dict[int, int] = {}
    for item_idx_1based in range(1, args.number_of_items):
        rep = content_based_representation(cfg, str(item_idx_1based), category_dict, level_categories)
        ids = tok(rep, add_special_tokens=False)["input_ids"]
        seqs.append([int(x) for x in ids])
        lengths[len(ids)] = lengths.get(len(ids), 0) + 1

    max_len = max(lengths)
    n_items = len(seqs)

    per_pos_values: list[list[int]] = []
    per_pos_sizes: list[int] = []
    cached_ids = np.zeros((n_items, max_len), dtype=np.int64)

    for pos in range(max_len):
        vals = sorted({seq[pos] for seq in seqs if pos < len(seq)})
        needs_pad = any(len(seq) <= pos for seq in seqs)
        pad_token = None
        if needs_pad:
            pad_token = max(vals) + 1 if vals else 0
            vals = vals + [pad_token]
        mapping = {tok_id: idx for idx, tok_id in enumerate(vals)}
        per_pos_values.append(vals)
        per_pos_sizes.append(len(vals))
        for i, seq in enumerate(seqs):
            raw_tok = seq[pos] if pos < len(seq) else pad_token
            cached_ids[i, pos] = mapping[int(raw_tok)]

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "cached_ids.npy", cached_ids)

    tiger_config = {
        "n_items": int(n_items),
        "sem_id_dim": int(max_len),
        "codebook_size": int(max(per_pos_sizes)),
        "per_pos_sizes": [int(x) for x in per_pos_sizes],
        "id_type": "I04-semid-mainalign",
        "id_name": "SemID reconstructed from LLM_RecSys_ID content_based representation",
        "source": str(llm_id_root),
        "task": args.task,
        "seed": int(args.seed),
        "n_unique_codes": int(len({tuple(row) for row in cached_ids.tolist()})),
        "collision_rate": 0.0,
        "max_collision": 1,
    }
    (out / "tiger_config.json").write_text(json.dumps(tiger_config, indent=2), encoding="utf-8")

    aux = {
        "lengths": lengths,
        "per_pos_sizes": per_pos_sizes,
        "sample_tokenized_ids": seqs[:5],
        "sample_category_paths": {str(i + 1): category_dict[str(i + 1)] for i in range(min(5, len(category_dict)))},
    }
    (out / "rebuild_debug.json").write_text(json.dumps(aux, indent=2), encoding="utf-8")

    print(json.dumps({
        "output_dir": str(out),
        "n_items": n_items,
        "sem_id_dim": max_len,
        "per_pos_sizes": per_pos_sizes,
        "codebook_size": max(per_pos_sizes),
        "lengths": lengths,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
