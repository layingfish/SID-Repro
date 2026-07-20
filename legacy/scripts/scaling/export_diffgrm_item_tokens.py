#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Instantiate DiffGRM tokenizer and export item_id2tokens only.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_npy", type=Path, required=True)
    parser.add_argument("--output_meta", type=Path, required=True)
    return parser.parse_known_args()


def main() -> None:
    args, unparsed_args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root / "third_party" / "DiffGRM")
    sys.path.insert(0, os.getcwd())

    from genrec.pipeline import Pipeline
    from genrec.utils import parse_command_line_args

    config_dict = parse_command_line_args(unparsed_args)
    pipeline = Pipeline(
        model_name=args.model,
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        config_dict=config_dict,
    )

    tok = pipeline.tokenizer
    item_id2tokens = np.zeros((tok.dataset.n_items, tok.n_digit), dtype=np.int64)
    for item, tokens in tok.item2tokens.items():
        item_id = tok.dataset.item2id[item]
        item_id2tokens[item_id] = np.asarray(tokens, dtype=np.int64)

    args.output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_npy, item_id2tokens)

    meta = {
        "dataset": args.dataset,
        "model": args.model,
        "n_items": int(tok.dataset.n_items),
        "n_digit": int(tok.n_digit),
        "codebook_size": int(tok.codebook_size),
        "sid_offset": int(tok.sid_offset),
        "config": {
            "category": pipeline.config.get("category"),
            "cache_dir": pipeline.config.get("cache_dir"),
            "sent_emb_model": pipeline.config.get("sent_emb_model"),
            "sent_emb_dim": pipeline.config.get("sent_emb_dim"),
            "sent_emb_pca": pipeline.config.get("sent_emb_pca"),
            "sid_quantizer": pipeline.config.get("sid_quantizer"),
            "disable_opq": pipeline.config.get("disable_opq"),
            "rq_kmeans_niters": pipeline.config.get("rq_kmeans_niters"),
            "rq_kmeans_seed": pipeline.config.get("rq_kmeans_seed"),
            "force_regenerate_opq": pipeline.config.get("force_regenerate_opq"),
        },
        "output_npy": str(args.output_npy),
    }
    args.output_meta.parent.mkdir(parents=True, exist_ok=True)
    args.output_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    pipeline.trainer.end()
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
