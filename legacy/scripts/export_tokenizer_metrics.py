#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tokenizer_metrics_utils import export_tokenizer_metrics


def _load_codes_from_index_json(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        # expected shape: {item_id: ["<a_1>", "<b_2>", ...]} or {item_id: [1,2,...]}
        rows = []
        for _, seq in sorted(data.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else str(kv[0])):
            norm = []
            for token in seq:
                if isinstance(token, int):
                    norm.append(token)
                else:
                    token = str(token)
                    if token.startswith("<") and token.endswith(">") and "_" in token:
                        norm.append(int(token.split("_")[-1].rstrip(">")))
                    else:
                        norm.append(int(token))
            rows.append(norm)
        return np.asarray(rows, dtype=np.int64)
    raise ValueError(f"Unsupported index.json format: {path}")


def _load_codes(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.int64)
    if path.suffix == ".json":
        return _load_codes_from_index_json(path)
    raise ValueError(f"Unsupported input type: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export tokenizer utilization metrics from code assignments")
    parser.add_argument("--input", required=True, type=Path, help="cached_ids.npy or index.json")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--method_family", required=True)
    parser.add_argument("--paper_variant", required=True)
    parser.add_argument("--implementation_variant", required=True)
    parser.add_argument("--embedding_source", default="")
    parser.add_argument("--embedding_dim", type=int, default=0)
    parser.add_argument("--balanced_assignment", action="store_true")
    parser.add_argument("--dedup_suffix", action="store_true")
    parser.add_argument("--tokenizer_ckpt_rule", default="final")
    parser.add_argument("--codebook_sizes", default="", help="comma-separated per-level codebook sizes")
    parser.add_argument("--extra_meta_json", default="", help="optional JSON string merged into meta")
    args = parser.parse_args()

    codes = _load_codes(args.input)
    codebook_sizes = [int(x) for x in args.codebook_sizes.split(",") if x.strip()] if args.codebook_sizes else None
    meta = {
        "method_family": args.method_family,
        "paper_variant": args.paper_variant,
        "implementation_variant": args.implementation_variant,
        "embedding_source": args.embedding_source,
        "embedding_dim": args.embedding_dim,
        "n_items": int(codes.shape[0]),
        "n_levels": int(codes.shape[1]),
        "codebook_size_per_level": codebook_sizes or [],
        "balanced_assignment": bool(args.balanced_assignment),
        "dedup_suffix": bool(args.dedup_suffix),
        "tokenizer_ckpt_rule": args.tokenizer_ckpt_rule,
        "input_path": str(args.input),
    }
    if args.extra_meta_json:
        meta.update(json.loads(args.extra_meta_json))

    summary = export_tokenizer_metrics(
        output_dir=args.output_dir,
        codes=codes,
        meta=meta,
        codebook_size_per_level=codebook_sizes,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
