
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Encode SETRec combine_tdcb_maps.npy with sentence-t5-base and save <dataset>.emb-t5-tdcb.npy."
    )
    ap.add_argument("--dataset", required=True, help="Domain name under setrec_data_root")
    ap.add_argument(
        "--setrec_data_root",
        type=Path,
        default=Path(os.environ.get("SETREC_DATA_ROOT", "data/setrec_data")),
    )
    ap.add_argument(
        "--model_name",
        default="sentence-transformers/sentence-t5-base",
        help="SentenceTransformer model name or local path",
    )
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    domain_dir = args.setrec_data_root / args.dataset
    in_path = domain_dir / "combine_tdcb_maps.npy"
    out_path = domain_dir / f"{args.dataset}.emb-t5-tdcb.npy"
    manifest_path = domain_dir / f"{args.dataset}.emb-t5-tdcb.manifest.json"

    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refuse to overwrite: {out_path}")

    eprint(f"[load] {in_path}")
    items = np.load(in_path, allow_pickle=True).item()
    recid2combine = items.get("recid2combine")
    if not isinstance(recid2combine, dict):
        raise TypeError("combine_tdcb_maps.npy must contain dict key 'recid2combine'")

    keys = sorted(int(k) for k in recid2combine.keys())
    texts = [str(recid2combine[k]) for k in keys]
    if any(t is None for t in texts):
        raise ValueError("Found None in recid2combine values")

    t0 = time.time()
    eprint(f"[encode] model={args.model_name} n_items={len(texts)} batch_size={args.batch_size}")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model_name)
    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    if not isinstance(emb, np.ndarray):
        emb = np.asarray(emb)

    emb = emb.astype(np.float32, copy=False)
    if emb.shape[0] != len(texts):
        raise ValueError(f"Unexpected embedding shape: {emb.shape} (n_items={len(texts)})")

    np.save(out_path, emb)
    dt = max(time.time() - t0, 1e-6)
    eprint(f"[OK] wrote {out_path} shape={tuple(emb.shape)} dt={dt:.1f}s")

    manifest = {
        "generated_utc": datetime.utcnow().isoformat() + "Z",
        "dataset": args.dataset,
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "n_items": int(emb.shape[0]),
        "dim": int(emb.shape[1]) if emb.ndim == 2 else None,
        "elapsed_sec": dt,
        "cuda_visible_devices": str(
            __import__("os").environ.get("CUDA_VISIBLE_DEVICES", "")
        ),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
