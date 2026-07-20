#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export SASRec item embeddings from a SEATER SASREC checkpoint to SASRec_item_embed.pkl."
    )
    ap.add_argument("--ckpt", type=Path, required=True, help="Path to SEATER SASREC ckpt best.pth")
    ap.add_argument("--out", type=Path, required=True, help="Output SASRec_item_embed.pkl path")
    ap.add_argument("--state_key", default="item_feat.emb_look_up.weight")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--drop_pad0", action="store_true", help="Drop embedding row 0 (padding)")
    args = ap.parse_args()

    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"Refuse to overwrite: {args.out}")

    state = torch.load(args.ckpt, map_location="cpu")
    if args.state_key not in state:
        keys = sorted(state.keys())
        raise KeyError(f"Missing {args.state_key!r} in ckpt. Available keys: {keys[:50]}")

    w = state[args.state_key].detach().cpu()
    if args.drop_pad0:
        if w.ndim != 2 or w.shape[0] < 2:
            raise ValueError(f"Unexpected embedding shape for drop_pad0: {tuple(w.shape)}")
        w = w[1:]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        pickle.dump(w, f)

    print(f"[OK] wrote {args.out} shape={tuple(w.shape)} dtype={w.dtype}")


if __name__ == "__main__":
    main()
