#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr, flush=True)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_npy_dict(path: Path) -> dict[int, list[int]]:
    obj = np.load(path, allow_pickle=True).item()
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in {path}, got {type(obj)}")
    out: dict[int, list[int]] = {}
    for k, v in obj.items():
        out[int(k)] = [int(x) for x in list(v)]
    return out


def _dump_tsv(path: Path, rows: list[tuple[Any, Any, Any]], header: tuple[str, str, str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate SEATER/SASRec TSV files from SETRec-style split npy files."
    )
    ap.add_argument("--domain", required=True, choices=["amazon23_vg", "microlens_50k", "microlens_100k", "microlens_1m", "yelp"])
    ap.add_argument(
        "--setrec_root",
        type=Path,
        default=Path("/data/xqp_data/RecSys26/third_party/SETRec/data"),
    )
    ap.add_argument(
        "--out_root",
        type=Path,
        default=Path("/data/xqp_data/RecSys26/datasets/seater_setrec"),
        help="Output root containing <domain>/dataset/*.tsv",
    )
    ap.add_argument("--max_his", type=int, default=50, help="History truncation length (after +1 shift)")
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    domain = args.domain
    setrec_root = args.setrec_root
    out_dir = args.out_root / domain / "dataset"
    _ensure_dir(out_dir)

    train_path = out_dir / "training.tsv"
    val_path = out_dir / "validation.tsv"
    test_path = out_dir / "test.tsv"
    for p in [train_path, val_path, test_path]:
        if p.exists() and not args.overwrite:
            raise FileExistsError(f"Refuse to overwrite: {p}")

    train_dict = _load_npy_dict(setrec_root / domain / "training_dict.npy")
    val_dict = _load_npy_dict(setrec_root / domain / "validation_dict.npy")
    test_dict = _load_npy_dict(setrec_root / domain / "testing_dict.npy")

    train_rows: list[tuple[Any, Any, Any]] = []
    valid_rows: list[tuple[Any, Any, Any]] = []
    test_rows: list[tuple[Any, Any, Any]] = []

    max_his = max(int(args.max_his), 1)

    for u, seq0 in train_dict.items():
        seq = [int(x) + 1 for x in seq0]

        if len(seq) >= 2:
            for i in range(1, len(seq)):
                start = max(0, i - max_his)
                train_rows.append((int(u), str(seq[start:i]), int(seq[i])))

        v0 = val_dict.get(u, [])
        t0 = test_dict.get(u, [])

        if v0:
            v = [int(x) + 1 for x in v0]
            valid_rows.append((int(u), str(seq[-max_his:]), str(v)))

        if t0:
            t = [int(x) + 1 for x in t0]
            his = seq + ([int(x) + 1 for x in v0] if v0 else [])
            test_rows.append((int(u), str(his[-max_his:]), str(t)))

    _dump_tsv(train_path, train_rows, ("uid", "his_seq", "next_item"))
    _dump_tsv(val_path, valid_rows, ("uid", "his_seq", "predicting_items"))
    _dump_tsv(test_path, test_rows, ("uid", "his_seq", "predicting_items"))

    eprint(
        f"[OK] domain={domain} train_rows={len(train_rows)} val_rows={len(valid_rows)} test_rows={len(test_rows)} out_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
