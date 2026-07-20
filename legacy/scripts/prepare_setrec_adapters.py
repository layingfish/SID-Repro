#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np


def _load_npy_dict(path: Path) -> dict:
    return np.load(path, allow_pickle=True).item()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _parse_title(text: str, fallback: str) -> str:
    m = re.search(r"The item title is (.*?)(?:\.|$)", text)
    if m:
        title = m.group(1).strip()
        return title if title else fallback
    return fallback


def _parse_brand(text: str, fallback: str = "Unknown") -> str:
    m = re.search(r"The brand of this item is (.*?)(?:\.|$)", text)
    if m:
        b = m.group(1).strip()
        return b if b else fallback

    # Steam subset uses publisher/dev
    m = re.search(r"The publisher is (.*?)(?:\.|$)", text)
    if m:
        b = m.group(1).strip()
        return b if b else fallback

    return fallback


def _parse_categories(text: str, domain_root: str) -> list[list[str]]:
    # Amazon-like: "It has the categories of Beauty , Makeup , Face , Powder."
    m = re.search(r"It has the categories of (.*?)(?:\.|$)", text)
    if m:
        cats = [c.strip() for c in m.group(1).split(",")]
        cats = [c for c in cats if c]
        if not cats:
            return [[domain_root]]
        # ensure domain root is first
        if cats[0].lower() != domain_root.lower():
            cats = [domain_root] + cats
        return [cats]

    # Steam-like: "The tags are Action , Adventure , ..."
    m = re.search(r"The tags are (.*?)(?:\.|$)", text)
    if m:
        tags = [c.strip() for c in m.group(1).split(",")]
        tags = [c for c in tags if c]
        if not tags:
            return [[domain_root]]
        return [[domain_root] + tags]

    return [[domain_root]]


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_gzip_jsonl_eval(path: Path, rows: list[dict]) -> None:
    # RQ_VAE_Recommender / LLM_RecSys_ID meta.json.gz expects one dict per line
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(repr(r) + "\n")


def _gen_leave_two_out_sequences(
    train_dict: dict[int, list[int]],
    val_dict: dict[int, list[int]],
    test_dict: dict[int, list[int]],
    *,
    min_train_len: int = 2,
) -> dict[int, list[int]]:
    """Return per-user full sequences (0-based item ids) with 2 held-out items at end.

    full_seq = train + [val0] + [test0]
    """
    out: dict[int, list[int]] = {}
    for u, train_seq in train_dict.items():
        train_seq = list(map(int, train_seq))
        if len(train_seq) < min_train_len:
            continue

        test_seq = test_dict.get(u, [])
        if not test_seq:
            continue

        val_seq = val_dict.get(u, [])
        if val_seq:
            train_used = train_seq
            val_item = int(val_seq[0])
        else:
            # Fallback: if validation is empty but test exists, synthesize a val item.
            # Prefer taking (val,test) from the test list if it has >=2 items; otherwise
            # hold out the last training item as val and keep the first test as test.
            if len(test_seq) >= 2:
                train_used = train_seq
                val_item = int(test_seq[0])
                test_item = int(test_seq[1])
            else:
                train_used = train_seq[:-1]
                val_item = int(train_seq[-1])
                test_item = int(test_seq[0])

            # Allow short train_used (>=1) in this fallback mode to maximize GT coverage.
            if len(train_used) < 1:
                continue

            out[u] = train_used + [val_item] + [test_item]
            continue

        # Normal mode: require train length >= min_train_len.
        if len(train_used) < min_train_len:
            continue

        out[u] = train_used + [val_item] + [int(test_seq[0])]
    return out


def _gen_rqvae_strict_sequences(
    train_dict: dict[int, list[int]],
    val_dict: dict[int, list[int]],
    test_dict: dict[int, list[int]],
    *,
    min_train_len: int = 2,
) -> dict[int, list[int]]:
    """Generate RQ-VAE sequences without placing any test item into test history.

    For users with validation:
      full_seq = train + [val0] + [test0]

    For users without validation but with test:
      full_seq = train[:-1] + [train_last] + [test0]

    This keeps the RQ-VAE leave-two-out layout while ensuring the exported test
    history never already contains a ground-truth test item from SETRec.
    """
    out: dict[int, list[int]] = {}
    for u, train_seq in train_dict.items():
        train_seq = list(map(int, train_seq))
        if len(train_seq) < min_train_len:
            continue

        test_seq = list(map(int, test_dict.get(u, [])))
        if not test_seq:
            continue

        val_seq = list(map(int, val_dict.get(u, [])))
        if val_seq:
            train_used = train_seq
            val_item = int(val_seq[0])
        else:
            train_used = train_seq[:-1]
            if len(train_used) < 1:
                continue
            val_item = int(train_seq[-1])

        out[u] = train_used + [val_item] + [int(test_seq[0])]
    return out


def _write_sequential_data_txt(path: Path, user2fullseq: dict[int, list[int]]) -> None:
    """Write in LLM_RecSys_ID / RQ_VAE_Recommender format: uid item1 item2 ... (all 1-based)."""
    with path.open("w", encoding="utf-8") as f:
        for u in sorted(user2fullseq.keys()):
            seq = user2fullseq[u]
            # 1-based ids in file
            items = [str(i + 1) for i in seq]
            f.write(str(u + 1) + " " + " ".join(items) + "\n")


def _gen_remapped_variants_from_sequential_data(seq_txt_path: Path) -> None:
    """Re-implement LLM_RecSys_ID/data/sequential_generation.py without cwd side-effects."""

    lines = seq_txt_path.read_text(encoding="utf-8").splitlines()
    lines = [l.strip() for l in lines if l.strip()]
    data = {l.split(" ")[0]: l.split(" ")[1:] for l in lines}

    all_train_sequence = [a for v in data.values() for a in v[:-2]]
    all_sequence = [a for v in data.values() for a in v]

    remap: dict[str, str] = {}
    index = 1
    for item in all_train_sequence:
        if item not in remap:
            remap[item] = str(index)
            index += 1
    for item in all_sequence:
        if item not in remap:
            remap[item] = str(index)
            index += 1

    remapped_data = {k: [remap[a] for a in v] for k, v in data.items()}

    # base remapped
    remapped_path = seq_txt_path.parent / "remapped_sequential_data.txt"
    with remapped_path.open("w", encoding="utf-8") as f:
        for k, v in remapped_data.items():
            f.write(k + " " + " ".join(v) + "\n")

    # variants operate on remapped_sequential_data
    data_lines = remapped_path.read_text(encoding="utf-8").splitlines()
    data_lines = [l.strip() for l in data_lines if l.strip()]

    # randomize
    import random
    randomize_lines = data_lines.copy()
    random.shuffle(randomize_lines)

    data_dict = {l.split(" ")[0]: l.split(" ")[1:] for l in data_lines}

    short_to_long = sorted(data_dict.items(), key=lambda x: len(x[1]))
    long_to_short = sorted(data_dict.items(), key=lambda x: len(x[1]), reverse=True)
    randomize_dict = {l.split(" ")[0]: l.split(" ")[1:] for l in randomize_lines}

    def _remap_ordered(ordered_items: list[tuple[str, list[str]]], out_name: str) -> None:
        mp: dict[str, int] = {}
        new_id = 0
        for _, v in ordered_items:
            for item in v[:-2]:
                if item not in mp:
                    new_id += 1
                    mp[item] = new_id
        out_lines = []
        for k, v in ordered_items:
            remapped_items = [str(mp[item]) if item in mp else item for item in v]
            out_lines.append(str(k) + " " + " ".join(remapped_items))

        (seq_txt_path.parent / out_name).write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    _remap_ordered(short_to_long, "short_to_long_remapped_sequential_data.txt")
    _remap_ordered(long_to_short, "long_to_short_remapped_sequential_data.txt")
    _remap_ordered(list(randomize_dict.items()), "randomize_remapped_sequential_data.txt")


def prepare_llm_recsys_id(
    setrec_root: Path,
    out_repo: Path,
    domain: str,
    *,
    overwrite: bool,
) -> None:
    task_name = f"setrec_{domain}"
    out_dir = out_repo / "data" / task_name
    if out_dir.exists() and overwrite:
        # remove only our generated files
        for fn in [
            "sequential_data.txt",
            "remapped_sequential_data.txt",
            "short_to_long_remapped_sequential_data.txt",
            "long_to_short_remapped_sequential_data.txt",
            "randomize_remapped_sequential_data.txt",
            "datamaps.json",
            "meta.json.gz",
        ]:
            p = out_dir / fn
            if p.exists():
                p.unlink()

    _ensure_dir(out_dir)

    train_dict = _load_npy_dict(setrec_root / domain / "training_dict.npy")
    val_dict = _load_npy_dict(setrec_root / domain / "validation_dict.npy")
    test_dict = _load_npy_dict(setrec_root / domain / "testing_dict.npy")

    user2full = _gen_leave_two_out_sequences(train_dict, val_dict, test_dict)
    _write_sequential_data_txt(out_dir / "sequential_data.txt", user2full)
    _gen_remapped_variants_from_sequential_data(out_dir / "sequential_data.txt")

    # datamaps.json
    # Map raw tokens I0.. to numeric 1.., and U0.. to numeric 1..
    n_users = max(train_dict.keys()) + 1
    n_items = int(np.load(setrec_root / domain / f"{domain}.emb-t5-tdcb.npy").shape[0])

    user2id = {f"U{u}": str(u + 1) for u in range(n_users)}
    id2user = {str(u + 1): f"U{u}" for u in range(n_users)}
    item2id = {f"I{i}": str(i + 1) for i in range(n_items)}
    id2item = {str(i + 1): f"I{i}" for i in range(n_items)}

    datamaps = {
        "user2id": user2id,
        "item2id": item2id,
        "id2user": id2user,
        "id2item": id2item,
        "attribute2id": {},
        "id2attribute": {},
        "attributeid2num": {},
    }
    _write_json(out_dir / "datamaps.json", datamaps)

    # meta.json.gz
    combined = _load_npy_dict(setrec_root / domain / "combine_tdcb_maps.npy")["recid2combine"]
    meta_rows = []
    root_cat = domain.capitalize() if domain != "steam" else "Steam"
    for i in range(n_items):
        text = str(combined.get(i, ""))
        meta_rows.append(
            {
                "asin": f"I{i}",
                "title": _parse_title(text, fallback=f"item_{i}"),
                "brand": _parse_brand(text),
                "categories": _parse_categories(text, root_cat),
                # keep the combined text for downstream use
                "description": text,
            }
        )
    _write_gzip_jsonl_eval(out_dir / "meta.json.gz", meta_rows)


def prepare_letter(
    setrec_root: Path,
    out_repo: Path,
    domain: str,
    *,
    overwrite: bool,
) -> None:
    dataset_name = f"SETRec_{domain}"
    out_dir = out_repo / "data" / dataset_name
    _ensure_dir(out_dir)

    train_dict = _load_npy_dict(setrec_root / domain / "training_dict.npy")
    val_dict = _load_npy_dict(setrec_root / domain / "validation_dict.npy")
    test_dict = _load_npy_dict(setrec_root / domain / "testing_dict.npy")

    user2full = _gen_leave_two_out_sequences(train_dict, val_dict, test_dict)

    # inter.json expects 0-based item ids
    inter = {str(u): seq for u, seq in user2full.items()}
    _write_json(out_dir / f"{dataset_name}.inter.json", inter)

    # Placeholder index file: unique 4-token codes in base-256
    n_items = int(np.load(setrec_root / domain / f"{domain}.emb-t5-tdcb.npy").shape[0])
    index: dict[str, list[str]] = {}

    def to_base256_4(x: int) -> tuple[int, int, int, int]:
        d0 = (x >> 24) & 0xFF
        d1 = (x >> 16) & 0xFF
        d2 = (x >> 8) & 0xFF
        d3 = x & 0xFF
        return d0, d1, d2, d3

    prefixes = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>"]
    for i in range(n_items):
        digits = to_base256_4(i)
        index[str(i)] = [prefixes[j].format(int(digits[j])) for j in range(4)]

    _write_json(out_dir / f"{dataset_name}.index.json", index)


def prepare_etegrec(
    setrec_root: Path,
    out_root: Path,
    domain: str,
    *,
    overwrite: bool,
) -> None:
    # Layout: data_path/<domain>/<domain>.train.jsonl, <domain>.emb_map.json, <domain>_emb_768.npy
    out_dir = out_root / "datasets" / "etegrec_setrec" / domain
    _ensure_dir(out_dir)

    train_dict = _load_npy_dict(setrec_root / domain / "training_dict.npy")
    val_dict = _load_npy_dict(setrec_root / domain / "validation_dict.npy")
    test_dict = _load_npy_dict(setrec_root / domain / "testing_dict.npy")

    n_items = int(np.load(setrec_root / domain / f"{domain}.emb-t5-tdcb.npy").shape[0])

    # emb_map.json: <pad>=0, I0.. -> 1..N
    item2id = {"<pad>": 0}
    for i in range(n_items):
        item2id[f"I{i}"] = i + 1
    _write_json(out_dir / f"{domain}.emb_map.json", item2id)

    # semantic embedding file (symlink)
    target_emb = setrec_root / domain / f"{domain}.emb-t5-tdcb.npy"
    link_emb = out_dir / f"{domain}_emb_768.npy"
    if link_emb.exists() or link_emb.is_symlink():
        if overwrite:
            link_emb.unlink()
    if not link_emb.exists():
        link_emb.symlink_to(target_emb)

    # Build jsonl splits
    def make_examples_from_seq(seq: list[int], *, max_his: int = 50) -> list[dict]:
        ex = []
        if len(seq) < 2:
            return ex
        for j in range(1, len(seq)):
            start = max(0, j - max_his)
            his = [f"I{int(x)}" for x in seq[start:j]]
            tgt = f"I{int(seq[j])}"
            ex.append({"inter_history": his, "target_id": tgt})
        return ex

    train_rows: list[dict] = []
    valid_rows: list[dict] = []
    test_rows: list[dict] = []

    for u, train_seq in train_dict.items():
        train_seq = list(map(int, train_seq))
        train_rows.extend(make_examples_from_seq(train_seq))

        v = val_dict.get(u, [])
        t = test_dict.get(u, [])
        if v:
            valid_rows.append({
                "inter_history": [f"I{int(x)}" for x in train_seq[-50:]],
                "target_id": f"I{int(v[0])}",
            })
        if t:
            his = train_seq + list(map(int, v))
            test_rows.append({
                "inter_history": [f"I{int(x)}" for x in his[-50:]],
                "target_id": f"I{int(t[0])}",
            })

    _write_jsonl(out_dir / f"{domain}.train.jsonl", train_rows)
    _write_jsonl(out_dir / f"{domain}.valid.jsonl", valid_rows)
    _write_jsonl(out_dir / f"{domain}.test.jsonl", test_rows)


def prepare_seater(
    setrec_root: Path,
    out_root: Path,
    domain: str,
    *,
    overwrite: bool,
) -> None:
    out_dir = out_root / "datasets" / "seater_setrec" / domain
    ds_dir = out_dir / "dataset"
    vocab_dir = out_dir / "vocab"
    tree_dir = out_dir / "tree_data_SASREC"
    _ensure_dir(ds_dir)
    _ensure_dir(vocab_dir)
    _ensure_dir(tree_dir)

    train_dict = _load_npy_dict(setrec_root / domain / "training_dict.npy")
    val_dict = _load_npy_dict(setrec_root / domain / "validation_dict.npy")
    test_dict = _load_npy_dict(setrec_root / domain / "testing_dict.npy")

    # SASRec embeddings (N, 64) -> add PAD row at 0 to make (N+1, 64)
    import pickle
    import torch

    sas_path = setrec_root / domain / "SASRec_item_embed.pkl"
    sas = pickle.load(open(sas_path, "rb"))
    if isinstance(sas, torch.Tensor):
        sas = sas.detach().cpu().numpy()
    sas = np.asarray(sas, dtype=np.float32)

    n_items = sas.shape[0]
    pad = np.zeros((1, sas.shape[1]), dtype=np.float32)
    sas_pad = np.concatenate([pad, sas], axis=0)

    out_emb = vocab_dir / f"{domain}_SASREC_item_emb.npy"
    if not out_emb.exists() or overwrite:
        np.save(out_emb, sas_pad)

    # optional placeholder for item_2_attr_mapping
    attr_path = vocab_dir / "item_2_attr_mapping.npy"
    if not attr_path.exists() or overwrite:
        np.save(attr_path, np.zeros((n_items + 1, 1), dtype=np.int64))

    # Write tsv files
    def dump_tsv(path: Path, rows: list[tuple[Any, Any, Any]], header: tuple[str, str, str]):
        with path.open("w", encoding="utf-8") as f:
            f.write("\t".join(header) + "\n")
            for r in rows:
                f.write("\t".join(map(str, r)) + "\n")

    train_rows: list[tuple[Any, Any, Any]] = []
    valid_rows: list[tuple[Any, Any, Any]] = []
    test_rows: list[tuple[Any, Any, Any]] = []

    for u, seq0 in train_dict.items():
        seq = [int(x) + 1 for x in seq0]
        if len(seq) >= 2:
            for i in range(1, len(seq)):
                train_rows.append((int(u), str(seq[:i]), int(seq[i])))

        v = val_dict.get(u, [])
        t = test_dict.get(u, [])
        if v:
            valid_rows.append((int(u), str(seq), str([int(x) + 1 for x in v])))
        if t:
            his = seq + [int(x) + 1 for x in v]
            test_rows.append((int(u), str(his), str([int(x) + 1 for x in t])))

    dump_tsv(ds_dir / "training.tsv", train_rows, ("uid", "his_seq", "next_item"))
    dump_tsv(ds_dir / "validation.tsv", valid_rows, ("uid", "his_seq", "predicting_items"))
    dump_tsv(ds_dir / "test.tsv", test_rows, ("uid", "his_seq", "predicting_items"))

def prepare_eager_setrec(
    setrec_root: Path,
    out_root: Path,
    domain: str,
    *,
    overwrite: bool,
) -> None:
    # Generate an Amazon-style jsonl file for EAGER _read().

    out_dir = out_root / "datasets" / "eager_setrec" / domain
    _ensure_dir(out_dir)

    out_file = out_dir / f"{domain}.jsonl"
    if out_file.exists() and not overwrite:
        return

    train_dict = _load_npy_dict(setrec_root / domain / "training_dict.npy")
    val_dict = _load_npy_dict(setrec_root / domain / "validation_dict.npy")
    test_dict = _load_npy_dict(setrec_root / domain / "testing_dict.npy")

    user2full = _gen_leave_two_out_sequences(train_dict, val_dict, test_dict)

    ts = 0
    with out_file.open("w", encoding="utf-8") as f:
        for u in sorted(user2full.keys()):
            seq = user2full[u]
            for item in seq:
                rec = {
                    "reviewerID": f"U{int(u)}",
                    "asin": f"I{int(item)}",
                    "overall": 1.0,
                    "unixReviewTime": int(ts),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                ts += 1



def prepare_rqvae_recommender_raw(
    setrec_root: Path,
    out_root: Path,
    domain: str,
    *,
    overwrite: bool,
    target_domain: str | None = None,
    strict_no_test_history: bool = False,
) -> None:
    # Create an AmazonReviews-like raw dataset under datasets/rqvae_recommender/setrec/raw/<domain>/
    source_domain = domain
    target_domain = target_domain or domain
    root = out_root / "datasets" / "rqvae_recommender" / "setrec"
    raw_dir = root / "raw" / target_domain
    _ensure_dir(raw_dir)

    train_dict = _load_npy_dict(setrec_root / source_domain / "training_dict.npy")
    val_dict = _load_npy_dict(setrec_root / source_domain / "validation_dict.npy")
    test_dict = _load_npy_dict(setrec_root / source_domain / "testing_dict.npy")

    if strict_no_test_history:
        user2full = _gen_rqvae_strict_sequences(train_dict, val_dict, test_dict)
    else:
        user2full = _gen_leave_two_out_sequences(train_dict, val_dict, test_dict)
    _write_sequential_data_txt(raw_dir / "sequential_data.txt", user2full)

    # datamaps.json / meta.json.gz
    n_users = max(train_dict.keys()) + 1
    n_items = int(np.load(setrec_root / source_domain / f"{source_domain}.emb-t5-tdcb.npy").shape[0])

    user2id = {f"U{u}": str(u + 1) for u in range(n_users)}
    id2user = {str(u + 1): f"U{u}" for u in range(n_users)}
    item2id = {f"I{i}": str(i + 1) for i in range(n_items)}
    id2item = {str(i + 1): f"I{i}" for i in range(n_items)}

    datamaps = {
        "user2id": user2id,
        "item2id": item2id,
        "id2user": id2user,
        "id2item": id2item,
        "attribute2id": {},
        "id2attribute": {},
        "attributeid2num": {},
    }
    _write_json(raw_dir / "datamaps.json", datamaps)

    combined = _load_npy_dict(setrec_root / source_domain / "combine_tdcb_maps.npy")["recid2combine"]
    meta_rows = []
    root_cat = source_domain.capitalize() if source_domain != "steam" else "Steam"
    for i in range(n_items):
        text = str(combined.get(i, ""))
        meta_rows.append(
            {
                "asin": f"I{i}",
                "title": _parse_title(text, fallback=f"item_{i}"),
                "brand": _parse_brand(text),
                "categories": _parse_categories(text, root_cat),
                "price": None,
            }
        )
    _write_gzip_jsonl_eval(raw_dir / "meta.json.gz", meta_rows)

    # Precomputed item embeddings for fast processing (used by our patch)
    emb = np.load(setrec_root / source_domain / f"{source_domain}.emb-t5-tdcb.npy").astype(np.float32, copy=False)
    np.save(raw_dir / "item_emb.npy", emb)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--setrec_root",
        type=Path,
        default=Path("/data/xqp_data/RecSys26/datasets/setrec_data"),
    )
    ap.add_argument(
        "--recsys_root",
        type=Path,
        default=Path("/data/xqp_data/RecSys26"),
    )
    ap.add_argument(
        "--domains",
        nargs="+",
        default=["beauty", "toys", "sports", "steam"],
    )
    ap.add_argument(
        "--rqvae_strict_domains",
        nargs="*",
        default=[],
        help="Domains for which an additional strict RQ-VAE raw split is generated.",
    )
    ap.add_argument(
        "--rqvae_strict_suffix",
        default="_strict",
        help="Suffix appended to strict RQ-VAE raw split names.",
    )
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    llm_repo = args.recsys_root / "third_party" / "LLM_RecSys_ID"
    letter_repo = args.recsys_root / "third_party" / "LETTER"

    for domain in args.domains:
        if not (args.setrec_root / domain).exists():
            raise FileNotFoundError(f"Unknown domain {domain}: {args.setrec_root/domain}")

        prepare_llm_recsys_id(args.setrec_root, llm_repo, domain, overwrite=args.overwrite)
        prepare_letter(args.setrec_root, letter_repo, domain, overwrite=args.overwrite)
        prepare_etegrec(args.setrec_root, args.recsys_root, domain, overwrite=args.overwrite)
        prepare_seater(args.setrec_root, args.recsys_root, domain, overwrite=args.overwrite)
        prepare_eager_setrec(args.setrec_root, args.recsys_root, domain, overwrite=args.overwrite)
        prepare_rqvae_recommender_raw(args.setrec_root, args.recsys_root, domain, overwrite=args.overwrite)
        if domain in set(args.rqvae_strict_domains):
            strict_domain = f"{domain}{args.rqvae_strict_suffix}"
            prepare_rqvae_recommender_raw(
                args.setrec_root,
                args.recsys_root,
                domain,
                overwrite=args.overwrite,
                target_domain=strict_domain,
                strict_no_test_history=True,
            )
            print(f"[OK] prepared strict RQ-VAE adapter: {domain} -> {strict_domain}")

    print("[OK] prepared adapters for domains:", ", ".join(args.domains))


if __name__ == "__main__":
    main()
