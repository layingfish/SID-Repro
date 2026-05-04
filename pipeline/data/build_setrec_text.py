
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr, flush=True)


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _join_nonempty(parts: list[str], *, sep: str = " ") -> str:
    return sep.join([p for p in parts if p.strip()]).strip()


def build_item_prompt(
    *,
    title: str,
    description: str = "",
    categories: list[str] | None = None,
    brand: str = "",
) -> str:
    categories = categories or []

    parts: list[str] = []
    if title:
        parts.append(f"The item title is {title}.")
    if description:
        desc = description.strip()
        if not desc.endswith("."):
            desc += "."
        parts.append(desc)
    if categories:
        cats = " , ".join([c for c in categories if c.strip()])
        if cats:
            parts.append(f"It has the categories of {cats}.")
    if brand:
        parts.append(f"The brand of this item is {brand}.")
    return _join_nonempty(parts)


def build_yelp_prompt(*, name: str, categories: list[str], city: str, state: str) -> str:
    parts: list[str] = []
    if name:
        parts.append(f"The business name is {name}.")
    if categories:
        cats = " , ".join([c for c in categories if c.strip()])
        if cats:
            parts.append(f"It has the categories of {cats}.")
    loc = _join_nonempty([city, state], sep=", ")
    if loc:
        parts.append(f"It is located in {loc}.")
    return _join_nonempty(parts)


def load_item_map_reverse(path: Path) -> dict[int, Any]:
    obj = np.load(path, allow_pickle=True).item()
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in {path}, got {type(obj)}")
    out: dict[int, Any] = {}
    for k, v in obj.items():
        out[int(k)] = v
    return out


def load_microlens_titles(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    first_line = path.open("r", encoding="utf-8", newline="").readline()
    first_cols = next(csv.reader([first_line]))
    lower = [c.strip().lower() for c in first_cols]

    if len(lower) >= 2 and lower[0] in {"item", "item_id", "video_id"}:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            header_lower = [c.strip().lower() for c in header]
            item_idx = 0
            title_idx = header_lower.index("title") if "title" in header_lower else 1
            for row in reader:
                try:
                    out[int(row[item_idx])] = _safe_str(row[title_idx])
                except Exception:
                    continue
    else:


        with path.open("r", encoding="utf-8", newline="") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if "," not in line:
                    continue
                item_raw, title = line.split(",", 1)
                try:
                    out[int(item_raw)] = _safe_str(title)
                except Exception:
                    continue
    return out


def load_amazon23_meta(path: Path, *, need_asins: set[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}

    t0 = time.time()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if i % 200_000 == 0:
                dt = max(time.time() - t0, 1e-6)
                eprint(f"[meta] lines={i} kept={len(out)} ({i/dt:.1f} lines/s)")

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                continue

            asin = obj.get("parent_asin")
            if not asin or asin not in need_asins:
                continue

            out[str(asin)] = obj
            if len(out) >= len(need_asins):
                break

    return out


def load_yelp_business(path: Path, *, need_biz: set[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}

    t0 = time.time()
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if i % 200_000 == 0:
                dt = max(time.time() - t0, 1e-6)
                eprint(f"[biz] lines={i} kept={len(out)} ({i/dt:.1f} lines/s)")

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                continue

            biz = obj.get("business_id")
            if not biz or biz not in need_biz:
                continue

            out[str(biz)] = obj
            if len(out) >= len(need_biz):
                break

    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build SETRec combine_tdcb_maps.npy (recid2combine) for custom datasets."
    )
    ap.add_argument("--dataset", required=True, choices=["amazon23_vg", "microlens_50k", "microlens_100k", "microlens_1m", "yelp"])
    ap.add_argument(
        "--setrec_data_root",
        type=Path,
        default=Path(os.environ.get("SETREC_DATA_ROOT", "data/setrec_data")),
        help="SETRec data root containing <dataset>/item_map_reverse.npy",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow_missing", action="store_true", help="Fill missing metadata with empty prompt")

    ap.add_argument(
        "--microlens_titles",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", "data/raw")) / "microlens_50k" / "MicroLens-50k_titles.csv",
    )
    ap.add_argument(
        "--microlens_title_fallback",
        type=Path,
        default=None,
        help="Optional MicroLens item-id,text CSV used when the primary title is empty.",
    )
    ap.add_argument(
        "--amazon_meta",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", "data/raw")) / "amazon23_vg" / "meta_Video_Games.jsonl.gz",
    )
    ap.add_argument(
        "--yelp_business",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", "data/raw")) / "yelp" / "yelp_academic_dataset_business.json",
    )

    args = ap.parse_args()

    domain_dir = args.setrec_data_root / args.dataset
    item_map_reverse = load_item_map_reverse(domain_dir / "item_map_reverse.npy")

    out_path = domain_dir / "combine_tdcb_maps.npy"
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refuse to overwrite: {out_path}")

    if args.dataset.startswith("microlens_"):
        title_map = load_microlens_titles(args.microlens_titles)
        fallback_map = load_microlens_titles(args.microlens_title_fallback) if args.microlens_title_fallback else {}
        recid2combine: dict[int, str] = {}
        missing = 0
        fallback_used = 0
        for recid, item_old in sorted(item_map_reverse.items()):
            title = title_map.get(int(item_old), "")
            if not title and fallback_map:
                title = fallback_map.get(int(item_old), "")
                if title:
                    fallback_used += 1
            if not title:
                missing += 1
            recid2combine[int(recid)] = build_item_prompt(title=title)
        eprint(
            f"[microlens] n_items={len(item_map_reverse)} missing_titles={missing} fallback_used={fallback_used} titles_src={args.microlens_titles}"
        )

    elif args.dataset == "amazon23_vg":
        need_asins = {str(v) for v in item_map_reverse.values()}
        meta = load_amazon23_meta(args.amazon_meta, need_asins=need_asins)
        recid2combine = {}
        missing = 0
        for recid, asin in sorted(item_map_reverse.items()):
            obj = meta.get(str(asin))
            if obj is None:
                missing += 1
                recid2combine[int(recid)] = "" if args.allow_missing else ""
                continue

            title = _safe_str(obj.get("title"))
            desc_raw = obj.get("description")
            if isinstance(desc_raw, list):
                desc = " ".join([_safe_str(x) for x in desc_raw])
            else:
                desc = _safe_str(desc_raw)
            categories = obj.get("categories")
            if isinstance(categories, list):
                cats = [_safe_str(x) for x in categories]
            else:
                cats = []
            brand = _safe_str(obj.get("store"))

            recid2combine[int(recid)] = build_item_prompt(
                title=title,
                description=desc,
                categories=cats,
                brand=brand,
            )

        eprint(
            f"[amazon23] n_items={len(item_map_reverse)} missing_in_meta={missing} meta_src={args.amazon_meta}"
        )

    else:
        need_biz = {str(v) for v in item_map_reverse.values()}
        biz_map = load_yelp_business(args.yelp_business, need_biz=need_biz)
        recid2combine = {}
        missing = 0
        for recid, biz in sorted(item_map_reverse.items()):
            obj = biz_map.get(str(biz))
            if obj is None:
                missing += 1
                recid2combine[int(recid)] = "" if args.allow_missing else ""
                continue

            name = _safe_str(obj.get("name"))
            cats_raw = _safe_str(obj.get("categories"))
            cats = [c.strip() for c in cats_raw.split(",") if c.strip()] if cats_raw else []
            city = _safe_str(obj.get("city"))
            state = _safe_str(obj.get("state"))

            recid2combine[int(recid)] = build_yelp_prompt(
                name=name,
                categories=cats,
                city=city,
                state=state,
            )

        eprint(
            f"[yelp] n_items={len(item_map_reverse)} missing_in_business={missing} business_src={args.yelp_business}"
        )

    if missing and not args.allow_missing:
        raise ValueError(f"Found missing metadata entries: {missing} (rerun with --allow_missing to bypass)")

    np.save(out_path, {"recid2combine": recid2combine})
    eprint(f"[OK] wrote {out_path} (n={len(recid2combine)})")


if __name__ == "__main__":
    main()
