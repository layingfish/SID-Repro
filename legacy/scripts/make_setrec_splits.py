#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Hashable

import numpy as np


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def load_amazon23_reviews_jsonl_gz(
    path: Path,
    *,
    user_key: str = "user_id",
    item_key: str = "parent_asin",
    item_fallback_key: str = "asin",
    time_key: str = "timestamp",
    limit: int | None = None,
    progress_every: int = 1_000_000,
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """Load Amazon Reviews'23 JSONL.GZ into user->item->timestamp.

    Keeps only the latest timestamp for duplicated (user,item).
    """

    interactions: dict[str, dict[str, int]] = {}
    stats = {"lines": 0, "kept": 0, "bad": 0, "dupe": 0}

    t0 = time.time()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if limit is not None and stats["lines"] >= limit:
                break
            stats["lines"] += 1
            if progress_every > 0 and stats["lines"] % progress_every == 0:
                dt = max(time.time() - t0, 1e-6)
                eprint(
                    f"[load] lines={stats['lines']} users={len(interactions)} kept={stats['kept']} "
                    f"bad={stats['bad']} dupe={stats['dupe']} ({stats['lines']/dt:.1f} lines/s)"
                )

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                stats["bad"] += 1
                continue

            user = obj.get(user_key)
            item = obj.get(item_key) or obj.get(item_fallback_key)
            time_raw = obj.get(time_key)

            if not user or not item or time_raw is None:
                stats["bad"] += 1
                continue

            ts = _safe_int(time_raw)
            if ts is None:
                stats["bad"] += 1
                continue

            if user not in interactions:
                interactions[user] = {}

            prev = interactions[user].get(item)
            if prev is None:
                interactions[user][item] = ts
                stats["kept"] += 1
            else:
                stats["dupe"] += 1
                if ts > prev:
                    interactions[user][item] = ts

    return interactions, stats


def load_amazon23_benchmark_csv_gz(
    path: Path,
    *,
    user_col: str = "user_id",
    item_col: str = "parent_asin",
    time_col: str = "timestamp",
    limit: int | None = None,
    progress_every: int = 1_000_000,
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """Load Amazon Reviews'23 benchmark CSV.GZ (e.g., 5-core rating_only) into user->item->timestamp.

    Expected columns include at least: user_id, parent_asin, timestamp.
    Keeps only the latest timestamp for duplicated (user,item).
    """

    interactions: dict[str, dict[str, int]] = {}
    stats = {"rows": 0, "kept": 0, "bad": 0, "dupe": 0}

    t0 = time.time()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if limit is not None and stats["rows"] >= limit:
                break
            stats["rows"] += 1
            if progress_every > 0 and stats["rows"] % progress_every == 0:
                dt = max(time.time() - t0, 1e-6)
                eprint(
                    f"[load] rows={stats['rows']} users={len(interactions)} kept={stats['kept']} "
                    f"bad={stats['bad']} dupe={stats['dupe']} ({stats['rows']/dt:.1f} rows/s)"
                )

            user = row.get(user_col)
            item = row.get(item_col)
            time_raw = row.get(time_col)

            if not user or not item or time_raw is None:
                stats["bad"] += 1
                continue

            ts = _safe_int(time_raw)
            if ts is None:
                stats["bad"] += 1
                continue

            if user not in interactions:
                interactions[user] = {}

            prev = interactions[user].get(item)
            if prev is None:
                interactions[user][item] = ts
                stats["kept"] += 1
            else:
                stats["dupe"] += 1
                if ts > prev:
                    interactions[user][item] = ts

    return interactions, stats


def load_yelp_review_jsonl(
    path: Path,
    *,
    user_key: str = "user_id",
    item_key: str = "business_id",
    time_key: str = "date",
    limit: int | None = None,
    progress_every: int = 1_000_000,
) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    """Load Yelp review JSON (NDJSON) into user->item->date_str.

    The Yelp Open Dataset `date` field is typically `YYYY-MM-DD HH:MM:SS`.
    Lexicographic order matches chronological order, so we keep it as a string
    for speed (avoids parsing millions of timestamps).

    Keeps only the latest date for duplicated (user,item).
    """

    interactions: dict[str, dict[str, str]] = {}
    stats = {"lines": 0, "kept": 0, "bad": 0, "dupe": 0}

    t0 = time.time()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and stats["lines"] >= limit:
                break
            stats["lines"] += 1
            if progress_every > 0 and stats["lines"] % progress_every == 0:
                dt = max(time.time() - t0, 1e-6)
                eprint(
                    f"[load] lines={stats['lines']} users={len(interactions)} kept={stats['kept']} "
                    f"bad={stats['bad']} dupe={stats['dupe']} ({stats['lines']/dt:.1f} lines/s)"
                )

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                stats["bad"] += 1
                continue

            user = obj.get(user_key)
            item = obj.get(item_key)
            time_str = obj.get(time_key)

            if not user or not item or not time_str:
                stats["bad"] += 1
                continue

            time_str = str(time_str)

            if user not in interactions:
                interactions[user] = {}

            prev = interactions[user].get(item)
            if prev is None:
                interactions[user][item] = time_str
                stats["kept"] += 1
            else:
                stats["dupe"] += 1
                if time_str > prev:
                    interactions[user][item] = time_str

    return interactions, stats


def load_microlens_pairs_csv(
    path: Path,
    *,
    user_col: str = "user",
    item_col: str = "item",
    time_col: str = "timestamp",
    limit: int | None = None,
    progress_every: int = 1_000_000,
) -> tuple[dict[int, dict[int, int]], dict[str, int]]:
    """Load MicroLens-50k_pairs.csv into user->item->timestamp (ms)."""

    interactions: dict[int, dict[int, int]] = {}
    stats = {"rows": 0, "kept": 0, "bad": 0, "dupe": 0}

    t0 = time.time()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if limit is not None and stats["rows"] >= limit:
                break
            stats["rows"] += 1
            if progress_every > 0 and stats["rows"] % progress_every == 0:
                dt = max(time.time() - t0, 1e-6)
                eprint(
                    f"[load] rows={stats['rows']} users={len(interactions)} kept={stats['kept']} "
                    f"bad={stats['bad']} dupe={stats['dupe']} ({stats['rows']/dt:.1f} rows/s)"
                )

            try:
                user = int(row[user_col])
                item = int(row[item_col])
                ts = int(row[time_col])
            except Exception:
                stats["bad"] += 1
                continue

            if user not in interactions:
                interactions[user] = {}

            prev = interactions[user].get(item)
            if prev is None:
                interactions[user][item] = ts
                stats["kept"] += 1
            else:
                stats["dupe"] += 1
                if ts > prev:
                    interactions[user][item] = ts

    return interactions, stats


def select_kcore(
    interactions: dict[Hashable, dict[Hashable, Any]],
    *,
    k: int,
    verbose: bool = True,
) -> dict[Hashable, dict[Hashable, Any]]:
    """Iteratively apply user/item k-core on a user->item->time dict.

    This is a practical (not asymptotically optimal) implementation intended to
    match the SETRec preprocessing notebook behavior.
    """

    if k <= 0:
        return interactions

    round_id = 0
    while True:
        round_id += 1

        item_cnt: Counter = Counter()
        for item_time in interactions.values():
            item_cnt.update(item_time.keys())

        bad_items = {item for item, cnt in item_cnt.items() if cnt < k}

        if bad_items:
            for user in list(interactions.keys()):
                item_time = interactions[user]
                for item in list(item_time.keys()):
                    if item in bad_items:
                        del item_time[item]

        bad_users = [user for user, item_time in interactions.items() if len(item_time) < k]
        for user in bad_users:
            del interactions[user]

        if verbose:
            eprint(
                f"[kcore] round={round_id} k={k} users={len(interactions)} items={len(item_cnt)} "
                f"drop_items={len(bad_items)} drop_users={len(bad_users)}"
            )

        if not bad_items and not bad_users:
            break

    return interactions


def build_global_time_list(interactions: dict[Hashable, dict[Hashable, Any]]) -> list[Any]:
    times: list[Any] = []
    for item_time in interactions.values():
        times.extend(item_time.values())
    times.sort()
    return times


def split_by_global_time(
    interactions: dict[Hashable, dict[Hashable, Any]],
    *,
    split_ratio: float = 0.17,
    val_multiplier: float = 1.8,
) -> tuple[
    Any,
    Any,
    dict[Hashable, list[Hashable]],
    dict[Hashable, list[Hashable]],
    dict[Hashable, list[Hashable]],
]:
    """Replicate SETRec notebook: global time cutoffs -> train/val/test."""

    times = build_global_time_list(interactions)
    if not times:
        raise ValueError("No timestamps found")

    test_num = int(len(times) * split_ratio)
    if test_num <= 0:
        raise ValueError(f"Too few interactions ({len(times)}) for split_ratio={split_ratio}")

    split_time1 = times[-test_num]
    split_time2 = times[-math.ceil(val_multiplier * test_num)]

    train_old: dict[Hashable, list[Hashable]] = {}
    val_old: dict[Hashable, list[Hashable]] = {}
    test_old: dict[Hashable, list[Hashable]] = {}

    for user, item_time in interactions.items():
        pairs = sorted(item_time.items(), key=lambda kv: kv[1])
        tr, va, te = [], [], []
        for item, t in pairs:
            if t < split_time2:
                tr.append(item)
            elif t < split_time1:
                va.append(item)
            else:
                te.append(item)
        train_old[user] = tr
        val_old[user] = va
        test_old[user] = te

    return split_time1, split_time2, train_old, val_old, test_old


def drop_short_train_users(
    train_old: dict[Hashable, list[Hashable]],
    val_old: dict[Hashable, list[Hashable]],
    test_old: dict[Hashable, list[Hashable]],
    *,
    min_train_len: int = 2,
) -> tuple[
    dict[Hashable, list[Hashable]],
    dict[Hashable, list[Hashable]],
    dict[Hashable, list[Hashable]],
    int,
]:
    dropped = 0
    for user in list(train_old.keys()):
        if len(train_old[user]) >= min_train_len:
            continue
        dropped += 1
        del train_old[user]
        del val_old[user]
        del test_old[user]
    return train_old, val_old, test_old, dropped


def make_maps(
    train_old: dict[Hashable, list[Hashable]],
    val_old: dict[Hashable, list[Hashable]],
    test_old: dict[Hashable, list[Hashable]],
    *,
    item_shuffle_seed: int = 2023,
) -> tuple[dict[Hashable, int], dict[Hashable, int]]:
    user_list = list(train_old.keys())

    item_set: set[Hashable] = set()
    for user in user_list:
        item_set.update(train_old[user])
        item_set.update(val_old[user])
        item_set.update(test_old[user])

    item_list = sorted(item_set)
    rnd = random.Random(item_shuffle_seed)
    rnd.shuffle(item_list)

    user_map = {old: new for new, old in enumerate(user_list)}
    item_map = {old: new for new, old in enumerate(item_list)}
    return user_map, item_map


def remap_splits(
    train_old: dict[Hashable, list[Hashable]],
    val_old: dict[Hashable, list[Hashable]],
    test_old: dict[Hashable, list[Hashable]],
    user_map: dict[Hashable, int],
    item_map: dict[Hashable, int],
) -> dict[str, Any]:
    train: dict[int, list[int]] = {}
    val: dict[int, list[int]] = {}
    test: dict[int, list[int]] = {}

    val_warm: dict[int, list[int]] = {}
    val_cold: dict[int, list[int]] = {}
    test_warm: dict[int, list[int]] = {}
    test_cold: dict[int, list[int]] = {}

    warm_items: set[int] = set()
    cold_items: set[int] = set()

    for user_old, user_new in user_map.items():
        items_old = train_old[user_old]
        items_new = [int(item_map[i]) for i in items_old]
        train[int(user_new)] = items_new
        for item_new in items_new:
            warm_items.add(int(item_new))

    for user_old, user_new in user_map.items():
        items = [int(item_map[i]) for i in val_old[user_old]]
        val[int(user_new)] = items
        w, c = [], []
        for item in items:
            if item in warm_items:
                w.append(item)
            else:
                c.append(item)
                cold_items.add(item)
        val_warm[int(user_new)] = w
        val_cold[int(user_new)] = c

    for user_old, user_new in user_map.items():
        items = [int(item_map[i]) for i in test_old[user_old]]
        test[int(user_new)] = items
        w, c = [], []
        for item in items:
            if item in warm_items:
                w.append(item)
            else:
                c.append(item)
                cold_items.add(item)
        test_warm[int(user_new)] = w
        test_cold[int(user_new)] = c

    return {
        "training_dict": train,
        "validation_dict": val,
        "testing_dict": test,
        "validation_warm_dict": val_warm,
        "validation_cold_dict": val_cold,
        "testing_warm_dict": test_warm,
        "testing_cold_dict": test_cold,
        "warm_item": warm_items,
        "cold_item": cold_items,
    }


def save_npy(path: Path, obj: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite: {path}")
    np.save(path, obj)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate SETRec-style split npy files from raw interactions")
    ap.add_argument(
        "--format",
        required=True,
        choices=["amazon23", "amazon23_benchmark_csv", "yelp", "microlens"],
        help="Input dataset format",
    )
    ap.add_argument("--input", required=True, type=Path, help="Path to raw interaction file")
    ap.add_argument("--out_root", required=True, type=Path, help="Output root directory")
    ap.add_argument("--dataset", required=True, help="Output dataset directory name")

    ap.add_argument("--k_core", type=int, default=0, help="User/item k-core (0 disables)")
    ap.add_argument("--split_ratio", type=float, default=0.17)
    ap.add_argument("--val_multiplier", type=float, default=1.8)
    ap.add_argument("--min_train_len", type=int, default=2)
    ap.add_argument("--item_shuffle_seed", type=int, default=2023)

    ap.add_argument("--limit", type=int, default=None, help="Debug: limit raw lines/rows")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--progress_every", type=int, default=1_000_000)

    ap.add_argument("--amazon_user_key", default="user_id")
    ap.add_argument("--amazon_item_key", default="parent_asin")
    ap.add_argument("--amazon_item_fallback_key", default="asin")
    ap.add_argument("--amazon_time_key", default="timestamp")

    ap.add_argument("--yelp_user_key", default="user_id")
    ap.add_argument("--yelp_item_key", default="business_id")
    ap.add_argument("--yelp_time_key", default="date")

    ap.add_argument("--microlens_user_col", default="user")
    ap.add_argument("--microlens_item_col", default="item")
    ap.add_argument("--microlens_time_col", default="timestamp")

    args = ap.parse_args()

    out_dir = args.out_root / args.dataset
    ensure_dir(out_dir)

    eprint(f"[load] format={args.format} input={args.input}")

    if args.format == "amazon23":
        interactions, st = load_amazon23_reviews_jsonl_gz(
            args.input,
            user_key=args.amazon_user_key,
            item_key=args.amazon_item_key,
            item_fallback_key=args.amazon_item_fallback_key,
            time_key=args.amazon_time_key,
            limit=args.limit,
            progress_every=args.progress_every,
        )
        eprint(f"[load] stats={st} users={len(interactions)}")

    elif args.format == "amazon23_benchmark_csv":
        interactions, st = load_amazon23_benchmark_csv_gz(
            args.input,
            user_col=args.amazon_user_key,
            item_col=args.amazon_item_key,
            time_col=args.amazon_time_key,
            limit=args.limit,
            progress_every=args.progress_every,
        )
        eprint(f"[load] stats={st} users={len(interactions)}")

    elif args.format == "yelp":
        interactions, st = load_yelp_review_jsonl(
            args.input,
            user_key=args.yelp_user_key,
            item_key=args.yelp_item_key,
            time_key=args.yelp_time_key,
            limit=args.limit,
            progress_every=args.progress_every,
        )
        eprint(f"[load] stats={st} users={len(interactions)}")

    else:
        interactions, st = load_microlens_pairs_csv(
            args.input,
            user_col=args.microlens_user_col,
            item_col=args.microlens_item_col,
            time_col=args.microlens_time_col,
            limit=args.limit,
            progress_every=args.progress_every,
        )
        eprint(f"[load] stats={st} users={len(interactions)}")

    if args.k_core > 0:
        eprint(f"[kcore] start k={args.k_core}")
        interactions = select_kcore(interactions, k=args.k_core, verbose=True)
        eprint(f"[kcore] done users={len(interactions)}")

    split_time1, split_time2, train_old, val_old, test_old = split_by_global_time(
        interactions,
        split_ratio=args.split_ratio,
        val_multiplier=args.val_multiplier,
    )
    eprint(f"[split] split_time1={split_time1} split_time2={split_time2}")

    train_old, val_old, test_old, dropped = drop_short_train_users(
        train_old,
        val_old,
        test_old,
        min_train_len=args.min_train_len,
    )
    eprint(f"[split] dropped_users_train<{args.min_train_len}: {dropped} remaining_users={len(train_old)}")

    user_map, item_map = make_maps(
        train_old,
        val_old,
        test_old,
        item_shuffle_seed=args.item_shuffle_seed,
    )
    user_map_rev = {v: k for k, v in user_map.items()}
    item_map_rev = {v: k for k, v in item_map.items()}

    save_npy(out_dir / "user_map.npy", user_map, overwrite=args.overwrite)
    save_npy(out_dir / "item_map.npy", item_map, overwrite=args.overwrite)
    save_npy(out_dir / "user_map_reverse.npy", user_map_rev, overwrite=args.overwrite)
    save_npy(out_dir / "item_map_reverse.npy", item_map_rev, overwrite=args.overwrite)

    remapped = remap_splits(train_old, val_old, test_old, user_map, item_map)
    save_npy(out_dir / "training_dict.npy", remapped["training_dict"], overwrite=args.overwrite)
    save_npy(out_dir / "validation_dict.npy", remapped["validation_dict"], overwrite=args.overwrite)
    save_npy(out_dir / "testing_dict.npy", remapped["testing_dict"], overwrite=args.overwrite)
    save_npy(out_dir / "validation_warm_dict.npy", remapped["validation_warm_dict"], overwrite=args.overwrite)
    save_npy(out_dir / "validation_cold_dict.npy", remapped["validation_cold_dict"], overwrite=args.overwrite)
    save_npy(out_dir / "testing_warm_dict.npy", remapped["testing_warm_dict"], overwrite=args.overwrite)
    save_npy(out_dir / "testing_cold_dict.npy", remapped["testing_cold_dict"], overwrite=args.overwrite)
    save_npy(out_dir / "warm_item.npy", remapped["warm_item"], overwrite=args.overwrite)
    save_npy(out_dir / "cold_item.npy", remapped["cold_item"], overwrite=args.overwrite)

    manifest = {
        "generated_utc": datetime.utcnow().isoformat() + "Z",
        "format": args.format,
        "input": str(args.input),
        "dataset": args.dataset,
        "k_core": args.k_core,
        "split_ratio": args.split_ratio,
        "val_multiplier": args.val_multiplier,
        "min_train_len": args.min_train_len,
        "item_shuffle_seed": args.item_shuffle_seed,
        "n_users": len(user_map),
        "n_items": len(item_map),
        "split_time1": str(split_time1),
        "split_time2": str(split_time2),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    eprint(f"[OK] wrote splits to {out_dir}")


if __name__ == "__main__":
    main()
