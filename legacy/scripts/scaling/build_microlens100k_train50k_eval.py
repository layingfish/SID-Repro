#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def load_npy(path: Path) -> Any:
    return np.load(path, allow_pickle=True).item()


def save_npy(path: Path, obj: Any) -> None:
    np.save(path, obj)


def reset_link_or_copy(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_raw_key(x: Any) -> int:
    if isinstance(x, (np.integer, int)):
        return int(x)
    return int(str(x))


def remap_eval50(
    *,
    eval50_dir: Path,
    user_map_100: dict[Any, int],
    item_map_100: dict[Any, int],
) -> tuple[dict[int, list[int]], set[int], dict[str, Any]]:
    user_rev_50 = load_npy(eval50_dir / "user_map_reverse.npy")
    item_rev_50 = load_npy(eval50_dir / "item_map_reverse.npy")
    test50 = load_npy(eval50_dir / "testing_dict.npy")
    warm50_raw = np.load(eval50_dir / "warm_item.npy", allow_pickle=True)
    try:
        warm50_obj = warm50_raw.item()
    except Exception:
        warm50_obj = warm50_raw.tolist()
    warm50_ids = set(warm50_obj.keys()) if hasattr(warm50_obj, "keys") else set(warm50_obj)

    eval_test: dict[int, list[int]] = {}
    missing_users: list[Any] = []
    missing_items: set[Any] = set()

    for uid50, items50 in test50.items():
        raw_user = normalize_raw_key(user_rev_50[int(uid50)])
        if raw_user not in user_map_100:
            missing_users.append(raw_user)
            continue
        uid100 = int(user_map_100[raw_user])

        out_items: list[int] = []
        values = items50.tolist() if isinstance(items50, np.ndarray) else list(items50)
        for item50 in values:
            raw_item = normalize_raw_key(item_rev_50[int(item50)])
            if raw_item not in item_map_100:
                missing_items.add(raw_item)
                continue
            out_items.append(int(item_map_100[raw_item]))
        eval_test[uid100] = out_items

    eval_warm_items: set[int] = set()
    for item50 in warm50_ids:
        raw_item = normalize_raw_key(item_rev_50[int(item50)])
        if raw_item in item_map_100:
            eval_warm_items.add(int(item_map_100[raw_item]))
        else:
            missing_items.add(raw_item)

    stats = {
        "eval50_users": len(eval_test),
        "eval50_gt_items": int(sum(len(v) for v in eval_test.values())),
        "eval50_warm_items": len(eval_warm_items),
        "missing_eval50_users_in_100": len(missing_users),
        "missing_eval50_items_in_100": len(missing_items),
        "sample_missing_users": missing_users[:10],
        "sample_missing_items": sorted(missing_items)[:10],
    }
    return eval_test, eval_warm_items, stats


def extend_user_map_with_eval50_users(
    *,
    user_map_100: dict[Any, int],
    eval50_dir: Path,
) -> tuple[dict[Any, int], dict[str, Any]]:
    user_map = dict(user_map_100)
    user_rev_50 = load_npy(eval50_dir / "user_map_reverse.npy")
    added: list[int] = []
    next_id = max(int(v) for v in user_map.values()) + 1 if user_map else 0
    for raw_user_raw in user_rev_50.values():
        raw_user = normalize_raw_key(raw_user_raw)
        if raw_user in user_map:
            continue
        user_map[raw_user] = next_id
        added.append(raw_user)
        next_id += 1
    return user_map, {
        "added_eval50_users_to_100k_map": len(added),
        "sample_added_eval50_users": added[:10],
    }


def build_splits_from_100k_raw(
    *,
    raw_pairs_path: Path,
    user_map: dict[Any, int],
    item_map: dict[Any, int],
    split_time1: int,
    split_time2: int,
    min_train_len: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    per_user: dict[int, dict[int, int]] = defaultdict(dict)
    stats = {
        "raw_rows": 0,
        "kept_rows": 0,
        "bad_rows": 0,
        "dupe_rows": 0,
        "unmapped_user_rows": 0,
        "unmapped_item_rows": 0,
    }

    with raw_pairs_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["raw_rows"] += 1
            try:
                raw_user = int(row["user"])
                raw_item = int(row["item"])
                ts = int(row["timestamp"])
            except Exception:
                stats["bad_rows"] += 1
                continue

            if raw_user not in user_map:
                stats["unmapped_user_rows"] += 1
                continue
            if raw_item not in item_map:
                stats["unmapped_item_rows"] += 1
                continue

            uid = int(user_map[raw_user])
            iid = int(item_map[raw_item])
            prev = per_user[uid].get(iid)
            if prev is None:
                per_user[uid][iid] = ts
                stats["kept_rows"] += 1
            else:
                stats["dupe_rows"] += 1
                if ts > prev:
                    per_user[uid][iid] = ts

    train: dict[int, list[int]] = {}
    val: dict[int, list[int]] = {}
    test: dict[int, list[int]] = {}
    dropped = 0

    for uid, item_time in per_user.items():
        pairs = sorted(item_time.items(), key=lambda kv: kv[1])
        tr: list[int] = []
        va: list[int] = []
        te: list[int] = []
        for iid, ts in pairs:
            if ts < split_time2:
                tr.append(iid)
            elif ts < split_time1:
                va.append(iid)
            else:
                te.append(iid)
        if len(tr) < min_train_len:
            dropped += 1
            continue
        train[uid] = tr
        val[uid] = va
        test[uid] = te

    warm_items: set[int] = set()
    for items in train.values():
        warm_items.update(int(x) for x in items)

    def split_warm_cold(src: dict[int, list[int]]) -> tuple[dict[int, list[int]], dict[int, list[int]], set[int]]:
        warm: dict[int, list[int]] = {}
        cold: dict[int, list[int]] = {}
        cold_items: set[int] = set()
        for uid, items in src.items():
            w = [int(x) for x in items if int(x) in warm_items]
            c = [int(x) for x in items if int(x) not in warm_items]
            warm[uid] = w
            cold[uid] = c
            cold_items.update(c)
        return warm, cold, cold_items

    val_warm, val_cold, val_cold_items = split_warm_cold(val)
    test_warm, test_cold, test_cold_items = split_warm_cold(test)
    cold_items = val_cold_items | test_cold_items

    split_data = {
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
    split_stats = {
        **stats,
        "dropped_users_train_lt_min": dropped,
        "train_users": len(train),
        "val_users": sum(1 for v in val.values() if v),
        "test_users": sum(1 for v in test.values() if v),
        "train_interactions": int(sum(len(v) for v in train.values())),
        "val_interactions": int(sum(len(v) for v in val.values())),
        "test_interactions": int(sum(len(v) for v in test.values())),
        "warm_items": len(warm_items),
        "cold_items": len(cold_items),
    }
    return split_data, split_stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/data/xqp_data/RecSys26"))
    ap.add_argument("--source_domain", default="microlens_100k")
    ap.add_argument("--eval_domain", default="microlens_50k")
    ap.add_argument("--out_domain", default="microlens_100k_train_50k_eval")
    ap.add_argument("--min_train_len", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    setrec_root = args.root / "third_party" / "SETRec" / "data"
    source_dir = setrec_root / args.source_domain
    eval_dir = setrec_root / args.eval_domain
    out_dir = setrec_root / args.out_domain
    raw_pairs = args.root / "datasets" / args.source_domain / "raw" / "MicroLens-100k_pairs.csv"

    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refuse to overwrite non-empty directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest50 = load_manifest(eval_dir / "manifest.json")
    split_time1 = int(manifest50["split_time1"])
    split_time2 = int(manifest50["split_time2"])

    source_user_map = load_npy(source_dir / "user_map.npy")
    item_map = load_npy(source_dir / "item_map.npy")
    user_map, user_extend_stats = extend_user_map_with_eval50_users(
        user_map_100=source_user_map,
        eval50_dir=eval_dir,
    )

    split_data, split_stats = build_splits_from_100k_raw(
        raw_pairs_path=raw_pairs,
        user_map=user_map,
        item_map=item_map,
        split_time1=split_time1,
        split_time2=split_time2,
        min_train_len=args.min_train_len,
    )

    for name, obj in split_data.items():
        save_npy(out_dir / f"{name}.npy", obj)

    user_map_rev = {int(v): normalize_raw_key(k) for k, v in user_map.items()}
    save_npy(out_dir / "user_map.npy", user_map)
    save_npy(out_dir / "user_map_reverse.npy", user_map_rev)
    for name in ["item_map.npy", "item_map_reverse.npy"]:
        shutil.copy2(source_dir / name, out_dir / name)

    reset_link_or_copy(source_dir / f"{args.source_domain}.emb-t5-tdcb.npy", out_dir / f"{args.out_domain}.emb-t5-tdcb.npy", overwrite=True)
    for name in ["combine_tdcb_maps.npy", "SASRec_item_embed.pkl"]:
        src = source_dir / name
        if src.exists():
            reset_link_or_copy(src, out_dir / name, overwrite=True)

    eval50_test, eval50_warm_items, eval50_stats = remap_eval50(
        eval50_dir=eval_dir,
        user_map_100=user_map,
        item_map_100=item_map,
    )
    save_npy(out_dir / "eval50_testing_dict.npy", eval50_test)
    save_npy(out_dir / "eval50_warm_item.npy", eval50_warm_items)
    save_npy(out_dir / "eval50_user_ids.npy", sorted(eval50_test.keys()))
    save_npy(out_dir / "eval50_item_ids.npy", sorted(eval50_warm_items))

    full_test_users = {int(u) for u, items in split_data["testing_dict"].items() if items}
    eval50_missing_from_full_test = sorted(set(eval50_test.keys()) - full_test_users)
    full_warm_items = set(int(x) for x in split_data["warm_item"])
    eval50_warm_gt_users = 0
    eval50_warm_gt_items = 0
    for items in eval50_test.values():
        kept = [int(x) for x in items if int(x) in eval50_warm_items]
        if kept:
            eval50_warm_gt_users += 1
            eval50_warm_gt_items += len(kept)

    manifest = {
        "generated_utc": datetime.utcnow().isoformat() + "Z",
        "out_domain": args.out_domain,
        "source_train_domain": args.source_domain,
        "fixed_eval_domain": args.eval_domain,
        "protocol": "MicroLens-100K raw interactions split with MicroLens-50K cutoffs; eval50 files are fixed MicroLens-50K eval remapped into the 100K item/user id space.",
        "raw_pairs": str(raw_pairs),
        "split_time1_from_eval_domain": split_time1,
        "split_time2_from_eval_domain": split_time2,
        "min_train_len": args.min_train_len,
        "n_users_source_map": len(source_user_map),
        "n_users_output_map": len(user_map),
        "n_items_source_map": len(item_map),
        "user_extend_stats": user_extend_stats,
        "split_stats": split_stats,
        "eval50_stats": {
            **eval50_stats,
            "eval50_users_missing_from_full_100k_test_after_50k_cutoff": len(eval50_missing_from_full_test),
            "sample_eval50_users_missing_from_full_100k_test": eval50_missing_from_full_test[:10],
            "eval50_warm_gt_users": eval50_warm_gt_users,
            "eval50_warm_gt_items": eval50_warm_gt_items,
            "eval50_warm_items_in_full_train_warm_rate": (
                len(set(eval50_warm_items) & full_warm_items) / max(len(eval50_warm_items), 1)
            ),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
