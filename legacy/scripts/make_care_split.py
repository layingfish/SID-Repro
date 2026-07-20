#!/usr/bin/env python3
"""
Create a CARE-paper-style split for MicroLens-50k:
  1. Load raw interactions
  2. Apply iterative k-core filtering (default 5)
  3. Per-user chronological 8:1:1 split
  4. Output SETRec-format npy files
"""
from __future__ import annotations
import argparse, csv, copy, json, math, random, sys, time
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


def load_microlens(path):
    interactions = {}  # user -> {item -> ts}
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            u, i, t = int(row["user"]), int(row["item"]), int(row["timestamp"])
            if u not in interactions:
                interactions[u] = {}
            prev = interactions[u].get(i)
            if prev is None or t > prev:
                interactions[u][i] = t
    return interactions


def apply_kcore(inter, k):
    inter = copy.deepcopy(inter)
    round_id = 0
    while True:
        round_id += 1
        item_cnt = Counter()
        for ui in inter.values():
            item_cnt.update(ui.keys())
        bad_items = {i for i, c in item_cnt.items() if c < k}
        if bad_items:
            for u in list(inter.keys()):
                for i in list(inter[u].keys()):
                    if i in bad_items:
                        del inter[u][i]
        bad_users = [u for u, ui in inter.items() if len(ui) < k]
        for u in bad_users:
            del inter[u]
        eprint(f"[kcore] round={round_id} k={k} users={len(inter)} items={len(item_cnt)-len(bad_items)} drop_items={len(bad_items)} drop_users={len(bad_users)}")
        if not bad_items and not bad_users:
            break
    return inter


def per_user_split(inter, train_ratio=0.8, val_ratio=0.1):
    """Per-user chronological split: 8:1:1"""
    train_old, val_old, test_old = {}, {}, {}
    for u, item_time in inter.items():
        pairs = sorted(item_time.items(), key=lambda x: x[1])
        n = len(pairs)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        # Ensure at least 1 test item
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        if n_train + n_val >= n:
            n_train = n - n_val - 1
        if n_train < 1:
            continue  # skip users with too few interactions

        train_old[u] = [i for i, _ in pairs[:n_train]]
        val_old[u] = [i for i, _ in pairs[n_train:n_train + n_val]]
        test_old[u] = [i for i, _ in pairs[n_train + n_val:]]
    return train_old, val_old, test_old


def make_maps(train_old, val_old, test_old, seed=2023):
    user_list = list(train_old.keys())
    item_set = set()
    for u in user_list:
        item_set.update(train_old[u])
        item_set.update(val_old[u])
        item_set.update(test_old[u])
    item_list = sorted(item_set)
    rnd = random.Random(seed)
    rnd.shuffle(item_list)
    user_map = {old: new for new, old in enumerate(user_list)}
    item_map = {old: new for new, old in enumerate(item_list)}
    return user_map, item_map


def remap_and_label(train_old, val_old, test_old, user_map, item_map):
    train, val, test = {}, {}, {}
    val_warm, val_cold, test_warm, test_cold = {}, {}, {}, {}
    warm_items, cold_items = set(), set()

    for u_old, u_new in user_map.items():
        tr = [int(item_map[i]) for i in train_old[u_old]]
        train[int(u_new)] = tr
        for i in tr:
            warm_items.add(i)

    for u_old, u_new in user_map.items():
        items = [int(item_map[i]) for i in val_old[u_old]]
        val[int(u_new)] = items
        w, c = [], []
        for i in items:
            (w if i in warm_items else c).append(i)
            if i not in warm_items:
                cold_items.add(i)
        val_warm[int(u_new)] = w
        val_cold[int(u_new)] = c

    for u_old, u_new in user_map.items():
        items = [int(item_map[i]) for i in test_old[u_old]]
        test[int(u_new)] = items
        w, c = [], []
        for i in items:
            (w if i in warm_items else c).append(i)
            if i not in warm_items:
                cold_items.add(i)
        test_warm[int(u_new)] = w
        test_cold[int(u_new)] = c

    return {
        "training_dict": train, "validation_dict": val, "testing_dict": test,
        "validation_warm_dict": val_warm, "validation_cold_dict": val_cold,
        "testing_warm_dict": test_warm, "testing_cold_dict": test_cold,
        "warm_item": warm_items, "cold_item": cold_items,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--k_core", type=int, default=5)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2023)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    eprint("[load] Loading raw data...")
    inter = load_microlens(args.input)
    eprint(f"[load] raw users={len(inter)} items={len(set(i for ui in inter.values() for i in ui))}")

    if args.k_core > 0:
        eprint(f"[kcore] Applying {args.k_core}-core...")
        inter = apply_kcore(inter, args.k_core)
    n_items_raw = len(set(i for ui in inter.values() for i in ui))
    eprint(f"[kcore] After filtering: users={len(inter)} items={n_items_raw}")

    eprint("[split] Per-user {:.0f}:{:.0f}:{:.0f} split...".format(
        args.train_ratio*10, args.val_ratio*10, (1-args.train_ratio-args.val_ratio)*10))
    train_old, val_old, test_old = per_user_split(inter, args.train_ratio, args.val_ratio)
    eprint(f"[split] users with valid split: {len(train_old)}")

    # Drop users with <2 training items (SETRec convention)
    dropped = 0
    for u in list(train_old.keys()):
        if len(train_old[u]) < 2:
            dropped += 1
            del train_old[u]; del val_old[u]; del test_old[u]
    eprint(f"[split] dropped {dropped} users with train<2, remaining={len(train_old)}")

    user_map, item_map = make_maps(train_old, val_old, test_old, args.seed)
    remapped = remap_and_label(train_old, val_old, test_old, user_map, item_map)

    # Save
    od = args.out_dir
    for name, obj in [
        ("user_map", user_map), ("item_map", item_map),
        ("user_map_reverse", {v:k for k,v in user_map.items()}),
        ("item_map_reverse", {v:k for k,v in item_map.items()}),
        ("training_dict", remapped["training_dict"]),
        ("validation_dict", remapped["validation_dict"]),
        ("testing_dict", remapped["testing_dict"]),
        ("validation_warm_dict", remapped["validation_warm_dict"]),
        ("validation_cold_dict", remapped["validation_cold_dict"]),
        ("testing_warm_dict", remapped["testing_warm_dict"]),
        ("testing_cold_dict", remapped["testing_cold_dict"]),
        ("warm_item", remapped["warm_item"]),
        ("cold_item", remapped["cold_item"]),
    ]:
        p = od / f"{name}.npy"
        if p.exists() and not args.overwrite:
            eprint(f"[SKIP] {p} exists")
            continue
        np.save(p, obj)

    # Stats
    n_users = len(user_map)
    n_items = len(item_map)
    n_warm = len(remapped["warm_item"])
    n_cold = len(remapped["cold_item"])
    train_int = sum(len(v) for v in remapped["training_dict"].values())
    val_int = sum(len(v) for v in remapped["validation_dict"].values())
    test_int = sum(len(v) for v in remapped["testing_dict"].values())

    test_cold_int = 0
    users_with_test = 0
    users_all_cold = 0
    for uid, items in remapped["testing_dict"].items():
        if not items: continue
        users_with_test += 1
        c = sum(1 for i in items if i not in remapped["warm_item"])
        test_cold_int += c
        if c == len(items):
            users_all_cold += 1

    eprint(f"\n=== CARE-style split results ===")
    eprint(f"  users={n_users}, items={n_items} (warm={n_warm}, cold={n_cold}, cold%={n_cold/n_items:.1%})")
    eprint(f"  train={train_int}, val={val_int}, test={test_int}")
    eprint(f"  ratio: {train_int/test_int:.1f}:{val_int/test_int:.1f}:1")
    eprint(f"  test cold interactions: {test_cold_int}/{test_int} ({test_cold_int/test_int:.1%})")
    eprint(f"  users all-cold test: {users_all_cold}/{users_with_test} ({users_all_cold/users_with_test:.1%})")

    # Manifest
    manifest = {
        "generated_utc": datetime.utcnow().isoformat() + "Z",
        "method": "CARE-style per-user split",
        "input": str(args.input),
        "k_core": args.k_core,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "n_users": n_users, "n_items": n_items,
        "n_warm": n_warm, "n_cold": n_cold,
        "train_int": train_int, "val_int": val_int, "test_int": test_int,
        "test_cold_ratio": round(test_cold_int/test_int, 4) if test_int else 0,
    }
    (od / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    eprint(f"\n[OK] Saved to {od}")


if __name__ == "__main__":
    main()
