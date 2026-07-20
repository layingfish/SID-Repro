#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np


def load_npy_obj(path: Path) -> Any:
    arr = np.load(path, allow_pickle=True)
    try:
        return arr.item()
    except Exception:
        return arr.tolist()


def load_ground_truth(path: Path, *, mode: str) -> dict[int, list[int]]:
    test_dict = load_npy_obj(path)
    out: dict[int, list[int]] = {}
    for uid, items in test_dict.items():
        if isinstance(items, np.ndarray):
            values = items.tolist()
        elif isinstance(items, (list, tuple, set)):
            values = list(items)
        elif items is None:
            values = []
        else:
            values = [items]
        values = [int(x) for x in values]
        if not values:
            continue
        out[int(uid)] = [values[0]] if mode == "loo" else values
    return out


def load_item_set(path: Path) -> set[int]:
    obj = load_npy_obj(path)
    if hasattr(obj, "keys"):
        return {int(x) for x in obj.keys()}
    return {int(x) for x in obj}


def load_predictions(path: Path, *, allowed_items: set[int] | None = None) -> dict[int, list[int]]:
    predictions: dict[int, list[int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            uid = int(obj["user_id"])
            seen: set[int] = set()
            items: list[int] = []
            for raw_item in obj["predicted_items"]:
                item = int(raw_item)
                if allowed_items is not None and item not in allowed_items:
                    continue
                if item in seen:
                    continue
                seen.add(item)
                items.append(item)
            predictions[uid] = items
    return predictions


def compute_metrics(ground_truth: dict[int, list[int]], predictions: dict[int, list[int]], top_n_list: list[int]) -> OrderedDict[str, float]:
    results: OrderedDict[str, float] = OrderedDict()
    evaluated_users = 0
    acc = {
        k: {"recall": 0.0, "ndcg": 0.0, "hr": 0.0, "precision": 0.0, "mrr": 0.0}
        for k in top_n_list
    }

    for uid, gt_items in ground_truth.items():
        if not gt_items:
            continue
        pred_items = predictions.get(uid, [])
        gt_set = set(gt_items)
        evaluated_users += 1

        for k in top_n_list:
            top_k = pred_items[:k]
            hits = 0
            dcg = 0.0
            idcg = 0.0
            idcg_count = len(gt_items)
            mrr = 0.0
            found_mrr = False

            for rank, item in enumerate(top_k):
                if item in gt_set:
                    hits += 1
                    dcg += 1.0 / math.log2(rank + 2)
                    if not found_mrr:
                        mrr = 1.0 / (rank + 1)
                        found_mrr = True
                if idcg_count > 0:
                    idcg += 1.0 / math.log2(rank + 2)
                    idcg_count -= 1

            acc[k]["recall"] += hits / len(gt_items)
            acc[k]["ndcg"] += (dcg / idcg) if idcg > 0 else 0.0
            acc[k]["hr"] += 1.0 if hits else 0.0
            acc[k]["precision"] += hits / k
            acc[k]["mrr"] += mrr

    if evaluated_users <= 0:
        raise ValueError("No users evaluated")

    for k in top_n_list:
        results[f"Recall@{k}"] = acc[k]["recall"] / evaluated_users
        results[f"NDCG@{k}"] = acc[k]["ndcg"] / evaluated_users
        results[f"HR@{k}"] = acc[k]["hr"] / evaluated_users
        results[f"Precision@{k}"] = acc[k]["precision"] / evaluated_users
        results[f"MRR@{k}"] = acc[k]["mrr"] / evaluated_users
    results["_evaluated_users"] = evaluated_users
    results["_total_gt_users"] = sum(1 for v in ground_truth.values() if v)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate pred_topk.jsonl against a fixed SETRec testing dict/candidate set.")
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--testing_dict", required=True, type=Path)
    ap.add_argument("--warm_items", required=True, type=Path)
    ap.add_argument("--mode", required=True, choices=["full", "loo"])
    ap.add_argument("--top_n", default="5,10")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--no_round", action="store_true")
    ap.add_argument("--non_strict", action="store_true")
    args = ap.parse_args()

    top_n_list = [int(x) for x in args.top_n.split(",") if x]
    max_k = max(top_n_list)
    warm_items = load_item_set(args.warm_items)
    ground_truth = load_ground_truth(args.testing_dict, mode=args.mode)
    before_users = len(ground_truth)
    before_items = sum(len(v) for v in ground_truth.values())
    ground_truth = {
        uid: [item for item in items if int(item) in warm_items]
        for uid, items in ground_truth.items()
    }
    ground_truth = {uid: items for uid, items in ground_truth.items() if items}
    predictions = load_predictions(args.pred, allowed_items=warm_items)

    missing = sorted(set(ground_truth) - set(predictions))
    short = sorted(uid for uid in ground_truth if len(predictions.get(uid, [])) < max_k)
    if (missing or short) and not args.non_strict:
        raise SystemExit(
            f"Strict validation failed: missing_users={len(missing)} sample={missing[:10]}, "
            f"short_users={len(short)} sample={short[:10]}, max_k={max_k}. "
            "Increase export --topk_items/--beam_size or use --non_strict only for debugging."
        )

    metrics = compute_metrics(ground_truth, predictions, top_n_list)
    output_metrics: dict[str, Any] = {}
    for k, v in metrics.items():
        output_metrics[k] = v if args.no_round or k.startswith("_") else round(v, 4)
    output_metrics["_meta"] = {
        "pred_file": str(args.pred.resolve()),
        "testing_dict": str(args.testing_dict.resolve()),
        "warm_items": str(args.warm_items.resolve()),
        "mode": args.mode,
        "top_n": top_n_list,
        "warm_only": True,
        "candidate_filter": "warm_items",
        "gt_users_before_warm_filter": before_users,
        "gt_items_before_warm_filter": before_items,
        "gt_users_after_warm_filter": len(ground_truth),
        "gt_items_after_warm_filter": sum(len(v) for v in ground_truth.values()),
        "prediction_users": len(predictions),
        "missing_users": len(missing),
        "short_users_after_candidate_filter": len(short),
        "strict": not args.non_strict,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_metrics, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(output_metrics, indent=2))
    print(f"[eval_fixed_setrec] saved: {args.output}")


if __name__ == "__main__":
    main()
