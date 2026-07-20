

import argparse
import json
import math
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np


def compute_metrics(ground_truth: dict, predictions: dict, top_n_list: list):


    results = OrderedDict()
    evaluated_users = 0

    acc = {k: {"recall": 0.0, "ndcg": 0.0, "hr": 0.0, "precision": 0.0, "mrr": 0.0}
           for k in top_n_list}

    for uid, gt_items in ground_truth.items():
        if len(gt_items) == 0:
            continue

        pred_items = predictions.get(uid, [])
        evaluated_users += 1
        gt_set = set(gt_items)

        for k in top_n_list:
            top_k = pred_items[:k]
            user_hit = 0
            dcg = 0.0
            idcg = 0.0
            idcg_count = len(gt_items)
            mrr = 0.0
            mrr_found = False

            for j, item in enumerate(top_k):
                if item in gt_set:
                    dcg += 1.0 / math.log2(j + 2)
                    user_hit += 1
                    if not mrr_found:
                        mrr = 1.0 / (j + 1.0)
                        mrr_found = True

                if idcg_count > 0:
                    idcg += 1.0 / math.log2(j + 2)
                    idcg_count -= 1

            ndcg = (dcg / idcg) if idcg > 0 else 0.0

            acc[k]["recall"] += user_hit / len(gt_items)
            acc[k]["ndcg"] += ndcg
            acc[k]["hr"] += (1.0 if user_hit > 0 else 0.0)
            acc[k]["precision"] += user_hit / k
            acc[k]["mrr"] += mrr

    if evaluated_users == 0:
        raise ValueError("No users evaluated")

    for k in top_n_list:
        results[f"Recall@{k}"] = acc[k]["recall"] / evaluated_users
        results[f"NDCG@{k}"] = acc[k]["ndcg"] / evaluated_users
        results[f"HR@{k}"] = acc[k]["hr"] / evaluated_users
        results[f"Precision@{k}"] = acc[k]["precision"] / evaluated_users
        results[f"MRR@{k}"] = acc[k]["mrr"] / evaluated_users

    results["_evaluated_users"] = evaluated_users
    results["_total_gt_users"] = sum(1 for v in ground_truth.values() if len(v) > 0)

    return results


def check_pbt(metrics: dict, top_n_list: list):

    issues = []

    for prefix in ("Recall", "NDCG", "HR"):
        for k in top_n_list:
            key = f"{prefix}@{k}"
            if key not in metrics:
                issues.append(("FAIL", f"P1: Missing metric {key}"))
            elif not math.isfinite(metrics[key]):
                issues.append(("FAIL", f"P1: {key} is not finite: {metrics[key]}"))

    sorted_k = sorted(top_n_list)
    # Recall and HR are monotonic in K. NDCG is not generally monotonic because
    # its ideal-DCG denominator can grow when K grows in multi-target evaluation.
    for prefix in ("Recall", "HR"):
        for i in range(len(sorted_k) - 1):
            k1, k2 = sorted_k[i], sorted_k[i + 1]
            v1 = metrics.get(f"{prefix}@{k1}", 0)
            v2 = metrics.get(f"{prefix}@{k2}", 0)
            if v1 > v2 + 1e-9:
                issues.append(("FAIL", f"P2: {prefix}@{k1}={v1:.6f} > {prefix}@{k2}={v2:.6f}"))

    for prefix in ("Recall", "NDCG", "HR", "Precision", "MRR"):
        for k in top_n_list:
            key = f"{prefix}@{k}"
            v = metrics.get(key, 0)
            if v < -1e-9 or v > 1.0 + 1e-9:
                issues.append(("FAIL", f"P4: {key}={v:.6f} out of [0,1]"))

    evaluated = metrics.get("_evaluated_users", 0)
    total = metrics.get("_total_gt_users", 0)
    if evaluated != total:
        issues.append(("WARN", f"P5: Evaluated {evaluated}/{total} users with ground truth"))

    return issues


def load_ground_truth_seq_last(llm_data_dir: str, dataset: str):


    seq_path = os.path.join(llm_data_dir, f"setrec_{dataset}", "sequential_data.txt")
    ground_truth = {}
    with open(seq_path) as f:
        for line_no, line in enumerate(f, 1):
            parts = [x for x in line.strip().split() if x]
            if len(parts) < 2:
                continue
            uid_1based = int(parts[0])
            last_item = parts[-1]
            item_1based = int(last_item[1:]) if last_item.startswith("I") else int(last_item)
            ground_truth[uid_1based - 1] = [item_1based - 1]
    return ground_truth


def load_ground_truth(data_dir: str, dataset: str, mode: str):

    test_path = os.path.join(data_dir, dataset, "testing_dict.npy")
    test_dict = np.load(test_path, allow_pickle=True).item()

    ground_truth = {}
    for uid, items in test_dict.items():
        if isinstance(items, np.ndarray):
            items = items.tolist()
        if not isinstance(items, list):
            items = [items] if items is not None else []

        if len(items) == 0:
            continue

        if mode == "loo":
            ground_truth[uid] = [items[0]]
        else:
            ground_truth[uid] = items

    return ground_truth


def load_predictions(pred_path: str):

    predictions = {}
    with open(pred_path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line_no}: invalid JSON: {e}")

            uid = obj["user_id"]
            items = obj["predicted_items"]
            if not isinstance(items, list):
                raise ValueError(f"Line {line_no}: predicted_items must be a list")

            seen = set()
            unique_items = []
            for item in items:
                if item not in seen:
                    seen.add(item)
                    unique_items.append(item)
            predictions[uid] = unique_items

    return predictions


def infer_item_id_max(data_dir: str, dataset: str):


    ddir = Path(data_dir) / dataset
    warm_path = ddir / "warm_item.npy"
    cold_path = ddir / "cold_item.npy"
    item_max = None

    if warm_path.exists() and cold_path.exists():
        warm = np.load(warm_path, allow_pickle=True)
        cold = np.load(cold_path, allow_pickle=True)
        warm = warm.tolist() if isinstance(warm, np.ndarray) else list(warm)
        cold = cold.tolist() if isinstance(cold, np.ndarray) else list(cold)
        all_items = [int(x) for x in warm] + [int(x) for x in cold]
        if all_items:
            item_max = max(all_items)

    return item_max


def validate_predictions_strict(ground_truth: dict, predictions: dict, top_n_list: list, item_id_max: int | None):


    issues = []

    if not top_n_list:
        issues.append(("FAIL", "V0: top_n_list is empty"))
        return issues

    if any(k <= 0 for k in top_n_list):
        issues.append(("FAIL", f"V0: invalid K in top_n_list={top_n_list}"))
        return issues

    required_uids = set(ground_truth.keys())
    pred_uids = set(predictions.keys())

    missing = sorted(required_uids - pred_uids)
    extra = sorted(pred_uids - required_uids)

    if missing:
        sample = missing[:10]
        issues.append(("FAIL", f"V1: Missing predictions for {len(missing)} ground-truth users; sample={sample}"))

    if extra:
        sample = extra[:10]
        issues.append(("WARN", f"V1: Predictions contain {len(extra)} extra users not in ground truth; sample={sample}"))

    max_k = max(top_n_list)
    short_users = []
    bad_type = 0
    bad_range = 0


    for uid in required_uids:
        items = predictions.get(uid, [])
        if not isinstance(items, list):
            bad_type += 1
            continue

        if len(items) < max_k:
            short_users.append(uid)

        for it in items:
            if not isinstance(it, (int, np.integer)):
                bad_type += 1
                continue
            it = int(it)
            if it < 0:
                bad_range += 1
                continue
            if item_id_max is not None and it > item_id_max:
                bad_range += 1

    if short_users:
        sample = short_users[:10]
        issues.append(("FAIL", f"V2: {len(short_users)} users have <{max_k} predictions; sample={sample}"))

    if bad_type:
        issues.append(("FAIL", f"V3: Found {bad_type} non-int or malformed predicted items"))

    if bad_range:
        if item_id_max is None:
            issues.append(("FAIL", f"V4: Found {bad_range} negative/out-of-range predicted items (max unknown)"))
        else:
            issues.append(("FAIL", f"V4: Found {bad_range} out-of-range predicted items (expected 0..{item_id_max})"))

    return issues


def _load_setrec_computeTopNAccuracy(setrec_eval_utils_path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("setrec_eval_utils", str(setrec_eval_utils_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {setrec_eval_utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.computeTopNAccuracy


def run_self_tests(setrec_eval_utils_path: str | None = None):

    top_n_list = [5, 10]


    ground_truth = {
        0: [1],
        1: [2, 3],
    }
    predictions = {
        0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        1: [3, 2, 1, 4, 5, 6, 7, 8, 9, 10],
    }

    m = compute_metrics(ground_truth, predictions, top_n_list)


    for k in top_n_list:
        for key in (f"Recall@{k}", f"NDCG@{k}", f"HR@{k}", f"Precision@{k}", f"MRR@{k}"):
            v = m.get(key)
            if v is None or not math.isfinite(v):
                raise AssertionError(f"self_test: {key} invalid: {v}")
            if v < -1e-12 or v > 1.0 + 1e-12:
                raise AssertionError(f"self_test: {key} out of [0,1]: {v}")


    if abs(m["Recall@5"] - 1.0) > 1e-12:
        got_recall5 = m["Recall@5"]
        raise AssertionError(f"self_test: Recall@5 expected 1.0, got {got_recall5}")


    if setrec_eval_utils_path:
        p = Path(setrec_eval_utils_path)
        if not p.exists():
            raise FileNotFoundError(f"--setrec_eval_utils not found: {p}")
        computeTopNAccuracy = _load_setrec_computeTopNAccuracy(p)


        gt_list = [ground_truth[i] for i in sorted(ground_truth.keys())]
        pred_list = [predictions[i] for i in sorted(predictions.keys())]

        precision, recall, ndcg, mrr = computeTopNAccuracy(gt_list, pred_list, top_n_list)

        ours_recall = [round(m[f"Recall@{k}"], 4) for k in top_n_list]
        ours_ndcg = [round(m[f"NDCG@{k}"], 4) for k in top_n_list]
        ours_precision = [round(m[f"Precision@{k}"], 4) for k in top_n_list]
        ours_mrr = [round(m[f"MRR@{k}"], 4) for k in top_n_list]

        if ours_recall != recall:
            raise AssertionError(f"self_test: recall mismatch: ours={ours_recall} vs official={recall}")
        if ours_ndcg != ndcg:
            raise AssertionError(f"self_test: ndcg mismatch: ours={ours_ndcg} vs official={ndcg}")
        if ours_precision != precision:
            raise AssertionError(f"self_test: precision mismatch: ours={ours_precision} vs official={precision}")
        if ours_mrr != mrr:
            raise AssertionError(f"self_test: mrr mismatch: ours={ours_mrr} vs official={mrr}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation for RecSys26")
    default_root = Path(os.environ.get("RECSYS26_ROOT", Path.cwd())).resolve()
    parser.add_argument("--self_test", action="store_true", help="Run deterministic self-tests and exit")
    parser.add_argument(
        "--setrec_eval_utils",
        default=str(default_root / "baselines" / "reference" / "code" / "utils" / "eval_utils.py"),
        help="Path to official SETRec eval_utils.py for self-test cross-check",
    )
    parser.add_argument("--pred", default=None, help="Path to pred_topk.jsonl")
    parser.add_argument("--dataset", default=None, choices=["beauty", "toys", "sports", "steam", "amazon23_vg", "microlens_50k", "microlens_100k", "microlens_1m", "yelp", "microlens_50k_care"])
    parser.add_argument(
        "--mode",
        default=None,
        choices=["full", "loo", "seq_last"],
        help="full=all test items; loo=first test item; seq_last=last item from adapted sequential_data.txt",
    )
    parser.add_argument(
        "--data_dir",
        default=str(default_root / "data"),
        help="Path to SETRec data root",
    )
    parser.add_argument(
        "--llm_data_dir",
        default=str(default_root / "data" / "sequential"),
        help="Path to LLM_RecSys_ID adapted data root (for mode=seq_last)",
    )
    parser.add_argument("--top_n", default="5,10", help="Comma-separated K values")
    parser.add_argument("--output", default=None, help="Output metrics.json path")
    parser.add_argument("--no_round", action="store_true", help="Disable rounding (for calibration)")
    parser.add_argument("--non_strict", action="store_true", help="Disable strict validation (debug only)")
    parser.add_argument(
        "--warm_only",
        action="store_true",
        help=(
            "Evaluate on warm items only: filter cold items from GT before computing metrics. "
            "Users whose entire GT is cold are excluded. "
            "Applied uniformly across yelp/amazon23_vg/microlens_50k/microlens_100k/microlens_1m as the primary evaluation protocol."
        ),
    )
    args = parser.parse_args()

    if args.self_test:
        run_self_tests(setrec_eval_utils_path=args.setrec_eval_utils)
        print("[unified_eval] self_test: PASS")
        return

    if not args.pred or not args.dataset or not args.mode:
        parser.error("--pred/--dataset/--mode are required unless --self_test is set")

    top_n_list = [int(k) for k in args.top_n.split(",")]

    print(f"[unified_eval] dataset={args.dataset} mode={args.mode} warm_only={args.warm_only} top_n={top_n_list}")

    if args.mode == "seq_last":
        ground_truth = load_ground_truth_seq_last(args.llm_data_dir, args.dataset)
    else:
        ground_truth = load_ground_truth(args.data_dir, args.dataset, args.mode)

    if args.warm_only:
        warm_path = os.path.join(args.data_dir, args.dataset, "warm_item.npy")
        warm_raw = np.load(warm_path, allow_pickle=True)
        w = warm_raw.item()
        warm_ids = set(w.keys()) if hasattr(w, "keys") else set(int(x) for x in w)
        before_users = len(ground_truth)
        before_items = sum(len(v) for v in ground_truth.values())
        ground_truth = {
            uid: [i for i in items if i in warm_ids]
            for uid, items in ground_truth.items()
        }
        ground_truth = {uid: items for uid, items in ground_truth.items() if len(items) > 0}
        after_users = len(ground_truth)
        after_items = sum(len(v) for v in ground_truth.values())
        print(
            f"[unified_eval] warm_only: users {before_users} -> {after_users} "
            f"({before_users - after_users} all-cold dropped, {after_users / before_users * 100:.1f}% retained)"
        )
        print(
            f"[unified_eval] warm_only: GT items {before_items} -> {after_items} "
            f"({before_items - after_items} cold GT items removed)"
        )

    predictions = load_predictions(args.pred)

    print(f"[unified_eval] ground_truth: {len(ground_truth)} users with test items")
    print(f"[unified_eval] predictions:  {len(predictions)} users with predictions")

    strict = not args.non_strict
    item_id_max = infer_item_id_max(args.data_dir, args.dataset)
    if item_id_max is not None:
        print(f"[unified_eval] inferred item_id_max={item_id_max} (expected range: 0..{item_id_max})")
    else:
        print("[unified_eval] inferred item_id_max=None (range checks may be weaker)")

    val_issues = validate_predictions_strict(ground_truth, predictions, top_n_list, item_id_max)
    if val_issues:
        print("\n" + "=" * 60)
        print("Prediction Validation")
        print("=" * 60)
        for severity, msg in val_issues:
            print(f"  [{severity}] {msg}")

        if strict and any(s == "FAIL" for s, _ in val_issues):
            sys.exit(1)

    metrics = compute_metrics(ground_truth, predictions, top_n_list)

    issues = check_pbt(metrics, top_n_list)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Results ({args.dataset}, mode={args.mode})")
    print(sep)
    evaluated_users = metrics["_evaluated_users"]
    total_gt_users = metrics["_total_gt_users"]
    print(f"Evaluated users: {evaluated_users}/{total_gt_users}")
    print()

    for prefix in ("Recall", "NDCG", "HR", "Precision", "MRR"):
        vals = []
        for k in top_n_list:
            key = f"{prefix}@{k}"
            v = metrics.get(key, 0)
            if not args.no_round:
                v = round(v, 4)
                metrics[key] = v
            vals.append(f"{key}={v:.4f}")
        joined = ", ".join(vals)
        print(f"  {joined}")

    if issues:
        print(f"\n{sep}")
        print("PBT Property Checks")
        print(sep)
        for severity, msg in issues:
            print(f"  [{severity}] {msg}")
    else:
        print("\n  PBT checks: ALL PASS")

    output_path = args.output or args.pred.replace(".jsonl", f"_metrics_{args.mode}.json")
    output_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
    output_metrics["_meta"] = {
        "dataset": args.dataset,
        "mode": args.mode,
        "warm_only": args.warm_only,
        "top_n": top_n_list,
        "evaluated_users": metrics["_evaluated_users"],
        "total_gt_users": metrics["_total_gt_users"],
        "strict": strict,
        "item_id_max": item_id_max,
        "pred_file": os.path.abspath(args.pred),
    }
    with open(output_path, "w") as f:
        json.dump(output_metrics, f, indent=2)
    print(f"\n  Saved: {output_path}")

    if any(s == "FAIL" for s, _ in issues):
        sys.exit(1)


if __name__ == "__main__":
    main()
