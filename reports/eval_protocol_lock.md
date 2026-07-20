# Eval Protocol Lock (RecSys26 reproduction)

Locked at: 2026-02-28 19:05 (UTC)

## 1) Dataset (base)
- Base dataset: SETRec processed datasets (beauty/toys/sports/steam)
- Canonical data root:
  - /data/xqp_data/RecSys26/third_party/SETRec/data/{beauty,toys,sports,steam}
- Audit (hash/stat/id checks):
  - /data/xqp_data/RecSys26/reports/dataset_audit_20260228_170226.md
  - /data/xqp_data/RecSys26/reports/dataset_audit_20260228_170226.json

## 2) Ground truth & user set
- Ground truth file (canonical): testing_dict.npy
- Evaluated users: ALL users with non-empty test items in testing_dict.npy
- Two evaluation modes (for different purposes):
  - full (FINAL): per-user ground truth = all test items
  - loo  (DEBUG ONLY): per-user ground truth = first test item only

## 3) Prediction export format (single source of truth)
- JSONL, one line per user:
  - { "user_id": <int>, "predicted_items": [<int>, ...] }
- All methods MUST export canonical 0-based item ids.
- Strict requirements (fail-fast):
  - must cover ALL ground-truth users
  - each user must have at least top-10 unique predictions
  - predicted item ids must be within [0, item_id_max] inferred from warm/cold item lists
  - duplicates are de-duplicated (stable) before evaluation

## 4) Metrics & exact definitions
- Main metrics (reported in final tables):
  - Recall@{5,10}
  - NDCG@{5,10}
- Definitions (binary relevance):
  - Recall@K = avg_u ( |GT_u ∩ TopK_u| / |GT_u| )
  - NDCG@K:
    - DCG = sum_{j=0..K-1} 1[TopK_u[j] in GT_u] / log2(j+2)
    - IDCG = sum_{j=0..min(|GT_u|,K)-1} 1 / log2(j+2)
    - NDCG = DCG / IDCG

## 5) Implementation (locked scripts)
- Unified evaluator:
  - /data/xqp_data/RecSys26/scripts/unified_eval.py
  - sha256: 3700361f9d811b9a4dae91137d0edbd7bbf71059ca5f2bd68d429d8e61c272f6
- Reference official implementation for alignment:
  - SETRec repo (clean): /data/xqp_data/RecSys26/third_party_clean/SETRec
  - commit: 2ed9a75ad1ad3784c61bba3c68cbedbe3cfce2d7
  - function: computeTopNAccuracy in code/utils/eval_utils.py

## 6) Alignment proof
- Self-test command (uses conda env python with numpy):
  - /data/xqp_data/RecSys26/envs/miniforge3/envs/recsys26_seater/bin/python /data/xqp_data/RecSys26/scripts/unified_eval.py --self_test --setrec_eval_utils /data/xqp_data/RecSys26/third_party_clean/SETRec/code/utils/eval_utils.py
- Expected output tail:
  - [unified_eval] self_test: PASS
