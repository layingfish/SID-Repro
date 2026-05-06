"""Export DiffGRM top-k predictions to JSONL for unified_eval.py.

Output JSONL schema:
  {"user_id": <int>, "predicted_items": [<int>, ...]}

Notes:
- DiffGRM internal item ids are 1-based; exporter converts back to canonical 0-based.
- Export filters out items seen in SETRec train/val history.
- Coverage is checked against canonical SETRec testing_dict.npy users (non-empty test items).

Decoding:
- --decode beam: generate SID tuples (may yield invalid / duplicate codes early in training).
- --decode score_all: rank all items by per-digit logp (always yields a full top-k).
- For robustness, beam decoding defaults to fallback to score_all when it cannot produce k
  valid unique items for a user (configurable via --beam_fallback).
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {v!r}")


def _to_list(x):
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, list):
        return x
    return [x]


def load_setrec_dicts(setrec_root: Path, domain: str):
    d = setrec_root / domain
    train = np.load(str(d / "training_dict.npy"), allow_pickle=True).item()
    val = np.load(str(d / "validation_dict.npy"), allow_pickle=True).item()
    test = np.load(str(d / "testing_dict.npy"), allow_pickle=True).item()
    return train, val, test


def infer_item_id_max(setrec_root: Path, domain: str) -> int:
    d = setrec_root / domain
    warm = np.load(str(d / "warm_item.npy"), allow_pickle=True)
    cold = np.load(str(d / "cold_item.npy"), allow_pickle=True)
    warm = warm.tolist() if isinstance(warm, np.ndarray) else list(warm)
    cold = cold.tolist() if isinstance(cold, np.ndarray) else list(cold)
    all_items = [int(x) for x in warm] + [int(x) for x in cold]
    if not all_items:
        raise RuntimeError("warm/cold item lists are empty")
    return int(max(all_items))


def compute_score_all_scores(pipeline, batch: dict, item_codes: torch.Tensor) -> torch.Tensor:
    """Compute per-item scores for the whole batch (CPU tensor of shape [B, N])."""
    enc_out = pipeline.model.forward(batch, return_loss=False)
    encoder_hidden = enc_out.hidden_states
    B = int(encoder_hidden.shape[0])
    n_digit = int(item_codes.shape[1])

    dec_batch = {
        "decoder_input_ids": torch.zeros(B, n_digit, device=pipeline.accelerator.device, dtype=torch.long),
        "encoder_hidden": encoder_hidden,
        "mask_positions": torch.ones(B, n_digit, device=pipeline.accelerator.device),
    }
    out = pipeline.model.forward_decoder_only(dec_batch, return_loss=False, digit=None, use_cache=False)
    logp = torch.log_softmax(out.logits, dim=-1)

    scores = None
    for d in range(n_digit):
        part = logp[:, d, :][:, item_codes[:, d]]
        scores = part if scores is None else (scores + part)

    return scores.detach().cpu()


def score_all_topk(row: torch.Tensor, hist: set[int], k: int, item_id_max: int) -> list[int]:
    if hist:
        for it in hist:
            it = int(it)
            if 0 <= it <= int(item_id_max):
                row[it] = -1e9
    top_idx = torch.topk(row, k=int(k)).indices.tolist()
    return [int(x) for x in top_idx]


def main() -> None:
    ap = argparse.ArgumentParser(description="Export DiffGRM predictions to JSONL")
    ap.add_argument(
        "--diffgrm_code_dir",
        default="baselines/ref06",
        help="Path to DiffGRM repo root (contains main.py)",
    )
    ap.add_argument("--checkpoint", required=True, help="Path to best checkpoint (best.bin)")
    ap.add_argument("--domain", required=True, choices=["beauty", "toys", "sports", "steam", "amazon23_vg", "microlens_50k", "yelp"])
    ap.add_argument("--cache_dir", required=True, help="DiffGRM cache_dir (root)")
    ap.add_argument(
        "--setrec_data_root",
        default="data/setrec_data",
        help="SETRec canonical data root (contains domain dirs)",
    )
    ap.add_argument("--eval_batch_size", type=int, default=32)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--n_return_sequences", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pred_out", required=True)
    ap.add_argument("--use_cpu", action="store_true")
    ap.add_argument("--beam_mode", default="", help="Override beam mode (default uses config primary)")
    ap.add_argument(
        "--decode",
        default="beam",
        choices=["beam", "score_all"],
        help="Decoding strategy: beam (generate SID tuples) or score_all (rank all items by per-digit logp)",
    )
    ap.add_argument(
        "--beam_fallback",
        default="score_all",
        choices=["score_all", "error"],
        help="When --decode beam yields <k valid unique items for a user: score_all (fallback) or error (raise)",
    )
    ap.add_argument("--log_dir", default="", help="Optional log_dir override (for Pipeline init_logger)")
    ap.add_argument("--tensorboard_log_dir", default="", help="Optional tensorboard_log_dir override")

    # Tokenization / SID mapping params (must match training config)
    ap.add_argument("--sent_emb_model", default="setrec_t5_tdcb")
    ap.add_argument("--sent_emb_dim", type=int, default=768)
    ap.add_argument("--sent_emb_pca", type=int, default=0)
    ap.add_argument("--normalize_after_pca", type=str2bool, default=True)
    ap.add_argument("--force_regenerate_opq", type=str2bool, default=False)

    # Model / diffusion params (optional; use to match training checkpoint)
    ap.add_argument("--encoder_n_layer", type=int, default=None)
    ap.add_argument("--decoder_n_layer", type=int, default=None)
    ap.add_argument("--n_head", type=int, default=None)
    ap.add_argument("--n_embd", type=int, default=None)
    ap.add_argument("--n_inner", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--masking_strategy", default="", help="Override masking_strategy (random|sequential|guided)")
    ap.add_argument("--guided_select", default="", help="Override guided_select (most|least)")
    ap.add_argument("--guided_conf_metric", default="", help="Override guided_conf_metric (msp|entropy)")
    ap.add_argument("--guided_refresh_each_step", type=str2bool, default=None)
    ap.add_argument("--train_sliding", type=str2bool, default=None)
    ap.add_argument("--min_hist_len", type=int, default=None)

    args = ap.parse_args()

    set_seed(int(args.seed))

    diffgrm_code_dir = Path(args.diffgrm_code_dir).resolve()
    if not diffgrm_code_dir.exists():
        raise FileNotFoundError(f"diffgrm_code_dir not found: {diffgrm_code_dir}")

    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    setrec_root = Path(args.setrec_data_root).resolve()
    if not setrec_root.exists():
        raise FileNotFoundError(f"setrec_data_root not found: {setrec_root}")

    train_dict, val_dict, test_dict = load_setrec_dicts(setrec_root, args.domain)
    gt_users = sorted(int(u) for u, items in test_dict.items() if len(_to_list(items)) > 0)
    if not gt_users:
        raise RuntimeError("no ground-truth users found in testing_dict.npy")

    history: dict[int, set[int]] = {}
    for u in gt_users:
        tr = _to_list(train_dict.get(u, []))
        va = _to_list(val_dict.get(u, []))
        history[u] = set(int(x) for x in tr) | set(int(x) for x in va)

    item_id_max = infer_item_id_max(setrec_root, args.domain)

    sys.path.insert(0, str(diffgrm_code_dir))
    os.chdir(str(diffgrm_code_dir))

    from genrec.pipeline import Pipeline  # noqa: E402

    config_dict = {
        "category": args.domain,
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "eval_batch_size": int(args.eval_batch_size),
        "topk": [5, 10],
        "metrics": ["ndcg", "recall"],
        "sent_emb_model": str(args.sent_emb_model),
        "sent_emb_dim": int(args.sent_emb_dim),
        "sent_emb_pca": int(args.sent_emb_pca),
        "normalize_after_pca": bool(args.normalize_after_pca),
        "force_regenerate_opq": bool(args.force_regenerate_opq),
    }

    # Optional config overrides to ensure the exporter builds the same model as training.
    # (When not provided, Pipeline falls back to genrec/models/DIFF_GRM/config.yaml defaults.)
    if args.encoder_n_layer is not None:
        config_dict["encoder_n_layer"] = int(args.encoder_n_layer)
    if args.decoder_n_layer is not None:
        config_dict["decoder_n_layer"] = int(args.decoder_n_layer)
    if args.n_head is not None:
        config_dict["n_head"] = int(args.n_head)
    if args.n_embd is not None:
        config_dict["n_embd"] = int(args.n_embd)
    if args.n_inner is not None:
        config_dict["n_inner"] = int(args.n_inner)
    if args.dropout is not None:
        config_dict["dropout"] = float(args.dropout)

    if str(args.masking_strategy).strip():
        config_dict["masking_strategy"] = str(args.masking_strategy).strip()
    if str(args.guided_select).strip():
        config_dict["guided_select"] = str(args.guided_select).strip()
    if str(args.guided_conf_metric).strip():
        config_dict["guided_conf_metric"] = str(args.guided_conf_metric).strip()
    if args.guided_refresh_each_step is not None:
        config_dict["guided_refresh_each_step"] = bool(args.guided_refresh_each_step)

    if args.train_sliding is not None:
        config_dict["train_sliding"] = bool(args.train_sliding)
    if args.min_hist_len is not None:
        config_dict["min_hist_len"] = int(args.min_hist_len)


    if args.use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if args.log_dir:
        config_dict["log_dir"] = args.log_dir
    if args.tensorboard_log_dir:
        config_dict["tensorboard_log_dir"] = args.tensorboard_log_dir

    pipeline = Pipeline(
        model_name="DIFF_GRM",
        dataset_name="SETRec",
        checkpoint_path=str(ckpt_path),
        config_dict=config_dict,
    )

    test_hf = pipeline.raw_dataset.split_data["test"]
    user_names = test_hf["user"]
    setrec_user_ids = [int(str(u)[1:]) for u in user_names]

    if sorted(setrec_user_ids) != gt_users:
        raise RuntimeError(
            "user coverage mismatch between DiffGRM dataset split and canonical testing_dict.npy: "
            f"split_users={len(set(setrec_user_ids))} gt_users={len(gt_users)}"
        )

    test_dataloader = DataLoader(
        pipeline.tokenized_datasets["test"],
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        collate_fn=pipeline.tokenizer.collate_fn["test"],
    )

    pipeline.model, test_dataloader = pipeline.accelerator.prepare(pipeline.model, test_dataloader)

    pipeline.model.eval()
    tok = pipeline.tokenizer

    # Ensure export uses test-time beam config (larger beam_act) and enough candidates.
    try:
        if hasattr(pipeline.model, "config") and isinstance(pipeline.model.config, dict):
            pipeline.model.config["current_split"] = "test"
            vbs = pipeline.model.config.get("vectorized_beam_search")
            if isinstance(vbs, dict):
                vbs["top_k_final"] = int(args.n_return_sequences)
    except Exception as e:
        print(f"[export_diffgrm] WARN: failed to override beam config: {e}")

    decode = str(args.decode).strip().lower()
    beam_fallback = str(args.beam_fallback).strip().lower()

    need_item_codes = (decode == "score_all") or (decode == "beam" and beam_fallback == "score_all")

    item_codes = None
    if need_item_codes:
        n_digit = int(getattr(pipeline.model, "n_digit", tok.n_digit))
        N = int(item_id_max) + 1
        codes = torch.empty((N, n_digit), dtype=torch.long)
        for item_0 in range(N):
            it = f"I{item_0}"
            toks = tok.item2tokens.get(it)
            if toks is None:
                raise RuntimeError(f"missing item2tokens for {it}")
            cb = []
            for d in range(n_digit):
                cb_id = int(toks[d]) - (tok.sid_offset + d * tok.codebook_size)
                if cb_id < 0 or cb_id >= int(tok.codebook_size):
                    raise RuntimeError(f"invalid cb_id={cb_id} for {it} digit={d}")
                cb.append(cb_id)
            codes[item_0] = torch.tensor(cb, dtype=torch.long)
        item_codes = codes.to(pipeline.accelerator.device)

    modes = pipeline.config.get("beam_search_modes", ["confidence"])
    mode = str(args.beam_mode).strip() or (modes[0] if modes else "confidence")

    pred_out = Path(args.pred_out).resolve()
    pred_out.parent.mkdir(parents=True, exist_ok=True)

    expected = len(setrec_user_ids)
    seen_users: set[int] = set()

    print(f"[export_diffgrm] decode={decode} beam_fallback={beam_fallback} users={expected} k={int(args.k)}")

    fallback_users = 0

    idx = 0
    with pred_out.open("w", encoding="utf-8") as f:
        for batch in tqdm(test_dataloader, desc=f"Export preds ({decode})"):
            with torch.no_grad():
                if isinstance(batch, dict):
                    batch = {k: v.to(pipeline.accelerator.device) for k, v in batch.items()}

                preds_cpu = None
                scores_cpu = None

                if decode == "beam":
                    preds = pipeline.model.generate(batch, n_return_sequences=int(args.n_return_sequences), mode=mode)
                    if isinstance(preds, tuple):
                        preds = preds[0]
                    preds_cpu = preds.detach().cpu()
                elif decode == "score_all":
                    if item_codes is None:
                        raise RuntimeError("item_codes not initialized for score_all")
                    scores_cpu = compute_score_all_scores(pipeline, batch, item_codes)
                else:
                    raise RuntimeError(f"unknown decode strategy: {decode}")

            B = int(preds_cpu.shape[0]) if preds_cpu is not None else int(scores_cpu.shape[0])

            batch_uids: list[int] = []
            batch_hists: list[set[int]] = []
            batch_items: list[list[int] | None] = [None for _ in range(B)]
            fallback_idx: list[int] = []

            for i in range(B):
                if idx >= expected:
                    raise RuntimeError("prediction overflow: more batches than expected users")

                uid = int(setrec_user_ids[idx])
                idx += 1

                if uid in seen_users:
                    raise RuntimeError(f"duplicate user_id in test split order: {uid}")
                seen_users.add(uid)

                hist = history.get(uid, set())
                batch_uids.append(uid)
                batch_hists.append(hist)

                if decode == "beam":
                    items: list[int] = []
                    uniq: set[int] = set()
                    for j in range(int(preds_cpu.shape[1])):
                        cb_ids = preds_cpu[i, j].tolist()
                        item_id = tok.codebooks_to_item_id(cb_ids)
                        if item_id is None:
                            continue
                        item_0 = int(item_id) - 1
                        if item_0 < 0 or item_0 > int(item_id_max):
                            continue
                        if item_0 in hist:
                            continue
                        if item_0 in uniq:
                            continue
                        uniq.add(item_0)
                        items.append(item_0)
                        if len(items) >= int(args.k):
                            break

                    if len(items) < int(args.k):
                        if beam_fallback == "score_all":
                            fallback_idx.append(i)
                            batch_items[i] = None
                        else:
                            if len(items) < 10:
                                raise RuntimeError(f"predicted_items too short for user_id={uid}: {len(items)}")
                            raise RuntimeError(
                                f"predicted_items <k for user_id={uid}: {len(items)} < {int(args.k)}; "
                                "try increasing --n_return_sequences or use --decode score_all"
                            )
                    else:
                        batch_items[i] = items[: int(args.k)]
                else:
                    row = scores_cpu[i]
                    items = score_all_topk(row, hist, int(args.k), int(item_id_max))
                    if len(items) < 10:
                        raise RuntimeError(f"predicted_items too short for user_id={uid}: {len(items)}")
                    if len(items) < int(args.k):
                        raise RuntimeError(f"predicted_items <k for user_id={uid}: {len(items)} < {int(args.k)}")
                    batch_items[i] = items[: int(args.k)]

            if fallback_idx:
                if item_codes is None:
                    raise RuntimeError("item_codes not initialized for beam fallback")
                with torch.no_grad():
                    scores_cpu_fb = compute_score_all_scores(pipeline, batch, item_codes)
                for i in fallback_idx:
                    uid = batch_uids[i]
                    hist = batch_hists[i]
                    row = scores_cpu_fb[i]
                    items = score_all_topk(row, hist, int(args.k), int(item_id_max))
                    if len(items) < 10 or len(items) < int(args.k):
                        raise RuntimeError(
                            f"beam_fallback failed for user_id={uid}: len={len(items)} k={int(args.k)}"
                        )
                    batch_items[i] = items[: int(args.k)]
                    fallback_users += 1

            for i in range(B):
                uid = batch_uids[i]
                items = batch_items[i]
                if items is None:
                    raise RuntimeError(f"internal error: missing items for user_id={uid}")
                obj = {"user_id": int(uid), "predicted_items": [int(x) for x in items]}
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    if idx != expected:
        raise RuntimeError(f"coverage mismatch: wrote={idx} expected={expected}")

    if decode == "beam" and beam_fallback == "score_all":
        print(f"[export_diffgrm] beam_fallback score_all used for {fallback_users}/{expected} users")

    print(f"[export_diffgrm] wrote users={idx} pred_out={pred_out}")


if __name__ == "__main__":
    main()
