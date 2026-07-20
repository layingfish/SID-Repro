#!/usr/bin/env python3
"""Prefix-constrained generation for SID length diagnostics.

For a fixed trained decoder with semantic length L, this script generates only
the first k semantic tokens for k=1..L, maps each generated prefix to all items
sharing that prefix, and evaluates the resulting ranked list.

This is intentionally different from post-hoc projection over a full decoded
ranking: the decoder is constrained to emit only k SID tokens during generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def _maybe_strip_module_prefix(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict
    if all(isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    if all(isinstance(k, str) and k.startswith("_orig_mod.") for k in state_dict.keys()):
        return {k[len("_orig_mod."):]: v for k, v in state_dict.items()}
    return state_dict


def _load_npy_obj(path: Path):
    arr = np.load(path, allow_pickle=True)
    try:
        return arr.item()
    except Exception:
        return arr.tolist()


def _load_warm_full_gt(data_dir: Path, dataset: str) -> dict[int, list[int]]:
    test = _load_npy_obj(data_dir / dataset / "testing_dict.npy")
    warm_raw = _load_npy_obj(data_dir / dataset / "warm_item.npy")
    warm_items = set(int(x) for x in (warm_raw.keys() if hasattr(warm_raw, "keys") else warm_raw))
    gt: dict[int, list[int]] = {}
    for uid, items in test.items():
        if isinstance(items, np.ndarray):
            values = items.tolist()
        elif isinstance(items, (list, tuple, set)):
            values = list(items)
        elif items is None:
            values = []
        else:
            values = [items]
        kept = [int(x) for x in values if int(x) in warm_items]
        if kept:
            gt[int(uid)] = kept
    return gt


def _load_seen_items(data_dir: Path, dataset: str) -> dict[int, set[int]]:
    path = data_dir / dataset / "training_dict.npy"
    if not path.exists():
        return {}
    raw = _load_npy_obj(path)
    out: dict[int, set[int]] = {}
    for uid, items in raw.items():
        if isinstance(items, np.ndarray):
            values = items.tolist()
        elif isinstance(items, (list, tuple, set)):
            values = list(items)
        elif items is None:
            values = []
        else:
            values = [items]
        out[int(uid)] = {int(x) for x in values}
    return out


def _item_popularity(data_dir: Path, dataset: str, n_items: int) -> np.ndarray:
    pop = np.zeros(n_items, dtype=np.int64)
    path = data_dir / dataset / "training_dict.npy"
    if not path.exists():
        return pop
    raw = _load_npy_obj(path)
    for items in raw.values():
        if isinstance(items, np.ndarray):
            values = items.tolist()
        elif isinstance(items, (list, tuple, set)):
            values = list(items)
        elif items is None:
            values = []
        else:
            values = [items]
        for item in values:
            item = int(item)
            if 0 <= item < n_items:
                pop[item] += 1
    return pop


def _sort_items(items: Iterable[int], popularity: np.ndarray) -> list[int]:
    return sorted((int(x) for x in items), key=lambda x: (-int(popularity[x]), x))


def _build_prefix_groups(token_seqs: torch.Tensor, popularity: np.ndarray, max_k: int) -> dict[int, dict[tuple[int, ...], list[int]]]:
    groups_by_k: dict[int, dict[tuple[int, ...], list[int]]] = {}
    for k in range(1, max_k + 1):
        groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for item_id in range(token_seqs.shape[0]):
            key = tuple(int(x) for x in token_seqs[item_id, :k].tolist())
            groups[key].append(int(item_id))
        groups_by_k[k] = {key: _sort_items(items, popularity) for key, items in groups.items()}
    return groups_by_k


def _load_predictions_jsonl(path: Path) -> dict[int, list[int]]:
    predictions: dict[int, list[int]] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            seen = set()
            items = []
            for raw in obj["predicted_items"]:
                item = int(raw)
                if item not in seen:
                    seen.add(item)
                    items.append(item)
            predictions[int(obj["user_id"])] = items
    return predictions


def _fill_to_k(preds: list[int], fallback_items: Iterable[int], *, blocked: set[int], topk: int) -> list[int]:
    used = set(preds)
    for item in fallback_items:
        item = int(item)
        if item in blocked or item in used:
            continue
        used.add(item)
        preds.append(item)
        if len(preds) >= topk:
            break
    return preds[:topk]


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached_ids_path", required=True, type=Path)
    ap.add_argument("--tiger_config_path", required=True, type=Path)
    ap.add_argument("--decoder_ckpt", required=True, type=Path)
    ap.add_argument("--hf_model_path", required=True)
    ap.add_argument("--dataset_folder", default="/data/xqp_data/RecSys26/datasets/rqvae_recommender/setrec")
    ap.add_argument("--export_domain", default="amazon23_vg_tiger_strict")
    ap.add_argument("--eval_dataset", default="amazon23_vg")
    ap.add_argument("--eval_data_dir", type=Path, default=Path("/data/xqp_data/RecSys26/third_party/SETRec/data"))
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--beam_size", type=int, default=64)
    ap.add_argument("--num_return_sequences", type=int, default=64)
    ap.add_argument("--topk_items", type=int, default=20)
    ap.add_argument("--k_values", default="", help="Comma-separated k values. Default: 1..semantic_code_length.")
    ap.add_argument("--max_users", type=int, default=0)
    args = ap.parse_args()

    root = Path("/data/xqp_data/RecSys26")
    tiger_root = root / "third_party" / "RQ_VAE_Recommender"
    scaling_root = root / "scripts" / "scaling"
    scripts_root = root / "scripts"
    for p in (str(tiger_root), str(scaling_root), str(scripts_root)):
        if p not in sys.path:
            sys.path.insert(0, p)

    with args.tiger_config_path.open() as f:
        tiger_config = json.load(f)

    from manifest_utils import patch_t5_semid_for_manifest, resolve_sid_layout

    sid_layout = resolve_sid_layout(tiger_config=tiger_config, cached_ids_path=args.cached_ids_path)
    per_pos_sizes = sid_layout["per_pos_sizes"]
    sem_id_dim = int(sid_layout["sem_id_dim"])
    semantic_code_length = int(tiger_config.get("semantic_code_length", sem_id_dim))
    codebook_size = int(sid_layout["max_codebook_size"])
    sid_tokens = int(sid_layout["n_sid_tokens"])
    duplicate_policy = sid_layout.get("duplicate_policy", tiger_config.get("duplicate_policy", "first"))

    if semantic_code_length > sem_id_dim:
        raise ValueError(f"semantic_code_length={semantic_code_length} > sem_id_dim={sem_id_dim}")
    if args.k_values:
        k_values = [int(x) for x in args.k_values.split(",") if x.strip()]
    else:
        k_values = list(range(1, semantic_code_length + 1))
    if any(k <= 0 or k > semantic_code_length for k in k_values):
        raise ValueError(f"Invalid k_values={k_values}; semantic_code_length={semantic_code_length}")

    patch_t5_semid_for_manifest(per_pos_sizes=per_pos_sizes, duplicate_policy=duplicate_policy)

    from cached_id_tokenizer import CachedIdTokenizer
    from data.processed import ItemData, SeqData, RecDataset
    from data.utils import batch_to
    from modules.tokenizer.t5_semid import (
        build_t5_tokenizer_and_model,
        build_token_seq_to_item_and_trie,
        make_prefix_allowed_tokens_fn,
        make_t5_inputs,
    )
    from unified_eval import compute_metrics
    from transformers import AutoTokenizer

    ckpt = torch.load(str(args.decoder_ckpt), map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    ckpt_vocab_size = int(state["encoder.embed_tokens.weight"].shape[0])
    base_tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model_path,
        local_files_only=("/" in args.hf_model_path),
        use_fast=True,
    )
    base_len = len(base_tokenizer)
    num_user_tokens = 0
    for candidate in range(0, 100000):
        new_vocab = base_len + sid_tokens + candidate
        if new_vocab == ckpt_vocab_size:
            num_user_tokens = candidate
            break
        if new_vocab > ckpt_vocab_size:
            num_user_tokens = max(0, candidate - 1)
            break

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(
        f"[prefix] device={device} sem_id_dim={sem_id_dim} semantic_code_length={semantic_code_length} "
        f"k_values={k_values} codebook_size={codebook_size} num_user_tokens={num_user_tokens}"
    )

    item_dataset = ItemData(
        root=args.dataset_folder,
        dataset=RecDataset.AMAZON,
        force_process=False,
        split=args.export_domain,
    )
    eval_dataset = SeqData(
        root=args.dataset_folder,
        dataset=RecDataset.AMAZON,
        is_train=False,
        subsample=False,
        split=args.export_domain,
    )

    gt = _load_warm_full_gt(args.eval_data_dir, args.eval_dataset)
    required_users = set(gt.keys())
    seq_user_ids = eval_dataset.sequence_data.get("rawUserId")
    if seq_user_ids is None:
        seq_user_ids = eval_dataset.sequence_data.get("userId")
    if seq_user_ids is None:
        raise KeyError("SeqData.sequence_data contains neither rawUserId nor userId")
    if isinstance(seq_user_ids, torch.Tensor):
        uid0 = seq_user_ids.detach().to(torch.long).cpu().reshape(-1) - 1
        keep_mask = torch.tensor([int(x) in required_users for x in uid0.tolist()], dtype=torch.bool)
        keep_idx = keep_mask.nonzero(as_tuple=False).reshape(-1)
        n_before = len(eval_dataset)
        for key, value in list(eval_dataset.sequence_data.items()):
            if isinstance(value, torch.Tensor) and value.shape[0] == n_before:
                eval_dataset.sequence_data[key] = value.index_select(0, keep_idx.to(value.device))
            elif isinstance(value, np.ndarray) and value.shape[0] == n_before:
                eval_dataset.sequence_data[key] = value[keep_mask.numpy()]
            elif isinstance(value, list) and len(value) == n_before:
                mask_list = keep_mask.tolist()
                eval_dataset.sequence_data[key] = [x for x, keep in zip(value, mask_list) if keep]
        print(f"[prefix] warm-only export users {n_before} -> {len(eval_dataset)}")

    if args.max_users and args.max_users > 0:
        n_before = len(eval_dataset)
        for key, value in list(eval_dataset.sequence_data.items()):
            if isinstance(value, torch.Tensor) and value.shape[0] == n_before:
                eval_dataset.sequence_data[key] = value[: args.max_users]
            elif isinstance(value, np.ndarray) and value.shape[0] == n_before:
                eval_dataset.sequence_data[key] = value[: args.max_users]
            elif isinstance(value, list) and len(value) == n_before:
                eval_dataset.sequence_data[key] = value[: args.max_users]
        print(f"[prefix] max_users export users {n_before} -> {len(eval_dataset)}")

    dataloader = DataLoader(eval_dataset, batch_size=int(args.batch_size), shuffle=False)

    semid_tokenizer = CachedIdTokenizer(
        cached_ids_path=str(args.cached_ids_path.resolve()),
        codebook_size=codebook_size,
        per_pos_sizes=per_pos_sizes,
    ).to(device)
    semid_tokenizer.precompute_corpus_ids(item_dataset)
    cached_ids = semid_tokenizer.cached_ids[:, :sem_id_dim].detach().cpu().to(torch.long)
    n_items = int(cached_ids.shape[0])

    hf_tokenizer, model, sid_token_ids, uid_token_ids = build_t5_tokenizer_and_model(
        hf_model_path=args.hf_model_path,
        codebook_size=codebook_size,
        sem_id_dim=sem_id_dim,
        num_user_tokens=num_user_tokens,
        local_files_only=("/" in args.hf_model_path),
    )
    pad_token_id = int(hf_tokenizer.pad_token_id)
    model.load_state_dict(_maybe_strip_module_prefix(state))
    del ckpt, state
    model = model.to(device)
    model.eval()

    token_seq_to_item, trie = build_token_seq_to_item_and_trie(cached_ids, sid_token_ids=sid_token_ids)
    prefix_allowed_tokens_fn = make_prefix_allowed_tokens_fn(
        trie=trie,
        decoder_start_token_id=int(model.config.decoder_start_token_id),
        eos_token_id=int(model.config.eos_token_id),
    )

    pos = torch.arange(sem_id_dim, dtype=torch.long).unsqueeze(0).expand(cached_ids.shape[0], -1)
    token_seqs = sid_token_ids[pos, cached_ids.to(torch.long)].cpu()
    popularity = _item_popularity(args.eval_data_dir, args.eval_dataset, n_items)
    fallback_items = _sort_items(range(n_items), popularity)
    seen_by_user = _load_seen_items(args.eval_data_dir, args.eval_dataset)
    prefix_groups = _build_prefix_groups(token_seqs, popularity, max(k_values))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    file_handles = {}
    pred_paths = {}
    for k in k_values:
        pred_path = args.output_dir / f"pred_prefix_k{k}.jsonl"
        pred_paths[k] = pred_path
        file_handles[k] = pred_path.open("w")

    aux_rows: list[dict] = []
    try:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Prefix export"):
                data = batch_to(batch, device)
                tokenized = semid_tokenizer(data)
                input_ids, attention_mask, _ = make_t5_inputs(
                    tokenized,
                    sid_token_ids=sid_token_ids,
                    uid_token_ids=uid_token_ids,
                    pad_token_id=pad_token_id,
                )
                user_ids = getattr(data, "raw_user_ids", None)
                if user_ids is None:
                    user_ids = data.user_ids
                if isinstance(user_ids, torch.Tensor) and user_ids.dim() > 1:
                    user_ids = user_ids.squeeze(-1)
                user_ids_0based = (user_ids.detach().to(torch.long).cpu() - 1).tolist()
                batch_seen = [
                    {int(x) for x in data.ids[i].detach().cpu().reshape(-1).tolist() if int(x) >= 0}
                    for i in range(int(input_ids.shape[0]))
                ]

                for k in k_values:
                    generated = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        num_beams=int(args.beam_size),
                        num_return_sequences=int(args.num_return_sequences),
                        max_new_tokens=int(k),
                        min_new_tokens=int(k),
                        do_sample=False,
                        early_stopping=True,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    )
                    generated = generated.reshape(int(input_ids.shape[0]), int(args.num_return_sequences), -1)
                    groups = prefix_groups[k]
                    fh = file_handles[k]

                    for i, uid in enumerate(user_ids_0based):
                        uid = int(uid)
                        seen_items = set(batch_seen[i])
                        seen_items.update(seen_by_user.get(uid, set()))
                        preds: list[int] = []
                        pred_seen: set[int] = set()
                        mapping_miss = 0
                        expanded_groups = 0
                        group_size_sum = 0

                        for j in range(int(args.num_return_sequences)):
                            seq = generated[i, j].detach().cpu().tolist()
                            if seq and int(seq[0]) == int(model.config.decoder_start_token_id):
                                seq = seq[1:]
                            while seq and int(seq[-1]) == int(model.config.eos_token_id):
                                seq = seq[:-1]
                            prefix = tuple(int(x) for x in seq[:k])
                            candidates = groups.get(prefix)
                            if not candidates:
                                mapping_miss += 1
                                continue
                            expanded_groups += 1
                            group_size_sum += len(candidates)
                            for item in candidates:
                                if item in seen_items or item in pred_seen:
                                    continue
                                pred_seen.add(item)
                                preds.append(int(item))
                                if len(preds) >= int(args.topk_items):
                                    break
                            if len(preds) >= int(args.topk_items):
                                break

                        preds = _fill_to_k(preds, fallback_items, blocked=seen_items, topk=int(args.topk_items))
                        fh.write(json.dumps({"user_id": uid, "predicted_items": preds}) + "\n")
                        aux_rows.append(
                            {
                                "k": k,
                                "user_id": uid,
                                "mapping_miss": mapping_miss,
                                "expanded_groups": expanded_groups,
                                "mean_expanded_group_size": (group_size_sum / expanded_groups) if expanded_groups else 0.0,
                                "num_predictions": len(preds),
                            }
                        )
    finally:
        for fh in file_handles.values():
            fh.close()

    metrics_rows: list[dict] = []
    metrics_by_k: dict[str, dict] = {}
    top_n_list = [5, 10]
    for k in k_values:
        predictions = _load_predictions_jsonl(pred_paths[k])
        metrics = compute_metrics(gt, predictions, top_n_list)
        out = {key: (round(float(value), 6) if isinstance(value, float) else value) for key, value in metrics.items()}
        out["_meta"] = {
            "k": k,
            "pred_file": str(pred_paths[k]),
            "eval_dataset": args.eval_dataset,
            "mode": "full",
            "warm_only": True,
            "top_n": top_n_list,
            "semantic_code_length": semantic_code_length,
            "decoder_code_length": sem_id_dim,
        }
        _write_json(args.output_dir / f"metrics_prefix_k{k}.json", out)
        row = {"k": k}
        for key in ("Recall@5", "NDCG@5", "Recall@10", "NDCG@10", "HR@10", "MRR@10"):
            row[key] = float(metrics.get(key, 0.0))
        metrics_rows.append(row)
        metrics_by_k[str(k)] = out
        print(
            f"[prefix:k={k}] "
            f"R@5={row['Recall@5']:.4f} N@5={row['NDCG@5']:.4f} "
            f"R@10={row['Recall@10']:.4f} N@10={row['NDCG@10']:.4f}"
        )

    _write_csv(
        args.output_dir / "prefix_metrics.csv",
        metrics_rows,
        ["k", "Recall@5", "NDCG@5", "Recall@10", "NDCG@10", "HR@10", "MRR@10"],
    )
    _write_csv(
        args.output_dir / "prefix_aux_by_user.csv",
        aux_rows,
        ["k", "user_id", "mapping_miss", "expanded_groups", "mean_expanded_group_size", "num_predictions"],
    )
    _write_json(
        args.output_dir / "summary.json",
        {
            "cached_ids_path": str(args.cached_ids_path),
            "tiger_config_path": str(args.tiger_config_path),
            "decoder_ckpt": str(args.decoder_ckpt),
            "semantic_code_length": semantic_code_length,
            "decoder_code_length": sem_id_dim,
            "k_values": k_values,
            "beam_size": int(args.beam_size),
            "num_return_sequences": int(args.num_return_sequences),
            "batch_size": int(args.batch_size),
            "topk_items": int(args.topk_items),
            "metrics_by_k": metrics_by_k,
        },
    )
    print(f"[prefix] done output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
