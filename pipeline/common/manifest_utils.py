

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def build_trie(token_seqs: list[list[int]]) -> dict:


    trie: dict = {}
    for seq in token_seqs:
        node = trie
        for tok in seq:
            if tok not in node:
                node[tok] = {}
            node = node[tok]
    return trie


def make_prefix_allowed_tokens_fn(
    trie: dict,
    decoder_start_token_id: int,
    eos_token_id: int,
) -> callable:


    def _fn(batch_id: int, sent: "torch.LongTensor") -> list[int]:


        generated = sent.tolist()

        if generated and generated[0] == decoder_start_token_id:
            generated = generated[1:]

        node = trie
        for tok in generated:
            if tok in node:
                node = node[tok]
            else:

                return [eos_token_id]

        if node:
            return list(node.keys())
        else:

            return [eos_token_id]

    return _fn


def save_manifest(
    manifest_dir: str | Path,
    item_to_token_seq: dict[int, list[int]],
    meta: dict[str, Any],
) -> Path:


    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)


    i2t = {str(k): v for k, v in item_to_token_seq.items()}
    (manifest_dir / "item_to_token_seq.json").write_text(
        json.dumps(i2t, ensure_ascii=False), encoding="utf-8"
    )


    t2i: dict[str, int] = {}
    collisions = 0
    for item_id, seq in item_to_token_seq.items():
        key = "_".join(str(t) for t in seq)
        if key in t2i:
            collisions += 1
        else:
            t2i[key] = item_id
    (manifest_dir / "token_seq_to_item.json").write_text(
        json.dumps(t2i, ensure_ascii=False), encoding="utf-8"
    )


    all_seqs = list(item_to_token_seq.values())
    trie = build_trie(all_seqs)
    with open(manifest_dir / "trie.pkl", "wb") as f:
        pickle.dump(trie, f)


    n_items = len(item_to_token_seq)
    unique_codes = len(t2i)
    target_len = len(all_seqs[0]) if all_seqs else 0
    all_tokens = set()
    per_pos_tokens: dict[int, set[int]] = {}
    for seq in all_seqs:
        for pos, tok in enumerate(seq):
            all_tokens.add(tok)
            if pos not in per_pos_tokens:
                per_pos_tokens[pos] = set()
            per_pos_tokens[pos].add(tok)


    codebook_usage = {}
    for pos in sorted(per_pos_tokens.keys()):
        codebook_usage[f"pos_{pos}"] = len(per_pos_tokens[pos])

    stats = {
        "n_items": n_items,
        "target_len": target_len,
        "vocab_size": len(all_tokens),
        "unique_codes": unique_codes,
        "collision_rate": (n_items - unique_codes) / max(n_items, 1),
        "collisions": collisions,
        "codebook_usage_per_pos": codebook_usage,
    }
    (manifest_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )


    (manifest_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return manifest_dir


def load_manifest(manifest_dir: str | Path) -> dict[str, Any]:

    manifest_dir = Path(manifest_dir)

    with open(manifest_dir / "item_to_token_seq.json", "r") as f:
        i2t_raw = json.load(f)
    item_to_token_seq = {int(k): v for k, v in i2t_raw.items()}

    with open(manifest_dir / "token_seq_to_item.json", "r") as f:
        t2i_raw = json.load(f)
    token_seq_to_item = {k: int(v) for k, v in t2i_raw.items()}

    with open(manifest_dir / "trie.pkl", "rb") as f:
        trie = pickle.load(f)

    with open(manifest_dir / "stats.json", "r") as f:
        stats = json.load(f)

    with open(manifest_dir / "meta.json", "r") as f:
        meta = json.load(f)

    return {
        "item_to_token_seq": item_to_token_seq,
        "token_seq_to_item": token_seq_to_item,
        "trie": trie,
        "stats": stats,
        "meta": meta,
    }


def resolve_sid_layout(
    *,
    tiger_config: dict[str, Any],
    cached_ids_path: str | Path,
) -> dict[str, Any]:


    cached_ids = np.load(cached_ids_path, mmap_mode="r")
    if cached_ids.ndim != 2:
        raise ValueError(f"cached_ids must be 2D, got shape={tuple(cached_ids.shape)}")

    sem_id_dim = int(cached_ids.shape[1])
    per_pos_sizes = tiger_config.get("per_pos_sizes")
    if per_pos_sizes is None:
        per_pos_sizes = tiger_config.get("codebook_size_per_level")

    if per_pos_sizes is None:
        codebook_size = int(tiger_config.get("codebook_size", 0))
        if codebook_size <= 0:
            raise ValueError(
                "tiger_config must contain either per_pos_sizes/codebook_size_per_level "
                "or a positive scalar codebook_size"
            )
        per_pos_sizes = [codebook_size] * sem_id_dim
    else:
        per_pos_sizes = [int(x) for x in per_pos_sizes]
        if len(per_pos_sizes) != sem_id_dim:
            raise ValueError(
                f"per_pos_sizes length mismatch: len={len(per_pos_sizes)} sem_id_dim={sem_id_dim}"
            )

    max_codes = np.asarray(cached_ids.max(axis=0)).astype(np.int64).tolist()
    min_codes = np.asarray(cached_ids.min(axis=0)).astype(np.int64).tolist()
    for pos, (lo, hi, size) in enumerate(zip(min_codes, max_codes, per_pos_sizes)):
        if lo < 0:
            raise ValueError(f"position {pos} contains negative code {lo}")
        if hi >= int(size):
            raise ValueError(
                f"position {pos} has code {hi} but per_pos_sizes[{pos}]={size}"
            )

    return {
        "sem_id_dim": sem_id_dim,
        "per_pos_sizes": per_pos_sizes,
        "max_codebook_size": max(int(x) for x in per_pos_sizes) if per_pos_sizes else 0,
        "n_sid_tokens": int(sum(int(x) for x in per_pos_sizes)),
        "duplicate_policy": tiger_config.get("duplicate_policy", "first"),
    }


def patch_t5_semid_for_manifest(
    *,
    per_pos_sizes: list[int] | tuple[int, ...],
    duplicate_policy: str = "first",
) -> None:

    import os
    import torch
    from torch import Tensor
    import modules.tokenizer.t5_semid as t5_semid_module

    per_pos_sizes = tuple(int(x) for x in per_pos_sizes)
    sem_id_dim = len(per_pos_sizes)
    max_codebook_size = max(per_pos_sizes) if per_pos_sizes else 0

    def _build_sid_tokens_variable(*, codebook_size: int, sem_id_dim: int) -> list[str]:
        if sem_id_dim != len(per_pos_sizes):
            raise ValueError(
                f"sem_id_dim mismatch: requested={sem_id_dim}, expected={len(per_pos_sizes)}"
            )
        tokens: list[str] = []
        for pos, size in enumerate(per_pos_sizes):
            tokens.extend(f"<sid_{pos}_{code}>" for code in range(int(size)))
        return tokens

    def _build_uid_tokens_variable(*, num_user_tokens: int) -> list[str]:
        if num_user_tokens < 0:
            raise ValueError(f"num_user_tokens must be non-negative, got {num_user_tokens}")
        return [f"<uid_{i}>" for i in range(num_user_tokens)]

    def build_token_id_tables_variable(
        tokenizer,
        *,
        codebook_size: int,
        sem_id_dim: int,
        num_user_tokens: int,
    ) -> tuple[Tensor, Tensor]:
        if sem_id_dim != len(per_pos_sizes):
            raise ValueError(
                f"sem_id_dim mismatch: requested={sem_id_dim}, expected={len(per_pos_sizes)}"
            )

        sid_token_ids = torch.full((sem_id_dim, max_codebook_size), -1, dtype=torch.long)
        for pos, size in enumerate(per_pos_sizes):
            for code in range(int(size)):
                tok = f"<sid_{pos}_{code}>"
                tid = tokenizer.convert_tokens_to_ids(tok)
                if tid is None or tid == tokenizer.unk_token_id:
                    raise ValueError(f"Missing token in tokenizer vocab: {tok}")
                sid_token_ids[pos, code] = int(tid)

        if num_user_tokens < 0:
            raise ValueError(f"num_user_tokens must be non-negative, got {num_user_tokens}")

        uid_token_ids = torch.empty((num_user_tokens,), dtype=torch.long)
        for b in range(num_user_tokens):
            tok = f"<uid_{b}>"
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is None or tid == tokenizer.unk_token_id:
                raise ValueError(f"Missing token in tokenizer vocab: {tok}")
            uid_token_ids[b] = int(tid)

        return sid_token_ids, uid_token_ids

    def build_t5_tokenizer_and_model_variable(
        *,
        hf_model_path: str,
        codebook_size: int,
        sem_id_dim: int,
        num_user_tokens: int,
        local_files_only: bool = True,
    ):
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        tokenizer = AutoTokenizer.from_pretrained(
            hf_model_path,
            local_files_only=local_files_only,
            use_fast=True,
        )

        special_tokens = _build_sid_tokens_variable(
            codebook_size=codebook_size,
            sem_id_dim=sem_id_dim,
        ) + _build_uid_tokens_variable(num_user_tokens=num_user_tokens)
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        model = T5ForConditionalGeneration.from_pretrained(
            hf_model_path,
            local_files_only=local_files_only,
        )
        model.resize_token_embeddings(len(tokenizer))

        if os.environ.get("TIGER_GRAD_CKPT", "") == "1":
            model.gradient_checkpointing_enable()
            print("[ACCEL] gradient_checkpointing enabled")

        sid_token_ids, uid_token_ids = build_token_id_tables_variable(
            tokenizer,
            codebook_size=codebook_size,
            sem_id_dim=sem_id_dim,
            num_user_tokens=num_user_tokens,
        )
        return tokenizer, model, sid_token_ids, uid_token_ids

    def make_t5_inputs_variable(
        tokenized,
        *,
        sid_token_ids: Tensor,
        uid_token_ids: Tensor,
        pad_token_id: int,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        if tokenized.seq_mask is None:
            raise ValueError("tokenized.seq_mask is None; did you forget to precompute_corpus_ids() first?")

        device = tokenized.sem_ids.device
        sid_token_ids = sid_token_ids.to(device)
        uid_token_ids = uid_token_ids.to(device)

        B, L = tokenized.sem_ids.shape
        use_uid_prefix = int(uid_token_ids.numel()) > 0
        prefix_len = 1 if use_uid_prefix else 0

        input_ids = torch.full((B, L + prefix_len), int(pad_token_id), dtype=torch.long, device=device)
        attention_mask = torch.zeros((B, L + prefix_len), dtype=torch.long, device=device)

        if use_uid_prefix:
            user_ids = tokenized.user_ids
            if user_ids.dim() > 1:
                user_ids = user_ids.squeeze(-1)
            bucketed_uids = t5_semid_module._bucketize_user_ids(
                user_ids,
                num_user_tokens=int(uid_token_ids.shape[0]),
            )
            input_ids[:, 0] = uid_token_ids[bucketed_uids]
            attention_mask[:, 0] = 1

        sem_ids_safe = tokenized.sem_ids.to(torch.long).clamp_min(0)
        token_types = tokenized.token_type_ids.to(torch.long)
        sem_tok = sid_token_ids[token_types, sem_ids_safe]

        mask = tokenized.seq_mask
        invalid_sem = (sem_tok < 0) & mask
        if bool(invalid_sem.any()):
            bad = invalid_sem.nonzero(as_tuple=False)[0].tolist()
            raise ValueError(f"Invalid SID lookup in inputs at index={bad}")

        input_ids[:, prefix_len:] = torch.where(mask, sem_tok, input_ids[:, prefix_len:])
        attention_mask[:, prefix_len:] = mask.to(torch.long)

        labels = None
        if tokenized.sem_ids_fut is not None:
            fut = tokenized.sem_ids_fut.to(torch.long).clamp_min(0)
            fut_types = tokenized.token_type_ids_fut.to(torch.long)
            labels = sid_token_ids[fut_types, fut]
            invalid_fut = labels < 0
            if bool(invalid_fut.any()):
                bad = invalid_fut.nonzero(as_tuple=False)[0].tolist()
                raise ValueError(f"Invalid SID lookup in labels at index={bad}")

        return input_ids, attention_mask, labels

    def build_token_seq_to_item_and_trie_variable(
        cached_ids: Tensor,
        *,
        sid_token_ids: Tensor,
    ) -> tuple[dict[tuple[int, ...], int], dict[int, dict]]:
        if cached_ids.dim() != 2:
            raise ValueError(f"cached_ids must be 2D, got shape={tuple(cached_ids.shape)}")

        sem_id_dim_local = int(cached_ids.shape[1])
        pos = torch.arange(sem_id_dim_local, dtype=torch.long).unsqueeze(0).expand(cached_ids.shape[0], -1)
        token_seqs = sid_token_ids[pos, cached_ids.to(torch.long)]
        if bool((token_seqs < 0).any()):
            raise ValueError("Found invalid token lookup while building trie")

        token_seq_to_item: dict[tuple[int, ...], int] = {}
        trie: dict[int, dict] = {}
        keep_last = duplicate_policy == "last"

        for item_id in range(token_seqs.shape[0]):
            seq = tuple(int(x) for x in token_seqs[item_id].tolist())
            if keep_last or seq not in token_seq_to_item:
                token_seq_to_item[seq] = int(item_id)

            node = trie
            for t in seq:
                node = node.setdefault(int(t), {})

        return token_seq_to_item, trie

    t5_semid_module._build_sid_tokens = _build_sid_tokens_variable
    t5_semid_module.build_token_id_tables = build_token_id_tables_variable
    t5_semid_module.build_t5_tokenizer_and_model = build_t5_tokenizer_and_model_variable
    t5_semid_module.make_t5_inputs = make_t5_inputs_variable
    t5_semid_module.build_token_seq_to_item_and_trie = build_token_seq_to_item_and_trie_variable

    print(
        "[Patch] t5_semid variable codebook layout enabled: "
        f"per_pos_sizes={list(per_pos_sizes)}, duplicate_policy={duplicate_policy}"
    )


def validate_manifest(manifest_dir: str | Path) -> list[str]:

    data = load_manifest(manifest_dir)
    errors: list[str] = []

    i2t = data["item_to_token_seq"]
    t2i = data["token_seq_to_item"]
    trie = data["trie"]
    stats = data["stats"]


    for item_id, seq in i2t.items():
        key = "_".join(str(t) for t in seq)
        if key not in t2i:
            errors.append(f"item {item_id} 的 token seq {seq} 不在 token_seq_to_item 中")
        elif t2i[key] != item_id:

            pass

    for key, item_id in t2i.items():
        seq = [int(x) for x in key.split("_")]
        if item_id not in i2t:
            errors.append(f"token_seq_to_item 中 item {item_id} 不在 item_to_token_seq 中")
        elif i2t[item_id] != seq:
            errors.append(f"item {item_id}: i2t={i2t[item_id]} != t2i key={seq}")


    for item_id, seq in i2t.items():
        node = trie
        for tok in seq:
            if tok not in node:
                errors.append(f"item {item_id} seq {seq} 在 trie 中不完整")
                break
            node = node[tok]


    lengths = set(len(seq) for seq in i2t.values())
    if len(lengths) > 1:
        errors.append(f"target_len 不一致: {lengths}")


    if stats["n_items"] != len(i2t):
        errors.append(f"stats.n_items={stats['n_items']} != len(i2t)={len(i2t)}")

    return errors


def check_abort_criteria(stats: dict, id_type: str) -> tuple[list[str], list[str]]:

    hard_fails: list[str] = []
    warnings: list[str] = []

    n_items = stats["n_items"]
    collision_rate = stats.get("collision_rate", 0)
    unique_codes = stats.get("unique_codes", n_items)

    if id_type in ("I02-semrq", "I03-cfrq"):

        if unique_codes < n_items:
            hard_fails.append(f"unique_final={unique_codes} < n_items={n_items}")


        usage = stats.get("codebook_usage_per_pos", {})
        codebook_size = stats.get("codebook_size", 256)
        n_rq_levels = len(usage) - 1
        usage_ratios = []
        for pos, count in usage.items():
            pos_idx = int(pos.split("_")[1]) if "_" in pos else 0
            if pos_idx >= n_rq_levels:
                continue
            ratio = count / codebook_size
            usage_ratios.append(ratio)
            if ratio < 0.50:
                hard_fails.append(f"{pos} codebook_usage={ratio:.2f} < 0.50")
            elif ratio < 0.70:
                warnings.append(f"{pos} codebook_usage={ratio:.2f} < 0.70")
        if usage_ratios:
            mean_usage = sum(usage_ratios) / len(usage_ratios)
            if mean_usage < 0.70:
                hard_fails.append(f"mean_codebook_usage={mean_usage:.2f} < 0.70")

    elif id_type == "I05-letter":

        if collision_rate > 0.05:
            hard_fails.append(f"collision_rate={collision_rate:.4f} > 0.05")
        elif collision_rate > 0.02:
            warnings.append(f"collision_rate={collision_rate:.4f} > 0.02")

    return hard_fails, warnings
