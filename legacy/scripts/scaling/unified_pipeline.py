#!/usr/bin/env python3
"""Scaling Law 实验 — 统一 T5 训练与导出 Pipeline

从 frozen manifest 驱动，支持所有 5 种 ID 类型和 3 种 T5 模型规模。
不依赖 TIGER 的 modules 包，直接使用 HuggingFace transformers。
"""
from __future__ import annotations

import json
import math
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import T5ForConditionalGeneration, T5Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest_utils import load_manifest, make_prefix_allowed_tokens_fn

# 与 generate_manifest.py 保持一致的 T5 vocab size
T5_VOCAB_SIZE = 32128


# ============================================================
# 数据加载：从 SETRec splits 构建训练/验证/测试数据
# ============================================================

def load_setrec_splits(setrec_root: str | Path) -> tuple[dict, dict, dict]:
    """加载 SETRec 标准 splits。"""
    root = Path(setrec_root)
    train_dict = np.load(root / "training_dict.npy", allow_pickle=True).item()
    val_dict = np.load(root / "validation_dict.npy", allow_pickle=True).item()
    test_dict = np.load(root / "testing_dict.npy", allow_pickle=True).item()
    return train_dict, val_dict, test_dict


def build_sequences_strict(
    train_dict: dict, val_dict: dict, test_dict: dict, min_train_len: int = 2
) -> tuple[dict[int, list[int]], dict[int, int]]:
    """构建严格序列（strict_no_test_history）。

    返回:
        user_histories: {uid: [item_ids]} — 用于 T5 encoder 输入的历史
        user_targets: {uid: target_item_id} — 用于 T5 decoder 的预测目标
    """
    user_histories: dict[int, list[int]] = {}
    user_targets: dict[int, int] = {}

    for u, train_seq in train_dict.items():
        u = int(u)
        train_seq = [int(x) for x in train_seq]
        if len(train_seq) < min_train_len:
            continue

        test_seq = test_dict.get(u, [])
        if not test_seq:
            continue
        test_item = int(test_seq[0])

        val_seq = val_dict.get(u, [])
        if val_seq:
            # 正常情况: history = train, target 用于训练时是 val, 测试时是 test
            # 训练数据: history=train[:-1], target=train[-1] (next-item prediction)
            # 这里为 test export 构建: history = train + [val0]
            history = train_seq + [int(val_seq[0])]
        else:
            # val 为空时: history = train (不用 test 回退)
            history = train_seq

        user_histories[u] = history
        user_targets[u] = test_item

    return user_histories, user_targets


def build_train_examples(
    train_dict: dict,
    val_dict: dict,
    item_to_token_seq: dict[int, list[int]],
    max_history_items: int = 20,
    target_len: int = 4,
) -> list[dict]:
    """构建训练样本：每个样本是 (source_token_ids, target_token_ids)。

    训练方式: 对每个用户的历史序列做 next-item prediction。
    source = [user_token] + flatten(history[-max_hist:] 各 item 的 token seq)
    target = next_item 的 token seq
    """
    examples = []

    for u, train_seq in train_dict.items():
        u = int(u)
        train_seq = [int(x) for x in train_seq]

        # 加入 val 作为额外训练信号（next-item 对）
        val_seq = val_dict.get(u, [])
        full_seq = train_seq + [int(v) for v in val_seq]

        if len(full_seq) < 2:
            continue

        # 为这个用户生成多个 next-item 训练样本
        for t in range(1, len(full_seq)):
            target_item = full_seq[t]
            if target_item not in item_to_token_seq:
                continue

            # 取最近 max_history_items 个 history item
            history = full_seq[max(0, t - max_history_items):t]

            # 构建 source tokens: user_token 占位（后续填入） + history item tokens
            history_tokens = []
            for item_id in history:
                if item_id in item_to_token_seq:
                    history_tokens.extend(item_to_token_seq[item_id])

            target_tokens = item_to_token_seq[target_item]

            examples.append({
                "user_id": u,
                "history_tokens": history_tokens,
                "target_tokens": target_tokens,
            })

    return examples


class ScalingDataset(Dataset):
    """统一的训练/测试 Dataset。"""

    def __init__(
        self,
        examples: list[dict],
        user_token_offset: int,
        pad_token_id: int,
        max_source_len: int,
        target_len: int,
    ):
        self.examples = examples
        self.user_token_offset = user_token_offset
        self.pad_token_id = pad_token_id
        self.max_source_len = max_source_len
        self.target_len = target_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        uid = ex["user_id"]

        # source: [user_token] + history_tokens, 右侧 pad
        user_tok = self.user_token_offset + uid
        source = [user_tok] + ex["history_tokens"]
        source = source[:self.max_source_len]  # 截断

        # target
        target = ex["target_tokens"][:self.target_len]

        return {
            "source": source,
            "target": target,
            "user_id": uid,
        }


def collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    """动态 padding collator。"""
    max_src = max(len(b["source"]) for b in batch)
    max_tgt = max(len(b["target"]) for b in batch)

    input_ids = []
    attention_mask = []
    labels = []
    user_ids = []

    for b in batch:
        src = b["source"]
        tgt = b["target"]

        pad_src = src + [pad_token_id] * (max_src - len(src))
        mask = [1] * len(src) + [0] * (max_src - len(src))
        pad_tgt = tgt + [-100] * (max_tgt - len(tgt))  # -100 = ignore in loss

        input_ids.append(pad_src)
        attention_mask.append(mask)
        labels.append(pad_tgt)
        user_ids.append(b["user_id"])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "user_ids": user_ids,
    }


# ============================================================
# 模型构建
# ============================================================

def build_model(
    hf_model_path: str,
    n_item_tokens: int,
    n_user_tokens: int,
    local_files_only: bool = True,
) -> tuple[T5ForConditionalGeneration, T5Tokenizer, int]:
    """构建扩展 vocab 的 T5 模型。

    返回: (model, tokenizer, pad_token_id)
    """
    tokenizer = T5Tokenizer.from_pretrained(hf_model_path, local_files_only=local_files_only)
    model = T5ForConditionalGeneration.from_pretrained(hf_model_path, local_files_only=local_files_only)

    # 扩展 vocab: item tokens + user tokens
    n_new = n_item_tokens + n_user_tokens
    original_vocab_size = model.config.vocab_size
    model.resize_token_embeddings(original_vocab_size + n_new)

    # 新 token 用小标准差随机初始化（对齐 T5 原始 embedding 的标准差）
    with torch.no_grad():
        embed = model.shared.weight
        original_std = embed[:original_vocab_size].std().item()
        embed[original_vocab_size:].normal_(mean=0.0, std=original_std)
        # decoder embed 也同步（T5 shared embedding）

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = 0

    print(f"[Model] {hf_model_path}: vocab {original_vocab_size} → {original_vocab_size + n_new} "
          f"(+{n_item_tokens} item tokens, +{n_user_tokens} user tokens)")

    return model, tokenizer, pad_token_id


# ============================================================
# 训练循环
# ============================================================

def train_model(
    model: T5ForConditionalGeneration,
    train_dataset: ScalingDataset,
    pad_token_id: int,
    config: dict,
    output_dir: Path,
    device: torch.device,
) -> Path:
    """T5 seq2seq 训练，返回最终 checkpoint 路径。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    lr = config["lr"]
    max_updates = config["max_updates"]
    warmup_ratio = config.get("warmup_ratio", 0.03)
    weight_decay = config.get("weight_decay", 0.035)
    grad_clip = config.get("grad_clip", 1.0)
    batch_size = config.get("per_device_batch_size", 32)
    grad_accum = config.get("grad_accum_steps", 1)
    eval_interval = config.get("eval_interval", max_updates // 20)
    seed = config.get("seed", 42)

    torch.manual_seed(seed)
    np.random.seed(seed)

    effective_batch = batch_size * grad_accum
    print(f"[Train] lr={lr}, max_updates={max_updates}, effective_batch={effective_batch}")

    collate = lambda batch: collate_fn(batch, pad_token_id)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate, num_workers=4, pin_memory=True, drop_last=True,
    )

    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_steps = int(max_updates * warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return 1.0  # constant after warmup (可以改成 cosine)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    accum_loss = 0.0
    log_interval = max(eval_interval // 10, 1)
    train_log = []

    data_iter = iter(train_loader)
    start_time = time.time()

    while global_step < max_updates:
        optimizer.zero_grad()

        for _ in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        global_step += 1

        if global_step % log_interval == 0:
            elapsed = time.time() - start_time
            avg_loss = accum_loss / log_interval
            print(f"  step={global_step}/{max_updates}  loss={avg_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  elapsed={elapsed:.0f}s")
            train_log.append({"step": global_step, "loss": avg_loss, "lr": scheduler.get_last_lr()[0]})
            accum_loss = 0.0

        if global_step % eval_interval == 0:
            ckpt_path = output_dir / f"checkpoint_{global_step}.pt"
            torch.save({"model": model.state_dict(), "step": global_step}, ckpt_path)
            print(f"  [Checkpoint] saved: {ckpt_path}")

    # 保存最终 checkpoint
    final_path = output_dir / "checkpoint_final.pt"
    torch.save({"model": model.state_dict(), "step": global_step}, final_path)
    print(f"[Train] 完成. 最终 checkpoint: {final_path}")

    # 保存训练日志
    (output_dir / "train_log.json").write_text(
        json.dumps(train_log, indent=2), encoding="utf-8"
    )

    return final_path


# ============================================================
# 导出预测
# ============================================================

def export_predictions(
    model: T5ForConditionalGeneration,
    manifest: dict,
    user_histories: dict[int, list[int]],
    user_targets: dict[int, int],
    item_to_token_seq: dict[int, list[int]],
    user_token_offset: int,
    pad_token_id: int,
    export_path: Path,
    device: torch.device,
    beam_size: int = 20,
    num_return_sequences: int = 20,
    topk_items: int = 10,
    target_len: int = 4,
    export_batch_size: int = 48,
    popularity_order: list[int] | None = None,
) -> dict:
    """Constrained beam search 导出 + dedup + seen-filter + popfill。

    返回 decode_stats。
    """
    model = model.to(device)
    model.eval()

    token_seq_to_item = manifest["token_seq_to_item"]
    trie = manifest["trie"]

    eos_token_id = model.config.eos_token_id
    decoder_start_token_id = model.config.decoder_start_token_id
    if decoder_start_token_id is None:
        decoder_start_token_id = pad_token_id

    prefix_fn = make_prefix_allowed_tokens_fn(trie, decoder_start_token_id, eos_token_id)

    # 收集所有测试用户
    test_users = sorted(user_histories.keys())
    n_users = len(test_users)

    # 统计
    total_mapping_miss = 0
    total_pre_filter_dup = 0
    shortfall_users = 0
    all_predicted_items: Counter = Counter()

    export_path.parent.mkdir(parents=True, exist_ok=True)

    with open(export_path, "w") as f:
        for batch_start in range(0, n_users, export_batch_size):
            batch_uids = test_users[batch_start:batch_start + export_batch_size]
            B = len(batch_uids)

            # 构建 input_ids
            batch_sources = []
            for uid in batch_uids:
                history = user_histories[uid][-20:]  # 最近 20 个
                user_tok = user_token_offset + uid
                src_tokens = [user_tok]
                for item_id in history:
                    if item_id in item_to_token_seq:
                        src_tokens.extend(item_to_token_seq[item_id])
                batch_sources.append(src_tokens)

            # Padding
            max_len = max(len(s) for s in batch_sources)
            input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((B, max_len), dtype=torch.long)
            for i, src in enumerate(batch_sources):
                input_ids[i, :len(src)] = torch.tensor(src, dtype=torch.long)
                attention_mask[i, :len(src)] = 1

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_beams=beam_size,
                    num_return_sequences=num_return_sequences,
                    max_new_tokens=target_len,
                    min_new_tokens=target_len,
                    do_sample=False,
                    early_stopping=True,
                    prefix_allowed_tokens_fn=prefix_fn,
                )

            generated = generated.reshape(B, num_return_sequences, -1)

            for i, uid in enumerate(batch_uids):
                seen_items = set(user_histories[uid])  # export history 全部已见
                preds: list[int] = []
                seen_pred: set[int] = set()

                batch_miss = 0
                batch_dup = 0

                for j in range(num_return_sequences):
                    seq = generated[i, j].tolist()
                    # 去掉 decoder_start_token_id
                    if seq and seq[0] == decoder_start_token_id:
                        seq = seq[1:]
                    seq = seq[:target_len]

                    key = "_".join(str(t) for t in seq)
                    item_id = token_seq_to_item.get(key)

                    if item_id is None:
                        batch_miss += 1
                        continue

                    if item_id in seen_pred:
                        batch_dup += 1
                        continue

                    seen_pred.add(item_id)

                    if item_id in seen_items:
                        continue  # seen-item filter

                    preds.append(item_id)
                    if len(preds) >= topk_items:
                        break

                total_mapping_miss += batch_miss
                total_pre_filter_dup += batch_dup

                # Popfill
                if len(preds) < topk_items and popularity_order is not None:
                    shortfall_users += 1
                    filled = set(preds) | seen_items
                    for pop_item in popularity_order:
                        if pop_item not in filled:
                            preds.append(pop_item)
                            filled.add(pop_item)
                            if len(preds) >= topk_items:
                                break

                preds = preds[:topk_items]
                for p in preds:
                    all_predicted_items[p] += 1

                f.write(json.dumps({"user_id": uid, "predicted_items": preds}) + "\n")

            if (batch_start // export_batch_size) % 10 == 0:
                print(f"  [Export] {batch_start + B}/{n_users} users processed")

    # 计算 decode_stats
    top1_counter = Counter()
    with open(export_path) as f:
        for line in f:
            obj = json.loads(line)
            if obj["predicted_items"]:
                top1_counter[obj["predicted_items"][0]] += 1

    top1_max = top1_counter.most_common(1)[0][1] if top1_counter else 0
    n_unique_items = len(all_predicted_items)

    decode_stats = {
        "n_test_users": n_users,
        "total_mapping_miss": total_mapping_miss,
        "total_pre_filter_duplicate": total_pre_filter_dup,
        "shortfall_users": shortfall_users,
        "pre_popfill_shortfall_ratio": shortfall_users / max(n_users, 1),
        "top1_concentration": top1_max / max(n_users, 1),
        "top10_unique_items": n_unique_items,
    }

    stats_path = export_path.parent / "decode_stats.json"
    stats_path.write_text(json.dumps(decode_stats, indent=2), encoding="utf-8")
    print(f"[Export] 完成: {export_path}")
    print(f"  mapping_miss={total_mapping_miss}, dup={total_pre_filter_dup}, "
          f"shortfall={shortfall_users}/{n_users}, unique_items={n_unique_items}")

    return decode_stats


# ============================================================
# Popularity 排序（用于 popfill）
# ============================================================

def compute_popularity_order(train_dict: dict, n_items: int) -> list[int]:
    """按训练集 popularity 降序排列所有 item。"""
    counter = Counter()
    for u, seq in train_dict.items():
        for item in seq:
            counter[int(item)] += 1
    # 所有 item 按频次降序，未出现的排最后
    return sorted(range(n_items), key=lambda x: -counter.get(x, 0))
