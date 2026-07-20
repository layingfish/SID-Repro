#!/usr/bin/env python3
"""Scaling Law 实验 — 统一 Manifest 生成入口

为 5 种 ID 类型生成统一格式的 frozen manifest。
纯 Python + numpy，不依赖 torch 或远端 modules 包。

用法:
    python generate_manifest.py --id_type seq \
        --setrec_root /data/.../SETRec/data/microlens_50k \
        --output_dir /data/.../logs/scaling_ml50k/I01-seq/manifest

    python generate_manifest.py --id_type semrq \
        --cached_ids_path /path/to/semantic_cached_ids.npy \
        --output_dir /data/.../logs/scaling_ml50k/I02-semrq/manifest

    python generate_manifest.py --id_type letter \
        --letter_index_path /path/to/SETRec_microlens_50k.index.json \
        --output_dir /data/.../logs/scaling_ml50k/I05-letter/manifest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# 把 scripts/scaling 加入 path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest_utils import save_manifest, validate_manifest, check_abort_criteria


# T5 原始 vocab size（用于计算 token offset）
# 注意: T5ForConditionalGeneration.config.vocab_size 可能因 resize 而不同于此值。
# 此处定义的是"新 token 的起始 offset"，必须与 unified_pipeline.py 中一致。
# 若使用不同的 T5 模型，需确认其原始 vocab_size。
T5_VOCAB_SIZE = 32128  # T5 实际 vocab_size（32100 sentencepiece + 100 extra_ids + 28 padding to multiple of 128）


# ============================================================
# Sequential ID (I01-seq)
# ============================================================

def generate_seq_manifest(n_items: int, output_dir: Path) -> None:
    """每个 item 分配一个原子 special token。"""
    offset = T5_VOCAB_SIZE
    item_to_token_seq = {}
    for i in range(n_items):
        item_to_token_seq[i] = [offset + i]

    meta = {
        "id_type": "I01-seq",
        "id_name": "Sequential ID",
        "target_len": 1,
        "token_offset": offset,
        "n_new_tokens": n_items,
        "description": "每个 item 一个原子 special token，vocab 大（~14k）是 baseline 本体特征",
    }
    save_manifest(output_dir, item_to_token_seq, meta)
    print(f"[I01-seq] 生成完成: {n_items} items, offset={offset}")


# ============================================================
# RQ-VAE Code (I02-semrq / I03-cfrq)
# ============================================================

def generate_rqvae_manifest(
    cached_ids: np.ndarray,
    output_dir: Path,
    id_type: str,
    id_name: str,
    source_desc: str,
    codebook_size: int = 256,
) -> None:
    """从 RQ-VAE cached_ids (N, code_len) 生成 manifest。

    Token 编码: 每个位置 pos 的 token_id = offset + pos * codebook_size + code_value
    这样不同位置的 token 互不重叠（disjoint）。
    """
    n_items, code_len = cached_ids.shape
    offset = T5_VOCAB_SIZE
    n_new_tokens = code_len * codebook_size  # 例如 4 * 256 = 1024

    item_to_token_seq = {}
    for i in range(n_items):
        seq = []
        for pos in range(code_len):
            code_val = int(cached_ids[i, pos])
            tok_id = offset + pos * codebook_size + code_val
            seq.append(tok_id)
        item_to_token_seq[i] = seq

    meta = {
        "id_type": id_type,
        "id_name": id_name,
        "target_len": code_len,
        "token_offset": offset,
        "n_new_tokens": n_new_tokens,
        "codebook_size": codebook_size,
        "code_len": code_len,
        "source": source_desc,
        "description": f"{code_len} 个位置各 {codebook_size} 个 disjoint tokens，总计 {n_new_tokens}",
    }
    save_manifest(output_dir, item_to_token_seq, meta)
    print(f"[{id_type}] 生成完成: {n_items} items, code_len={code_len}, codebook={codebook_size}")


# ============================================================
# SemID† (I04-semid)
# ============================================================

def generate_semid_manifest(
    semid_paths: np.ndarray | list[list[str]],
    output_dir: Path,
) -> None:
    """从层级路径生成 manifest。

    输入 semid_paths: 每个 item 的层级路径列表，如:
        [["Electronics", "Phones", "Samsung", "leaf_0"],
         ["Electronics", "Phones", "Apple", "leaf_1"], ...]
    或者 npy 文件中的等价结构。

    每个唯一的层级节点映射为一个 special token。
    """
    if isinstance(semid_paths, np.ndarray):
        semid_paths = semid_paths.tolist()

    # 收集所有唯一节点，按 (position, node_name) 排序分配 token
    node_to_token: dict[tuple[int, str], int] = {}
    offset = T5_VOCAB_SIZE
    token_counter = 0

    # 先扫一遍收集所有节点
    all_nodes: list[tuple[int, str]] = []
    for path in semid_paths:
        for pos, node_name in enumerate(path):
            key = (pos, node_name)
            if key not in node_to_token:
                all_nodes.append(key)
                node_to_token[key] = -1  # 占位

    # 按 (pos, name) 排序后分配 token（确保确定性）
    all_nodes_sorted = sorted(set(all_nodes))
    for key in all_nodes_sorted:
        node_to_token[key] = offset + token_counter
        token_counter += 1

    # 生成 item_to_token_seq
    item_to_token_seq = {}
    for i, path in enumerate(semid_paths):
        seq = [node_to_token[(pos, name)] for pos, name in enumerate(path)]
        item_to_token_seq[i] = seq

    # 检查 target_len 一致性
    lengths = set(len(seq) for seq in item_to_token_seq.values())
    if len(lengths) != 1:
        print(f"[WARNING] SemID target_len 不一致: {lengths}")

    target_len = max(lengths) if lengths else 4

    meta = {
        "id_type": "I04-semid",
        "id_name": "SemID†",
        "target_len": target_len,
        "token_offset": offset,
        "n_new_tokens": token_counter,
        "n_unique_nodes": len(all_nodes_sorted),
        "description": "title-derived hierarchy, metadata-assisted; 每个层级节点映射为原子 special token",
        "node_to_token": {f"{pos}:{name}": tok for (pos, name), tok in node_to_token.items()},
    }
    save_manifest(output_dir, item_to_token_seq, meta)
    print(f"[I04-semid] 生成完成: {len(semid_paths)} items, {token_counter} unique tokens")


# ============================================================
# LETTER Learned Code (I05-letter)
# ============================================================

def generate_letter_manifest(
    letter_index_path: Path,
    output_dir: Path,
) -> None:
    """从 LETTER 的 index.json 转换为 protocol manifest。

    LETTER index.json 格式: {"0": ["<a_X>", "<b_Y>", "<c_Z>", "<d_W>"], ...}
    需要把字符串 token 映射为整数 token_id。
    """
    with open(letter_index_path, "r") as f:
        letter_index = json.load(f)

    # 收集所有唯一的 LETTER token 字符串
    str_token_set: set[str] = set()
    for item_id_str, token_strs in letter_index.items():
        for t in token_strs:
            str_token_set.add(t)

    # 按字母排序后分配 token_id（确保确定性）
    sorted_tokens = sorted(str_token_set)
    offset = T5_VOCAB_SIZE
    str_to_int: dict[str, int] = {}
    for idx, t in enumerate(sorted_tokens):
        str_to_int[t] = offset + idx

    # 生成 item_to_token_seq
    item_to_token_seq = {}
    for item_id_str, token_strs in letter_index.items():
        item_id = int(item_id_str)
        seq = [str_to_int[t] for t in token_strs]
        item_to_token_seq[item_id] = seq

    meta = {
        "id_type": "I05-letter",
        "id_name": "LETTER Learned Code",
        "target_len": len(next(iter(letter_index.values()))),
        "token_offset": offset,
        "n_new_tokens": len(sorted_tokens),
        "str_to_int_mapping": str_to_int,
        "source": str(letter_index_path),
        "description": "LETTER tokenizer 学习后的离散码，从 index.json 转换",
    }
    save_manifest(output_dir, item_to_token_seq, meta)
    print(f"[I05-letter] 生成完成: {len(item_to_token_seq)} items, {len(sorted_tokens)} unique tokens")


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Scaling Law — 统一 Manifest 生成")
    parser.add_argument("--id_type", required=True, choices=["seq", "semrq", "cfrq", "semid", "letter"])
    parser.add_argument("--output_dir", required=True, type=Path)

    # 数据源参数
    parser.add_argument("--setrec_root", type=Path, help="SETRec 数据目录 (含 *.emb-t5-tdcb.npy 等)")
    parser.add_argument("--domain", type=str, default="microlens_50k")
    parser.add_argument("--cached_ids_path", type=Path, help="RQ-VAE 输出的 cached_ids.npy (N, 4)")
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--semid_paths_path", type=Path, help="SemID 层级路径 .npy/.json 文件")
    parser.add_argument("--letter_index_path", type=Path, help="LETTER index.json 路径")

    args = parser.parse_args()

    if args.id_type == "seq":
        assert args.setrec_root is not None, "--setrec_root required for seq"
        emb_path = args.setrec_root / f"{args.domain}.emb-t5-tdcb.npy"
        emb = np.load(emb_path)
        n_items = emb.shape[0]
        del emb
        generate_seq_manifest(n_items, args.output_dir)

    elif args.id_type in ("semrq", "cfrq"):
        assert args.cached_ids_path is not None, "--cached_ids_path required for semrq/cfrq"
        cached_ids = np.load(args.cached_ids_path)
        id_type = "I02-semrq" if args.id_type == "semrq" else "I03-cfrq"
        id_name = "Semantic Code" if args.id_type == "semrq" else "Collaborative Code"
        source = "T5 embedding → RQ-VAE" if args.id_type == "semrq" else "SASRec embedding → RQ-VAE"
        generate_rqvae_manifest(cached_ids, args.output_dir, id_type, id_name, source, args.codebook_size)

    elif args.id_type == "semid":
        assert args.semid_paths_path is not None, "--semid_paths_path required for semid"
        p = args.semid_paths_path
        if p.suffix == ".json":
            with open(p) as f:
                semid_paths = json.load(f)
        else:
            semid_paths = np.load(p, allow_pickle=True).tolist()
        generate_semid_manifest(semid_paths, args.output_dir)

    elif args.id_type == "letter":
        assert args.letter_index_path is not None, "--letter_index_path required for letter"
        generate_letter_manifest(args.letter_index_path, args.output_dir)

    # 验证
    errors = validate_manifest(args.output_dir)
    if errors:
        print(f"[ERROR] Manifest 验证失败:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f"[OK] Manifest 验证通过")

    # 熔断检查
    with open(args.output_dir / "stats.json") as f:
        stats = json.load(f)
    id_type_full = {"seq": "I01-seq", "semrq": "I02-semrq", "cfrq": "I03-cfrq",
                    "semid": "I04-semid", "letter": "I05-letter"}[args.id_type]
    hard_fails, warnings = check_abort_criteria(stats, id_type_full)
    for w in warnings:
        print(f"[WARN] {w}")
    if hard_fails:
        print(f"[HARD FAIL] 熔断:")
        for hf in hard_fails:
            print(f"  - {hf}")
        sys.exit(2)


if __name__ == "__main__":
    main()
