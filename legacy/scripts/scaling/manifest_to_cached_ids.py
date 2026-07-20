#!/usr/bin/env python3
"""将 manifest 转换为 TIGER pipeline 需要的 cached_ids 格式。

TIGER pipeline 需要:
  - cached_ids: (n_items, sem_id_dim) int tensor, 每个位置的值是 0-indexed code
  - codebook_size: 每个位置的最大 code 数
  - sem_id_dim: code 长度

本脚本从 manifest 的 item_to_token_seq 逆向提取这些信息。

用法:
    python manifest_to_cached_ids.py \
        --manifest_dir logs/scaling_ml50k/I01-seq/manifest \
        --output_dir logs/scaling_ml50k/I01-seq/tiger_compat
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest_utils import load_manifest


def manifest_to_tiger_format(manifest_dir: Path, output_dir: Path) -> dict:
    """从 manifest 生成 TIGER 兼容的 cached_ids。

    返回 tiger_config dict，包含 codebook_size, sem_id_dim 等。
    """
    manifest = load_manifest(manifest_dir)
    i2t = manifest["item_to_token_seq"]
    meta = manifest["meta"]

    n_items = len(i2t)
    target_len = meta["target_len"]

    # 收集每个位置的所有 token_id，建立 per-position 映射: token_id → 0-indexed code
    per_pos_tokens: list[dict[int, int]] = [dict() for _ in range(target_len)]

    for item_id in range(n_items):
        seq = i2t[item_id]
        for pos in range(target_len):
            tok = seq[pos]
            if tok not in per_pos_tokens[pos]:
                per_pos_tokens[pos][tok] = len(per_pos_tokens[pos])

    # codebook_size = 每个位置的最大 code 数（取所有位置的最大值）
    per_pos_sizes = [len(m) for m in per_pos_tokens]
    codebook_size = max(per_pos_sizes)

    print(f"Per-position vocab sizes: {per_pos_sizes}")
    print(f"Unified codebook_size: {codebook_size}")

    # 构建 cached_ids (n_items, target_len)
    cached_ids = np.zeros((n_items, target_len), dtype=np.int64)
    for item_id in range(n_items):
        seq = i2t[item_id]
        for pos in range(target_len):
            cached_ids[item_id, pos] = per_pos_tokens[pos][seq[pos]]

    # 保存
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "cached_ids.npy", cached_ids)

    # 保存 per-position token 映射（用于导出时反向查找）
    # code_to_token[pos][code] = original_token_id
    code_to_token = []
    for pos in range(target_len):
        reverse = {v: k for k, v in per_pos_tokens[pos].items()}
        code_to_token.append(reverse)

    # 也保存 token_to_code 映射
    token_to_code = [dict(m) for m in per_pos_tokens]

    tiger_config = {
        "n_items": n_items,
        "sem_id_dim": target_len,
        "codebook_size": codebook_size,
        "per_pos_sizes": per_pos_sizes,
        "id_type": meta.get("id_type", "unknown"),
        "id_name": meta.get("id_name", "unknown"),
        "manifest_dir": str(manifest_dir),
    }

    (output_dir / "tiger_config.json").write_text(
        json.dumps(tiger_config, indent=2), encoding="utf-8"
    )

    # 保存映射表（用于导出阶段 token 转换）
    # 序列化: list of dict[int, int]
    mapping = {
        "token_to_code": [{str(k): v for k, v in m.items()} for m in token_to_code],
        "code_to_token": [{str(k): v for k, v in m.items()} for m in code_to_token],
    }
    (output_dir / "token_mapping.json").write_text(
        json.dumps(mapping), encoding="utf-8"
    )

    print(f"[OK] cached_ids: {cached_ids.shape}, codebook_size={codebook_size}, sem_id_dim={target_len}")
    print(f"[OK] Saved to {output_dir}")

    return tiger_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()

    manifest_to_tiger_format(args.manifest_dir, args.output_dir)


if __name__ == "__main__":
    main()
