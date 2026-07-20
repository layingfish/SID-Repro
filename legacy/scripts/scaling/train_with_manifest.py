#!/usr/bin/env python3
"""用 CachedIdTokenizer 替换 SemanticIdTokenizer，其余完全复用 TIGER pipeline。

用法:
    python train_with_manifest.py \
        --cached_ids_path logs/scaling_ml50k/I02-semrq/tiger_compat/cached_ids.npy \
        --tiger_config_path logs/scaling_ml50k/I02-semrq/tiger_compat/tiger_config.json \
        --gin_config configs/decoder_setrec_microlens_t5small_paper.gin \
        [--gin_override "train.iterations=80000"]

原理: 在 import train_decoder 之前，将 CachedIdTokenizer monkey-patch 到
modules.tokenizer.semids.SemanticIdTokenizer，使得 train_decoder.py 的代码
在构造 tokenizer 时自动使用我们的 drop-in 替代品。
"""
from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path


def _normalize_gin_override(override: str) -> str:
    if "=" not in override:
        return override
    key, value = override.split("=", 1)
    if key == "train.save_dir_root":
        stripped = value.strip()
        if (stripped.startswith('"') and stripped.endswith('"')) or (stripped.startswith("'") and stripped.endswith("'")):
            stripped = stripped[1:-1]
        if not stripped.endswith("/"):
            stripped = stripped + "/"
        value = '"' + stripped + '"'
    return f"{key}={value}"


def main():
    parser = argparse.ArgumentParser(description="Scaling Law — TIGER pipeline with manifest IDs")
    parser.add_argument("--cached_ids_path", required=True, type=Path)
    parser.add_argument("--tiger_config_path", required=True, type=Path)
    parser.add_argument("--gin_config", required=True, type=str, help="gin config file for train_decoder")
    parser.add_argument("--gin_override", type=str, nargs="*", default=[], help="additional gin overrides")
    parser.add_argument("--resume_from", type=str, default=None, help="checkpoint .pt path to resume from")
    parser.add_argument(
        "--vg_main_table_t5small_align",
        action="store_true",
        help="Align the training recipe to the VG main-table TIGER-noUID t5-small setup.",
    )
    parser.add_argument(
        "--vg_main_table_align_model_size",
        choices=["t5-small", "t5-base", "t5-large"],
        default="t5-small",
        help="Model size used under VG main-table alignment mode.",
    )
    args = parser.parse_args()

    # 读取 tiger_config
    with open(args.tiger_config_path) as f:
        tiger_config = json.load(f)

    cached_ids_path = str(args.cached_ids_path.resolve())

    # 设置 PYTHONPATH 指向 RQ_VAE_Recommender
    tiger_root = "/data/xqp_data/RecSys26/third_party/RQ_VAE_Recommender"
    scaling_root = str(Path(__file__).resolve().parent)

    if tiger_root not in sys.path:
        sys.path.insert(0, tiger_root)
    if scaling_root not in sys.path:
        sys.path.insert(0, scaling_root)

    from manifest_utils import patch_t5_semid_for_manifest, resolve_sid_layout

    sid_layout = resolve_sid_layout(
        tiger_config=tiger_config,
        cached_ids_path=args.cached_ids_path,
    )
    codebook_size = sid_layout["max_codebook_size"]
    sem_id_dim = sid_layout["sem_id_dim"]
    per_pos_sizes = sid_layout["per_pos_sizes"]
    print(f"[Manifest] codebook_size={codebook_size}, sem_id_dim={sem_id_dim}")
    print(f"[Manifest] cached_ids: {cached_ids_path}")
    print(f"[Manifest] per_pos_sizes={per_pos_sizes}")
    patch_t5_semid_for_manifest(
        per_pos_sizes=per_pos_sizes,
        duplicate_policy=sid_layout["duplicate_policy"],
    )

    # 在 import train_decoder 之前，注入 CachedIdTokenizer
    # 关键：不能 import modules.tokenizer.semids（会触发 polars 等依赖链）
    # 所以直接在 sys.modules 里注入一个 fake module

    from cached_id_tokenizer import CachedIdTokenizer

    class MockSemanticIdTokenizer(CachedIdTokenizer):
        """伪装成 SemanticIdTokenizer，接受并忽略 RQ-VAE 相关参数。"""

        def __init__(
            self,
            input_dim=None,
            output_dim=None,
            hidden_dims=None,
            codebook_size=None,
            n_layers=None,
            n_cat_feats=None,
            commitment_weight=None,
            rqvae_weights_path=None,
            rqvae_codebook_normalize=None,
            rqvae_sim_vq=None,
            **kwargs,
        ):
            super().__init__(
                cached_ids_path=cached_ids_path,
                codebook_size=codebook_size if codebook_size is not None else tiger_config["codebook_size"],
                per_pos_sizes=per_pos_sizes,
            )

    # 创建一个 fake module 并注入到 sys.modules，避免触发真实 import
    import types
    fake_semids = types.ModuleType("modules.tokenizer.semids")
    fake_semids.SemanticIdTokenizer = MockSemanticIdTokenizer
    sys.modules["modules.tokenizer.semids"] = fake_semids

    print("[Patch] SemanticIdTokenizer → MockSemanticIdTokenizer (via sys.modules injection)")

    # parse_config() 只接受一个位置参数 config_path
    # 所以我们生成一个临时 gin 文件，include 基础配置 + 所有 override
    import tempfile

    gin_config = args.gin_config
    gin_overrides = [
        f"train.vae_codebook_size={codebook_size}",
        f"train.vae_n_layers={sem_id_dim - 1}",  # n_layers = sem_id_dim - 1 (因为 +1 dedup suffix)
    ]

    if args.vg_main_table_t5small_align:
        gin_config = os.path.join(
            tiger_root,
            "configs",
            "decoder_setrec_yelp_t5small_v2_lr3e4_paper.gin",
        )
        model_overrides = {
            "t5-small": [
                'train.hf_model_path="/data/xqp_data/RecSys26/models/hf/transformers/models--t5-small/snapshots/df1b051c49625cf57a3d0d8d3863ed4d13564fe4"',
                "train.hf_local_files_only=True",
                "train.batch_size=64",
                "train.gradient_accumulate_every=4",
            ],
            "t5-base": [
                'train.hf_model_path="/data/xqp_data/RecSys26/models/hf/transformers/models--t5-base/snapshots/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"',
                "train.hf_local_files_only=True",
                "train.batch_size=32",
                "train.gradient_accumulate_every=8",
            ],
            "t5-large": [
                'train.hf_model_path="/data/xqp_data/RecSys26/models/hf/transformers/models--t5-large/snapshots/150ebc2c4b72291e770f58e6057481c8d2ed331a"',
                "train.hf_local_files_only=True",
                "train.batch_size=16",
                "train.gradient_accumulate_every=16",
            ],
        }
        gin_overrides.extend(
            [
                'train.dataset_split="amazon23_vg_tiger_strict"',
                "train.force_dataset_process=False",
                "train.num_user_tokens=0",
                "train.iterations=5000",
                "train.save_model_every=5000",
                "train.learning_rate=0.0003",
                "train.weight_decay=0.035",
            ]
        )
        gin_overrides.extend(model_overrides[args.vg_main_table_align_model_size])
        print(
            f"[Align] Using VG main-table TIGER-noUID recipe "
            f"with model_size={args.vg_main_table_align_model_size}"
        )

    # Resume 支持
    if args.resume_from:
        gin_overrides.append(f'train.pretrained_decoder_path="{args.resume_from}"')
        print(f"[Resume] from {args.resume_from}")

    gin_overrides += [_normalize_gin_override(x) for x in args.gin_override]

    # 写临时 gin 文件
    tmp_gin = tempfile.NamedTemporaryFile(mode="w", suffix=".gin", delete=False, dir=tiger_root)
    tmp_gin.write(f'include "{gin_config}"\n\n')
    for override in gin_overrides:
        tmp_gin.write(override + "\n")
    tmp_gin.flush()
    tmp_gin_path = tmp_gin.name
    tmp_gin.close()

    print(f"[Gin] Temp config: {tmp_gin_path}")
    with open(tmp_gin_path) as f:
        print(f.read())

    sys.argv = ["train_decoder.py", tmp_gin_path]

    # 切换工作目录到 TIGER 根目录
    os.chdir(tiger_root)

    # 必须先 import train_decoder（注册 @gin.configurable），再 parse_config
    from train_decoder import train
    from modules.utils import parse_config
    parse_config()

    train()

    # 清理临时文件
    os.unlink(tmp_gin_path)


if __name__ == "__main__":
    main()
