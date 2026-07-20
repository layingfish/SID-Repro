#!/usr/bin/env python3
"""Scaling Law 实验 — 单个实验 cell 执行入口

用法:
    python run_experiment.py \
        --id_type I02-semrq --model_size M01-t5s \
        --manifest_dir logs/scaling_ml50k/I02-semrq/manifest \
        --output_dir logs/scaling_ml50k/I02-semrq/M01-t5s \
        --setrec_root /data/.../SETRec/data/microlens_50k \
        --lr 3e-4 --max_updates 80000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest_utils import load_manifest
from unified_pipeline import (
    build_model,
    build_sequences_strict,
    build_train_examples,
    collate_fn,
    compute_popularity_order,
    export_predictions,
    load_setrec_splits,
    ScalingDataset,
    train_model,
)

# T5 模型路径映射
T5_MODEL_PATHS = {
    "M01-t5s": "/data/xqp_data/RecSys26/models/hf/transformers/models--t5-small/snapshots/df1b051c49625cf57a3d0d8d3863ed4d13564fe4",
    "M02-t5b": "t5-base",   # 若无本地缓存，从 HF 下载
    "M03-t5l": "t5-large",
}

# T5 默认学习率
T5_DEFAULT_LR = {
    "M01-t5s": 3e-4,
    "M02-t5b": 1.5e-4,
    "M03-t5l": 1e-4,
}

T5_VOCAB_SIZE = 32128  # 与 generate_manifest.py 保持一致
MAX_HISTORY_ITEMS = 20


def main():
    parser = argparse.ArgumentParser(description="Scaling Law — 单实验 cell 执行")
    parser.add_argument("--id_type", required=True, help="I01-seq / I02-semrq / I03-cfrq / I04-semid / I05-letter")
    parser.add_argument("--model_size", required=True, help="M01-t5s / M02-t5b / M03-t5l")
    parser.add_argument("--manifest_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--setrec_root", required=True, type=Path)

    # 训练超参数
    parser.add_argument("--lr", type=float, default=None, help="学习率（默认按 model_size 选）")
    parser.add_argument("--max_updates", type=int, default=80000)
    parser.add_argument("--per_device_batch_size", type=int, default=32)
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="梯度累积步数，effective_batch = per_device * accum")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.035)
    parser.add_argument("--seed", type=int, default=42)

    # 导出参数
    parser.add_argument("--beam_size", type=int, default=20)
    parser.add_argument("--num_return_sequences", type=int, default=20)
    parser.add_argument("--topk_items", type=int, default=10)
    parser.add_argument("--export_batch_size", type=int, default=None,
                        help="导出 batch size（默认按 backbone: t5s=48, t5b=24, t5l=12）")

    # 模式控制
    parser.add_argument("--skip_train", action="store_true", help="跳过训练，直接导出")
    parser.add_argument("--checkpoint", type=Path, help="指定 checkpoint 路径（跳过训练时使用）")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--hf_local_files_only", action="store_true", default=False,
                        help="强制只使用本地文件（不从 HF 下载）")
    parser.add_argument("--domain", type=str, default="microlens_50k")

    args = parser.parse_args()

    # GPU 检查：保留禁止 GPU 2 的历史约束，其余实验卡按当前可用性调度。
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        parts = [p.strip() for p in cvd.replace(" ", ",").split(",") if p.strip()]
        allowed = {"0", "1", "3", "4", "5", "6"}
        forbidden = set(parts) - allowed
        if forbidden:
            raise RuntimeError(f"GPU {forbidden} is forbidden. Allowed: {allowed}")
        if len(parts) > 6:
            raise RuntimeError(f"同时最多 6 张 GPU，当前 {len(parts)} 张")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[Experiment] id_type={args.id_type}, model_size={args.model_size}, device={device}")

    # 加载 manifest
    manifest = load_manifest(args.manifest_dir)
    meta = manifest["meta"]
    item_to_token_seq = manifest["item_to_token_seq"]
    n_items = len(item_to_token_seq)
    target_len = meta["target_len"]
    n_item_tokens = meta["n_new_tokens"]

    print(f"[Manifest] n_items={n_items}, target_len={target_len}, n_item_tokens={n_item_tokens}")

    # 加载数据
    train_dict, val_dict, test_dict = load_setrec_splits(args.setrec_root)
    n_users = max(int(k) for k in train_dict.keys()) + 1
    print(f"[Data] n_users={n_users}")

    # user tokens: offset = T5_VOCAB_SIZE + n_item_tokens
    user_token_offset = T5_VOCAB_SIZE + n_item_tokens

    # 构建模型
    hf_model_path = T5_MODEL_PATHS.get(args.model_size, args.model_size)
    local_only = args.hf_local_files_only and "/" in hf_model_path
    model, tokenizer, pad_token_id = build_model(
        hf_model_path, n_item_tokens, n_users, local_files_only=local_only,
    )

    # 学习率
    lr = args.lr if args.lr is not None else T5_DEFAULT_LR.get(args.model_size, 3e-4)

    # Export batch size 按 backbone 选择
    EXPORT_BS = {"M01-t5s": 48, "M02-t5b": 24, "M03-t5l": 12}
    export_batch_size = args.export_batch_size if args.export_batch_size is not None else EXPORT_BS.get(args.model_size, 48)

    # S2: 训练
    decoder_dir = args.output_dir / "S2-decoder" / f"{time.strftime('%Y%m%d_%H%M%S')}__lr{lr}_u{args.max_updates}_s{args.seed}"

    if not args.skip_train:
        # 构建训练数据
        train_examples = build_train_examples(
            train_dict, val_dict, item_to_token_seq,
            max_history_items=MAX_HISTORY_ITEMS, target_len=target_len,
        )
        print(f"[Train] {len(train_examples)} training examples")

        max_source_len = 1 + MAX_HISTORY_ITEMS * target_len  # user_token + history
        train_dataset = ScalingDataset(
            train_examples, user_token_offset, pad_token_id,
            max_source_len=max_source_len, target_len=target_len,
        )

        train_config = {
            "lr": lr,
            "max_updates": args.max_updates,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "grad_clip": 1.0,
            "per_device_batch_size": args.per_device_batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "eval_interval": args.max_updates // 20,
            "seed": args.seed,
        }

        ckpt_path = train_model(model, train_dataset, pad_token_id, train_config, decoder_dir, device)
    else:
        ckpt_path = args.checkpoint
        if ckpt_path is None:
            raise ValueError("--skip_train requires --checkpoint")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        del ckpt

    # S3: 导出 + 评测
    eval_dir = args.output_dir / "S3-eval" / decoder_dir.name
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 构建测试数据
    user_histories, user_targets = build_sequences_strict(train_dict, val_dict, test_dict)
    popularity_order = compute_popularity_order(train_dict, n_items)

    pred_path = eval_dir / "pred_topk.jsonl"
    decode_stats = export_predictions(
        model, manifest, user_histories, user_targets,
        item_to_token_seq, user_token_offset, pad_token_id,
        pred_path, device,
        beam_size=args.beam_size, num_return_sequences=args.num_return_sequences,
        topk_items=args.topk_items, target_len=target_len,
        export_batch_size=export_batch_size,
        popularity_order=popularity_order,
    )

    # 写 manifest.json
    run_manifest = {
        "protocol_version": "scaling_ml50k_v1.1",
        "id_type": args.id_type,
        "t5_size": args.model_size,
        "uid_mode": "unique_token",
        "effective_num_user_tokens": n_users,
        "decode_mode": "constrained_trie+seen_filter+popfill",
        "beam_size": args.beam_size,
        "history_mode": "strict_no_test_history",
        "max_history_items": MAX_HISTORY_ITEMS,
        "batch_mode": "examples_per_update",
        "global_batch_examples": args.per_device_batch_size * args.grad_accum_steps,
        "tokenizer_manifest": str(args.manifest_dir),
        "final_ckpt": str(ckpt_path),
        "learning_rate": lr,
        "max_updates": args.max_updates,
        "seed": args.seed,
        "pred_path": str(pred_path),
        "decode_stats": decode_stats,
    }
    (eval_dir / "manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    # Decoder/export 熔断检查
    abort = False
    if decode_stats["total_mapping_miss"] > 0:
        print(f"[HARD FAIL] mapping_miss@10 = {decode_stats['total_mapping_miss']} > 0")
        abort = True
    if decode_stats["pre_popfill_shortfall_ratio"] > 0.50:
        print(f"[HARD FAIL] pre_popfill_shortfall = {decode_stats['pre_popfill_shortfall_ratio']:.2f} > 0.50")
        abort = True
    if decode_stats["top1_concentration"] > 0.25:
        print(f"[WARN] top1_concentration = {decode_stats['top1_concentration']:.2f} > 0.25")
    if abort:
        print("[ABORT] Decoder 熔断，请检查 tokenizer 或模型质量")
        sys.exit(3)

    # 运行 unified_eval.py
    project_root = Path(os.environ.get("RECSYS26_ROOT", Path(__file__).resolve().parents[2]))
    eval_script = project_root / "scripts" / "unified_eval.py"
    if eval_script.exists():
        import subprocess
        for mode in ("loo", "full"):
            metrics_path = eval_dir / f"metrics_{mode}_warm.json"
            cmd = [
                sys.executable, str(eval_script),
                "--pred", str(pred_path),
                "--dataset", args.domain,
                "--mode", mode,
                "--warm_only",
                "--top_n", "5,10",
                "--output", str(metrics_path),
            ]
            print(f"[Eval] {mode}: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[Eval ERROR] {result.stderr}")
            else:
                print(f"[Eval] {mode} -> {metrics_path}")
                if result.stdout:
                    print(result.stdout)
    else:
        print(f"[WARN] unified_eval.py not found at {eval_script}")
        print(f"  手动运行: python unified_eval.py --pred {pred_path} --dataset {args.domain} --mode loo --warm_only --top_n 5,10")

    print(f"\n[Done] 实验完成: {args.id_type} / {args.model_size}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Predictions: {pred_path}")


if __name__ == "__main__":
    main()
