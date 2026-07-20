#!/usr/bin/env bash
# Scaling Law 实验 — 全 15 组实验编排脚本
#
# 用法:
#   bash scripts/scaling/run_all.sh [--phase P0|P1|P2]
#
# 远端执行:
#   cd /data/xqp_data/RecSys26 && bash scripts/scaling/run_all.sh --phase P1

set -euo pipefail

# ============================================================
# 路径配置
# ============================================================
RECSYS_ROOT="${RECSYS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
SETREC_ROOT="$RECSYS_ROOT/third_party/SETRec/data/microlens_50k"
LOG_ROOT="$RECSYS_ROOT/logs/scaling_ml50k"
SCRIPT_DIR="$RECSYS_ROOT/scripts/scaling"
EVAL_SCRIPT="$RECSYS_ROOT/scripts/unified_eval.py"
DOMAIN="microlens_50k"

# LETTER index 路径
LETTER_INDEX="$RECSYS_ROOT/third_party/LETTER/data/SETRec_microlens_50k/SETRec_microlens_50k.index.json"

# T5 模型路径
T5_SMALL="$RECSYS_ROOT/models/hf/transformers/models--t5-small/snapshots/df1b051c49625cf57a3d0d8d3863ed4d13564fe4"
T5_BASE="t5-base"
T5_LARGE="t5-large"

# GPU 锁脚本
GPU_LOCK="$RECSYS_ROOT/scripts/with_gpu_lock.sh"

PHASE="${1:---phase}"
PHASE="${2:-all}"
if [[ "$1" == "--phase" ]]; then
    PHASE="$2"
fi

echo "=========================================="
echo "Scaling Law 实验: phase=$PHASE"
echo "RECSYS_ROOT=$RECSYS_ROOT"
echo "=========================================="

# ============================================================
# P0: 协议冻结 + Manifest 生成
# ============================================================
run_p0() {
    echo ""
    echo "=== P0: Manifest 生成 ==="

    # I01-seq: Sequential ID
    echo "[P0] Generating I01-seq manifest..."
    python "$SCRIPT_DIR/generate_manifest.py" \
        --id_type seq \
        --setrec_root "$SETREC_ROOT" \
        --domain "$DOMAIN" \
        --output_dir "$LOG_ROOT/I01-seq/manifest"

    # I05-letter: LETTER Learned Code (从已有 index.json 转换)
    echo "[P0] Generating I05-letter manifest..."
    python "$SCRIPT_DIR/generate_manifest.py" \
        --id_type letter \
        --letter_index_path "$LETTER_INDEX" \
        --output_dir "$LOG_ROOT/I05-letter/manifest"

    # I02-semrq 和 I03-cfrq 需要先训练 RQ-VAE，此处仅打印提示
    echo ""
    echo "[P0] I02-semrq: 需先训练 Semantic RQ-VAE，然后运行:"
    echo "  python $SCRIPT_DIR/generate_manifest.py --id_type semrq --cached_ids_path <path> --output_dir $LOG_ROOT/I02-semrq/manifest"
    echo ""
    echo "[P0] I03-cfrq: 需先训练 Collaborative RQ-VAE，然后运行:"
    echo "  python $SCRIPT_DIR/generate_manifest.py --id_type cfrq --cached_ids_path <path> --output_dir $LOG_ROOT/I03-cfrq/manifest"
    echo ""
    echo "[P0] I04-semid: 需准备 SemID 层级路径文件，然后运行:"
    echo "  python $SCRIPT_DIR/generate_manifest.py --id_type semid --semid_paths_path <path> --output_dir $LOG_ROOT/I04-semid/manifest"

    echo ""
    echo "=== P0 完成 ==="
}

# ============================================================
# P1: t5-small 全跑
# ============================================================
run_p1() {
    echo ""
    echo "=== P1: t5-small 全跑 ==="

    ID_TYPES=("I01-seq" "I02-semrq" "I03-cfrq" "I04-semid" "I05-letter")

    for ID in "${ID_TYPES[@]}"; do
        MANIFEST="$LOG_ROOT/$ID/manifest"

        if [ ! -f "$MANIFEST/meta.json" ]; then
            echo "[SKIP] $ID: manifest 不存在，跳过"
            continue
        fi

        echo ""
        echo "[P1] Running $ID / M01-t5s ..."
        python "$SCRIPT_DIR/run_experiment.py" \
            --id_type "$ID" \
            --model_size M01-t5s \
            --manifest_dir "$MANIFEST" \
            --output_dir "$LOG_ROOT/$ID/M01-t5s" \
            --setrec_root "$SETREC_ROOT" \
            --max_updates 80000 \
            --per_device_batch_size 32 \
            --grad_accum_steps 8 \
            --seed 42 \
            --gpu 0

        # 运行评测
        PRED=$(find "$LOG_ROOT/$ID/M01-t5s/S3-eval" -name "pred_topk.jsonl" -type f | head -1)
        if [ -n "$PRED" ]; then
            echo "[P1] Evaluating $ID / M01-t5s (loo) ..."
            python "$EVAL_SCRIPT" --pred "$PRED" --dataset "$DOMAIN" --mode loo --warm_only --top_n 5,10
            echo "[P1] Evaluating $ID / M01-t5s (full) ..."
            python "$EVAL_SCRIPT" --pred "$PRED" --dataset "$DOMAIN" --mode full --warm_only --top_n 5,10
        fi
    done

    echo ""
    echo "=== P1 完成 ==="
}

# ============================================================
# P2: t5-base + t5-large
# ============================================================
run_p2() {
    echo ""
    echo "=== P2: t5-base + t5-large ==="

    ID_TYPES=("I01-seq" "I02-semrq" "I03-cfrq" "I04-semid" "I05-letter")
    MODEL_SIZES=("M02-t5b" "M03-t5l")

    for ID in "${ID_TYPES[@]}"; do
        MANIFEST="$LOG_ROOT/$ID/manifest"

        if [ ! -f "$MANIFEST/meta.json" ]; then
            echo "[SKIP] $ID: manifest 不存在，跳过"
            continue
        fi

        for MODEL in "${MODEL_SIZES[@]}"; do
            echo ""
            echo "[P2] Running $ID / $MODEL ..."
            python "$SCRIPT_DIR/run_experiment.py" \
                --id_type "$ID" \
                --model_size "$MODEL" \
                --manifest_dir "$MANIFEST" \
                --output_dir "$LOG_ROOT/$ID/$MODEL" \
                --setrec_root "$SETREC_ROOT" \
                --max_updates 80000 \
                --per_device_batch_size 16 \
                --grad_accum_steps 16 \
                --seed 42 \
                --gpu 0

            # 评测
            PRED=$(find "$LOG_ROOT/$ID/$MODEL/S3-eval" -name "pred_topk.jsonl" -type f | head -1)
            if [ -n "$PRED" ]; then
                python "$EVAL_SCRIPT" --pred "$PRED" --dataset "$DOMAIN" --mode loo --warm_only --top_n 5,10
                python "$EVAL_SCRIPT" --pred "$PRED" --dataset "$DOMAIN" --mode full --warm_only --top_n 5,10
            fi
        done
    done

    echo ""
    echo "=== P2 完成 ==="
}

# ============================================================
# 主入口
# ============================================================
case "$PHASE" in
    P0|p0) run_p0 ;;
    P1|p1) run_p1 ;;
    P2|p2) run_p2 ;;
    all)   run_p0 && run_p1 && run_p2 ;;
    *)     echo "Unknown phase: $PHASE (use P0/P1/P2/all)" && exit 1 ;;
esac
