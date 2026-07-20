#!/usr/bin/env bash
set -euo pipefail

source /data/xqp_data/RecSys26/scripts/conda_activate.sh recsys26_letter

cd /data/xqp_data/RecSys26/third_party/LETTER/LETTER-TIGER
mkdir -p /data/xqp_data/RecSys26/logs/letter_tiger_smoke

log=/data/xqp_data/RecSys26/logs/letter_tiger_smoke/run.log

# IMPORTANT: without restricting visible GPUs, CUDA init can fail on this host
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(/data/xqp_data/RecSys26/scripts/pick_gpu.sh)}"

setsid python finetune.py \
  --base_model t5-small \
  --output_dir /data/xqp_data/RecSys26/logs/letter_tiger_smoke/out \
  --dataset SETRec_beauty \
  --index_file .index.json \
  --epochs 1 \
  --per_device_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --logging_step 20 \
  --warmup_ratio 0.0 \
  --save_and_eval_strategy steps \
  --save_and_eval_steps 2000 \
  --lr_scheduler_type cosine \
  --learning_rate 1e-3 \
  --seed 42 \
  > "$log" 2>&1 &

pid=$!
echo "[LETTER-TIGER] started pid=$pid"
sleep 20

echo "[LETTER-TIGER] stopping (smoke test)"
kill -- -"$pid" 2>/dev/null || true
sleep 2

tail -n 60 "$log" || true
