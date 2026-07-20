#!/usr/bin/env bash
set -euo pipefail

source /data/xqp_data/RecSys26/scripts/conda_activate.sh recsys26_seater

cd /data/xqp_data/RecSys26/third_party/SEATER
mkdir -p /data/xqp_data/RecSys26/logs/seater_setrec_smoke

log=/data/xqp_data/RecSys26/logs/seater_setrec_smoke/run.log

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(/data/xqp_data/RecSys26/scripts/pick_gpu.sh)}"

setsid python main.py \
  --name SEATER_SETRec_beauty_smoke \
  --description smoke \
  --workspace /data/xqp_data/RecSys26/logs/seater_setrec_smoke/workspace \
  --dataset_name SETRec_beauty \
  --model SEATER \
  --vocab 8 \
  --gpu_id 0 \
  --epochs 1 \
  --batch_size 64 \
  --test_batch_size 128 \
  --num_workers 0 \
  --no-tb \
  --no-train_tb \
  > "$log" 2>&1 &

pid=$!
echo "[SEATER] started pid=$pid (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"

sleep 180

echo "[SEATER] stopping (smoke test)"
kill -- -"$pid" 2>/dev/null || true
sleep 2

tail -n 120 "$log" || true
