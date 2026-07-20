#!/usr/bin/env bash
set -euo pipefail

source /data/xqp_data/RecSys26/scripts/conda_activate.sh recsys26_setrec

cd /data/xqp_data/RecSys26/third_party/SETRec/code
mkdir -p /data/xqp_data/RecSys26/logs/setrec_smoke

log=/data/xqp_data/RecSys26/logs/setrec_smoke/run.log

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(/data/xqp_data/RecSys26/scripts/pick_gpu.sh)}"

setsid torchrun --nproc_per_node=1 --master_port=29501 finetune_t5.py \
  --base_model t5-small \
  --data_path ../data/toys/ \
  --output_dir /data/xqp_data/RecSys26/logs/setrec_smoke/out \
  --cache_dir /data/xqp_data/RecSys26/models/hf/transformers \
  --n_query 2 --n_sem 1 --n_cf 1 --alpha 0.7 \
  --layers 128 64 \
  --batch_size 4 --micro_batch_size 4 \
  --num_epochs 1 \
  --learning_rate 1e-3 \
  --cutoff_len 64 \
  --val_set_size 20 \
  --warmup_steps 0 \
  --lr_scheduler cosine \
  --seed 42 \
  --wandb_project "" \
  > "$log" 2>&1 &

pid=$!
echo "[SETRec] started pid=$pid CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
sleep 20

echo "[SETRec] stopping (smoke test)"
kill -- -"$pid" 2>/dev/null || true
sleep 2

tail -n 80 "$log" || true
