#!/usr/bin/env bash
set -euo pipefail

source /data/xqp_data/RecSys26/scripts/conda_activate.sh recsys26_diffgrm

cd /data/xqp_data/RecSys26/third_party/DiffGRM
mkdir -p /data/xqp_data/RecSys26/logs/diffgrm_setrec_smoke

log=/data/xqp_data/RecSys26/logs/diffgrm_setrec_smoke/run.log

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(/data/xqp_data/RecSys26/scripts/pick_gpu.sh)}"

setsid python main.py \
  --model=DIFF_GRM \
  --dataset=SETRec \
  --category=beauty \
  --epochs=1 \
  --train_batch_size=8 \
  --eval_batch_size=8 \
  --cache_dir=/data/xqp_data/RecSys26/datasets/diffgrm_cache \
  --log_dir=/data/xqp_data/RecSys26/logs/diffgrm_setrec_smoke \
  --ckpt_dir=/data/xqp_data/RecSys26/logs/diffgrm_setrec_smoke/ckpt \
  --run_id=diffgrm_setrec_smoke \
  --num_proc=1 \
  --sent_emb_model=setrec_t5_tdcb \
  --sent_emb_dim=768 \
  --sent_emb_pca=0 \
  --force_regenerate_opq=False \
  --sent_emb_batch_size=256 \
  > "$log" 2>&1 &

pid=$!
echo "[DiffGRM] started pid=$pid CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
sleep 60

echo "[DiffGRM] stopping (smoke test)"
kill -- -"$pid" 2>/dev/null || true
sleep 2

tail -n 160 "$log" || true
