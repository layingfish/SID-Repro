#!/usr/bin/env bash
set -euo pipefail

source /data/xqp_data/RecSys26/scripts/conda_activate.sh recsys26_rqvae

mkdir -p /data/xqp_data/RecSys26/logs/rqvae_recommender_setrec_smoke
log=/data/xqp_data/RecSys26/logs/rqvae_recommender_setrec_smoke/run.log

cd /data/xqp_data/RecSys26/third_party/RQ_VAE_Recommender

# IMPORTANT: without restricting visible GPUs, CUDA init fails on this host
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(/data/xqp_data/RecSys26/scripts/pick_gpu.sh)}"

setsid python train_rqvae.py configs/rqvae_setrec_smoke.gin > "$log" 2>&1 &

pid=$!
echo "[RQ-VAE Recommender] started pid=$pid"

sleep 60

echo "[RQ-VAE Recommender] stopping (smoke test)"
kill -- -"$pid" 2>/dev/null || true
sleep 2

tail -n 120 "$log" || true
