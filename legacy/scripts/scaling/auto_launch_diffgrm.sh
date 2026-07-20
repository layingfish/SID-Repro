#!/bin/bash
# 等 I03-seater 完成后自动启动 I06-diffgrm
source /data/xqp_data/RecSys26/scripts/conda_activate.sh recsys26_rqvae

echo "[Auto] Waiting for p1_I03-seater to finish..."
while tmux has-session -t p1_I03-seater 2>/dev/null; do
  sleep 60
done
echo "[Auto] I03-seater finished. Launching I06-diffgrm on GPU0..."

mkdir -p /data/xqp_data/RecSys26/logs/scaling_ml50k/I06-diffgrm/M01-t5s
SAVE_DIR="/data/xqp_data/RecSys26/logs/scaling_ml50k/I06-diffgrm/M01-t5s/decoder/"

CUDA_VISIBLE_DEVICES=0 python3 /data/xqp_data/RecSys26/scripts/scaling/train_with_manifest.py \
  --cached_ids_path /data/xqp_data/RecSys26/logs/scaling_ml50k/I06-diffgrm/tiger_compat/cached_ids.npy \
  --tiger_config_path /data/xqp_data/RecSys26/logs/scaling_ml50k/I06-diffgrm/tiger_compat/tiger_config.json \
  --gin_config configs/scaling_ml50k_t5small_base.gin \
  --gin_override "train.iterations=80000" "train.save_model_every=80000" "train.save_dir_root=\"$SAVE_DIR\"" \
  2>&1 | tee /data/xqp_data/RecSys26/logs/scaling_ml50k/I06-diffgrm/M01-t5s/train.log

echo "[Auto] I06-diffgrm finished."
