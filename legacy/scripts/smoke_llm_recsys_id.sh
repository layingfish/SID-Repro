#!/usr/bin/env bash
set -euo pipefail

source /data/xqp_data/RecSys26/scripts/conda_activate.sh recsys26_llm_id

cd /data/xqp_data/RecSys26/third_party/LLM_RecSys_ID
mkdir -p /data/xqp_data/RecSys26/logs/llm_recsys_id_smoke

log=/data/xqp_data/RecSys26/logs/llm_recsys_id_smoke/run.log

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29531


export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(/data/xqp_data/RecSys26/scripts/pick_gpu.sh)}"

setsid python main.py \
  --distributed --multiGPU \
  --task setrec_beauty \
  --data_dir data/ \
  --model_type t5-small \
  --item_representation no_tokenization \
  --data_order random \
  --epochs 1 \
  --lr 1e-3 \
  --logging_step 20 \
  --logging_dir /data/xqp_data/RecSys26/logs/llm_recsys_id_smoke/train.log \
  --model_dir /data/xqp_data/RecSys26/logs/llm_recsys_id_smoke/model.pt \
  --train_sequential_item_batch 8 \
  --train_sequential_yesno_batch 4 \
  --train_direct_yesno_batch 4 \
  --train_direct_candidate_batch 2 \
  --train_direct_straightforward_batch 4 \
  > "$log" 2>&1 &

pid=$!
echo "[LLM_RecSys_ID] started pid=$pid"
sleep 30

echo "[LLM_RecSys_ID] stopping (smoke test)"
kill -- -"$pid" 2>/dev/null || true
sleep 2

tail -n 80 "$log" || true
