#!/usr/bin/env bash
set -euo pipefail

source /data/xqp_data/RecSys26/scripts/conda_activate.sh recsys26_etegrec

# missing lightweight deps
python -c "import colorama" >/dev/null 2>&1 || python -m pip install -q colorama

cd /data/xqp_data/RecSys26/third_party/ETEGRec
mkdir -p /data/xqp_data/RecSys26/logs/etegrec_setrec_smoke
log=/data/xqp_data/RecSys26/logs/etegrec_setrec_smoke/run.log

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(/data/xqp_data/RecSys26/scripts/pick_gpu.sh)}"

setsid python main.py --config config/setrec_beauty.yaml > "$log" 2>&1 &

pid=$!
echo "[ETEGRec] started pid=$pid (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
sleep 30

echo "[ETEGRec] stopping (smoke test)"
kill -- -"$pid" 2>/dev/null || true
sleep 2

tail -n 120 "$log" || true
