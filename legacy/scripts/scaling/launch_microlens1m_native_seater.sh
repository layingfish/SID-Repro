#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-paper}"
GPU="${2:-0}"

ROOT="${RECSYS26_ROOT:-/data/xqp_data/RecSys26}"
SEATER_CODE_DIR="${RECSYS26_SEATER_CODE_DIR:-${ROOT}/third_party/SEATER}"
DOMAIN="microlens_1m"
DATASET_NAME="SETRec_microlens_1m"
RUN_ROOT="${ROOT}/logs/native_microlens1m/seater/$(date -u +%Y%m%d_%H%M%S)__${PROFILE}__gpu${GPU}"
WORKSPACE_ROOT="${RUN_ROOT}/workspace"

case "${PROFILE}" in
  paper)
    EPOCHS=50
    BATCH_SIZE=512
    TEST_BATCH_SIZE=2048
    NUM_WORKERS=4
    ;;
  paper_evalsafe)
    EPOCHS=50
    BATCH_SIZE=512
    TEST_BATCH_SIZE=64
    NUM_WORKERS=4
    ;;
  smoke)
    EPOCHS=1
    BATCH_SIZE=128
    TEST_BATCH_SIZE=512
    NUM_WORKERS=0
    ;;
  smoke_evalsafe)
    EPOCHS=1
    BATCH_SIZE=128
    TEST_BATCH_SIZE=64
    NUM_WORKERS=0
    ;;
  *)
    echo "Unknown profile: ${PROFILE}. Use paper, paper_evalsafe, smoke, or smoke_evalsafe." >&2
    exit 2
    ;;
esac

mkdir -p "${RUN_ROOT}" "${WORKSPACE_ROOT}" "${RUN_ROOT}/export"
ln -sfn "${RUN_ROOT}" "${ROOT}/logs/native_microlens1m/seater/latest"

echo "[native_seater] root=${ROOT}"
echo "[native_seater] seater_code_dir=${SEATER_CODE_DIR}"
echo "[native_seater] run_root=${RUN_ROOT}"
echo "[native_seater] profile=${PROFILE} gpu=${GPU}"
echo "[native_seater] epochs=${EPOCHS} batch_size=${BATCH_SIZE} test_batch_size=${TEST_BATCH_SIZE}"

(
  source "${ROOT}/scripts/conda_activate.sh" recsys26_seater
  cd "${SEATER_CODE_DIR}"

  CUDA_VISIBLE_DEVICES="${GPU}" python main.py \
    --name "SEATER_${DATASET_NAME}_${PROFILE}" \
    --description "${PROFILE}" \
    --workspace "${WORKSPACE_ROOT}" \
    --dataset_name "${DATASET_NAME}" \
    --model SEATER \
    --vocab 8 \
    --gpu_id 0 \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --test_batch_size "${TEST_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --random_seed 42 \
    --no-tb \
    --no-train_tb

  WS_SUBDIR="$(find "${WORKSPACE_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
  CKPT="${WS_SUBDIR}/ckpt/best.pth"
  if [[ ! -s "${CKPT}" ]]; then
    echo "[native_seater] missing checkpoint: ${CKPT}" >&2
    exit 3
  fi
  echo "[native_seater] ckpt=${CKPT}"

  CUDA_VISIBLE_DEVICES="${GPU}" python "${ROOT}/scripts/exports/export_seater_topk.py" \
    --seater_code_dir "${SEATER_CODE_DIR}" \
    --dataset_name "${DATASET_NAME}" \
    --ckpt "${CKPT}" \
    --vocab 8 \
    --gpu_id 0 \
    --k 20 \
    --batch_size "${TEST_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --seed 42 \
    --pred_out "${RUN_ROOT}/export/pred_topk.jsonl"

  for mode in full loo; do
    python "${ROOT}/scripts/unified_eval.py" \
      --pred "${RUN_ROOT}/export/pred_topk.jsonl" \
      --dataset "${DOMAIN}" \
      --mode "${mode}" \
      --warm_only \
      --top_n 5,10 \
      --output "${RUN_ROOT}/export/metrics_${mode}_warm_only.json"
  done

  echo "[native_seater] DONE ${RUN_ROOT}"
) 2>&1 | tee "${RUN_ROOT}/run.log"
