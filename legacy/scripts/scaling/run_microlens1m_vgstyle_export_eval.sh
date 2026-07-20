#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 <id_dir> <compat_dir> <model_size> <gpu> <run_root>" >&2
  exit 2
fi

ID_DIR="$1"
COMPAT_DIR="$2"
MODEL_SIZE="$3"
GPU="$4"
RUN_ROOT="$5"

ROOT="/data/xqp_data/RecSys26"
PY="${ROOT}/envs/miniforge3/envs/recsys26_rqvae/bin/python"
LOG_ROOT="${ROOT}/logs/scaling_microlens1m_vgstyle"
SOURCE_ID_ROOT="${ROOT}/logs/scaling_microlens1m_large_drop/${ID_DIR}"
COMPAT_ROOT="${SOURCE_ID_ROOT}/${COMPAT_DIR}"
DATASET_FOLDER="${ROOT}/datasets/rqvae_recommender/setrec"

case "${MODEL_SIZE}" in
  M01-t5s)
    HF="${ROOT}/models/hf/transformers/models--t5-small/snapshots/df1b051c49625cf57a3d0d8d3863ed4d13564fe4"
    EXPORT_BATCH_SIZE=96
    ;;
  M02-t5b)
    HF="${ROOT}/models/hf/transformers/models--t5-base/snapshots/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"
    EXPORT_BATCH_SIZE=48
    ;;
  M03-t5l)
    HF="${ROOT}/models/hf/transformers/models--t5-large/snapshots/150ebc2c4b72291e770f58e6057481c8d2ed331a"
    EXPORT_BATCH_SIZE=16
    ;;
  *)
    echo "Unknown model_size: ${MODEL_SIZE}" >&2
    exit 2
    ;;
esac

mkdir -p "${RUN_ROOT}/export"

CKPT="$(find "${RUN_ROOT}/decoder" -maxdepth 1 -name 'checkpoint_*.pt' | sort -V | tail -n 1)"
if [[ -z "${CKPT}" ]]; then
  echo "[export_eval] no checkpoint found in ${RUN_ROOT}/decoder" >&2
  exit 1
fi

echo "[export_eval] id=${ID_DIR} compat=${COMPAT_DIR} model=${MODEL_SIZE} gpu=${GPU}"
echo "[export_eval] run_root=${RUN_ROOT}"
echo "[export_eval] ckpt=${CKPT}"
echo "[export_eval] export_batch_size=${EXPORT_BATCH_SIZE} export_scope=warm_only"

CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" "${ROOT}/scripts/scaling/export_with_manifest.py" \
  --cached_ids_path "${COMPAT_ROOT}/cached_ids.npy" \
  --tiger_config_path "${COMPAT_ROOT}/tiger_config.json" \
  --domain microlens_1m \
  --decoder_ckpt "${CKPT}" \
  --dataset_folder "${DATASET_FOLDER}" \
  --export_path "${RUN_ROOT}/export/pred_topk.jsonl" \
  --hf_model_path "${HF}" \
  --beam_size 64 \
  --num_return_sequences 64 \
  --topk_items 20 \
  --batch_size "${EXPORT_BATCH_SIZE}" \
  --gpu 0 \
  --force_num_user_tokens 0 \
  --warm_only_export \
  --resume \
  2>&1 | tee "${RUN_ROOT}/export.log"

for mode in full loo; do
  "${PY}" "${ROOT}/scripts/unified_eval.py" \
    --pred "${RUN_ROOT}/export/pred_topk.jsonl" \
    --dataset microlens_1m \
    --mode "${mode}" \
    --warm_only \
    --top_n 5,10 \
    --output "${RUN_ROOT}/export/metrics_${mode}_warm_only.json" \
    2>&1 | tee "${RUN_ROOT}/eval_${mode}.log"
done

echo "[export_eval] done ${RUN_ROOT}"
