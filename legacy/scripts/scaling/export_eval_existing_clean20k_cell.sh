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
ASSET_ROOT="${ROOT}/logs/scaling_microlens1m_retrain_clean"
COMPAT_ROOT="${ASSET_ROOT}/${ID_DIR}/${COMPAT_DIR}"
DATASET_FOLDER="${ROOT}/datasets/rqvae_recommender/setrec"

case "${MODEL_SIZE}" in
  M01-t5s)
    HF="${ROOT}/models/hf/transformers/models--t5-small/snapshots/df1b051c49625cf57a3d0d8d3863ed4d13564fe4"
    EXPORT_BS="${EXPORT_BS_OVERRIDE:-192}"
    ;;
  M02-t5b)
    HF="${ROOT}/models/hf/transformers/models--t5-base/snapshots/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"
    EXPORT_BS="${EXPORT_BS_OVERRIDE:-96}"
    ;;
  M03-t5l)
    HF="${ROOT}/models/hf/transformers/models--t5-large/snapshots/150ebc2c4b72291e770f58e6057481c8d2ed331a"
    EXPORT_BS="${EXPORT_BS_OVERRIDE:-32}"
    ;;
  *)
    echo "Unknown model_size: ${MODEL_SIZE}" >&2
    exit 2
    ;;
esac

if [[ ! -s "${COMPAT_ROOT}/cached_ids.npy" || ! -s "${COMPAT_ROOT}/tiger_config.json" ]]; then
  echo "[export-existing] missing tiger compat assets under ${COMPAT_ROOT}" >&2
  exit 1
fi

CKPT="$(find "${RUN_ROOT}/decoder" -maxdepth 1 -name 'checkpoint_*.pt' | sort -V | tail -n 1)"
if [[ -z "${CKPT}" ]]; then
  echo "[export-existing] no checkpoint found in ${RUN_ROOT}/decoder" >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}/export"

echo "[export-existing] id=${ID_DIR} model=${MODEL_SIZE} gpu=${GPU}"
echo "[export-existing] run_root=${RUN_ROOT}"
echo "[export-existing] compat_root=${COMPAT_ROOT}"
echo "[export-existing] ckpt=${CKPT}"
echo "[export-existing] hf=${HF}"
echo "[export-existing] export_batch_size=${EXPORT_BS} export_scope=warm_only"

CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" "${ROOT}/scripts/scaling/export_with_manifest.py" \
  --cached_ids_path "${COMPAT_ROOT}/cached_ids.npy" \
  --tiger_config_path "${COMPAT_ROOT}/tiger_config.json" \
  --domain microlens_1m \
  --decoder_ckpt "${CKPT}" \
  --dataset_folder "${DATASET_FOLDER}" \
  --export_path "${RUN_ROOT}/export/pred_topk.jsonl" \
  --hf_model_path "${HF}" \
  --beam_size 20 \
  --num_return_sequences 20 \
  --topk_items 10 \
  --batch_size "${EXPORT_BS}" \
  --gpu 0 \
  --warm_only_export \
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

echo "[export-existing] done ${RUN_ROOT}"
