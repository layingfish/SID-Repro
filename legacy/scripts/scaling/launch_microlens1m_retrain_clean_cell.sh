#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <id_dir> <compat_dir> <model_size> <gpu>" >&2
  echo "  example: $0 I03-seater tiger_compat M01-t5s 0" >&2
  exit 2
fi

ID_DIR="$1"
COMPAT_DIR="$2"
MODEL_SIZE="$3"
GPU="$4"

ROOT="/data/xqp_data/RecSys26"
PY="${ROOT}/envs/miniforge3/envs/recsys26_rqvae/bin/python"
LOG_ROOT="${ROOT}/logs/scaling_microlens1m_retrain_clean"
ID_ROOT="${LOG_ROOT}/${ID_DIR}"
COMPAT_ROOT="${ID_ROOT}/${COMPAT_DIR}"
RUN_ROOT="${ID_ROOT}/${MODEL_SIZE}/$(date +%Y%m%d_%H%M%S)__gpu${GPU}"
DATASET_FOLDER="${ROOT}/datasets/rqvae_recommender/setrec"

case "${MODEL_SIZE}" in
  M01-t5s)
    GIN="${ROOT}/third_party/RQ_VAE_Recommender/configs/scaling_ml50k_t5small_base.gin"
    HF="${ROOT}/models/hf/transformers/models--t5-small/snapshots/df1b051c49625cf57a3d0d8d3863ed4d13564fe4"
    EXPORT_BS=192
    MODEL_OVERRIDES=(
      "train.hf_model_path=\"${HF}\""
      "train.hf_local_files_only=True"
    )
    ;;
  M02-t5b)
    GIN="${ROOT}/third_party/RQ_VAE_Recommender/configs/scaling_ml50k_t5base_base.gin"
    HF="${ROOT}/models/hf/transformers/models--t5-base/snapshots/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"
    EXPORT_BS=96
    MODEL_OVERRIDES=(
      "train.hf_model_path=\"${HF}\""
      "train.hf_local_files_only=True"
    )
    ;;
  M03-t5l)
    GIN="${ROOT}/third_party/RQ_VAE_Recommender/configs/scaling_ml50k_t5large_80k.gin"
    HF="${ROOT}/models/hf/transformers/models--t5-large/snapshots/150ebc2c4b72291e770f58e6057481c8d2ed331a"
    EXPORT_BS=32
    MODEL_OVERRIDES=(
      "train.hf_model_path=\"${HF}\""
      "train.hf_local_files_only=True"
      "train.save_model_every=80000"
    )
    ;;
  *)
    echo "Unknown model_size: ${MODEL_SIZE}" >&2
    exit 2
    ;;
esac

if [[ ! -s "${COMPAT_ROOT}/cached_ids.npy" || ! -s "${COMPAT_ROOT}/tiger_config.json" ]]; then
  echo "[cell] missing tiger compat assets under ${COMPAT_ROOT}" >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}/decoder" "${RUN_ROOT}/export"

echo "[cell] protocol=microlens1m_retrain_clean"
echo "[cell] id=${ID_DIR} compat=${COMPAT_DIR} model=${MODEL_SIZE} gpu=${GPU}"
echo "[cell] run_root=${RUN_ROOT}"
echo "[cell] compat_root=${COMPAT_ROOT}"
echo "[cell] gin=${GIN}"
echo "[cell] hf=${HF}"
echo "[cell] export_batch_size=${EXPORT_BS} export_scope=warm_only"

CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" "${ROOT}/scripts/scaling/train_with_manifest.py" \
  --cached_ids_path "${COMPAT_ROOT}/cached_ids.npy" \
  --tiger_config_path "${COMPAT_ROOT}/tiger_config.json" \
  --gin_config "${GIN}" \
  --gin_override \
    "train.dataset_split=\"microlens_1m\"" \
    "train.force_dataset_process=False" \
    "train.num_user_tokens=2000" \
    "train.save_dir_root=\"${RUN_ROOT}/decoder/\"" \
    "${MODEL_OVERRIDES[@]}" \
  2>&1 | tee "${RUN_ROOT}/train.log"

CKPT="$(find "${RUN_ROOT}/decoder" -maxdepth 1 -name 'checkpoint_*.pt' | sort -V | tail -n 1)"
if [[ -z "${CKPT}" ]]; then
  echo "[cell] no checkpoint found in ${RUN_ROOT}/decoder" >&2
  exit 1
fi
echo "[cell] ckpt=${CKPT}"

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

echo "[cell] done ${RUN_ROOT}"
