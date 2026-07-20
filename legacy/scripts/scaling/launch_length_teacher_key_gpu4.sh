#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/xqp_data/RecSys26
PY=${ROOT}/envs/miniforge3/envs/recsys26_rqvae/bin/python
SCRIPT=${ROOT}/scripts/scaling/length_capacity_teacher_diagnostics.py
STUDY=${ROOT}/logs/code_length_study/amazon23_vg/20260421_clean_suffix_mainalign_t5small
OUT=${ROOT}/logs/code_length_study/amazon23_vg/20260429_capacity_teacher/teacher_key

export CUDA_VISIBLE_DEVICES=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUT}"
echo "[teacher-key] start $(date)"

"${PY}" "${SCRIPT}" \
  --study_root "${STUDY}" \
  --output_dir "${OUT}/rqvae" \
  --methods rqvae \
  --lengths 2,4,12 \
  --gpu 0 \
  --batch_size 16
echo "[teacher-key] done rqvae $(date)"

"${PY}" "${SCRIPT}" \
  --study_root "${STUDY}" \
  --output_dir "${OUT}/rqkmeans" \
  --methods rqkmeans \
  --lengths 2,3,4,12 \
  --gpu 0 \
  --batch_size 16
echo "[teacher-key] done rqkmeans $(date)"

"${PY}" "${SCRIPT}" \
  --study_root "${STUDY}" \
  --output_dir "${OUT}/opq" \
  --methods opq \
  --lengths 2,4,6,8,16 \
  --gpu 0 \
  --batch_size 16
echo "[teacher-key] done opq $(date)"
