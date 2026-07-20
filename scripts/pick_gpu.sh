#!/usr/bin/env bash
set -euo pipefail

# NOTE: Avoid querying GPUs via `nvidia-smi` on this host (it can hang).
# Prefer a fixed safe GPU list; callers that need mutual exclusion should use
# `scripts/with_gpu_lock.sh`.

gpus_str="${RECSYS26_GPUS:-0 1 3 4}"
# shellcheck disable=SC2206
GPUS=( $gpus_str )

if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo 0
  exit 0
fi

# Deterministic-ish selection to spread load when multiple jobs start together.
idx=$(( ( $(date +%s) + $$ ) % ${#GPUS[@]} ))
echo "${GPUS[$idx]}"
