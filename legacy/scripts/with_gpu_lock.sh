#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: with_gpu_lock.sh --gpus \"3 4 5 6\" -- <command...>" >&2
}


if [[ "$#" -lt 3 ]]; then
  usage
  exit 2
fi

if [[ "$1" != "--gpus" ]]; then
  usage
  exit 2
fi

gpu_list="$2"
shift 2

if [[ "$1" != "--" ]]; then
  usage
  exit 2
fi
shift

if [[ "$#" -lt 1 ]]; then
  usage
  exit 2
fi
# If the caller already pinned GPUs, just run.
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "" ]]; then
  echo "[with_gpu_lock] CUDA_VISIBLE_DEVICES already set to ${CUDA_VISIBLE_DEVICES}; skipping lock" >&2
  exec "$@"
fi



RECSYS26_ROOT="${RECSYS26_ROOT:-/data/xqp_data/RecSys26}"
lock_dir="$RECSYS26_ROOT/tmp/gpu_locks"
mkdir -p "$lock_dir"

# Try to acquire one GPU lock non-blocking; if none available, wait and retry.
while true; do
  for gpu in $gpu_list; do
    lock_file="$lock_dir/gpu${gpu}.lock"
    # Allocate a new FD for each attempt and close it if we fail.
    exec {fd}>"$lock_file" || continue
    if flock -n "$fd"; then
      export CUDA_VISIBLE_DEVICES="$gpu"
      echo "[with_gpu_lock] acquired gpu=$gpu lock=$lock_file" >&2
      "$@"
      status=$?
      echo "[with_gpu_lock] releasing gpu=$gpu status=$status" >&2
      exit "$status"
    fi
    # Failed to lock; close FD.
    eval "exec ${fd}>&-"
  done
  sleep 10
done
