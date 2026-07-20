#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: with_gpu_lock_multi.sh --gpus \"3 4 5 6\" --count N -- <command...>" >&2
}

if [[ "$#" -lt 5 ]]; then
  usage
  exit 2
fi

if [[ "$1" != "--gpus" ]]; then
  usage
  exit 2
fi

gpu_list="$2"
shift 2

if [[ "$1" != "--count" ]]; then
  usage
  exit 2
fi

count="$2"
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

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "" ]]; then
  echo "[with_gpu_lock_multi] CUDA_VISIBLE_DEVICES already set to ${CUDA_VISIBLE_DEVICES}; skipping lock" >&2
  exec "$@"
fi

if ! [[ "$count" =~ ^[0-9]+$ ]] || [[ "$count" -le 0 ]]; then
  echo "[with_gpu_lock_multi] --count must be a positive int" >&2
  exit 2
fi

# shellcheck disable=SC2206
GPUS=( $gpu_list )

if [[ "${#GPUS[@]}" -lt "$count" ]]; then
  echo "[with_gpu_lock_multi] not enough GPUs in list: need $count have ${#GPUS[@]}" >&2
  exit 2
fi

RECSYS26_ROOT="${RECSYS26_ROOT:-/data/xqp_data/RecSys26}"
lock_dir="$RECSYS26_ROOT/tmp/gpu_locks"
mkdir -p "$lock_dir"

# Generate combinations of size N from GPUS (N is small: typically 2 or 4).
# Returns combos as space-separated gpu ids per line.
combinations() {
  local arr_name=$1
  local k=$2
  local start=${3:-0}
  local prefix=${4:-}

  # name reference: arr_name points to the caller-provided array variable
  local -n arr_ref="${arr_name}"

  if [[ "$k" -eq 0 ]]; then
    echo "${prefix# }"
    return 0
  fi

  local i
  for ((i=start; i<=${#arr_ref[@]}-k; i++)); do
    combinations "${arr_name}" "$((k-1))" "$((i+1))" "$prefix ${arr_ref[$i]}"
  done
}

while true; do
  while IFS= read -r combo; do
    # Try acquire all locks in this combo.
    # Lock in sorted order to avoid deadlocks.
    # shellcheck disable=SC2206
    combo_arr=( $combo )

    fds=()
    ok=1
    for gpu in "${combo_arr[@]}"; do
      lock_file="$lock_dir/gpu${gpu}.lock"
      exec {fd}>"$lock_file" || { ok=0; break; }
      if flock -n "$fd"; then
        fds+=("$fd")
      else
        ok=0
        # close fd
        eval "exec ${fd}>&-"
        break
      fi
    done

    if [[ "$ok" -eq 1 ]]; then
      # Success: keep FDs open across exec.
      export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${combo_arr[*]}")"
      echo "[with_gpu_lock_multi] acquired gpus=${CUDA_VISIBLE_DEVICES}" >&2
      exec "$@"
    fi

    # Failed: close any acquired FDs and continue.
    for fd in "${fds[@]}"; do
      eval "exec ${fd}>&-" || true
    done
  done < <(combinations GPUS "$count")

  sleep 10
done
