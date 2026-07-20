#!/usr/bin/env bash
set -euo pipefail

RECSYS26_ROOT="${RECSYS26_ROOT:-/data/xqp_data/RecSys26}"

# Comma-separated list for CUDA_VISIBLE_DEVICES.
SETREC_GPUS_CSV="${RECSYS26_SETREC_GPUS_CSV:-3,4,5,6}"
FREE_MIB="${RECSYS26_GPU_FREE_MIB:-500}"
SLEEP_SEC="${RECSYS26_GPU_POLL_SEC:-60}"

# Domains to run (space-separated).
DOMAINS_STR="${RECSYS26_SETREC_DOMAINS:-beauty toys sports steam}"
# shellcheck disable=SC2206
DOMAINS=( $DOMAINS_STR )

IFS="," read -ra GPUS <<<"$SETREC_GPUS_CSV"

need=${#GPUS[@]}
if [[ "$need" -lt 4 ]]; then
  echo "[queue_setrec_paper_all] need 4 GPUs; got SETREC_GPUS_CSV=$SETREC_GPUS_CSV" >&2
  exit 2
fi

get_used_mib() {
  local gpu="$1"
  # Use timeout to avoid hangs.
  local out
  if ! out=$(timeout 5 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | head -n 1); then
    echo "-1"
    return 0
  fi
  out="${out//[[:space:]]/}"
  if [[ "$out" == "" ]]; then
    echo "-1"
  else
    echo "$out"
  fi
}

wait_for_free() {
  echo "[queue_setrec_paper_all] waiting for GPUs=$SETREC_GPUS_CSV to be <= ${FREE_MIB}MiB used..." >&2
  while true; do
    local ok=1
    for g in "${GPUS[@]}"; do
      used=$(get_used_mib "$g")
      if [[ "$used" -lt 0 ]]; then
        ok=0
        break
      fi
      if [[ "$used" -gt "$FREE_MIB" ]]; then
        ok=0
        break
      fi
    done
    if [[ "$ok" -eq 1 ]]; then
      echo "[queue_setrec_paper_all] GPUs appear free; starting SETRec paper runs." >&2
      return 0
    fi
    sleep "$SLEEP_SEC"
  done
}

wait_for_free

export CUDA_VISIBLE_DEVICES="$SETREC_GPUS_CSV"

for domain in "${DOMAINS[@]}"; do
  echo "[queue_setrec_paper_all] RUN setrec $domain (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)" >&2
  "$RECSYS26_ROOT/scripts/repro_dispatch_paper.sh" setrec "$domain"
  echo "[queue_setrec_paper_all] DONE setrec $domain" >&2
  echo >&2
  # small gap to avoid reusing master_port too quickly
  sleep 5
done

echo "[queue_setrec_paper_all] ALL DONE" >&2
