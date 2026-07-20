#!/usr/bin/env bash
set -euo pipefail

RECSYS26_ROOT="${RECSYS26_ROOT:-/data/xqp_data/RecSys26}"

LLM_GPUS_CSV="${RECSYS26_LLM_ID_GPUS_CSV:-1,2}"
FREE_MIB="${RECSYS26_GPU_FREE_MIB_LLM_ID:-2000}"
SLEEP_SEC="${RECSYS26_GPU_POLL_SEC:-60}"

DOMAINS_STR="${RECSYS26_SETREC_DOMAINS:-beauty toys sports steam}"
# shellcheck disable=SC2206
DOMAINS=( $DOMAINS_STR )

VARIANTS_STR="${RECSYS26_LLM_ID_VARIANTS:-sid semid cid hid}"
# shellcheck disable=SC2206
VARIANTS=( $VARIANTS_STR )

IFS="," read -ra GPUS <<<"$LLM_GPUS_CSV"
need=${#GPUS[@]}
if [[ "$need" -lt 2 ]]; then
  echo "[queue_llm_id_paper_all] need 2 GPUs; got LLM_GPUS_CSV=$LLM_GPUS_CSV" >&2
  exit 2
fi

get_used_mib() {
  local gpu="$1"
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
  echo "[queue_llm_id_paper_all] waiting for GPUs=$LLM_GPUS_CSV to be <= ${FREE_MIB}MiB used..." >&2
  while true; do
    local ok=1

    # Avoid starting while RPG is still running, since it can hold CUDA_VISIBLE_DEVICES=2 but not allocate VRAM yet.
    if pgrep -f "$RECSYS26_ROOT/scripts/repro_dispatch_paper.sh rpg" >/dev/null 2>&1; then
      ok=0
    fi

    if [[ "$ok" -eq 1 ]]; then
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
    fi

    if [[ "$ok" -eq 1 ]]; then
      echo "[queue_llm_id_paper_all] GPUs appear free; starting LLM_ID paper runs." >&2
      return 0
    fi

    sleep "$SLEEP_SEC"
  done
}

wait_for_free

export CUDA_VISIBLE_DEVICES="$LLM_GPUS_CSV"

for domain in "${DOMAINS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    echo "[queue_llm_id_paper_all] RUN llm_id $domain $variant (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)" >&2
    "$RECSYS26_ROOT/scripts/repro_dispatch_paper.sh" llm_id "$domain" "$variant"
    echo "[queue_llm_id_paper_all] DONE llm_id $domain $variant" >&2
    echo >&2
    sleep 5
  done
done

echo "[queue_llm_id_paper_all] ALL DONE" >&2
