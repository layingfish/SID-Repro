#!/usr/bin/env bash
set -euo pipefail

RECSYS26_ROOT="${RECSYS26_ROOT:-/data/xqp_data/RecSys26}"
GPUS="${RECSYS26_LLM_ID_GPU_POOL:-5 6}"   # space-separated physical GPU ids
COUNT=2

out_dir="$RECSYS26_ROOT/logs/repro_paper/queue_llm_id"
mkdir -p "$out_dir"
ts="$(date +%Y%m%d_%H%M%S)"

DOMAINS_STR="${RECSYS26_SETREC_DOMAINS:-beauty toys sports steam}"
# shellcheck disable=SC2206
DOMAINS=( $DOMAINS_STR )

VARIANTS_STR="${RECSYS26_LLM_ID_VARIANTS:-sid semid cid hid}"
# shellcheck disable=SC2206
VARIANTS=( $VARIANTS_STR )

should_skip() {
  local domain="$1"
  local variant="$2"

  local root="$RECSYS26_ROOT/logs/repro_paper/llm_id/$variant/$domain"
  local latest="$root/latest"

  if [[ -d "$latest" ]]; then
    if [[ -f "$latest/exit_code.txt" ]]; then
      local code
      code="$(cat "$latest/exit_code.txt" 2>/dev/null || true)"
      if [[ "$code" == "0" ]]; then
        echo "[skip done] llm_id $domain $variant"
        return 0
      fi
    else
      if [[ -s "$latest/run.log" ]]; then
        echo "[skip running] llm_id $domain $variant"
        return 0
      fi
    fi
  fi
  return 1
}

queued=0
for domain in "${DOMAINS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    if should_skip "$domain" "$variant"; then
      continue
    fi

    out="$out_dir/${ts}_llm_id_${variant}_${domain}.out"
    echo "[queue] llm_id $domain $variant -> $out (GPUS='$GPUS')"

    setsid "$RECSYS26_ROOT/scripts/with_gpu_lock_multi.sh" --gpus "$GPUS" --count "$COUNT" -- \
      "$RECSYS26_ROOT/scripts/repro_dispatch_paper.sh" llm_id "$domain" "$variant" \
      >"$out" 2>&1 &

    queued=$((queued + 1))
    sleep 0.2
  done
done

echo "[queue] queued=$queued (GPUS='$GPUS' count=$COUNT)"
