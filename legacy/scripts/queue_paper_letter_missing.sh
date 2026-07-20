#!/usr/bin/env bash
set -euo pipefail

RECSYS26_ROOT="${RECSYS26_ROOT:-/data/xqp_data/RecSys26}"
GPUS="${RECSYS26_LETTER_GPU_POOL:-3 4 5 6}"  # space-separated physical GPU ids
COUNT=2

out_dir="$RECSYS26_ROOT/logs/repro_paper/queue_letter"
mkdir -p "$out_dir"
ts="$(date +%Y%m%d_%H%M%S)"

DOMAINS_STR="${RECSYS26_SETREC_DOMAINS:-beauty toys sports steam}"
# shellcheck disable=SC2206
DOMAINS=( $DOMAINS_STR )

should_skip() {
  local domain="$1"

  local root="$RECSYS26_ROOT/logs/repro_paper/letter/$domain"
  local latest="$root/latest"

  if [[ -d "$latest" ]]; then
    if [[ -f "$latest/exit_code.txt" ]]; then
      local code
      code="$(cat "$latest/exit_code.txt" 2>/dev/null || true)"
      if [[ "$code" == "0" ]]; then
        echo "[skip done] letter $domain"
        return 0
      fi
    else
      if [[ -s "$latest/run.log" ]]; then
        echo "[skip running] letter $domain"
        return 0
      fi
    fi
  fi

  return 1
}

queued=0
for domain in "${DOMAINS[@]}"; do
  if should_skip "$domain"; then
    continue
  fi

  out="$out_dir/${ts}_letter_${domain}.out"
  echo "[queue] letter $domain -> $out (GPUS=$GPUS count=$COUNT)"

  setsid "$RECSYS26_ROOT/scripts/with_gpu_lock_multi.sh" --gpus "$GPUS" --count "$COUNT" -- \
    "$RECSYS26_ROOT/scripts/repro_dispatch_paper.sh" letter "$domain" \
    >"$out" 2>&1 &

  queued=$((queued + 1))
  sleep 0.2
done

echo "[queue] queued=$queued (GPUS=$GPUS count=$COUNT)"
