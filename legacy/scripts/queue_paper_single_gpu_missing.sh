#!/usr/bin/env bash
set -euo pipefail

RECSYS26_ROOT="${RECSYS26_ROOT:-/data/xqp_data/RecSys26}"
GPUS="${RECSYS26_PAPER_GPU_POOL:-3 4 5 6}"  # space-separated physical GPU ids

out_dir="$RECSYS26_ROOT/logs/repro_paper/queue_single_gpu"
mkdir -p "$out_dir"
ts="$(date +%Y%m%d_%H%M%S)"

# NOTE: Only single-GPU methods here.
# Add more entries as needed: "<method> <domain> [variant]"
TASKS=(
  "eager sports"
  "eager steam"
  "rpg sports"
  "rpg steam"
  "seater steam"
  "etegrec sports"
  "etegrec steam"
)

should_skip() {
  local method="$1"
  local domain="$2"

  local method_root="$RECSYS26_ROOT/logs/repro_paper/$method/$domain"
  local latest="$method_root/latest"

  if [[ -d "$latest" ]]; then
    if [[ -f "$latest/exit_code.txt" ]]; then
      local code
      code="$(cat "$latest/exit_code.txt" 2>/dev/null || true)"
      if [[ "$code" == "0" ]]; then
        echo "[skip done] $method $domain"
        return 0
      fi
    else
      # has a latest dir but no exit_code yet -> treat as running
      if [[ -s "$latest/run.log" ]]; then
        echo "[skip running] $method $domain"
        return 0
      fi
    fi
  fi
  return 1
}

queued=0
for task in "${TASKS[@]}"; do
  # shellcheck disable=SC2206
  parts=( $task )
  method="${parts[0]}"
  domain="${parts[1]}"

  if should_skip "$method" "$domain"; then
    continue
  fi

  out="$out_dir/${ts}_${method}_${domain}.out"
  echo "[queue] $method $domain -> $out"

  setsid "$RECSYS26_ROOT/scripts/with_gpu_lock.sh" --gpus "$GPUS" -- \
    "$RECSYS26_ROOT/scripts/repro_dispatch_paper.sh" "$method" "$domain" \
    >"$out" 2>&1 &

  queued=$((queued + 1))
  sleep 0.2

done

echo "[queue] queued=$queued (GPUS='$GPUS')"
