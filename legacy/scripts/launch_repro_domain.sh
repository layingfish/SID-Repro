#!/usr/bin/env bash
set -euo pipefail

domain="${1:-beauty}"
case "$domain" in
  beauty|toys|sports|steam|amazon23_vg|microlens_50k|yelp) ;;
  *) echo "Unknown domain: $domain" >&2; exit 2 ;;
esac

RECSYS26_ROOT="${RECSYS26_ROOT:-/data/xqp_data/RecSys26}"
mode="${RECSYS26_MODE:-full}"                 # quick|full
gpus="${RECSYS26_GPUS:-3 4 5 6}"              # space-separated global GPU ids
llm_id_variant="${RECSYS26_LLM_ID_VARIANT:-semid}"

ts="$(date +%Y%m%d_%H%M%S)"
launcher_dir="$RECSYS26_ROOT/logs/repro_launcher/$domain/$ts"
mkdir -p "$launcher_dir"

echo "$mode" >"$launcher_dir/mode.txt"
echo "$gpus" >"$launcher_dir/gpus.txt"
echo "$llm_id_variant" >"$launcher_dir/llm_id_variant.txt"

methods=(setrec llm_id eager letter etegrec seater tiger rpg diffgrm)

echo "[launcher] domain=$domain mode=$mode gpus=($gpus) llm_id_variant=$llm_id_variant"
echo "[launcher] launcher_dir=$launcher_dir"

for method in "${methods[@]}"; do
  log="$launcher_dir/${method}.log"
  pidfile="$launcher_dir/${method}.pid"

  args=("$method" "$domain")
  if [[ "$method" == "llm_id" ]]; then
    args+=("$llm_id_variant")
  fi

  echo "[launcher] starting $method -> $log"
  setsid "$RECSYS26_ROOT/scripts/with_gpu_lock.sh" --gpus "$gpus" -- "$RECSYS26_ROOT/scripts/repro_dispatch.sh" "${args[@]}" >"$log" 2>&1 &
  echo $! >"$pidfile"
done

echo "[launcher] started ${#methods[@]} jobs"
echo "[launcher] tail: tail -n 80 $launcher_dir/<method>.log"
