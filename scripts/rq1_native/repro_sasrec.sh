set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: repro_sasrec.sh <domain>" >&2
  exit 2
fi

domain="$1"

case "$domain" in
  beauty|toys|sports|steam|amazon23_vg|microlens_50k|yelp) ;;
  *) echo "Unknown domain: $domain" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECSYS26_ROOT="${RECSYS26_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$RECSYS26_ROOT/data}"
SEATER_CODE_DIR="${SEATER_CODE_DIR:-$RECSYS26_ROOT/baselines/ref01}"
SEATER_SETREC_DATA_ROOT="${SEATER_SETREC_DATA_ROOT:-$DATA_ROOT/seater_setrec}"
export SEATER_SETREC_DATA_ROOT

if [[ "${CUDA_VISIBLE_DEVICES:-}" == "" ]]; then
  export CUDA_VISIBLE_DEVICES="$("$RECSYS26_ROOT/scripts/pick_gpu.sh")"
fi

now_ts() { date -u +%Y%m%d_%H%M%S; }

run_dir="${RECSYS26_RUN_DIR:-$RECSYS26_ROOT/logs/repro_from_scratch/sasrec/${domain}/$(now_ts)}"
workspace_root="$run_dir/workspace"
mkdir -p "$workspace_root"
ln -sfn "$run_dir" "$RECSYS26_ROOT/logs/repro_from_scratch/sasrec/${domain}/latest"

log_file="$run_dir/run.log"
metrics_full="$run_dir/metrics_full.json"

(
  set -euo pipefail
  source "$RECSYS26_ROOT/scripts/conda_activate.sh" recsys26_seater
  cd "$SEATER_CODE_DIR"

  name="SASREC_SETRec_${domain}"
  python main.py \
    --name "$name" \
    --description "paper" \
    --workspace "$workspace_root" \
    --dataset_name "SETRec_${domain}" \
    --model SASREC \
    --gpu_id 0 \
    --epochs 50 \
    --batch_size 256 \
    --test_batch_size 1024 \
    --num_workers 4 \
    --random_seed 2023 \
    --no-tb \
    --no-train_tb

  latest_ws="$(find "$workspace_root" -maxdepth 1 -type d -name "${name}*" -print | sort | tail -n 1)"
  pred="$latest_ws/pred_topk.jsonl"
  if [[ ! -f "$pred" ]]; then
    echo "[sasrec] expected prediction file not found: $pred" >&2
    exit 3
  fi

  python "$RECSYS26_ROOT/scripts/unified_eval.py" \
    --pred "$pred" \
    --dataset "$domain" \
    --data_dir "$DATA_ROOT/setrec_data" \
    --mode full \
    --top_n 5,10 \
    --warm_only \
    --output "$metrics_full"
) 2>&1 | tee "$log_file"
