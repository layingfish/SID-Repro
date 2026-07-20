set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: repro_letter_lc_rec.sh <domain>" >&2
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
MODEL_ROOT="${MODEL_ROOT:-$RECSYS26_ROOT/models}"
CODE_DIR="${LETTER_LC_REC_CODE_DIR:-$RECSYS26_ROOT/baselines/ref03/LETTER-LC-Rec}"
BASE_MODEL="${BASE_MODEL:-${LETTER_LC_REC_BASE_MODEL:-$MODEL_ROOT/llama-7b}}"

if [[ "${CUDA_VISIBLE_DEVICES:-}" == "" ]]; then
  export CUDA_VISIBLE_DEVICES="${RECSYS26_LETTER_LC_GPUS:-0,1,2,3}"
fi

IFS=',' read -ra _CVD <<<"${CUDA_VISIBLE_DEVICES}"
NGPU="${#_CVD[@]}"

pick_port() {
  local base=20000
  local span=20000
  echo $(( base + ( ( $(date +%s) + $$ + RANDOM ) % span ) ))
}

now_ts() { date -u +%Y%m%d_%H%M%S; }

run_dir="${RECSYS26_RUN_DIR:-$RECSYS26_ROOT/logs/repro_paper/letter_lc_rec/${domain}/$(now_ts)}"
out_dir="$run_dir/out"
pred="$run_dir/pred_topk.jsonl"
metrics="$run_dir/metrics_full.json"
mkdir -p "$out_dir"

log_file="$run_dir/run.log"

(
  set -euo pipefail
  export WANDB_MODE=disabled
  export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export RECSYS26_LCREC_SAVE_STEPS="${RECSYS26_LCREC_SAVE_STEPS:-100}"

  source "$RECSYS26_ROOT/scripts/conda_activate.sh" recsys26_letter
  cd "$CODE_DIR"

  master_port="$(pick_port)"
  torchrun --nproc_per_node="$NGPU" --master_port="$master_port" lora_finetune.py \
    --base_model "$BASE_MODEL" \
    --output_dir "$out_dir" \
    --dataset "SETRec_${domain}" \
    --data_path ../data \
    --per_device_batch_size "${LETTER_LC_REC_BATCH_SIZE:-4}" \
    --gradient_accumulation_steps "${LETTER_LC_REC_GRAD_ACCUM:-8}" \
    --learning_rate "${LETTER_LC_REC_LR:-2e-5}" \
    --epochs "${LETTER_LC_REC_EPOCHS:-1}" \
    --tasks seqrec \
    --train_prompt_sample_num 1 \
    --train_data_sample_num 0 \
    --index_file .index.json \
    --wandb_run_name "letter_lc_rec_${domain}" \
    --temperature 1.0

  ckpt="$(find "$out_dir" -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -V | tail -n 1)"
  if [[ "$ckpt" == "" ]]; then
    echo "[letter_lc_rec] checkpoint not found under $out_dir" >&2
    exit 3
  fi

  python test_only_lc.py \
    --base_model "$BASE_MODEL" \
    --ckpt_path "$ckpt" \
    --dataset "SETRec_${domain}" \
    --data_path ../data \
    --index_file .index.json \
    --export_path "$pred" \
    --gpu_id 0 \
    --test_batch_size "${LETTER_LC_REC_TEST_BATCH_SIZE:-1}" \
    --num_beams 20

  python "$RECSYS26_ROOT/scripts/unified_eval.py" \
    --pred "$pred" \
    --dataset "$domain" \
    --data_dir "$DATA_ROOT/setrec_data" \
    --mode full \
    --top_n 5,10 \
    --warm_only \
    --output "$metrics"
) 2>&1 | tee "$log_file"
