#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: repro_setrec_t5.sh <domain> <profile>" >&2
  echo "  domain: beauty | toys | sports | steam | amazon23_vg | microlens_50k | yelp" >&2
  echo "  profile: smoke | paper" >&2
  exit 2
fi

domain="$1"
profile="$2"

case "$domain" in
  beauty|toys|sports|steam|amazon23_vg|microlens_50k|yelp) ;;
  *) echo "Unknown domain: $domain" >&2; exit 2 ;;
esac

case "$profile" in
  smoke|paper) ;;
  *) echo "Unknown profile: $profile" >&2; exit 2 ;;
esac

RECSYS26_ROOT="${RECSYS26_ROOT:-/data/xqp_data/RecSys26}"
SETREC_CODE_DIR="$RECSYS26_ROOT/third_party_clean/SETRec/code"
DATA_PATH="../data/${domain}/"
CACHE_DIR="$RECSYS26_ROOT/models/hf/transformers"
PATCH_FILE="${RECSYS26_SETREC_PATCH:-$RECSYS26_ROOT/patches/from_scratch_20260228/SETRec.patch}"

now_ts() { date -u +%Y%m%d_%H%M%S; }

pick_port() {
  local base=20000
  local span=20000
  echo $(( base + ( ( $(date +%s) + $$ + RANDOM ) % span ) ))
}

run_dir="$RECSYS26_ROOT/logs/repro_from_scratch/setrec/${domain}/$(now_ts)"
mkdir -p "$run_dir"
ln -sfn "$run_dir" "$RECSYS26_ROOT/logs/repro_from_scratch/setrec/${domain}/latest"

echo "$(date -Is)" >"$run_dir/start_time.txt"

# GPU selection
if [[ "${CUDA_VISIBLE_DEVICES:-}" == "" ]]; then
  if [[ "$profile" == "paper" ]]; then
    export CUDA_VISIBLE_DEVICES="${RECSYS26_SETREC_GPUS:-3,4,5,6}"
  else
    export CUDA_VISIBLE_DEVICES="$($RECSYS26_ROOT/scripts/pick_gpu.sh)"
  fi
fi

# profile hyperparams
if [[ "$profile" == "paper" ]]; then
  nproc=4
  batch_size=512
  micro_batch_size=128
  num_epochs=30
  learning_rate="${RECSYS26_SETREC_LR:-1e-3}"
  val_set_size=2000
  max_val_users=0
else
  nproc=1
  batch_size=64
  micro_batch_size=64
  num_epochs=1
  learning_rate="3e-4"
  val_set_size=200
  max_val_users=200
fi

# NOTE: keep these aligned with paper/dispatch defaults unless explicitly changed.
n_sem="${RECSYS26_SETREC_N_SEM:-4}"
alpha="${RECSYS26_SETREC_ALPHA:-0.7}"
n_cf=1
n_query=$((n_sem + 1))
seed=42

out_dir="$run_dir/out"
mkdir -p "$out_dir"

log_file="$run_dir/run.log"

pred_tmp="$run_dir/pred_topk.jsonl"
beta_tmp="$run_dir/beta.json"
metrics_full="$run_dir/metrics_full.json"

export_root="$RECSYS26_ROOT/pred_exports/setrec"
pred_export="$export_root/${domain}_pred_topk.jsonl"
metrics_export="$export_root/${domain}_pred_topk_metrics_full.json"

audit_json=""
if ls "$RECSYS26_ROOT"/reports/dataset_audit_*.json >/dev/null 2>&1; then
  audit_json="$(ls -t "$RECSYS26_ROOT"/reports/dataset_audit_*.json | head -n 1)"
fi

(
  set -euo pipefail

  source "$RECSYS26_ROOT/scripts/conda_activate.sh" recsys26_setrec
  cd "$SETREC_CODE_DIR"

  if command -v python >/dev/null 2>&1; then
    manifest_py="python"
  else
    manifest_py="python3"
  fi

  master_port="$(pick_port)"

  train_cmd_str=""
  export_cmd_str=""
  eval_cmd_str=""

  on_exit() {
    local code="$?"
    set +e
    set +u
    set +o pipefail

    manifest_args=(
      --out "$run_dir/run_manifest.json"
      --method setrec
      --domain "$domain"
      --profile "$profile"
      --run_dir "$run_dir"
      --repo_dir "$SETREC_CODE_DIR"
      --data_path "$DATA_PATH"
      --train_cmd "$train_cmd_str"
      --export_cmd "$export_cmd_str"
      --eval_cmd "$eval_cmd_str"
      --pred_file "$pred_tmp"
      --metrics_file "$metrics_full"
      --pred_export_file "$pred_export"
      --metrics_export_file "$metrics_export"
      --exit_code "$code"
    )

    if [[ -n "${audit_json:-}" ]]; then
      manifest_args+=(--dataset_audit_json "$audit_json")
    fi
    if [[ -n "${PATCH_FILE:-}" && -f "$PATCH_FILE" ]]; then
      manifest_args+=(--patch_file "$PATCH_FILE")
    fi

    "$manifest_py" "$RECSYS26_ROOT/scripts/write_run_manifest.py" "${manifest_args[@]}" || true
  }
  trap on_exit EXIT

  echo "[from_scratch][setrec] start $(date -Is)"
  echo "[from_scratch][setrec] domain=$domain profile=$profile CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "[from_scratch][setrec] data_path=$DATA_PATH"
  echo "[from_scratch][setrec] out_dir=$out_dir"
  echo "[from_scratch][setrec] torchrun nproc=$nproc master_port=$master_port"

  train_cmd=(
    torchrun --nproc_per_node="$nproc" --master_port="$master_port" finetune_t5.py
    --base_model t5-small
    --data_path "$DATA_PATH"
    --output_dir "$out_dir"
    --cache_dir "$CACHE_DIR"
    --sem_encoder t5
    --n_query "$n_query" --n_sem "$n_sem" --n_cf "$n_cf" --alpha "$alpha"
    --layers 512 256 128
    --batch_size "$batch_size" --micro_batch_size "$micro_batch_size"
    --num_epochs "$num_epochs"
    --learning_rate "$learning_rate"
    --cutoff_len 512
    --val_set_size "$val_set_size"
    --warmup_steps 100
    --lr_scheduler cosine
    --seed "$seed"
    --wandb_project ""
  )
  train_cmd_str="$(printf "%q " "${train_cmd[@]}")"
  "${train_cmd[@]}"


  # For export, force single GPU to avoid multi-GPU device_map sharding issues.
  export_cuda="${CUDA_VISIBLE_DEVICES%%,*}"
  if [[ "$export_cuda" != "$CUDA_VISIBLE_DEVICES" ]]; then
    echo "[from_scratch][setrec] narrowing CUDA_VISIBLE_DEVICES for export: $CUDA_VISIBLE_DEVICES -> $export_cuda"
    export CUDA_VISIBLE_DEVICES="$export_cuda"
  fi

  export_cmd=(
    python "$RECSYS26_ROOT/scripts/exports/export_setrec_t5_topk.py"
    --data_path "$DATA_PATH"
    --ckpt_dir "$out_dir"
    --cache_dir "$CACHE_DIR"
    --n_sem "$n_sem"
    --alpha "$alpha"
    --k 20
    --max_val_users "$max_val_users"
    --pred_out "$pred_tmp"
    --beta_out "$beta_tmp"
  )
  export_cmd_str="$(printf "%q " "${export_cmd[@]}")"
  "${export_cmd[@]}"

  eval_cmd=(
    python "$RECSYS26_ROOT/scripts/unified_eval.py"
    --pred "$pred_tmp"
    --dataset "$domain"
    --mode full
    --top_n 5,10
    --output "$metrics_full"
  )  # warm_only for main table datasets (yelp/amazon23_vg/microlens_50k)
  case "$domain" in
    microlens_50k|amazon23_vg|yelp)
      eval_cmd+=(--warm_only)
      ;;
  esac

  eval_cmd_str="$(printf "%q " "${eval_cmd[@]}")"
  "${eval_cmd[@]}"

  mkdir -p "$export_root"
  cp -f "$pred_tmp" "$pred_export"
  cp -f "$metrics_full" "$metrics_export"

  echo "[from_scratch][setrec] DONE $(date -Is)"
) 2>&1 | tee "$log_file"
