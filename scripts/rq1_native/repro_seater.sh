#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: repro_seater.sh <domain> <profile>" >&2
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECSYS26_ROOT="${RECSYS26_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$RECSYS26_ROOT/data}"
SEATER_CODE_DIR="${SEATER_CODE_DIR:-$RECSYS26_ROOT/baselines/ref01}"
DATA_PATH="${SEATER_DATA_PATH:-$DATA_ROOT/seater_setrec/${domain}}"
DATASET_NAME="SETRec_${domain}"
PATCH_FILE="${RECSYS26_SEATER_PATCH:-$RECSYS26_ROOT/patches/from_scratch_20260228/SEATER.patch}"

now_ts() { date -u +%Y%m%d_%H%M%S; }

run_dir="${RECSYS26_RUN_DIR:-$RECSYS26_ROOT/logs/repro_from_scratch/seater/${domain}/$(now_ts)}"
mkdir -p "$run_dir"
ln -sfn "$run_dir" "$RECSYS26_ROOT/logs/repro_from_scratch/seater/${domain}/latest"

echo "$(date -Is)" >"$run_dir/start_time.txt"

# GPU selection
if [[ "${CUDA_VISIBLE_DEVICES:-}" == "" ]]; then
  export CUDA_VISIBLE_DEVICES="$($RECSYS26_ROOT/scripts/pick_gpu.sh)"
fi

# profile hyperparams
if [[ "$profile" == "paper" ]]; then
  epochs=100
  batch_size=256
  test_batch_size=1024
  num_workers=4
else
  epochs=1
  batch_size=64
  test_batch_size=128
  num_workers=0
fi

vocab=8
seed=42

workspace_root="$run_dir/workspace"
mkdir -p "$workspace_root"

name="SEATER_SETRec_${domain}_${profile}"

log_file="$run_dir/run.log"

pred_tmp="$run_dir/pred_topk.jsonl"
metrics_full="$run_dir/metrics_full.json"

export_root="$RECSYS26_ROOT/pred_exports/seater"
pred_export="$export_root/${domain}_pred_topk.jsonl"
metrics_export="$export_root/${domain}_pred_topk_metrics_full.json"

audit_json=""
if ls "$RECSYS26_ROOT"/reports/dataset_audit_*.json >/dev/null 2>&1; then
  audit_json="$(ls -t "$RECSYS26_ROOT"/reports/dataset_audit_*.json | head -n 1)"
fi

(
  set -euo pipefail

  source "$RECSYS26_ROOT/scripts/conda_activate.sh" recsys26_seater
  cd "$SEATER_CODE_DIR"

  if command -v python >/dev/null 2>&1; then
    manifest_py="python"
  else
    manifest_py="python3"
  fi

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
      --method seater
      --domain "$domain"
      --profile "$profile"
      --run_dir "$run_dir"
      --repo_dir "$SEATER_CODE_DIR"
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

  echo "[from_scratch][seater] start $(date -Is)"
  echo "[from_scratch][seater] domain=$domain profile=$profile CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "[from_scratch][seater] dataset_name=$DATASET_NAME data_path=$DATA_PATH"
  echo "[from_scratch][seater] workspace_root=$workspace_root"

  train_cmd=(
    python main.py
    --name "$name"
    --description "$profile"
    --workspace "$workspace_root"
    --dataset_name "$DATASET_NAME"
    --model SEATER
    --vocab "$vocab"
    --gpu_id 0
    --epochs "$epochs"
    --batch_size "$batch_size"
    --test_batch_size "$test_batch_size"
    --num_workers "$num_workers"
    --random_seed "$seed"
    --no-tb
    --no-train_tb
  )
  train_cmd_str="$(printf "%q " "${train_cmd[@]}")"
  "${train_cmd[@]}"

  ws_subdir=""
  if ls "$workspace_root"/* >/dev/null 2>&1; then
    ws_subdir="$(ls -td "$workspace_root"/* | head -n 1)"
  fi

  ckpt_path="$ws_subdir/ckpt/best.pth"
  if [[ ! -f "$ckpt_path" ]]; then
    echo "[from_scratch][seater] ERROR: ckpt not found: $ckpt_path" >&2
    exit 3
  fi

  export_cmd=(
    python "$RECSYS26_ROOT/scripts/exports/export_seater_topk.py"
    --seater_code_dir "$SEATER_CODE_DIR"
    --dataset_name "$DATASET_NAME"
    --ckpt "$ckpt_path"
    --vocab "$vocab"
    --gpu_id 0
    --k 20
    --batch_size "$test_batch_size"
    --num_workers "$num_workers"
    --seed "$seed"
    --pred_out "$pred_tmp"
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
    microlens_50k|microlens_100k|amazon23_vg|yelp)
      eval_cmd+=(--warm_only)
      ;;
  esac

  eval_cmd_str="$(printf "%q " "${eval_cmd[@]}")"
  "${eval_cmd[@]}"

  mkdir -p "$export_root"
  cp -f "$pred_tmp" "$pred_export"
  cp -f "$metrics_full" "$metrics_export"

  echo "[from_scratch][seater] DONE $(date -Is)"
) 2>&1 | tee "$log_file"
