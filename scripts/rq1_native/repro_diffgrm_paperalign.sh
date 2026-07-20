#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: repro_diffgrm_paperalign.sh <domain>" >&2
  echo "  domain: beauty | toys | sports | steam | amazon23_vg | microlens_50k | yelp" >&2
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
DIFFGRM_CODE_DIR="${DIFFGRM_CODE_DIR:-$RECSYS26_ROOT/baselines/ref06}"
SETREC_DATA_ROOT="${SETREC_DATA_ROOT:-$DATA_ROOT/setrec_data}"
CACHE_DIR="${DIFFGRM_CACHE_DIR:-$DATA_ROOT/diffgrm_cache_from_scratch/${domain}_k5}"
PATCH_FILE="${RECSYS26_DIFFGRM_PATCH:-$RECSYS26_ROOT/patches/from_scratch_20260228/DiffGRM.patch}"

now_ts() { date -u +%Y%m%d_%H%M%S; }

run_dir="${RECSYS26_RUN_DIR:-$RECSYS26_ROOT/logs/repro_from_scratch/diffgrm/${domain}/$(now_ts)_paperalign}"
mkdir -p "$run_dir"
ln -sfn "$run_dir" "$RECSYS26_ROOT/logs/repro_from_scratch/diffgrm/${domain}/latest_paperalign"

echo "$(date -Is)" >"$run_dir/start_time.txt"

# GPU selection
if [[ "${CUDA_VISIBLE_DEVICES:-}" == "" ]]; then
  export CUDA_VISIBLE_DEVICES="$($RECSYS26_ROOT/scripts/pick_gpu.sh)"
fi

# Paper-aligned-ish hyperparams (within fixed dataset/backbone constraints)
# Note: paper says max_epochs=100 + early-stopping(patience=15)
epochs=100
train_batch_size=256
# eval batch size for speed
eval_batch_size=32
sent_emb_batch_size=256
seed=42

workspace_root="$run_dir/workspace"
mkdir -p "$workspace_root"
log_dir="$workspace_root/logs"
tb_dir="$workspace_root/tensorboard"
ckpt_dir="$workspace_root/ckpt"
mkdir -p "$log_dir" "$tb_dir" "$ckpt_dir"

log_file="$run_dir/run.log"

pred_tmp="$run_dir/pred_topk.jsonl"
metrics_full="$run_dir/metrics_full.json"

export_root="$RECSYS26_ROOT/pred_exports/diffgrm"
pred_export="$export_root/${domain}_pred_topk_paperalign.jsonl"
metrics_export="$export_root/${domain}_pred_topk_paperalign_metrics_full.json"

audit_json=""
if ls "$RECSYS26_ROOT"/reports/dataset_audit_*.json >/dev/null 2>&1; then
  audit_json="$(ls -t "$RECSYS26_ROOT"/reports/dataset_audit_*.json | head -n 1)"
fi

(
  set -euo pipefail

  source "$RECSYS26_ROOT/scripts/conda_activate.sh" recsys26_diffgrm
  cd "$DIFFGRM_CODE_DIR"

  manifest_py="python"

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
      --method diffgrm
      --domain "$domain"
      --profile "paperalign"
      --run_dir "$run_dir"
      --repo_dir "$DIFFGRM_CODE_DIR"
      --data_path "$SETREC_DATA_ROOT/$domain"
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

  echo "[from_scratch][diffgrm] start $(date -Is)"
  echo "[from_scratch][diffgrm] domain=$domain profile=paperalign CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "[from_scratch][diffgrm] cache_dir=$CACHE_DIR"
  echo "[from_scratch][diffgrm] workspace_root=$workspace_root"

  # NOTE: do NOT change dataset/backbone here (keep setrec_t5_tdcb); align the rest to paper/README.
  train_cmd=(
    python main.py
    --model=DIFF_GRM
    --dataset=SETRec
    --category=$domain
    --epochs=$epochs
    --train_batch_size=$train_batch_size
    --eval_batch_size=$eval_batch_size
    --cache_dir=$CACHE_DIR
    --log_dir=$log_dir
    --tensorboard_log_dir=$tb_dir
    --ckpt_dir=$ckpt_dir
    --run_id=diffgrm_setrec_${domain}_paperalign
    --num_proc=1
    --rand_seed=$seed

    --masking_strategy=guided
    --guided_refresh_each_step=false
    --guided_select=least
    --guided_conf_metric=msp

    --train_sliding=true
    --min_hist_len=2

    --encoder_n_layer=1
    --decoder_n_layer=4
    --n_head=4
    --n_embd=256
    --n_inner=1024

    --lr=0.003
    --warmup_steps=10000
    --dropout=0.1
    --label_smoothing=0.1
    --eval_start_epoch=20
    --eval_interval=1
    --patience=15

    --sent_emb_model=setrec_t5_tdcb
    --sent_emb_dim=768
    --sent_emb_pca=256
    --sent_emb_batch_size=$sent_emb_batch_size
    --normalize_after_pca=true
    --force_regenerate_opq=true
  )

  train_cmd_str="$(printf "%q " "${train_cmd[@]}")"
  "${train_cmd[@]}"

  ckpt_path="$ckpt_dir/best.bin"
  if [[ ! -f "$ckpt_path" ]]; then
    echo "[from_scratch][diffgrm] ERROR: ckpt not found: $ckpt_path" >&2
    exit 3
  fi

  export_cmd=(
    python "$RECSYS26_ROOT/scripts/exports/export_diffgrm_topk.py"
    --decode beam
    --diffgrm_code_dir "$DIFFGRM_CODE_DIR"
    --checkpoint "$ckpt_path"
    --domain "$domain"
    --cache_dir "$CACHE_DIR"
    --setrec_data_root "$SETREC_DATA_ROOT"
    --eval_batch_size "$eval_batch_size"
    --k 20
    --n_return_sequences 200
    --seed "$seed"
    --pred_out "$pred_tmp"
    --log_dir "$log_dir"
    --tensorboard_log_dir "$tb_dir"

    --sent_emb_model setrec_t5_tdcb
    --sent_emb_dim 768
    --sent_emb_pca 256
    --normalize_after_pca true

    --encoder_n_layer 1
    --decoder_n_layer 4
    --n_head 4
    --n_embd 256
    --n_inner 1024
    --dropout 0.1

    --masking_strategy guided
    --guided_refresh_each_step false
    --guided_select least
    --guided_conf_metric msp

    --train_sliding true
    --min_hist_len 2
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

  echo "[from_scratch][diffgrm] DONE $(date -Is)"
) 2>&1 | tee "$log_file"
