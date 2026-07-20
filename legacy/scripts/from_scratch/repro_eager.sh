#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: repro_eager.sh <domain> <profile>" >&2
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
EAGER_CODE_DIR="$RECSYS26_ROOT/third_party_clean/EAGER/EAGER"
EAGER_REPO_DIR="$RECSYS26_ROOT/third_party_clean/EAGER"
SETREC_DATA_ROOT="$RECSYS26_ROOT/datasets/setrec_data"
PATCH_FILE="${RECSYS26_EAGER_PATCH:-$RECSYS26_ROOT/patches/from_scratch_20260228/EAGER.patch}"

now_ts() { date -u +%Y%m%d_%H%M%S; }

run_dir="$RECSYS26_ROOT/logs/repro_from_scratch/eager/${domain}/$(now_ts)"
mkdir -p "$run_dir"
ln -sfn "$run_dir" "$RECSYS26_ROOT/logs/repro_from_scratch/eager/${domain}/latest"

echo "$(date -Is)" >"$run_dir/start_time.txt"

# GPU selection (callers can also set CUDA_VISIBLE_DEVICES explicitly)
if [[ "${CUDA_VISIBLE_DEVICES:-}" == "" ]]; then
  export CUDA_VISIBLE_DEVICES="$($RECSYS26_ROOT/scripts/pick_gpu.sh)"
fi

# profile hyperparams
if [[ "$profile" == "paper" ]]; then
  total_batch_num=60000
  eval_step=3000
  train_batch_size=256
  k=128
else
  total_batch_num=600
  eval_step=200
  train_batch_size=64
  k=128
fi

seed=42

log_file="$run_dir/run.log"
pred_tmp="$run_dir/pred_topk.jsonl"
metrics_full="$run_dir/metrics_full.json"

export_root="$RECSYS26_ROOT/pred_exports/eager"
pred_export="$export_root/${domain}_pred_topk.jsonl"
metrics_export="$export_root/${domain}_pred_topk_metrics_full.json"

# dataset audit json (best-effort)
audit_json=""
if ls "$RECSYS26_ROOT"/reports/dataset_audit_*.json >/dev/null 2>&1; then
  audit_json="$(ls -t "$RECSYS26_ROOT"/reports/dataset_audit_*.json | head -n 1)"
fi

(
  set -euo pipefail

  source "$RECSYS26_ROOT/scripts/conda_activate.sh" recsys26_eager
  cd "$EAGER_CODE_DIR"

  manifest_py=python

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
      --method eager
      --domain "$domain"
      --profile "$profile"
      --run_dir "$run_dir"
      --repo_dir "$EAGER_REPO_DIR"
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

  echo "[from_scratch][eager] start $(date -Is)"
  echo "[from_scratch][eager] domain=$domain profile=$profile CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "[from_scratch][eager] run_dir=$run_dir"

  behavior_emb="sasrec"
  din_model_path=""
  din_cmd_str=""

  if [[ "$profile" == "paper" ]]; then
    behavior_emb="din"
    din_step="${RECSYS26_EAGER_DIN_STEP:-29000}"
    din_model_path="$SETREC_DATA_ROOT/$domain/DIN_MODEL_${din_step}.pt"
    if [[ "${RECSYS26_EAGER_FORCE_DIN:-0}" == "1" || ! -f "$din_model_path" ]]; then
      din_cmd=(
        python train_din_setrec.py
        --domain "$domain"
        --data_root "$SETREC_DATA_ROOT"
        --output_dir "$run_dir"
        --seed "$seed"
        --seq_len 20
        --min_seq_len 5
        --train_sample_seg_cnt 10
        --parall 8
        --train_batch_size 128
        --total_batch_num "$din_step"
        --save_dir "$SETREC_DATA_ROOT/$domain"
        --save_steps "$din_step"
        --promote_step "$din_step"
        --log_every 100
      )
      if [[ "${RECSYS26_EAGER_FORCE_DIN:-0}" == "1" ]]; then
        din_cmd+=(--force_train)
      fi
      din_cmd_str="$(printf "%q " "${din_cmd[@]}")"
      echo "[from_scratch][eager] pretrain DIN: $din_cmd_str"
      "${din_cmd[@]}"
    else
      echo "[from_scratch][eager] DIN checkpoint exists: $din_model_path (skip)"
    fi
  fi

  train_cmd=(
    python train_rec_setrec.py
    --domain "$domain"
    --data_root "$SETREC_DATA_ROOT"
    --output_dir "$run_dir"
    --seed "$seed"
    --behavior_emb "$behavior_emb"
    --din_model_path "$din_model_path"
    --seq_len 20
    --min_seq_len 5
    --train_sample_seg_cnt 10
    --parall 8
    --tree_num 2
    --k "$k"
    --d_model 96
    --enc_num_layers 1
    --dec_num_layers 2,2
    --n_head 4
    --init_way embkm,embkm
    --feature_ratio 1.0
    --max_iters 100
    --train_batch_size "$train_batch_size"
    --total_batch_num "$total_batch_num"
    --eval_step "$eval_step"
    --eval_topk 10
    --eval_rerank_topk 10
    --export_topk 20
    --export_rerank_topk 20
    --pred_out "$pred_tmp"
    --ckpt_dir "$run_dir/ckpt"
    --lr 1e-3
    --weight_decay 1e-7
    --warmup_updates 2000
    --warmup_init_lr 1e-7
  )
  eager_cmd_str="$(printf "%q " "${train_cmd[@]}")"
  if [[ -n "${din_cmd_str:-}" ]]; then
    train_cmd_str="${din_cmd_str} && ${eager_cmd_str}"
  else
    train_cmd_str="${eager_cmd_str}"
  fi
  "${train_cmd[@]}"

  if [[ ! -f "$pred_tmp" ]]; then
    echo "[from_scratch][eager] ERROR: pred_topk.jsonl not found: $pred_tmp" >&2
    exit 3
  fi

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

  echo "[from_scratch][eager] DONE $(date -Is)"
) 2>&1 | tee "$log_file"
