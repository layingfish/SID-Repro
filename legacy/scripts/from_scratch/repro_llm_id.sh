#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "Usage: repro_llm_id.sh <domain> <profile> <variant>" >&2
  echo "  domain: beauty | toys | sports | steam | amazon23_vg | microlens_50k | yelp" >&2
  echo "  profile: smoke | paper" >&2
  echo "  variant: iid | sid | semid | cid | hid" >&2
  exit 2
fi

domain="$1"
profile="$2"
variant="$3"

case "$domain" in
  beauty|toys|sports|steam|amazon23_vg|microlens_50k|microlens_50k_semid_adapt|yelp) ;;
  *) echo "Unknown domain: $domain" >&2; exit 2 ;;
esac
case "$profile" in
  smoke|paper) ;;
  *) echo "Unknown profile: $profile" >&2; exit 2 ;;
esac
case "$variant" in
  iid|sid|semid|cid|hid) ;;
  *) echo "Unknown variant: $variant" >&2; exit 2 ;;
esac

RECSYS26_ROOT="${RECSYS26_ROOT:-/data/xqp_data/RecSys26}"
CODE_DIR="$RECSYS26_ROOT/third_party/LLM_RecSys_ID"

now_ts() { date -u +%Y%m%d_%H%M%S; }
pick_port() {
  local base=20000
  local span=20000
  echo $(( base + ( ( $(date +%s) + $$ + RANDOM ) % span ) ))
}

run_root="$RECSYS26_ROOT/logs/repro_paper/llm_id/$variant/$domain"
run_dir="$run_root/$(now_ts)"
mkdir -p "$run_dir"
ln -sfn "$run_dir" "$run_root/latest"
echo "$(date -Is)" > "$run_dir/start_time.txt"
log_file="$run_dir/run.log"

pred_tmp="$run_dir/pred_topk.jsonl"
metrics_full="$run_dir/metrics_full.json"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if [[ "$profile" == "paper" ]]; then
    export CUDA_VISIBLE_DEVICES="${RECSYS26_LLM_ID_GPUS:-2,5,6}"
  else
    export CUDA_VISIBLE_DEVICES="${RECSYS26_LLM_ID_GPUS:-5}"
  fi
fi

if [[ "$profile" == "paper" ]]; then
  epochs="${RECSYS26_LLM_ID_EPOCHS:-20}"
  logging_step="${RECSYS26_LLM_ID_LOGGING_STEP:-1000}"
  train_seq_batch="${RECSYS26_LLM_ID_TRAIN_SEQ_BATCH:-64}"
  train_yesno_batch="${RECSYS26_LLM_ID_TRAIN_YESNO_BATCH:-16}"
  train_direct_yesno_batch="${RECSYS26_LLM_ID_TRAIN_DIRECT_YESNO_BATCH:-16}"
  train_direct_candidate_batch="${RECSYS26_LLM_ID_TRAIN_DIRECT_CAND_BATCH:-8}"
  train_direct_straight_batch="${RECSYS26_LLM_ID_TRAIN_DIRECT_STRAIGHT_BATCH:-16}"
  export_batch="${RECSYS26_LLM_ID_EXPORT_BATCH:-24}"
else
  epochs=1
  logging_step=50
  train_seq_batch=16
  train_yesno_batch=4
  train_direct_yesno_batch=4
  train_direct_candidate_batch=2
  train_direct_straight_batch=4
  export_batch=8
fi

task="setrec_${domain}"
item_representation="content_based"
data_order="random"
remapped_data_order="original"
whole_word_embedding="shijie"
extra_train_args=()
extra_export_args=()

case "$variant" in
  iid)
    item_representation="no_tokenization"
    data_order="random"
    ;;
  sid)
    item_representation="None"
    data_order="remapped_sequential"
    extra_train_args+=(--remapped_data_order original)
    extra_export_args+=(--remapped_data_order original)
    ;;
  semid)
    item_representation="content_based"
    data_order="random"
    ;;
  cid)
    item_representation="CF"
    data_order="remapped_sequential"
    extra_train_args+=(--remapped_data_order original --cluster_size 500 --cluster_number 20)
    extra_export_args+=(--remapped_data_order original --cluster_size 500 --cluster_number 20)
    ;;
  hid)
    item_representation="CF"
    data_order="remapped_sequential"
    extra_train_args+=(--remapped_data_order original --cluster_size 500 --cluster_number 20 --last_token_no_repetition)
    extra_export_args+=(--remapped_data_order original --cluster_size 500 --cluster_number 20 --last_token_no_repetition)
    ;;
esac

(
  set -euo pipefail
  source "$RECSYS26_ROOT/scripts/conda_activate.sh" recsys26_llm_id
  cd "$CODE_DIR"

  export MASTER_ADDR=127.0.0.1
  export MASTER_PORT="$(pick_port)"

  echo "[repro_llm_id] start $(date -Is)"
  echo "[repro_llm_id] domain=$domain profile=$profile variant=$variant CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "[repro_llm_id] run_dir=$run_dir"

  train_cmd=(
    python main.py
    --distributed --multiGPU
    --task "$task"
    --data_dir data/
    --seed 42
    --warmup_prop 0.05
    --lr 0.001
    --clip 1.0
    --model_type t5-small
    --epochs "$epochs"
    --gpu 0,1
    --logging_step "$logging_step"
    --eval_interval "${RECSYS26_LLM_ID_EVAL_INTERVAL:-1}"
    --eval_batch_size "${RECSYS26_LLM_ID_EVAL_BATCH_SIZE:-24}"
    --logging_dir "$run_dir/train.log"
    --model_dir "$run_dir/model.pt"
    --train_sequential_item_batch "$train_seq_batch"
    --train_sequential_yesno_batch "$train_yesno_batch"
    --train_direct_yesno_batch "$train_direct_yesno_batch"
    --train_direct_candidate_batch "$train_direct_candidate_batch"
    --train_direct_straightforward_batch "$train_direct_straight_batch"
    --item_representation "$item_representation"
    --data_order "$data_order"
    --whole_word_embedding "$whole_word_embedding"
    "${extra_train_args[@]}"
  )
  printf '[repro_llm_id] train_cmd=%q ' "${train_cmd[@]}"; printf '\n'
  "${train_cmd[@]}"

  best_ckpt="$run_dir/best_model.pt"
  if [[ ! -f "$best_ckpt" ]]; then
    echo "[repro_llm_id] ERROR: best checkpoint not found: $best_ckpt" >&2
    exit 1
  fi

  export_cmd=(
    python test_only_llm_id.py
    --ckpt_path "$best_ckpt"
    --task "$task"
    --data_dir data/
    --export_path "$pred_tmp"
    --gpu_id 0
    --batch_size "$export_batch"
    --item_representation "$item_representation"
    --data_order "$data_order"
    --whole_word_embedding "$whole_word_embedding"
    --num_beams 20
    --topk 20
    "${extra_export_args[@]}"
  )
  printf '[repro_llm_id] export_cmd=%q ' "${export_cmd[@]}"; printf '\n'
  "${export_cmd[@]}"

  eval_cmd=(
    python "$RECSYS26_ROOT/scripts/unified_eval.py"
    --pred "$pred_tmp"
    --dataset "$domain"
    --mode full
    --top_n 5,10
    --output "$metrics_full"
  )
  case "$domain" in
    microlens_50k|microlens_50k_semid_adapt|amazon23_vg|yelp)
      eval_cmd+=(--warm_only)
      ;;
  esac
  printf '[repro_llm_id] eval_cmd=%q ' "${eval_cmd[@]}"; printf '\n'
  "${eval_cmd[@]}"

  echo "0" > "$run_dir/exit_code.txt"
  echo "$(date -Is)" > "$run_dir/end_time.txt"
  echo "[repro_llm_id] DONE $(date -Is)"
) 2>&1 | tee "$log_file"
