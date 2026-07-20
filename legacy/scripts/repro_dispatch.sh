#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: repro_dispatch.sh <method> <domain> [variant]" >&2
  echo "  methods: setrec | llm_id | eager | letter | etegrec | seater | tiger | rpg | diffgrm" >&2
  echo "  domains: beauty | toys | sports | steam | amazon23_vg | microlens_50k | yelp" >&2
  echo "  llm_id variants: iid | sid | semid | cid | hid (default: semid)" >&2
  exit 2
fi

method="$1"
domain="$2"
variant="${3:-}" # only used by llm_id

case "$domain" in
  beauty|toys|sports|steam|amazon23_vg|microlens_50k|yelp) ;;
  *) echo "Unknown domain: $domain" >&2; exit 2 ;;
esac

RECSYS26_ROOT="${RECSYS26_ROOT:-/data/xqp_data/RecSys26}"
MODE="${RECSYS26_MODE:-full}" # quick|full

pick_port() {
  local base=20000
  local span=20000
  echo $(( base + ( ( $(date +%s) + $$ + RANDOM ) % span ) ))
}

now_ts() {
  date +%Y%m%d_%H%M%S
}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$("$RECSYS26_ROOT/scripts/pick_gpu.sh")}"

run_root="$RECSYS26_ROOT/logs/repro/$method"
if [[ "$method" == "llm_id" ]]; then
  variant="${variant:-semid}"
  run_root="$RECSYS26_ROOT/logs/repro/$method/$variant"
fi

run_dir="$run_root/$domain/$(now_ts)"
mkdir -p "$run_dir"
ln -sfn "$run_dir" "$run_root/$domain/latest"

echo "$(date -Is)" >"$run_dir/start_time.txt"

log_file="$run_dir/run.log"

# mode defaults
epochs_quick=1

setrec_epochs_full=20
llm_id_epochs_full=20
letter_epochs_full=50
etegrec_epochs_full=50
seater_epochs_full=50

eager_total_batch_num_full=6000
rpg_epochs_full=50
diffgrm_epochs_full=50
tiger_iterations_full=20000

echo "[repro_dispatch] method=$method domain=$domain variant=${variant:-n/a} mode=$MODE gpu=$CUDA_VISIBLE_DEVICES"
echo "[repro_dispatch] run_dir=$run_dir"

set +e
(
  set -euo pipefail
  echo "[run] start $(date -Is)"
  echo "[run] method=$method domain=$domain variant=${variant:-n/a} mode=$MODE CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

  case "$method" in
    setrec)
      source "$RECSYS26_ROOT/scripts/conda_activate.sh" recsys26_setrec
      cd "$RECSYS26_ROOT/third_party/SETRec/code"

      epochs="$epochs_quick"
      batch_size=32
      micro_batch_size=4
      lr=0.0003
      if [[ "$MODE" == "full" ]]; then
        epochs="$setrec_epochs_full"
        batch_size=64
        micro_batch_size=8
      fi

      master_port="$(pick_port)"
      mkdir -p "$run_dir/out"
      torchrun --nproc_per_node=1 --master_port="$master_port" finetune_t5.py \
        --base_model t5-small \
        --data_path "../data/$domain/" \
        --output_dir "$run_dir/out" \
        --cache_dir "$RECSYS26_ROOT/models/hf/transformers" \
        --n_query 2 --n_sem 1 --n_cf 1 --alpha 0.7 \
        --layers 128 64 \
        --batch_size "$batch_size" --micro_batch_size "$micro_batch_size" \
        --num_epochs "$epochs" \
        --learning_rate "$lr" \
        --cutoff_len 64 \
        --val_set_size 2000 \
        --warmup_steps 0 \
        --lr_scheduler cosine \
        --seed 42 \
        --wandb_project ""
      ;;

    llm_id)
      source "$RECSYS26_ROOT/scripts/conda_activate.sh" recsys26_llm_id
      cd "$RECSYS26_ROOT/third_party/LLM_RecSys_ID"

      export MASTER_ADDR=127.0.0.1
      export MASTER_PORT="$(pick_port)"

      epochs="$epochs_quick"
      lr=0.001
      logging_step=100
      if [[ "$MODE" == "full" ]]; then
        epochs="$llm_id_epochs_full"
        logging_step=500
      fi

      task="setrec_${domain}"
      item_representation="content_based"
      data_order="random"
      extra_args=()

      case "$variant" in
        iid)
          item_representation="no_tokenization"
          data_order="random"
          ;;
        sid)
          item_representation="None"
          data_order="remapped_sequential"
          extra_args+=(--remapped_data_order original)
          ;;
        semid)
          item_representation="content_based"
          data_order="random"
          ;;
        cid)
          item_representation="CF"
          data_order="remapped_sequential"
          extra_args+=(--remapped_data_order original --cluster_size 500 --cluster_number 20)
          ;;
        hid)
          item_representation="CF"
          data_order="remapped_sequential"
          extra_args+=(--remapped_data_order original --cluster_size 500 --cluster_number 20 --last_token_no_repetition)
          ;;
        *)
          echo "Unknown llm_id variant: $variant" >&2
          exit 2
          ;;
      esac

      python main.py \
        --distributed --multiGPU \
        --task "$task" \
        --data_dir data/ \
        --seed 42 \
        --warmup_prop 0.05 \
        --lr "$lr" \
        --clip 1.0 \
        --model_type t5-small \
        --epochs="$epochs" \
        --logging_step "$logging_step" \
        --logging_dir "$run_dir/train.log" \
        --model_dir "$run_dir/model.pt" \
        --train_sequential_item_batch 32 \
        --train_sequential_yesno_batch 16 \
        --train_direct_yesno_batch 16 \
        --train_direct_candidate_batch 8 \
        --train_direct_straightforward_batch 16 \
        --item_representation "$item_representation" \
        --data_order "$data_order" \
        "${extra_args[@]}"
      ;;

    eager)
      source "$RECSYS26_ROOT/scripts/conda_activate.sh" recsys26_eager
      cd "$RECSYS26_ROOT/third_party/EAGER/EAGER"

      total_batch_num=600
      eval_step=300
      if [[ "$MODE" == "full" ]]; then
        total_batch_num="$eager_total_batch_num_full"
        eval_step=300
      fi

      python train_rec_setrec.py \
        --domain "$domain" \
        --data_root "$RECSYS26_ROOT/datasets/setrec_data" \
        --output_dir "$RECSYS26_ROOT/logs/eager_setrec" \
        --seed 2024 \
        --seq_len 20 --min_seq_len 5 \
        --train_sample_seg_cnt 10 \
        --parall 8 \
        --tree_num 2 \
        --k 128 \
        --d_model 96 \
        --enc_num_layers 1 \
        --dec_num_layers 2,2 \
        --n_head 4 \
        --init_way embkm,embkm \
        --feature_ratio 1.0 \
        --max_iters 50 \
        --train_batch_size 256 \
        --total_batch_num "$total_batch_num" \
        --eval_step "$eval_step" \
        --topk 5 \
        --rerank_topk 5
      ;;

    letter)
      source "$RECSYS26_ROOT/scripts/conda_activate.sh" recsys26_letter
      cd "$RECSYS26_ROOT/third_party/LETTER/LETTER-TIGER"

      epochs="$epochs_quick"
      if [[ "$MODE" == "full" ]]; then
        epochs="$letter_epochs_full"
      fi

      dataset_name="SETRec_${domain}"
      out_dir="$run_dir/out"
      mkdir -p "$out_dir"

      python finetune.py \
        --base_model t5-small \
        --output_dir "$out_dir" \
        --dataset "$dataset_name" \
        --index_file .index.json \
        --epochs="$epochs" \
        --per_device_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --logging_step 50 \
        --model_max_length 512 \
        --warmup_ratio 0.0 \
        --lr_scheduler_type cosine \
        --learning_rate 0.0005 \
        --seed 42

      python test.py \
        --gpu_id 0 \
        --ckpt_path "$out_dir" \
        --dataset "$dataset_name" \
        --data_path ../data \
        --results_file "$run_dir/results.json" \
        --test_batch_size 32 \
        --num_beams 20 \
        --test_prompt_ids 0 \
        --index_file .index.json
      ;;

    etegrec)
      source "$RECSYS26_ROOT/scripts/conda_activate.sh" recsys26_etegrec
      cd "$RECSYS26_ROOT/third_party/ETEGRec"

      python -c "import colorama" >/dev/null 2>&1 || python -m pip install -q colorama

      epochs="$epochs_quick"
      if [[ "$MODE" == "full" ]]; then
        epochs="$etegrec_epochs_full"
      fi

      python main.py \
        --config "config/setrec_${domain}.yaml" \
        --log_dir="$run_dir" \
        --epochs="$epochs" \
        --early_stop=20 \
        --batch_size=32 \
        --eval_batch_size=32 \
        --eval_step=1
      ;;

    seater)
      source "$RECSYS26_ROOT/scripts/conda_activate.sh" recsys26_seater
      cd "$RECSYS26_ROOT/third_party/SEATER"

      epochs="$epochs_quick"
      if [[ "$MODE" == "full" ]]; then
        epochs="$seater_epochs_full"
      fi

      python main.py \
        --name "SEATER_SETRec_${domain}_${MODE}" \
        --description "repro" \
        --workspace "$run_dir/workspace" \
        --dataset_name "SETRec_${domain}" \
        --model SEATER \
        --vocab 8 \
        --gpu_id 0 \
        --epochs="$epochs" \
        --batch_size 256 \
        --test_batch_size 1024 \
        --num_workers 4 \
        --no-tb \
        --no-train_tb
      ;;

    tiger)
      source "$RECSYS26_ROOT/scripts/conda_activate.sh" recsys26_rqvae
      cd "$RECSYS26_ROOT/third_party/RQ_VAE_Recommender"

      iterations=2000
      if [[ "$MODE" == "full" ]]; then
        iterations="$tiger_iterations_full"
      fi

      cfg="$run_dir/decoder_setrec_${domain}.gin"
      python3 - <<PY
from pathlib import Path

domain = "${domain}"
run_dir = "${run_dir}"
iterations = int("${iterations}")

src = Path("${RECSYS26_ROOT}/third_party/RQ_VAE_Recommender/configs/decoder_setrec_smoke.gin")
dst = Path(run_dir) / f"decoder_setrec_{domain}.gin"

text = src.read_text()
text = text.replace('train.dataset_split="beauty"', f'train.dataset_split="{domain}"')
text = text.replace(
    'train.save_dir_root="/data/xqp_data/RecSys26/logs/rqvae_recommender_setrec/decoder/"',
    f'train.save_dir_root="{run_dir}/ckpt/"',
)
text = text.replace('train.iterations=2000', f'train.iterations={iterations}')

dst.write_text(text)
print("[tiger] wrote", dst)
PY

      python train_decoder.py "$cfg"
      ;;

    rpg)
      source "$RECSYS26_ROOT/scripts/conda_activate.sh" recsys26_rpg
      cd "$RECSYS26_ROOT/third_party/RPG_KDD2025"

      epochs="$epochs_quick"
      train_batch_size=64
      eval_batch_size=64
      n_codebook=4
      codebook_size=16
      if [[ "$MODE" == "full" ]]; then
        epochs="$rpg_epochs_full"
        train_batch_size=256
        eval_batch_size=64
        n_codebook=64
        codebook_size=256
      fi

      python main.py \
        --dataset=SETRec \
        --category="$domain" \
        --epochs="$epochs" \
        --train_batch_size="$train_batch_size" \
        --eval_batch_size="$eval_batch_size" \
        --cache_dir="$RECSYS26_ROOT/datasets/rpg_cache" \
        --log_dir="$run_dir" \
        --ckpt_dir="$run_dir/ckpt" \
        --run_id="rpg_setrec_${domain}_$(now_ts)" \
        --num_proc=1 \
        --sent_emb_model=setrec_t5_tdcb \
        --sent_emb_dim=768 \
        --sent_emb_pca=0 \
        --sent_emb_batch_size=256 \
        --n_codebook="$n_codebook" \
        --codebook_size="$codebook_size"
      ;;

    diffgrm)
      source "$RECSYS26_ROOT/scripts/conda_activate.sh" recsys26_diffgrm
      cd "$RECSYS26_ROOT/third_party/DiffGRM"

      epochs="$epochs_quick"
      train_batch_size=64
      eval_batch_size=64
      if [[ "$MODE" == "full" ]]; then
        epochs="$diffgrm_epochs_full"
        train_batch_size=256
        eval_batch_size=64
      fi

      python main.py \
        --model=DIFF_GRM \
        --dataset=SETRec \
        --category="$domain" \
        --epochs="$epochs" \
        --train_batch_size="$train_batch_size" \
        --eval_batch_size="$eval_batch_size" \
        --cache_dir="$RECSYS26_ROOT/datasets/diffgrm_cache" \
        --log_dir="$run_dir" \
        --ckpt_dir="$run_dir/ckpt" \
        --run_id="diffgrm_setrec_${domain}_$(now_ts)" \
        --num_proc=1 \
        --sent_emb_model=setrec_t5_tdcb \
        --sent_emb_dim=768 \
        --sent_emb_pca=0 \
        --force_regenerate_opq=False \
        --sent_emb_batch_size=256
      ;;

    *)
      echo "Unknown method: $method" >&2
      exit 2
      ;;
  esac

  echo "[run] done $(date -Is)"
) >"$log_file" 2>&1

code=$?
set -e

echo "$code" >"$run_dir/exit_code.txt"
echo "$(date -Is)" >"$run_dir/end_time.txt"

if [[ "$code" -ne 0 ]]; then
  echo "[repro_dispatch] FAILED exit_code=$code (see $log_file)" >&2
else
  echo "[repro_dispatch] OK (see $log_file)" >&2
fi

exit "$code"
