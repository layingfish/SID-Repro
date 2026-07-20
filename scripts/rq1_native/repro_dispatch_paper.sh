#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: repro_dispatch_paper.sh <method> <domain> [variant]" >&2
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECSYS26_ROOT="${RECSYS26_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$RECSYS26_ROOT/data}"
MODEL_ROOT="${MODEL_ROOT:-$RECSYS26_ROOT/models}"
PROFILE="${RECSYS26_PROFILE:-paper}"

pick_port() {
  local base=20000
  local span=20000
  echo $(( base + ( ( $(date +%s) + $$ + RANDOM ) % span ) ))
}

now_ts() {
  date +%Y%m%d_%H%M%S
}

if [[ "${CUDA_VISIBLE_DEVICES:-}" == "" ]]; then
  export CUDA_VISIBLE_DEVICES="$(${RECSYS26_ROOT}/scripts/pick_gpu.sh)"
fi

# Count visible GPUs.
IFS=',' read -ra _CVD <<<"${CUDA_VISIBLE_DEVICES}"
NGPU=${#_CVD[@]}

run_root="$RECSYS26_ROOT/logs/repro_paper/$method"
if [[ "$method" == "llm_id" ]]; then
  variant="${variant:-semid}"
  run_root="$RECSYS26_ROOT/logs/repro_paper/$method/$variant"
fi

run_dir="${RECSYS26_RUN_DIR:-$run_root/$domain/$(now_ts)}"
mkdir -p "$run_dir"
ln -sfn "$run_dir" "$run_root/$domain/latest"

echo "$(date -Is)" >"$run_dir/start_time.txt"

log_file="$run_dir/run.log"

echo "[repro_dispatch_paper] method=$method domain=$domain variant=${variant:-n/a} profile=$PROFILE gpu=$CUDA_VISIBLE_DEVICES"
echo "[repro_dispatch_paper] run_dir=$run_dir"

set +e
(
  set -euo pipefail
  echo "[run] start $(date -Is)"
  echo "[run] method=$method domain=$domain variant=${variant:-n/a} profile=$PROFILE CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

  case "$method" in
    setrec)
      if [[ "$NGPU" -lt 4 ]]; then
        echo "[setrec][paper] need 4 GPUs (got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)" >&2
        exit 2
      fi
      source "$RECSYS26_ROOT/scripts/conda_activate.sh" recsys26_setrec
      cd "${SETREC_CODE_DIR:-$RECSYS26_ROOT/baselines/reference/code}"

      lr="${RECSYS26_SETREC_LR:-1e-3}"
      n_sem="${RECSYS26_SETREC_N_SEM:-4}"
      alpha="${RECSYS26_SETREC_ALPHA:-0.7}"
      n_query=$((n_sem + 1))

      master_port="$(pick_port)"
      mkdir -p "$run_dir/out"
      torchrun --nproc_per_node=4 --master_port="$master_port" finetune_t5.py \
        --base_model t5-small \
        --data_path "../data/${domain}/" \
        --output_dir "$run_dir/out" \
        --cache_dir "$MODEL_ROOT/hf/transformers" \
        --sem_encoder t5 \
        --n_query "$n_query" --n_sem "$n_sem" --n_cf 1 --alpha "$alpha" \
        --layers 512 256 128 \
        --batch_size 512 --micro_batch_size 128 \
        --num_epochs 30 \
        --learning_rate "$lr" \
        --cutoff_len 512 \
        --val_set_size 2000 \
        --warmup_steps 100 \
        --lr_scheduler cosine \
        --disable_early_stopping \
        --seed 42 \
        --wandb_project ""
      ;;

    llm_id)
      if [[ "$NGPU" -lt 2 ]]; then
        echo "[llm_id][paper] need 2 GPUs (got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)" >&2
        exit 2
      fi
      source "$RECSYS26_ROOT/scripts/conda_activate.sh" recsys26_llm_id
      cd "${LLM_ID_CODE_DIR:-$RECSYS26_ROOT/baselines/ref02}"

      export MASTER_ADDR=127.0.0.1
      export MASTER_PORT="$(pick_port)"

      task="setrec_${domain}"
      lr=0.001
      epochs=20
      logging_step=1000
      gpu_arg="0,1"

      item_representation="content_based"
      data_order="random"
      extra_args=()

      case "${variant:-semid}" in
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
          echo "Unknown llm_id variant: ${variant}" >&2
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
        --gpu "$gpu_arg" \
        --logging_step "$logging_step" \
        --logging_dir "$run_dir/train.log" \
        --model_dir "$run_dir/model.pt" \
        --train_sequential_item_batch 64 \
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
      cd "${EAGER_CODE_DIR:-$RECSYS26_ROOT/baselines/ref08}"

      eager_parall="${RECSYS26_EAGER_PARALL:-50}"
      eager_k="${RECSYS26_EAGER_K:-128}"
      if [[ "$domain" == "sports" ]]; then
        eager_k="${RECSYS26_EAGER_SPORTS_K:-160}"
      fi

      python train_rec_setrec.py \
        --domain "$domain" \
        --data_root "$DATA_ROOT/setrec_data" \
        --output_dir "$RECSYS26_ROOT/logs/eager_setrec_paper" \
        --seed 2024 \
        --seq_len 20 --min_seq_len 5 \
        --train_sample_seg_cnt 10 \
        --parall "$eager_parall" \
        --tree_num 2 \
        --k "$eager_k" \
        --d_model 96 \
        --enc_num_layers 1 \
        --dec_num_layers 2,2 \
        --n_head 4 \
        --init_way embkm,embkm \
        --feature_ratio 1.0 \
        --max_iters 100 \
        --train_batch_size 256 \
        --total_batch_num 60000 \
        --eval_step 300 \
        --topk 10 \
        --rerank_topk 10
      ;;

    rpg)
      source "$RECSYS26_ROOT/scripts/conda_activate.sh" recsys26_rpg
      cd "${RPG_CODE_DIR:-$RECSYS26_ROOT/baselines/ref04}"

      epochs=150
      train_batch_size=256
      eval_batch_size=32
      sent_emb_batch_size=512
      codebook_size=256

      case "$domain" in
        beauty)
          lr=0.01; temperature=0.03; n_codebook=32; num_beams=20; n_edges=200; prop_steps=3
          ;;
        toys)
          lr=0.003; temperature=0.03; n_codebook=16; num_beams=200; n_edges=20; prop_steps=3
          ;;
        sports)
          lr=0.003; temperature=0.03; n_codebook=16; num_beams=100; n_edges=30; prop_steps=5
          ;;
        steam)
          # No official hyperparams; use the paper's longest-ID setting as a reasonable default.
          lr=0.001; temperature=0.03; n_codebook=64; num_beams=20; n_edges=500; prop_steps=5
          ;;
        amazon23_vg|microlens_50k|yelp)
          lr=0.001; temperature=0.03; n_codebook=64; num_beams=20; n_edges=500; prop_steps=5
          ;;
      esac

      python main.py \
        --dataset=SETRec \
        --category="$domain" \
        --epochs="$epochs" \
        --train_batch_size="$train_batch_size" \
        --eval_batch_size="$eval_batch_size" \
        --lr="$lr" \
        --temperature="$temperature" \
        --n_codebook="$n_codebook" \
        --codebook_size="$codebook_size" \
        --num_beams="$num_beams" \
        --n_edges="$n_edges" \
        --propagation_steps="$prop_steps" \
        --cache_dir="$DATA_ROOT/rpg_cache_paper" \
        --log_dir="$run_dir" \
        --ckpt_dir="$run_dir/ckpt" \
        --run_id="rpg_setrec_${domain}_$(now_ts)" \
        --num_proc=1 \
        --sent_emb_model=setrec_t5_tdcb \
        --sent_emb_dim=768 \
        --sent_emb_pca=0 \
        --sent_emb_batch_size="$sent_emb_batch_size"
      ;;

    diffgrm)
      source "$RECSYS26_ROOT/scripts/conda_activate.sh" recsys26_diffgrm
      cd "${DIFFGRM_CODE_DIR:-$RECSYS26_ROOT/baselines/ref06}"

      epochs=200
      train_batch_size=1024
      eval_batch_size=32
      sent_emb_batch_size=256
      sent_emb_pca=256

      case "$domain" in
        beauty)
          lr=0.01; label_smoothing=0.2; encoder_n_layer=1; decoder_n_layer=4; n_head=4; n_embd=256; n_inner=1024; eval_start_epoch=20
          ;;
        toys)
          lr=0.003; label_smoothing=0.15; encoder_n_layer=1; decoder_n_layer=4; n_head=8; n_embd=1024; n_inner=1024; eval_start_epoch=10; eval_batch_size=8
          ;;
        sports)
          lr=0.003; label_smoothing=0.1; encoder_n_layer=1; decoder_n_layer=4; n_head=4; n_embd=256; n_inner=1024; eval_start_epoch=20
          ;;
        steam)
          # No official hyperparams; reuse sports setting.
          lr=0.003; label_smoothing=0.1; encoder_n_layer=1; decoder_n_layer=4; n_head=4; n_embd=256; n_inner=1024; eval_start_epoch=20
          ;;
      esac

      python main.py \
        --model=DIFF_GRM \
        --dataset=SETRec \
        --category="$domain" \
        --epochs="$epochs" \
        --train_batch_size="$train_batch_size" \
        --eval_batch_size="$eval_batch_size" \
        --n_digit=4 \
        --masking_strategy=guided \
        --guided_refresh_each_step=false \
        --guided_select=least \
        --guided_conf_metric=msp \
        --encoder_n_layer="$encoder_n_layer" \
        --decoder_n_layer="$decoder_n_layer" \
        --n_head="$n_head" \
        --n_embd="$n_embd" \
        --n_inner="$n_inner" \
        --train_sliding=true \
        --min_hist_len=2 \
        --eval_start_epoch="$eval_start_epoch" \
        --lr="$lr" \
        --label_smoothing="$label_smoothing" \
        --sent_emb_model=setrec_t5_tdcb \
        --sent_emb_dim=768 \
        --sent_emb_pca="$sent_emb_pca" \
        --sent_emb_batch_size="$sent_emb_batch_size" \
        --normalize_after_pca=true \
        --force_regenerate_opq=true \
        --share_decoder_output_embedding=true \
        --cache_dir="$DATA_ROOT/diffgrm_cache_paper" \
        --log_dir="$run_dir" \
        --ckpt_dir="$run_dir/ckpt" \
        --run_id="diffgrm_setrec_${domain}_$(now_ts)" \
        --num_proc=1
      ;;

    seater)
      source "$RECSYS26_ROOT/scripts/conda_activate.sh" recsys26_seater
      cd "${SEATER_CODE_DIR:-$RECSYS26_ROOT/baselines/ref01}"

      # Paper repo stores most hyperparameters under ./config; keep main CLI minimal.
      python main.py \
        --name "SEATER_SETRec_${domain}_paper" \
        --description "paper-profile" \
        --workspace "$run_dir/workspace" \
        --dataset_name "SETRec_${domain}" \
        --model SEATER \
        --vocab 8 \
        --gpu_id 0 \
        --epochs 100 \
        --batch_size 256 \
        --test_batch_size 1024 \
        --num_workers 4 \
        --no-tb \
        --no-train_tb
      ;;

    tiger)
      source "$RECSYS26_ROOT/scripts/conda_activate.sh" recsys26_rqvae
      cd "${TIGER_CODE_DIR:-$RECSYS26_ROOT/baselines/decoder}"

      rqvae_iterations="${RECSYS26_TIGER_RQVAE_ITERS:-2000}"
      decoder_iterations="${RECSYS26_TIGER_DECODER_ITERS:-20000}"

      rqvae_cfg="$run_dir/rqvae_setrec_${domain}.gin"
      decoder_cfg="$run_dir/decoder_setrec_${domain}.gin"

      python3 - <<PY
from pathlib import Path

domain = "${domain}"
run_dir = "${run_dir}"
data_root = "${DATA_ROOT}"
iterations = int("${rqvae_iterations}")

src = Path("${TIGER_CODE_DIR:-$RECSYS26_ROOT/baselines/decoder}/configs/rqvae_setrec_smoke.gin")
dst = Path(run_dir) / f"rqvae_setrec_{domain}.gin"

text = src.read_text()
text = text.replace('train.dataset_split="beauty"', f'train.dataset_split="{domain}"')
text = text.replace(
    'train.dataset_folder="data/rqvae_recommender/setrec"',
    f'train.dataset_folder="{data_root}/rqvae_recommender/setrec"',
)
text = text.replace(
    'train.save_dir_root="logs/rqvae_recommender_setrec/rqvae/"',
    f'train.save_dir_root="{run_dir}/ckpt/rqvae/"',
)
text = text.replace('train.iterations=2000', f'train.iterations={iterations}')

dst.write_text(text)
print("[tiger] wrote", dst)
PY

      python train_rqvae.py "$rqvae_cfg"

      rqvae_ckpt="$run_dir/ckpt/rqvae/checkpoint_$((rqvae_iterations-1)).pt"
      if [[ ! -f "$rqvae_ckpt" ]]; then
        echo "[tiger][paper] expected RQ-VAE checkpoint not found: $rqvae_ckpt" >&2
        exit 2
      fi

      python3 - <<PY
from pathlib import Path

domain = "${domain}"
run_dir = "${run_dir}"
data_root = "${DATA_ROOT}"
iterations = int("${decoder_iterations}")
rqvae_ckpt = "${rqvae_ckpt}"

src = Path("${TIGER_CODE_DIR:-$RECSYS26_ROOT/baselines/decoder}/configs/decoder_setrec_smoke.gin")
dst = Path(run_dir) / f"decoder_setrec_{domain}.gin"

text = src.read_text()
text = text.replace('train.dataset_split="beauty"', f'train.dataset_split="{domain}"')
text = text.replace(
    'train.dataset_folder="data/rqvae_recommender/setrec"',
    f'train.dataset_folder="{data_root}/rqvae_recommender/setrec"',
)
text = text.replace(
    'train.save_dir_root="logs/rqvae_recommender_setrec/decoder/"',
    f'train.save_dir_root="{run_dir}/ckpt/decoder/"',
)
text = text.replace('train.iterations=2000', f'train.iterations={iterations}')
text = text.replace(
    'train.pretrained_rqvae_path="trained_models/rqvae_amazon_beauty/checkpoint_399999.pt"',
    f'train.pretrained_rqvae_path="{rqvae_ckpt}"',
)

dst.write_text(text)
print("[tiger] wrote", dst)
PY

      python train_decoder.py "$decoder_cfg"
      ;;

    letter)
      if [[ "$NGPU" -lt 2 ]]; then
        echo "[letter][paper] need 2 GPUs (got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)" >&2
        exit 2
      fi
      source "$RECSYS26_ROOT/scripts/conda_activate.sh" recsys26_letter
      cd "${LETTER_TIGER_CODE_DIR:-$RECSYS26_ROOT/baselines/ref03/LETTER-TIGER}"

      dataset_name="SETRec_${domain}"
      out_dir="$run_dir/out"
      mkdir -p "$out_dir"

      # NOTE: This assumes `${RECSYS26_ROOT}/third_party/LETTER/data/${dataset_name}/${dataset_name}.index.json` exists.
      # Full paper alignment requires training the LETTER tokenizer (RQ-VAE) and generating indices.

      master_port="$(pick_port)"
      torchrun --nproc_per_node=2 --master_port="$master_port" finetune.py \
        --base_model t5-small \
        --output_dir "$out_dir" \
        --dataset "$dataset_name" \
        --index_file .index.json \
        --per_device_batch_size 256 \
        --learning_rate 5e-4 \
        --epochs 200 \
        --temperature 1.0

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
      cd "${ETEGREC_CODE_DIR:-$RECSYS26_ROOT/baselines/ref07}"

      python -c "import colorama" >/dev/null 2>&1 || python -m pip install -q colorama

      python main.py \
        --config "config/setrec_${domain}.yaml" \
        --log_dir="$run_dir" \
        --epochs=999 \
        --early_stop=50 \
        --batch_size=256 \
        --eval_batch_size=256 \
        --eval_step=1
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
  echo "[repro_dispatch_paper] FAILED exit_code=$code (see $log_file)" >&2
else
  echo "[repro_dispatch_paper] OK (see $log_file)" >&2
fi

exit "$code"
