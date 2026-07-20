set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECSYS26_ROOT="${RECSYS26_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$RECSYS26_ROOT/data}"

if [[ "$#" -ge 1 ]]; then
  DATASET="$1"
else
  DATASET="${DATASET:-yelp}"
fi

DATA_DIR=${DATA_DIR:-$DATA_ROOT/setrec_data/$DATASET}
EMB_PATH=${EMB_PATH:-$DATA_DIR/$DATASET.emb-t5-tdcb.npy}
ONEREC_DIR=${ONEREC_DIR:-$RECSYS26_ROOT/baselines/ref05}
PYTHON=${PYTHON:-python}
EVAL_PYTHON=${EVAL_PYTHON:-$PYTHON}
GPU=${GPU:-${CUDA_VISIBLE_DEVICES%%,*}}
GPU=${GPU:-0}
ARCHITECTURE=${ARCHITECTURE:-seq2seq}

RUN_TAG=${RUN_TAG:-20260401_stage1_t5small_L3K8192}
RUN_DIR=${RECSYS26_RUN_DIR:-${RUN_DIR:-$RECSYS26_ROOT/logs/repro_from_scratch/onerec/${DATASET}/${RUN_TAG}}}
LOG_PATH=$RUN_DIR/run.log

NUM_LEVELS=${NUM_LEVELS:-3}
CODEBOOK_WIDTH=${CODEBOOK_WIDTH:-8192}
TOKENIZER_BACKEND=${TOKENIZER_BACKEND:-balanced}
TOKENIZER_ASSIGN_METHOD=${TOKENIZER_ASSIGN_METHOD:-nearest}
TOKENIZER_NITER=${TOKENIZER_NITER:-50}
TOKENIZER_NREDO=${TOKENIZER_NREDO:-10}
BALANCED_SEARCH_TOPK=${BALANCED_SEARCH_TOPK:-32}

SESSION_SIZE=${SESSION_SIZE:-5}
TRAIN_SESSIONS_PATH=${TRAIN_SESSIONS_PATH:-}
TRAIN_FUTURE_SUFFIX=${TRAIN_FUTURE_SUFFIX:-0}
MIN_SESSION_SIZE=${MIN_SESSION_SIZE:-}
MAX_SESSION_SIZE=${MAX_SESSION_SIZE:-}
FUTURE_MIN_ITEMS=${FUTURE_MIN_ITEMS:-1}
FUTURE_MAX_ITEMS=${FUTURE_MAX_ITEMS:-5}
FUTURE_LENGTH_STRATEGY=${FUTURE_LENGTH_STRATEGY:-eval_empirical}
FUTURE_LENGTH_SOURCE=${FUTURE_LENGTH_SOURCE:-val}
MAX_HIST=${MAX_HIST:-256}
MIN_HIST=${MIN_HIST:-3}
BATCH_SIZE=${BATCH_SIZE:-16}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
EVAL_GENERATE_BATCH_SIZE=${EVAL_GENERATE_BATCH_SIZE:-1}
LR=${LR:-2e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
GRAD_CLIP=${GRAD_CLIP:-1.0}
MAX_STEPS=${MAX_STEPS:-30000}
EVAL_EVERY=${EVAL_EVERY:-2000}
LOG_EVERY=${LOG_EVERY:-100}
PATIENCE=${PATIENCE:-8}
EVAL_BEAM_SIZE=${EVAL_BEAM_SIZE:-128}
EVAL_TOPK=${EVAL_TOPK:-20}
EVAL_USERS=${EVAL_USERS:-2000}
DEVICE=${DEVICE:-cuda:0}
AMP_MODE=${AMP_MODE:-bf16}
FREEZE_NON_EMBEDDING_STEPS=${FREEZE_NON_EMBEDDING_STEPS:-0}

mkdir -p "$RUN_DIR/data"

echo "=== Stage-1 T5-small OneRec pipeline ===" | tee -a "$LOG_PATH"
echo "RUN_DIR=$RUN_DIR" | tee -a "$LOG_PATH"
echo "Tokenizer: backend=$TOKENIZER_BACKEND L=$NUM_LEVELS K=$CODEBOOK_WIDTH" | tee -a "$LOG_PATH"
echo "Architecture: $ARCHITECTURE" | tee -a "$LOG_PATH"
if [ -n "$TRAIN_SESSIONS_PATH" ]; then
  echo "Train: session_jsonl=$TRAIN_SESSIONS_PATH min_session=$MIN_SESSION_SIZE max_session=$MAX_SESSION_SIZE max_hist=$MAX_HIST beam=$EVAL_BEAM_SIZE amp=$AMP_MODE" | tee -a "$LOG_PATH"
elif [ "$TRAIN_FUTURE_SUFFIX" = "1" ]; then
  echo "Train: future_suffix=1 future_min=$FUTURE_MIN_ITEMS future_max=$FUTURE_MAX_ITEMS future_strategy=$FUTURE_LENGTH_STRATEGY future_source=$FUTURE_LENGTH_SOURCE min_session=$MIN_SESSION_SIZE max_session=$MAX_SESSION_SIZE max_hist=$MAX_HIST beam=$EVAL_BEAM_SIZE amp=$AMP_MODE" | tee -a "$LOG_PATH"
else
  echo "Train: session_size=$SESSION_SIZE max_hist=$MAX_HIST beam=$EVAL_BEAM_SIZE amp=$AMP_MODE" | tee -a "$LOG_PATH"
fi

echo "=== Step 1: Convert embeddings ===" | tee -a "$LOG_PATH"
"$PYTHON" - <<PYEOF 2>&1 | tee -a "$LOG_PATH"
import numpy as np
import torch

emb = np.load("$EMB_PATH")
print(f"  embedding shape: {emb.shape}")
torch.save(torch.from_numpy(emb).float(), "$RUN_DIR/data/embeddings.pt")
print("  saved:", "$RUN_DIR/data/embeddings.pt")
PYEOF

echo "=== Step 2: Build semantic IDs ===" | tee -a "$LOG_PATH"
cd "$ONEREC_DIR"
TOKENIZER_CMD=(
  "$PYTHON" -m onerec_v2.tokenizer
  --emb_path "$EMB_PATH"
  --output_path "$RUN_DIR/data/semantic_ids.pt"
  --L "$NUM_LEVELS"
  --K "$CODEBOOK_WIDTH"
  --backend "$TOKENIZER_BACKEND"
  --assign_method "$TOKENIZER_ASSIGN_METHOD"
  --niter "$TOKENIZER_NITER"
  --nredo "$TOKENIZER_NREDO"
  --balanced_search_topk "$BALANCED_SEARCH_TOPK"
)
if [ "$TOKENIZER_BACKEND" = "faiss" ]; then
  TOKENIZER_CMD+=(--gpu)
fi
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "${TOKENIZER_CMD[@]}" 2>&1 | tee -a "$LOG_PATH"

echo "=== Step 3: Train seq2seq Stage-1 model ===" | tee -a "$LOG_PATH"
TRAIN_CMD=(
  "$PYTHON" -m onerec_v2.train
  --data_dir "$DATA_DIR"
  --semantic_ids_path "$RUN_DIR/data/semantic_ids.pt"
  --output_dir "$RUN_DIR"
  --architecture "$ARCHITECTURE"
  --pretrained t5-small
  --num_levels "$NUM_LEVELS"
  --codebook_width "$CODEBOOK_WIDTH"
  --max_hist "$MAX_HIST"
  --min_hist "$MIN_HIST"
  --batch_size "$BATCH_SIZE"
  --eval_batch_size "$EVAL_BATCH_SIZE"
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --warmup_ratio "$WARMUP_RATIO"
  --grad_clip "$GRAD_CLIP"
  --max_steps "$MAX_STEPS"
  --eval_every "$EVAL_EVERY"
  --log_every "$LOG_EVERY"
  --patience "$PATIENCE"
  --eval_beam_size "$EVAL_BEAM_SIZE"
  --eval_topk "$EVAL_TOPK"
  --eval_generate_batch_size "$EVAL_GENERATE_BATCH_SIZE"
  --device "$DEVICE"
  --freeze_non_embedding_steps "$FREEZE_NON_EMBEDDING_STEPS"
)
if [ -n "$TRAIN_SESSIONS_PATH" ]; then
  TRAIN_CMD+=(--train_sessions_path "$TRAIN_SESSIONS_PATH")
elif [ "$TRAIN_FUTURE_SUFFIX" = "1" ]; then
  TRAIN_CMD+=(
    --train_future_suffix
    --future_min_items "$FUTURE_MIN_ITEMS"
    --future_max_items "$FUTURE_MAX_ITEMS"
    --future_length_strategy "$FUTURE_LENGTH_STRATEGY"
    --future_length_source "$FUTURE_LENGTH_SOURCE"
  )
else
  TRAIN_CMD+=(--session_size "$SESSION_SIZE")
fi
if [ -n "$MIN_SESSION_SIZE" ]; then
  TRAIN_CMD+=(--min_session_size "$MIN_SESSION_SIZE")
fi
if [ -n "$MAX_SESSION_SIZE" ]; then
  TRAIN_CMD+=(--max_session_size "$MAX_SESSION_SIZE")
fi
if [ -n "$EVAL_USERS" ]; then
  TRAIN_CMD+=(--eval_users "$EVAL_USERS")
fi
if [ "$AMP_MODE" = "bf16" ]; then
  TRAIN_CMD+=(--bf16)
elif [ "$AMP_MODE" = "fp16" ]; then
  TRAIN_CMD+=(--fp16)
fi
ONEREC_K="$CODEBOOK_WIDTH" ONEREC_L="$NUM_LEVELS" CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "${TRAIN_CMD[@]}" 2>&1 | tee -a "$LOG_PATH"

echo "=== Step 4: Export predictions ===" | tee -a "$LOG_PATH"
GENERATE_CMD=(
  "$PYTHON" -m onerec_v2.generate
  --data_dir "$DATA_DIR"
  --semantic_ids_path "$RUN_DIR/data/semantic_ids.pt"
  --ckpt_path "$RUN_DIR/best_model.pt"
  --output_path "$RUN_DIR/pred_topk_test.jsonl"
  --split test
  --max_hist "$MAX_HIST"
  --batch_size "$EVAL_BATCH_SIZE"
  --beam_size "$EVAL_BEAM_SIZE"
  --topk "$EVAL_TOPK"
  --generate_batch_size "$EVAL_GENERATE_BATCH_SIZE"
  --device "$DEVICE"
)
if [ -n "$MIN_SESSION_SIZE" ]; then
  GENERATE_CMD+=(--min_session_size "$MIN_SESSION_SIZE")
fi
if [ -n "$MAX_SESSION_SIZE" ]; then
  GENERATE_CMD+=(--max_session_size "$MAX_SESSION_SIZE")
fi
if [ -z "$TRAIN_SESSIONS_PATH" ] && [ -n "$SESSION_SIZE" ] && [ -z "$MIN_SESSION_SIZE" ] && [ -z "$MAX_SESSION_SIZE" ]; then
  GENERATE_CMD+=(--session_size "$SESSION_SIZE")
fi
ONEREC_K="$CODEBOOK_WIDTH" ONEREC_L="$NUM_LEVELS" CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "${GENERATE_CMD[@]}" 2>&1 | tee -a "$LOG_PATH"

echo "=== Step 5: Unified eval ===" | tee -a "$LOG_PATH"
"$EVAL_PYTHON" "$RECSYS26_ROOT/scripts/unified_eval.py" \
  --pred "$RUN_DIR/pred_topk_test.jsonl" \
  --dataset "$DATASET" \
  --data_dir "$DATA_ROOT/setrec_data" \
  --mode full \
  --warm_only \
  --top_n 5,10 \
  --output "$RUN_DIR/metrics_warm_only.json" \
  2>&1 | tee -a "$LOG_PATH"

echo "=== ALL DONE ===" | tee -a "$LOG_PATH"
