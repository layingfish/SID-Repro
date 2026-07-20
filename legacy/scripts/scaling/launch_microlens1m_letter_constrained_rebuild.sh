#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-rebuild}"  # rebuild: skip tokenizer training; full: retrain then export
GPU="${2:-1}"

ROOT="${RECSYS26_ROOT:-/data/xqp_data/RecSys26}"
DOMAIN="microlens_1m"
LOG_DIR="$ROOT/logs/letter_tokenizer_setrec/$DOMAIN/launchers"
mkdir -p "$LOG_DIR"

ts="$(date +%Y%m%d_%H%M%S)"
log="$LOG_DIR/letter_constrained_${MODE}_gpu${GPU}_${ts}.nohup"

case "$MODE" in
  rebuild) skip_train=1 ;;
  full) skip_train=0 ;;
  *) echo "Usage: $0 [rebuild|full] [gpu]" >&2; exit 2 ;;
esac

(
  set -euo pipefail
  cd "$ROOT"

  export CUDA_VISIBLE_DEVICES="$GPU"
  export RECSYS26_LETTER_TOKENIZER_SKIP_TRAIN="$skip_train"
  export PYTHONPATH="$ROOT/envs/miniforge3/envs/recsys26_seater/lib/python3.10/site-packages:${PYTHONPATH:-}"

  echo "[letter_constrained] start $(date -Is)"
  echo "[letter_constrained] mode=$MODE gpu=$GPU skip_train=$skip_train"
  echo "[letter_constrained] alpha=${RECSYS26_LETTER_TOKENIZER_ALPHA:-0.1} beta=${RECSYS26_LETTER_TOKENIZER_BETA:-1e-4} epochs=${RECSYS26_LETTER_TOKENIZER_EPOCHS:-500} eval_step=${RECSYS26_LETTER_TOKENIZER_EVAL_STEP:-50} batch=${RECSYS26_LETTER_TOKENIZER_BATCH:-1024}"
  echo "[letter_constrained] km_n_init=${RECSYS26_LETTER_KM_N_INIT:-10} km_max_iter=${RECSYS26_LETTER_KM_MAX_ITER:-10} km_n_jobs=${RECSYS26_LETTER_KM_N_JOBS:-10}"
  echo "[letter_constrained] init_backend=${RECSYS26_LETTER_INIT_BACKEND:-constrained}"
  echo "[letter_constrained] cluster_backend=${RECSYS26_LETTER_CLUSTER_BACKEND:-constrained}"
  echo "[letter_constrained] PYTHONPATH=$PYTHONPATH"

  source "$ROOT/scripts/conda_activate.sh" recsys26_letter

  python - <<'PY'
from k_means_constrained import KMeansConstrained
import ortools
print("[letter_constrained] k_means_constrained import OK")
print("[letter_constrained] ortools", ortools.__version__)
PY

  scripts/build_letter_indices_setrec.sh "$DOMAIN"

  python - <<'PY'
import collections
import json
from pathlib import Path

path = Path("/data/xqp_data/RecSys26/third_party/LETTER/data/SETRec_microlens_1m/SETRec_microlens_1m.index.json")
raw = json.loads(path.read_text())
seqs = [tuple(raw[str(i)]) for i in range(len(raw))]
cnt = collections.Counter(seqs)
total = len(seqs)
unique = len(cnt)
print("[letter_constrained] index", path)
print("[letter_constrained] items", total)
print("[letter_constrained] unique_codes", unique)
print("[letter_constrained] collision_rate", f"{(total - unique) / total:.6f}")
print("[letter_constrained] max_collision_group", max(cnt.values()))
print("[letter_constrained] per_position_usage", [len({s[i] for s in seqs}) for i in range(len(seqs[0]))])
PY

  echo "[letter_constrained] done $(date -Is)"
) >"$log" 2>&1 < /dev/null &

pid="$!"
echo "[letter_constrained] pid=$pid"
echo "[letter_constrained] log=$log"
