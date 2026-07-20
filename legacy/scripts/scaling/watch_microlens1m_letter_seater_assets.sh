#!/usr/bin/env bash
set -euo pipefail

ROOT="${RECSYS26_ROOT:-/data/xqp_data/RecSys26}"
SASREC_PID="${1:?usage: $0 <sasrec_pid> [letter_gpu]}"
LETTER_GPU="${2:-1}"
DOMAIN="microlens_1m"
RUN_LOG_DIR="$ROOT/logs/microlens1m_letter_seater_retrain"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$RUN_LOG_DIR/postprocess_${TS}.log"

mkdir -p "$RUN_LOG_DIR"

exec > >(tee -a "$LOG") 2>&1

echo "[postprocess] start $(date -Is)"
echo "[postprocess] sasrec_pid=$SASREC_PID letter_gpu=$LETTER_GPU"

while kill -0 "$SASREC_PID" >/dev/null 2>&1; do
  echo "[postprocess] waiting for SASRec pid=$SASREC_PID $(date -Is)"
  sleep 60
done

echo "[postprocess] SASRec process exited $(date -Is)"

SASREC_RUN_DIR="$(ls -1dt "$ROOT/logs/sasrec_embed/$DOMAIN"/SASREC_SETRec_microlens_1m_* 2>/dev/null | head -n 1 || true)"
if [[ -z "$SASREC_RUN_DIR" ]]; then
  echo "[postprocess][ERROR] no SASRec run dir found"
  exit 1
fi

CKPT="$SASREC_RUN_DIR/ckpt/best.pth"
if [[ ! -f "$CKPT" ]]; then
  echo "[postprocess][ERROR] missing best checkpoint: $CKPT"
  exit 1
fi

echo "[postprocess] sasrec_run_dir=$SASREC_RUN_DIR"
echo "[postprocess] ckpt=$CKPT"

source "$ROOT/scripts/conda_activate.sh" recsys26_seater

PICKLE_OUT="$ROOT/third_party/SETRec/data/$DOMAIN/SASRec_item_embed.pkl"
python "$ROOT/scripts/export_sasrec_item_embed.py" \
  --ckpt "$CKPT" \
  --out "$PICKLE_OUT" \
  --drop_pad0 \
  --overwrite

python - <<'PY'
from pathlib import Path
import pickle
import numpy as np

root = Path("/data/xqp_data/RecSys26")
domain = "microlens_1m"
pkl_path = root / "third_party/SETRec/data" / domain / "SASRec_item_embed.pkl"
vocab_dir = root / "datasets/seater_setrec" / domain / "vocab"
vocab_dir.mkdir(parents=True, exist_ok=True)

with pkl_path.open("rb") as f:
    emb = pickle.load(f)
emb = np.asarray(emb, dtype=np.float32)
if emb.shape[0] == 88866:
    emb_with_pad = np.vstack([np.zeros((1, emb.shape[1]), dtype=np.float32), emb])
elif emb.shape[0] == 88867:
    emb_with_pad = emb
else:
    raise ValueError(f"Unexpected SASRec embedding shape: {emb.shape}")

np.save(vocab_dir / f"{domain}_SASREC_item_emb.npy", emb_with_pad)
np.save(vocab_dir / "item_2_attr_mapping.npy", np.zeros((88867, 1), dtype=np.int64))
print(f"[postprocess] wrote SEATER vocab embedding {emb_with_pad.shape}")
PY

LETTER_LOG="$RUN_LOG_DIR/letter_tokenizer_${TS}.log"
echo "[postprocess] launching LETTER tokenizer on gpu=$LETTER_GPU log=$LETTER_LOG"
(
  cd "$ROOT"
  CUDA_VISIBLE_DEVICES="$LETTER_GPU" \
  RECSYS26_LETTER_TOKENIZER_EPOCHS="${RECSYS26_LETTER_TOKENIZER_EPOCHS:-500}" \
  RECSYS26_LETTER_TOKENIZER_BATCH="${RECSYS26_LETTER_TOKENIZER_BATCH:-1024}" \
  bash "$ROOT/scripts/build_letter_indices_setrec.sh" "$DOMAIN"
) >"$LETTER_LOG" 2>&1 &
LETTER_PID=$!
echo "[postprocess] letter_pid=$LETTER_PID"

echo "[postprocess] building SEATER tree and tiger compat $(date -Is)"
python - <<'PY'
from pathlib import Path
import json
import numpy as np
import sys

root = Path("/data/xqp_data/RecSys26")
sys.path.insert(0, str(root / "third_party/SEATER"))

from utils.build_tree import build_hieraichical_clustering_tree

domain = "microlens_1m"
emb_path = root / "datasets/seater_setrec" / domain / "vocab" / f"{domain}_SASREC_item_emb.npy"
tree_dir = root / "datasets/seater_setrec" / domain / "tree_data_SASREC" / "8_branch_tree"
tree_dir.mkdir(parents=True, exist_ok=True)

if not (tree_dir / "itemID_2_tree_indexID.npy").exists():
    build_hieraichical_clustering_tree(
        item_emb_path=str(emb_path),
        output_file_path=str(tree_dir),
        vocab_size=8,
    )
else:
    print(f"[postprocess] reuse existing SEATER tree: {tree_dir}")

matrix = np.load(tree_dir / "itemID_2_tree_indexID.npy")
if matrix.shape[0] != 88867:
    raise ValueError(f"Unexpected SEATER matrix shape: {matrix.shape}")

raw = matrix[1:]  # convert shifted SEATER item ids back to canonical 0-based item ids.
target_len = raw.shape[1]
per_pos_maps = [dict() for _ in range(target_len)]
cached = np.zeros_like(raw, dtype=np.int64)

for item_idx in range(raw.shape[0]):
    for pos in range(target_len):
        node = int(raw[item_idx, pos])
        mapping = per_pos_maps[pos]
        if node not in mapping:
            mapping[node] = len(mapping)
        cached[item_idx, pos] = mapping[node]

per_pos_sizes = [len(m) for m in per_pos_maps]
out_dir = root / "logs/scaling_microlens1m_retrain_clean/I03-seater/tiger_compat"
out_dir.mkdir(parents=True, exist_ok=True)
np.save(out_dir / "cached_ids.npy", cached)
(out_dir / "tiger_config.json").write_text(json.dumps({
    "n_items": int(cached.shape[0]),
    "sem_id_dim": int(cached.shape[1]),
    "codebook_size": int(max(per_pos_sizes)),
    "per_pos_sizes": [int(x) for x in per_pos_sizes],
    "id_type": "I03-seater",
    "id_name": "SEATER Tree Code",
    "source_tree": str(tree_dir / "itemID_2_tree_indexID.npy"),
}, indent=2), encoding="utf-8")

print(f"[postprocess] wrote I03 tiger_compat {cached.shape} per_pos_sizes={per_pos_sizes}")
PY

echo "[postprocess] waiting LETTER tokenizer pid=$LETTER_PID"
wait "$LETTER_PID"
echo "[postprocess] LETTER tokenizer finished $(date -Is)"

source "$ROOT/scripts/conda_activate.sh" recsys26_rqvae

LETTER_INDEX="$ROOT/third_party/LETTER/data/SETRec_${DOMAIN}/SETRec_${DOMAIN}.index.json"
LETTER_MANIFEST="$ROOT/logs/scaling_microlens1m_retrain_clean/I05-letter/manifest"
LETTER_COMPAT="$ROOT/logs/scaling_microlens1m_retrain_clean/I05-letter/tiger_compat"

python "$ROOT/scripts/scaling/generate_manifest.py" \
  --id_type letter \
  --letter_index_path "$LETTER_INDEX" \
  --output_dir "$LETTER_MANIFEST"

python "$ROOT/scripts/scaling/manifest_to_cached_ids.py" \
  --manifest_dir "$LETTER_MANIFEST" \
  --output_dir "$LETTER_COMPAT"

echo "[postprocess] done $(date -Is)"
echo "[postprocess] log=$LOG"
