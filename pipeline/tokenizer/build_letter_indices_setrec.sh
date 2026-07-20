#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: build_letter_indices_setrec.sh <domain>" >&2
  echo "  domain: beauty | toys | sports | steam | amazon23_vg | microlens_50k | microlens_100k | microlens_1m | yelp" >&2
  exit 2
fi

domain="$1"
case "$domain" in
  beauty|toys|sports|steam|amazon23_vg|microlens_50k|microlens_100k|microlens_1m|yelp) ;;
  *) echo "Unknown domain: $domain" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECSYS26_ROOT="${RECSYS26_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$RECSYS26_ROOT/data}"
LETTER_CODE_DIR="${LETTER_CODE_DIR:-$RECSYS26_ROOT/baselines/ref03}"
# Prefer externally provided CUDA_VISIBLE_DEVICES (e.g. with_gpu_lock.sh).
# Fall back to a safe GPU chosen by pick_gpu.sh (avoids GPU2 by default).
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "" ]]; then
  GPU="${CUDA_VISIBLE_DEVICES%%,*}"   # physical GPU id (first entry)
else
  GPU="${RECSYS26_LETTER_TOKENIZER_GPU:-$("$RECSYS26_ROOT/scripts/pick_gpu.sh")}"   # physical GPU id
fi

alpha="${RECSYS26_LETTER_TOKENIZER_ALPHA:-0.1}"
beta="${RECSYS26_LETTER_TOKENIZER_BETA:-1e-4}"
epochs="${RECSYS26_LETTER_TOKENIZER_EPOCHS:-500}"
batch_size="${RECSYS26_LETTER_TOKENIZER_BATCH:-1024}"
eval_step="${RECSYS26_LETTER_TOKENIZER_EVAL_STEP:-50}"

num_workers="${RECSYS26_LETTER_TOKENIZER_NUM_WORKERS:-0}"

emb_path="$DATA_ROOT/setrec_data/$domain/${domain}.emb-t5-tdcb.npy"
cf_emb_path="$DATA_ROOT/setrec_data/$domain/SASRec_item_embed.pkl"

if [[ ! -f "$emb_path" ]]; then
  echo "Missing embedding: $emb_path" >&2
  exit 2
fi
if [[ ! -f "$cf_emb_path" ]]; then
  echo "Missing CF embedding: $cf_emb_path" >&2
  exit 2
fi

ckpt_base="${RECSYS26_LETTER_TOKENIZER_CKPT_BASE:-$RECSYS26_ROOT/datasets/letter_tokenizer_setrec/$domain}"
log_dir="${RECSYS26_LETTER_TOKENIZER_LOG_DIR:-$RECSYS26_ROOT/logs/letter_tokenizer_setrec/$domain}"
mkdir -p "$ckpt_base" "$log_dir"

ts="$(date +%Y%m%d_%H%M%S)"
train_log="$log_dir/train_${ts}.log"
gen_log="$log_dir/generate_${ts}.log"

index_out="${RECSYS26_LETTER_TOKENIZER_INDEX_OUT:-$LETTER_CODE_DIR/data/SETRec_${domain}/SETRec_${domain}.index.json}"

source "$RECSYS26_ROOT/scripts/conda_activate.sh" recsys26_letter
cd "$LETTER_CODE_DIR"

skip_train="${RECSYS26_LETTER_TOKENIZER_SKIP_TRAIN:-0}"


# Train tokenizer
if [[ "$skip_train" == "1" ]]; then
  {
    echo "[train_tokenizer] skipped $(date -Is)"
    echo "[train_tokenizer] using existing checkpoints under $ckpt_base"
  } >"$train_log" 2>&1
else
  (
    echo "[train_tokenizer] start $(date -Is)"
    echo "[train_tokenizer] domain=$domain gpu=$GPU alpha=$alpha beta=$beta epochs=$epochs batch_size=$batch_size eval_step=$eval_step"
    echo "[train_tokenizer] emb_path=$emb_path"
    echo "[train_tokenizer] cf_emb_path=$cf_emb_path"

    CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 python -u ./RQ-VAE/main.py \
      --device cuda:0 \
      --data_path "$emb_path" \
      --cf_emb "$cf_emb_path" \
      --alpha "$alpha" \
      --beta "$beta" \
      --epochs "$epochs" \
      --batch_size "$batch_size" \
      --num_workers "$num_workers" \
      --eval_step "$eval_step" \
      --ckpt_dir "$ckpt_base"

    echo "[train_tokenizer] done $(date -Is)"
  ) >"$train_log" 2>&1
fi

# Find latest checkpoint dir + file
latest_dir="$(ls -1dt "$ckpt_base"/* 2>/dev/null | head -n 1 || true)"
if [[ "$latest_dir" == "" ]]; then
  echo "No checkpoint directory found under $ckpt_base" >&2
  exit 2
fi

ckpt_path="$latest_dir/best_collision_model.pth"
if [[ ! -f "$ckpt_path" ]]; then
  ckpt_path="$(find "$latest_dir" -maxdepth 1 -type f -name "*.pth" | head -n 1 || true)"
fi
if [[ ! -f "$ckpt_path" ]]; then
  echo "No checkpoint file found under $latest_dir" >&2
  exit 2
fi

# Backup existing index file if present
if [[ -f "$index_out" ]]; then
  cp -f "$index_out" "$index_out.bak_${ts}"
fi

# Generate indices
(
  echo "[generate_indices] start $(date -Is)"
  echo "[generate_indices] ckpt_path=$ckpt_path"
  echo "[generate_indices] out=$index_out"
  echo "[generate_indices] cluster_backend=${RECSYS26_LETTER_CLUSTER_BACKEND:-constrained}"

  RECSYS26_CKPT_PATH="$ckpt_path" RECSYS26_INDEX_OUT="$index_out" CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 python -u - <<PY
import ast
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

RQVAE_DIR = Path("${LETTER_CODE_DIR}/RQ-VAE")
import sys
sys.path.insert(0, str(RQVAE_DIR))

from datasets import EmbDataset
from models.rqvae import RQVAE

ckpt_path = Path(os.environ["RECSYS26_CKPT_PATH"])
out_path = Path(os.environ["RECSYS26_INDEX_OUT"])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(str(ckpt_path), map_location=torch.device("cpu"))
args = ckpt["args"]
state_dict = ckpt["state_dict"]

data = EmbDataset(args.data_path)

model = RQVAE(
    in_dim=data.dim,
    num_emb_list=args.num_emb_list,
    e_dim=args.e_dim,
    layers=args.layers,
    dropout_prob=args.dropout_prob,
    bn=args.bn,
    loss_type=args.loss_type,
    quant_loss_weight=args.quant_loss_weight,
    kmeans_init=args.kmeans_init,
    kmeans_iters=args.kmeans_iters,
    sk_epsilons=args.sk_epsilons,
    sk_iters=args.sk_iters,
)
model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model.eval()

# Helpers copied from upstream generate_indices.py (with stdlib-friendly fallbacks).
prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>","<f_{}>"]


def constrained_km(data, n_clusters=10):
    x = data
    if os.environ.get("RECSYS26_LETTER_CLUSTER_BACKEND", "constrained") == "sklearn":
        from sklearn.cluster import KMeans
        clf = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            max_iter=300,
            random_state=42,
        )
        clf.fit(x)
        t_centers = torch.from_numpy(clf.cluster_centers_)
        t_labels = torch.from_numpy(clf.labels_).tolist()
        return t_centers, t_labels

    try:
        from k_means_constrained import KMeansConstrained
        size_min = min(len(x) // (n_clusters * 2), 10)
        clf = KMeansConstrained(
            n_clusters=n_clusters,
            size_min=size_min,
            size_max=n_clusters * 6,
            max_iter=10,
            n_init=10,
            n_jobs=10,
            verbose=False,
        )
    except Exception:
        from sklearn.cluster import KMeans
        clf = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            max_iter=300,
            random_state=42,
        )

    clf.fit(x)
    t_centers = torch.from_numpy(clf.cluster_centers_)
    t_labels = torch.from_numpy(clf.labels_).tolist()
    return t_centers, t_labels


def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item==tot_indice


def get_indices_count(all_indices_str):
    from collections import defaultdict
    indices_count = defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count


def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        index2id.setdefault(index, []).append(i)

    collision_item_groups = []
    for index, ids in index2id.items():
        if len(ids) > 1:
            collision_item_groups.append(ids)
    return collision_item_groups


# Build labels for each RQ layer
labels = {str(i): [] for i in range(len(model.rq.vq_layers))}
embs = [layer.embedding.weight.cpu().detach().numpy() for layer in model.rq.vq_layers]
for idx, emb in enumerate(embs):
    _, label = constrained_km(emb)
    labels[str(idx)] = label

# Generate raw indices
loader = DataLoader(data, num_workers=getattr(args, "num_workers", 4), batch_size=64, shuffle=False, pin_memory=True)
all_indices = []
all_indices_str = []

for batch in tqdm(loader, desc="encode", ncols=100):
    d = batch[0].to(device)
    indices = model.get_indices(d, labels, use_sk=False)
    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    for index in indices:
        code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
        all_indices.append(code)
        all_indices_str.append(str(code))

all_indices = np.array(all_indices)
all_indices_str = np.array(all_indices_str)

# Collision fixing: enable sinkhorn on last layer only
for vq in model.rq.vq_layers[:-1]:
    vq.sk_epsilon = 0.0
if getattr(model.rq.vq_layers[-1], "sk_epsilon", 0.0) == 0.0:
    model.rq.vq_layers[-1].sk_epsilon = 0.003

# Iteratively resolve collisions
attempt = 0
while attempt < 20 and not check_collision(all_indices_str):
    collision_groups = get_collision_item(all_indices_str)
    for group in collision_groups:
        # EmbDataset supports list/ndarray indexing
        d = data[group][0].to(device)
        indices = model.get_indices(d, labels, use_sk=True)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for item_idx, index in zip(group, indices):
            code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
            all_indices[item_idx] = code
            all_indices_str[item_idx] = str(code)
    attempt += 1

print("All indices number:", len(all_indices))
print("Max number of conflicts:", max(get_indices_count(all_indices_str).values()))

tot_item = len(all_indices_str)
tot_indice = len(set(all_indices_str.tolist()))
print("Collision Rate", (tot_item - tot_indice) / tot_item)

out_path.parent.mkdir(parents=True, exist_ok=True)
all_indices_dict = {str(item): list(code) for item, code in enumerate(all_indices.tolist())}
out_path.write_text(json.dumps(all_indices_dict, ensure_ascii=False))
print("Wrote", out_path)
PY

  echo "[generate_indices] done $(date -Is)"
) >"$gen_log" 2>&1

echo "[ok] index_out=$index_out"
echo "[ok] train_log=$train_log"
echo "[ok] gen_log=$gen_log"
