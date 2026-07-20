#!/usr/bin/env bash
set -euo pipefail

source /data/xqp_data/RecSys26/scripts/conda_activate.sh recsys26_eager

mkdir -p /data/xqp_data/RecSys26/datasets/eager_setrec_smoke
mkdir -p /data/xqp_data/RecSys26/logs/eager_setrec_smoke
log=/data/xqp_data/RecSys26/logs/eager_setrec_smoke/run.log

cd /data/xqp_data/RecSys26/third_party/EAGER/EAGER

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(/data/xqp_data/RecSys26/scripts/pick_gpu.sh)}"

python - <<\PY > "$log" 2>&1
from pathlib import Path

jsonl = Path("/data/xqp_data/RecSys26/datasets/eager_setrec/beauty/beauty.jsonl")

from lib.generate_train_and_test_data import _read

behavior_dict, train_sample, test_sample, val_sample, user_num, item_num, _ = _read(
    str(jsonl), test_record_num=0
)
print("[OK] _read", "users", user_num, "items", item_num, "train", len(train_sample["USERID"]))

import numpy as np
import torch
from lib.Trm4Rec_trainer import Trm4Rec

torch.manual_seed(0)
np.random.seed(0)

item_num = int(item_num)
seq_len = 10
batch = 4

tmp_dir = Path("/data/xqp_data/RecSys26/datasets/eager_setrec_smoke")
code_to_item_file = str(tmp_dir / "code_to_item.npy")
item_to_code_file = str(tmp_dir / "item_to_code.npy")

model = Trm4Rec(
    item_num=item_num,
    user_seq_len=seq_len,
    d_model=32,
    d_model2=96,
    nhead=4,
    device="cuda",
    enc_num_layers=1,
    dec_num_layers=1,
    k=28,
    item_to_code_file=item_to_code_file,
    code_to_item_file=code_to_item_file,
    tree_has_generated=False,
    init_way="random",
)

batch_x = torch.randint(
    low=0, high=item_num, size=(batch, seq_len), device="cuda", dtype=torch.int64
)
batch_y = torch.randint(low=0, high=item_num, size=(batch,), device="cuda", dtype=torch.int64)

# Use random embeddings to validate forward/backward only.
data_emb = torch.randn(item_num, 768, device="cuda")
loss, contra, _ = model.update_model(
    batch_x,
    batch_y,
    data_emb=data_emb,
    type=0,
    use_con=False,
    use_guide=False,
    guide_feat=None,
)
loss.backward()
print("[OK] update_model", float(loss.detach().cpu()))
PY

tail -n 120 "$log" || true
