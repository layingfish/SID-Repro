"""训练 Collaborative RQ-VAE: SASRec embedding (64d) → 3-level RQ codes"""
import os, sys, time, pickle, json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter, namedtuple

sys.path.insert(0, "/data/xqp_data/RecSys26/third_party/RQ_VAE_Recommender")
from modules.rqvae import RqVae

# 创建一个最小 SeqBatch 兼容对象
SeqBatch = namedtuple("SeqBatch", ["x"])

def main():
    emb_path = "/data/xqp_data/RecSys26/third_party/SETRec/data/microlens_50k/SASRec_item_embed.pkl"
    out_dir = "/data/xqp_data/RecSys26/logs/scaling_ml50k/I03-cfrq/rqvae_training"
    os.makedirs(out_dir, exist_ok=True)

    iterations = 20000
    batch_size = 1024
    lr = 5e-4
    grad_clip = 1.0
    diag_interval = 1000
    save_interval = 5000
    seed = 42

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 加载 SASRec embedding
    sas = pickle.load(open(emb_path, "rb"))
    if isinstance(sas, torch.Tensor):
        sas = sas.detach().cpu().float()
    else:
        sas = torch.tensor(np.asarray(sas, dtype=np.float32))
    n_items, emb_dim = sas.shape
    print(f"SASRec embedding: {n_items} items, {emb_dim}d")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = RqVae(
        input_dim=emb_dim,
        embed_dim=32,
        hidden_dims=[128, 128, 64],
        codebook_size=256,
        n_layers=3,
        n_cat_features=0,
        commitment_weight=0.25,
    ).to(device)
    print(f"RqVae: {sum(p.numel() for p in model.parameters())/1e3:.1f}K params, device={device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    dataset = TensorDataset(sas)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model.train()
    data_iter = iter(loader)
    log = []
    start = time.time()

    for step in range(1, iterations + 1):
        try:
            (batch_emb,) = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            (batch_emb,) = next(data_iter)

        batch_emb = batch_emb.to(device)

        # 用 RqVae 的原生 forward（需要 SeqBatch）
        batch = SeqBatch(x=batch_emb)
        output = model(batch, gumbel_t=1.0)
        loss = output.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if step % diag_interval == 0:
            elapsed = time.time() - start
            model.eval()
            with torch.no_grad():
                sem_output = model.get_semantic_ids(sas.to(device))
                indices = sem_output[2].cpu().numpy()
            model.train()

            code_counter = Counter(tuple(row) for row in indices.tolist())
            n_unique = len(code_counter)
            per_level = [len(set(indices[:, lv].tolist())) for lv in range(indices.shape[1])]

            entry = {
                "step": step, "loss": float(loss.item()),
                "recon_loss": float(output.reconstruction_loss.item()),
                "unique_codes": n_unique, "per_level_usage": per_level,
                "elapsed": round(elapsed, 1),
            }
            log.append(entry)
            print(f"  step={step}/{iterations}  loss={loss.item():.4f}  recon={output.reconstruction_loss.item():.4f}  "
                  f"unique={n_unique}/{n_items}  usage={per_level}  {elapsed:.0f}s")

        if step % save_interval == 0:
            ckpt_path = os.path.join(out_dir, f"checkpoint_{step}.pt")
            torch.save({"model": model.state_dict(), "step": step}, ckpt_path)

    # 最终保存
    final_path = os.path.join(out_dir, "checkpoint_final.pt")
    torch.save({"model": model.state_dict(), "step": iterations}, final_path)

    # 导出 cached_ids
    model.eval()
    with torch.no_grad():
        sem_output = model.get_semantic_ids(sas.to(device))
        indices = sem_output[2].cpu().numpy().astype(np.int64)

    code_counter = Counter(tuple(row) for row in indices.tolist())
    n_unique = len(code_counter)
    max_c = code_counter.most_common(1)[0][1]

    code_index = {}
    suffix = np.zeros((n_items, 1), dtype=np.int64)
    for i in range(n_items):
        key = tuple(indices[i].tolist())
        if key not in code_index: code_index[key] = 0
        suffix[i, 0] = code_index[key]
        code_index[key] += 1

    cached_ids = np.concatenate([indices, suffix], axis=1)
    ids_path = os.path.join(out_dir, "collaborative_cached_ids.npy")
    np.save(ids_path, cached_ids)

    print(f"\n[Done] CF RQ-VAE training complete")
    print(f"  Unique: {n_unique}/{n_items}, max collision: {max_c}, max suffix: {suffix.max()}")
    print(f"  Cached IDs: {ids_path} shape={cached_ids.shape}")

    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)

if __name__ == "__main__":
    main()
