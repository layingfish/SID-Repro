"""Extract semantic IDs from RQ-VAE checkpoint -> LETTER index.json"""
import sys, os, json, torch, numpy as np
sys.path.insert(0, "/data/xqp_data/RecSys26/third_party/RQ_VAE_Recommender")

from modules.rqvae import RqVae

def main():
    RQVAE_CKPT = "/data/xqp_data/RecSys26/logs/repro_paper/tiger/yelp_strict/20260313_023500/ckpt/rqvae/checkpoint_19999.pt"
    ITEM_EMB = "/data/xqp_data/RecSys26/datasets/rqvae_recommender/setrec/raw/yelp_strict/item_emb.npy"
    OUTPUT = "/data/xqp_data/RecSys26/third_party/LETTER/data/SETRec_yelp_rqvae/SETRec_yelp_rqvae.index.json"
    device = "cuda"
    codebook_size = 256

    # Build & load RQ-VAE
    model = RqVae(
        input_dim=768, embed_dim=32, hidden_dims=[512, 256, 128],
        codebook_size=codebook_size, codebook_kmeans_init=False,
        codebook_normalize=False, codebook_sim_vq=False,
        n_layers=3, n_cat_features=0, commitment_weight=0.25,
    )
    model.load_pretrained(RQVAE_CKPT)
    model = model.to(device).eval()

    # Load item embeddings
    item_emb = torch.from_numpy(np.load(ITEM_EMB)).float()
    n_items = item_emb.shape[0]
    print(f"Items: {n_items}")

    # Get 3-level semantic IDs
    all_ids = []
    for start in range(0, n_items, 512):
        batch = item_emb[start:start+512].to(device)
        with torch.no_grad():
            out = model.get_semantic_ids(batch)
            all_ids.append(out.sem_ids.cpu())
    cached_ids = torch.cat(all_ids, dim=0)  # (n_items, 3)
    unique3 = len(set(map(tuple, cached_ids.tolist())))
    print(f"3-digit unique: {unique3}/{n_items}")

    # Add dedup column
    id_tuples = [tuple(row.tolist()) for row in cached_ids]
    seen = {}
    dedup = []
    for t in id_tuples:
        count = seen.get(t, 0)
        dedup.append(min(count, codebook_size - 1))
        seen[t] = count + 1
    dedup_tensor = torch.tensor(dedup).unsqueeze(-1)
    full_ids = torch.cat([cached_ids, dedup_tensor], dim=-1)  # (n_items, 4)
    unique4 = len(set(map(tuple, full_ids.tolist())))
    print(f"4-digit unique: {unique4}/{n_items}, max_dedup: {max(dedup)}")

    # Convert to LETTER format
    prefixes = ["a", "b", "c", "d"]
    index = {}
    for i in range(n_items):
        index[str(i)] = [f"<{prefixes[j]}_{full_ids[i][j].item()}>" for j in range(4)]

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(index, f)

    all_tokens = set()
    for v in index.values():
        all_tokens.update(v)
    print(f"Saved {len(index)} items, {len(all_tokens)} unique tokens")

if __name__ == "__main__":
    main()
