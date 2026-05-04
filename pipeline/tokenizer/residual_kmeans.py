

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from collections import Counter

RELEASE_ROOT = Path(__file__).resolve().parents[2]
METRICS_ROOT = RELEASE_ROOT / "experiments" / "rq2"
if str(METRICS_ROOT) not in sys.path:
    sys.path.insert(0, str(METRICS_ROOT))

from tokenizer_metrics_utils import export_tokenizer_metrics


def minibatch_kmeans(
    data: np.ndarray,
    n_clusters: int,
    max_iter: int = 100,
    batch_size: int = 2048,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:


    rng = np.random.RandomState(seed)
    n_samples, n_features = data.shape


    centroids = np.zeros((n_clusters, n_features), dtype=np.float32)
    idx = rng.randint(n_samples)
    centroids[0] = data[idx]

    for k in range(1, n_clusters):

        dists = np.min(
            np.sum((data[:, None, :] - centroids[None, :k, :]) ** 2, axis=2),
            axis=1,
        )

        probs = dists / dists.sum()
        idx = rng.choice(n_samples, p=probs)
        centroids[k] = data[idx]


    cluster_counts = np.ones(n_clusters, dtype=np.float64)

    for it in range(max_iter):

        indices = rng.choice(n_samples, size=min(batch_size, n_samples), replace=False)
        batch = data[indices]


        dists = np.sum((batch[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        assignments = np.argmin(dists, axis=1)


        for k in range(n_clusters):
            mask = assignments == k
            count = mask.sum()
            if count > 0:
                cluster_counts[k] += count
                eta = count / cluster_counts[k]
                centroids[k] = (1 - eta) * centroids[k] + eta * batch[mask].mean(axis=0)


    dists = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    final_assignments = np.argmin(dists, axis=1)

    return centroids, final_assignments


def sklearn_minibatch_kmeans(
    data: np.ndarray,
    n_clusters: int,
    max_iter: int = 100,
    batch_size: int = 2048,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:

    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=3,
        max_iter=max_iter,
        batch_size=min(batch_size, max(n_clusters * 8, data.shape[0])),
        random_state=seed,
        reassignment_ratio=0.01,
        verbose=0,
    )
    assignments = km.fit_predict(data).astype(np.int64)
    centroids = km.cluster_centers_.astype(np.float32)
    return centroids, assignments


def residual_kmeans(
    embeddings: np.ndarray,
    n_layers: int = 3,
    codebook_width: int = 256,
    normalize_residuals: bool = True,
    max_iter: int = 100,
    batch_size: int = 2048,
    seed: int = 42,
    backend: str = "naive",
) -> tuple[np.ndarray, list[np.ndarray]]:


    n_items, emb_dim = embeddings.shape
    residuals = embeddings.copy().astype(np.float32)
    codes = np.zeros((n_items, n_layers), dtype=np.int64)
    centroids_list = []

    for layer in range(n_layers):
        print(f"  Level {layer}/{n_layers}: ", end="", flush=True)


        if normalize_residuals:
            norms = np.linalg.norm(residuals, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            input_data = residuals / norms
        else:
            input_data = residuals


        if backend == "naive":
            centroids, assignments = minibatch_kmeans(
                input_data, n_clusters=codebook_width,
                max_iter=max_iter, batch_size=batch_size,
                seed=seed + layer,
            )
        elif backend == "sklearn":
            centroids, assignments = sklearn_minibatch_kmeans(
                input_data, n_clusters=codebook_width,
                max_iter=max_iter, batch_size=batch_size,
                seed=seed + layer,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

        codes[:, layer] = assignments
        centroids_list.append(centroids)


        quantized = centroids[assignments]
        residuals = input_data - quantized


        n_unique = len(set(assignments.tolist()))
        print(f"usage={n_unique}/{codebook_width}, residual_norm={np.linalg.norm(residuals, axis=1).mean():.4f}", flush=True)

    return codes, centroids_list


def main():
    parser = argparse.ArgumentParser(description="Residual K-means (OneRec style)")
    parser.add_argument("--emb_path", required=True, type=Path, help="Item embedding .npy (N, D)")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--codebook_width", type=int, default=256)
    parser.add_argument(
        "--dedup_mode",
        choices=["suffix", "none"],
        default="suffix",
        help="suffix: append a dedup suffix for unique decoder codes; none: keep raw semantic codes only.",
    )
    parser.add_argument("--normalize_residuals", action="store_true", default=True)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backend",
        choices=["naive", "sklearn", "balanced"],
        default="naive",
        help="K-means backend. balanced matches the OneRec-style balanced RQ-KMeans path.",
    )
    parser.add_argument("--search_topk", type=int, default=32, help="Candidate centroids for balanced assignment.")
    args = parser.parse_args()


    emb = np.load(args.emb_path).astype(np.float32)
    n_items, emb_dim = emb.shape
    print(f"Loaded {n_items} items, {emb_dim}d", flush=True)


    print(f"Running Residual K-means: {args.n_layers} layers × {args.codebook_width} clusters, backend={args.backend}", flush=True)
    if args.backend == "balanced":
        balanced_root = Path(__file__).resolve().parent
        sys.path.insert(0, str(balanced_root))
        from balanced_rqkmeans import residual_quantization as balanced_residual_quantization

        centroids_list, assignments = balanced_residual_quantization(
            emb,
            K=args.codebook_width,
            L=args.n_layers,
            max_iter=args.max_iter,
            seed=args.seed,
            verbose=True,
            search_topk=args.search_topk,
        )
        codes = assignments.T.astype(np.int64)
    else:
        codes, centroids_list = residual_kmeans(
            emb, n_layers=args.n_layers, codebook_width=args.codebook_width,
            normalize_residuals=args.normalize_residuals,
            max_iter=args.max_iter, batch_size=args.batch_size, seed=args.seed,
            backend=args.backend,
        )


    code_counter = Counter(tuple(row) for row in codes.tolist())
    n_unique = len(code_counter)
    max_collision = code_counter.most_common(1)[0][1]
    print(f"\nUnique codes: {n_unique}/{n_items} ({n_unique/n_items*100:.1f}%)")
    print(f"Max collision: {max_collision}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "raw_codes.npy", codes)

    if args.dedup_mode == "suffix":
        code_index = {}
        suffix = np.zeros((n_items, 1), dtype=np.int64)
        for i in range(n_items):
            key = tuple(codes[i].tolist())
            if key not in code_index:
                code_index[key] = 0
            suffix[i, 0] = code_index[key]
            code_index[key] += 1
        cached_ids = np.concatenate([codes, suffix], axis=1)
        duplicate_policy = "unique"
        dedup_suffix = True
        print(f"With dedup suffix: {cached_ids.shape}, max suffix: {suffix.max()}")
    else:
        cached_ids = codes.copy()
        duplicate_policy = "first"
        dedup_suffix = False
        print(f"Raw semantic codes only: {cached_ids.shape}")


    np.save(args.output_dir / "cached_ids.npy", cached_ids)


    for i, c in enumerate(centroids_list):
        np.save(args.output_dir / f"centroids_level_{i}.npy", c)


    per_pos = [len(set(cached_ids[:, pos].tolist())) for pos in range(cached_ids.shape[1])]
    config = {
        "n_items": n_items,
        "sem_id_dim": cached_ids.shape[1],
        "codebook_size": max(per_pos),
        "per_pos_sizes": per_pos,
        "id_type": f"I07-rkmeans-{args.dedup_mode}",
        "id_name": "Balanced RQ-KMeans Code (OneRec)" if args.backend == "balanced" else ("Residual K-means Code (OneRec)" if dedup_suffix else "Residual K-means Raw Semantic Code (OneRec)"),
        "source": (
            f"Balanced RQ-KMeans {args.n_layers}x{args.codebook_width}, search_topk={args.search_topk}"
            if args.backend == "balanced"
            else f"Residual K-means: {args.n_layers} layers × {args.codebook_width}, normalize={args.normalize_residuals}"
        ),
        "n_unique_codes": n_unique,
        "collision_rate": (n_items - n_unique) / n_items,
        "max_collision": int(max_collision),
        "duplicate_policy": duplicate_policy,
        "dedup_suffix": dedup_suffix,
        "semantic_code_length": int(args.n_layers),
        "decoder_code_length": int(cached_ids.shape[1]),
    }
    with open(args.output_dir / "tiger_config.json", "w") as f:
        json.dump(config, f, indent=2)

    meta = {
        "method_family": "RQ-Kmeans",
        "paper_variant": "OneRec-paper-claim",
        "implementation_variant": "local-residual-kmeans-reproduction",
        "embedding_source": str(args.emb_path),
        "embedding_dim": emb_dim,
        "n_items": n_items,
        "n_levels": args.n_layers,
        "codebook_size_per_level": [args.codebook_width] * args.n_layers,
        "balanced_assignment": False,
        "dedup_suffix": dedup_suffix,
        "tokenizer_ckpt_rule": "final",
        "normalize_residuals": args.normalize_residuals,
        "max_iter": args.max_iter,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "backend": args.backend,
        "search_topk": args.search_topk,
        "id_type": f"I07-rkmeans-{args.dedup_mode}",
        "duplicate_policy": duplicate_policy,
        "semantic_code_length": int(args.n_layers),
    }
    summary = export_tokenizer_metrics(
        output_dir=args.output_dir,
        codes=cached_ids,
        meta=meta,
        codebook_size_per_level=(
            [args.codebook_width] * args.n_layers
            if not dedup_suffix
            else [args.codebook_width] * args.n_layers + [int(cached_ids[:, -1].max()) + 1]
        ),
    )
    print(
        "Tokenizer metrics: first_level_used_codes={used} usage_rate={rate:.4f} collision_rate={cr:.6f}".format(
            used=summary["first_level_used_codes"],
            rate=summary["first_level_usage_rate"],
            cr=summary["collision_rate"],
        )
    )

    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
