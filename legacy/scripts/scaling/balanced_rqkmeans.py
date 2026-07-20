#!/usr/bin/env python3
"""
Balanced RQ-KMeans tokenizer (OneRec Algorithm 1).

Key difference from standard k-means: each cluster is forced to have exactly
w = |V| / K items, eliminating the "hourglass phenomenon" where some clusters
are overcrowded and others are empty.

Usage:
  python balanced_rqkmeans.py \
    --emb_path /path/to/embeddings.npy \
    --output_path /path/to/semantic_ids.pt \
    --num_hierarchies 3 \
    --codebook_width 256 \
    --max_iter 50
"""

import argparse
import os

import numpy as np
import torch
from scipy.spatial.distance import cdist


def balanced_kmeans(embeddings, K, max_iter=50, seed=42, verbose=True):
    """
    Balanced K-Means (OneRec Algorithm 1).

    Forces each cluster to have exactly w = N // K items.
    Remaining N % K items go to the last cluster.

    Args:
        embeddings: np.ndarray of shape [N, D]
        K: number of clusters
        max_iter: max iterations
        seed: random seed
        verbose: print progress

    Returns:
        centroids: np.ndarray [K, D]
        assignments: np.ndarray [N] (int, cluster index for each item)
    """
    N, D = embeddings.shape
    w = N // K  # base items per cluster
    remainder = N % K  # first `remainder` clusters get w+1 items
    rng = np.random.RandomState(seed)

    # Initialize centroids with K random items
    indices = rng.choice(N, size=K, replace=False)
    centroids = embeddings[indices].copy()

    assignments = np.zeros(N, dtype=np.int64)

    # Pre-compute cluster capacities: first `remainder` clusters get w+1, rest get w
    capacities = np.array([w + 1 if k < remainder else w for k in range(K)])

    for it in range(max_iter):
        old_assignments = assignments.copy()

        # Compute all pairwise distances: [N, K]
        dists = cdist(embeddings, centroids, metric='sqeuclidean')

        # --- Balanced assignment via linear assignment approximation ---
        # For each cluster k (in random order), greedily assign its w nearest
        # unassigned items. This is the approach from OneRec Algorithm 1.
        assigned = np.zeros(N, dtype=bool)
        assignments[:] = -1

        # Sort clusters by their "tightness" (sum of distances to nearest w items)
        # Process tighter clusters first to reduce overall distortion
        cluster_order = rng.permutation(K)

        for k in cluster_order:
            unassigned_idx = np.where(~assigned)[0]
            if len(unassigned_idx) == 0:
                break

            cap = min(capacities[k], len(unassigned_idx))
            d = dists[unassigned_idx, k]
            if cap >= len(d):
                nearest = np.arange(len(d))
            else:
                nearest = np.argpartition(d, cap)[:cap]
            selected = unassigned_idx[nearest]

            assignments[selected] = k
            assigned[selected] = True

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for k in range(K):
            mask = assignments == k
            if np.any(mask):
                new_centroids[k] = embeddings[mask].mean(axis=0)
            else:
                new_centroids[k] = centroids[k]
        centroids = new_centroids

        # Check convergence
        changed = np.sum(assignments != old_assignments)
        if verbose and (it % 10 == 0 or it == max_iter - 1):
            mse = np.mean(dists[np.arange(N), assignments])
            sizes = np.bincount(assignments, minlength=K)
            print(f"  iter {it:3d}: changed={changed}, MSE={mse:.6f}, "
                  f"cluster_size min={sizes.min()} max={sizes.max()} "
                  f"std={sizes.std():.1f}")

        if changed == 0:
            if verbose:
                print(f"  Converged at iteration {it}")
            break

    return centroids, assignments


def residual_quantization(embeddings, K, L, max_iter=50, seed=42, verbose=True):
    """
    Residual Quantization with Balanced K-Means at each level.

    Args:
        embeddings: np.ndarray [N, D]
        K: codebook width per level
        L: number of hierarchy levels
        max_iter: max iterations per level
        seed: random seed

    Returns:
        all_centroids: list of L np.ndarray [K, D]
        all_assignments: np.ndarray [L, N]
    """
    N, D = embeddings.shape
    residual = embeddings.copy()
    all_centroids = []
    all_assignments = np.zeros((L, N), dtype=np.int64)

    for level in range(L):
        if verbose:
            print(f"\nLevel {level}: residual norm mean = {np.linalg.norm(residual, axis=1).mean():.6f}")

        centroids, assignments = balanced_kmeans(
            residual, K, max_iter=max_iter, seed=seed + level, verbose=verbose
        )

        all_centroids.append(centroids)
        all_assignments[level] = assignments

        # Update residual
        residual = residual - centroids[assignments]

        if verbose:
            print(f"  After quantization: residual norm mean = {np.linalg.norm(residual, axis=1).mean():.6f}")

    return all_centroids, all_assignments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', required=True, help='.npy embedding file')
    parser.add_argument('--output_path', required=True, help='Output .pt file for semantic IDs')
    parser.add_argument('--centroids_path', default=None, help='Output .pt file for centroids (optional)')
    parser.add_argument('--num_hierarchies', type=int, default=3)
    parser.add_argument('--codebook_width', type=int, default=256)
    parser.add_argument('--max_iter', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f"Loading embeddings from {args.emb_path}")
    emb = np.load(args.emb_path).astype(np.float32)
    N, D = emb.shape
    print(f"  N={N}, D={D}")
    print(f"  Config: L={args.num_hierarchies}, K={args.codebook_width}")

    print("\nRunning Balanced RQ-KMeans...")
    all_centroids, all_assignments = residual_quantization(
        emb, K=args.codebook_width, L=args.num_hierarchies,
        max_iter=args.max_iter, seed=args.seed, verbose=True
    )

    # Verify uniqueness
    semantic_ids = torch.from_numpy(all_assignments).long()  # [L, N]
    K = args.codebook_width
    if args.num_hierarchies == 3:
        combined = semantic_ids[0] * K * K + semantic_ids[1] * K + semantic_ids[2]
    else:
        combined = semantic_ids[0]
        for l in range(1, args.num_hierarchies):
            combined = combined * K + semantic_ids[l]
    n_unique = combined.unique().shape[0]
    collision_rate = 1.0 - n_unique / N

    print(f"\n{'='*60}")
    print(f"Balanced RQ-KMeans Results:")
    print(f"  n_items: {N}")
    print(f"  unique 3-level IDs: {n_unique}")
    print(f"  collision rate: {collision_rate:.6f} ({collision_rate*100:.2f}%)")

    # Cluster size stats
    for l in range(args.num_hierarchies):
        sizes = np.bincount(all_assignments[l], minlength=K)
        active = np.sum(sizes > 0)
        print(f"  Level {l}: active clusters={active}/{K} ({active/K*100:.1f}%), "
              f"sizes min={sizes[sizes>0].min()} max={sizes.max()} std={sizes[sizes>0].std():.1f}")
    print(f"{'='*60}")

    # Save semantic IDs
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    torch.save(semantic_ids, args.output_path)
    print(f"\nSaved semantic IDs to {args.output_path} (shape: {semantic_ids.shape})")

    # Optionally save centroids
    if args.centroids_path:
        centroid_tensors = [torch.from_numpy(c).float() for c in all_centroids]
        torch.save(centroid_tensors, args.centroids_path)
        print(f"Saved centroids to {args.centroids_path}")

    # Sample output
    print(f"\nSample (first 10 items):")
    for i in range(min(10, N)):
        ids = [semantic_ids[j, i].item() for j in range(args.num_hierarchies)]
        print(f"  item {i}: {ids}")


if __name__ == '__main__':
    main()
