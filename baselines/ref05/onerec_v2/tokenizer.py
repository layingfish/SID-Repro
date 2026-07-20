"""
Stage-1 tokenizer for OneRec-style semantic IDs.

Supports both the paper-style balanced RQ-kmeans path and faster FAISS/sklearn
approximations. The paper-aligned defaults follow L=3, K=8192.

Usage:
  python -m onerec_v2.tokenizer \
    --emb_path /path/to/embeddings.npy \
    --output_path /path/to/semantic_ids.pt \
    --L 3 --K 8192 --backend balanced
"""

import argparse
import os

import numpy as np
import torch

from balanced_rqkmeans import residual_quantization as balanced_residual_quantization


def nearest_assign(embeddings, centroids):
    """Assign each item to its nearest centroid."""
    from scipy.spatial.distance import cdist

    distances = cdist(embeddings, centroids, metric='sqeuclidean')
    return distances.argmin(axis=1).astype(np.int64)


def faiss_kmeans(
    embeddings,
    K,
    niter=50,
    nredo=10,
    gpu=False,
    seed=42,
    verbose=True,
    assign_method='nearest',
):
    """Run FAISS KMeans on embeddings."""
    import faiss

    N, D = embeddings.shape
    kmeans = faiss.Kmeans(
        D, K,
        niter=niter,
        nredo=nredo,
        seed=seed,
        verbose=verbose,
        gpu=gpu,
    )
    kmeans.train(embeddings)

    centroids = kmeans.centroids.copy()

    if assign_method == 'sinkhorn':
        assignments = sinkhorn_assign(embeddings, centroids, verbose=verbose)
    else:
        assignments = nearest_assign(embeddings, centroids)

    # Stats
    sizes = np.bincount(assignments, minlength=K)
    active = np.sum(sizes > 0)
    if verbose:
        print(f"  active clusters: {active}/{K} ({active/K*100:.1f}%)")
        print(f"  cluster sizes: min={sizes[sizes>0].min()} max={sizes.max()} "
              f"mean={sizes[sizes>0].mean():.1f} std={sizes[sizes>0].std():.1f}")

    return centroids, assignments


def openonerec_faiss_kmeans(
    embeddings,
    K,
    niter=20,
    gpu=False,
    seed=42,
    verbose=True,
):
    """Match the public OpenOneRec tokenizer training loop as closely as possible."""
    import faiss

    _, D = embeddings.shape
    kmeans = faiss.Kmeans(
        D,
        K,
        niter=niter,
        nredo=1,
        seed=seed,
        verbose=verbose,
        gpu=gpu,
        spherical=False,
    )
    kmeans.train(embeddings.astype(np.float32))
    _, assignments = kmeans.index.search(embeddings.astype(np.float32), 1)
    assignments = assignments.reshape(-1).astype(np.int64)
    centroids = kmeans.centroids.copy()

    sizes = np.bincount(assignments, minlength=K)
    active = np.sum(sizes > 0)
    if verbose:
        print(f"  active clusters: {active}/{K} ({active/K*100:.1f}%)")
        print(f"  cluster sizes: min={sizes[sizes>0].min()} max={sizes.max()} "
              f"mean={sizes[sizes>0].mean():.1f} std={sizes[sizes>0].std():.1f}")

    return centroids, assignments


def sinkhorn_assign(embeddings, centroids, reg=0.1, max_iter=100, verbose=True):
    """Sinkhorn balanced assignment: items -> clusters with uniform capacity.

    Uses Sinkhorn-Knopp algorithm to find a transport plan that:
    - Minimizes total squared distance
    - Ensures each cluster gets approximately N/K items
    """
    N = embeddings.shape[0]
    K = centroids.shape[0]

    # Cost matrix: squared distances [N, K]
    from scipy.spatial.distance import cdist
    C = cdist(embeddings, centroids, metric='sqeuclidean').astype(np.float64)

    # Normalize cost for numerical stability
    C = C / (C.max() + 1e-10)

    # Source: uniform over items (each item has mass 1/N)
    a = np.ones(N) / N
    # Target: uniform over clusters (each cluster has mass 1/K)
    b = np.ones(K) / K

    # Sinkhorn iterations
    # K = exp(-C/reg)
    M = np.exp(-C / reg)

    u = np.ones(N)
    for it in range(max_iter):
        # Scale rows
        v = b / (M.T @ u + 1e-10)
        u_new = a / (M @ v + 1e-10)

        # Check convergence
        if it > 0 and np.max(np.abs(u_new - u)) < 1e-6:
            if verbose:
                print(f"  Sinkhorn converged at iter {it}")
            u = u_new
            break
        u = u_new

    # Transport plan (memory-efficient: avoid N×N diagonal matrix)
    T = (u[:, None] * M) * v[None, :]  # [N, K], element-wise

    # Assign each item to its highest-transport cluster
    assignments = T.argmax(axis=1).astype(np.int64)

    if verbose:
        sizes = np.bincount(assignments, minlength=K)
        print(f"  Sinkhorn: cluster sizes min={sizes.min()} max={sizes.max()} "
              f"std={sizes.std():.1f} (target={N//K})")

    return assignments


def sklearn_kmeans(
    embeddings,
    K,
    max_iter=100,
    n_init=10,
    seed=42,
    verbose=True,
    assign_method='nearest',
):
    """Fallback: sklearn KMeans (if FAISS not available)."""
    from sklearn.cluster import KMeans

    N, D = embeddings.shape
    km = KMeans(
        n_clusters=K,
        max_iter=max_iter,
        n_init=n_init,
        random_state=seed,
        algorithm='lloyd',
        verbose=1 if verbose else 0,
    )
    km.fit(embeddings)

    centroids = km.cluster_centers_.copy()
    if assign_method == 'sinkhorn':
        assignments = sinkhorn_assign(embeddings, centroids, verbose=verbose)
    else:
        assignments = nearest_assign(embeddings, centroids)

    sizes = np.bincount(assignments, minlength=K)
    active = np.sum(sizes > 0)
    if verbose:
        print(f"  active clusters: {active}/{K} ({active/K*100:.1f}%)")
        print(f"  cluster sizes: min={sizes[sizes>0].min()} max={sizes.max()} "
              f"mean={sizes[sizes>0].mean():.1f} std={sizes[sizes>0].std():.1f}")

    return centroids, assignments


def residual_quantization(
    embeddings,
    K,
    L,
    backend='faiss',
    niter=50,
    nredo=10,
    gpu=False,
    seed=42,
    verbose=True,
    assign_method='nearest',
    balanced_search_topk=32,
):
    """Residual Quantization with full KMeans at each level."""
    if backend == 'balanced':
        return balanced_residual_quantization(
            embeddings,
            K=K,
            L=L,
            max_iter=niter,
            seed=seed,
            verbose=verbose,
            search_topk=balanced_search_topk,
        )

    N, D = embeddings.shape
    residual = embeddings.copy()
    all_centroids = []
    all_assignments = np.zeros((L, N), dtype=np.int64)

    for level in range(L):
        norm = np.linalg.norm(residual, axis=1).mean()
        if verbose:
            print(f"\nLevel {level}: residual norm mean = {norm:.6f}")

        if backend == 'faiss':
            centroids, assignments = faiss_kmeans(
                residual.astype(np.float32), K,
                niter=niter,
                nredo=nredo,
                gpu=gpu,
                seed=seed + level,
                verbose=verbose,
                assign_method=assign_method,
            )
        elif backend == 'openonerec':
            centroids, assignments = openonerec_faiss_kmeans(
                residual.astype(np.float32),
                K,
                niter=niter,
                gpu=gpu,
                seed=seed + level,
                verbose=verbose,
            )
        else:
            centroids, assignments = sklearn_kmeans(
                residual,
                K,
                seed=seed + level,
                verbose=verbose,
                assign_method=assign_method,
            )

        all_centroids.append(centroids)
        all_assignments[level] = assignments

        residual = residual - centroids[assignments]

        if verbose:
            print(f"  After quantization: residual norm mean = {np.linalg.norm(residual, axis=1).mean():.6f}")

    return all_centroids, all_assignments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', required=True, help='.npy embedding file')
    parser.add_argument('--output_path', required=True, help='Output .pt file')
    parser.add_argument('--centroids_path', default=None, help='Output centroids .pt')
    parser.add_argument('--L', type=int, default=3, help='Number of hierarchy levels (paper Stage 1 default: 3)')
    parser.add_argument('--K', type=int, default=8192, help='Codebook width per level (paper Stage 1 default: 8192)')
    parser.add_argument('--backend', choices=['balanced', 'faiss', 'openonerec', 'sklearn'], default='balanced')
    parser.add_argument('--assign_method', choices=['nearest', 'sinkhorn'], default='nearest')
    parser.add_argument('--niter', type=int, default=50)
    parser.add_argument('--nredo', type=int, default=10)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--balanced_search_topk', type=int, default=32)
    args = parser.parse_args()

    print(f"Loading embeddings from {args.emb_path}")
    emb = np.load(args.emb_path).astype(np.float32)
    N, D = emb.shape
    print(f"  N={N}, D={D}")
    print(f"  Config: L={args.L}, K={args.K}, backend={args.backend}, assign={args.assign_method}")

    all_centroids, all_assignments = residual_quantization(
        emb, K=args.K, L=args.L, backend=args.backend,
        niter=args.niter, nredo=args.nredo, gpu=args.gpu,
        seed=args.seed, verbose=True, assign_method=args.assign_method,
        balanced_search_topk=args.balanced_search_topk,
    )

    semantic_ids = torch.from_numpy(all_assignments).long()  # [L, N]

    # Collision stats
    K = args.K
    combined = semantic_ids[0].clone()
    for l in range(1, args.L):
        combined = combined * K + semantic_ids[l]
    n_unique = combined.unique().shape[0]
    collision_rate = 1.0 - n_unique / N

    print(f"\n{'='*60}")
    print(f"RQ-KMeans Results ({args.backend}):")
    print(f"  n_items: {N}")
    print(f"  L={args.L}, K={args.K}, capacity={args.K**args.L:,}")
    print(f"  unique {args.L}-level IDs: {n_unique}")
    print(f"  collision rate: {collision_rate:.6f} ({collision_rate*100:.2f}%)")

    for l in range(args.L):
        sizes = np.bincount(all_assignments[l], minlength=K)
        active = np.sum(sizes > 0)
        print(f"  Level {l}: active={active}/{K} ({active/K*100:.1f}%), "
              f"sizes min={sizes[sizes>0].min()} max={sizes.max()}")
    print(f"{'='*60}")

    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    torch.save(semantic_ids, args.output_path)
    print(f"\nSaved semantic IDs to {args.output_path} (shape: {semantic_ids.shape})")

    if args.centroids_path:
        torch.save([torch.from_numpy(c).float() for c in all_centroids], args.centroids_path)

    print(f"\nSample (first 5 items):")
    for i in range(min(5, N)):
        ids = [semantic_ids[j, i].item() for j in range(args.L)]
        print(f"  item {i}: {ids}")


if __name__ == '__main__':
    main()
