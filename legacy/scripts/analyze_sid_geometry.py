#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np


def _load_embeddings(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix == ".pkl":
        with path.open("rb") as f:
            arr = pickle.load(f)
    else:
        raise ValueError(f"Unsupported reference embedding format: {path}")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape={arr.shape} from {path}")
    return arr


def _load_codes(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path)
        arr = np.asarray(arr, dtype=np.int64)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D codes, got shape={arr.shape} from {path}")
        return arr
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Unsupported code JSON format: {path}")
        rows: list[list[int]] = []
        for _, seq in sorted(
            data.items(),
            key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else str(kv[0]),
        ):
            norm: list[int] = []
            for token in seq:
                if isinstance(token, int):
                    norm.append(token)
                    continue
                token = str(token)
                if token.startswith("<") and token.endswith(">") and "_" in token:
                    norm.append(int(token.split("_")[-1].rstrip(">")))
                else:
                    norm.append(int(token))
            rows.append(norm)
        return np.asarray(rows, dtype=np.int64)
    raise ValueError(f"Unsupported code format: {path}")


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float((x * x).sum() * (y * y).sum()))
    if denom <= 0:
        return 0.0
    return float((x * y).sum() / denom)


def _rankdata_average(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.shape[0], dtype=np.float64)
    sorted_a = a[order]
    n = a.shape[0]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_a[j] == sorted_a[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rankdata_average(x), _rankdata_average(y))


def _sample_pairs(n_items: int, n_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    left = rng.integers(0, n_items, size=n_pairs, endpoint=False)
    right = rng.integers(0, n_items - 1, size=n_pairs, endpoint=False)
    right = right + (right >= left)
    return left.astype(np.int64), right.astype(np.int64)


def _sample_triplets(n_items: int, n_triplets: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    anchor = rng.integers(0, n_items, size=n_triplets, endpoint=False)
    j = rng.integers(0, n_items - 1, size=n_triplets, endpoint=False)
    j = j + (j >= anchor)
    k = rng.integers(0, n_items - 2, size=n_triplets, endpoint=False)
    low = np.minimum(anchor, j)
    high = np.maximum(anchor, j)
    k = k + (k >= low)
    k = k + (k >= high - 1)
    k = np.where(k == j, (k + 1) % n_items, k)
    return anchor.astype(np.int64), j.astype(np.int64), k.astype(np.int64)


def _hamming_distance_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a != b).mean(axis=1, dtype=np.float64)


def _prefix_distance_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError(f"Prefix distance shape mismatch: {a.shape} vs {b.shape}")
    same = a == b
    all_equal = same.all(axis=1)
    first_diff = (~same).argmax(axis=1)
    lcp = first_diff.astype(np.int64)
    lcp[all_equal] = a.shape[1]
    return 1.0 - (lcp.astype(np.float64) / float(a.shape[1]))


def _pair_distances(codes: np.ndarray, left: np.ndarray, right: np.ndarray) -> dict[str, np.ndarray]:
    a = codes[left]
    b = codes[right]
    return {
        "hamming": _hamming_distance_rows(a, b),
        "prefix": _prefix_distance_rows(a, b),
    }


def _all_sid_distances(anchor_codes: np.ndarray, codes: np.ndarray, metric: str) -> np.ndarray:
    if metric == "hamming":
        return (codes != anchor_codes[None, :]).mean(axis=1, dtype=np.float64)
    if metric == "prefix":
        same = codes == anchor_codes[None, :]
        all_equal = same.all(axis=1)
        first_diff = (~same).argmax(axis=1)
        lcp = first_diff.astype(np.int64)
        lcp[all_equal] = codes.shape[1]
        return 1.0 - (lcp.astype(np.float64) / float(codes.shape[1]))
    raise ValueError(f"Unsupported SID distance metric: {metric}")


def _topk_reference_neighbors(
    embeddings_norm: np.ndarray,
    anchors: np.ndarray,
    k: int,
) -> np.ndarray:
    sims = embeddings_norm[anchors] @ embeddings_norm.T
    rows = np.arange(anchors.shape[0])
    sims[rows, anchors] = -np.inf
    result = np.empty((anchors.shape[0], k), dtype=np.int64)
    for idx in range(anchors.shape[0]):
        order = np.argsort(-sims[idx], kind="mergesort")
        result[idx] = order[:k]
    return result


def _topk_sid_neighbors(
    codes: np.ndarray,
    anchors: np.ndarray,
    k: int,
    metric: str,
) -> np.ndarray:
    item_ids = np.arange(codes.shape[0], dtype=np.int64)
    result = np.empty((anchors.shape[0], k), dtype=np.int64)
    for row_idx, anchor in enumerate(anchors.tolist()):
        dist = _all_sid_distances(codes[anchor], codes, metric)
        dist[anchor] = np.inf
        order = np.lexsort((item_ids, dist))
        result[row_idx] = order[:k]
    return result


def _triplet_accuracy(
    ref_ij: np.ndarray,
    ref_ik: np.ndarray,
    sid_ij: np.ndarray,
    sid_ik: np.ndarray,
) -> float:
    ref_sign = np.sign(ref_ij - ref_ik)
    sid_sign = np.sign(sid_ij - sid_ik)
    exact = sid_sign == ref_sign
    ties = sid_sign == 0
    score = exact.astype(np.float64) + 0.5 * ties.astype(np.float64)
    return float(score.mean())


def _jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    union = sa | sb
    if not union:
        return 0.0
    return float(len(sa & sb) / len(union))


def _recall(a: Iterable[int], b: Iterable[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    if not sa:
        return 0.0
    return float(len(sa & sb) / len(sa))


def _rank_corr_for_anchor(
    emb_dist: np.ndarray,
    sid_dist: np.ndarray,
    top_m: int,
) -> float:
    order = np.argsort(emb_dist, kind="mergesort")
    order = order[:top_m]
    return _spearman(emb_dist[order], sid_dist[order])


def _pca_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    if u.shape[1] < 2:
        pad = np.zeros((x.shape[0], 2), dtype=np.float64)
        pad[:, : u.shape[1]] = u * s
        return pad
    return (u[:, :2] * s[:2]).astype(np.float64)


def _svg_circle(cx: float, cy: float, r: float, fill: str, opacity: float = 1.0, stroke: str = "none", stroke_width: float = 0.0) -> str:
    return (
        f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" '
        f'fill-opacity="{opacity:.3f}" stroke="{stroke}" stroke-width="{stroke_width:.2f}" />'
    )


def _svg_polygon(points: list[tuple[float, float]], fill: str, opacity: float = 1.0, stroke: str = "none", stroke_width: float = 0.0) -> str:
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (
        f'<polygon points="{pts}" fill="{fill}" fill-opacity="{opacity:.3f}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.2f}" />'
    )


def _svg_triangle(cx: float, cy: float, size: float, fill: str, opacity: float = 1.0, stroke: str = "none", stroke_width: float = 0.0) -> str:
    h = size * math.sqrt(3.0) / 2.0
    points = [
        (cx, cy - 2.0 * h / 3.0),
        (cx - size / 2.0, cy + h / 3.0),
        (cx + size / 2.0, cy + h / 3.0),
    ]
    return _svg_polygon(points, fill=fill, opacity=opacity, stroke=stroke, stroke_width=stroke_width)


def _svg_diamond(cx: float, cy: float, size: float, fill: str, opacity: float = 1.0, stroke: str = "none", stroke_width: float = 0.0) -> str:
    half = size / 2.0
    points = [
        (cx, cy - half),
        (cx - half, cy),
        (cx, cy + half),
        (cx + half, cy),
    ]
    return _svg_polygon(points, fill=fill, opacity=opacity, stroke=stroke, stroke_width=stroke_width)


def _svg_star(cx: float, cy: float, r_outer: float, fill: str, stroke: str = "white", stroke_width: float = 0.8) -> str:
    r_inner = r_outer * 0.45
    points: list[tuple[float, float]] = []
    for i in range(10):
        angle = -math.pi / 2.0 + i * math.pi / 5.0
        radius = r_outer if i % 2 == 0 else r_inner
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return _svg_polygon(points, fill=fill, opacity=1.0, stroke=stroke, stroke_width=stroke_width)


def _scale_points(coords: np.ndarray, x0: float, y0: float, width: float, height: float, pad: float) -> np.ndarray:
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    xr = max(float(xmax - xmin), 1e-6)
    yr = max(float(ymax - ymin), 1e-6)
    usable_w = max(width - 2.0 * pad, 1.0)
    usable_h = max(height - 2.0 * pad, 1.0)
    scale = min(usable_w / xr, usable_h / yr)
    x_mid = 0.5 * (xmin + xmax)
    y_mid = 0.5 * (ymin + ymax)
    out = np.empty_like(coords, dtype=np.float64)
    out[:, 0] = x0 + width / 2.0 + (coords[:, 0] - x_mid) * scale
    out[:, 1] = y0 + height / 2.0 - (coords[:, 1] - y_mid) * scale
    return out


def _try_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


@dataclass
class SidSpec:
    name: str
    path: Path
    codes: np.ndarray


def _parse_sid_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid --sid spec: {spec}. Expected NAME=PATH")
    name, raw_path = spec.split("=", 1)
    return name.strip(), Path(raw_path.strip())


def _save_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    std = float(values.std())
    if std <= 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - float(values.mean())) / std


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze how well different semantic IDs preserve the native item geometry."
    )
    ap.add_argument("--reference_embeddings", type=Path, required=True)
    ap.add_argument("--sid", action="append", required=True, help="NAME=PATH_TO_cached_ids.npy")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--pair_samples", type=int, default=100000)
    ap.add_argument("--triplet_samples", type=int, default=50000)
    ap.add_argument("--anchor_samples", type=int, default=128)
    ap.add_argument("--neighbors_k", type=int, default=20)
    ap.add_argument("--anchor_rank_topm", type=int, default=200)
    ap.add_argument("--plot_anchors", type=int, default=3)
    ap.add_argument("--plot_context_k", type=int, default=120)
    ap.add_argument(
        "--plot_anchor_item_ids",
        type=str,
        default="",
        help="Optional comma-separated anchor item ids to plot directly. Overrides auto-selected anchors for visualization only.",
    )
    ap.add_argument(
        "--plot_method_order",
        type=str,
        default="",
        help="Optional comma-separated method order for plotting only.",
    )
    ap.add_argument(
        "--paper_sid_metric",
        type=str,
        default="hamming",
        choices=["hamming", "prefix"],
        help="Metric used for the paper-ready main table and case-study selection.",
    )
    ap.add_argument("--seed", type=int, default=20260423)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference = _load_embeddings(args.reference_embeddings)
    reference_norm = _normalize_rows(reference.astype(np.float32))
    n_items = reference.shape[0]

    sid_specs: list[SidSpec] = []
    for raw_spec in args.sid:
        name, path = _parse_sid_spec(raw_spec)
        codes = _load_codes(path)
        if codes.shape[0] != n_items:
            raise ValueError(
                f"{name}: item count mismatch. codes={codes.shape[0]} reference={n_items} path={path}"
            )
        sid_specs.append(SidSpec(name=name, path=path, codes=codes))

    left, right = _sample_pairs(n_items=n_items, n_pairs=args.pair_samples, seed=args.seed)
    ref_pair_dist = 1.0 - np.sum(reference_norm[left] * reference_norm[right], axis=1, dtype=np.float64)

    anchors_rng = np.random.default_rng(args.seed + 1)
    anchors = anchors_rng.choice(n_items, size=min(args.anchor_samples, n_items), replace=False)
    ref_neighbors = _topk_reference_neighbors(reference_norm, anchors=anchors, k=args.neighbors_k)

    triplet_anchor, triplet_j, triplet_k = _sample_triplets(
        n_items=n_items,
        n_triplets=args.triplet_samples,
        seed=args.seed + 2,
    )
    ref_triplet_ij = 1.0 - np.sum(
        reference_norm[triplet_anchor] * reference_norm[triplet_j],
        axis=1,
        dtype=np.float64,
    )
    ref_triplet_ik = 1.0 - np.sum(
        reference_norm[triplet_anchor] * reference_norm[triplet_k],
        axis=1,
        dtype=np.float64,
    )

    summary_rows: list[dict[str, object]] = []
    anchor_rows: list[dict[str, object]] = []
    plot_cache: dict[str, np.ndarray] = {}
    method_order = [sid_spec.name for sid_spec in sid_specs]

    for sid_spec in sid_specs:
        pair_dist = _pair_distances(sid_spec.codes, left=left, right=right)
        sid_neighbors_h = _topk_sid_neighbors(
            codes=sid_spec.codes,
            anchors=anchors,
            k=args.neighbors_k,
            metric="hamming",
        )
        sid_neighbors_p = _topk_sid_neighbors(
            codes=sid_spec.codes,
            anchors=anchors,
            k=args.neighbors_k,
            metric="prefix",
        )
        triplet_h_ij = _hamming_distance_rows(sid_spec.codes[triplet_anchor], sid_spec.codes[triplet_j])
        triplet_h_ik = _hamming_distance_rows(sid_spec.codes[triplet_anchor], sid_spec.codes[triplet_k])
        triplet_p_ij = _prefix_distance_rows(sid_spec.codes[triplet_anchor], sid_spec.codes[triplet_j])
        triplet_p_ik = _prefix_distance_rows(sid_spec.codes[triplet_anchor], sid_spec.codes[triplet_k])

        for metric_name, sid_distances, sid_neighbors, sid_tij, sid_tik in (
            ("hamming", pair_dist["hamming"], sid_neighbors_h, triplet_h_ij, triplet_h_ik),
            ("prefix", pair_dist["prefix"], sid_neighbors_p, triplet_p_ij, triplet_p_ik),
        ):
            jaccards: list[float] = []
            recalls: list[float] = []
            anchor_corrs: list[float] = []
            for anchor_idx, anchor in enumerate(anchors.tolist()):
                ref_set = ref_neighbors[anchor_idx]
                sid_set = sid_neighbors[anchor_idx]
                j = _jaccard(ref_set, sid_set)
                r = _recall(ref_set, sid_set)
                jaccards.append(j)
                recalls.append(r)

                emb_dist_all = 1.0 - (reference_norm[anchor] @ reference_norm.T).astype(np.float64)
                emb_dist_all[anchor] = np.inf
                sid_dist_all = _all_sid_distances(sid_spec.codes[anchor], sid_spec.codes, metric_name)
                sid_dist_all[anchor] = np.inf
                anchor_corr = _rank_corr_for_anchor(
                    emb_dist=emb_dist_all,
                    sid_dist=sid_dist_all,
                    top_m=min(args.anchor_rank_topm, n_items - 1),
                )
                anchor_corrs.append(anchor_corr)
                anchor_rows.append(
                    {
                        "method": sid_spec.name,
                        "id_metric": metric_name,
                        "anchor_item_id": anchor,
                        "jaccard_at_k": round(j, 6),
                        "recall_at_k": round(r, 6),
                        "local_rank_spearman": round(anchor_corr, 6),
                    }
                )

            summary_rows.append(
                {
                    "method": sid_spec.name,
                    "id_metric": metric_name,
                    "reference_embedding_path": str(args.reference_embeddings),
                    "sid_path": str(sid_spec.path),
                    "n_items": n_items,
                    "code_length": int(sid_spec.codes.shape[1]),
                    "pair_samples": int(args.pair_samples),
                    "triplet_samples": int(args.triplet_samples),
                    "anchor_samples": int(anchors.shape[0]),
                    "neighbors_k": int(args.neighbors_k),
                    "pearson_distance_corr": round(_pearson(ref_pair_dist, sid_distances), 6),
                    "spearman_distance_corr": round(_spearman(ref_pair_dist, sid_distances), 6),
                    "mean_jaccard_at_k": round(float(np.mean(jaccards)), 6),
                    "mean_recall_at_k": round(float(np.mean(recalls)), 6),
                    "mean_local_rank_spearman": round(float(np.mean(anchor_corrs)), 6),
                    "triplet_accuracy": round(
                        _triplet_accuracy(ref_triplet_ij, ref_triplet_ik, sid_tij, sid_tik),
                        6,
                    ),
                }
            )
            if metric_name == "hamming":
                plot_cache[sid_spec.name] = sid_neighbors

    summary_rows.sort(key=lambda row: (str(row["id_metric"]), -float(row["spearman_distance_corr"])))
    _save_csv(
        args.output_dir / "summary_metrics.csv",
        summary_rows,
        fieldnames=[
            "method",
            "id_metric",
            "reference_embedding_path",
            "sid_path",
            "n_items",
            "code_length",
            "pair_samples",
            "triplet_samples",
            "anchor_samples",
            "neighbors_k",
            "pearson_distance_corr",
            "spearman_distance_corr",
            "mean_jaccard_at_k",
            "mean_recall_at_k",
            "mean_local_rank_spearman",
            "triplet_accuracy",
        ],
    )
    _save_csv(
        args.output_dir / "per_anchor_metrics.csv",
        anchor_rows,
        fieldnames=[
            "method",
            "id_metric",
            "anchor_item_id",
            "jaccard_at_k",
            "recall_at_k",
            "local_rank_spearman",
        ],
    )

    summary_json = {
        "reference_embeddings": str(args.reference_embeddings),
        "n_items": n_items,
        "pair_samples": int(args.pair_samples),
        "triplet_samples": int(args.triplet_samples),
        "anchor_samples": int(anchors.shape[0]),
        "neighbors_k": int(args.neighbors_k),
        "methods": summary_rows,
    }
    (args.output_dir / "summary_metrics.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    paper_rows = [row for row in summary_rows if row["id_metric"] == args.paper_sid_metric]
    if paper_rows:
        paper_rows_by_method = {str(row["method"]): row for row in paper_rows}
        paper_table_rows: list[dict[str, object]] = []
        for method in method_order:
            if method not in paper_rows_by_method:
                continue
            row = paper_rows_by_method[method]
            paper_table_rows.append(
                {
                    "method": method,
                    "id_metric": args.paper_sid_metric,
                    "spearman_distance_corr": float(row["spearman_distance_corr"]),
                    "jaccard_overlap_at_k": float(row["mean_jaccard_at_k"]),
                }
            )
        _save_csv(
            args.output_dir / "paper_main_table.csv",
            paper_table_rows,
            fieldnames=[
                "method",
                "id_metric",
                "spearman_distance_corr",
                "jaccard_overlap_at_k",
            ],
        )

    if paper_rows:
        top_method_order = [method for method in method_order if method in {str(row["method"]) for row in paper_rows}]
        if args.plot_method_order.strip():
            requested_order = [x.strip() for x in args.plot_method_order.split(",") if x.strip()]
            ordered = [method for method in requested_order if method in top_method_order]
            remaining = [method for method in top_method_order if method not in ordered]
            top_method_order = ordered + remaining
        per_anchor = {method: {} for method in top_method_order}
        per_anchor_rank = {method: {} for method in top_method_order}
        for row in anchor_rows:
            if row["id_metric"] != args.paper_sid_metric:
                continue
            method = str(row["method"])
            anchor_id = int(row["anchor_item_id"])
            per_anchor[method][anchor_id] = float(row["jaccard_at_k"])
            per_anchor_rank[method][anchor_id] = float(row["local_rank_spearman"])

        global_jaccard = np.asarray(
            [float(paper_rows_by_method[method]["mean_jaccard_at_k"]) for method in top_method_order],
            dtype=np.float64,
        )
        global_rank = np.asarray(
            [float(paper_rows_by_method[method]["spearman_distance_corr"]) for method in top_method_order],
            dtype=np.float64,
        )

        anchor_candidate_rows: list[dict[str, object]] = []
        for anchor in anchors.tolist():
            j_vals = np.asarray([per_anchor[m].get(anchor, 0.0) for m in top_method_order], dtype=np.float64)
            s_vals = np.asarray([per_anchor_rank[m].get(anchor, 0.0) for m in top_method_order], dtype=np.float64)
            anchor_candidate_rows.append(
                {
                    "anchor_item_id": int(anchor),
                    "jaccard_rank_alignment": float(_spearman(j_vals, global_jaccard)),
                    "local_spearman_rank_alignment": float(_spearman(s_vals, global_rank)),
                    "jaccard_gap": float(j_vals.max() - j_vals.min()),
                    "local_spearman_gap": float(s_vals.max() - s_vals.min()),
                    "mean_jaccard": float(j_vals.mean()),
                    "mean_local_spearman": float(s_vals.mean()),
                }
            )

        if anchor_candidate_rows:
            global_mean_j = float(global_jaccard.mean())
            global_mean_s = float(global_rank.mean())
            j_gap_z = _zscore(np.asarray([row["jaccard_gap"] for row in anchor_candidate_rows], dtype=np.float64))
            s_gap_z = _zscore(np.asarray([row["local_spearman_gap"] for row in anchor_candidate_rows], dtype=np.float64))
            typicality_penalty = _zscore(
                np.asarray(
                    [
                        abs(float(row["mean_jaccard"]) - global_mean_j)
                        + 0.5 * abs(float(row["mean_local_spearman"]) - global_mean_s)
                        for row in anchor_candidate_rows
                    ],
                    dtype=np.float64,
                )
            )
            for idx, row in enumerate(anchor_candidate_rows):
                rank_consistency = 0.6 * float(row["jaccard_rank_alignment"]) + 0.4 * float(
                    row["local_spearman_rank_alignment"]
                )
                selection_score = rank_consistency + 0.30 * float(j_gap_z[idx]) + 0.10 * float(s_gap_z[idx]) - 0.15 * float(
                    typicality_penalty[idx]
                )
                row["selection_score"] = float(selection_score)

        anchor_candidate_rows.sort(
            key=lambda row: (
                -float(row.get("selection_score", 0.0)),
                -float(row["jaccard_rank_alignment"]),
                -float(row["jaccard_gap"]),
                int(row["anchor_item_id"]),
            )
        )
        _save_csv(
            args.output_dir / "case_anchor_candidates.csv",
            anchor_candidate_rows,
            fieldnames=[
                "anchor_item_id",
                "selection_score",
                "jaccard_rank_alignment",
                "local_spearman_rank_alignment",
                "jaccard_gap",
                "local_spearman_gap",
                "mean_jaccard",
                "mean_local_spearman",
            ],
        )
        if anchor_candidate_rows:
            (args.output_dir / "selected_case_anchor.json").write_text(
                json.dumps(anchor_candidate_rows[0], ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        n_plot = min(args.plot_anchors, len(anchor_candidate_rows))
        if n_plot > 0:
            if args.plot_anchor_item_ids.strip():
                chosen_anchors = [
                    int(x.strip()) for x in args.plot_anchor_item_ids.split(",") if x.strip()
                ]
            else:
                chosen_anchors = [int(row["anchor_item_id"]) for row in anchor_candidate_rows[:n_plot]]
            cell_w = 320.0
            cell_h = 300.0
            cell_gap_x = 22.0
            cell_gap_y = 44.0
            header_h = 74.0
            legend_h = 46.0
            total_w = 24.0 + len(top_method_order) * cell_w + (len(top_method_order) - 1) * cell_gap_x + 24.0
            total_h = header_h + n_plot * cell_h + (n_plot - 1) * cell_gap_y + legend_h + 18.0
            svg_parts: list[str] = [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w:.0f}" height="{total_h:.0f}" viewBox="0 0 {total_w:.0f} {total_h:.0f}">',
                f'<rect x="0" y="0" width="{total_w:.0f}" height="{total_h:.0f}" fill="white" />',
                f'<text x="{total_w / 2.0:.2f}" y="28" text-anchor="middle" font-size="22" font-family="Arial, Helvetica, sans-serif" font-weight="700">RQ4: SID space vs. native semantic space</text>',
                f'<text x="{total_w / 2.0:.2f}" y="52" text-anchor="middle" font-size="13" font-family="Arial, Helvetica, sans-serif" fill="#444">Representative anchor selected by global-rank consistency + local contrast; red = reference semantic neighbors, blue = SID neighbors, purple = overlap</text>',
            ]
            matplotlib_cache: list[dict[str, object]] = []

            legend_x = 28.0
            legend_y = total_h - 18.0
            legend_items = [
                ("circle", "#314a9b", "SID top-K"),
                ("triangle", "#f04b4b", "Reference top-K"),
                ("diamond", "#7a3db8", "Overlap"),
                ("circle", "#d9d9d9", "Local context"),
                ("star", "#000000", "Anchor item"),
            ]
            lx = legend_x
            for shape, color, label in legend_items:
                cy = legend_y - 8.0
                if shape == "circle":
                    svg_parts.append(_svg_circle(lx + 7.0, cy, 5.2, fill=color, opacity=0.95, stroke="white", stroke_width=0.4))
                elif shape == "triangle":
                    svg_parts.append(_svg_triangle(lx + 7.0, cy, 12.0, fill=color, opacity=0.95, stroke="white", stroke_width=0.4))
                elif shape == "diamond":
                    svg_parts.append(_svg_diamond(lx + 7.0, cy, 11.0, fill=color, opacity=0.95, stroke="white", stroke_width=0.4))
                elif shape == "star":
                    svg_parts.append(_svg_star(lx + 7.0, cy, 7.5, fill=color))
                svg_parts.append(
                    f'<text x="{lx + 18.0:.2f}" y="{legend_y - 3.0:.2f}" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#222">{html.escape(label)}</text>'
                )
                lx += 104.0

            for row_idx, anchor in enumerate(chosen_anchors):
                ref_all_order = np.argsort(
                    1.0 - (reference_norm[anchor] @ reference_norm.T),
                    kind="mergesort",
                )
                ref_all_order = ref_all_order[ref_all_order != anchor]
                context_ids = ref_all_order[: max(args.plot_context_k, args.neighbors_k)]
                ref_topk = ref_all_order[: args.neighbors_k]
                union_ids = {anchor}
                union_ids.update(context_ids.tolist())
                union_ids.update(ref_topk.tolist())
                for method in top_method_order:
                    sid_topk_for_anchor = _topk_sid_neighbors(
                        codes=sid_specs[[spec.name for spec in sid_specs].index(method)].codes,
                        anchors=np.asarray([anchor], dtype=np.int64),
                        k=args.neighbors_k,
                        metric=args.paper_sid_metric,
                    )[0]
                    union_ids.update(sid_topk_for_anchor.tolist())
                union_ids_sorted = np.asarray(sorted(union_ids), dtype=np.int64)
                coords = _pca_2d(reference[union_ids_sorted])
                id_to_pos = {int(item_id): idx for idx, item_id in enumerate(union_ids_sorted.tolist())}
                for col_idx, method in enumerate(top_method_order):
                    sid_topk = _topk_sid_neighbors(
                        codes=sid_specs[[spec.name for spec in sid_specs].index(method)].codes,
                        anchors=np.asarray([anchor], dtype=np.int64),
                        k=args.neighbors_k,
                        metric=args.paper_sid_metric,
                    )[0]
                    sid_positions = {id_to_pos[x] for x in sid_topk.tolist() if x in id_to_pos}
                    ref_positions = {id_to_pos[x] for x in ref_topk.tolist() if x in id_to_pos}
                    overlap_positions = sid_positions & ref_positions
                    sid_only_positions = sid_positions - overlap_positions
                    ref_only_positions = ref_positions - overlap_positions
                    context_positions = [
                        idx
                        for idx in range(union_ids_sorted.shape[0])
                        if idx != id_to_pos[anchor] and idx not in sid_positions and idx not in ref_positions
                    ]
                    x0 = 24.0 + col_idx * (cell_w + cell_gap_x)
                    y0 = header_h + row_idx * (cell_h + cell_gap_y)
                    svg_parts.append(
                        f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{cell_w:.2f}" height="{cell_h:.2f}" fill="white" stroke="#999" stroke-width="1.0" />'
                    )
                    scaled = _scale_points(coords, x0=x0, y0=y0 + 38.0, width=cell_w, height=cell_h - 54.0, pad=16.0)
                    overlap = _jaccard(ref_topk, sid_topk)
                    matplotlib_cache.append(
                        {
                            "row_idx": row_idx,
                            "col_idx": col_idx,
                            "anchor": anchor,
                            "method": method,
                            "coords": coords.copy(),
                            "context_positions": np.asarray(context_positions, dtype=np.int64),
                            "ref_only_positions": np.asarray(sorted(ref_only_positions), dtype=np.int64),
                            "sid_only_positions": np.asarray(sorted(sid_only_positions), dtype=np.int64),
                            "overlap_positions": np.asarray(sorted(overlap_positions), dtype=np.int64),
                            "anchor_pos_idx": int(id_to_pos[anchor]),
                            "overlap": float(overlap),
                        }
                    )
                    for idx in context_positions:
                        svg_parts.append(_svg_circle(scaled[idx, 0], scaled[idx, 1], 3.6, fill="#d9d9d9", opacity=0.58))
                    for idx in ref_only_positions:
                        svg_parts.append(_svg_triangle(scaled[idx, 0], scaled[idx, 1], 10.0, fill="#f04b4b", opacity=0.92, stroke="white", stroke_width=0.35))
                    for idx in sid_only_positions:
                        svg_parts.append(_svg_circle(scaled[idx, 0], scaled[idx, 1], 4.8, fill="#314a9b", opacity=0.92, stroke="white", stroke_width=0.35))
                    for idx in overlap_positions:
                        svg_parts.append(_svg_diamond(scaled[idx, 0], scaled[idx, 1], 10.0, fill="#7a3db8", opacity=0.96, stroke="white", stroke_width=0.35))
                    anchor_xy = scaled[id_to_pos[anchor]]
                    svg_parts.append(_svg_star(anchor_xy[0], anchor_xy[1], 7.8, fill="black"))
                    svg_parts.append(
                        f'<text x="{x0 + cell_w / 2.0:.2f}" y="{y0 + 20.0:.2f}" text-anchor="middle" font-size="14" font-family="Arial, Helvetica, sans-serif" font-weight="700">{html.escape(method)}</text>'
                    )
                    svg_parts.append(
                        f'<text x="{x0 + cell_w / 2.0:.2f}" y="{y0 + 36.0:.2f}" text-anchor="middle" font-size="11" font-family="Arial, Helvetica, sans-serif" fill="#444">Anchor {anchor} | J@{args.neighbors_k}={overlap:.2f}</text>'
                    )

            svg_parts.append("</svg>")
            (args.output_dir / "rq4_sid_geometry_neighbors.svg").write_text(
                "\n".join(svg_parts) + "\n",
                encoding="utf-8",
            )

            plt = _try_import_matplotlib()
            if plt is not None:
                fig, axes = plt.subplots(
                    nrows=n_plot,
                    ncols=len(top_method_order),
                    figsize=(4.2 * len(top_method_order), 4.0 * n_plot),
                    squeeze=False,
                )
                for panel in matplotlib_cache:
                    ax = axes[int(panel["row_idx"]), int(panel["col_idx"])]
                    coords_panel = np.asarray(panel["coords"], dtype=np.float64)
                    context_positions = np.asarray(panel["context_positions"], dtype=np.int64)
                    ref_only_positions = np.asarray(panel["ref_only_positions"], dtype=np.int64)
                    sid_only_positions = np.asarray(panel["sid_only_positions"], dtype=np.int64)
                    overlap_positions = np.asarray(panel["overlap_positions"], dtype=np.int64)
                    anchor_pos_idx = int(panel["anchor_pos_idx"])

                    if context_positions.size > 0:
                        ax.scatter(
                            coords_panel[context_positions, 0],
                            coords_panel[context_positions, 1],
                            s=26,
                            c="#d9d9d9",
                            marker="o",
                            alpha=0.55,
                            edgecolors="none",
                        )
                    if ref_only_positions.size > 0:
                        ax.scatter(
                            coords_panel[ref_only_positions, 0],
                            coords_panel[ref_only_positions, 1],
                            s=68,
                            c="#f04b4b",
                            marker="^",
                            alpha=0.9,
                            edgecolors="white",
                            linewidths=0.3,
                        )
                    if sid_only_positions.size > 0:
                        ax.scatter(
                            coords_panel[sid_only_positions, 0],
                            coords_panel[sid_only_positions, 1],
                            s=52,
                            c="#314a9b",
                            marker="o",
                            alpha=0.9,
                            edgecolors="white",
                            linewidths=0.3,
                        )
                    if overlap_positions.size > 0:
                        ax.scatter(
                            coords_panel[overlap_positions, 0],
                            coords_panel[overlap_positions, 1],
                            s=58,
                            c="#7a3db8",
                            marker="D",
                            alpha=0.95,
                            edgecolors="white",
                            linewidths=0.3,
                        )
                    ax.scatter(
                        [coords_panel[anchor_pos_idx, 0]],
                        [coords_panel[anchor_pos_idx, 1]],
                        s=160,
                        c="black",
                        marker="*",
                        edgecolors="white",
                        linewidths=0.5,
                        zorder=5,
                    )
                    ax.set_title(
                        f'{panel["method"]}\nAnchor {panel["anchor"]} | J@{args.neighbors_k}={panel["overlap"]:.2f}',
                        fontsize=11,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_alpha(0.35)

                handles = [
                    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#314a9b", markersize=8, label="SID top-K"),
                    plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#f04b4b", markersize=9, label="Reference top-K"),
                    plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#7a3db8", markersize=8, label="Overlap"),
                    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#d9d9d9", markersize=7, label="Local context"),
                    plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="black", markersize=12, label="Anchor item"),
                ]
                fig.legend(handles=handles, loc="upper center", ncol=5, frameon=True, bbox_to_anchor=(0.5, 1.02))
                fig.suptitle("RQ4: SID space vs. native semantic space (shared reference projection)", y=1.06, fontsize=14)
                fig.tight_layout()
                fig.savefig(args.output_dir / "rq4_sid_geometry_neighbors.png", dpi=220, bbox_inches="tight")
                plt.close(fig)


if __name__ == "__main__":
    main()
