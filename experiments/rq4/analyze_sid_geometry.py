
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
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


def _rbo_at_k(a: Iterable[int], b: Iterable[int], p: float) -> float:
    left = [int(x) for x in a]
    right = [int(x) for x in b]
    if len(left) != len(right):
        raise ValueError(f"RBO expects equal-length ranked lists, got {len(left)} vs {len(right)}")
    if not 0.0 < p < 1.0:
        raise ValueError(f"RBO persistence p must be in (0, 1), got {p}")
    k = len(left)
    if k == 0:
        return 0.0
    seen_left: set[int] = set()
    seen_right: set[int] = set()
    weighted_overlap = 0.0
    for depth in range(1, k + 1):
        seen_left.add(left[depth - 1])
        seen_right.add(right[depth - 1])
        overlap_ratio = len(seen_left & seen_right) / float(depth)
        weighted_overlap += (p ** (depth - 1)) * overlap_ratio
    normalizer = (1.0 - p**k) / (1.0 - p)
    if normalizer <= 0.0:
        return 0.0
    return float(weighted_overlap / normalizer)


def _logcomb(n: int, r: int) -> float:
    if r < 0 or r > n:
        return float("-inf")
    return math.lgamma(n + 1.0) - math.lgamma(r + 1.0) - math.lgamma(n - r + 1.0)


def _expected_jaccard_given_ties(
    k: int,
    strict_hits: int,
    cutoff_hits: int,
    cutoff_size: int,
    draws: int,
) -> float:
    if draws <= 0 or cutoff_hits <= 0:
        inter = strict_hits
        return float(inter / (2 * k - inter)) if (2 * k - inter) > 0 else 0.0
    if draws >= cutoff_size:
        inter = strict_hits + cutoff_hits
        return float(inter / (2 * k - inter)) if (2 * k - inter) > 0 else 0.0

    y_min = max(0, draws - (cutoff_size - cutoff_hits))
    y_max = min(cutoff_hits, draws)
    log_denom = _logcomb(cutoff_size, draws)
    expected = 0.0
    for y in range(y_min, y_max + 1):
        log_p = _logcomb(cutoff_hits, y) + _logcomb(cutoff_size - cutoff_hits, draws - y) - log_denom
        p_y = math.exp(log_p)
        inter = strict_hits + y
        expected += p_y * (inter / float(2 * k - inter))
    return float(expected)


def _expected_rbo_from_overlaps(expected_overlaps: np.ndarray, p: float) -> float:
    k = int(expected_overlaps.shape[0])
    if k <= 0:
        return 0.0
    depths = np.arange(1, k + 1, dtype=np.float64)
    weights = np.power(p, depths - 1.0, dtype=np.float64)
    normalizer = float(weights.sum())
    if normalizer <= 0.0:
        return 0.0
    return float(np.sum(weights * (expected_overlaps / depths)) / normalizer)


def _tie_aware_metrics_for_anchor(
    ref_topk: np.ndarray,
    sid_dist: np.ndarray,
    k: int,
    rbo_persistence: float,
) -> tuple[float, float]:
    finite_ids = np.flatnonzero(np.isfinite(sid_dist))
    if finite_ids.size == 0:
        return 0.0, 0.0
    ordered = np.argsort(sid_dist[finite_ids], kind="mergesort")
    sorted_ids = finite_ids[ordered]
    sorted_dist = sid_dist[sorted_ids]

    boundaries = np.flatnonzero(np.diff(sorted_dist) != 0.0) + 1
    group_starts = np.concatenate(([0], boundaries))
    group_ends = np.concatenate((boundaries, [sorted_ids.shape[0]]))

    ref_prefix_sets = [set(int(x) for x in ref_topk[:depth]) for depth in range(1, k + 1)]
    strict_items: set[int] = set()
    expected_overlaps = np.zeros(k, dtype=np.float64)

    cum = 0
    final_strict_hits = 0
    final_cutoff_hits = 0
    final_cutoff_size = 0
    final_draws = 0

    for start, end in zip(group_starts.tolist(), group_ends.tolist()):
        group = sorted_ids[start:end]
        group_size = int(group.shape[0])
        group_items = set(int(x) for x in group.tolist())
        if cum >= k:
            break
        max_depth = min(k, cum + group_size)
        for depth in range(cum + 1, max_depth + 1):
            ref_prefix = ref_prefix_sets[depth - 1]
            strict_hits = len(strict_items & ref_prefix)
            cutoff_hits = len(group_items & ref_prefix)
            draws = depth - cum
            expected_overlaps[depth - 1] = strict_hits + (draws * cutoff_hits / float(group_size))
            if depth == k:
                final_strict_hits = strict_hits
                final_cutoff_hits = cutoff_hits
                final_cutoff_size = group_size
                final_draws = draws
        cum += group_size
        if cum < k:
            strict_items |= group_items
        else:
            break

    expected_jaccard = _expected_jaccard_given_ties(
        k=k,
        strict_hits=final_strict_hits,
        cutoff_hits=final_cutoff_hits,
        cutoff_size=final_cutoff_size,
        draws=final_draws,
    )
    expected_rbo = _expected_rbo_from_overlaps(expected_overlaps, p=rbo_persistence)
    return expected_jaccard, expected_rbo


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


def _rotate_2d(coords: np.ndarray, angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    return coords @ rot.T


def _reference_local_scatter_coords(
    reference: np.ndarray,
    union_ids: np.ndarray,
    anchor: int,
    orient_ids: np.ndarray | None = None,
) -> np.ndarray:
    if union_ids.ndim != 1:
        raise ValueError(f"Expected 1D union_ids, got shape={union_ids.shape}")
    centered = reference[union_ids].astype(np.float64) - reference[int(anchor)].astype(np.float64)
    local_pca = _pca_2d(centered)
    anchor_mask = union_ids == int(anchor)
    coords = local_pca.astype(np.float64, copy=True)

    if orient_ids is not None and orient_ids.size > 0:
        orient_mask = np.isin(union_ids, orient_ids)
        orient_mask &= ~anchor_mask
        if orient_mask.any():
            centroid = coords[orient_mask].mean(axis=0)
            if float(np.linalg.norm(centroid)) > 1e-12:
                angle = math.atan2(float(centroid[1]), float(centroid[0]))
                coords = _rotate_2d(coords, (math.pi / 2.0) - angle)

    coords[anchor_mask] = 0.0
    non_anchor = np.flatnonzero(~anchor_mask)
    max_norm = float(np.max(np.linalg.norm(coords[non_anchor], axis=1))) if non_anchor.size > 0 else 1.0
    if max_norm <= 1e-12:
        max_norm = 1.0
    coords /= max_norm
    coords[anchor_mask] = 0.0
    return coords


def _local_reference_shape_stats(
    reference: np.ndarray,
    anchor_vec: np.ndarray,
    context_ids: np.ndarray,
) -> tuple[float, float]:
    if context_ids.ndim != 1:
        raise ValueError(f"Expected 1D context_ids, got shape={context_ids.shape}")
    if context_ids.size == 0:
        return 0.0, 0.0
    centered = reference[context_ids].astype(np.float64) - anchor_vec.astype(np.float64)[None, :]
    coords = _pca_2d(centered)
    variances = np.var(coords, axis=0, dtype=np.float64)
    v1 = float(max(variances[0], variances[1]))
    v2 = float(min(variances[0], variances[1]))
    isotropy = (v2 / v1) if v1 > 1e-12 else 0.0
    norms = np.linalg.norm(coords, axis=1)
    valid = norms > 1e-12
    if not np.any(valid):
        return isotropy, 0.0
    angles = np.arctan2(coords[valid, 1], coords[valid, 0])
    circular_spread = 1.0 - float(np.abs(np.mean(np.exp(1j * angles))))
    return isotropy, circular_spread


def _scale_unit_coords(coords: np.ndarray, x0: float, y0: float, width: float, height: float, pad: float) -> np.ndarray:
    usable_w = max(width - 2.0 * pad, 1.0)
    usable_h = max(height - 2.0 * pad, 1.0)
    half = min(usable_w, usable_h) / 2.0
    cx = x0 + width / 2.0
    cy = y0 + height / 2.0
    out = np.empty_like(coords, dtype=np.float64)
    out[:, 0] = cx + half * coords[:, 0]
    out[:, 1] = cy - half * coords[:, 1]
    return out


def _panel_view_coords(
    coords: np.ndarray,
    focus_positions: np.ndarray,
    *,
    focus_pad: float = 0.12,
    min_half_span: float = 0.22,
) -> tuple[np.ndarray, float]:
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected coords of shape [N,2], got {coords.shape}")
    if focus_positions.size > 0:
        focus = coords[np.asarray(focus_positions, dtype=np.int64)]
    else:
        focus = coords
    if focus.size == 0:
        half_span = 1.0
    else:
        max_abs = float(np.max(np.abs(focus)))
        half_span = max(max_abs * (1.0 + focus_pad), min_half_span)
    view = coords / max(half_span, 1e-12)
    return view, half_span


def _relax_display_coords(
    coords: np.ndarray,
    active_positions: np.ndarray,
    fixed_positions: np.ndarray,
    *,
    min_distance: float = 0.07,
    max_shift: float = 0.16,
    iterations: int = 80,
) -> np.ndarray:

    out = np.asarray(coords, dtype=np.float64).copy()
    active = np.asarray(sorted(set(np.asarray(active_positions, dtype=np.int64).tolist())), dtype=np.int64)
    fixed = set(np.asarray(fixed_positions, dtype=np.int64).tolist())
    active = np.asarray([idx for idx in active.tolist() if idx not in fixed], dtype=np.int64)
    if active.size <= 1:
        return out

    original = out.copy()
    all_positions = np.asarray(sorted(set(active.tolist()) | fixed), dtype=np.int64)
    for _ in range(iterations):
        delta = np.zeros_like(out, dtype=np.float64)
        for i in range(all_positions.size):
            left = int(all_positions[i])
            for right_raw in all_positions[i + 1 :]:
                right = int(right_raw)
                diff = out[left] - out[right]
                dist = float(np.linalg.norm(diff))
                if dist >= min_distance:
                    continue
                if dist <= 1e-12:
                    angle = ((left * 1103515245 + right * 12345) % 360) * math.pi / 180.0
                    unit = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)
                else:
                    unit = diff / dist
                push = 0.5 * (min_distance - dist)
                if left not in fixed:
                    delta[left] += unit * push
                if right not in fixed:
                    delta[right] -= unit * push

        out[active] += 0.45 * delta[active]
        displacement = out[active] - original[active]
        norms = np.linalg.norm(displacement, axis=1)
        too_far = norms > max_shift
        if np.any(too_far):
            out[active[too_far]] = (
                original[active[too_far]]
                + displacement[too_far] / np.maximum(norms[too_far, None], 1e-12) * max_shift
            )
        out[active] = np.clip(out[active], -1.02, 1.02)
    return out


def _try_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _plot_style_tokens() -> dict[str, object]:
    return {
        "fig_bg": "#ffffff",
        "ax_bg": "#ffffff",
        "spine": "#7f7f7f",
        "title": "#111111",
        "subtitle": "#222222",
        "legend_bg": "#ffffff",
        "legend_edge": "#7f7f7f",

        "sid": "#4f5f99",
        "ref": "#ff8f92",
        "overlap": "#7b5dbb",
        "context": "#c9c9c9",
        "anchor_fill": "#111111",
        "anchor_edge": "#111111",
    }


@dataclass
class SidSpec:
    name: str
    path: Path
    raw_codes: np.ndarray
    codes: np.ndarray
    native_metric: str
    drop_last_col: bool


def _parse_name_value_specs(
    specs: list[str] | None,
    *,
    value_parser,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f"Invalid NAME=VALUE spec: {spec}")
        name, raw_value = spec.split("=", 1)
        out[name.strip()] = value_parser(raw_value.strip())
    return out


def _parse_bool(value: str) -> bool:
    norm = value.strip().lower()
    if norm in {"1", "true", "yes", "on"}:
        return True
    if norm in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid bool value: {value}")


def _default_native_metric(name: str) -> str:
    if name in {"OPQ", "I06-DiffGRM", "OPQ-L6"}:
        return "hamming"
    if name in {
        "TIGER",
        "SEATER",
        "HowToIndex",
        "LETTER",
        "RQ-Kmeans",
        "I02-TIGER",
        "I03-SEATER",
        "I04-SemID",
        "I05-LETTER",
        "I07-RQKmeans",
    }:
        return "prefix"
    return "hamming"


def _default_drop_last_col(name: str) -> bool:
    if name in {
        "TIGER",
        "SEATER",
        "LETTER",
        "OPQ",
        "RQ-Kmeans",
        "I02-TIGER",
        "I03-SEATER",
        "I05-LETTER",
        "I06-DiffGRM",
        "I07-RQKmeans",
    }:
        return True
    return False


def _effective_metric_name(paper_metric: str, sid_spec: SidSpec) -> str:
    return sid_spec.native_metric if paper_metric == "native" else paper_metric


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
        description="Analyze how well different semantic IDs preserve local neighborhoods in the reference item-text embedding space."
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
        "--plot_display_name",
        action="append",
        default=[],
        help="Optional NAME=LABEL override used only for plot labels.",
    )
    ap.add_argument(
        "--plot_font_scale",
        type=float,
        default=1.0,
        help="Scale factor for plot text font sizes.",
    )
    ap.add_argument(
        "--paper_sid_metric",
        type=str,
        default="native",
        choices=["hamming", "prefix", "native"],
        help="Metric used for the paper-ready main table and case-study selection.",
    )
    ap.add_argument(
        "--native_metric_override",
        action="append",
        default=[],
        help="Optional override NAME=METRIC for native SID distance (METRIC in {hamming,prefix}).",
    )
    ap.add_argument(
        "--drop_last_override",
        action="append",
        default=[],
        help="Optional override NAME=BOOL controlling whether to drop the last code column before analysis.",
    )
    ap.add_argument(
        "--paper_tie_aware",
        action="store_true",
        help="Use tie-aware Jaccard/RBO for paper table outputs and overview plots.",
    )
    ap.add_argument(
        "--rbo_persistence",
        type=float,
        default=0.9,
        help="Persistence parameter p used by truncated RBO on the top-K neighbor lists.",
    )
    ap.add_argument("--seed", type=int, default=20260423)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_display_names = _parse_name_value_specs(
        args.plot_display_name,
        value_parser=lambda value: value,
    )
    if args.plot_font_scale <= 0:
        raise ValueError(f"--plot_font_scale must be positive, got {args.plot_font_scale}")
    plot_font_scale = float(args.plot_font_scale)

    def display_name(method: str) -> str:
        return str(plot_display_names.get(method, method))

    reference = _load_embeddings(args.reference_embeddings)
    reference_norm = _normalize_rows(reference.astype(np.float32))
    n_items = reference.shape[0]

    native_metric_overrides = _parse_name_value_specs(
        args.native_metric_override,
        value_parser=lambda value: value,
    )
    drop_last_overrides = _parse_name_value_specs(
        args.drop_last_override,
        value_parser=_parse_bool,
    )

    sid_specs: list[SidSpec] = []
    for raw_spec in args.sid:
        name, path = _parse_sid_spec(raw_spec)
        raw_codes = _load_codes(path)
        if raw_codes.shape[0] != n_items:
            raise ValueError(
                f"{name}: item count mismatch. codes={raw_codes.shape[0]} reference={n_items} path={path}"
            )
        native_metric = native_metric_overrides.get(name, _default_native_metric(name))
        if native_metric not in {"hamming", "prefix"}:
            raise ValueError(f"{name}: unsupported native metric {native_metric}")
        drop_last_col = drop_last_overrides.get(name, _default_drop_last_col(name))
        if drop_last_col:
            if raw_codes.shape[1] <= 1:
                raise ValueError(f"{name}: cannot drop last column for shape={raw_codes.shape}")
            codes = raw_codes[:, :-1]
        else:
            codes = raw_codes
        sid_specs.append(
            SidSpec(
                name=name,
                path=path,
                raw_codes=raw_codes,
                codes=codes,
                native_metric=native_metric,
                drop_last_col=drop_last_col,
            )
        )

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
        metric_payloads = {
            "hamming": (pair_dist["hamming"], sid_neighbors_h, triplet_h_ij, triplet_h_ik),
            "prefix": (pair_dist["prefix"], sid_neighbors_p, triplet_p_ij, triplet_p_ik),
        }
        metric_summary_map: dict[str, dict[str, object]] = {}
        metric_anchor_map: dict[str, list[dict[str, object]]] = {}

        for metric_name, (sid_distances, sid_neighbors, sid_tij, sid_tik) in metric_payloads.items():
            jaccards: list[float] = []
            recalls: list[float] = []
            anchor_corrs: list[float] = []
            anchor_rbos: list[float] = []
            tieaware_jaccards: list[float] = []
            tieaware_rbos: list[float] = []
            metric_anchor_rows: list[dict[str, object]] = []
            for anchor_idx, anchor in enumerate(anchors.tolist()):
                ref_set = ref_neighbors[anchor_idx]
                sid_set = sid_neighbors[anchor_idx]
                j = _jaccard(ref_set, sid_set)
                r = _recall(ref_set, sid_set)
                rbo = _rbo_at_k(ref_set, sid_set, p=args.rbo_persistence)
                emb_dist_all = 1.0 - (reference_norm[anchor] @ reference_norm.T).astype(np.float64)
                emb_dist_all[anchor] = np.inf
                sid_dist_all = _all_sid_distances(sid_spec.codes[anchor], sid_spec.codes, metric_name)
                sid_dist_all[anchor] = np.inf
                tie_j, tie_rbo = _tie_aware_metrics_for_anchor(
                    ref_topk=ref_set,
                    sid_dist=sid_dist_all,
                    k=args.neighbors_k,
                    rbo_persistence=args.rbo_persistence,
                )
                jaccards.append(j)
                recalls.append(r)
                anchor_rbos.append(rbo)
                tieaware_jaccards.append(tie_j)
                tieaware_rbos.append(tie_rbo)
                anchor_corr = _rank_corr_for_anchor(
                    emb_dist=emb_dist_all,
                    sid_dist=sid_dist_all,
                    top_m=min(args.anchor_rank_topm, n_items - 1),
                )
                anchor_corrs.append(anchor_corr)
                metric_anchor_rows.append(
                    {
                        "method": sid_spec.name,
                        "id_metric": metric_name,
                        "native_metric": sid_spec.native_metric,
                        "drop_last_col": bool(sid_spec.drop_last_col),
                        "anchor_item_id": anchor,
                        "jaccard_at_k": round(j, 6),
                        "recall_at_k": round(r, 6),
                        "rbo_at_k": round(rbo, 6),
                        "tieaware_jaccard_at_k": round(tie_j, 6),
                        "tieaware_rbo_at_k": round(tie_rbo, 6),
                        "local_rank_spearman": round(anchor_corr, 6),
                    }
                )

            metric_summary_map[metric_name] = (
                {
                    "method": sid_spec.name,
                    "id_metric": metric_name,
                    "native_metric": sid_spec.native_metric,
                    "drop_last_col": bool(sid_spec.drop_last_col),
                    "reference_embedding_path": str(args.reference_embeddings),
                    "sid_path": str(sid_spec.path),
                    "n_items": n_items,
                    "raw_code_length": int(sid_spec.raw_codes.shape[1]),
                    "code_length": int(sid_spec.codes.shape[1]),
                    "pair_samples": int(args.pair_samples),
                    "triplet_samples": int(args.triplet_samples),
                    "anchor_samples": int(anchors.shape[0]),
                    "neighbors_k": int(args.neighbors_k),
                    "pearson_distance_corr": round(_pearson(ref_pair_dist, sid_distances), 6),
                    "spearman_distance_corr": round(_spearman(ref_pair_dist, sid_distances), 6),
                    "mean_jaccard_at_k": round(float(np.mean(jaccards)), 6),
                    "mean_recall_at_k": round(float(np.mean(recalls)), 6),
                    "mean_rbo_at_k": round(float(np.mean(anchor_rbos)), 6),
                    "mean_tieaware_jaccard_at_k": round(float(np.mean(tieaware_jaccards)), 6),
                    "mean_tieaware_rbo_at_k": round(float(np.mean(tieaware_rbos)), 6),
                    "mean_local_rank_spearman": round(float(np.mean(anchor_corrs)), 6),
                    "triplet_accuracy": round(
                        _triplet_accuracy(ref_triplet_ij, ref_triplet_ik, sid_tij, sid_tik),
                        6,
                    ),
                }
            )
            metric_anchor_map[metric_name] = metric_anchor_rows
            if metric_name == "hamming":
                plot_cache[sid_spec.name] = sid_neighbors

        for metric_name in ("hamming", "prefix"):
            summary_rows.append(metric_summary_map[metric_name])
            anchor_rows.extend(metric_anchor_map[metric_name])
        native_summary = dict(metric_summary_map[sid_spec.native_metric])
        native_summary["id_metric"] = "native"
        summary_rows.append(native_summary)
        for row in metric_anchor_map[sid_spec.native_metric]:
            native_row = dict(row)
            native_row["id_metric"] = "native"
            anchor_rows.append(native_row)

    summary_rows.sort(key=lambda row: (str(row["id_metric"]), -float(row["spearman_distance_corr"])))
    _save_csv(
        args.output_dir / "summary_metrics.csv",
        summary_rows,
        fieldnames=[
            "method",
            "id_metric",
            "native_metric",
            "drop_last_col",
            "reference_embedding_path",
            "sid_path",
            "n_items",
            "raw_code_length",
            "code_length",
            "pair_samples",
            "triplet_samples",
            "anchor_samples",
            "neighbors_k",
            "pearson_distance_corr",
            "spearman_distance_corr",
            "mean_jaccard_at_k",
            "mean_recall_at_k",
            "mean_rbo_at_k",
            "mean_tieaware_jaccard_at_k",
            "mean_tieaware_rbo_at_k",
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
            "native_metric",
            "drop_last_col",
            "anchor_item_id",
            "jaccard_at_k",
            "recall_at_k",
            "rbo_at_k",
            "tieaware_jaccard_at_k",
            "tieaware_rbo_at_k",
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
        "rbo_persistence": float(args.rbo_persistence),
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
            paper_j_key = "mean_tieaware_jaccard_at_k" if args.paper_tie_aware else "mean_jaccard_at_k"
            paper_rbo_key = "mean_tieaware_rbo_at_k" if args.paper_tie_aware else "mean_rbo_at_k"
            paper_table_rows.append(
                {
                    "method": method,
                    "id_metric": args.paper_sid_metric,
                    "native_metric": str(row["native_metric"]),
                    "jaccard_overlap_at_k": float(row[paper_j_key]),
                    "rbo_at_k": float(row[paper_rbo_key]),
                }
            )
        _save_csv(
            args.output_dir / "paper_main_table.csv",
            paper_table_rows,
            fieldnames=[
                "method",
                "id_metric",
                "native_metric",
                "jaccard_overlap_at_k",
                "rbo_at_k",
            ],
        )
        _save_csv(
            args.output_dir / "paper_main_table_local_clean.csv",
            [
                {
                    "method": str(row["method"]),
                    f"jaccard_overlap_at_{args.neighbors_k}": float(row["jaccard_overlap_at_k"]),
                    f"rbo_overlap_at_{args.neighbors_k}": float(row["rbo_at_k"]),
                }
                for row in paper_table_rows
            ],
            fieldnames=[
                "method",
                f"jaccard_overlap_at_{args.neighbors_k}",
                f"rbo_overlap_at_{args.neighbors_k}",
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
        per_anchor_rbo = {method: {} for method in top_method_order}
        for row in anchor_rows:
            if row["id_metric"] != args.paper_sid_metric:
                continue
            method = str(row["method"])
            anchor_id = int(row["anchor_item_id"])
            j_key = "tieaware_jaccard_at_k" if args.paper_tie_aware else "jaccard_at_k"
            rbo_key = "tieaware_rbo_at_k" if args.paper_tie_aware else "rbo_at_k"
            per_anchor[method][anchor_id] = float(row[j_key])
            per_anchor_rbo[method][anchor_id] = float(row[rbo_key])

        global_jaccard = np.asarray(
            [
                float(
                    paper_rows_by_method[method][
                        "mean_tieaware_jaccard_at_k" if args.paper_tie_aware else "mean_jaccard_at_k"
                    ]
                )
                for method in top_method_order
            ],
            dtype=np.float64,
        )
        global_rbo = np.asarray(
            [
                float(
                    paper_rows_by_method[method][
                        "mean_tieaware_rbo_at_k" if args.paper_tie_aware else "mean_rbo_at_k"
                    ]
                )
                for method in top_method_order
            ],
            dtype=np.float64,
        )

        anchor_candidate_rows: list[dict[str, object]] = []
        for anchor in anchors.tolist():
            j_vals = np.asarray([per_anchor[m].get(anchor, 0.0) for m in top_method_order], dtype=np.float64)
            rbo_vals = np.asarray([per_anchor_rbo[m].get(anchor, 0.0) for m in top_method_order], dtype=np.float64)
            ref_all_order = np.argsort(
                1.0 - (reference_norm[anchor] @ reference_norm.T),
                kind="mergesort",
            )
            ref_all_order = ref_all_order[ref_all_order != anchor]
            context_ids = ref_all_order[: max(args.plot_context_k, args.neighbors_k)]
            isotropy, circular_spread = _local_reference_shape_stats(
                reference=reference,
                anchor_vec=reference[anchor],
                context_ids=context_ids,
            )
            anchor_candidate_rows.append(
                {
                    "anchor_item_id": int(anchor),
                    "jaccard_rank_alignment": float(_spearman(j_vals, global_jaccard)),
                    "rbo_rank_alignment": float(_spearman(rbo_vals, global_rbo)),
                    "jaccard_gap": float(j_vals.max() - j_vals.min()),
                    "rbo_gap": float(rbo_vals.max() - rbo_vals.min()),
                    "mean_jaccard": float(j_vals.mean()),
                    "mean_rbo": float(rbo_vals.mean()),
                    "local_isotropy": float(isotropy),
                    "local_circular_spread": float(circular_spread),
                }
            )

        if anchor_candidate_rows:
            global_mean_j = float(global_jaccard.mean())
            global_mean_rbo = float(global_rbo.mean())
            j_gap_z = _zscore(np.asarray([row["jaccard_gap"] for row in anchor_candidate_rows], dtype=np.float64))
            rbo_gap_z = _zscore(np.asarray([row["rbo_gap"] for row in anchor_candidate_rows], dtype=np.float64))
            isotropy_z = _zscore(np.asarray([row["local_isotropy"] for row in anchor_candidate_rows], dtype=np.float64))
            circular_z = _zscore(
                np.asarray([row["local_circular_spread"] for row in anchor_candidate_rows], dtype=np.float64)
            )
            typicality_penalty = _zscore(
                np.asarray(
                    [
                        abs(float(row["mean_jaccard"]) - global_mean_j)
                        + 0.5 * abs(float(row["mean_rbo"]) - global_mean_rbo)
                        for row in anchor_candidate_rows
                    ],
                    dtype=np.float64,
                )
            )
            for idx, row in enumerate(anchor_candidate_rows):
                rank_consistency = 0.5 * float(row["jaccard_rank_alignment"]) + 0.5 * float(row["rbo_rank_alignment"])
                selection_score = (
                    rank_consistency
                    + 0.25 * float(j_gap_z[idx])
                    + 0.15 * float(rbo_gap_z[idx])
                    + 0.18 * float(isotropy_z[idx])
                    + 0.18 * float(circular_z[idx])
                    - 0.15 * float(typicality_penalty[idx])
                )
                row["selection_score"] = float(selection_score)

        anchor_candidate_rows.sort(
            key=lambda row: (
                -float(row.get("selection_score", 0.0)),
                -float(row["jaccard_rank_alignment"]),
                -float(row["rbo_rank_alignment"]),
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
                "rbo_rank_alignment",
                "jaccard_gap",
                "rbo_gap",
                "mean_jaccard",
                "mean_rbo",
                "local_isotropy",
                "local_circular_spread",
            ],
        )
        if anchor_candidate_rows:
            (args.output_dir / "selected_case_anchor.json").write_text(
                json.dumps(anchor_candidate_rows[0], ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        plt = _try_import_matplotlib()
        if plt is not None and top_method_order:
            fig, ax = plt.subplots(figsize=(7.6, 5.6))
            paper_j_key = "mean_tieaware_jaccard_at_k" if args.paper_tie_aware else "mean_jaccard_at_k"
            paper_rbo_key = "mean_tieaware_rbo_at_k" if args.paper_tie_aware else "mean_rbo_at_k"
            xs = [float(paper_rows_by_method[method][paper_j_key]) for method in top_method_order]
            ys = [float(paper_rows_by_method[method][paper_rbo_key]) for method in top_method_order]
            colors = ["#355fb3", "#c93f5b", "#2f8f6b", "#c07f00", "#7a3db8", "#444444"]
            for idx, method in enumerate(top_method_order):
                ax.scatter(xs[idx], ys[idx], s=86, color=colors[idx % len(colors)], edgecolors="white", linewidths=0.6, zorder=3)
                ax.text(xs[idx] + 0.0025, ys[idx] + 0.0025, display_name(method), fontsize=10 * plot_font_scale)
            metric_label = "Native SID Metric" if args.paper_sid_metric == "native" else args.paper_sid_metric.capitalize()
            j_label = f"{'Tie-aware ' if args.paper_tie_aware else ''}Jaccard@{args.neighbors_k}"
            rbo_label = f"{'Tie-aware ' if args.paper_tie_aware else ''}RBO@{args.neighbors_k} (p={args.rbo_persistence:.1f})"
            ax.set_xlabel(j_label, fontsize=11 * plot_font_scale)
            ax.set_ylabel(rbo_label, fontsize=11 * plot_font_scale)
            ax.set_title(f"RQ4 Main Table: Local Semantic Neighborhood Preservation ({metric_label})", fontsize=13 * plot_font_scale)
            ax.tick_params(axis="both", labelsize=9 * plot_font_scale)
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
            ax.set_axisbelow(True)
            fig.tight_layout()
            fig.savefig(args.output_dir / "rq4_paper_overview_local.png", dpi=220, bbox_inches="tight")
            fig.savefig(args.output_dir / "rq4_paper_overview_local.pdf", bbox_inches="tight")
            plt.close(fig)

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
                f'<rect x="0" y="0" width="{total_w:.0f}" height="{total_h:.0f}" fill="#ffffff" />',
                f'<text x="{total_w / 2.0:.2f}" y="28" text-anchor="middle" font-size="22" font-family="Arial, Helvetica, sans-serif" font-weight="700" fill="#1f1f1f">RQ4: SID space vs. native semantic space</text>',
                f'<text x="{total_w / 2.0:.2f}" y="52" text-anchor="middle" font-size="13" font-family="Arial, Helvetica, sans-serif" fill="#5b554b">Anchor-centered local scatter in the reference space; red = reference top-K, blue = SID top-K, purple = overlap</text>',
            ]
            matplotlib_cache: list[dict[str, object]] = []

            legend_x = 28.0
            legend_y = total_h - 18.0
            legend_items = [
                ("circle", "#4f5f99", "SID top-K"),
                ("triangle", "#ff8f92", "Reference top-K"),
                ("diamond", "#7b5dbb", "Overlap"),
                ("triangle", "#c9c9c9", "Local context"),
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
                        metric=_effective_metric_name(
                            args.paper_sid_metric,
                            sid_specs[[spec.name for spec in sid_specs].index(method)],
                        ),
                    )[0]
                    union_ids.update(sid_topk_for_anchor.tolist())
                union_ids_sorted = np.asarray(sorted(union_ids), dtype=np.int64)
                coords = _reference_local_scatter_coords(
                    reference=reference,
                    union_ids=union_ids_sorted,
                    anchor=anchor,
                    orient_ids=ref_topk,
                )
                id_to_pos = {int(item_id): idx for idx, item_id in enumerate(union_ids_sorted.tolist())}
                for col_idx, method in enumerate(top_method_order):
                    sid_topk = _topk_sid_neighbors(
                        codes=sid_specs[[spec.name for spec in sid_specs].index(method)].codes,
                        anchors=np.asarray([anchor], dtype=np.int64),
                        k=args.neighbors_k,
                        metric=_effective_metric_name(
                            args.paper_sid_metric,
                            sid_specs[[spec.name for spec in sid_specs].index(method)],
                        ),
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
                        f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{cell_w:.2f}" height="{cell_h:.2f}" rx="0" ry="0" fill="#ffffff" stroke="#7f7f7f" stroke-width="1.3" />'
                    )
                    plot_x0 = x0
                    plot_y0 = y0 + 38.0
                    plot_w = cell_w
                    plot_h = cell_h - 54.0


                    panel_focus = np.asarray(
                        sorted({id_to_pos[anchor], *ref_positions, *sid_positions}),
                        dtype=np.int64,
                    )
                    view_coords, panel_half_span = _panel_view_coords(coords, panel_focus)
                    scaled = _scale_unit_coords(view_coords, x0=plot_x0, y0=plot_y0, width=plot_w, height=plot_h, pad=16.0)
                    overlap = _jaccard(ref_topk, sid_topk)
                    rbo = _rbo_at_k(ref_topk, sid_topk, p=args.rbo_persistence)
                    matplotlib_cache.append(
                        {
                            "row_idx": row_idx,
                            "col_idx": col_idx,
                            "anchor": anchor,
                            "method": method,
                            "coords": view_coords.copy(),
                            "panel_half_span": float(panel_half_span),
                            "context_positions": np.asarray(context_positions, dtype=np.int64),
                            "ref_only_positions": np.asarray(sorted(ref_only_positions), dtype=np.int64),
                            "sid_only_positions": np.asarray(sorted(sid_only_positions), dtype=np.int64),
                            "overlap_positions": np.asarray(sorted(overlap_positions), dtype=np.int64),
                            "anchor_pos_idx": int(id_to_pos[anchor]),
                            "overlap": float(overlap),
                            "rbo": float(rbo),
                        }
                    )
                    for idx in context_positions:
                        svg_parts.append(_svg_triangle(scaled[idx, 0], scaled[idx, 1], 7.0, fill="#c9c9c9", opacity=0.46))
                    for idx in ref_only_positions:
                        svg_parts.append(_svg_triangle(scaled[idx, 0], scaled[idx, 1], 10.0, fill="#ff8f92", opacity=0.76))
                    for idx in sid_only_positions:
                        svg_parts.append(_svg_circle(scaled[idx, 0], scaled[idx, 1], 4.8, fill="#4f5f99", opacity=0.82))
                    for idx in overlap_positions:
                        svg_parts.append(_svg_diamond(scaled[idx, 0], scaled[idx, 1], 10.5, fill="#7b5dbb", opacity=0.82))
                    anchor_xy = scaled[id_to_pos[anchor]]
                    svg_parts.append(_svg_star(anchor_xy[0], anchor_xy[1], 7.8, fill="black"))
                    svg_parts.append(
                        f'<text x="{x0 + cell_w / 2.0:.2f}" y="{y0 + 19.0:.2f}" text-anchor="middle" font-size="14" font-family="Arial, Helvetica, sans-serif" font-weight="700" fill="#1f1f1f">{html.escape(display_name(method))}</text>'
                    )
                    svg_parts.append(
                        f'<text x="{x0 + cell_w / 2.0:.2f}" y="{y0 + 35.0:.2f}" text-anchor="middle" font-size="11" font-family="Arial, Helvetica, sans-serif" fill="#5b554b">Anchor {anchor} | J@{args.neighbors_k}={overlap:.2f} | RBO={rbo:.2f}</text>'
                    )

            svg_parts.append("</svg>")
            (args.output_dir / "rq4_sid_geometry_neighbors.svg").write_text(
                "\n".join(svg_parts) + "\n",
                encoding="utf-8",
            )

            plt = _try_import_matplotlib()
            if plt is not None:
                style = _plot_style_tokens()
                method_grid_cols = min(3, len(top_method_order))
                method_grid_rows = int(math.ceil(len(top_method_order) / method_grid_cols))
                total_plot_rows = n_plot * method_grid_rows
                figsize_scale = max(1.0, min(1.25, 1.0 + 0.22 * (plot_font_scale - 1.0)))
                fig, axes = plt.subplots(
                    nrows=total_plot_rows,
                    ncols=method_grid_cols,
                    figsize=(3.25 * method_grid_cols * figsize_scale, 3.70 * total_plot_rows * figsize_scale),
                    squeeze=False,
                    facecolor=style["fig_bg"],
                )
                fig.patch.set_facecolor(style["fig_bg"])
                used_axes: set[tuple[int, int]] = set()
                for panel in matplotlib_cache:
                    method_slot = int(panel["col_idx"])
                    grid_row = int(panel["row_idx"]) * method_grid_rows + method_slot // method_grid_cols
                    grid_col = method_slot % method_grid_cols
                    used_axes.add((grid_row, grid_col))
                    ax = axes[grid_row, grid_col]
                    ax.set_facecolor(style["ax_bg"])
                    coords_panel = np.asarray(panel["coords"], dtype=np.float64)
                    context_positions = np.asarray(panel["context_positions"], dtype=np.int64)
                    ref_only_positions = np.asarray(panel["ref_only_positions"], dtype=np.int64)
                    sid_only_positions = np.asarray(panel["sid_only_positions"], dtype=np.int64)
                    overlap_positions = np.asarray(panel["overlap_positions"], dtype=np.int64)
                    anchor_pos_idx = int(panel["anchor_pos_idx"])
                    if context_positions.size > 0:
                        context_xy = coords_panel[context_positions]
                        visible_context = (
                            (np.abs(context_xy[:, 0]) <= 1.02)
                            & (np.abs(context_xy[:, 1]) <= 1.02)
                        )
                        context_positions = context_positions[visible_context]
                    display_positions = np.concatenate(
                        [ref_only_positions, sid_only_positions, overlap_positions]
                    )
                    display_coords = _relax_display_coords(
                        coords_panel,
                        active_positions=display_positions,
                        fixed_positions=np.asarray([anchor_pos_idx], dtype=np.int64),
                        min_distance=0.066,
                        max_shift=0.15,
                        iterations=90,
                    )

                    if context_positions.size > 0:
                        coll = ax.scatter(
                            display_coords[context_positions, 0],
                            display_coords[context_positions, 1],
                            s=17,
                            c=style["context"],
                            marker="^",
                            alpha=0.46,
                            edgecolors="none",
                            zorder=1,
                            clip_on=True,
                        )
                        coll.set_clip_path(ax.patch)
                    if ref_only_positions.size > 0:
                        coll = ax.scatter(
                            display_coords[ref_only_positions, 0],
                            display_coords[ref_only_positions, 1],
                            s=40,
                            c=style["ref"],
                            marker="^",
                            alpha=0.76,
                            edgecolors="none",
                            linewidths=0.0,
                            zorder=3,
                            clip_on=True,
                        )
                        coll.set_clip_path(ax.patch)
                    if sid_only_positions.size > 0:
                        coll = ax.scatter(
                            display_coords[sid_only_positions, 0],
                            display_coords[sid_only_positions, 1],
                            s=38,
                            c=style["sid"],
                            marker="o",
                            alpha=0.82,
                            edgecolors=style["sid"],
                            linewidths=0.2,
                            zorder=2,
                            clip_on=True,
                        )
                        coll.set_clip_path(ax.patch)
                    if overlap_positions.size > 0:
                        coll = ax.scatter(
                            display_coords[overlap_positions, 0],
                            display_coords[overlap_positions, 1],
                            s=44,
                            c=style["overlap"],
                            marker="D",
                            alpha=0.82,
                            edgecolors=style["overlap"],
                            linewidths=0.2,
                            zorder=4,
                            clip_on=True,
                        )
                        coll.set_clip_path(ax.patch)
                    coll = ax.scatter(
                        [coords_panel[anchor_pos_idx, 0]],
                        [coords_panel[anchor_pos_idx, 1]],
                        s=105,
                        c=style["anchor_fill"],
                        marker="*",
                        edgecolors=style["anchor_edge"],
                        linewidths=0.6,
                        zorder=5,
                        clip_on=True,
                    )
                    coll.set_clip_path(ax.patch)
                    ax.set_title(
                        display_name(str(panel["method"])),
                        fontsize=12.0 * plot_font_scale,
                        color=style["title"],
                        pad=5,
                    )
                    ax.set_xlim(-1.04, 1.04)
                    ax.set_ylim(-1.04, 1.04)
                    ax.set_aspect("equal")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_color(style["spine"])
                        spine.set_linewidth(1.3)
                        spine.set_alpha(0.9)

                for row_idx in range(total_plot_rows):
                    for col_idx in range(method_grid_cols):
                        if (row_idx, col_idx) not in used_axes:
                            axes[row_idx, col_idx].set_visible(False)

                handles = [
                    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=style["sid"], markeredgecolor=style["sid"], markersize=6, label="SID top-K"),
                    plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=style["ref"], markeredgecolor=style["ref"], markersize=7, label="Reference top-K"),
                    plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=style["overlap"], markeredgecolor=style["overlap"], markersize=6, label="Overlap"),
                    plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=style["context"], markeredgecolor=style["context"], markersize=6, alpha=0.7, label="Local context"),
                    plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=style["anchor_fill"], markeredgecolor=style["anchor_edge"], markersize=9, label="Anchor item"),
                ]
                legend = fig.legend(
                    handles=handles,
                    loc="upper center",
                    ncol=len(handles),
                    bbox_to_anchor=(0.5, 0.985),
                    frameon=True,
                    fontsize=8.9 * plot_font_scale,
                    handletextpad=0.42,
                    columnspacing=0.95,
                )
                legend.get_frame().set_facecolor(style["legend_bg"])
                legend.get_frame().set_edgecolor(style["legend_edge"])
                legend.get_frame().set_linewidth(1.1)
                legend.get_frame().set_alpha(0.96)
                top_margin = max(0.74, 0.885 - 0.075 * (plot_font_scale - 1.0))
                hspace = min(0.32, 0.155 + 0.060 * (plot_font_scale - 1.0))
                fig.subplots_adjust(
                    left=0.032,
                    right=0.988,
                    bottom=0.035,
                    top=top_margin,
                    wspace=0.075,
                    hspace=hspace,
                )
                fig.savefig(args.output_dir / "rq4_sid_geometry_neighbors.png", dpi=240, bbox_inches="tight", facecolor=style["fig_bg"])
                fig.savefig(args.output_dir / "rq4_sid_geometry_neighbors.pdf", bbox_inches="tight", facecolor=style["fig_bg"])
                if len(chosen_anchors) == 1:
                    anchor_name = f"case_{chosen_anchors[0]}.png"
                    fig.savefig(args.output_dir / anchor_name, dpi=240, bbox_inches="tight", facecolor=style["fig_bg"])
                    fig.savefig(args.output_dir / "rq4_case_study_selected.png", dpi=240, bbox_inches="tight", facecolor=style["fig_bg"])
                    anchor_pdf_name = f"case_{chosen_anchors[0]}.pdf"
                    fig.savefig(args.output_dir / anchor_pdf_name, bbox_inches="tight", facecolor=style["fig_bg"])
                    fig.savefig(args.output_dir / "rq4_case_study_selected.pdf", bbox_inches="tight", facecolor=style["fig_bg"])
                plt.close(fig)


if __name__ == "__main__":
    main()
