#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def _parse_title(combined: str) -> str:
    prefix = "The item title is "
    text = combined.strip()
    if text.startswith(prefix):
        text = text[len(prefix):]
    if text.endswith("."):
        text = text[:-1]
    return text.strip()


_TAG_RE = re.compile(r"#([^#\s，。,.!！?？;；:：/\\]+)")
_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _stable_bucket(text: str, n_buckets: int) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % n_buckets


def _clean_topic_token(text: str, max_len: int = 48) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "_", text)
    text = text.strip("_")
    if not text:
        return "nohash"
    return text[:max_len].strip("_") or "nohash"


def _extract_topic(title: str, *, rare_tag_counts: collections.Counter[str], min_topic_count: int) -> str:
    tags = [_clean_topic_token(t) for t in _TAG_RE.findall(title)]
    tags = [t for t in tags if t and t != "nohash"]
    if tags:
        head = tags[0]
        if rare_tag_counts[head] >= min_topic_count:
            return f"topic_{head}"
        # Keep rare hashtag signal without letting one-off tags explode the SID vocab.
        return f"topic_rare_{_stable_bucket(head, 2048):04d}"

    words = [w.lower() for w in _ASCII_WORD_RE.findall(title)]
    words = [w for w in words if len(w) > 1][:6]
    if words:
        return "topic_" + "_".join(words)
    return "topic_nohash"


def _fit_minibatch_kmeans(data: np.ndarray, n_clusters: int, seed: int, batch_size: int) -> np.ndarray:
    if data.shape[0] <= n_clusters:
        return np.arange(data.shape[0], dtype=np.int64)
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=min(batch_size, max(n_clusters * 8, data.shape[0])),
        n_init=3,
        max_iter=100,
        reassignment_ratio=0.01,
        verbose=0,
    )
    return km.fit_predict(data).astype(np.int64)


def build_paths(
    *,
    combine_maps_path: Path,
    emb_path: Path,
    output_path: Path,
    stats_path: Path,
    n_clusters: int,
    n_subclusters: int,
    min_topic_count: int,
    batch_size: int,
    seed: int,
) -> None:
    payload = np.load(combine_maps_path, allow_pickle=True).item()
    recid2combine = payload["recid2combine"]
    n_items = len(recid2combine)

    emb = np.load(emb_path).astype(np.float32)
    if emb.shape[0] != n_items:
        raise ValueError(f"embedding/item mismatch: emb={emb.shape[0]}, combine={n_items}")

    titles = [_parse_title(str(recid2combine[i])) for i in range(n_items)]
    tag_counter: collections.Counter[str] = collections.Counter()
    for title in titles:
        for tag in _TAG_RE.findall(title):
            tag_counter[_clean_topic_token(tag)] += 1

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_norm = emb / np.maximum(norms, 1e-8)

    cluster_assign = _fit_minibatch_kmeans(
        emb_norm,
        n_clusters=n_clusters,
        seed=seed,
        batch_size=batch_size,
    )

    sub_assign = np.zeros(n_items, dtype=np.int64)
    for cluster_id in range(n_clusters):
        idx = np.where(cluster_assign == cluster_id)[0]
        if idx.size == 0:
            continue
        local_k = min(n_subclusters, idx.size)
        local_assign = _fit_minibatch_kmeans(
            emb_norm[idx],
            n_clusters=local_k,
            seed=seed + 1000 + cluster_id,
            batch_size=batch_size,
        )
        sub_assign[idx] = local_assign

    prefixes: list[tuple[str, str, str]] = []
    for i, title in enumerate(titles):
        prefixes.append(
            (
                f"cluster_{int(cluster_assign[i]):02d}",
                _extract_topic(title, rare_tag_counts=tag_counter, min_topic_count=min_topic_count),
                f"sub_{int(sub_assign[i]):02d}",
            )
        )

    grouped: dict[tuple[str, str, str], list[int]] = collections.defaultdict(list)
    for i, prefix in enumerate(prefixes):
        grouped[prefix].append(i)

    paths: list[list[str]] = [[] for _ in range(n_items)]
    max_leaf = 0
    for prefix, item_ids in grouped.items():
        for leaf_idx, item_id in enumerate(sorted(item_ids)):
            max_leaf = max(max_leaf, leaf_idx)
            paths[item_id] = [prefix[0], prefix[1], prefix[2], f"leaf_{leaf_idx}"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(paths, ensure_ascii=False), encoding="utf-8")

    per_pos_counts = []
    for pos in range(4):
        per_pos_counts.append(len({p[pos] for p in paths}))

    stats = {
        "n_items": n_items,
        "n_clusters": n_clusters,
        "n_subclusters": n_subclusters,
        "min_topic_count": min_topic_count,
        "per_pos_unique": per_pos_counts,
        "n_raw_hashtags": len(tag_counter),
        "n_topic_nodes": per_pos_counts[1],
        "n_prefixes": len(grouped),
        "max_leaf_index": max_leaf,
        "cluster_size_top10": collections.Counter(cluster_assign.tolist()).most_common(10),
        "topic_top20": collections.Counter(p[1] for p in paths).most_common(20),
        "source": {
            "combine_maps_path": str(combine_maps_path),
            "emb_path": str(emb_path),
        },
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build title-derived MicroLens SemID paths.")
    parser.add_argument("--combine_maps_path", required=True, type=Path)
    parser.add_argument("--emb_path", required=True, type=Path)
    parser.add_argument("--output_path", required=True, type=Path)
    parser.add_argument("--stats_path", required=True, type=Path)
    parser.add_argument("--n_clusters", type=int, default=64)
    parser.add_argument("--n_subclusters", type=int, default=8)
    parser.add_argument("--min_topic_count", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_paths(
        combine_maps_path=args.combine_maps_path,
        emb_path=args.emb_path,
        output_path=args.output_path,
        stats_path=args.stats_path,
        n_clusters=args.n_clusters,
        n_subclusters=args.n_subclusters,
        min_topic_count=args.min_topic_count,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
