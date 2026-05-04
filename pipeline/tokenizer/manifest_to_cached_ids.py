

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np

RELEASE_ROOT = Path(__file__).resolve().parents[2]
COMMON_ROOT = RELEASE_ROOT / "pipeline" / "common"
if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))
from manifest_utils import load_manifest


def manifest_to_tiger_format(manifest_dir: Path, output_dir: Path) -> dict:


    manifest = load_manifest(manifest_dir)
    i2t = manifest["item_to_token_seq"]
    meta = manifest["meta"]

    n_items = len(i2t)
    target_len = meta["target_len"]


    per_pos_tokens: list[dict[int, int]] = [dict() for _ in range(target_len)]

    for item_id in range(n_items):
        seq = i2t[item_id]
        for pos in range(target_len):
            tok = seq[pos]
            if tok not in per_pos_tokens[pos]:
                per_pos_tokens[pos][tok] = len(per_pos_tokens[pos])


    per_pos_sizes = [len(m) for m in per_pos_tokens]
    codebook_size = max(per_pos_sizes)

    print(f"Per-position vocab sizes: {per_pos_sizes}")
    print(f"Unified codebook_size: {codebook_size}")


    cached_ids = np.zeros((n_items, target_len), dtype=np.int64)
    for item_id in range(n_items):
        seq = i2t[item_id]
        for pos in range(target_len):
            cached_ids[item_id, pos] = per_pos_tokens[pos][seq[pos]]


    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "cached_ids.npy", cached_ids)

    code_counter = Counter(tuple(row.tolist()) for row in cached_ids.tolist())
    n_unique_codes = len(code_counter)
    max_collision = int(code_counter.most_common(1)[0][1]) if code_counter else 0
    collision_rate = float((n_items - n_unique_codes) / max(n_items, 1))


    code_to_token = []
    for pos in range(target_len):
        reverse = {v: k for k, v in per_pos_tokens[pos].items()}
        code_to_token.append(reverse)


    token_to_code = [dict(m) for m in per_pos_tokens]

    tiger_config = {
        "n_items": n_items,
        "sem_id_dim": target_len,
        "codebook_size": codebook_size,
        "per_pos_sizes": per_pos_sizes,
        "id_type": meta.get("id_type", "unknown"),
        "id_name": meta.get("id_name", "unknown"),
        "manifest_dir": str(manifest_dir),
        "n_unique_codes": n_unique_codes,
        "collision_rate": collision_rate,
        "max_collision": max_collision,
        "source": meta.get("source", meta.get("description", "manifest_to_cached_ids")),
    }

    (output_dir / "tiger_config.json").write_text(
        json.dumps(tiger_config, indent=2), encoding="utf-8"
    )


    mapping = {
        "token_to_code": [{str(k): v for k, v in m.items()} for m in token_to_code],
        "code_to_token": [{str(k): v for k, v in m.items()} for m in code_to_token],
    }
    (output_dir / "token_mapping.json").write_text(
        json.dumps(mapping), encoding="utf-8"
    )

    print(f"[OK] cached_ids: {cached_ids.shape}, codebook_size={codebook_size}, sem_id_dim={target_len}")
    print(f"[OK] Saved to {output_dir}")

    return tiger_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()

    manifest_to_tiger_format(args.manifest_dir, args.output_dir)


if __name__ == "__main__":
    main()
