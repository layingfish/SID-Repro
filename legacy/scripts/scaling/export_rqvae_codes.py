#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler


REPO_ROOT = Path(__file__).resolve().parents[2]
RQ_ROOT = REPO_ROOT / "third_party" / "RQ_VAE_Recommender"
SCRIPTS_ROOT = REPO_ROOT / "scripts"

if str(RQ_ROOT) not in sys.path:
    sys.path.insert(0, str(RQ_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from data.processed import ItemData, RecDataset
from data.utils import batch_to
from modules.rqvae import RqVae
from tokenizer_metrics_utils import export_tokenizer_metrics


def _load_enum(name: str) -> RecDataset:
    try:
        return getattr(RecDataset, name)
    except AttributeError as exc:
        valid = ", ".join(x.name for x in RecDataset)
        raise ValueError(f"Unknown dataset enum {name!r}. Valid values: {valid}") from exc


def _dump_codes_with_mode(raw_codes: np.ndarray, mode: str) -> tuple[np.ndarray, dict[str, int | float | str | bool]]:
    n_items = int(raw_codes.shape[0])
    code_counter = Counter(tuple(row) for row in raw_codes.tolist())
    n_unique = len(code_counter)
    max_collision = int(code_counter.most_common(1)[0][1]) if code_counter else 0
    collision_rate = float((n_items - n_unique) / max(n_items, 1))

    if mode == "suffix":
        code_index: dict[tuple[int, ...], int] = {}
        suffix = np.zeros((n_items, 1), dtype=np.int64)
        for i in range(n_items):
            key = tuple(raw_codes[i].tolist())
            suffix[i, 0] = code_index.get(key, 0)
            code_index[key] = int(suffix[i, 0]) + 1
        cached_ids = np.concatenate([raw_codes, suffix], axis=1)
        meta = {
            "duplicate_policy": "unique",
            "dedup_suffix": True,
            "max_suffix": int(suffix.max()) if suffix.size else 0,
            "n_unique_codes": int(n_unique),
            "collision_rate": float(collision_rate),
            "max_collision": int(max_collision),
        }
    else:
        cached_ids = raw_codes.copy()
        meta = {
            "duplicate_policy": "first",
            "dedup_suffix": False,
            "max_suffix": 0,
            "n_unique_codes": int(n_unique),
            "collision_rate": float(collision_rate),
            "max_collision": int(max_collision),
        }
    return cached_ids, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RQ-VAE raw/suffix semantic codes as TIGER-compatible cached_ids.")
    parser.add_argument("--rqvae_ckpt", required=True, type=Path)
    parser.add_argument("--dataset_folder", required=True, type=Path)
    parser.add_argument("--dataset_enum", default="AMAZON", help="RecDataset enum name, e.g. AMAZON")
    parser.add_argument("--dataset_split", required=True, help="Dataset split/category, e.g. amazon23_vg")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--mode", choices=["suffix", "raw"], default="raw")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vae_input_dim", type=int, default=768)
    parser.add_argument("--vae_embed_dim", type=int, default=32)
    parser.add_argument("--vae_hidden_dims", default="512,256,128")
    parser.add_argument("--vae_codebook_size", type=int, default=256)
    parser.add_argument("--vae_n_layers", type=int, required=True)
    parser.add_argument("--vae_n_cat_feats", type=int, default=0)
    parser.add_argument("--commitment_weight", type=float, default=0.25)
    parser.add_argument("--rqvae_codebook_normalize", action="store_true")
    parser.add_argument("--rqvae_sim_vq", action="store_true")
    args = parser.parse_args()

    hidden_dims = [int(x) for x in args.vae_hidden_dims.split(",") if x.strip()]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = ItemData(
        root=str(args.dataset_folder),
        dataset=_load_enum(args.dataset_enum),
        force_process=False,
        train_test_split="all",
        split=args.dataset_split,
    )

    model = RqVae(
        input_dim=args.vae_input_dim,
        embed_dim=args.vae_embed_dim,
        hidden_dims=hidden_dims,
        codebook_size=args.vae_codebook_size,
        codebook_kmeans_init=False,
        codebook_normalize=args.rqvae_codebook_normalize,
        codebook_sim_vq=args.rqvae_sim_vq,
        n_layers=args.vae_n_layers,
        n_cat_features=args.vae_n_cat_feats,
        commitment_weight=args.commitment_weight,
    ).to(device)
    state = torch.load(args.rqvae_ckpt, map_location=device, weights_only=False)
    model_state = state["model"] if "model" in state else state
    model.load_state_dict(model_state, strict=True)
    model.eval()

    sampler = BatchSampler(SequentialSampler(range(len(dataset))), batch_size=512, drop_last=False)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=None, collate_fn=lambda batch: batch)
    raw_batches = []
    with torch.no_grad():
        for batch in dataloader:
            sem_ids = model.get_semantic_ids(batch_to(batch, device).x).sem_ids
            raw_batches.append(sem_ids.detach().cpu().numpy().astype(np.int64))

    raw_codes = np.concatenate(raw_batches, axis=0)
    cached_ids, mode_meta = _dump_codes_with_mode(raw_codes, args.mode)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "raw_codes.npy", raw_codes)
    np.save(args.output_dir / "cached_ids.npy", cached_ids)

    # RQ-VAE 导出的 raw code 可能是稀疏标签，例如只用了 240 个不同取值，
    # 但实际 label id 仍可能落在 [0, 255]。这里必须记录为 `max(code)+1`，
    # 否则下游 TIGER 侧会误以为该位置只有 240 个合法 token，导致 layout
    # 非法、词表构造错误，最终污染 decoder 训练与导出。
    per_pos_sizes = [
        int(cached_ids[:, pos].max()) + 1
        for pos in range(cached_ids.shape[1])
    ]
    tiger_config = {
        "n_items": int(cached_ids.shape[0]),
        "sem_id_dim": int(cached_ids.shape[1]),
        "codebook_size": int(max(per_pos_sizes) if per_pos_sizes else args.vae_codebook_size),
        "per_pos_sizes": per_pos_sizes,
        "id_type": f"I02-rqvae-{args.mode}",
        "id_name": "RQ-VAE Semantic Code" if args.mode == "suffix" else "RQ-VAE Raw Semantic Code",
        "source": str(args.rqvae_ckpt),
        "dataset_split": args.dataset_split,
        "semantic_code_length": int(args.vae_n_layers),
        "decoder_code_length": int(cached_ids.shape[1]),
        **mode_meta,
    }
    (args.output_dir / "tiger_config.json").write_text(
        json.dumps(tiger_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    meta = {
        "method_family": "RQ-VAE",
        "paper_variant": "TIGER-mainline",
        "implementation_variant": "official-rqvae-recommender",
        "embedding_source": f"{args.dataset_folder}/{args.dataset_split}",
        "embedding_dim": args.vae_input_dim,
        "n_items": int(cached_ids.shape[0]),
        "n_levels": int(args.vae_n_layers),
        "codebook_size_per_level": [args.vae_codebook_size] * args.vae_n_layers,
        "balanced_assignment": False,
        "dedup_suffix": bool(mode_meta["dedup_suffix"]),
        "tokenizer_ckpt_rule": "explicit",
        "dataset_split": args.dataset_split,
        "rqvae_ckpt": str(args.rqvae_ckpt),
        "semantic_code_length": int(args.vae_n_layers),
        "duplicate_policy": str(mode_meta["duplicate_policy"]),
    }
    codebook_sizes = [args.vae_codebook_size] * args.vae_n_layers
    if mode_meta["dedup_suffix"]:
        codebook_sizes.append(int(mode_meta["max_suffix"]) + 1)
    summary = export_tokenizer_metrics(
        output_dir=args.output_dir,
        codes=cached_ids,
        meta=meta,
        codebook_size_per_level=codebook_sizes,
    )

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "raw_codes_shape": list(raw_codes.shape),
                "cached_ids_shape": list(cached_ids.shape),
                "mode": args.mode,
                "first_level_used_codes": summary["first_level_used_codes"],
                "first_level_usage_rate": summary["first_level_usage_rate"],
                "collision_rate": summary["collision_rate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
