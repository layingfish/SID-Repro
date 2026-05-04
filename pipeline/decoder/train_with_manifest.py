

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path


def _normalize_gin_override(override: str) -> str:
    if "=" not in override:
        return override
    key, value = override.split("=", 1)
    if key == "train.save_dir_root":
        stripped = value.strip()
        if (stripped.startswith('"') and stripped.endswith('"')) or (stripped.startswith("'") and stripped.endswith("'")):
            stripped = stripped[1:-1]
        if not stripped.endswith("/"):
            stripped = stripped + "/"
        value = '"' + stripped + '"'
    return f"{key}={value}"


def main():
    parser = argparse.ArgumentParser(description="Scaling Law — TIGER pipeline with manifest IDs")
    parser.add_argument("--cached_ids_path", required=True, type=Path)
    parser.add_argument("--tiger_config_path", required=True, type=Path)
    parser.add_argument("--gin_config", required=True, type=str, help="gin config file for train_decoder")
    parser.add_argument("--gin_override", type=str, nargs="*", default=[], help="additional gin overrides")
    parser.add_argument("--resume_from", type=str, default=None, help="checkpoint .pt path to resume from")
    parser.add_argument(
        "--vg_main_table_t5small_align",
        action="store_true",
        help="Align the training recipe to the VG main-table TIGER-noUID t5-small setup.",
    )
    parser.add_argument(
        "--vg_main_table_align_model_size",
        choices=["t5-small", "t5-base", "t5-large"],
        default="t5-small",
        help="Model size used under VG main-table alignment mode.",
    )
    args = parser.parse_args()


    with open(args.tiger_config_path) as f:
        tiger_config = json.load(f)

    cached_ids_path = str(args.cached_ids_path.resolve())

    release_root = Path(os.environ.get("RECSYS26_ROOT", Path(__file__).resolve().parents[2])).resolve()
    baseline_decoder_root = Path(
        os.environ.get("BASELINE_DECODER_ROOT", release_root / "baselines" / "decoder")
    ).resolve()
    tiger_root = baseline_decoder_root
    common_root = release_root / "pipeline" / "common"
    tokenizer_root = release_root / "pipeline" / "tokenizer"

    for path in (baseline_decoder_root, common_root, tokenizer_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    from manifest_utils import patch_t5_semid_for_manifest, resolve_sid_layout

    sid_layout = resolve_sid_layout(
        tiger_config=tiger_config,
        cached_ids_path=args.cached_ids_path,
    )
    codebook_size = sid_layout["max_codebook_size"]
    sem_id_dim = sid_layout["sem_id_dim"]
    per_pos_sizes = sid_layout["per_pos_sizes"]
    print(f"[Manifest] codebook_size={codebook_size}, sem_id_dim={sem_id_dim}")
    print(f"[Manifest] cached_ids: {cached_ids_path}")
    print(f"[Manifest] per_pos_sizes={per_pos_sizes}")
    patch_t5_semid_for_manifest(
        per_pos_sizes=per_pos_sizes,
        duplicate_policy=sid_layout["duplicate_policy"],
    )


    from cached_id_tokenizer import CachedIdTokenizer

    class MockSemanticIdTokenizer(CachedIdTokenizer):


        def __init__(
            self,
            input_dim=None,
            output_dim=None,
            hidden_dims=None,
            codebook_size=None,
            n_layers=None,
            n_cat_feats=None,
            commitment_weight=None,
            rqvae_weights_path=None,
            rqvae_codebook_normalize=None,
            rqvae_sim_vq=None,
            **kwargs,
        ):
            super().__init__(
                cached_ids_path=cached_ids_path,
                codebook_size=codebook_size if codebook_size is not None else tiger_config["codebook_size"],
                per_pos_sizes=per_pos_sizes,
            )


    import types
    fake_semids = types.ModuleType("modules.tokenizer.semids")
    fake_semids.SemanticIdTokenizer = MockSemanticIdTokenizer
    sys.modules["modules.tokenizer.semids"] = fake_semids

    print("[Patch] SemanticIdTokenizer → MockSemanticIdTokenizer (via sys.modules injection)")


    import tempfile

    gin_config = args.gin_config
    gin_overrides = [
        f"train.vae_codebook_size={codebook_size}",
        f"train.vae_n_layers={sem_id_dim - 1}",
    ]

    if args.vg_main_table_t5small_align:
        gin_config = os.path.join(
            str(baseline_decoder_root),
            "configs",
            "decoder_setrec_yelp_t5small_v2_lr3e4_paper.gin",
        )
        model_root = Path(os.environ.get("MODEL_ROOT", release_root / "models")).resolve()
        hf_model_paths = {
            "t5-small": os.environ.get("HF_MODEL_T5_SMALL", str(model_root / "t5-small")),
            "t5-base": os.environ.get("HF_MODEL_T5_BASE", str(model_root / "t5-base")),
            "t5-large": os.environ.get("HF_MODEL_T5_LARGE", str(model_root / "t5-large")),
        }
        model_overrides = {
            "t5-small": [
                f'train.hf_model_path="{hf_model_paths["t5-small"]}"',
                "train.hf_local_files_only=True",
                "train.batch_size=64",
                "train.gradient_accumulate_every=4",
            ],
            "t5-base": [
                f'train.hf_model_path="{hf_model_paths["t5-base"]}"',
                "train.hf_local_files_only=True",
                "train.batch_size=32",
                "train.gradient_accumulate_every=8",
            ],
            "t5-large": [
                f'train.hf_model_path="{hf_model_paths["t5-large"]}"',
                "train.hf_local_files_only=True",
                "train.batch_size=16",
                "train.gradient_accumulate_every=16",
            ],
        }
        gin_overrides.extend(
            [
                'train.dataset_split="amazon23_vg_tiger_strict"',
                "train.force_dataset_process=False",
                "train.num_user_tokens=0",
                "train.iterations=5000",
                "train.save_model_every=5000",
                "train.learning_rate=0.0003",
                "train.weight_decay=0.035",
            ]
        )
        gin_overrides.extend(model_overrides[args.vg_main_table_align_model_size])
        print(
            f"[Align] Using VG main-table TIGER-noUID recipe "
            f"with model_size={args.vg_main_table_align_model_size}"
        )


    if args.resume_from:
        gin_overrides.append(f'train.pretrained_decoder_path="{args.resume_from}"')
        print(f"[Resume] from {args.resume_from}")

    gin_overrides += [_normalize_gin_override(x) for x in args.gin_override]


    tmp_gin = tempfile.NamedTemporaryFile(mode="w", suffix=".gin", delete=False, dir=tiger_root)
    tmp_gin.write(f'include "{gin_config}"\n\n')
    for override in gin_overrides:
        tmp_gin.write(override + "\n")
    tmp_gin.flush()
    tmp_gin_path = tmp_gin.name
    tmp_gin.close()

    print(f"[Gin] Temp config: {tmp_gin_path}")
    with open(tmp_gin_path) as f:
        print(f.read())

    sys.argv = ["train_decoder.py", tmp_gin_path]


    os.chdir(tiger_root)


    from train_decoder import train
    from modules.utils import parse_config
    parse_config()

    train()


    os.unlink(tmp_gin_path)


if __name__ == "__main__":
    main()
