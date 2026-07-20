#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("[CMD]", " ".join(shlex.quote(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run scaling tokenizer under the exact VG main-table TIGER-noUID recipe."
    )
    parser.add_argument("--cached_ids_path", required=True, type=Path)
    parser.add_argument("--tiger_config_path", required=True, type=Path)
    parser.add_argument("--decoder_output_root", required=True, type=Path)
    parser.add_argument("--decoder_ckpt", default="", help="Optional checkpoint path for export-only mode")
    parser.add_argument("--export_path", default="", help="Optional prediction path for export")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_size", choices=["M01-t5s", "M02-t5b", "M03-t5l"], default="M01-t5s")
    parser.add_argument(
        "--gin_override",
        nargs="*",
        default=[],
        help="Additional gin overrides forwarded to train_with_manifest.py",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / "train_with_manifest.py"
    export_script = script_dir / "export_with_manifest.py"

    model_size_map = {
        "M01-t5s": ("t5-small", "/data/xqp_data/RecSys26/models/hf/transformers/models--t5-small/snapshots/df1b051c49625cf57a3d0d8d3863ed4d13564fe4"),
        "M02-t5b": ("t5-base", "/data/xqp_data/RecSys26/models/hf/transformers/models--t5-base/snapshots/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"),
        "M03-t5l": ("t5-large", "/data/xqp_data/RecSys26/models/hf/transformers/models--t5-large/snapshots/150ebc2c4b72291e770f58e6057481c8d2ed331a"),
    }
    align_model_size, hf_model_path = model_size_map[args.model_size]

    save_dir_root = str(args.decoder_output_root)
    if not save_dir_root.endswith("/"):
        save_dir_root += "/"
    gin_overrides = [
        f'train.save_dir_root="{save_dir_root}"',
        *args.gin_override,
    ]
    train_cmd = [
        "python3",
        str(train_script),
        "--cached_ids_path",
        str(args.cached_ids_path),
        "--tiger_config_path",
        str(args.tiger_config_path),
        "--gin_config",
        "UNUSED",
        "--vg_main_table_t5small_align",
        "--vg_main_table_align_model_size",
        align_model_size,
        "--gin_override",
        *gin_overrides,
    ]
    if not args.skip_train:
        run(train_cmd)

    if args.decoder_ckpt and args.export_path:
        export_cmd = [
            "python3",
            str(export_script),
            "--cached_ids_path",
            str(args.cached_ids_path),
            "--tiger_config_path",
            str(args.tiger_config_path),
            "--domain",
            "amazon23_vg",
            "--decoder_ckpt",
            args.decoder_ckpt,
            "--dataset_folder",
            "/data/xqp_data/RecSys26/datasets/rqvae_recommender/setrec",
            "--export_path",
            args.export_path,
            "--hf_model_path",
            hf_model_path,
            "--gpu",
            str(args.gpu),
            "--vg_main_table_t5small_align",
        ]
        run(export_cmd)


if __name__ == "__main__":
    main()
