

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Scaling Law — TIGER export with manifest IDs")
    parser.add_argument("--cached_ids_path", required=True, type=Path)
    parser.add_argument("--tiger_config_path", required=True, type=Path)


    parser.add_argument("--domain", required=True, type=str)
    parser.add_argument("--decoder_ckpt", required=True, type=str)
    parser.add_argument("--dataset_folder", required=True, type=str)
    parser.add_argument("--export_path", required=True, type=str)
    parser.add_argument("--hf_model_path", required=True, type=str)
    parser.add_argument("--beam_size", type=int, default=20)
    parser.add_argument("--num_return_sequences", type=int, default=20)
    parser.add_argument("--topk_items", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--warm_only_export",
        action="store_true",
        help="Export only users whose test targets still contain warm items.",
    )
    parser.add_argument(
        "--warm_items_path",
        type=Path,
        default=None,
        help="Optional warm_item.npy path. Defaults to <dataset_folder>/<domain>/warm_item.npy.",
    )
    parser.add_argument(
        "--testing_dict_path",
        type=Path,
        default=None,
        help="Optional testing_dict.npy path used to select warm-only export users.",
    )
    parser.add_argument(
        "--vg_main_table_t5small_align",
        action="store_true",
        help="Align export to the VG main-table TIGER-noUID t5-small setup.",
    )
    parser.add_argument(
        "--force_num_user_tokens",
        type=int,
        default=None,
        help="Override inferred num_user_tokens (use 0 for noUID).",
    )

    args = parser.parse_args()

    with open(args.tiger_config_path) as f:
        tiger_config = json.load(f)

    duplicate_policy = tiger_config.get("duplicate_policy", "unique")
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
    per_pos_sizes = sid_layout["per_pos_sizes"]
    sem_id_dim = sid_layout["sem_id_dim"]
    codebook_size = sid_layout["max_codebook_size"]
    sid_tokens = sid_layout["n_sid_tokens"]
    print(f"[Manifest] per_pos_sizes={per_pos_sizes}")


    import torch
    ckpt = torch.load(args.decoder_ckpt, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    ckpt_vocab_size = state["encoder.embed_tokens.weight"].shape[0]
    del ckpt, state


    from transformers import AutoTokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path, local_files_only=("/" in args.hf_model_path), use_fast=True)
    base_len = len(base_tokenizer)

    num_user_tokens = 0
    for candidate in range(0, 100000):
        total_special = sid_tokens + candidate
        new_vocab = base_len + total_special

        if new_vocab == ckpt_vocab_size:
            num_user_tokens = candidate
            break
        if new_vocab > ckpt_vocab_size:
            num_user_tokens = max(0, candidate - 1)
            break

    if args.force_num_user_tokens is not None:
        num_user_tokens = int(args.force_num_user_tokens)

    if args.vg_main_table_t5small_align:
        args.domain = "amazon23_vg_tiger_strict"
        args.batch_size = 16
        args.beam_size = 64
        args.num_return_sequences = 64
        args.topk_items = 20
        num_user_tokens = 0
        print("[Align] Using VG main-table TIGER-noUID t5-small export recipe")

    print(f"[Manifest] ckpt_vocab={ckpt_vocab_size}, base={base_len}, sid={sid_tokens}, matched num_user_tokens={num_user_tokens}")
    print(f"[Manifest] codebook_size={codebook_size}, sem_id_dim={sem_id_dim}")
    print(f"[Manifest] ckpt_vocab={ckpt_vocab_size}, inferred num_user_tokens={num_user_tokens}")
    patch_t5_semid_for_manifest(
        per_pos_sizes=per_pos_sizes,
        duplicate_policy=duplicate_policy,
    )

    from cached_id_tokenizer import CachedIdTokenizer

    class MockSemanticIdTokenizer(CachedIdTokenizer):
        def __init__(self, input_dim=None, output_dim=None, hidden_dims=None,
                     codebook_size=None, n_layers=None, n_cat_feats=None,
                     commitment_weight=None, rqvae_weights_path=None,
                     rqvae_codebook_normalize=None, rqvae_sim_vq=None, **kwargs):
            super().__init__(
                cached_ids_path=cached_ids_path,
                codebook_size=codebook_size if codebook_size is not None else tiger_config["codebook_size"],
                per_pos_sizes=per_pos_sizes,
            )

    import modules.tokenizer.semids as semids_module
    semids_module.SemanticIdTokenizer = MockSemanticIdTokenizer

    print("[Patch] SemanticIdTokenizer → MockSemanticIdTokenizer")

    if args.warm_only_export:
        import numpy as np
        import torch
        import data.processed as processed_module

        warm_items_path = args.warm_items_path
        if warm_items_path is None:
            setrec_warm_items_path = Path(os.environ.get("DATA_ROOT", release_root / "data")) / args.domain / "warm_item.npy"
            dataset_warm_items_path = Path(args.dataset_folder) / args.domain / "warm_item.npy"
            warm_items_path = setrec_warm_items_path if setrec_warm_items_path.exists() else dataset_warm_items_path
        if not warm_items_path.exists():
            raise FileNotFoundError(f"warm_only_export requested but warm item file is missing: {warm_items_path}")

        warm_raw = np.load(warm_items_path, allow_pickle=True)
        try:
            warm_obj = warm_raw.item()
        except Exception:
            warm_obj = warm_raw.tolist()
        if hasattr(warm_obj, "keys"):
            warm_ids = {int(x) for x in warm_obj.keys()}
        else:
            warm_ids = {int(x) for x in warm_obj}
        if not warm_ids:
            raise ValueError(f"No warm item ids loaded from {warm_items_path}")

        data_dir_candidates = [
            Path(os.environ.get("DATA_ROOT", release_root / "data")) / args.domain,
            Path(args.dataset_folder) / args.domain,
        ]
        testing_dict_path = args.testing_dict_path
        if testing_dict_path is None:
            testing_dict_path = next(
                (p / "testing_dict.npy" for p in data_dir_candidates if (p / "testing_dict.npy").exists()),
                None,
            )
        if testing_dict_path is None:
            searched = ", ".join(str(p / "testing_dict.npy") for p in data_dir_candidates)
            raise FileNotFoundError(
                f"warm_only_export requested but testing_dict.npy is missing. searched: {searched}"
            )
        testing_dict_path = Path(testing_dict_path)

        test_dict = np.load(testing_dict_path, allow_pickle=True).item()
        full_warm_user_ids = set()
        full_warm_gt_items = 0
        for uid, items in test_dict.items():
            if isinstance(items, np.ndarray):
                values = items.tolist()
            elif isinstance(items, (list, tuple, set)):
                values = list(items)
            elif items is None:
                values = []
            else:
                values = [items]
            kept = [int(x) for x in values if int(x) in warm_ids]
            if kept:
                full_warm_user_ids.add(int(uid))
                full_warm_gt_items += len(kept)
        if not full_warm_user_ids:
            raise ValueError(f"No full warm GT users loaded from {testing_dict_path}")

        OriginalSeqData = processed_module.SeqData

        class WarmOnlySeqData(OriginalSeqData):
            def __init__(self, *seq_args, **seq_kwargs):
                super().__init__(*seq_args, **seq_kwargs)
                if getattr(self, "split", None) != "test":
                    return

                export_user_ids = self.sequence_data.get("rawUserId")
                if export_user_ids is None:
                    export_user_ids = self.sequence_data.get("userId")
                if export_user_ids is None:
                    raise KeyError("SeqData.sequence_data contains neither rawUserId nor userId")

                n_before = len(self)
                if isinstance(export_user_ids, torch.Tensor):
                    uid_0based = export_user_ids.detach().to(torch.long).cpu().reshape(-1) - 1
                    keep_set_tensor = torch.tensor(sorted(full_warm_user_ids), dtype=torch.long)
                    keep_mask = torch.isin(uid_0based, keep_set_tensor)
                    keep_index = keep_mask.nonzero(as_tuple=False).reshape(-1)
                    keep_list = keep_mask.tolist()
                else:
                    keep_list = []
                    for uid in export_user_ids:
                        uid_value = uid.item() if hasattr(uid, "item") else uid
                        keep_list.append((int(uid_value) - 1) in full_warm_user_ids)
                    keep_index = None

                n_after = int(sum(bool(x) for x in keep_list))
                if n_after <= 0:
                    raise ValueError(f"warm_only_export removed all users for domain={args.domain}")

                for key, value in list(self.sequence_data.items()):
                    if isinstance(value, torch.Tensor) and value.shape[0] == n_before:
                        if keep_index is not None:
                            self.sequence_data[key] = value.index_select(0, keep_index.to(value.device))
                        else:
                            self.sequence_data[key] = value[torch.tensor(keep_list, dtype=torch.bool, device=value.device)]
                    elif isinstance(value, np.ndarray) and value.shape[0] == n_before:
                        self.sequence_data[key] = value[np.asarray(keep_list, dtype=bool)]
                    elif isinstance(value, list) and len(value) == n_before:
                        self.sequence_data[key] = [x for x, keep in zip(value, keep_list) if keep]

                print(
                    f"[WarmOnlyExport] users {n_before} -> {n_after} "
                    f"using full_warm_gt_users={len(full_warm_user_ids)} "
                    f"full_warm_gt_items={full_warm_gt_items} "
                    f"warm_items={len(warm_ids)} "
                    f"testing_dict={testing_dict_path} warm_items_path={warm_items_path}"
                )

        processed_module.SeqData = WarmOnlySeqData
        print("[Patch] SeqData → WarmOnlySeqData")


    sys.argv = [
        "test_only_tiger.py",
        "--domain", args.domain,
        "--backbone", "t5_small",
        "--rqvae_ckpt", "UNUSED",
        "--decoder_ckpt", args.decoder_ckpt,
        "--dataset_folder", args.dataset_folder,
        "--export_path", args.export_path,
        "--hf_model_path", args.hf_model_path,
        "--beam_size", str(args.beam_size),
        "--num_return_sequences", str(args.num_return_sequences),
        "--topk_items", str(args.topk_items),
        "--batch_size", str(args.batch_size),
        "--gpu", str(args.gpu),
        "--vae_codebook_size", str(codebook_size),
        "--vae_n_layers", str(sem_id_dim - 1),
        "--num_user_tokens", str(num_user_tokens),
    ]
    if args.resume:
        sys.argv.append("--resume")

    print(f"[Args] {sys.argv}")

    os.chdir(tiger_root)

    from test_only_tiger import main as tiger_main
    tiger_main()


if __name__ == "__main__":
    main()
