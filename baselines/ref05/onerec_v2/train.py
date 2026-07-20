import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from . import data as data_utils
from .data import (
    OneRecEvalDataset,
    OneRecFutureTrainDataset,
    OneRecNextItemTrainDataset,
    OneRecSessionTrainDataset,
    OneRecTrainDataset,
    build_code_trie,
    build_code_tuple_to_items,
    build_length_distribution,
    build_popularity,
    build_trie,
    build_valid_token_mask,
    collate_eval,
    collate_next_item_train,
    collate_train,
    load_semantic_ids,
    load_split_dicts,
)
from .generate import generate_predictions
from .model import create_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(results, gt_lookup, k_list=(5, 10)):
    metrics = {f"R@{k}": 0.0 for k in k_list}
    metrics.update({f"N@{k}": 0.0 for k in k_list})

    n = 0
    for row in results:
        user_id = int(row["user_id"])
        pred = [int(x) for x in row["predicted_items"]]
        gt = [int(x) for x in gt_lookup.get(user_id, [])]
        if not gt:
            continue

        gt_set = set(gt)
        n += 1
        for k in k_list:
            topk = pred[:k]
            hits = len(set(topk) & gt_set)
            metrics[f"R@{k}"] += hits / max(len(gt_set), 1)

            dcg = 0.0
            for idx, item_id in enumerate(topk):
                if item_id in gt_set:
                    dcg += 1.0 / np.log2(idx + 2)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt_set), k)))
            metrics[f"N@{k}"] += dcg / idcg if idcg > 0 else 0.0

    n = max(n, 1)
    for key in metrics:
        metrics[key] /= n
    return metrics


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def set_requires_grad_for_phase(model, embedding_only: bool):
    for _, param in model.named_parameters():
        param.requires_grad = not embedding_only

    if embedding_only:
        if hasattr(model, "shared"):
            model.shared.weight.requires_grad = True
        if hasattr(model, "lm_head") and getattr(model, "lm_head", None) is not None:
            model.lm_head.weight.requires_grad = True


@torch.no_grad()
def run_eval(
    model,
    eval_loader,
    gt_lookup,
    tuple_to_items,
    trie,
    popularity_counts,
    popular_items,
    device,
    beam_size: int,
    topk: int,
    min_session_size: int,
    max_session_size: int,
    generate_batch_size: int,
    architecture: str,
):
    results = generate_predictions(
        model=model,
        eval_loader=eval_loader,
        tuple_to_items=tuple_to_items,
        trie=trie,
        popularity_counts=popularity_counts,
        popular_items=popular_items,
        device=str(device),
        beam_size=beam_size,
        topk=topk,
        min_session_size=min_session_size,
        max_session_size=max_session_size,
        generate_batch_size=generate_batch_size,
        architecture=architecture,
    )
    return compute_metrics(results, gt_lookup, k_list=(5, 10))


def _build_seq2seq_train_dataset(args, train_dict, val_dict, test_dict, item_to_tokens):
    if args.train_sessions_path:
        train_ds = OneRecSessionTrainDataset(
            samples_path=args.train_sessions_path,
            item_to_tokens=item_to_tokens,
            max_hist=args.max_hist,
            min_hist=args.min_hist,
        )
        if args.min_session_size is None:
            args.min_session_size = int(train_ds.min_target_items)
        if args.max_session_size is None:
            args.max_session_size = int(train_ds.max_target_items)
        print(
            "training mode: session-jsonl "
            f"(samples={len(train_ds)} min_target_items={train_ds.min_target_items} "
            f"max_target_items={train_ds.max_target_items})"
        )
        return train_ds

    if args.train_future_suffix:
        length_dicts = [val_dict]
        if args.future_length_source == "valtest":
            length_dicts = [val_dict, test_dict]
        target_lengths, target_probs = build_length_distribution(
            length_dicts,
            n_items=item_to_tokens.shape[0],
            min_len=args.future_min_items,
            max_len=args.future_max_items,
        )
        if target_lengths is None or target_probs is None:
            raise ValueError("failed to build empirical target length distribution from eval splits")

        length_summary = ", ".join(
            f"{int(length)}:{prob:.3f}" for length, prob in zip(target_lengths.tolist(), target_probs.tolist())
        )
        print(
            "future length distribution "
            f"({args.future_length_source}, clipped_to={args.future_max_items}): {length_summary}"
        )

        train_ds = OneRecFutureTrainDataset(
            user_dict=train_dict,
            item_to_tokens=item_to_tokens,
            max_hist=args.max_hist,
            min_hist=args.min_hist,
            min_target_items=args.future_min_items,
            max_target_items=args.future_max_items,
            length_strategy=args.future_length_strategy,
            target_lengths=target_lengths,
            target_probs=target_probs,
            seed=args.seed,
        )
        if args.min_session_size is None:
            args.min_session_size = int(train_ds.min_target_items)
        if args.max_session_size is None:
            args.max_session_size = int(train_ds.max_target_items)
        print(
            "training mode: future-suffix "
            f"(samples={len(train_ds)} min_target_items={train_ds.min_target_items} "
            f"max_target_items={train_ds.max_target_items} "
            f"strategy={args.future_length_strategy})"
        )
        return train_ds

    train_ds = OneRecTrainDataset(
        user_dict=train_dict,
        item_to_tokens=item_to_tokens,
        max_hist=args.max_hist,
        min_hist=args.min_hist,
        session_size=args.session_size,
    )
    if args.min_session_size is None:
        args.min_session_size = int(args.session_size)
    if args.max_session_size is None:
        args.max_session_size = int(args.session_size)
    print(f"training mode: fixed-window (session_size={args.session_size})")
    return train_ds


def train(args):
    set_seed(args.seed)
    data_utils.set_runtime_config(
        num_levels=args.num_levels,
        codebook_width=args.codebook_width,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and (args.fp16 or args.bf16)
    amp_dtype = torch.float16 if args.fp16 else torch.bfloat16

    train_dict, val_dict, test_dict = load_split_dicts(args.data_dir)
    item_to_tokens, item_to_codes, token_tuple_to_items = load_semantic_ids(args.semantic_ids_path)

    if args.architecture == "hierarchical":
        trie = build_code_trie(item_to_codes)
        tuple_to_items = build_code_tuple_to_items(item_to_codes)
    else:
        trie = build_trie(item_to_tokens)
        tuple_to_items = token_tuple_to_items

    popularity_counts, popular_items = build_popularity(train_dict, item_to_tokens.shape[0])

    if args.architecture == "hierarchical":
        train_ds = OneRecNextItemTrainDataset(
            user_dict=train_dict,
            item_to_tokens=item_to_tokens,
            item_to_codes=item_to_codes,
            max_hist=args.max_hist,
            min_hist=args.min_hist,
        )
        args.min_session_size = 1 if args.min_session_size is None else args.min_session_size
        args.max_session_size = 1 if args.max_session_size is None else args.max_session_size
        print(
            "training mode: next-item hierarchical "
            f"(samples={len(train_ds)} num_levels={args.num_levels} codebook_width={args.codebook_width})"
        )
    else:
        train_ds = _build_seq2seq_train_dataset(args, train_dict, val_dict, test_dict, item_to_tokens)

    if args.min_session_size <= 0:
        raise ValueError("min_session_size must be positive")
    if args.max_session_size < args.min_session_size:
        raise ValueError("max_session_size must be >= min_session_size")
    print(
        "generation session bounds: "
        f"min_session_size={args.min_session_size} "
        f"max_session_size={args.max_session_size}"
    )

    val_ds = OneRecEvalDataset(
        train_dict=train_dict,
        val_dict=val_dict,
        test_dict=test_dict,
        item_to_tokens=item_to_tokens,
        split="val",
        max_hist=args.max_hist,
        sample_users=args.eval_users,
        seed=args.seed,
    )

    if args.architecture == "hierarchical":
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_next_item_train,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_train,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
        collate_fn=collate_eval,
    )

    model = create_model(
        pretrained=args.pretrained,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        d_kv=args.d_kv,
        dropout=args.dropout,
        architecture=args.architecture,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.max_steps, args.warmup_ratio)
    scaler = GradScaler(enabled=use_amp and args.fp16)

    embedding_only_phase = args.architecture == "seq2seq" and args.freeze_non_embedding_steps > 0
    if embedding_only_phase:
        set_requires_grad_for_phase(model, embedding_only=True)
        print(f"phase: embedding-only optimization for first {args.freeze_non_embedding_steps} steps")

    os.makedirs(args.output_dir, exist_ok=True)
    best_r10 = -1.0
    no_improve = 0
    train_iter = iter(train_loader)
    valid_token_mask_cache = {}

    def get_valid_token_mask(seq_len: int):
        seq_len = int(seq_len)
        if seq_len not in valid_token_mask_cache:
            valid_token_mask_cache[seq_len] = build_valid_token_mask(seq_len).to(device)
        return valid_token_mask_cache[seq_len]

    for step in range(1, args.max_steps + 1):
        if embedding_only_phase and step == args.freeze_non_embedding_steps + 1:
            set_requires_grad_for_phase(model, embedding_only=False)
            embedding_only_phase = False
            print(f"phase: unfroze full model at step={step}")

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        model.train()

        with autocast(dtype=amp_dtype, enabled=use_amp):
            if args.architecture == "hierarchical":
                _, logits_list = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    target_codes=batch["target_codes"],
                )
                loss = 0.0
                for level, logits in enumerate(logits_list):
                    loss = loss + F.cross_entropy(logits, batch["target_codes"][:, level])
            else:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                logits = outputs.logits

                valid_token_mask = get_valid_token_mask(logits.shape[1])
                time_steps = min(logits.shape[1], valid_token_mask.shape[0])
                logits = logits.clone()
                logits[:, :time_steps] = logits[:, :time_steps].masked_fill(
                    ~valid_token_mask[:time_steps].unsqueeze(0),
                    -1e9,
                )

                loss = F.cross_entropy(
                    logits.reshape(-1, data_utils.VOCAB_SIZE),
                    batch["labels"].reshape(-1),
                    ignore_index=-100,
                )

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        if step % args.log_every == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"step={step} loss={loss.item():.4f} lr={lr:.6e}")

        if step % args.eval_every == 0 or step == args.max_steps:
            model.eval()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            metrics = run_eval(
                model=model,
                eval_loader=val_loader,
                gt_lookup=val_dict,
                tuple_to_items=tuple_to_items,
                trie=trie,
                popularity_counts=popularity_counts,
                popular_items=popular_items,
                device=device,
                beam_size=args.eval_beam_size,
                topk=args.eval_topk,
                min_session_size=args.min_session_size,
                max_session_size=args.max_session_size,
                generate_batch_size=args.eval_generate_batch_size,
                architecture=args.architecture,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

            print(
                "eval "
                f"R@5={metrics['R@5']:.4f} "
                f"R@10={metrics['R@10']:.4f} "
                f"N@5={metrics['N@5']:.4f} "
                f"N@10={metrics['N@10']:.4f}"
            )

            if metrics["R@10"] > best_r10:
                best_r10 = metrics["R@10"]
                no_improve = 0
                ckpt_path = os.path.join(args.output_dir, "best_model.pt")
                save_args = {
                    **vars(args),
                    "num_levels": data_utils.NUM_LEVELS,
                    "codebook_width": data_utils.CODEBOOK_WIDTH,
                }
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "metric": metrics,
                        "args": save_args,
                    },
                    ckpt_path,
                )
                with open(os.path.join(args.output_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
                    json.dump({"step": step, **metrics}, f, indent=2)
                print(f"saved best checkpoint to {ckpt_path}")
            else:
                no_improve += 1
                print(f"no improvement ({no_improve}/{args.patience})")

            if no_improve >= args.patience:
                print("early stopping")
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--semantic_ids_path", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument(
        "--architecture",
        type=str,
        default="seq2seq",
        choices=["seq2seq", "hierarchical"],
    )
    parser.add_argument("--pretrained", type=str, default="t5-small")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_decoder_layers", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_kv", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--num_levels", type=int, default=data_utils.NUM_LEVELS)
    parser.add_argument("--codebook_width", type=int, default=data_utils.CODEBOOK_WIDTH)
    parser.add_argument("--session_size", type=int, default=5)
    parser.add_argument("--train_sessions_path", type=str, default=None)
    parser.add_argument("--train_future_suffix", action="store_true")
    parser.add_argument("--min_session_size", type=int, default=None)
    parser.add_argument("--max_session_size", type=int, default=None)
    parser.add_argument("--future_min_items", type=int, default=1)
    parser.add_argument("--future_max_items", type=int, default=5)
    parser.add_argument(
        "--future_length_strategy",
        type=str,
        default="eval_empirical",
        choices=["eval_empirical", "uniform", "max"],
    )
    parser.add_argument(
        "--future_length_source",
        type=str,
        default="val",
        choices=["val", "valtest"],
    )
    parser.add_argument("--max_hist", type=int, default=256)
    parser.add_argument("--min_hist", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max_ss", type=float, default=0.0, help="Reserved for compatibility; hierarchical path keeps teacher forcing only.")

    parser.add_argument("--eval_beam_size", type=int, default=128)
    parser.add_argument("--eval_topk", type=int, default=20)
    parser.add_argument("--eval_users", type=int, default=None)
    parser.add_argument("--eval_generate_batch_size", type=int, default=4)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--freeze_non_embedding_steps",
        type=int,
        default=0,
        help="Train only semantic-ID embeddings for the first N steps, then unfreeze the full T5 model.",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
