#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel


def load_amrit_module(repo_dir: Path):
    script_path = repo_dir / "scripts" / "run_microlens_tiger.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    sys.path.insert(0, str(repo_dir))
    sys.path.insert(0, str(script_path.parent))
    spec = importlib.util.spec_from_file_location("amrit_run_microlens_tiger", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resume-capable t5 training wrapper for amrit_tiger")
    parser.add_argument("--repo_dir", required=True, type=Path)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--dataset", default="amazon23_vg")
    parser.add_argument("--seed", type=int, default=20260311)
    parser.add_argument("--item_semantic_ids", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--hf_model_path", default="")
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--num_user_tokens", type=int, default=2000)
    parser.add_argument("--max_history_items", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--per_device_batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--save_last", action="store_true")
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_eval_examples", type=int, default=None)
    parser.add_argument("--resume_ckpt", default="", help="Optional explicit resume checkpoint path")
    parser.add_argument("--no_auto_resume", action="store_true", help="Disable auto resume from output_dir/last_resume.pt")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    mod = load_amrit_module(args.repo_dir)

    distributed, rank, local_rank, world_size = mod.maybe_init_distributed()
    try:
        mod.set_seed(args.seed + rank)
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        artifacts = mod.load_microlens_artifacts(args.data_root, dataset=args.dataset)
        item_semantic_ids = mod.load_item_semantic_ids(args.item_semantic_ids)
        position_sizes = mod.get_position_sizes(item_semantic_ids)
        output_dir = mod.ensure_dir(args.output_dir)

        hf_model_path = args.hf_model_path or mod.DEFAULT_HF_MODEL_PATH
        tokenizer, model, token_tables = mod.build_t5_tokenizer_and_model(
            hf_model_path=hf_model_path,
            position_sizes=position_sizes,
            num_user_tokens=args.num_user_tokens,
            local_files_only=args.local_files_only,
        )
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer missing pad_token_id")

        train_examples = mod.build_training_examples(artifacts.training_dict, max_history_items=args.max_history_items)
        val_examples = mod.build_validation_examples(
            artifacts.training_dict,
            artifacts.validation_dict,
            max_history_items=args.max_history_items,
        )
        if args.max_train_examples is not None:
            train_examples = train_examples[: args.max_train_examples]
        if args.max_eval_examples is not None:
            val_examples = val_examples[: args.max_eval_examples]

        encoded_train_examples = mod.encode_next_item_examples(
            train_examples,
            item_semantic_ids=item_semantic_ids,
            token_tables=token_tables,
        )
        encoded_val_examples = mod.encode_next_item_examples(
            val_examples,
            item_semantic_ids=item_semantic_ids,
            token_tables=token_tables,
        )

        train_dataset = mod.EncodedExamplesDataset(encoded_train_examples)
        val_dataset = mod.EncodedExamplesDataset(encoded_val_examples)
        collate_fn = lambda batch: mod.collate_encoded_examples(batch, pad_token_id=int(pad_token_id))

        train_loader, train_sampler = mod.build_dataloader(
            train_dataset,
            batch_size=args.per_device_batch_size,
            distributed=distributed,
            shuffle=True,
            seed=args.seed,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
        val_loader, _ = mod.build_dataloader(
            val_dataset,
            batch_size=args.eval_batch_size,
            distributed=distributed,
            shuffle=False,
            seed=args.seed,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )

        resume_state_path = Path(args.resume_ckpt) if args.resume_ckpt else output_dir / "last_resume.pt"
        auto_resume = not args.no_auto_resume
        resume_payload = None
        if auto_resume and resume_state_path.exists():
            resume_payload = torch.load(resume_state_path, map_location="cpu")
            model.load_state_dict(resume_payload["model_state_dict"], strict=True)

        model = model.to(device)
        if distributed:
            model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        steps_per_epoch = math.ceil(len(train_loader) / max(1, args.gradient_accumulation_steps))
        total_steps = steps_per_epoch * args.epochs
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = mod.get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(1, total_steps),
        )

        start_epoch = 0
        best_val_loss = float("inf")
        best_metrics = None
        elapsed_offset = 0.0
        if resume_payload is not None:
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
            scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
            start_epoch = int(resume_payload["epoch"]) + 1
            best_val_loss = float(resume_payload.get("best_val_loss", best_val_loss))
            best_metrics = resume_payload.get("best_metrics")
            elapsed_offset = float(resume_payload.get("elapsed_seconds", 0.0))
            if mod.is_main_process(rank):
                print(
                    json.dumps(
                        {
                            "stage": "resume",
                            "resume_ckpt": str(resume_state_path),
                            "start_epoch": start_epoch,
                            "target_epochs": args.epochs,
                            "best_val_loss": best_val_loss,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        if mod.is_main_process(rank):
            mod.save_json(
                output_dir / "t5_run_config.json",
                {
                    "command": "t5_resume",
                    "dataset": args.dataset,
                    "world_size": world_size,
                    "output_dir": str(output_dir),
                    "item_semantic_ids": str(Path(args.item_semantic_ids).expanduser().resolve()),
                    "num_train_examples": len(train_examples),
                    "num_eval_examples": len(val_examples),
                    "max_history_items": args.max_history_items,
                    "position_sizes": position_sizes,
                    "hf_model_path": hf_model_path,
                    "num_user_tokens": args.num_user_tokens,
                    "seed": args.seed,
                    "epochs": args.epochs,
                    "resume_ckpt": str(resume_state_path),
                    "auto_resume": auto_resume,
                    "started_from_epoch": start_epoch,
                },
            )

        start_time = time.time() - elapsed_offset
        if start_epoch >= args.epochs:
            if mod.is_main_process(rank):
                print(
                    json.dumps(
                        {
                            "stage": "skip_train",
                            "reason": "already_reached_target_epochs",
                            "start_epoch": start_epoch,
                            "target_epochs": args.epochs,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            return

        for epoch in range(start_epoch, args.epochs):
            if distributed and isinstance(train_sampler, mod.DistributedSampler):
                train_sampler.set_epoch(epoch)

            model.train()
            epoch_loss = 0.0
            step_count = 0
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(train_loader, start=1):
                batch = mod.move_batch_to_device(batch, device)
                with mod.autocast_context(args.bf16):
                    output = model(**batch)
                    loss = output.loss / args.gradient_accumulation_steps

                loss.backward()
                epoch_loss += float(loss.detach().cpu().item()) * args.gradient_accumulation_steps
                step_count += 1

                if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                    if args.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            train_loss = epoch_loss / max(1, step_count)
            train_loss = mod.maybe_reduce_scalar(train_loss, distributed, device)
            val_loss = mod.evaluate_t5_model(
                model=model,
                dataloader=val_loader,
                distributed=distributed,
                device=device,
                bf16=args.bf16,
            )

            if mod.is_main_process(rank):
                elapsed_seconds = time.time() - start_time
                elapsed = mod.format_seconds(elapsed_seconds)
                learning_rate = optimizer.param_groups[0]["lr"]
                print(
                    f"[t5] epoch={epoch + 1}/{args.epochs} "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                    f"lr={learning_rate:.6g} elapsed={elapsed}",
                    flush=True,
                )

                model_to_save = model.module if isinstance(model, DistributedDataParallel) else model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_metrics = {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "learning_rate": learning_rate,
                        "elapsed_seconds": round(elapsed_seconds, 2),
                    }
                    best_dir = mod.ensure_dir(output_dir / "best")
                    model_to_save.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)
                    mod.save_json(
                        best_dir / "metadata.json",
                        {
                            "dataset": args.dataset,
                            "position_sizes": position_sizes,
                            "num_user_tokens": args.num_user_tokens,
                            "max_history_items": args.max_history_items,
                            "best_metrics": best_metrics,
                        },
                    )

                if args.save_last:
                    last_dir = mod.ensure_dir(output_dir / "last")
                    model_to_save.save_pretrained(last_dir)
                    tokenizer.save_pretrained(last_dir)
                    mod.save_json(
                        last_dir / "metadata.json",
                        {
                            "dataset": args.dataset,
                            "position_sizes": position_sizes,
                            "num_user_tokens": args.num_user_tokens,
                            "max_history_items": args.max_history_items,
                            "epoch": epoch + 1,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                        },
                    )

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model_to_save.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_val_loss": best_val_loss,
                        "best_metrics": best_metrics,
                        "elapsed_seconds": round(elapsed_seconds, 2),
                        "args": vars(args),
                    },
                    resume_state_path,
                )

            mod.barrier_if_needed(distributed)

        if mod.is_main_process(rank) and best_metrics is not None:
            mod.save_json(output_dir / "best_metrics.json", best_metrics)
    finally:
        mod.cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
