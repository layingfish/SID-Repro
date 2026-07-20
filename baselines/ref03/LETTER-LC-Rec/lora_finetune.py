import argparse
import os
import sys
from pathlib import Path
from typing import List
# import wandb
import torch
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from modeling_letter import LETTER
try:
    from fastchat.train.llama2_flash_attn_monkey_patch import (
        replace_llama_attn_with_flash_attn,
    )
    replace_llama_attn_with_flash_attn()
except Exception:
    pass

import transformers


from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, AutoTokenizer, LlamaConfig, AddedToken

from utils import *
from collator import Collator

def train(args):

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    dataloader_num_workers = int(os.environ.get("RECSYS26_LCREC_DATALOADER_WORKERS", "4"))
    dataloader_persistent_workers = dataloader_num_workers > 0 and os.environ.get("RECSYS26_LCREC_DATALOADER_PERSISTENT_WORKERS", "1") != "0"
    use_fast_tokenizer = os.environ.get("RECSYS26_LCREC_USE_FAST_TOKENIZER", "1") != "0"
    gradient_checkpointing = os.environ.get("RECSYS26_LCREC_GRADIENT_CHECKPOINTING", "1") != "0"
    compile_model = os.environ.get("RECSYS26_LCREC_COMPILE", "1") != "0"
    max_steps = int(os.environ.get("RECSYS26_LCREC_MAX_STEPS", "-1"))
    if local_rank == 0:
        print(vars(args))
        print({
            "dataloader_num_workers": dataloader_num_workers,
            "dataloader_persistent_workers": dataloader_persistent_workers,
            "use_fast_tokenizer": use_fast_tokenizer,
            "gradient_checkpointing": gradient_checkpointing,
            "compile_model": compile_model,
            "max_steps": max_steps,
        })

    if ddp:
        device_map = {"": local_rank}

    config = LlamaConfig.from_pretrained(args.base_model)
    tokenizer_kwargs = dict(
        model_max_length=args.model_max_length,
        padding_side="right",
    )
    if use_fast_tokenizer:
        tokenizer_kwargs["legacy"] = True
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=use_fast_tokenizer,
        **tokenizer_kwargs,
    )
    tokenizer.pad_token_id = 0
    args.deepspeed = None

    train_data, valid_data = load_datasets(args)
    force_save_steps = os.environ.get("RECSYS26_LCREC_SAVE_STEPS")
    new_tokens = train_data.datasets[0].get_new_tokens()
    if tokenizer.is_fast:
        add_num = tokenizer.add_tokens([AddedToken(token, normalized=False) for token in new_tokens])
    else:
        add_num = tokenizer.add_tokens(new_tokens)
    config.vocab_size = len(tokenizer)
    if local_rank == 0:
        print("add {} new token.".format(add_num))
        print("data num:", len(train_data))
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)

    assert_letter_tokenizer_compatible(train_data.datasets[0].indices, tokenizer)
    if local_rank == 0:
        print("[tokenizer_diagnostics]", summarize_letter_tokenizer_behavior(train_data.datasets[0].indices, tokenizer))

    collator = Collator(args, tokenizer)
    model = LETTER.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
    model.set_hyper(args.temperature)
    model.resize_token_embeddings(len(tokenizer))
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(","),
        modules_to_save=args.lora_modules_to_save.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, config)

    trainer_resume_path = args.resume_from_checkpoint
    if args.resume_from_checkpoint:
        ckpt_dir = Path(args.resume_from_checkpoint)
        trainer_state_path = ckpt_dir / "trainer_state.json"
        optimizer_path = ckpt_dir / "optimizer.pt"
        scheduler_path = ckpt_dir / "scheduler.pt"
        adapter_bin = ckpt_dir / "adapter_model.bin"
        adapter_safe = ckpt_dir / "adapter_model.safetensors"

        has_trainer_state = (
            trainer_state_path.exists()
            and optimizer_path.exists()
            and scheduler_path.exists()
        )
        if has_trainer_state:
            if local_rank == 0:
                print(f"Resuming trainer state from {ckpt_dir}")
        else:
            trainer_resume_path = False
            checkpoint_name = adapter_bin if adapter_bin.exists() else adapter_safe
            if checkpoint_name.exists():
                if local_rank == 0:
                    print(f"Loading adapter weights from {checkpoint_name}")
                if checkpoint_name.suffix == ".safetensors":
                    from safetensors.torch import load_file as safe_load_file
                    adapters_weights = safe_load_file(str(checkpoint_name))
                else:
                    adapters_weights = torch.load(str(checkpoint_name), map_location="cpu")
                set_peft_model_state_dict(model, adapters_weights)
            else:
                if local_rank == 0:
                    print(f"Checkpoint {ckpt_dir} missing trainer state and adapter weights")

    for n, p in model.named_parameters():
        if "original_module" in n and any(module_name in n for module_name in config.modules_to_save):
            p.requires_grad = False

    if hasattr(model, "active_adapters") and callable(model.active_adapters):
        try:
            active_adapters = model.active_adapters()
        except Exception:
            active_adapters = list(getattr(model, "peft_config", {}).keys())
        if isinstance(active_adapters, str):
            active_adapters = [active_adapters]
        elif active_adapters is None:
            active_adapters = []
        else:
            active_adapters = list(active_adapters)
        if not active_adapters:
            active_adapters = ["default"]
        model.active_adapters = active_adapters

    if local_rank == 0:
        model.print_trainable_parameters()


    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    eval_dataset_arg = valid_data
    eval_strategy = args.save_and_eval_strategy
    save_strategy = args.save_and_eval_strategy
    eval_steps = args.save_and_eval_steps
    save_steps = args.save_and_eval_steps
    load_best_model_at_end = True
    save_total_limit = int(os.environ.get("RECSYS26_LCREC_SAVE_TOTAL_LIMIT", "1"))
    if force_save_steps:
        eval_dataset_arg = None
        eval_strategy = "no"
        save_strategy = "steps"
        save_steps = int(force_save_steps)
        load_best_model_at_end = False

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_dataset_arg,
        args=transformers.TrainingArguments(
            seed=args.seed,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs,
            max_steps=max_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            report_to=['wandb'],
            fp16=args.fp16,
            bf16=args.bf16,
            logging_steps=args.logging_step,
            optim=args.optim,
            gradient_checkpointing=gradient_checkpointing,
            evaluation_strategy=eval_strategy,
            save_strategy=save_strategy,
            eval_steps=eval_steps,
            save_steps=save_steps,
            output_dir=args.output_dir,
            save_total_limit=save_total_limit,
            load_best_model_at_end=load_best_model_at_end,
            deepspeed=args.deepspeed,
            ddp_find_unused_parameters=False if ddp else None,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_persistent_workers=dataloader_persistent_workers,
            # report_to=None,
            eval_delay=1 if args.save_and_eval_strategy=="epoch" else 2000,
        ),
        tokenizer=tokenizer,
        data_collator=collator,
    )
    model.config.use_cache = False

    if compile_model and torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(
        resume_from_checkpoint=trainer_resume_path,
    )

    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMRec')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)

    args = parser.parse_args()

    train(args)
