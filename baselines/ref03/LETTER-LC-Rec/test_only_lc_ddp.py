import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from peft import PeftModel
from torch.utils.data import DataLoader, Subset
from transformers import LlamaForCausalLM, AutoTokenizer
from tqdm import tqdm

from utils import (
    assert_letter_tokenizer_compatible,
    summarize_letter_tokenizer_behavior,
    build_response_token_mappings,
    extract_response_token_tuples,
    load_test_dataset,
    make_response_prefix_allowed_tokens_fn,
    set_seed,
)
from collator import TestCollator


def parse_args():
    parser = argparse.ArgumentParser(description="LC-Rec DDP test-only export")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="../data")
    parser.add_argument("--index_file", type=str, default=".index.json")
    parser.add_argument("--export_path", type=str, required=True)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_his_len", type=int, default=20)
    parser.add_argument("--his_sep", type=str, default=", ")
    parser.add_argument("--add_prefix", action="store_true", default=False)
    parser.add_argument("--test_prompt_ids", type=str, default="0")
    parser.add_argument("--sample_num", type=int, default=-1)
    parser.add_argument("--test_task", type=str, default="SeqRec")
    parser.add_argument("--filter_items", action="store_true", default=False)
    parser.add_argument("--metrics", type=str, default="hit@1,hit@5,hit@10,ndcg@5,ndcg@10")
    parser.add_argument("--lora", action="store_true", default=True)
    parser.add_argument("--user_ids_path", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    rank = int(os.environ.get("RANK") or local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)

    device = torch.device("cuda", local_rank)
    device_map = {"": local_rank}

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path, use_fast=False, padding_side="left")
    tokenizer.pad_token_id = 0

    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(
        model,
        args.ckpt_path,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    test_data = load_test_dataset(args)
    collator = TestCollator(args, tokenizer)

    inter_path = os.path.join(args.data_path, args.dataset, args.dataset + ".inter.json")
    with open(inter_path) as f:
        inter_obj = json.load(f)
    user_ids_ordered = list(inter_obj.keys())

    index_path = os.path.join(args.data_path, args.dataset, args.dataset + args.index_file)
    with open(index_path) as f:
        index_obj = json.load(f)
    assert_letter_tokenizer_compatible(index_obj, tokenizer)
    if rank == 0:
        print("[tokenizer_diagnostics]", summarize_letter_tokenizer_behavior(index_obj, tokenizer))
    token_maps = build_response_token_mappings(index_obj, tokenizer)

    if args.user_ids_path:
        import json as _json
        keep_users = set(int(x) for x in _json.load(open(args.user_ids_path)))
        base_indices = [i for i, uid in enumerate(user_ids_ordered) if int(uid) in keep_users]
    else:
        base_indices = list(range(len(test_data)))

    subset_indices = base_indices[rank::world_size]
    subset_user_ids = [user_ids_ordered[i] for i in subset_indices]
    subset = Subset(test_data, subset_indices)
    test_loader = DataLoader(
        subset,
        batch_size=args.test_batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    if rank == 0:
        print(f"[LC-Rec DDP export] dataset={args.dataset} users={len(user_ids_ordered)} world_size={world_size}")
        print(f"[LC-Rec DDP export] reverse_map={len(token_maps['token_tuple_to_item'])} / {len(index_obj)}")
        print(f"[LC-Rec DDP export] response max_new_tokens={token_maps['max_new_tokens']}")

    all_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"rank{rank} export", disable=(rank != 0)):
            inputs = batch[0].to(device)
            batch_size = inputs["input_ids"].shape[0]
            input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
            prefix_allowed_tokens = make_response_prefix_allowed_tokens_fn(
                token_maps["trie"], input_lengths, token_maps["eos_token_id"]
            )
            num_beams = args.num_beams
            while True:
                try:
                    output = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=token_maps["max_new_tokens"],
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                        num_beams=num_beams,
                        num_return_sequences=num_beams,
                        output_scores=True,
                        return_dict_in_generate=True,
                        early_stopping=True,
                    )
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    num_beams -= 1
                    if num_beams <= 0:
                        raise RuntimeError("Beam reduced to 0 due to OOM")
                    if rank == 0:
                        print(f"[OOM] reduce beams to {num_beams}")

            output_ids = output["sequences"]
            scores = output["sequences_scores"]
            pred_token_tuples = extract_response_token_tuples(
                output_ids, input_lengths, num_beams, token_maps["eos_token_id"]
            )

            for b in range(batch_size):
                batch_token_tuples = pred_token_tuples[b * num_beams:(b + 1) * num_beams]
                batch_scores = scores[b * num_beams:(b + 1) * num_beams].tolist()
                pairs = sorted(zip(batch_token_tuples, batch_scores), key=lambda x: x[1], reverse=True)
                preds = []
                seen = set()
                for token_tuple, _ in pairs:
                    item_id = token_maps["token_tuple_to_item"].get(token_tuple)
                    if item_id is not None and item_id not in seen:
                        seen.add(item_id)
                        preds.append(item_id)
                        if len(preds) >= 20:
                            break
                all_preds.append(preds)

    if len(all_preds) != len(subset_user_ids):
        raise RuntimeError(f"rank{rank}: preds={len(all_preds)} users={len(subset_user_ids)} mismatch")

    export_path = Path(args.export_path)
    part_path = export_path.with_suffix(export_path.suffix + f".rank{rank}")
    with part_path.open("w") as f:
        for uid, preds in zip(subset_user_ids, all_preds):
            f.write(json.dumps({"user_id": int(uid), "predicted_items": preds[:20]}) + "\n")

    dist.barrier()

    if rank == 0:
        merged = {}
        for r in range(world_size):
            pp = export_path.with_suffix(export_path.suffix + f".rank{r}")
            with pp.open() as f:
                for line in f:
                    obj = json.loads(line)
                    merged[int(obj["user_id"])] = obj["predicted_items"]
        with export_path.open("w") as f:
            for uid in user_ids_ordered:
                uid_int = int(uid)
                preds = merged.get(uid_int, [])
                f.write(json.dumps({"user_id": uid_int, "predicted_items": preds[:20]}) + "\n")
        print(f"[EXPORT] merged users={len(merged)} -> {export_path}")
        with export_path.open() as f:
            print(f.readline().strip())

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
