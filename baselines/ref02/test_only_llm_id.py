"""
LLM_RecSys_ID test-only prediction export.
Loads a trained P5 checkpoint and exports per-user top-K predictions to pred_topk.jsonl.
Supports iid/sid/semid/cid/hid-style item representations used in RecSys26.
"""
import argparse
import json
import os
import re
from collections import OrderedDict

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, T5Config

from modeling_p5 import P5
from generation_trie import Trie
from data import load_data, Collator
from prompt import task_subgroup_1
from utils import (
    prefix_allowed_tokens_fn,
    set_seed,
    content_based_representation_non_hierarchical,
)
from item_rep_method import (
    build_category_map,
    create_CF_embedding,
    create_CF_embedding_optimal_width,
    create_hybrid_embedding,
    load_hybrid,
    load_meta,
)


def parse_args():
    parser = argparse.ArgumentParser(description="LLM_RecSys_ID test-only export")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, help="e.g. setrec_microlens_50k")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--export_path", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_history", type=int, default=20)
    parser.add_argument("--model_type", type=str, default="t5-small")
    parser.add_argument("--evaluation_template_id", type=int, default=0)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--topk", type=int, default=20)

    parser.add_argument("--cluster_size", type=int, default=500)
    parser.add_argument("--cluster_number", type=int, default=20)
    parser.add_argument("--number_of_items", type=int, default=0)
    parser.add_argument("--last_token_no_repetition", action="store_true")
    parser.add_argument("--optimal_width_in_CF", action="store_true")
    parser.add_argument("--category_no_repetition", action="store_true")
    parser.add_argument("--hybrid_order", type=str, default="category_first")

    parser.add_argument("--data_order", type=str, default="remapped_sequential")
    parser.add_argument("--remapped_data_order", type=str, default="original")
    parser.add_argument("--item_representation", type=str, default="CF")
    parser.add_argument("--whole_word_embedding", type=str, default="shijie")
    parser.add_argument("--remove_last_item", action="store_true", default=False)
    parser.add_argument("--remove_first_item", action="store_true", default=False)
    parser.add_argument("--base", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--overlap", type=int, default=0)
    return parser.parse_args()


def saved_tokenizer_dir_from_ckpt(ckpt_path):
    return os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "tokenizer")


def init_number_of_items(args):
    if args.number_of_items > 0:
        return
    datamaps_path = os.path.join(args.data_dir, args.task, "datamaps.json")
    if not os.path.exists(datamaps_path):
        raise ValueError(f"Cannot infer number_of_items: {datamaps_path} not found")
    with open(datamaps_path, "r") as f:
        datamaps = json.load(f)
    args.number_of_items = len(datamaps["id2item"]) + 1


def init_tokenizer(args):
    tokenizer_dir = saved_tokenizer_dir_from_ckpt(args.ckpt_path)
    if os.path.isdir(tokenizer_dir):
        print(f"[LLM_ID] Load tokenizer from saved dir: {tokenizer_dir}")
        return AutoTokenizer.from_pretrained(tokenizer_dir)

    print(
        f"[LLM_ID] WARNING: saved tokenizer not found at {tokenizer_dir}; "
        "falling back to tokenizer reconstruction. Old checkpoints may be incompatible."
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    number_of_items = args.number_of_items

    if args.item_representation == "no_tokenization":
        new_tokens = [f"<extra_id_{x}>" for x in range(number_of_items)]
        ordered_new_tokens = [
            token for token in dict.fromkeys(new_tokens) if token not in tokenizer.vocab
        ]
        if ordered_new_tokens:
            tokenizer.add_tokens(ordered_new_tokens)

    elif args.item_representation == "content_based":
        if args.task == "yelp":
            from item_rep_method import create_category_embedding
            tokenizer = create_category_embedding(args, tokenizer)
        else:
            meta_data, meta_dict, id2item = load_meta(args)
            category_dict, level_categories = build_category_map(
                args, meta_data, meta_dict, id2item
            )
            tokenizer = content_based_representation_non_hierarchical(
                args, category_dict, level_categories, tokenizer
            )

    elif args.item_representation == "CF":
        if args.optimal_width_in_CF:
            tokenizer = create_CF_embedding_optimal_width(args, tokenizer)
        else:
            tokenizer = create_CF_embedding(args, tokenizer)

    elif args.item_representation == "hybrid":
        _, vocabulary = load_hybrid(args)
        tokenizer = create_hybrid_embedding(vocabulary, tokenizer)

    return tokenizer


def build_reverse_map(all_items, remapped_all_items):
    reverse_map = {}
    ordered_candidates = []
    for orig_id, rep in zip(all_items, remapped_all_items):
        rep = rep.replace(" ", "")
        if rep not in reverse_map:
            reverse_map[rep] = int(orig_id) - 1
            ordered_candidates.append(rep)
    return reverse_map, ordered_candidates


def normalize_decoded(s):
    s = s.strip().replace(" ", "")
    if s.startswith("item_"):
        s = s[5:]
    return s


def main():
    args = parse_args()
    init_number_of_items(args)
    set_seed(args)
    device = torch.device(f"cuda:{args.gpu_id}")

    print(f"[LLM_ID:{args.task}] ckpt={args.ckpt_path}")
    print(f"[LLM_ID:{args.task}] export={args.export_path}")
    print(f"[LLM_ID:{args.task}] item_representation={args.item_representation}")
    print(f"[LLM_ID:{args.task}] data_order={args.data_order} remapped_data_order={args.remapped_data_order}")
    print(f"[LLM_ID:{args.task}] number_of_items={args.number_of_items}")

    tokenizer = init_tokenizer(args)
    print(f"[LLM_ID] Tokenizer vocab size: {len(tokenizer)}")

    users, all_items, train_sequence, val_sequence, test_sequence, remapped_all_items = load_data(args, tokenizer)
    reverse_map, ordered_candidates = build_reverse_map(all_items, remapped_all_items)

    print(f"[LLM_ID] {len(users)} users, {len(reverse_map)} unique candidate items")

    candidate_trie = Trie(
        [[0] + tokenizer.encode(f"item_{candidate}") for candidate in ordered_candidates]
    )
    prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)
    print(f"[LLM_ID] Trie built with {len(ordered_candidates)} candidates")

    config = T5Config.from_pretrained(args.model_type)
    model = P5.from_pretrained(args.model_type, config=config)
    model.resize_token_embeddings(len(tokenizer))

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        new_ckpt[k.replace("module.", "")] = v
    model.load_state_dict(new_ckpt, strict=True)
    del ckpt, new_ckpt

    model = model.to(device)
    model.eval()
    print(f"[LLM_ID] Model loaded from {args.ckpt_path}")

    template = task_subgroup_1[args.evaluation_template_id]
    collator = Collator(args, tokenizer)

    test_examples = []
    test_user_ids = []
    for user, seq in zip(users, test_sequence):
        purchase_history = seq[:-1]
        if len(purchase_history) > args.max_history:
            purchase_history = purchase_history[-args.max_history:]

        if template["input_first"] == "user":
            input_sent = template["source"].format(
                user,
                " , ".join(["item_" + item for item in purchase_history]),
            )
        else:
            input_sent = template["source"].format(
                " , ".join(["item_" + item for item in purchase_history]),
                user,
            )
        target_sent = template["target"].format("item_" + seq[-1])
        test_examples.append((input_sent, target_sent))
        test_user_ids.append(int(user) - 1)

    print(f"[LLM_ID] {len(test_examples)} test examples")

    class ListDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    test_loader = DataLoader(
        ListDataset(test_examples),
        batch_size=args.batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=0,
    )

    all_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Export preds"):
            input_ids = batch[0].to(device)
            attn = batch[1].to(device)
            batch_size = attn.shape[0]

            whole_input_ids = batch[2].to(device)
            generate_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attn,
                max_length=10,
                prefix_allowed_tokens_fn=prefix_allowed_tokens,
                num_beams=args.num_beams,
                num_return_sequences=args.num_beams,
                whole_word_embedding_type=args.whole_word_embedding,
                output_scores=True,
                return_dict_in_generate=True,
            )
            if args.whole_word_embedding != "None":
                generate_kwargs["whole_word_ids"] = whole_input_ids
            prediction = model.generate(**generate_kwargs)

            prediction_ids = prediction["sequences"]
            prediction_scores = prediction["sequences_scores"]
            decoded = tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)

            for b in range(batch_size):
                batch_seqs = decoded[b * args.num_beams : (b + 1) * args.num_beams]
                batch_scores = prediction_scores[b * args.num_beams : (b + 1) * args.num_beams].tolist()
                pairs = sorted(zip(batch_seqs, batch_scores), key=lambda x: x[1], reverse=True)

                items_0based = []
                seen = set()
                for seq_str, _ in pairs:
                    rep = normalize_decoded(seq_str)
                    item_id = reverse_map.get(rep)
                    if item_id is None:
                        match = re.match(r"^<A(\d+)>$", rep)
                        if match:
                            item_id = int(match.group(1))
                    if item_id is not None and item_id not in seen:
                        seen.add(item_id)
                        items_0based.append(item_id)
                        if len(items_0based) >= args.topk:
                            break

                if len(items_0based) < args.topk:
                    for candidate in ordered_candidates:
                        fallback_item = reverse_map[candidate]
                        if fallback_item not in seen:
                            seen.add(fallback_item)
                            items_0based.append(fallback_item)
                            if len(items_0based) >= args.topk:
                                break

                all_preds.append(items_0based)

    os.makedirs(os.path.dirname(args.export_path), exist_ok=True)
    with open(args.export_path, "w") as f:
        for uid, preds in zip(test_user_ids, all_preds):
            f.write(json.dumps({"user_id": uid, "predicted_items": preds[: args.topk]}) + "\n")

    n_lines = len(test_user_ids)
    avg_preds = sum(len(p) for p in all_preds) / max(len(all_preds), 1)
    print(f"[EXPORT] Saved {n_lines} predictions to {args.export_path}")
    print(f"[EXPORT] Avg predictions per user: {avg_preds:.2f}")
    with open(args.export_path, "r") as f:
        first = f.readline().strip()
    print(first)


if __name__ == "__main__":
    main()
