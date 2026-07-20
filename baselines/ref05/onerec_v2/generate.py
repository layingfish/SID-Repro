from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import data as data_utils
from .data import (
    OneRecEvalDataset,
    build_code_trie,
    build_code_tuple_to_items,
    build_popularity,
    build_trie,
    collate_eval,
    load_semantic_ids,
    load_split_dicts,
)
from .model import load_checkpoint_model


def _level_token_range(level: int):
    start = 4 + level * data_utils.CODEBOOK_WIDTH
    return list(range(start, start + data_utils.CODEBOOK_WIDTH))


def make_prefix_allowed_tokens_fn(trie: dict, min_session_size: int, max_session_size: int):
    total_steps = max_session_size * (data_utils.NUM_LEVELS + 1)

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        del batch_id

        generated = input_ids.tolist()[1:]
        pos = len(generated)
        if pos >= total_steps:
            return [data_utils.EOS]

        pos_in_item = pos % (data_utils.NUM_LEVELS + 1)
        if pos_in_item == data_utils.NUM_LEVELS:
            completed_items = pos // (data_utils.NUM_LEVELS + 1) + 1
            if completed_items < min_session_size:
                return [data_utils.BOS]
            if completed_items >= max_session_size:
                return [data_utils.EOS]
            return [data_utils.BOS, data_utils.EOS]

        item_start = (pos // (data_utils.NUM_LEVELS + 1)) * (data_utils.NUM_LEVELS + 1)
        prefix = tuple(generated[item_start:item_start + pos_in_item])
        valid = trie.get(prefix)
        if valid:
            return valid
        return _level_token_range(pos_in_item)

    return prefix_allowed_tokens_fn


def decode_session_tokens(token_ids, max_session_size: int | None = None):
    if token_ids and token_ids[0] == data_utils.BOS:
        token_ids = token_ids[1:]

    items = []
    current = []
    terminated = False
    for token_id in token_ids:
        token_id = int(token_id)

        if token_id == data_utils.EOS:
            if len(current) == data_utils.NUM_LEVELS:
                items.append(tuple(current))
            current = []
            terminated = True
            break

        if token_id == data_utils.BOS:
            if len(current) == data_utils.NUM_LEVELS:
                items.append(tuple(current))
            current = []
            if max_session_size is not None and len(items) >= max_session_size:
                break
            continue

        if token_id in (data_utils.PAD, data_utils.SEP):
            continue

        current.append(token_id)
        if len(current) > data_utils.NUM_LEVELS:
            current = []

    if (
        not terminated
        and len(current) == data_utils.NUM_LEVELS
        and (max_session_size is None or len(items) < max_session_size)
    ):
        items.append(tuple(current))

    if max_session_size is None:
        return items
    return items[:max_session_size]


def _rank_item_candidates(item_candidates, popularity_counts):
    if popularity_counts is None or len(item_candidates) <= 1:
        return item_candidates
    return sorted(item_candidates, key=lambda item_id: -int(popularity_counts[int(item_id)]))


def _build_candidate_groups(
    ordered_tuples,
    tuple_to_items: dict,
    popularity_counts: np.ndarray | None,
    seen_items: set[int],
    seen_pred: set[int],
):
    groups = []
    seen_tuples = set()

    for raw_tuple in ordered_tuples:
        key = tuple(int(x) for x in raw_tuple)
        if key in seen_tuples:
            continue
        seen_tuples.add(key)

        item_candidates = list(tuple_to_items.get(key, []))
        item_candidates = _rank_item_candidates(item_candidates, popularity_counts)

        filtered = []
        for item_id in item_candidates:
            item_id = int(item_id)
            if item_id in seen_items or item_id in seen_pred:
                continue
            filtered.append(item_id)

        if filtered:
            groups.append(filtered)

    return groups


def _extend_predictions_two_stage(
    predicted: list[int],
    seen_pred: set[int],
    candidate_groups,
    topk: int,
):
    for group in candidate_groups:
        item_id = int(group[0])
        if item_id in seen_pred:
            continue
        predicted.append(item_id)
        seen_pred.add(item_id)
        if len(predicted) >= topk:
            return

    for group in candidate_groups:
        for item_id in group[1:]:
            item_id = int(item_id)
            if item_id in seen_pred:
                continue
            predicted.append(item_id)
            seen_pred.add(item_id)
            if len(predicted) >= topk:
                return


def _generate_predictions_seq2seq(
    model,
    eval_loader,
    tuple_to_items: dict,
    trie: dict,
    popularity_counts: np.ndarray | None,
    popular_items: list[int] | None,
    device: str,
    beam_size: int = 128,
    topk: int = 20,
    min_session_size: int = 5,
    max_session_size: int = 5,
    generate_batch_size: int | None = 4,
):
    model.eval()
    results = []
    prefix_fn = make_prefix_allowed_tokens_fn(
        trie,
        min_session_size=min_session_size,
        max_session_size=max_session_size,
    )
    total_steps = max_session_size * (data_utils.NUM_LEVELS + 1)
    min_steps = min_session_size * (data_utils.NUM_LEVELS + 1)
    max_new_tokens = total_steps + 1

    for batch in tqdm(eval_loader, desc="Generating"):
        full_batch_size = int(batch["input_ids"].shape[0])
        chunk_size = full_batch_size if not generate_batch_size or generate_batch_size <= 0 else int(generate_batch_size)

        for start in range(0, full_batch_size, chunk_size):
            end = min(full_batch_size, start + chunk_size)
            input_ids = batch["input_ids"][start:end].to(device)
            attention_mask = batch["attention_mask"][start:end].to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_steps,
                num_beams=beam_size,
                num_return_sequences=beam_size,
                do_sample=False,
                early_stopping=False,
                forced_eos_token_id=None,
                prefix_allowed_tokens_fn=prefix_fn,
            )

            batch_size = input_ids.shape[0]
            for batch_idx in range(batch_size):
                global_idx = start + batch_idx
                user_id = int(batch["user_ids"][global_idx])
                seen_items = set(int(x) for x in batch["history_items"][global_idx])
                predicted = []
                seen_pred = set()
                ordered_token_tuples = []

                beam_outputs = outputs[batch_idx * beam_size:(batch_idx + 1) * beam_size]
                for beam in beam_outputs:
                    ordered_token_tuples.extend(
                        decode_session_tokens(
                            beam.tolist(),
                            max_session_size=max_session_size,
                        )
                    )

                candidate_groups = _build_candidate_groups(
                    ordered_tuples=ordered_token_tuples,
                    tuple_to_items=tuple_to_items,
                    popularity_counts=popularity_counts,
                    seen_items=seen_items,
                    seen_pred=seen_pred,
                )
                _extend_predictions_two_stage(
                    predicted=predicted,
                    seen_pred=seen_pred,
                    candidate_groups=candidate_groups,
                    topk=topk,
                )

                if popular_items:
                    for item_id in popular_items:
                        item_id = int(item_id)
                        if item_id in seen_items or item_id in seen_pred:
                            continue
                        predicted.append(item_id)
                        seen_pred.add(item_id)
                        if len(predicted) >= topk:
                            break

                results.append(
                    {
                        "user_id": user_id,
                        "predicted_items": predicted[:topk],
                    }
                )

            del outputs, input_ids, attention_mask
            if isinstance(device, str) and device.startswith("cuda"):
                torch.cuda.empty_cache()
            elif isinstance(device, torch.device) and device.type == "cuda":
                torch.cuda.empty_cache()

    return results


def _beam_search_code_tuples(model, context: torch.Tensor, code_trie: dict, beam_size: int):
    beams = [([], 0.0)]
    device = context.device

    for level in range(data_utils.NUM_LEVELS):
        next_beams = []
        for prefix_codes, prefix_score in beams:
            prefix_tensor = None
            if prefix_codes:
                prefix_tensor = torch.tensor([prefix_codes], dtype=torch.long, device=device)

            logits = model.predict_level_logits(
                context=context,
                level=level,
                prefix_codes=prefix_tensor,
            )[0]
            log_probs = torch.log_softmax(logits, dim=-1)

            valid_codes = code_trie.get(tuple(prefix_codes))
            if valid_codes:
                valid_idx = torch.as_tensor(valid_codes, dtype=torch.long, device=device)
                valid_scores = log_probs.index_select(0, valid_idx)
                k = min(beam_size, valid_idx.numel())
                top_scores, top_pos = torch.topk(valid_scores, k=k)
                top_codes = valid_idx.index_select(0, top_pos)
            else:
                k = min(beam_size, log_probs.numel())
                top_scores, top_codes = torch.topk(log_probs, k=k)

            for score, code in zip(top_scores.tolist(), top_codes.tolist()):
                next_beams.append((prefix_codes + [int(code)], prefix_score + float(score)))

        next_beams.sort(key=lambda row: row[1], reverse=True)
        beams = next_beams[:beam_size]

    return [tuple(prefix_codes) for prefix_codes, _ in beams]


@torch.no_grad()
def _generate_predictions_hierarchical(
    model,
    eval_loader,
    code_tuple_to_items: dict,
    code_trie: dict,
    popularity_counts: np.ndarray | None,
    popular_items: list[int] | None,
    device,
    beam_size: int = 50,
    topk: int = 20,
):
    model.eval()
    results = []

    for batch in tqdm(eval_loader, desc="Generating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        contexts = model.encode_context(input_ids=input_ids, attention_mask=attention_mask)

        for batch_idx in range(contexts.size(0)):
            user_id = int(batch["user_ids"][batch_idx])
            seen_items = set(int(x) for x in batch["history_items"][batch_idx])
            predicted = []
            seen_pred = set()

            code_tuples = _beam_search_code_tuples(
                model=model,
                context=contexts[batch_idx:batch_idx + 1],
                code_trie=code_trie,
                beam_size=beam_size,
            )
            candidate_groups = _build_candidate_groups(
                ordered_tuples=code_tuples,
                tuple_to_items=code_tuple_to_items,
                popularity_counts=popularity_counts,
                seen_items=seen_items,
                seen_pred=seen_pred,
            )
            _extend_predictions_two_stage(
                predicted=predicted,
                seen_pred=seen_pred,
                candidate_groups=candidate_groups,
                topk=topk,
            )

            if popular_items:
                for item_id in popular_items:
                    item_id = int(item_id)
                    if item_id in seen_items or item_id in seen_pred:
                        continue
                    predicted.append(item_id)
                    seen_pred.add(item_id)
                    if len(predicted) >= topk:
                        break

            results.append(
                {
                    "user_id": user_id,
                    "predicted_items": predicted[:topk],
                }
            )

    return results


@torch.no_grad()
def generate_predictions(
    model,
    eval_loader,
    tuple_to_items: dict,
    trie: dict,
    popularity_counts: np.ndarray | None,
    popular_items: list[int] | None,
    device: str,
    beam_size: int = 128,
    topk: int = 20,
    min_session_size: int = 5,
    max_session_size: int = 5,
    generate_batch_size: int | None = 4,
    architecture: str = "seq2seq",
):
    if architecture == "hierarchical":
        return _generate_predictions_hierarchical(
            model=model,
            eval_loader=eval_loader,
            code_tuple_to_items=tuple_to_items,
            code_trie=trie,
            popularity_counts=popularity_counts,
            popular_items=popular_items,
            device=torch.device(device) if isinstance(device, str) else device,
            beam_size=beam_size,
            topk=topk,
        )

    return _generate_predictions_seq2seq(
        model=model,
        eval_loader=eval_loader,
        tuple_to_items=tuple_to_items,
        trie=trie,
        popularity_counts=popularity_counts,
        popular_items=popular_items,
        device=device,
        beam_size=beam_size,
        topk=topk,
        min_session_size=min_session_size,
        max_session_size=max_session_size,
        generate_batch_size=generate_batch_size,
    )


def export_pred_topk(results, output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--semantic_ids_path", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--max_hist", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--beam_size", type=int, default=128)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--session_size", type=int, default=None)
    parser.add_argument("--min_session_size", type=int, default=None)
    parser.add_argument("--max_session_size", type=int, default=None)
    parser.add_argument("--generate_batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint_model(args.ckpt_path, device="cpu")
    architecture = ckpt.get("args", {}).get("architecture", "seq2seq")
    model = model.to(device)

    session_size = args.session_size
    min_session_size = args.min_session_size
    max_session_size = args.max_session_size
    if session_size is not None:
        min_session_size = session_size if min_session_size is None else min_session_size
        max_session_size = session_size if max_session_size is None else max_session_size
    if min_session_size is None:
        min_session_size = int(ckpt.get("args", {}).get("min_session_size", ckpt.get("args", {}).get("session_size", 5)))
    if max_session_size is None:
        max_session_size = int(ckpt.get("args", {}).get("max_session_size", ckpt.get("args", {}).get("session_size", 5)))
    if max_session_size < min_session_size:
        raise ValueError("max_session_size must be >= min_session_size")

    train_dict, val_dict, test_dict = load_split_dicts(args.data_dir)
    item_to_tokens, item_to_codes, tuple_to_items = load_semantic_ids(args.semantic_ids_path)

    if architecture == "hierarchical":
        trie = build_code_trie(item_to_codes)
        tuple_to_items = build_code_tuple_to_items(item_to_codes)
    else:
        trie = build_trie(item_to_tokens)

    popularity_counts, popular_items = build_popularity(train_dict, item_to_tokens.shape[0])

    dataset = OneRecEvalDataset(
        train_dict=train_dict,
        val_dict=val_dict,
        test_dict=test_dict,
        item_to_tokens=item_to_tokens,
        split=args.split,
        max_hist=args.max_hist,
        sample_users=None,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_eval,
        pin_memory=True,
    )

    results = generate_predictions(
        model=model,
        eval_loader=loader,
        tuple_to_items=tuple_to_items,
        trie=trie,
        popularity_counts=popularity_counts,
        popular_items=popular_items,
        device=str(device),
        beam_size=args.beam_size,
        topk=args.topk,
        min_session_size=min_session_size,
        max_session_size=max_session_size,
        generate_batch_size=args.generate_batch_size,
        architecture=architecture,
    )
    export_pred_topk(results, args.output_path)


if __name__ == "__main__":
    main()
