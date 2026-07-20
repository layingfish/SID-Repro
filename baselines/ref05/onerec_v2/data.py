from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


PAD = 0
BOS = 1
SEP = 2
EOS = 3

PAPER_STAGE1_CODEBOOK_WIDTH = 8192
PAPER_STAGE1_NUM_LEVELS = 3

DEFAULT_CODEBOOK_WIDTH = int(os.environ.get("ONEREC_K", str(PAPER_STAGE1_CODEBOOK_WIDTH)))
DEFAULT_NUM_LEVELS = int(os.environ.get("ONEREC_L", str(PAPER_STAGE1_NUM_LEVELS)))

CODEBOOK_WIDTH = DEFAULT_CODEBOOK_WIDTH
NUM_LEVELS = DEFAULT_NUM_LEVELS
VOCAB_SIZE = 4 + NUM_LEVELS * CODEBOOK_WIDTH


def set_runtime_config(num_levels: int | None = None, codebook_width: int | None = None):
    """Update the runtime vocabulary layout before building data/model objects."""
    global NUM_LEVELS, CODEBOOK_WIDTH, VOCAB_SIZE

    if num_levels is not None:
        NUM_LEVELS = int(num_levels)
    if codebook_width is not None:
        CODEBOOK_WIDTH = int(codebook_width)

    VOCAB_SIZE = 4 + NUM_LEVELS * CODEBOOK_WIDTH


def get_runtime_config():
    return {
        "num_levels": NUM_LEVELS,
        "codebook_width": CODEBOOK_WIDTH,
        "vocab_size": VOCAB_SIZE,
    }


def code_to_token(level: int, code: int) -> int:
    return 4 + level * CODEBOOK_WIDTH + int(code)


def token_to_code(token_id: int):
    if token_id < 4:
        return None
    token_id -= 4
    return token_id // CODEBOOK_WIDTH, token_id % CODEBOOK_WIDTH


def load_npy_dict(path: str):
    return np.load(path, allow_pickle=True).item()


def load_split_dicts(data_dir: str):
    train_dict = load_npy_dict(os.path.join(data_dir, "training_dict.npy"))
    val_dict = load_npy_dict(os.path.join(data_dir, "validation_dict.npy"))
    test_dict = load_npy_dict(os.path.join(data_dir, "testing_dict.npy"))
    return train_dict, val_dict, test_dict


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid jsonl at {path}:{lineno}: {exc}") from exc
    return rows


def load_semantic_ids(path: str):
    """Load semantic IDs and build token/code mappings for the active K/L config."""
    sids = torch.load(path, map_location="cpu", weights_only=True)
    if int(sids.shape[0]) < NUM_LEVELS:
        raise ValueError(
            f"semantic IDs at {path} only have {int(sids.shape[0])} levels, "
            f"but runtime config expects {NUM_LEVELS}"
        )

    sid_l = sids[:NUM_LEVELS].T.long().numpy()
    if sid_l.max(initial=0) >= CODEBOOK_WIDTH:
        raise ValueError(
            f"semantic IDs at {path} contain code >= {CODEBOOK_WIDTH}; "
            "runtime codebook_width does not match this checkpoint/data pair"
        )

    n_items = sid_l.shape[0]
    item_to_codes = sid_l.astype(np.int64, copy=True)
    item_to_tokens = np.zeros((n_items, NUM_LEVELS), dtype=np.int64)
    tuple_to_items = defaultdict(list)

    for item_id in range(n_items):
        token_tuple = []
        for level in range(NUM_LEVELS):
            token_id = code_to_token(level, item_to_codes[item_id, level])
            item_to_tokens[item_id, level] = token_id
            token_tuple.append(int(token_id))
        tuple_to_items[tuple(token_tuple)].append(item_id)

    return item_to_tokens, item_to_codes, dict(tuple_to_items)


def build_trie(item_to_tokens: np.ndarray):
    """Build a token-level prefix trie for constrained decoder generation."""
    trie = defaultdict(set)
    for row in item_to_tokens:
        tokens = [int(x) for x in row.tolist()]
        for depth in range(len(tokens)):
            trie[tuple(tokens[:depth])].add(tokens[depth])
    return {k: sorted(v) for k, v in trie.items()}


def build_code_trie(item_to_codes: np.ndarray):
    """Build a prefix trie over raw semantic codes for hierarchical classifiers."""
    trie = defaultdict(set)
    for row in item_to_codes:
        codes = [int(x) for x in row.tolist()]
        for depth in range(len(codes)):
            trie[tuple(codes[:depth])].add(codes[depth])
    return {k: sorted(v) for k, v in trie.items()}


def build_code_tuple_to_items(item_to_codes: np.ndarray):
    mapping = defaultdict(list)
    for item_id, row in enumerate(item_to_codes):
        mapping[tuple(int(x) for x in row.tolist())].append(int(item_id))
    return dict(mapping)


def build_popularity(train_dict, n_items: int):
    counts = np.zeros(n_items, dtype=np.int64)
    for items in train_dict.values():
        for item_id in items:
            item_id = int(item_id)
            if 0 <= item_id < n_items:
                counts[item_id] += 1
    popular_items = np.argsort(-counts).tolist()
    return counts, popular_items


def encode_history(item_ids, item_to_tokens: np.ndarray, max_hist: int = 256):
    item_ids = item_ids[-max_hist:]
    tokens = []
    for item_id in item_ids:
        tokens.append(SEP)
        tokens.extend(item_to_tokens[item_id].tolist())
    return tokens


def encode_session_labels(item_ids, item_to_tokens: np.ndarray):
    """Encode a session target as flat decoder labels."""
    labels = []
    for idx, item_id in enumerate(item_ids):
        labels.extend(item_to_tokens[item_id].tolist())
        labels.append(EOS if idx == len(item_ids) - 1 else BOS)
    return labels


def build_length_distribution(target_dicts, n_items: int, min_len: int = 1, max_len: int | None = None):
    counts = defaultdict(int)
    for target_dict in target_dicts:
        for target_items in target_dict.values():
            target_len = sum(1 for x in target_items if 0 <= int(x) < n_items)
            if target_len <= 0:
                continue
            if max_len is not None:
                target_len = min(int(max_len), target_len)
            if target_len < min_len:
                continue
            counts[int(target_len)] += 1

    if not counts:
        return None, None

    lengths = np.asarray(sorted(counts), dtype=np.int64)
    probs = np.asarray([counts[int(length)] for length in lengths], dtype=np.float64)
    probs /= probs.sum()
    return lengths, probs


class OneRecTrainDataset(Dataset):
    def __init__(
        self,
        user_dict,
        item_to_tokens: np.ndarray,
        max_hist: int = 256,
        min_hist: int = 3,
        session_size: int = 5,
    ):
        self.item_to_tokens = item_to_tokens
        self.max_hist = max_hist
        self.session_size = session_size
        self.samples = []

        n_items = int(item_to_tokens.shape[0])
        for items in user_dict.values():
            items = [int(x) for x in items if 0 <= int(x) < n_items]
            for t in range(min_hist, len(items) - session_size + 1):
                self.samples.append((items[:t], items[t:t + session_size]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        history, target_items = self.samples[idx]
        return {
            "input_ids": torch.tensor(
                encode_history(history, self.item_to_tokens, self.max_hist),
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                encode_session_labels(target_items, self.item_to_tokens),
                dtype=torch.long,
            ),
        }


class OneRecNextItemTrainDataset(Dataset):
    """Train on history -> next-item semantic codes for hierarchical decoding."""

    def __init__(
        self,
        user_dict,
        item_to_tokens: np.ndarray,
        item_to_codes: np.ndarray,
        max_hist: int = 256,
        min_hist: int = 3,
    ):
        self.item_to_tokens = item_to_tokens
        self.item_to_codes = item_to_codes
        self.max_hist = max_hist
        self.samples = []

        n_items = int(item_to_tokens.shape[0])
        for items in user_dict.values():
            items = [int(x) for x in items if 0 <= int(x) < n_items]
            for t in range(min_hist, len(items)):
                self.samples.append((items[:t], items[t]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        history, target_item = self.samples[idx]
        return {
            "input_ids": torch.tensor(
                encode_history(history, self.item_to_tokens, self.max_hist),
                dtype=torch.long,
            ),
            "target_codes": torch.tensor(
                self.item_to_codes[int(target_item)],
                dtype=torch.long,
            ),
        }


class OneRecSessionTrainDataset(Dataset):
    """Training dataset backed by explicit (history, target-session) samples."""

    def __init__(
        self,
        samples_path: str,
        item_to_tokens: np.ndarray,
        max_hist: int = 256,
        min_hist: int = 3,
    ):
        self.item_to_tokens = item_to_tokens
        self.max_hist = max_hist
        self.samples = []
        self.min_target_items = None
        self.max_target_items = 0

        n_items = int(item_to_tokens.shape[0])
        for row in load_jsonl(samples_path):
            history = [int(x) for x in row.get("history_items", []) if 0 <= int(x) < n_items]
            target_items = [int(x) for x in row.get("target_items", []) if 0 <= int(x) < n_items]
            if len(history) < min_hist or not target_items:
                continue

            self.samples.append((history, target_items))
            target_len = len(target_items)
            self.min_target_items = target_len if self.min_target_items is None else min(self.min_target_items, target_len)
            self.max_target_items = max(self.max_target_items, target_len)

        if not self.samples:
            raise ValueError(f"no valid session samples found in {samples_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        history, target_items = self.samples[idx]
        return {
            "input_ids": torch.tensor(
                encode_history(history, self.item_to_tokens, self.max_hist),
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                encode_session_labels(target_items, self.item_to_tokens),
                dtype=torch.long,
            ),
        }


class OneRecFutureTrainDataset(Dataset):
    """Train on user history -> future suffix chunks that match benchmark target lengths."""

    def __init__(
        self,
        user_dict,
        item_to_tokens: np.ndarray,
        max_hist: int = 256,
        min_hist: int = 3,
        min_target_items: int = 1,
        max_target_items: int = 5,
        length_strategy: str = "eval_empirical",
        target_lengths: np.ndarray | None = None,
        target_probs: np.ndarray | None = None,
        seed: int = 42,
    ):
        self.item_to_tokens = item_to_tokens
        self.max_hist = max_hist
        self.samples = []
        self.min_target_items = None
        self.max_target_items = 0
        self.length_strategy = str(length_strategy)

        if max_target_items < min_target_items:
            raise ValueError("max_target_items must be >= min_target_items")

        rng = np.random.default_rng(seed)
        n_items = int(item_to_tokens.shape[0])

        if target_lengths is not None:
            target_lengths = np.asarray(target_lengths, dtype=np.int64)
        if target_probs is not None:
            target_probs = np.asarray(target_probs, dtype=np.float64)

        for items in user_dict.values():
            items = [int(x) for x in items if 0 <= int(x) < n_items]
            for t in range(min_hist, len(items)):
                remaining = len(items) - t
                if remaining < min_target_items:
                    continue

                allowed_max = min(int(max_target_items), int(remaining))
                target_len = self._sample_target_len(
                    rng=rng,
                    min_target_items=min_target_items,
                    max_target_items=allowed_max,
                    target_lengths=target_lengths,
                    target_probs=target_probs,
                )
                if target_len is None:
                    continue

                target_items = items[t:t + target_len]
                self.samples.append((items[:t], target_items))
                self.min_target_items = target_len if self.min_target_items is None else min(self.min_target_items, target_len)
                self.max_target_items = max(self.max_target_items, target_len)

        if not self.samples:
            raise ValueError("no valid future-suffix training samples were constructed")

    def _sample_target_len(
        self,
        rng,
        min_target_items: int,
        max_target_items: int,
        target_lengths: np.ndarray | None,
        target_probs: np.ndarray | None,
    ):
        if self.length_strategy == "max":
            return int(max_target_items)

        if self.length_strategy == "uniform":
            return int(rng.integers(min_target_items, max_target_items + 1))

        if target_lengths is None or target_probs is None:
            raise ValueError("eval_empirical length sampling requires target_lengths and target_probs")

        mask = (target_lengths >= int(min_target_items)) & (target_lengths <= int(max_target_items))
        if not np.any(mask):
            return int(max_target_items)

        lengths = target_lengths[mask]
        probs = target_probs[mask]
        probs = probs / probs.sum()
        return int(rng.choice(lengths, p=probs))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        history, target_items = self.samples[idx]
        return {
            "input_ids": torch.tensor(
                encode_history(history, self.item_to_tokens, self.max_hist),
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                encode_session_labels(target_items, self.item_to_tokens),
                dtype=torch.long,
            ),
        }


class OneRecEvalDataset(Dataset):
    def __init__(
        self,
        train_dict,
        val_dict,
        test_dict,
        item_to_tokens: np.ndarray,
        split: str = "val",
        max_hist: int = 256,
        sample_users: int | None = None,
        seed: int = 42,
    ):
        self.item_to_tokens = item_to_tokens
        self.max_hist = max_hist
        self.samples = []

        n_items = int(item_to_tokens.shape[0])
        target_dict = val_dict if split == "val" else test_dict

        for user_id, target_items in target_dict.items():
            targets = [int(x) for x in target_items if 0 <= int(x) < n_items]
            if not targets:
                continue

            history = [int(x) for x in train_dict.get(user_id, []) if 0 <= int(x) < n_items]
            if split == "test":
                history += [int(x) for x in val_dict.get(user_id, []) if 0 <= int(x) < n_items]
            if not history:
                continue

            self.samples.append(
                {
                    "user_id": int(user_id),
                    "history_items": history,
                    "target_items": targets,
                }
            )

        if sample_users is not None and len(self.samples) > sample_users:
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(self.samples), size=sample_users, replace=False)
            self.samples = [self.samples[int(i)] for i in sorted(indices.tolist())]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        return {
            "user_id": sample["user_id"],
            "history_items": sample["history_items"],
            "target_items": sample["target_items"],
            "input_ids": torch.tensor(
                encode_history(sample["history_items"], self.item_to_tokens, self.max_hist),
                dtype=torch.long,
            ),
        }


def collate_train(batch):
    input_ids = [x["input_ids"] for x in batch]
    labels = [x["labels"] for x in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=PAD)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = (input_ids != PAD).long()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def collate_next_item_train(batch):
    input_ids = [x["input_ids"] for x in batch]
    target_codes = torch.stack([x["target_codes"] for x in batch], dim=0)

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=PAD)
    attention_mask = (input_ids != PAD).long()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_codes": target_codes,
    }


def collate_eval(batch):
    input_ids = [x["input_ids"] for x in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=PAD)
    attention_mask = (input_ids != PAD).long()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "user_ids": [x["user_id"] for x in batch],
        "history_items": [x["history_items"] for x in batch],
        "target_items": [x["target_items"] for x in batch],
    }


def build_valid_token_mask(target_seq_len: int):
    """Build a [T, V] mask describing valid decoder tokens for variable-length sessions."""
    total_steps = int(target_seq_len)
    mask = torch.zeros(total_steps, VOCAB_SIZE, dtype=torch.bool)

    for pos in range(total_steps):
        pos_in_item = pos % (NUM_LEVELS + 1)
        if pos_in_item < NUM_LEVELS:
            level = pos_in_item
            start = 4 + level * CODEBOOK_WIDTH
            end = start + CODEBOOK_WIDTH
            mask[pos, start:end] = True
        else:
            mask[pos, BOS] = True
            mask[pos, EOS] = True

    return mask
