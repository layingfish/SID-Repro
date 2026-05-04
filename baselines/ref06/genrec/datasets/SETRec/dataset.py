

import json
import os

import numpy as np

from genrec.dataset import AbstractDataset


class SETRec(AbstractDataset):
    def __init__(self, config: dict):
        super(SETRec, self).__init__(config)

        self.category = config.get("category", "beauty")
        self._check_available_category()
        self.log(f"[DATASET] SETRec for domain: {self.category}")

        self.setrec_root = config.get(
            "setrec_data_root",
            config.get("setrec_root", os.environ.get("SETREC_DATA_ROOT", "data/setrec_data")),
        )
        self.cache_dir = os.path.join(config["cache_dir"], "SETRec", self.category)
        self._download_and_process_raw()

    def _check_available_category(self):
        available_categories = ["beauty", "toys", "sports", "steam", "amazon23_vg", "microlens_50k", "yelp"]
        assert self.category in available_categories, (
            f"Category \"{self.category}\" not available. "
            f"Available categories: {available_categories}"
        )

    def _load_processed(self, processed_dir: str) -> bool:
        seq_file = os.path.join(processed_dir, "all_item_seqs.json")
        id_mapping_file = os.path.join(processed_dir, "id_mapping.json")
        meta_file = os.path.join(processed_dir, "metadata.sentence.json")

        if not (os.path.exists(seq_file) and os.path.exists(id_mapping_file) and os.path.exists(meta_file)):
            return False

        with open(seq_file, "r", encoding="utf-8") as f:
            self.all_item_seqs = json.load(f)
        with open(id_mapping_file, "r", encoding="utf-8") as f:
            self.id_mapping = json.load(f)
        with open(meta_file, "r", encoding="utf-8") as f:
            self.item2meta = json.load(f)
        return True

    def _ensure_sent_emb(self, processed_dir: str) -> None:
        sent_emb_model = self.config.get("sent_emb_model", "setrec_t5_tdcb")
        model_basename = os.path.basename(sent_emb_model)
        sent_emb_dim = int(self.config.get("sent_emb_dim", 768))
        raw_path = os.path.join(processed_dir, f"{model_basename}_raw_d{sent_emb_dim}.sent_emb")

        if os.path.exists(raw_path):
            return

        domain_dir = os.path.join(self.setrec_root, self.category)
        emb_npy = os.path.join(domain_dir, f"{self.category}.emb-t5-tdcb.npy")
        emb = np.load(emb_npy).astype(np.float32, copy=False)
        if emb.ndim != 2:
            raise ValueError(f"Unexpected embedding shape: {emb.shape}")
        if emb.shape[1] != sent_emb_dim:
            raise ValueError(
                f"sent_emb_dim mismatch: file {emb.shape[1]} vs config {sent_emb_dim}"
            )
        emb.tofile(raw_path)
        self.log(f"[DATASET] Wrote precomputed sentence embeddings to {raw_path}")

    def _download_and_process_raw(self):
        processed_dir = os.path.join(self.cache_dir, "processed")
        os.makedirs(processed_dir, exist_ok=True)

        if self._load_processed(processed_dir):
            self._ensure_sent_emb(processed_dir)
            return

        domain_dir = os.path.join(self.setrec_root, self.category)
        train_dict = np.load(
            os.path.join(domain_dir, "training_dict.npy"), allow_pickle=True
        ).item()
        val_dict = np.load(
            os.path.join(domain_dir, "validation_dict.npy"), allow_pickle=True
        ).item()
        test_dict = np.load(
            os.path.join(domain_dir, "testing_dict.npy"), allow_pickle=True
        ).item()

        emb_npy = os.path.join(domain_dir, f"{self.category}.emb-t5-tdcb.npy")
        emb = np.load(emb_npy, mmap_mode="r")
        n_items = int(emb.shape[0])

        self.id_mapping = {
            "user2id": {"[PAD]": 0},
            "item2id": {"[PAD]": 0},
            "id2user": ["[PAD]"],
            "id2item": ["[PAD]"],
        }

        for i in range(n_items):
            tok = f"I{i}"
            self.id_mapping["item2id"][tok] = i + 1
            self.id_mapping["id2item"].append(tok)

        def _to_list(x):
            if x is None:
                return []
            if isinstance(x, np.ndarray):
                x = x.tolist()
            if isinstance(x, list):
                return x
            return [x]


        self.all_item_seqs = {}
        user_tokens = []
        for u, test_seq in test_dict.items():
            test_seq = _to_list(test_seq)
            if len(test_seq) == 0:
                continue

            train_seq = _to_list(train_dict.get(u, []))
            if len(train_seq) == 0:
                continue

            val_seq = _to_list(val_dict.get(u, []))

            test_target = int(test_seq[0])

            if len(val_seq) > 0:
                val_target = int(val_seq[0])
                train_used = list(map(int, train_seq))
            else:


                train_list = list(map(int, train_seq))
                val_target = int(train_list[-1])
                train_used = train_list[:-1]


            if val_target == test_target and len(train_used) > 0:
                val_target = int(train_used[-1])
                train_used = train_used[:-1]

            full = list(train_used) + [val_target, test_target]
            user_tok = f"U{int(u)}"
            self.all_item_seqs[user_tok] = [f"I{i}" for i in full]
            user_tokens.append(user_tok)

        user_tokens = sorted(set(user_tokens))
        for new_uid, user_tok in enumerate(user_tokens, start=1):
            self.id_mapping["user2id"][user_tok] = new_uid
            self.id_mapping["id2user"].append(user_tok)

        combined = np.load(
            os.path.join(domain_dir, "combine_tdcb_maps.npy"), allow_pickle=True
        ).item()
        recid2combine = combined.get("recid2combine", {})
        self.item2meta = {f"I{i}": str(recid2combine.get(i, "")) for i in range(n_items)}

        with open(os.path.join(processed_dir, "all_item_seqs.json"), "w", encoding="utf-8") as f:
            json.dump(self.all_item_seqs, f)
        with open(os.path.join(processed_dir, "id_mapping.json"), "w", encoding="utf-8") as f:
            json.dump(self.id_mapping, f)
        with open(os.path.join(processed_dir, "metadata.sentence.json"), "w", encoding="utf-8") as f:
            json.dump(self.item2meta, f, ensure_ascii=False)

        self._ensure_sent_emb(processed_dir)
