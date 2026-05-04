

import torch
import numpy as np


class DIFF_GRMEvaluator:


    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.metric2func = {
            'recall': self.recall_at_k,
            'ndcg': self.ndcg_at_k
        }

        self.pad_token = self.tokenizer.pad_token
        self.maxk = max(config['topk'])


        self.total_seqs = 0
        self.total_legals = 0
        self.total_unique = 0


        self.batch_legal_ratios = []
        self.batch_duplicate_ratios = []
        self.batch_dup10_ratios = []


        print(f'>> Using evaluator = {self.__class__.__name__} (fixed duplicate scoring bug)')
        print(f'>> Recall: uses any() to avoid duplicate scoring')
        print(f'>> NDCG: uses first-hit-only to avoid duplicate DCG accumulation')
        print(f'>> Fixed: index bounds checking to prevent out-of-bounds errors')
        print(f'>> Added: illegal sequence filtering for more accurate evaluation')


        self.eval_expand = bool(self.config.get('eval_expand_sid_to_items', False))

        self.eval_expand_dedup = str(self.config.get('eval_expand_dedup', 'first')).lower()


        self.cb2items = getattr(self.tokenizer, 'cb2items', None)
        if self.cb2items is None:
            try:
                self.cb2items = self.tokenizer._build_cb2items_map()
            except Exception:
                self.cb2items = {}


    def calculate_pos_index(self, preds, labels):


        preds = preds.detach().cpu()
        labels = labels.detach().cpu()

        B, maxk, n_digit = preds.shape


        pos_index = torch.zeros((B, maxk), dtype=torch.bool)

        for i in range(B):

            cur_label = labels[i].tolist()

            for j in range(maxk):

                cur_pred = preds[i, j].tolist()


                if cur_pred == cur_label:
                    pos_index[i, j] = True
                    break

        return pos_index

    def recall_at_k(self, pos_index, k):


        return pos_index[:, :k].any(dim=1).cpu().float()

    def ndcg_at_k(self, pos_index, k):


        batch_size, maxk = pos_index.shape
        device = pos_index.device


        ranks = torch.arange(1, maxk + 1, device=device).float()
        dcg_weights = 1.0 / torch.log2(ranks + 1)


        position_matrix = torch.arange(maxk, device=device).expand(batch_size, -1)


        masked_positions = torch.where(pos_index, position_matrix, torch.full_like(position_matrix, maxk))
        first_hit_positions = masked_positions.min(dim=1).values


        has_hit = first_hit_positions < maxk
        in_topk = first_hit_positions < k
        valid_mask = has_hit & in_topk


        safe_positions = torch.clamp(first_hit_positions, 0, maxk - 1)
        dcg_scores = torch.where(valid_mask, dcg_weights[safe_positions], torch.tensor(0.0, device=device))


        return dcg_scores.cpu().float()


    def _sid_row_to_cb(self, sid_row):

        return tuple(int(x) for x in sid_row.tolist())


    def _expand_sid_list_to_items(self, sid_list, K):
        seen = set()
        expanded = []
        if sid_list.ndim != 2:
            return expanded
        M, _ = sid_list.shape
        for i in range(M):
            cb = self._sid_row_to_cb(sid_list[i])

            if cb not in self.cb2items:
                continue

            if self.eval_expand_dedup == 'first':
                if cb in seen:
                    continue
                seen.add(cb)

            iid_list = self.tokenizer.cb_tuple_to_item_ids(cb) if hasattr(self.tokenizer, 'cb_tuple_to_item_ids') else []
            for iid in iid_list:
                expanded.append(iid)
                if len(expanded) >= K:
                    return expanded
        return expanded


    def _build_pos_index_from_items(self, item_ranked, target_item_id, Kmax):
        hit_pos = None
        for idx, iid in enumerate(item_ranked):
            if iid == target_item_id:
                hit_pos = idx
                break
        pos = torch.zeros(Kmax, dtype=torch.bool)
        if hit_pos is not None and hit_pos < Kmax:
            pos[hit_pos] = True
        return pos

    def _dup_ratio_per_user(self, preds, k=10):


        B, _, n_digit = preds.shape
        dup_ratios = []

        for b in range(B):

            seqs = preds[b, :k]
            k_seqs = [tuple(s.tolist()) for s in seqs]
            unique_cnt = len(set(k_seqs))
            dup_ratios.append(1 - unique_cnt / k)

        return torch.tensor(dup_ratios, dtype=torch.float32)

    def calculate_weighted_score(self, preds, labels):


        pos_index = self.calculate_pos_index(preds, labels)


        ndcg_10 = self.ndcg_at_k(pos_index, k=10)


        recall_10 = self.recall_at_k(pos_index, k=10)


        weighted_score = 0.8 * ndcg_10 + 0.2 * recall_10

        return weighted_score

    def calculate_metrics(self, preds, labels, suffix=""):

        results = {}
        pos_index = self.calculate_pos_index(preds, labels)


        B, maxk, n_digit = preds.shape
        self.total_seqs += preds.numel() // n_digit
        self.total_legals += sum(
            self.tokenizer.codebooks_to_item_id(seq.tolist()) is not None
            for seq in preds.view(-1, n_digit)
        )
        self.total_unique += len({
            tuple(seq.tolist()) for seq in preds.view(-1, n_digit)
        })


        total_seqs = preds.numel() // n_digit
        current_legal_ratio = sum(
            self.tokenizer.codebooks_to_item_id(seq.tolist()) is not None
            for seq in preds.view(-1, n_digit)
        ) / total_seqs

        current_duplicate_ratio = 1 - len({
            tuple(seq.tolist()) for seq in preds.view(-1, n_digit)
        }) / total_seqs


        self.batch_legal_ratios.append(current_legal_ratio)
        self.batch_duplicate_ratios.append(current_duplicate_ratio)


        results[f'legal_ratio{suffix}'] = torch.tensor([current_legal_ratio], dtype=torch.float32)
        results[f'duplicate_ratio{suffix}'] = torch.tensor([current_duplicate_ratio], dtype=torch.float32)


        dup10 = self._dup_ratio_per_user(preds, k=10)
        results[f'dup@10{suffix}'] = dup10


        self.batch_dup10_ratios.append(dup10.mean().item())

        for metric in self.config['metrics']:
            for k in self.config['topk']:
                results[f"{metric}@{k}{suffix}"] = self.metric2func[metric](pos_index, k)


        if suffix == "":
            weighted_score = self.calculate_weighted_score(preds, labels)
            results['weighted_score'] = weighted_score


        if self.eval_expand:
            B, maxk, n_digit = preds.shape

            target_item_ids = []
            for i in range(B):
                lab = labels[i].tolist()
                target_iid = self.tokenizer.codebooks_to_item_id(lab)
                target_item_ids.append(int(target_iid) if target_iid is not None else 0)


            Kmax = self.maxk
            pos_index_item = torch.zeros(B, Kmax, dtype=torch.bool)
            dup10_list = []

            for i in range(B):
                sid_top = preds[i]
                expanded_items = self._expand_sid_list_to_items(sid_top, K=Kmax)

                k10 = min(10, len(expanded_items))
                if k10 > 0:
                    uniq = len(set(expanded_items[:k10]))
                    dup10_list.append(1 - uniq / k10)
                else:
                    dup10_list.append(1.0)

                pos_index_item[i] = self._build_pos_index_from_items(
                    expanded_items, target_item_ids[i], Kmax
                )


            for metric in self.config['metrics']:
                for k in self.config['topk']:
                    results[f"{metric}@{k}{suffix}_xitem"] = self.metric2func[metric](pos_index_item, k)

            ndcg_10_x = self.ndcg_at_k(pos_index_item, k=10)
            recall_10_x = self.recall_at_k(pos_index_item, k=10)
            weighted_x = 0.8 * ndcg_10_x + 0.2 * recall_10_x
            results[f"weighted_score{suffix}_xitem"] = weighted_x
            results[f"dup@10{suffix}_xitem"] = torch.tensor(dup10_list, dtype=torch.float32)

        return results

    def print_final_stats(self):

        if self.total_seqs > 0:

            legal_ratio = self.total_legals / self.total_seqs

            duplicate_ratio = 1 - self.total_unique / self.total_seqs


            if self.batch_legal_ratios:
                avg_legal_ratio = sum(self.batch_legal_ratios) / len(self.batch_legal_ratios)
                avg_duplicate_ratio = sum(self.batch_duplicate_ratios) / len(self.batch_duplicate_ratios)

                print(f"[SID_STATS] 平均合法率: {avg_legal_ratio:.3f}, 平均重复率: {avg_duplicate_ratio:.3f}")


                if self.batch_dup10_ratios:
                    avg_dup10 = sum(self.batch_dup10_ratios) / len(self.batch_dup10_ratios)
                    print(f"[SID_STATS] 用户内部 Top-10 平均重复率: {avg_dup10:.3f}")
            else:
                print(f"[SID_STATS] 总体合法率: {legal_ratio:.3f}, 总体重复率: {duplicate_ratio:.3f}")
