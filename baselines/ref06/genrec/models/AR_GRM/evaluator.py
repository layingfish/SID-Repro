

import torch
import numpy as np


class AR_GRMEvaluator:


    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.metric2func = {
            'recall': self.recall_at_k,
            'ndcg': self.ndcg_at_k
        }

        self.pad_token = self.tokenizer.pad_token
        self.maxk = max(config['topk'])


        self.batch_legal_at10 = []


        print(f'>> Using evaluator = {self.__class__.__name__} (AR beam)')
        print(f'>> Recall: any() 按首次命中计分')
        print(f'>> NDCG: first-hit-only 计算')


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
        K = min(10, maxk)
        legal_mask = []
        for b in range(B):
            cnt = 0
            for j in range(K):
                if self.tokenizer.codebooks_to_item_id(preds[b, j].tolist()) is not None:
                    cnt += 1
            legal_mask.append(cnt / float(K))
        legal_at10 = torch.tensor(legal_mask, dtype=torch.float32)
        results['legal@10'] = legal_at10
        self.batch_legal_at10.append(legal_at10.mean().item())

        for metric in self.config['metrics']:
            for k in self.config['topk']:
                results[f"{metric}@{k}{suffix}"] = self.metric2func[metric](pos_index, k)


        if suffix == "":
            weighted_score = self.calculate_weighted_score(preds, labels)
            results['weighted_score'] = weighted_score

        return results

    def print_final_stats(self):

        if self.batch_legal_at10:
            avg_legal_at10 = sum(self.batch_legal_at10) / len(self.batch_legal_at10)
            print(f"[SID_STATS] Top-10 合法率: {avg_legal_at10:.3f}")
