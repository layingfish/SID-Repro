from collections import defaultdict
from einops import rearrange
from torch import Tensor


class TopKAccumulator:
    def __init__(self, ks=[1, 5, 10]):
        self.ks = ks
        self.reset()

    def reset(self):
        self.total = 0
        self.metrics = defaultdict(int)

    def accumulate(self, actual: Tensor, top_k: Tensor) -> None:
        B, D = actual.shape
        pos_match = (rearrange(actual, "b d -> b 1 d") == top_k)
        for i in range(D):
            match_found, rank = pos_match[...,:i+1].all(axis=-1).max(axis=-1)
            matched_rank = rank[match_found]
            for k in self.ks:
                in_topk = matched_rank < k
                self.metrics[f"h@{k}_slice_:{i+1}"] += len(matched_rank[in_topk])
                if len(matched_rank) > 0:
                    discount = 1.0 / (matched_rank[in_topk].float() + 2.0).log2()
                    self.metrics[f"ndcg@{k}_slice_:{i+1}"] += float(discount.sum().item())

            match_found, rank = pos_match[...,i:i+1].all(axis=-1).max(axis=-1)
            matched_rank = rank[match_found]
            for k in self.ks:
                in_topk = matched_rank < k
                self.metrics[f"h@{k}_pos_{i}"] += len(matched_rank[in_topk])
                if len(matched_rank) > 0:
                    discount = 1.0 / (matched_rank[in_topk].float() + 2.0).log2()
                    self.metrics[f"ndcg@{k}_pos_{i}"] += float(discount.sum().item())
        self.total += B

    def reduce(self) -> dict:
        return {k: v/self.total for k, v in self.metrics.items()}
