"""Drop-in 兼容的 SemanticIdTokenizer 替代品。

从 cached_ids.npy 直接加载预计算的 item codes，
不依赖 RQ-VAE、ItemData、polars、gin 等上游依赖。

接口与 SemanticIdTokenizer 完全一致：
  - cached_ids: (N, sem_ids_dim) Tensor
  - sem_ids_dim: int
  - precompute_corpus_ids(item_dataset): 无操作（已预加载）
  - forward(batch: SeqBatch) -> TokenizedSeqBatch
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn, Tensor
from einops import rearrange

# 直接导入 schema（这个模块依赖很轻，只有 torch）
from data.schemas import SeqBatch, TokenizedSeqBatch


class CachedIdTokenizer(nn.Module):
    """从预计算的 cached_ids.npy 加载 item codes，替代 SemanticIdTokenizer。"""

    def __init__(
        self,
        cached_ids_path: str,
        codebook_size: int,
        per_pos_sizes: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()

        # 加载 cached_ids (N, D)
        ids_np = np.load(cached_ids_path).astype(np.int64)
        self.register_buffer("cached_ids", torch.tensor(ids_np, dtype=torch.long))

        self.codebook_size = codebook_size
        self.per_pos_sizes = tuple(int(x) for x in per_pos_sizes) if per_pos_sizes is not None else None
        self.n_items = self.cached_ids.shape[0]

        # sem_ids_dim = cached_ids 的列数（含 dedup suffix）
        self._sem_ids_dim = self.cached_ids.shape[1]

        print(
            f"[CachedIdTokenizer] loaded {self.n_items} items, "
            f"sem_ids_dim={self._sem_ids_dim}, codebook_size={codebook_size}, "
            f"per_pos_sizes={self.per_pos_sizes}"
        )

    @property
    def sem_ids_dim(self) -> int:
        return self._sem_ids_dim

    def precompute_corpus_ids(self, item_dataset) -> Tensor:
        """兼容接口：无操作，cached_ids 已在 __init__ 加载。"""
        return self.cached_ids

    def _tokenize_seq_batch_from_cached(self, ids: Tensor) -> Tensor:
        """将 item id 序列转换为 semantic id 序列（与原始实现完全一致）。"""
        return rearrange(
            self.cached_ids[ids.flatten(), : self.sem_ids_dim],
            "(b n) d -> b (n d)",
            n=ids.shape[1],
        )

    @torch.no_grad()
    def forward(self, batch: SeqBatch) -> TokenizedSeqBatch:
        """将 SeqBatch 转换为 TokenizedSeqBatch（与原始实现完全一致）。"""
        B, N = batch.ids.shape
        D = self.sem_ids_dim

        sem_ids = self._tokenize_seq_batch_from_cached(batch.ids)
        seq_mask = batch.seq_mask.repeat_interleave(D, dim=1)
        sem_ids[~seq_mask] = -1

        sem_ids_fut = self._tokenize_seq_batch_from_cached(batch.ids_fut)

        token_type_ids = torch.arange(D, device=sem_ids.device).repeat(B, N)
        token_type_ids_fut = torch.arange(D, device=sem_ids.device).repeat(B, 1)

        return TokenizedSeqBatch(
            user_ids=batch.user_ids,
            sem_ids=sem_ids,
            sem_ids_fut=sem_ids_fut,
            seq_mask=seq_mask,
            token_type_ids=token_type_ids,
            token_type_ids_fut=token_type_ids_fut,
        )
