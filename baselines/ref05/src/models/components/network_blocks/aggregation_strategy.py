from abc import ABC, abstractmethod
from typing import Optional

import torch

from src.utils.masking_utils import create_last_k_mask


class AggregationStrategy(ABC):
    @abstractmethod
    def aggregate(
        self,
        embeddings: torch.Tensor,
        row_ids: torch.Tensor,
        last_item_index: torch.Tensor,
    ) -> torch.Tensor:
        pass


class MeanAggregation(AggregationStrategy):


    def __init__(self, last_k: Optional[int] = None):


        self.last_k = last_k

    def aggregate(
        self,
        embeddings: torch.Tensor,
        row_ids: torch.Tensor,
        last_item_index: torch.Tensor,
    ) -> torch.Tensor:


        embeddings = embeddings[
            row_ids
        ]

        mask = create_last_k_mask(embeddings.size(1), last_item_index, self.last_k)
        mask = mask.to(dtype=embeddings.dtype, device=embeddings.device)


        masked_embeddings = embeddings * mask.unsqueeze(
            2
        )


        sum_embeddings = torch.sum(
            masked_embeddings, dim=1
        )
        count = (
            torch.sum(mask, dim=1).clamp(min=1).unsqueeze(1)
        )
        return sum_embeddings / count


class LastAggregation(AggregationStrategy):
    def aggregate(
        self,
        embeddings: torch.Tensor,
        row_ids: torch.Tensor,
        last_item_index: torch.Tensor,
    ) -> torch.Tensor:
        return embeddings[row_ids, last_item_index]


class FirstAggregation(AggregationStrategy):
    def aggregate(
        self,
        embeddings: torch.Tensor,
        row_ids: torch.Tensor,
        last_item_index: torch.Tensor,
    ) -> torch.Tensor:


        return embeddings[row_ids, 0]
