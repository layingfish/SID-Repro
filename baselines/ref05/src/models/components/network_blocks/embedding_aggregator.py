import torch
import torch.nn as nn

from src.models.components.network_blocks.aggregation_strategy import (
    AggregationStrategy,
)


class EmbeddingAggregator(nn.Module):


    def __init__(
        self,
        aggregation_strategy: AggregationStrategy,
    ):
        super(EmbeddingAggregator, self).__init__()
        self.aggregation_strategy = aggregation_strategy

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:


        last_item_index = attention_mask.sum(dim=1) - 1


        dummy_tensor_for_batch_shape = attention_mask[:, 0]


        ones_tensor = torch.ones_like(dummy_tensor_for_batch_shape, dtype=torch.long)


        row_ids = torch.cumsum(ones_tensor, dim=0) - 1

        return self.aggregation_strategy.aggregate(embeddings, row_ids, last_item_index)
