from typing import Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from src.models.components.network_blocks.embedding_aggregator import (
    EmbeddingAggregator,
)


class HFLanguageModel(nn.Module):
    def __init__(
        self,
        huggingface_model: PreTrainedModel,
        aggregator: EmbeddingAggregator,
        postprocessor: nn.Module = nn.Identity(),
        return_last_hidden_states: bool = False,
    ):


        super(HFLanguageModel, self).__init__()
        self.huggingface_model = huggingface_model
        self.aggregator = aggregator
        self.postprocessor = postprocessor
        self.return_last_hidden_states = return_last_hidden_states

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:


        outputs: BaseModelOutput = self.huggingface_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        embeddings = outputs.last_hidden_state
        aggregated_embeddings = self.aggregator(embeddings, attention_mask)
        postprocessed_embeddings = self.postprocessor(aggregated_embeddings)
        if self.return_last_hidden_states:
            return postprocessed_embeddings, embeddings
        return postprocessed_embeddings
