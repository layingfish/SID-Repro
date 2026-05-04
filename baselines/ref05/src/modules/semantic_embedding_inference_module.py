from typing import Dict, Optional, Union

import torch
import transformers
from lightning import LightningModule
from torch import nn

from src.data.loading.components.interfaces import ItemData
from src.models.components.interfaces import OneKeyPerPredictionOutput


class SemanticEmbeddingInferenceModule(LightningModule):
    def __init__(
        self,
        semantic_embedding_model: Union[nn.Module, transformers.PreTrainedModel],
        semantic_embedding_model_input_map: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> None:


        super().__init__()

        self.semantic_embedding_model = semantic_embedding_model

        for param in self.semantic_embedding_model.parameters():
            param.requires_grad = False
        self.semantic_embedding_model_input_map = semantic_embedding_model_input_map

    def forward(self, model_input: ItemData) -> torch.Tensor:


        semantic_embedding_model_input_name_to_feature = {
            input_embedding_model_input_name: model_input.transformed_features[
                feature_name
            ]
            for input_embedding_model_input_name, feature_name in self.semantic_embedding_model_input_map.items()
        }
        with torch.no_grad():
            semantic_embeddings = self.semantic_embedding_model(
                **semantic_embedding_model_input_name_to_feature
            )
        return semantic_embeddings

    def model_step(self, model_input: ItemData) -> torch.Tensor:
        semantic_embeddings = self.forward(model_input)
        return semantic_embeddings

    def predict_step(self, batch: ItemData) -> OneKeyPerPredictionOutput:


        semantic_embeddings = self.model_step(batch)
        item_ids = [
            item_id.item() if isinstance(item_id, torch.Tensor) else item_id
            for item_id in batch.item_ids
        ]

        model_output = OneKeyPerPredictionOutput(
            keys=item_ids,
            predictions=semantic_embeddings,
            key_name="item_id",
            prediction_name="embedding",
        )
        return model_output
