import functools
from typing import Optional, Tuple

import torch

from src.components.distance_functions import DistanceFunction
from src.components.clustering_initializers import ClusteringInitializer
from src.components.loss_functions import WeightedSquaredError
from src.components.quantization_strategies import QuantizationStrategy
from src.models.modules.clustering.base_clustering_module import BaseClusteringModule


class VectorQuantization(BaseClusteringModule):
    def __init__(
        self,
        n_clusters: int,
        n_features: int,
        distance_function: DistanceFunction,
        initializer: ClusteringInitializer,
        quantization_strategy: QuantizationStrategy,
        loss_function: torch.nn.Module = WeightedSquaredError(),
        optimizer: torch.optim.Optimizer = functools.partial(
            torch.optim.SGD,
            lr=0.5,
        ),
        init_buffer_size: int = 1000,
    ):


        super().__init__(
            n_clusters=n_clusters,
            n_features=n_features,
            distance_function=distance_function,
            loss_function=loss_function,
            optimizer=optimizer,
            initializer=initializer,
            init_buffer_size=init_buffer_size,
        )

        self.quantization_strategy = quantization_strategy

    def forward(
        self, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:


        codebook = self.get_centroids()
        (
            ids,
            embeddings,
            reconstruction_loss_embeddings,
        ) = self.quantization_strategy.quantize(
            codebook=codebook,
            batch=batch,
        )
        return ids, embeddings, reconstruction_loss_embeddings

    def model_step(
        self,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:


        if batch.device != self.device:
            batch = batch.to(self.device)


        if self.is_initial_step:
            self.is_initial_step = False
            self.is_initialized = True
        if not self.is_initialized:
            return self.initialization_step(batch)

        assignments, embeddings, reconstruction_loss_embeddings = self.forward(batch)
        loss = self.loss_function(batch, embeddings)
        return (
            assignments,
            reconstruction_loss_embeddings
            if reconstruction_loss_embeddings is not None
            else embeddings,
            loss,
        )
