import functools
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.components.distance_functions import DistanceFunction
from src.components.clustering_initializers import (
    ClusteringInitializer,
    KMeansPlusPlusInitInitializer,
)
from src.components.loss_functions import WeightedSquaredError
from src.models.modules.clustering.base_clustering_module import BaseClusteringModule


class MiniBatchKMeans(BaseClusteringModule):
    def __init__(
        self,
        n_clusters: int,
        n_features: int,
        distance_function: DistanceFunction,
        initializer: ClusteringInitializer = KMeansPlusPlusInitInitializer,
        loss_function: torch.nn.Module = WeightedSquaredError(),
        optimizer: torch.optim.Optimizer = functools.partial(
            torch.optim.SGD,
            lr=0.5,
        ),
        init_buffer_size: int = 1000,
        update_manually: bool = False,
    ):


        super().__init__(
            n_clusters=n_clusters,
            n_features=n_features,
            distance_function=distance_function,
            loss_function=loss_function,
            optimizer=optimizer,
            initializer=initializer,
            init_buffer_size=init_buffer_size,
            update_manually=update_manually,
        )
        self.cluster_counts = torch.zeros(self.n_clusters)

    def forward(
        self, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:


        assignments = self.predict_step(batch, return_embeddings=False)
        assignments_one_hot = (
            nn.functional.one_hot(assignments, self.n_clusters)
        ).detach()

        batch_cluster_counts = torch.sum(assignments_one_hot, dim=0)
        self.cluster_counts += batch_cluster_counts

        batch_cluster_sums = torch.mm(assignments_one_hot.float().t(), batch)

        return assignments, batch_cluster_counts, batch_cluster_sums

    def model_step(
        self, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:


        batch = batch.to(self.device)


        if self.is_initial_step:
            self.is_initial_step = False
            self.is_initialized = True
        if not self.is_initialized:
            return self.initialization_step(batch)

        assignments, batch_cluster_counts, batch_cluster_sums = self.forward(batch)

        centroids = self.get_centroids()

        mask = batch_cluster_counts != 0
        mask_target = batch_cluster_sums[mask] / batch_cluster_counts[mask].unsqueeze(1)
        centroid_weights = batch_cluster_counts[mask] / self.cluster_counts[mask]

        if self.update_manually:
            self.centroids[mask] = self.centroids[mask].data - (
                (centroids[mask].data - mask_target) * centroid_weights.unsqueeze(1)
            )
            return assignments, centroids[assignments], None
        else:


            loss = self.loss_function(centroids[mask], mask_target, centroid_weights)
            return assignments, centroids[assignments], loss

    def on_train_start(self) -> None:

        self.cluster_counts = torch.zeros(self.n_clusters, device=self.device)
        super().on_train_start()
