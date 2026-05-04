from abc import abstractmethod
from functools import partial

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from src.components.distance_functions import DistanceFunction


class ClusteringInitializer(nn.Module):


    def __init__(self, n_clusters: int, initialize_on_cpu: bool = False):


        super().__init__()
        self.n_clusters = n_clusters
        self.initialize_on_cpu = initialize_on_cpu

    @abstractmethod
    def forward(self, buffer: torch.Tensor) -> torch.Tensor:


        pass


class RandomInitializer(ClusteringInitializer):


    def __init__(self, n_clusters: int, initialize_on_cpu: bool = True):


        super().__init__(n_clusters=n_clusters, initialize_on_cpu=initialize_on_cpu)

    def forward(self, buffer: torch.Tensor) -> torch.Tensor:


        if self.initialize_on_cpu:
            old_device = buffer.device

            buffer = buffer.to("cpu")

        n_samples = buffer.shape[0]


        indices = torch.randperm(n_samples, device=buffer.device)[: self.n_clusters]
        centroids = buffer[indices].clone().data
        if self.initialize_on_cpu:

            centroids = centroids.to(old_device)

        return centroids


class KMeansPlusPlusInitInitializer(ClusteringInitializer):


    def __init__(
        self,
        n_clusters: int,
        distance_function: DistanceFunction,
        initialize_on_cpu: bool = True,
    ):


        super().__init__(n_clusters=n_clusters, initialize_on_cpu=initialize_on_cpu)
        self.distance_function = distance_function

    def forward(self, buffer: torch.Tensor) -> torch.Tensor:


        if self.initialize_on_cpu:
            old_device = buffer.device

            buffer = buffer.to("cpu")

        n_samples = buffer.shape[0]
        n_features = buffer.shape[1]
        centroids = torch.zeros(
            (self.n_clusters, n_features), dtype=buffer.dtype, device=buffer.device
        )


        first_centroid_idx = torch.randint(0, n_samples, (1,), device=buffer.device)
        centroids[0] = buffer[first_centroid_idx]


        for i in range(1, self.n_clusters):

            min_distances = torch.min(
                self.distance_function.compute(buffer, centroids[:i]), dim=1
            )[0]
            if min_distances.sum() == 0:


                centroids[i:] = buffer[
                    torch.randint(
                        0, n_samples, (self.n_clusters - i,), device=buffer.device
                    )
                ]
                break


            next_centroid_idx = torch.multinomial(min_distances, num_samples=1)


            centroids[i] = buffer[next_centroid_idx]

        if self.initialize_on_cpu:

            centroids = centroids.to(old_device)

        return centroids


class ClusteringModuleInitializer(ClusteringInitializer):


    def __init__(
        self,
        n_clusters: int,
        clustering_module: LightningModule,
        initialize_on_cpu: bool = False,
        max_iter: int = 100,
        atol: float = 1e-8,
    ):


        super().__init__(n_clusters=n_clusters, initialize_on_cpu=initialize_on_cpu)

        from src.models.modules.clustering.base_clustering_module import (
            BaseClusteringModule,
        )

        assert isinstance(
            clustering_module, BaseClusteringModule
        ), "clustering_module must be an instance of BaseClusteringModule"

        self.clustering_module = clustering_module
        self.max_iter = max_iter
        self.atol = atol

    def forward(self, buffer: torch.Tensor) -> torch.Tensor:


        self.clustering_module.on_train_start()
        cur_centroids = self.clustering_module.get_centroids()
        for step in range(self.max_iter):

            self.clustering_module.model_step(buffer)
            new_centroids = self.clustering_module.get_centroids()


            if step > 0 and torch.allclose(
                cur_centroids, new_centroids, atol=self.atol
            ):
                print(f"Initialization converged after {step} iterations")
                break
            cur_centroids = new_centroids.clone()

        return cur_centroids.detach()
