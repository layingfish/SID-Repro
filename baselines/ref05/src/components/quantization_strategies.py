from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn.functional as F

from src.components.distance_functions import DistanceFunction
from src.utils.utils import gumbel_softmax_sample


class QuantizationStrategy(ABC):


    def __init__(
        self,
        distance_function: DistanceFunction,
        compute_reconstruction_loss_embeddings: bool = False,
    ):


        self.distance_function = distance_function
        self.compute_reconstruction_loss_embeddings = (
            compute_reconstruction_loss_embeddings
        )

    def get_nearest_neighbors(
        self,
        codebook: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:


        dists = self.distance_function.compute(batch, codebook)
        ids = torch.argmin(dists, dim=-1)
        return ids, codebook[ids]

    @abstractmethod
    def quantize(
        self,
        codebook: torch.Tensor,
        batch: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:


        pass


class GumbelSoftmaxQuantization(QuantizationStrategy):
    def __init__(
        self,
        temperature: float = 0.7,
        **kwargs,
    ):


        super().__init__(**kwargs)
        self.temperature = temperature

    def quantize(
        self,
        codebook: torch.Tensor,
        batch: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dists = self.distance_function.compute(batch, codebook)
        weights = gumbel_softmax_sample(-dists, temperature=self.temperature)
        embeddings = weights @ codebook
        reconstruction_loss_embeddings = embeddings
        ids = torch.argmax(weights, dim=-1)
        return ids, embeddings, reconstruction_loss_embeddings


class STEQuantization(QuantizationStrategy):
    def quantize(
        self,
        codebook: torch.Tensor,
        batch: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:


        ids, embeddings = self.get_nearest_neighbors(codebook, batch)
        reconstruction_loss_embeddings = batch + (embeddings - batch).detach()
        return ids, embeddings, reconstruction_loss_embeddings


class RotationTrickQuantization(QuantizationStrategy):
    def rotate_and_scale_batch(
        self,
        batch: torch.Tensor,
        quantized_embeddings: torch.Tensor,
    ) -> torch.Tensor:


        quantized_embeddings = quantized_embeddings.detach()
        detached_batch = batch.detach()

        quantized_norms = torch.linalg.vector_norm(
            quantized_embeddings, dim=-1
        ).unsqueeze(
            1
        )
        batch_norms = torch.linalg.vector_norm(detached_batch, dim=-1).unsqueeze(
            1
        )
        lambda_ = quantized_norms / batch_norms

        normalized_batch = detached_batch / batch_norms
        normalized_embeddings = (
            quantized_embeddings / quantized_norms
        )

        normalized_sum = F.normalize(
            normalized_batch + normalized_embeddings, p=2, dim=1
        )
        batch = batch.unsqueeze(1)


        sum_projection = (
            batch @ normalized_sum.unsqueeze(2) @ normalized_sum.unsqueeze(1)
        )
        rescaled_embeddings = (
            batch @ normalized_batch.unsqueeze(2) @ normalized_embeddings.unsqueeze(1)
        )
        return (
            lambda_ * (batch - 2 * sum_projection + 2 * rescaled_embeddings).squeeze()
        )

    def quantize(
        self,
        codebook: torch.Tensor,
        batch: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:


        ids, embeddings = self.get_nearest_neighbors(codebook, batch)
        reconstruction_loss_embeddings = self.rotate_and_scale_batch(
            batch, embeddings
        )
        return ids, embeddings, reconstruction_loss_embeddings
