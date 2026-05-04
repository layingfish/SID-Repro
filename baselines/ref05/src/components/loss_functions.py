import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Dict

class FullBatchCrossEntropyLoss(nn.Module):


    def __init__(
        self,
        normalize: bool = True,
        **kwargs,
    ):


        super().__init__()
        self.normalize = normalize
        self.cross_entroy_loss = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        query_embeddings: torch.Tensor,
        key_embeddings: torch.Tensor,
        label_locations: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:


        query_embeddings = query_embeddings[
            label_locations[:, 0], label_locations[:, 1]
        ]

        if self.normalize:
            query_embeddings = F.normalize(query_embeddings, dim=-1)
            key_embeddings = F.normalize(key_embeddings, dim=-1)

        logits = torch.mm(query_embeddings, key_embeddings.t())

        loss = self.cross_entroy_loss(logits, labels.long())

        return loss

class WeightedSquaredError(torch.nn.Module):
    def __init__(self):

        super().__init__()

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:


        error = x - y
        squared_error = torch.sum(error**2, dim=-1)


        if weights is None:
            return torch.sum(squared_error)
        return torch.sum(weights * squared_error)

class BetaQuantizationLoss(torch.nn.Module):
    def __init__(self, beta: float = 0.25, reduction: str = "sum"):


        super().__init__()
        self.beta = beta
        self.criterion = torch.nn.MSELoss(reduction=reduction)

    def forward(self, x: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:


        x_no_grad = x.detach()
        xq_no_grad = xq.detach()
        loss = self.criterion(x_no_grad, xq) + self.beta * self.criterion(x, xq_no_grad)
        return loss
