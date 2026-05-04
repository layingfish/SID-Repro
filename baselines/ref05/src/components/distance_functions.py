from abc import ABC, abstractmethod
from typing import Optional
import torch


class DistanceFunction(ABC):
    @abstractmethod
    def compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:


        pass


class SquaredEuclideanDistance(DistanceFunction):
    def compute(
        self, x: torch.Tensor, y: torch.Tensor, batch_size: int = 256
    ) -> torch.Tensor:


        assert x.dim() == 2, f"Data must be 2D, got {x.dim()} dimensions"
        assert y.dim() == 2, f"Data must be 2D, got {y.dim()} dimensions"
        assert x.size(1) == y.size(1), f"Data must have the same number of columns"

        n1, d = x.shape
        n2, _ = y.shape

        if batch_size is None or batch_size >= n1:

            x_expanded = x.unsqueeze(1)
            y_expanded = y.unsqueeze(0)
            sq_diffs = (x_expanded - y_expanded).pow(2)
            sq_distances = torch.sum(sq_diffs, dim=2)
            return sq_distances
        else:

            all_sq_distances = []
            num_batches = (n1 + batch_size - 1) // batch_size

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n1)
                x_batch = x[start_idx:end_idx]


                x_batch_expanded = x_batch.unsqueeze(
                    1
                )
                y_expanded = y.unsqueeze(0)

                sq_diffs_batch = (x_batch_expanded - y_expanded).pow(
                    2
                )
                sq_distances_batch = torch.sum(
                    sq_diffs_batch, dim=2
                )

                all_sq_distances.append(sq_distances_batch)


            return torch.cat(all_sq_distances, dim=0)

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
