from typing import Any, Dict, Optional, Tuple

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities import rank_zero_only
from torch import nn
from torchmetrics import MeanMetric

from src.components.distance_functions import DistanceFunction
from src.components.clustering_initializers import ClusteringInitializer
from src.components.loss_functions import WeightedSquaredError


class BaseClusteringModule(LightningModule):
    def __init__(
        self,
        n_clusters: int,
        n_features: int,
        distance_function: DistanceFunction,
        initializer: ClusteringInitializer,
        loss_function: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        init_buffer_size: int = 1000,
        update_manually: bool = False,
    ):


        super(BaseClusteringModule, self).__init__()

        self.n_clusters = n_clusters
        self.n_features = n_features
        self.distance_function = distance_function
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.initializer = initializer
        self.init_buffer_size = init_buffer_size

        self.centroids = torch.nn.Parameter(
            torch.zeros(self.n_clusters, self.n_features), requires_grad=True
        )
        self.update_manually = update_manually
        if self.update_manually:
            self.centroids.requires_grad = False

        self.init_loss_function = WeightedSquaredError()
        self.init_buffer = torch.tensor([])
        self.is_initialized = False
        self.is_initial_step = False
        self.train_loss = MeanMetric()

    def _buffer_points(self, batch: torch.Tensor) -> None:


        batch = batch.detach()
        n_to_add = min(
            self.init_buffer_size - self.init_buffer.shape[0], batch.shape[0]
        )
        self.init_buffer = torch.cat([self.init_buffer, batch[:n_to_add]], dim=0)

    @rank_zero_only
    def compute_initial_centroids(self, buffer: torch.Tensor) -> None:


        if buffer.shape[0] < self.n_clusters:
            raise ValueError(
                f"Buffer size {buffer.shape[0]} is less than the number of clusters"
                f" {self.n_clusters}."
            )

        self.init_centroids = self.initializer(buffer)

    def initialization_step(
        self,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:


        self._buffer_points(batch)
        if self.init_buffer.shape[0] < self.init_buffer_size:
            centroid_zero_embeddings = torch.zeros_like(
                self.centroids.data, dtype=batch.dtype, device=self.device
            )
            loss = self.init_loss_function(self.centroids, centroid_zero_embeddings)


            batch_zero_embeddings = torch.zeros_like(
                batch, dtype=batch.dtype, device=self.device
            )
            batch_zero_assignments = torch.zeros(
                batch.shape[0], dtype=torch.long, device=self.device
            )
            return batch_zero_assignments, batch_zero_embeddings, loss
        else:
            self.init_centroids = torch.zeros_like(
                self.centroids.data, dtype=batch.dtype, device=self.device
            )


            self.compute_initial_centroids(self.init_buffer)
            self.is_initial_step = True
            self.init_buffer = torch.tensor([], device=self.device)

            if self.update_manually:


                self.centroids[:] = self.init_centroids.data
                distances = self.distance_function.compute(batch, self.centroids.data)
                assignments = torch.argmin(distances, dim=1).to(self.device)
                return assignments, self.centroids[assignments], None

            loss = self.init_loss_function(self.centroids, self.init_centroids)
            distances = self.distance_function.compute(batch, self.init_centroids)
            assignments = torch.argmin(distances, dim=1).to(self.device)
            return assignments, self.init_centroids[assignments], loss

    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool]:


        raise NotImplementedError(
            "Inherit from this class and implement the forward method."
        )

    def model_step(
        self, batch: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:


        raise NotImplementedError(
            "Inherit from this class and implement the forward method."
        )

    def training_step(
        self, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:


        _, _, loss = self.model_step(batch)

        self.train_loss(loss)
        train_dict_to_log = {
            "train/loss": self.train_loss,
        }
        self.log_dict(
            train_dict_to_log,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def predict_step(
        self, batch: torch.Tensor, return_embeddings: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:


        batch = batch.to(self.device)
        with torch.no_grad():
            centroids = self.get_centroids().data
            distances = self.distance_function.compute(batch, centroids)
            assignments = torch.argmin(distances, dim=1)
            if not return_embeddings:
                return assignments
            return assignments, centroids[assignments]

    def get_centroids(self) -> nn.Parameter:


        return self.centroids

    def get_residuals(self, batch: torch.Tensor) -> torch.Tensor:


        _, centroids = self.predict_step(batch)
        return batch - centroids

    def on_train_start(self) -> None:

        self.train_loss.reset()
        self.init_buffer = torch.tensor([], device=self.device)
        self.centroids = self.centroids.to(self.device)

    def configure_optimizers(self) -> Dict[str, Any]:


        optimizer = self.optimizer(params=(self.centroids,))
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss",
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
