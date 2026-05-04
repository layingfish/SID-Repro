from typing import Any, Dict, Optional, Union

import torch
import transformers
from lightning import LightningModule
from torchmetrics import MeanMetric
from torchmetrics.aggregation import BaseAggregator

from src.components.eval_metrics import Evaluator
from src.utils.pylogger import RankedLogger

command_line_logger = RankedLogger(__name__, rank_zero_only=True)


class BaseModule(LightningModule):
    def __init__(
        self,
        model: Union[torch.nn.Module, transformers.PreTrainedModel],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        loss_function: torch.nn.Module,
        evaluator: Evaluator,
        training_loop_function: callable = None,
    ) -> None:


        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.evaluator = evaluator
        self.training_loop_function = training_loop_function

        self._prediction_key_name = None
        self._prediction_name = None

        if self.training_loop_function is not None:
            self.automatic_optimization = False

        if self.evaluator:
            for metric_name, metric_object in self.evaluator.metrics.items():
                setattr(self, metric_name, metric_object)


            self.train_loss = MeanMetric()
            self.val_loss = MeanMetric()
            self.test_loss = MeanMetric()

    @property
    def prediction_key_name(self) -> Optional[str]:
        return self._prediction_key_name

    @prediction_key_name.setter
    def prediction_key_name(self, value: str) -> None:
        command_line_logger.debug(f"Setting prediction_key_name to {value}")
        self._prediction_key_name = value

    @property
    def prediction_name(self) -> Optional[str]:
        return self._prediction_name

    @prediction_name.setter
    def prediction_name(self, value: str) -> None:
        command_line_logger.debug(f"Setting prediction_name to {value}")
        self._prediction_name = value

    def forward(
        self,
        **kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Inherit from this class and implement the forward method."
        )

    def model_step(
        self,
        model_input: Any,
        label_data: Optional[Any] = None,
    ):
        raise NotImplementedError(
            "Inherit from this class and implement the model_step method."
        )

    def on_train_start(self) -> None:


        self.val_loss.reset()
        self.evaluator.reset()
        self.train_loss.reset()
        self.test_loss.reset()

    def on_validation_epoch_start(self) -> None:

        self.val_loss.reset()
        self.evaluator.reset()

    def on_test_epoch_start(self):
        self.test_loss.reset()
        self.evaluator.reset()

    def on_validation_epoch_end(self) -> None:

        self.log("val/loss", self.val_loss, sync_dist=False, prog_bar=True, logger=True)
        self.log_metrics("val")

    def on_test_epoch_end(self) -> None:
        self.log(
            "test/loss", self.test_loss, sync_dist=False, prog_bar=True, logger=True
        )
        self.log_metrics("test")

    def on_exception(self, exception):
        self.trainer.should_stop = True
        self.trainer.logger.finalize(status="failure")

    def log_metrics(
        self,
        prefix: str,
        on_step=False,
        on_epoch=True,


        sync_dist=False,
        logger=True,
        prog_bar=False,
        call_compute=False,
    ) -> Dict[str, Any]:

        metrics_dict = {
            f"{prefix}/{metric_name}": metric_object.compute()
            if call_compute
            else metric_object
            for metric_name, metric_object in self.evaluator.metrics.items()
        }

        self.log_dict(
            metrics_dict,
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=sync_dist,
            logger=logger,
            prog_bar=prog_bar,
        )

    def setup(self, stage: str) -> None:


        pass

    def configure_optimizers(self) -> Dict[str, Any]:


        optimizer = self.optimizer(params=self.trainer.model.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def eval_step(self, batch: Any, loss_to_aggregate: BaseAggregator):
        raise NotImplementedError("eval_step method must be implemented.")

    def validation_step(
        self,
        batch: Any,
        batch_idx: int,
    ) -> None:


        self.eval_step(batch, self.val_loss)

    def test_step(
        self,
        batch: Any,
        batch_idx: int,
    ) -> None:


        self.eval_step(batch, self.test_loss)
