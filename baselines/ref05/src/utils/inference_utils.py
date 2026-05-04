import datetime
import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Union

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from google.cloud import bigquery

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter

from src.models.components.interfaces import ModelOutput
from src.utils.decorators import RetriesFailedException, retry
from src.utils.tensor_utils import merge_list_of_keyed_tensors_to_single_tensor

log = logging.getLogger(__name__)


class BaseBufferedWriter(BasePredictionWriter):
    def __init__(
        self,

        flush_frequency: int = 5000,
        write_interval: str = "batch",
        schema: Optional[List[Union[bigquery.SchemaField, pa.Field]]] = None,
        prediction_key_name: Optional[str] = None,
        prediction_name: Optional[str] = None,
    ):


        super().__init__(write_interval)
        self.flush_frequency = flush_frequency

        self.rows_buffer: List[dict] = []
        self.schema = schema
        self.global_rank = None
        self.prediction_key_name = prediction_key_name
        self.prediction_name = prediction_name

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        self.global_rank = trainer.global_rank if trainer.global_rank else 0
        log.info(f"Rank {self.global_rank} initialized for inference.")


        if (
            hasattr(pl_module, "prediction_key_name")
            and pl_module.prediction_key_name is None
            and self.prediction_key_name is not None
        ):
            pl_module.prediction_key_name = self.prediction_key_name
        if (
            hasattr(pl_module, "prediction_name")
            and pl_module.prediction_name is None
            and self.prediction_name is not None
        ):
            pl_module.prediction_name = self.prediction_name

    def flush_buffer(self) -> None:

        if self.rows_buffer:
            self._flush_buffer()
            self.rows_buffer.clear()

        else:
            log.info("Buffer is empty, nothing to flush.")

    def _flush_buffer(self) -> None:

        raise NotImplementedError(
            "You need to implement the `_flush_buffer` method in your subclass."
        )

    def handle_batch(self, model_output: ModelOutput) -> None:


        if model_output is None:
            log.warning(
                f"Rank {self.global_rank} received an empty model output. Skipping this batch. This is expected if the batch is a dummy batch."
            )
            return None
        rows = model_output.list_of_row_format

        if len(rows) + len(self.rows_buffer) < self.flush_frequency:
            self.rows_buffer.extend(rows)
        else:
            rows_left = self.flush_frequency - len(self.rows_buffer)
            self.rows_buffer.extend(rows[:rows_left])
            self.flush_buffer()
            self.rows_buffer.extend(rows[rows_left:])

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: ModelOutput,
        batch_indices: List[int],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:


        self.handle_batch(prediction)

    def write_on_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        predictions: List[ModelOutput],
        batch_indices: List[List[int]],
    ) -> None:


        for batch_pred in predictions:
            self.handle_batch(batch_pred)

        self.flush_buffer()

    def on_predict_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:


        self.flush_buffer()
        log.info(f"Rank {self.global_rank} finished writing predictions.")


class LocalPickleWriter(BaseBufferedWriter):


    def __init__(
        self,
        output_dir: str,
        flush_frequency: int = 1000,
        write_interval: str = "batch",
        should_merge_files_on_main: bool = True,
        should_merge_list_of_keyed_tensors_to_single_tensor: bool = True,
        post_processing_functions: Optional[List[Dict[str, callable]]] = [],
        **kwargs,
    ):


        super().__init__(
            write_interval=write_interval, flush_frequency=flush_frequency, **kwargs
        )
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.should_merge_files_on_main = should_merge_files_on_main
        self.should_merge_list_of_keyed_tensors_to_single_tensor = (
            should_merge_list_of_keyed_tensors_to_single_tensor
        )
        self.post_processing_functions = post_processing_functions

    def _create_file_path(self) -> str:

        return f"predictions_{self.global_rank}_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')[:-3]}.pkl"

    def _local_file_path(self, file_path: Optional[str] = None) -> str:

        return (
            f"{self.output_dir}/{file_path if file_path else self._create_file_path()}"
        )

    @retry()
    def _flush_buffer(self):


        file_path = self._create_file_path()
        with open(self._local_file_path(file_path=file_path), "wb") as f:
            pickle.dump(self.rows_buffer, f)

        log.info(
            f"Global Rank: {self.global_rank} wrote {len(self.rows_buffer)} rows to {self._local_file_path(file_path=file_path)}."
        )

    @retry()
    def on_predict_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        super().on_predict_end(trainer, pl_module)

        if self.should_merge_files_on_main:


            if trainer.global_rank != None:
                torch.distributed.barrier()
            if self.global_rank == 0:
                log.info("Merging pickle files on main process.")
                self._merge_files()


            if trainer.global_rank != None:
                torch.distributed.barrier()


        for process_func in self.post_processing_functions:
            all_files = [f for f in os.listdir(self.output_dir)]
            for file in all_files:
                file_path = os.path.join(self.output_dir, file)
                if process_func.get("main_only", False):
                    if self.global_rank == 0:
                        process_func["function"](file_path)
                else:
                    process_func["function"](file_path)
                if trainer.global_rank != None:
                    torch.distributed.barrier()

    def _merge_files(self):

        all_files = [f for f in os.listdir(self.output_dir) if f.endswith(".pkl")]
        merged_data = []
        for file in all_files:
            with open(os.path.join(self.output_dir, file), "rb") as f:
                merged_data.extend(pickle.load(f))
            os.remove(os.path.join(self.output_dir, file))
        with open(os.path.join(self.output_dir, "merged_predictions.pkl"), "wb") as f:
            pickle.dump(merged_data, f)
        log.info(f"Merged {len(merged_data)} rows into merged_predictions.pkl.")

        if self.should_merge_list_of_keyed_tensors_to_single_tensor:
            merged_data_tensor = merge_list_of_keyed_tensors_to_single_tensor(
                data=merged_data,
                index_key=self.prediction_key_name,
                value_key=self.prediction_name,
            )
            torch.save(
                merged_data_tensor.cpu(),
                os.path.join(self.output_dir, "merged_predictions_tensor.pt"),
            )
        log.info(
            f"Merged {len(merged_data_tensor)} rows into merged_predictions_tensor.pt. as pytorch tensor"
        )
