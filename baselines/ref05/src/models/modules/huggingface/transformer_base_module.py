from typing import Dict, Optional, Tuple

import torch
import transformers
from torchmetrics.aggregation import BaseAggregator

from src.components.eval_metrics import RetrievalEvaluator
from src.data.loading.components.interfaces import (
    SequentialModelInputData,
    SequentialModuleLabelData,
)
from src.models.components.interfaces import SharedKeyAcrossPredictionsOutput
from src.models.components.network_blocks.embedding_aggregator import (
    EmbeddingAggregator,
)
from src.models.modules.base_module import BaseModule


class TransformerBaseModule(BaseModule):
    def __init__(
        self,
        huggingface_model: transformers.PreTrainedModel,
        postprocessor: torch.nn.Module,
        aggregator: EmbeddingAggregator,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss_function: torch.nn.Module,
        evaluator: RetrievalEvaluator,
        weight_tying: bool,
        compile: bool,
        training_loop_function: callable = None,
        feature_to_model_input_map: Dict[str, str] = {},
        decoder: torch.nn.Module = None,
    ) -> None:

        super().__init__(
            model=huggingface_model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=loss_function,
            evaluator=evaluator,
            training_loop_function=training_loop_function,
        )


        self.save_hyperparameters(
            logger=False,
            ignore=[
                "huggingface_model",
                "postprocessor",
                "aggregator",
                "decoder",
                "loss_function",
            ],
        )

        self.encoder = huggingface_model
        self.embedding_post_processor = postprocessor
        self.decoder = decoder
        self.aggregator = aggregator
        self.feature_to_model_input_map = feature_to_model_input_map

    def get_embedding_table(self):
        if self.hparams.weight_tying:
            return self.encoder.get_input_embeddings().weight
        else:
            return self.decoder.weight

    def training_step(
        self,
        batch: Tuple[Tuple[SequentialModelInputData, SequentialModuleLabelData]],
        batch_idx: int,
    ) -> torch.Tensor:


        batch = batch[0]

        model_input: SequentialModelInputData = batch[0]
        label_data: SequentialModuleLabelData = batch[1]

        model_output, loss = self.model_step(
            model_input=model_input, label_data=label_data
        )


        self.train_loss(loss)

        self.log(
            "train/loss",
            self.train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )


        if self.training_loop_function is not None:
            self.training_loop_function(self, loss)

        return loss

    def eval_step(
        self,
        batch: Tuple[SequentialModelInputData, SequentialModuleLabelData],
        loss_to_aggregate: BaseAggregator,
    ):


        model_input: SequentialModelInputData = batch[0]
        label_data: SequentialModuleLabelData = batch[1]

        model_output_before_aggregation, loss = self.model_step(
            model_input=model_input, label_data=label_data
        )

        model_output_after_aggregation = self.aggregator(
            model_output_before_aggregation, model_input.mask
        )


        self.evaluator(
            query_embeddings=model_output_after_aggregation,
            key_embeddings=self.get_embedding_table().to(
                model_output_after_aggregation.device
            ),

            labels=list(label_data.labels.values())[0].to(
                model_output_after_aggregation.device
            ),
        )
        loss_to_aggregate(loss)

    def predict_step(
        self,
        batch: Tuple[SequentialModelInputData, SequentialModuleLabelData],
        batch_idx: int,
    ):


        model_input: SequentialModelInputData = batch[0]
        model_output_before_aggregation, _ = self.model_step(model_input=model_input)

        model_output_after_aggregation = self.aggregator(
            model_output_before_aggregation, model_input.mask
        )

        model_output = SharedKeyAcrossPredictionsOutput(
            key=batch_idx,
            predictions=model_output_after_aggregation,
            key_name=self.prediction_key_name,
            prediction_name=self.prediction_name,
        )
        return model_output
