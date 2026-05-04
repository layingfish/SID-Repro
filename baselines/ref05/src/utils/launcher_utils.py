from contextlib import contextmanager
from dataclasses import dataclass
from typing import List

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils import (
    RankedLogger,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
)
from src.utils.file_utils import (
    get_last_modified_file,
    has_no_extension,
    list_subfolders,
)
from src.utils.logging_utils import finalize_loggers
from src.utils.restart_job_utils import get_attribute_from_metadata_file
from src.utils.utils import has_class_object_inside_list

command_line_logger = RankedLogger(__name__, rank_zero_only=True)


@dataclass
class PipelineModules:
    cfg: DictConfig
    datamodule: LightningDataModule
    model: LightningModule

    callbacks: List[Callback]
    loggers: List[Logger]
    trainer: Trainer


def update_cfg_with_most_recent_checkpoint_path(cfg: DictConfig) -> str:


    ckpt_path = cfg.get("ckpt_path", None)

    if (
        ckpt_path is not None
        and has_no_extension(ckpt_path)
        and cfg.get("should_retrieve_latest_ckpt_path", False)
    ):


        checkpoint_folders = list_subfolders(ckpt_path)
        if len(checkpoint_folders) > 0:

            checkpoint_folders.sort(reverse=True)

            latest_ckpt_folder = checkpoint_folders[0]
            last_modified = get_last_modified_file(
                folder_path=latest_ckpt_folder, suffix="*.ckpt"
            )
            if len(last_modified) > 0:
                ckpt_path = last_modified
                command_line_logger.info(
                    f"Found most recent checkpoint path: {ckpt_path}. Starting job from this checkpoint."
                )


    if (
        cfg.get("callbacks")
        and cfg.callbacks.get("model_checkpoint")
        and cfg.callbacks.get("restart_job")
        and get_attribute_from_metadata_file(
            f"{cfg.callbacks.restart_job.metadata_dir}/restart_metadata.json",
            "current_run",
        )
        > 0
    ):
        checkpoint_folder = cfg.callbacks.model_checkpoint.dirpath

        last_modified = get_last_modified_file(
            folder_path=checkpoint_folder, suffix="*.ckpt"
        )
        if len(last_modified) > 0:
            ckpt_path = last_modified
            command_line_logger.info(
                f"Found most recent checkpoint path: {ckpt_path}. Starting job from this checkpoint."
            )

    cfg.ckpt_path = ckpt_path
    return cfg


def initialize_pipeline_modules(
    cfg: DictConfig,
) -> PipelineModules:


    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    command_line_logger.info(
        f"Instantiating datamodule <{cfg.data_loading.datamodule._target_}>"
    )
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.data_loading.datamodule
    )

    command_line_logger.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    command_line_logger.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    command_line_logger.info("Instantiating loggers...")
    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"))

    command_line_logger.info(f"Instantiating trainer <{cfg.trainer._target_}>")

    cfg = update_cfg_with_most_recent_checkpoint_path(cfg)

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,


        enable_checkpointing=cfg.trainer.get(
            "enable_checkpointing",
            has_class_object_inside_list(callbacks, ModelCheckpoint),
        ),
        enable_model_summary=cfg.trainer.get(
            "enable_model_summary",
            has_class_object_inside_list(callbacks, ModelSummary),
        ),
        logger=loggers,
    )

    pipeline_modules = PipelineModules(
        cfg=cfg,
        datamodule=datamodule,
        model=model,
        callbacks=callbacks,
        loggers=loggers,
        trainer=trainer,
    )

    return pipeline_modules


@contextmanager
def pipeline_launcher(cfg: DictConfig):


    try:
        pipeline_modules: PipelineModules = initialize_pipeline_modules(cfg)

        if len(pipeline_modules.loggers) > 0:
            command_line_logger.info("Logging hyperparameters!")
            log_hyperparameters(cfg, pipeline_modules.model, pipeline_modules.trainer)
        yield pipeline_modules
    except Exception as ex:
        raise ex
    finally:

        finalize_loggers(pipeline_modules.trainer)
