import os
from typing import Any, Dict, Optional, Tuple

import hydra
import rootutils
import torch
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from src.utils import RankedLogger, extras
from src.utils.custom_hydra_resolvers import *
from src.utils.launcher_utils import pipeline_launcher
from src.utils.restart_job import LocalJobLauncher

command_line_logger = RankedLogger(__name__, rank_zero_only=True)

torch.set_float32_matmul_precision("medium")


def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:


    with pipeline_launcher(cfg) as pipeline_modules:

        if cfg.get("train"):
            command_line_logger.info("Starting training!")
            pipeline_modules.trainer.fit(
                model=pipeline_modules.model,
                datamodule=pipeline_modules.datamodule,
                ckpt_path=cfg.get("ckpt_path"),
            )
        train_metrics = pipeline_modules.trainer.callback_metrics

        if cfg.get("test"):
            command_line_logger.info("Starting testing!")
            ckpt_path = None


            checkpoint_callback = getattr(
                pipeline_modules.trainer, "checkpoint_callback", None
            )
            if checkpoint_callback:
                ckpt_path = getattr(checkpoint_callback, "best_model_path", None)
                if ckpt_path == "":
                    ckpt_path = None
            if not ckpt_path:
                command_line_logger.warning(
                    "Best checkpoint not found! Using current weights for testing..."
                )
            pipeline_modules.trainer.test(
                model=pipeline_modules.model,
                datamodule=pipeline_modules.datamodule,
                ckpt_path=ckpt_path,
            )
            command_line_logger.info(f"Best ckpt path: {ckpt_path}")

        test_metrics = pipeline_modules.trainer.callback_metrics


        metric_dict = {**train_metrics, **test_metrics}

        command_line_logger.info(f"Metrics: {metric_dict}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:


    extras(cfg)
    job_launcher = LocalJobLauncher(cfg=cfg)
    job_launcher.launch(function_to_run=train)


if __name__ == "__main__":
    main()
