import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar

import psutil
from lightning import Trainer
from omegaconf import DictConfig

from src.utils.file_utils import (
    file_exists_local_or_remote,
    load_json,
    open_local_or_remote,
)
from src.utils.pylogger import RankedLogger

command_line_logger = RankedLogger(__name__, rank_zero_only=True)
F = TypeVar("F", bound=Callable[..., Any])

import os
import sys
from datetime import datetime
from typing import Any, Callable, TypeVar

import torch
import torch.distributed as dist
from lightning import Trainer
from lightning.pytorch.strategies.launchers import _SubprocessScriptLauncher
from lightning.pytorch.trainer.connectors.signal_connector import _get_sigkill_signal
from omegaconf import DictConfig

from src.utils.pylogger import RankedLogger


@dataclass
class JobCheckpointMetadata:


    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    restarts: List[Dict[str, Any]] = field(default_factory=list)
    current_run: int = 0
    used_ports: List[str] = field(default_factory=list)
    world_size: int = 0
    node_rank: int = 0
    master_addr: str = ""
    original_args: List[str] = field(default_factory=lambda: sys.argv)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time,
            "restarts": self.restarts,
            "current_run": self.current_run,
            "used_ports": self.used_ports,
            "world_size": self.world_size,
            "node_rank": self.node_rank,
            "master_addr": self.master_addr,
            "original_args": self.original_args,
        }


@dataclass
class RestartMetadata:


    time: str
    exception: str
    run_number: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time": self.time,
            "exception": self.exception,
            "run_number": self.run_number,
        }


def load_metadata_from_local_or_remote(metadata_path: str) -> JobCheckpointMetadata:


    command_line_logger.info(f"Trying to load metadata from {metadata_path}")
    if file_exists_local_or_remote(metadata_path):
        metadata_dict = load_json(metadata_path)
        command_line_logger.info(f"Metadata loaded successfully from {metadata_path}")
        return JobCheckpointMetadata(**metadata_dict)
    else:
        command_line_logger.warning(
            f"Metadata file not found at {metadata_path}. Creating empty metadata."
        )
        return JobCheckpointMetadata()


def save_metadata_to_local_or_remote(
    metadata: JobCheckpointMetadata, metadata_path: str
) -> None:


    command_line_logger.info(
        f"Saving metadata to {metadata_path}. {metadata.to_dict()}"
    )


    json_content = json.dumps(metadata.to_dict(), indent=2)


    with open_local_or_remote(metadata_path, "w") as f:
        f.write(json_content)


def get_attribute_from_metadata_file(metadata_path: str, attribute: str) -> Any:


    metadata = load_metadata_from_local_or_remote(metadata_path)
    attribute_value = getattr(metadata, attribute, None)
    command_line_logger.info(
        f"Retrieved {attribute}: {attribute_value} from metadata {metadata_path}"
    )
    return attribute_value


def _is_process_running(proc: psutil.Process) -> bool:


    proc.poll()
    return proc.returncode is None


def clean_up_resources(
    trainer: Optional[Trainer] = None, exception: Optional[Exception] = None
) -> None:

    if dist.is_initialized():
        command_line_logger.info("Cleaning up distributed process group")
        dist.destroy_process_group()

    if torch.cuda.is_available():
        command_line_logger.info("Clearing CUDA cache")
        torch.cuda.empty_cache()

    if trainer is not None:
        command_line_logger.info("Tearing down trainer")
        trainer.strategy.on_exception(exception)
        launcher = trainer.strategy.launcher if trainer.strategy is not None else None
        trainer._teardown()
        if isinstance(launcher, _SubprocessScriptLauncher):
            launcher.kill(_get_sigkill_signal())
