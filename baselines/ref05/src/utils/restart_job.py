import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
from typing import Any, Callable, TypeVar, Union

from lightning import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig

from src.utils.pylogger import RankedLogger
from src.utils.restart_job_utils import (
    JobCheckpointMetadata,
    RestartMetadata,
    _is_process_running,
    clean_up_resources,
    get_attribute_from_metadata_file,
    load_metadata_from_local_or_remote,
    save_metadata_to_local_or_remote,
)

command_line_logger = RankedLogger(__name__, rank_zero_only=True)
F = TypeVar("F", bound=Callable[..., Any])


class RestartAndLoadCheckpointCallback(Callback):


    def __init__(self, metadata_dir: Union[str, None] = None):


        super().__init__()
        if metadata_dir is None:
            metadata_dir = os.environ.get("METADATA_DIR", "")
        else:
            if metadata_dir != os.environ.get("METADATA_DIR", ""):
                os.environ["METADATA_DIR"] = str(metadata_dir)
        self.metadata_dir = metadata_dir
        os.makedirs(self.metadata_dir, exist_ok=True)

        self.metadata_path = os.path.join(self.metadata_dir, "restart_metadata.json")
        self._load_metadata()
        self._save_metadata()

    def _load_metadata(self) -> None:

        self.metadata: JobCheckpointMetadata = load_metadata_from_local_or_remote(
            self.metadata_path
        )

    @rank_zero_only
    def _save_metadata(self) -> None:

        save_metadata_to_local_or_remote(self.metadata, self.metadata_path)

    @rank_zero_only
    def on_train_start(self, trainer, pl_module) -> None:

        self.metadata.world_size = trainer.world_size
        self.metadata.node_rank = getattr(trainer, "node_rank", 0)
        self.metadata.master_addr = os.environ.get("MASTER_ADDR", "")


        master_port = os.environ.get("MASTER_PORT", "")
        if master_port and master_port not in self.metadata.used_ports:
            self.metadata.used_ports.append(master_port)

        self._save_metadata()

    def on_exception(self, trainer, pl_module, exception) -> None:

        command_line_logger.error(f"Exception caught: {exception}")
        command_line_logger.error(f"Stack trace: {traceback.format_exc()}")

        command_line_logger.info("Cleaning up and handling restart logic")

        self.metadata.current_run = get_attribute_from_metadata_file(
            self.metadata_path, "current_run"
        )

        self.metadata.restarts.append(
            RestartMetadata(
                time=datetime.now().isoformat(),
                exception=str(exception),
                run_number=self.metadata.current_run,
            ).to_dict()
        )
        self.metadata.current_run += 1
        command_line_logger.info(f"Restarting job. Current metadata: {self.metadata}")
        self._save_metadata()


        self._cleanup_resources(trainer, exception)

    def _cleanup_resources(self, trainer: Trainer, exception: Exception) -> None:
        clean_up_resources(trainer, exception)

        command_line_logger.info(f"Resources cleaned up.")
        os._exit(1)


class BaseJobLauncher:


    def __init__(self, cfg: DictConfig, max_retries: int = 3, retry_delay: int = 5):
        self.cfg = cfg
        self.logger = RankedLogger(__name__, rank_zero_only=True)
        self.max_retries = cfg.get("max_retries", max_retries)
        self.retry_delay = cfg.get("retry_delay", retry_delay)
        self.run_count = 0
        self.metadata_dir = None
        self.process = None

    def setup_metadata_dir(self):


        if (
            self.cfg.get("callbacks")
            and self.cfg.callbacks.get("restart_job", None)
            and self.cfg.callbacks.restart_job.metadata_dir
        ):
            self.metadata_dir = self.cfg.callbacks.restart_job.metadata_dir
        elif self.cfg.paths.get("metadata_dir"):
            self.metadata_dir = self.cfg.paths.metadata_dir
        else:
            self.logger.info("Creating temporary metadata dir.")
            self.metadata_dir = tempfile.mkdtemp(prefix=f"job_launcher_{os.getpid()}_")

        os.environ["METADATA_DIR"] = self.metadata_dir
        self.cfg.paths.metadata_dir = self.metadata_dir
        self.metadata_path = os.path.join(self.metadata_dir, "restart_metadata.json")
        self.logger.info(f"Using metadata dir: {self.metadata_dir}")

    def prepare_command(self):

        original_argv = sys.argv
        cmd = [sys.executable] + original_argv + ["++should_skip_retry=True"]
        return cmd

    def execute_job(self) -> bool:

        self.setup_metadata_dir()
        cmd = self.prepare_command()

        while self.run_count <= self.max_retries:
            self.logger.info(
                f"Starting job (attempt {self.run_count + 1}/{self.max_retries + 1}): {' '.join(cmd)}"
            )

            success = self.run_single_attempt(cmd)

            if success:
                self.logger.info("Job finished successfully.")
                return True

            self.run_count += 1
            metadata_run = get_attribute_from_metadata_file(
                self.metadata_path, "current_run"
            )
            if metadata_run != self.run_count:
                self.logger.error(
                    f"Metadata run count {metadata_run} does not match current run count {self.run_count}."
                    f"This can happen if the exception does not trigger the callback on_exception. Cleaning up and"
                    f"updating metadata before retrying."
                )

                clean_up_resources()
                metadata = load_metadata_from_local_or_remote(self.metadata_path)
                metadata.current_run = self.run_count
                save_metadata_to_local_or_remote(metadata, self.metadata_path)
            if self.run_count <= self.max_retries:
                self.logger.info(
                    f"Retrying in {self.retry_delay} seconds. Attempt {self.run_count + 1}/{self.max_retries + 1}"
                )
                time.sleep(self.retry_delay)
            else:
                self.logger.error(f"Job failed after {self.max_retries} retries.")
                return False

        return False

    def _clean_process(self):

        if self.process and self.process.poll() is None:
            self.process.send_signal(signal.SIGKILL)

    def run_single_attempt(self, cmd) -> bool:

        self.process = subprocess.Popen(cmd, env=os.environ)

        try:
            while _is_process_running(self.process):
                time.sleep(5)

            if self.process.returncode == 0:
                return True

            self._clean_process()
            self.logger.error(f"Job failed with return code {self.process.returncode}")
            self.process = None
            return False

        except Exception as e:
            self.logger.error(f"Error running job: {e} {traceback.format_exc()}")
            self._clean_process()
            return False

    def cleanup(self):

        return_code = self.process.returncode if self.process else None
        self._clean_process()
        sys.exit(return_code if return_code is not None else 1)

    def launch(self, function_to_run: callable) -> bool:
        raise NotImplementedError(
            "Method not implemented, should be implemented in child classes."
        )


class LocalJobLauncher(BaseJobLauncher):


    def __init__(self, cfg: DictConfig, max_retries: int = 3, retry_delay: int = 5):
        super().__init__(cfg, max_retries, retry_delay)
        self.should_skip_retry = cfg.get("should_skip_retry", False)
        if (
            not self.should_skip_retry
            and cfg.get("trainer")
            and cfg.trainer.num_nodes > 1
        ):
            self.logger.warning(
                "Retry logic is not supported for multi-node training, setting should_skip_retry to True."
            )
            self.should_skip_retry = True

        if self.should_skip_retry:
            self.max_retries = 0

    def launch(self, function_to_run: callable) -> bool:

        if self.should_skip_retry:
            self.logger.info("should_skip_retry set to True. Skipping retry logic.")
            return function_to_run(cfg=self.cfg)
        try:
            return self.execute_job()
        except Exception as e:
            self.logger.error(
                f"Error launching job. Exception {e} {traceback.format_exc()}"
            )
            self.cleanup()
            return False
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt caught. Exiting.")
            self._clean_process()
            return False
        finally:
            self.cleanup()
