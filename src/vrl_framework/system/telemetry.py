# Copyright 2026 Jacek Rejnhard.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import multiprocessing as mp
import os
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Optional

import torch
import wandb


class MetricsAdapter(ABC):

    @abstractmethod
    def log(self, data: dict, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def finish(self) -> None:
        pass


class NoOpMetrics(MetricsAdapter):

    def log(self, data: dict, *args: Any, **kwargs: Any) -> None:
        return None

    def finish(self) -> None:
        if wandb.run is not None:
            try:
                wandb.finish()
            except Exception:
                pass
        return None


class WandBMetrics(MetricsAdapter):

    def __init__(self) -> None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        wandb_dir = os.path.join(project_root, "outputs")
        os.makedirs(wandb_dir, exist_ok=True)
        os.environ["WANDB_DIR"] = wandb_dir

        if wandb.run is None:
            wandb.init(project="VRL-Framework", dir=wandb_dir)

    @staticmethod
    def _sanitize(val: Any) -> Any:
        if isinstance(val, torch.Tensor):
            if val.numel() == 1:
                return val.detach().item()
            return val.detach().cpu().numpy()
        elif hasattr(val, "item") and callable(val.item):
            try:
                return val.item()
            except Exception:
                return float(val)
        elif isinstance(val, (int, float, str, bool)):
            return val
        elif isinstance(val, dict):
            return {k: WandBMetrics._sanitize(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [WandBMetrics._sanitize(v) for v in val]
        return val

    def log(self, data: dict, *args: Any, **kwargs: Any) -> None:
        if wandb.run is not None and data:
            clean_data = {k: self._sanitize(v) for k, v in data.items()}
            try:
                from vrl_framework.trainer.ppo_engine import metrics_aggregator

                metrics_aggregator.log(clean_data)
            except Exception:
                pass

    def finish(self) -> None:
        if wandb.run is not None:
            try:
                wandb.finish()
            except Exception as e:
                import logging

                logging.getLogger("Telemetry").warning(
                    f"Ignored WandB shutdown error (likely missing/deleted log files): {e}"
                )


class CheckpointManager:

    def shutdown(self) -> None:
        import logging

        from vrl_framework.system.ipc_core import COMPUTE_WORKER, io_thread_pool

        logging.getLogger("VRL_Engine").critical("Shutdown protocol triggered. Escalating to ExperimentRunner.")
        io_thread_pool.shutdown(wait=False)
        COMPUTE_WORKER.shutdown(wait=False)
        raise RuntimeError("User requested system termination via CLI.")


runner = CheckpointManager()


class RuntimeState(Enum):
    CREATED = auto()
    STARTED = auto()
    STOPPING = auto()
    STOPPED = auto()


class RuntimeContext:

    def __init__(
        self, metrics_adapter: Any, io_worker: Any, compute_worker: Any, lmdb_bank: Any, sim_dir: str
    ) -> None:
        self.metrics = metrics_adapter
        self.io_worker = io_worker
        self.compute_worker = compute_worker
        self.lmdb_bank = lmdb_bank
        self.sim_dir = sim_dir


class ExperimentRunner:

    def __init__(self) -> None:
        self.state = RuntimeState.CREATED
        self.context: Optional[RuntimeContext] = None

    def attach_context(self, context: RuntimeContext) -> None:
        self.context = context

    def register_signals(self) -> None:
        import signal

        signal.signal(signal.SIGINT, self._handle_termination_signal)
        signal.signal(signal.SIGTERM, self._handle_termination_signal)

    def _handle_termination_signal(self, signum: Any, frame: Any) -> None:
        if self.state != RuntimeState.STOPPING and self.state != RuntimeState.STOPPED:
            self.shutdown_runtime()

    def start_runtime(self) -> None:
        self.state = RuntimeState.STARTED
        import logging

        logging.info("Runtime infrastructure transitioned to STARTED.")

    def shutdown_runtime(self) -> int:
        self.state = RuntimeState.STOPPING
        import logging

        logger = logging.getLogger("VRL_Engine")
        logger.info("Initiating runtime teardown sequence.")
        if self.context is not None:
            if hasattr(self.context, "lmdb_bank") and hasattr(self.context.lmdb_bank, "close"):
                self.context.lmdb_bank.close()
            if hasattr(self.context, "world_ref") and hasattr(self.context.world_ref, "close"):
                self.context.world_ref.close()
            if hasattr(self.context, "metrics") and hasattr(self.context.metrics, "finish"):
                self.context.metrics.finish()
            if hasattr(self.context, "io_worker") and hasattr(self.context.io_worker, "shutdown"):
                self.context.io_worker.is_running = False
                self.context.io_worker.submit(None)
                if hasattr(self.context.io_worker, "thread"):
                    self.context.io_worker.thread.join(timeout=10.0)

            if hasattr(self.context.compute_worker, "task_queues"):
                for q in self.context.compute_worker.task_queues:
                    try:
                        q.put(None)
                    except Exception as e:
                        import logging

                        logging.getLogger("TelemetryCleanup").warning(f"Failed to signal task queue: {e}")
                for p in self.context.compute_worker.worker_pool:
                    p.join(timeout=5.0)
                    if p.is_alive():
                        import logging

                        logging.getLogger("TelemetryCleanup").warning(f"Force terminating worker {p.name}")
                        p.terminate()
                        p.join(timeout=1.0)

            try:
                # Suppress socket descriptor exceptions during forced process termination
                pass
            except Exception as e:
                logger.error(f"Metrics socket closure fault: {e}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

        for p in mp.active_children():
            if p.is_alive():
                p.terminate()
                p.join(timeout=2.0)

        self.state = RuntimeState.STOPPED
        logger.info("Runtime terminated.")
        return 0
