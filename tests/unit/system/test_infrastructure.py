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

"""Unit tests for Inter-Process Communication (IPC) and distributed telemetry logic."""

import multiprocessing as mp
import os
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

current_file = Path(__file__).resolve()
repo_root = current_file.parents[3]
src_root = str(repo_root / "src")

if src_root not in sys.path:
    sys.path.insert(0, src_root)
if str(repo_root) not in sys.path:
    sys.path.insert(1, str(repo_root))

from vrl_framework.core.contracts import ComputeTaskContract  # noqa: E402
from vrl_framework.system.telemetry import WandBMetrics  # noqa: E402


class SharedMemoryReceiver:
    def __init__(self, buffer_name: str, buffer_size: int):
        self.shm = shared_memory.SharedMemory(name=buffer_name)
        self.buffer_size = buffer_size

    def close(self):
        if self.shm is not None:
            self.shm.close()


# Must be defined at module-level to support Windows multiprocessing 'spawn' pickling protocol.
def _blocking_worker():
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


class TestInterProcessCommunication:

    def test_shared_memory_allocation_and_unlink_safety(self) -> None:
        """Validates cross-boundary memory mapping and strict POSIX /dev/shm garbage collection."""
        buffer_name = "test_shm_stream_01"
        tensor_size = 1024 * 1024  # 1 MB test block
        shm_block = None
        receiver = None

        try:
            shm_block = shared_memory.SharedMemory(name=buffer_name, create=True, size=tensor_size)

            test_array = np.ndarray((256, 1024), dtype=np.float32, buffer=shm_block.buf)
            test_array.fill(42.0)

            receiver = SharedMemoryReceiver(buffer_name=buffer_name, buffer_size=tensor_size)
            assert receiver.shm is not None, "Receiver failed to map existing SHM block."

            read_array = np.ndarray((256, 1024), dtype=np.float32, buffer=receiver.shm.buf)
            assert read_array[0, 0] == 42.0, "IPC mapped data mismatch."

        finally:
            if receiver is not None and receiver.shm is not None:
                receiver.shm.close()
            if shm_block is not None:
                shm_block.close()
                shm_block.unlink()

    def test_compute_task_contract_serialization(self) -> None:
        """Verifies ComputeTaskContract enforces graph detachment prior to IPC transmission."""
        target_tensor = torch.randn(4, 256, requires_grad=True)
        computation_result = target_tensor * 2.0

        # Tensors must be contiguous and detached before mapping to shared memory limits.
        safe_tensor = computation_result.detach().cpu().contiguous().share_memory_()

        contract = ComputeTaskContract(
            task_name="latent_projection",
            request_id="req_999",
            input_shape=safe_tensor.shape,
            input_dtype=safe_tensor.dtype,
            expected_out_shape=(4, 64),
            expected_out_dtype=torch.float32,
            shared_tensor=safe_tensor,
            kwargs={},
        )

        assert (
            contract.shared_tensor.requires_grad is False
        ), "Contract failed to reject tensor with requires_grad=True."
        assert contract.shared_tensor.is_shared(), "Tensor lacks shared memory allocation flag."


class TestTelemetryAndTeardown:

    def test_wandb_metrics_tensor_sanitization(self) -> None:
        """Tests coercion of arbitrary torch.Tensor objects into JSON-serializable primitives."""
        adapter = WandBMetrics()

        scalar_tensor = torch.tensor(42.5, requires_grad=True, device="cpu")
        clean_scalar = adapter._sanitize(scalar_tensor)
        assert isinstance(clean_scalar, float), "Expected float coercion for 0D tensor."
        assert clean_scalar == 42.5

        multi_tensor = torch.randn(2, 2, requires_grad=True)
        clean_multi = adapter._sanitize(multi_tensor)
        assert isinstance(clean_multi, np.ndarray), "Expected ndarray coercion for ND tensor."

        clean_str = adapter._sanitize("epoch_event")
        assert clean_str == "epoch_event"

    @patch("wandb.run", new_callable=MagicMock)
    @patch("wandb.log")
    def test_metrics_logging_resilience(self, mock_wandb_log, mock_wandb_run) -> None:
        """Tests telemetry payload sanitization with nested iterables."""
        adapter = WandBMetrics()

        payload = {"loss": torch.tensor(0.5), "generation": 1000, "complex_struct": [1, 2, 3]}

        try:
            adapter.log(payload)
        except Exception as e:
            pytest.fail(f"WandBMetrics.log crashed during payload sanitization: {e}")

    def test_worker_escalation_teardown(self) -> None:
        """Simulates SIGTERM to SIGKILL escalation sequence on an unresponsive process."""
        worker_process = mp.Process(target=_blocking_worker)
        worker_process.start()

        assert worker_process.is_alive() is True, "Failed to spawn child worker process."

        if worker_process.is_alive():
            worker_process.terminate()
            worker_process.join(timeout=2.0)

            if worker_process.is_alive():
                if os.name == "nt":
                    import subprocess

                    subprocess.call(["taskkill", "/F", "/T", "/PID", str(worker_process.pid)])
                else:
                    import signal

                    os.kill(worker_process.pid, signal.SIGKILL)
                worker_process.join(timeout=1.0)

        assert worker_process.is_alive() is False, "Process failed to terminate after escalation."
