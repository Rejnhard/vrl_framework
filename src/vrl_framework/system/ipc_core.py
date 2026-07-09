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

import atexit
import concurrent.futures
import logging
import multiprocessing as mp
import os
import platform
import queue
import signal
import sys
import threading
import time
import traceback
import uuid
from multiprocessing import shared_memory
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch

from vrl_framework.core.contracts import ComputeTaskContract


class SharedMemoryReceiver:
    """Zero-copy IPC tensor reader."""

    def __init__(self, buffer_name: str = "multimodal_sensor_stream", buffer_size: int = 134217728):
        self.buffer_name = buffer_name
        self.buffer_size = buffer_size
        self.shm: Optional[shared_memory.SharedMemory] = None
        self.connect_to_stream()

    def connect_to_stream(self) -> None:
        """Connects to shared memory and registers atexit cleanup."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.shm = shared_memory.SharedMemory(name=self.buffer_name)
                break
            except FileNotFoundError:
                if attempt == max_retries - 1:
                    logging.warning(f"Shared memory {self.buffer_name} not found. Operating in isolated vacuum mode.")
                    return
                time.sleep(0.1)

        if getattr(self, "shm", None) is not None:

            def cleanup_shm():
                # Release shared memory resources gracefully.
                if self.shm is not None:
                    try:
                        self.shm.close()
                        self.shm.unlink()
                    except Exception as e:
                        logging.error(f"SHM Teardown Error [{self.buffer_name}]: {type(e).__name__} - {e}")

            atexit.register(cleanup_shm)

    def read_latest_frame(
        self, shape: Tuple[int, ...], dtype: np.dtype, device: torch.device = torch.device("cuda")
    ) -> Optional[torch.Tensor]:
        """Reads tensor streams from shared IPC memory into pinned tensors."""
        if getattr(self, "_zero_cache", None) is None:
            self._zero_cache: dict[Tuple[Tuple[int, ...], torch.device], torch.Tensor] = {}
            self._pinned_cache: dict[Tuple[Tuple[int, ...], torch.dtype], torch.Tensor] = {}
            self._DTYPE_MAP = {
                np.dtype("float16"): torch.float16,
                np.dtype("uint8"): torch.uint8,
                np.dtype("int8"): torch.int8,
                np.dtype("int32"): torch.int32,
            }

        zero_key = (shape, device)
        if zero_key not in self._zero_cache:
            self._zero_cache[zero_key] = torch.zeros(shape, dtype=torch.float32, device=device)

        if self.shm is None:
            self.connect_to_stream()
            if self.shm is None:
                return self._zero_cache[zero_key]

        expected_bytes = int(np.prod(shape)) * dtype.itemsize
        if expected_bytes > self.buffer_size:
            raise ValueError("Requested tensor shape exceeds allocated IPC buffer limits.")

        try:
            torch_dtype = self._DTYPE_MAP.get(np.dtype(dtype), torch.float32)

            buf = self.shm.buf
            if buf is None:
                return self._zero_cache[zero_key]

            raw_buffer = memoryview(buf[:expected_bytes])
            if len(raw_buffer) < expected_bytes:
                return self._zero_cache[zero_key]
            tensor_mapped = torch.frombuffer(raw_buffer, dtype=torch_dtype).view(shape)

            # Pin staging tensor for zero-copy Host-to-Device (H2D) transfer
            cache_key = (shape, torch_dtype)
            if cache_key not in self._pinned_cache:
                self._pinned_cache[cache_key] = torch.empty(shape, dtype=torch_dtype, pin_memory=True)

            pinned_staging_buffer = self._pinned_cache[cache_key]
            pinned_staging_buffer.copy_(tensor_mapped)
            return pinned_staging_buffer.to(device, non_blocking=True)
        except Exception as e:
            logging.error(f"Zero-Copy read fault: {e}")
            return self._zero_cache[zero_key]


class ThreadPoolManager:
    """Asynchronous I/O thread pool."""

    def __init__(self, max_workers: int = 4):
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, task: Callable, *args: Any, **kwargs: Any) -> concurrent.futures.Future:
        return self._executor.submit(task, *args, **kwargs)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)


io_thread_pool = ThreadPoolManager(max_workers=4)
COMPUTE_WORKER = ThreadPoolManager(max_workers=os.cpu_count() or 4)


class DiskIOWorkerQueue:
    """Background disk I/O queue."""

    def __init__(self, maxsize: int = 1000):
        self.queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=maxsize)
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.is_running = True
        self.error_count = 0
        self.thread.start()

    def _apply_thread_affinity(self) -> None:
        # Pin I/O to lower logical cores
        if platform.system() == "Windows":
            try:
                import psutil

                logical_cores = psutil.cpu_count(logical=True)
                if logical_cores is not None and logical_cores >= 4:
                    target_cores = [logical_cores - 1, logical_cores - 2]
                    p = psutil.Process(os.getpid())
                    p.cpu_affinity(target_cores)
            except Exception as e:
                import logging

                logging.error(f"Affinity binding fault: {e}")

    def _worker_loop(self):
        self._apply_thread_affinity()
        while True:
            try:
                task_data = self.queue.get()
                if task_data is None or task_data.get("task") is None:
                    self.queue.task_done()
                    break

                task_id = task_data["task_id"]
                task_fn = task_data["task"]
                args = task_data["args"]
                kwargs = task_data["kwargs"]

                try:
                    task_fn(*args, **kwargs)
                except Exception as ex:
                    self.error_count += 1
                    import logging

                    logging.error(f"DiskIO Fault on Task {task_id}: {ex}")
            except Exception as e:
                import logging

                logging.error(f"DiskIOWorkerQueue core fault: {e}")
            finally:
                self.queue.task_done()

    def submit(self, task, *args, **kwargs):
        if self.is_running:
            task_payload = {
                "task_id": uuid.uuid4().hex[:8],
                "task": task,
                "args": args,
                "kwargs": kwargs,
                "status": "pending",
            }
            self.queue.put(task_payload)


class MockSyncWorker:
    """Synchronous executor for debugging."""

    def submit(self, task, *args, **kwargs):
        if "target_tensor" in kwargs:
            target_tensor = kwargs.pop("target_tensor")
            kwargs["shared_tensor"] = target_tensor

        # Resolve string-based IPC identifiers to functions
        if isinstance(task, str):
            task = getattr(sys.modules[__name__], task, None)

        if callable(task):
            task(*args, **kwargs)


class ComputeWorkerQueue:
    """Asynchronous IPC tensor processing queue."""

    def __init__(self, max_workers: int = 2):
        self.worker_pool = []
        self.task_queues = []
        self.result_queues = []

        try:
            if hasattr(mp, "set_sharing_strategy"):
                mp.set_sharing_strategy("file_system")
        except Exception:
            pass

        for i in range(max_workers):
            task_q: mp.Queue[Any] = mp.Queue()
            res_q: mp.Queue[dict[str, Any]] = mp.Queue()
            self.task_queues.append(task_q)
            self.result_queues.append(res_q)
            p = mp.Process(target=self._dynamic_mmap_worker_loop, args=(task_q, res_q), daemon=True)
            p.start()
            self.worker_pool.append(p)

        self.rr_index = 0

    @staticmethod
    def _dynamic_mmap_worker_loop(task_queue: mp.Queue, result_queue: mp.Queue) -> None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        while True:
            try:
                contract = task_queue.get()
                if contract is None:
                    break

                if not isinstance(contract, ComputeTaskContract):
                    raise ValueError(f"Invalid contract format received in compute worker: {type(contract)}")

                if (
                    contract.shared_tensor.shape != contract.input_shape
                    or contract.shared_tensor.dtype != contract.input_dtype
                ):
                    raise ValueError(f"Shape/dtype mismatch in IPC memory block for request {contract.request_id}.")

                task_fn = getattr(sys.modules[__name__], contract.task_name, None)
                if task_fn is not None:
                    result_tensor = task_fn(contract.shared_tensor, **contract.kwargs)
                    if result_tensor is not None:
                        if (
                            result_tensor.shape != contract.expected_out_shape
                            or result_tensor.dtype != contract.expected_out_dtype
                        ):
                            raise ValueError("Output constraint violation generated by compute task.")
                    result_queue.put({"request_id": contract.request_id, "status": "success"})
                else:
                    result_queue.put({"request_id": contract.request_id, "status": "missing_function"})
            except Exception as e:
                logging.error(f"Compute worker execution fault: {e}")
                logging.error(traceback.format_exc())
                try:
                    req_id = contract.request_id
                except (NameError, AttributeError):
                    req_id = "UNKNOWN"
                result_queue.put({"request_id": req_id, "status": "error", "msg": str(e)})

    def submit(
        self,
        task_name: str,
        target_tensor: torch.Tensor,
        expected_shape: Tuple[int, ...],
        expected_dtype: torch.dtype,
        **kwargs,
    ) -> str:
        if not isinstance(target_tensor, torch.Tensor):
            raise TypeError("IPC submission requires native PyTorch tensors.")

        # Detach to prevent serializing the entire autograd graph across IPC
        shared_tensor = target_tensor.detach().cpu().contiguous().share_memory_()
        request_id = uuid.uuid4().hex[:8]

        contract = ComputeTaskContract(
            task_name=task_name,
            request_id=request_id,
            input_shape=target_tensor.shape,
            input_dtype=target_tensor.dtype,
            expected_out_shape=expected_shape,
            expected_out_dtype=expected_dtype,
            shared_tensor=shared_tensor,
            kwargs=kwargs,
        )

        self.task_queues[self.rr_index].put(contract)
        self.rr_index = (self.rr_index + 1) % len(self.worker_pool)
        return request_id
