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

"""Tests for CUDA Memory Allocation, Fragmentation, and Interoperability.

Targets:
1. Memory fragmentation thresholds during dynamic graph sizing.
2. Compression efficiency in 8-bit optimizers.
3. DLPack zero-copy semantics (PyTorch <-> NVIDIA Warp).
4. Memory leak detection during recurrent passes.
"""

import gc
import sys
from pathlib import Path
from typing import Tuple

import bitsandbytes as bnb
import pytest
import torch
import warp as wp

current_file = Path(__file__).resolve()
repo_root = current_file.parents[2]
src_root = str(repo_root / "src")

if src_root not in sys.path:
    sys.path.insert(0, src_root)

if str(repo_root) not in sys.path:
    sys.path.insert(1, str(repo_root))

from vrl_framework.core.settings import MODEL_DEVICE  # noqa: E402
from vrl_framework.models.agents import ActorCriticModule  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires CUDA hardware for VRAM allocator profiling."
)


@pytest.fixture(scope="module", autouse=True)
def setup_allocators():
    """Initializes external backends and resets PyTorch allocator metrics."""
    wp.init()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    yield
    torch.cuda.empty_cache()


def get_cuda_memory_stats() -> Tuple[float, float, float]:
    """Returns allocation metrics (Allocated MB, Reserved MB, Fragmentation Ratio)."""
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)

    fragmentation = 0.0 if reserved == 0 else (reserved - allocated) / reserved
    return allocated, reserved, fragmentation


class TestCudaAndBnbAllocators:

    def test_dynamic_graph_fragmentation_tolerance(self):
        """Evaluates allocator fragmentation tolerance across varying sequence lengths."""
        model = ActorCriticModule(input_dim=128, num_actions=16).to(MODEL_DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        _, initial_reserved, _ = get_cuda_memory_stats()

        horizons = [1, 50, 2, 100, 5, 200, 1]

        for h in horizons:
            optimizer.zero_grad(set_to_none=True)

            states = torch.randn(32, h, 128, device=MODEL_DEVICE)
            out = model(states)
            loss = out.policy_logits.sum()
            loss.backward()
            optimizer.step()

            del states, out, loss

        gc.collect()
        torch.cuda.empty_cache()

        allocated, reserved, fragmentation_ratio = get_cuda_memory_stats()

        assert (
            fragmentation_ratio < 0.40
        ), f"CUDA Memory heavily fragmented. Ratio: {fragmentation_ratio:.2%}. Risk of OOM."

    def test_bnb_8bit_optimizer_memory_compression_and_leakage(self):
        """Validates momentum state compression bounds in AdamW8bit."""
        # Using a dense layer to isolate quantization metrics over metadata overhead
        huge_layer_fp32 = torch.nn.Linear(8192, 8192, device=MODEL_DEVICE)
        huge_layer_8bit = torch.nn.Linear(8192, 8192, device=MODEL_DEVICE)

        huge_layer_8bit.load_state_dict(huge_layer_fp32.state_dict())

        opt_32bit = torch.optim.AdamW(huge_layer_fp32.parameters(), lr=1e-4)
        opt_8bit = bnb.optim.AdamW8bit(huge_layer_8bit.parameters(), lr=1e-4)

        dummy_input = torch.randn(128, 8192, device=MODEL_DEVICE)

        def measure_optimizer_footprint(layer, optimizer) -> float:
            torch.cuda.empty_cache()
            mem_before = torch.cuda.memory_allocated()

            loss = layer(dummy_input).sum()
            loss.backward()
            optimizer.step()

            mem_after = torch.cuda.memory_allocated()
            return mem_after - mem_before

        fp32_footprint = measure_optimizer_footprint(huge_layer_fp32, opt_32bit)
        bnb8_footprint = measure_optimizer_footprint(huge_layer_8bit, opt_8bit)

        compression_ratio = bnb8_footprint / max(fp32_footprint, 1)

        # 55% upper bound accommodates block-wise metadata and fallback allocations
        assert compression_ratio < 0.55, f"Failed AdamW8bit compression SLA. Ratio: {compression_ratio:.2%}."

    def test_warp_pytorch_zero_copy_interoperability(self):
        """Verifies zero-copy memory mapping via the DLPack protocol."""

        @wp.kernel
        def mutation_kernel(arr: wp.array3d(dtype=wp.uint8)):
            i, j, k = wp.tid()
            arr[i, j, k] = wp.uint8(255)
            dim_x, dim_y, dim_z = 64, 64, 64
            torch_tensor = torch.zeros((dim_x, dim_y, dim_z), dtype=torch.uint8, device=MODEL_DEVICE)

            warp_array = wp.from_torch(torch_tensor, dtype=wp.uint8)
            assert (
                torch_tensor.data_ptr() == warp_array.ptr
            ), "Pointer mapping failed. Warp triggered implicit allocation."

            wp.launch(kernel=mutation_kernel, dim=(dim_x, dim_y, dim_z), inputs=[warp_array], device=MODEL_DEVICE)
            wp.synchronize(device=MODEL_DEVICE)

            assert torch_tensor[0, 0, 0].item() == 255, "Data coherency failure between Warp and PyTorch."

    def test_strict_epoch_memory_leakage(self):
        """Monitors VRAM for un-detached computational graphs during recurrent forward/backward loops."""
        model = ActorCriticModule(input_dim=64, num_actions=4).to(MODEL_DEVICE)
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)

        # CUDA context initialization and cuDNN benchmarking
        for _ in range(5):
            x = torch.randn(16, 1, 64, device=MODEL_DEVICE)
            loss = model(x).policy_logits.sum()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        gc.collect()
        torch.cuda.empty_cache()

        baseline_allocated = torch.cuda.memory_allocated()

        for _ in range(50):
            x = torch.randn(16, 1, 64, device=MODEL_DEVICE)

            with torch.autocast(device_type=MODEL_DEVICE, dtype=torch.float16):
                out = model(x)
                loss = out.policy_logits.var() + out.pessimistic_value.mean()

            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

            del x, out, loss

        final_allocated = torch.cuda.memory_allocated()

        # 1.0 MB threshold for static PyTorch graph bookkeeping
        delta_mb = (final_allocated - baseline_allocated) / (1024**2)

        assert (
            delta_mb < 1.0
        ), f"VRAM Leak: Allocated memory increased by {delta_mb:.2f} MB. Potential dangling references."
