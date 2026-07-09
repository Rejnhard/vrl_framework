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

"""
Unit tests for vectorized world dynamics and Warp-based physics kernels.

Validates memory safety, tensor shape invariants, and correct mapping of
object properties to contiguous memory buffers during JIT kernel execution.
"""

import sys
from pathlib import Path

import pytest
import torch
import warp as wp

current_file = Path(__file__).resolve()
repo_root = current_file.parents[3]
src_root = str(repo_root / "src")

if src_root not in sys.path:
    sys.path.insert(0, src_root)
if str(repo_root) not in sys.path:
    sys.path.insert(1, str(repo_root))

wp.init()

from vrl_framework.core.settings import MODEL_DEVICE  # noqa: E402
from vrl_framework.environment.world_dynamics import VectorizedPopulation, step_sparse_substrate_kernel  # noqa: E402


class TestWarpPhysicsKernels:

    def test_sparse_substrate_transitions(self) -> None:
        """Verifies state transition rules and bounds checking within the substrate kernel."""
        dim = 4

        pt_state = torch.zeros((dim, dim, dim), dtype=torch.uint8, device=MODEL_DEVICE)

        energy_base = torch.zeros((dim, dim, dim), dtype=torch.uint8, device=MODEL_DEVICE)
        energy_base[0, 0, 0] = 150
        energy_base[0, 0, 1] = 50
        energy_base[0, 0, 2] = 5
        energy_base[0, 0, 3] = 0

        pt_energy = energy_base
        pt_mask = torch.ones((dim, dim, dim), dtype=torch.uint8, device=MODEL_DEVICE)
        pt_mask[:, :, 3] = 0

        wp_state = wp.from_torch(pt_state)
        wp_energy = wp.from_torch(pt_energy)
        wp_mask = wp.from_torch(pt_mask)

        wp.launch(
            kernel=step_sparse_substrate_kernel,
            dim=(dim, dim, dim),
            inputs=[wp_state, wp_energy, wp_mask, dim, dim, dim],
        )
        wp.synchronize()

        out_state = wp_state.numpy()
        out_energy = wp_energy.numpy()

        assert out_state[0, 0, 0] == 1, "State activation failed for energy > 128."
        assert out_energy[0, 0, 0] == 140, "Energy deduction failed during state activation."
        assert out_energy[0, 0, 1] == 49, "Energy deduction failed for standard state."
        assert out_energy[0, 0, 3] == 0, "Kernel modified memory outside the active mask."


class TestVectorizedPopulation:

    @pytest.fixture
    def mock_population(self) -> VectorizedPopulation:
        return VectorizedPopulation(initial_agents=2, world_dim=(10, 10, 10, 10), max_agents=4)

    def test_tensor_allocation_capacity(self, mock_population: VectorizedPopulation) -> None:
        """Verifies contiguous memory allocation matches expected population bounds."""
        assert mock_population.positions.shape == (4, 4)
        assert mock_population.active_mask.sum().item() == 2

    def test_entity_tensor_synchronization(self, mock_population: VectorizedPopulation) -> None:
        """Verifies structural mapping between individual objects and flat tensors."""

        class DummyEntity:
            def __init__(self, idx, pos):
                self.idx = idx
                self._pos_fallback = pos
                self.needs_epigenetic_inheritance = False

        entities = [
            DummyEntity(None, [1.0, 1.0, 1.0, 0.0]),
            DummyEntity(None, [2.0, 2.0, 2.0, 0.0]),
            DummyEntity(None, [3.0, 3.0, 3.0, 0.0]),
        ]

        mock_population.sync_with_entities_list(entities)

        assert mock_population.active_mask.sum().item() == 3
        assert mock_population.positions[2, 0].item() == 3.0
        assert mock_population.active_mask[3].item() is False
