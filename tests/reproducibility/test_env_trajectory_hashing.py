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

"""Tests determinism of the environment physics engine.

Validates that identical PRNG seeds produce exact trajectory matches.
"""

import hashlib
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
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

from vrl_framework.core.settings import MODEL_DEVICE, WORLD_DIM  # noqa: E402
from vrl_framework.environment.world_dynamics import VectorizedPopulation  # noqa: E402


def enforce_deterministic_state(seed: int = 42) -> None:
    """Configures environment and framework seeds for reproducible execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def compute_trajectory_signature(trajectory_tensors: List[torch.Tensor], precision: int = 4) -> str:
    """Generates a SHA-256 signature from quantized sequence of state tensors."""
    hasher = hashlib.sha256()

    for tensor in trajectory_tensors:
        cpu_tensor = tensor.detach().cpu().float()
        quantized_array = np.round(cpu_tensor.numpy(), decimals=precision)
        hasher.update(quantized_array.tobytes())

    return hasher.hexdigest()


@pytest.fixture(scope="module", autouse=True)
def init_warp_backend():
    """Initializes Warp physics backend."""
    wp.init()
    yield


class TestEnvironmentTrajectoryHashing:
    """Validates bit-exact reproducibility of VectorizedPopulation rollouts."""

    def simulate_rollout(self, seed: int, steps: int = 5, agents: int = 16) -> Dict[str, str]:
        """Executes a fixed-length environment rollout and hashes the trajectory state."""
        enforce_deterministic_state(seed)

        dim_topology = WORLD_DIM if len(WORLD_DIM) == 4 else (100, 100, 100, 10)
        population = VectorizedPopulation(initial_agents=agents, world_dim=dim_topology, max_agents=agents)
        population.to(MODEL_DEVICE)

        trajectory_pos = []
        trajectory_energy = []

        for step in range(steps):
            actions = torch.randint(0, 16, (agents,), device=MODEL_DEVICE)

            next_states, rewards, inactive_mask = population.step(actions)

            trajectory_pos.append(population.positions.clone())
            trajectory_energy.append(population.energies.clone())

        return {
            "positions_hash": compute_trajectory_signature(trajectory_pos),
            "energy_hash": compute_trajectory_signature(trajectory_energy),
        }

    def test_strict_trajectory_reproducibility(self):
        """Asserts identical seeds produce identical state representations."""
        seed = 1337

        rollout_alpha = self.simulate_rollout(seed=seed)
        rollout_beta = self.simulate_rollout(seed=seed)

        assert (
            rollout_alpha["positions_hash"] == rollout_beta["positions_hash"]
        ), "Positions diverged for identical seeds."

        assert rollout_alpha["energy_hash"] == rollout_beta["energy_hash"], "Energies diverged for identical seeds."

    def test_hash_collision_resistance_across_divergent_seeds(self):
        """Asserts sequence sensitivity to seed mutations."""
        seed_a = 42
        seed_b = 43

        rollout_a = self.simulate_rollout(seed=seed_a)
        rollout_b = self.simulate_rollout(seed=seed_b)

        assert (
            rollout_a["positions_hash"] != rollout_b["positions_hash"]
        ), "Different seeds yielded identical position hashes."

        assert rollout_a["energy_hash"] != rollout_b["energy_hash"], "Different seeds yielded identical energy hashes."

    def test_trajectory_sensitivity_to_micro_perturbations(self):
        """Asserts hash sensitivity to coordinate-level epsilon perturbations."""
        seed = 101010
        enforce_deterministic_state(seed)

        dim_topology = WORLD_DIM if len(WORLD_DIM) == 4 else (100, 100, 100, 10)
        pop_control = VectorizedPopulation(initial_agents=8, world_dim=dim_topology, max_agents=8).to(MODEL_DEVICE)

        enforce_deterministic_state(seed)
        pop_perturbed = VectorizedPopulation(initial_agents=8, world_dim=dim_topology, max_agents=8).to(MODEL_DEVICE)

        traj_control = []
        traj_perturbed = []

        for step in range(3):
            actions = torch.ones((8,), dtype=torch.long, device=MODEL_DEVICE)

            pop_control.step(actions)
            traj_control.append(pop_control.positions.clone())

            if step == 1:
                with torch.no_grad():
                    pop_perturbed.positions[0, 0] += 0.001

            pop_perturbed.step(actions)
            traj_perturbed.append(pop_perturbed.positions.clone())

        hash_ctrl = compute_trajectory_signature(traj_control, precision=4)
        hash_prtb = compute_trajectory_signature(traj_perturbed, precision=4)

        assert hash_ctrl != hash_prtb, "Hash failed to change after a micro-perturbation."
