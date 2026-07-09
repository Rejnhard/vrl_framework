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

"""Latency benchmarks for Warp physics kernels and PPO autograd propagation."""

import os
import sys

import pytest
import torch
import torch.nn as nn
import warp as wp

# Initialize Warp runtime.
try:
    wp.init()
except Exception:
    pass

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from vrl_framework.environment.world_dynamics import step_sparse_substrate_kernel  # noqa: E402


class MinimalMoEActorCritic(nn.Module):
    """Minimal actor-critic for autograd overhead profiling."""

    def __init__(self, state_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 128)
        )
        self.actor_head = nn.Linear(128, num_actions)
        self.critic_head = nn.Linear(128, 1)

    def forward(self, state: torch.Tensor):
        # [batch_size, 128]
        features = self.net(state)

        class MockOutput:
            def __init__(self, logits, value):
                self.policy_logits = logits
                self.value = value

        # policy_logits: [batch_size, num_actions], value: [batch_size]
        return MockOutput(self.actor_head(features), self.critic_head(features).squeeze(-1))


class TestHardwarePhysicsLatency:
    """Benchmark Warp JIT-compiled CUDA physics kernels."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="Hardware benchmarks strictly require CUDA architecture."
    )
    def test_warp_kernel_throughput(self, benchmark) -> None:
        """Benchmark sparse substrate execution speed."""
        dim = 64

        wp_state = wp.zeros((dim, dim, dim), dtype=wp.uint8, device="cuda")
        wp_energy = wp.full((dim, dim, dim), value=100, dtype=wp.uint8, device="cuda")
        wp_mask = wp.ones((dim, dim, dim), dtype=wp.uint8, device="cuda")

        def execute_physics_step() -> None:
            wp.launch(
                kernel=step_sparse_substrate_kernel,
                dim=(dim, dim, dim),
                inputs=[wp_state, wp_energy, wp_mask, dim, dim, dim],
                device="cuda",
            )
            # Synchronize stream for accurate kernel timing.
            wp.synchronize()

        benchmark.pedantic(execute_physics_step, warmup_rounds=2, iterations=50, rounds=10)
        mean_latency_sec = benchmark.stats.stats.mean

        assert mean_latency_sec < 0.0025, f"Warp kernel latency: {mean_latency_sec * 1000:.2f} ms"


class TestOptimizationLatency:
    """Benchmark PyTorch autograd and optimizer step latency."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="Hardware benchmarks strictly require CUDA architecture."
    )
    def test_ppo_backward_pass_latency(self, benchmark) -> None:
        """Benchmark Forward-Backward-Step optimization sequence overhead."""
        device = torch.device("cuda")
        model = MinimalMoEActorCritic(state_dim=256, num_actions=16).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

        batch_size = 2048

        # [batch_size, state_dim]
        states = torch.randn(batch_size, 256, device=device)

        def execute_ppo_update() -> None:
            optimizer.zero_grad(set_to_none=True)
            out = model(states)

            # Trigger full graph traversal.
            loss = (out.policy_logits.sum() * 0.1) + (out.value.sum() * 0.1)
            loss.backward()
            optimizer.step()

            # Synchronize stream for accurate timing.
            torch.cuda.synchronize(device)

        benchmark.pedantic(execute_ppo_update, warmup_rounds=2, iterations=20, rounds=10)
        mean_latency_sec = benchmark.stats.stats.mean

        assert mean_latency_sec < 0.015, f"PPO autograd latency: {mean_latency_sec * 1000:.2f} ms."
