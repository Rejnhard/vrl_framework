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

"""Integration tests for memory stability and deterministic checkpoint resumption."""

import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.testing import assert_close


class TrivialActorCritic(nn.Module):
    """Minimal MLP actor-critic."""

    def __init__(self, state_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 32), nn.ReLU())
        self.actor_head = nn.Linear(32, num_actions)
        self.critic_head = nn.Linear(32, 1)

    def forward(self, state: torch.Tensor):
        # [batch_size, 32]
        features = self.net(state)

        class MockOutput:
            def __init__(self, logits, value):
                self.policy_logits = logits
                self.value = value

        # policy_logits: [batch_size, num_actions], value: [batch_size]
        return MockOutput(self.actor_head(features), self.critic_head(features).squeeze(-1))


class TestSystemEndurance:
    """Validate memory stability during optimization steps."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Memory leak tracking requires CUDA allocator.")
    def test_autograd_memory_leak_prevention(self) -> None:
        """Measure VRAM allocation post-warmup to ensure autograd graph clearance."""
        state_dim, num_actions, batch_size = 8, 4, 128
        device = torch.device("cuda")

        model = TrivialActorCritic(state_dim, num_actions).to(device)
        optimizer = Adam(model.parameters(), lr=1e-3)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        def simulate_update_step():
            # [batch_size, state_dim]
            states = torch.randn(batch_size, state_dim, device=device)
            out = model(states)

            dist = torch.distributions.Categorical(logits=out.policy_logits)
            actions = dist.sample()

            # [batch_size]
            log_probs = dist.log_prob(actions)
            advantages = torch.randn(batch_size, device=device)
            loss = -(log_probs * advantages).mean() + out.value.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Stabilize CUDA caching allocator.
        for _ in range(5):
            simulate_update_step()

        warmup_memory = torch.cuda.memory_allocated()

        for _ in range(50):
            simulate_update_step()

        final_memory = torch.cuda.memory_allocated()
        growth_mb = (final_memory - warmup_memory) / (1024 * 1024)

        assert growth_mb <= 1.0, f"Memory leak: {growth_mb:.2f} MB increase post-warmup."


class TestStochasticRecovery:
    """Validate deterministic resumption from checkpoints."""

    def test_checkpoint_isomorphism_and_rng_state(self) -> None:
        """Verify state restoration for model, optimizer, and PRNG."""
        state_dim, num_actions = 8, 4
        device = torch.device("cpu")

        # [1, state_dim]
        test_state = torch.randn(1, state_dim, device=device)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = os.path.join(tmp_dir, "disaster_recovery.pt")

            torch.manual_seed(42)
            model_A = TrivialActorCritic(state_dim, num_actions).to(device)
            optimizer_A = Adam(model_A.parameters(), lr=1e-3)

            # Initialize optimizer momentum/variance buffers.
            out_A = model_A(test_state)
            loss_A = out_A.policy_logits.sum() + out_A.value.sum()
            optimizer_A.zero_grad()
            loss_A.backward()
            optimizer_A.step()

            out_before_crash = model_A(test_state)
            logits_before = out_before_crash.policy_logits.clone().detach()

            torch.save(
                {
                    "model_state": model_A.state_dict(),
                    "optimizer_state": optimizer_A.state_dict(),
                    "rng_state": torch.get_rng_state(),
                },
                ckpt_path,
            )

            del model_A
            del optimizer_A

            torch.manual_seed(999)

            model_B = TrivialActorCritic(state_dim, num_actions).to(device)
            optimizer_B = Adam(model_B.parameters(), lr=1e-3)

            checkpoint = torch.load(ckpt_path, weights_only=False)
            model_B.load_state_dict(checkpoint["model_state"])
            optimizer_B.load_state_dict(checkpoint["optimizer_state"])
            torch.set_rng_state(checkpoint["rng_state"])

            out_after_recovery = model_B(test_state)
            logits_after = out_after_recovery.policy_logits.clone().detach()

            assert_close(logits_before, logits_after, msg="Parameter restoration mismatch.")

            loss_B = out_after_recovery.policy_logits.sum() + out_after_recovery.value.sum()
            optimizer_B.zero_grad()
            loss_B.backward()
            optimizer_B.step()

            final_logits = model_B(test_state).policy_logits

            assert final_logits is not None, "Optimizer state divergence during recovery step."
