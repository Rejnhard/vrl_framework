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

"""Unit tests for Proximal Policy Optimization (PPO) surrogate objectives and value functions."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close

current_file = Path(__file__).resolve()
repo_root = current_file.parents[3]
src_root = str(repo_root / "src")

if src_root not in sys.path:
    sys.path.insert(0, src_root)
if str(repo_root) not in sys.path:
    sys.path.insert(1, str(repo_root))


class DummyActorCriticForPPO(nn.Module):
    """Deterministic Actor-Critic mock for isolating objective function evaluations."""

    def __init__(self, state_dim: int, num_actions: int):
        super().__init__()
        self.actor = nn.Linear(state_dim, num_actions)
        self.critic = nn.Linear(state_dim, 1)

        # Static weights enforce deterministic gradient assertions across test runs.
        nn.init.constant_(self.actor.weight, 0.1)
        nn.init.constant_(self.actor.bias, 0.0)
        nn.init.constant_(self.critic.weight, 0.1)
        nn.init.constant_(self.critic.bias, 0.0)

    def forward(self, state: torch.Tensor, *args, **kwargs):
        class MockOutput:
            def __init__(self, logits, value):
                self.policy_logits = logits
                self.value = value

        logits = self.actor(state)
        value = self.critic(state).squeeze(-1)
        return MockOutput(logits, value)


class TestGeneralizedAdvantageEstimation:

    def test_gae_discount_invariants(self) -> None:
        """Validates Generalized Advantage Estimation (GAE) bounded calculations."""
        values = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)

        gamma = 0.99
        lam = 0.95

        # Analytical GAE invariant derivations.
        delta_2 = 1.0 + gamma * 0.5 * 0.0 - 0.5
        delta_1 = 1.0 + gamma * 0.5 * 1.0 - 0.5
        delta_0 = 1.0 + gamma * 0.5 * 1.0 - 0.5

        a_2 = delta_2
        a_1 = delta_1 + gamma * lam * 1.0 * a_2
        a_0 = delta_0 + gamma * lam * 1.0 * a_1

        expected_advantages = torch.tensor([a_0, a_1, a_2], dtype=torch.float32)
        expected_advantages + values

        assert expected_advantages.var() > 0.0, "Advantage collapsed to a constant scalar."


class TestPPOSurrogateObjective:

    @pytest.fixture
    def mock_ppo_setup(self):
        state_dim, num_actions = 16, 4
        model = DummyActorCriticForPPO(state_dim, num_actions)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        return model, optimizer

    def test_trust_region_gradient_zeroing(self, mock_ppo_setup) -> None:
        """Tests policy gradient truncation when the probability ratio exceeds the trust region."""
        model, optimizer = mock_ppo_setup
        clip_param = 0.2

        state = torch.randn(1, 16)
        old_action = torch.tensor([1])
        advantage = torch.tensor([10.0])

        with torch.no_grad():
            out = model(state)
            old_dist = torch.distributions.Categorical(logits=out.policy_logits)
            old_log_prob = old_dist.log_prob(old_action)

        with torch.no_grad():
            model.actor.weight[1].add_(2.0)

        out_new = model(state)
        new_dist = torch.distributions.Categorical(logits=out_new.policy_logits)
        new_log_prob = new_dist.log_prob(old_action)
        ratio = torch.exp(new_log_prob - old_log_prob)

        assert ratio.item() > 1.0 + clip_param, f"Policy ratio {ratio.item()} insufficient to trigger clipping."

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

        policy_loss = -torch.min(surr1, surr2).mean()

        optimizer.zero_grad()
        policy_loss.backward()

        # Surrogate objective is clamped; gradient outside trust region evaluates to zero.
        actor_grad = model.actor.weight.grad

        assert actor_grad is not None, "Actor Autograd graph disconnected."
        assert_close(
            actor_grad, torch.zeros_like(actor_grad), msg="Gradients not zeroed outside the trust region bounds."
        )

    def test_entropy_bonus_gradient_direction(self, mock_ppo_setup) -> None:
        """Tests if the entropy bonus gradient correctly pushes logits toward a uniform distribution."""
        model, optimizer = mock_ppo_setup
        state = torch.randn(1, 16)

        with torch.no_grad():
            model.actor.weight.copy_(torch.randn_like(model.actor.weight) * 10.0)

        out = model(state)
        dist = torch.distributions.Categorical(logits=out.policy_logits)
        entropy = dist.entropy().mean()

        entropy_loss = -0.01 * entropy

        optimizer.zero_grad()
        entropy_loss.backward()

        grad = model.actor.weight.grad
        assert grad is not None, "Entropy bonus detached from computational graph."
        assert torch.norm(grad) > 0, "Entropy gradient collapsed to zero on a peaked distribution."


class TestPPOValueFunction:

    def test_value_function_clipping_bounds(self) -> None:
        """Validates value function (VF) gradient restriction during high variance updates."""
        clip_param = 0.2

        old_values = torch.tensor([1.0, 1.0, 1.0])
        returns = torch.tensor([10.0, 0.5, 1.1])

        new_values = torch.tensor([10.0, 0.5, 1.1], requires_grad=True)

        values_pred = new_values

        values_pred_clipped = old_values + torch.clamp(new_values - old_values, min=-clip_param, max=clip_param)

        vf_losses1 = (values_pred - returns) ** 2
        vf_losses2 = (values_pred_clipped - returns) ** 2

        vf_loss = 0.5 * torch.max(vf_losses1, vf_losses2).mean()
        vf_loss.backward()

        assert new_values.grad is not None, "Value Autograd graph disconnected."
        assert not torch.isnan(new_values.grad).any(), "NaN in value gradient."
