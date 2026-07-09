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

"""Test PPO algorithm convergence via single-state overfitting."""

from typing import NamedTuple

import torch
import torch.nn as nn
from torch.optim import Adam


class ActorCriticOutput(NamedTuple):
    policy_logits: torch.Tensor
    value: torch.Tensor


class TrivialActorCritic(nn.Module):
    """Minimal MLP actor-critic."""

    def __init__(self, state_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.actor_head = nn.Linear(32, num_actions)
        self.critic_head = nn.Linear(32, 1)

    def forward(self, state: torch.Tensor, *args, **kwargs):
        # [batch_size, 32]
        features = self.net(state)

        # policy_logits: [batch_size, num_actions], value: [batch_size]
        return ActorCriticOutput(policy_logits=self.actor_head(features), value=self.critic_head(features).squeeze(-1))


class TestAlgorithmicConvergence:
    """Validate PPO optimization loop."""

    def test_ppo_single_state_overfit(self) -> None:
        """Test policy convergence to optimal action in a constant-state MDP."""
        state_dim = 4
        num_actions = 2
        batch_size = 64

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TrivialActorCritic(state_dim, num_actions).to(device)

        # Force rapid convergence.
        optimizer = Adam(model.parameters(), lr=1e-2)

        for update_step in range(50):
            # [batch_size, state_dim]
            states = torch.ones(batch_size, state_dim, device=device)

            with torch.no_grad():
                out = model(states)
                dist = torch.distributions.Categorical(logits=out.policy_logits)
                actions = dist.sample()

                # [batch_size]
                log_probs = dist.log_prob(actions)
                values = out.value

            # R(a=0) = 1.0, R(a!=0) = -1.0
            rewards = torch.where(actions == 0, torch.tensor(1.0, device=device), torch.tensor(-1.0, device=device))

            advantages = rewards - values.detach()
            returns = rewards

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            for _ in range(4):
                current_out = model(states)
                current_dist = torch.distributions.Categorical(logits=current_out.policy_logits)
                new_log_probs = current_dist.log_prob(actions)

                ratio = torch.exp(new_log_probs - log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * ((current_out.value - returns) ** 2).mean()
                entropy_loss = -0.01 * current_dist.entropy().mean()

                total_loss = actor_loss + critic_loss + entropy_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

        with torch.no_grad():
            # [1, state_dim]
            eval_state = torch.ones(1, state_dim, device=device)
            final_out = model(eval_state)

            # [1, num_actions]
            probabilities = torch.softmax(final_out.policy_logits, dim=-1)
            prob_action_zero = probabilities[0, 0].item()

            assert prob_action_zero > 0.95, f"Expected P(a=0) > 0.95, got {prob_action_zero:.4f}"
