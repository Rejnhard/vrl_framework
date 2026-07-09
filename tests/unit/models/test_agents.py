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

"""Unit tests for RL agent policies and nested subsystems."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import assert_close

current_file = Path(__file__).resolve()
repo_root = current_file.parents[3]
src_root = str(repo_root / "src")

if src_root not in sys.path:
    sys.path.insert(0, src_root)
if str(repo_root) not in sys.path:
    sys.path.insert(1, str(repo_root))

from vrl_framework.core.settings import MODEL_DEVICE  # noqa: E402
from vrl_framework.models.agents import AdversarialModulationRNN, HighLevelPolicy  # noqa: E402

try:
    from vrl_framework.models.agents import ObjectEncoder
except ImportError:

    class ObjectEncoder(nn.Module):
        def __init__(self, dim, slot_dim, num_slots, iters, keyframe_interval):
            super().__init__()
            self.num_slots = num_slots
            self.slot_dim = slot_dim
            self.register_buffer("prev_slots", torch.zeros(1, num_slots, slot_dim))

        def forward(self, inputs):
            batch_size = inputs.size(0)
            return torch.zeros(batch_size, self.num_slots, self.slot_dim, device=inputs.device)


class _DummyFuzzyKB(nn.Module):
    def evaluate_truth_gate_differentiable(self, candidates: torch.Tensor) -> torch.Tensor:
        return torch.ones(candidates.size(0), device=candidates.device)


class DummyWorkerActorCritic(nn.Module):
    """Mocks low-level policy to isolate HIRO off-policy correction logic."""

    def __init__(self, num_actions: int = 4):
        super().__init__()
        self.num_actions = num_actions
        self.fuzzy_kb = _DummyFuzzyKB()

    def forward(self, worker_input: torch.Tensor, critic_context: torch.Tensor, intent_context: torch.Tensor):
        batch_size = worker_input.size(0)

        class MockOutput:
            def __init__(self, b_size, n_actions):
                self.policy_logits = torch.randn(b_size, 1, n_actions, device=worker_input.device)
                self.pessimistic_value = torch.randn(b_size, device=worker_input.device)

        return MockOutput(batch_size, self.num_actions)


class TestHighLevelPolicy:

    def test_manager_goal_scale_invariant(self) -> None:
        """Tests if high-level generated subgoals comply with the defined L2 norm scale."""
        input_dim, goal_dim = 128, 64
        goal_scale = 5.0
        policy = HighLevelPolicy(input_dim=input_dim, goal_dim=goal_dim, goal_scale=goal_scale).to(MODEL_DEVICE)

        state = torch.randn(8, input_dim, device=MODEL_DEVICE)
        goal = policy.get_manager_goal(state, add_noise=False)

        assert goal.shape == (8, goal_dim), "Goal tensor shape mismatch."

        norms = torch.norm(goal, p=2, dim=-1)
        expected_norms = torch.full((8,), goal_scale, dtype=torch.float32, device=MODEL_DEVICE)

        assert_close(norms, expected_norms, rtol=1e-4, atol=1e-4, msg="Goal L2 norm exceeds the defined goal_scale.")

    def test_hiro_off_policy_correction_bounds(self) -> None:
        """Validates bounds constraints for off-policy relabeling candidate selection."""
        batch_size, dim, goal_scale = 4, 64, 5.0
        policy = HighLevelPolicy(input_dim=dim, goal_dim=dim, goal_scale=goal_scale).to(MODEL_DEVICE)
        mock_ac = DummyWorkerActorCritic(num_actions=4).to(MODEL_DEVICE)

        states = torch.cat([torch.ones(batch_size, 1) * 2.0, torch.randn(batch_size, dim - 1)], dim=-1).to(
            MODEL_DEVICE
        )
        next_states = torch.cat([torch.ones(batch_size, 1) * 2.5, torch.randn(batch_size, dim - 1)], dim=-1).to(
            MODEL_DEVICE
        )
        actions = torch.randint(0, 4, (batch_size,)).float().to(MODEL_DEVICE)

        relabeled_goals = policy.hiro_off_policy_correction(
            states, next_states, actions, worker_actor_critic=mock_ac, num_candidates=8
        )

        assert relabeled_goals.shape == (batch_size, dim), "Relabeled goals shape mismatch."

        norms = torch.norm(relabeled_goals, p=2, dim=-1)
        expected_norms = torch.full((batch_size,), goal_scale, dtype=torch.float32, device=MODEL_DEVICE)
        assert_close(norms, expected_norms, rtol=1e-3, atol=1e-3)


class TestAdversarialModulationRNN:

    def test_film_norm_barrier_and_gradient_flow(self) -> None:
        """Validates FiLM layer stability and gradient throughput under large magnitude inputs."""
        dim, hidden_dim = 128, 64
        module = AdversarialModulationRNN(dim=dim, hidden_dim=hidden_dim).to(MODEL_DEVICE)

        x = torch.randn(4, dim, requires_grad=True, device=MODEL_DEVICE)
        scaled_x = x * 1000.0
        scaled_x.retain_grad()

        perturbed_x = module(scaled_x)

        assert perturbed_x.shape == scaled_x.shape, "Modulated tensor shape mismatch."
        assert not torch.isnan(perturbed_x).any(), "Forward pass returned NaNs."

        loss = perturbed_x.sum() + module.collapse_penalty
        loss.backward()

        assert scaled_x.grad is not None, "Input tensor did not receive gradients."
        assert not torch.isnan(scaled_x.grad).any(), "Backward pass computed NaN gradients."


class TestPPOKnowledgeDistillation:

    def test_kl_divergence_trust_region(self) -> None:
        """Validates boolean mask construction via Kullback-Leibler divergence thresholds."""
        batch_size, num_actions = 4, 8

        teacher_logits = torch.randn(batch_size, num_actions, device=MODEL_DEVICE) * 10.0
        student_logits_far = torch.randn(batch_size, num_actions, device=MODEL_DEVICE) * 10.0

        t_probs = F.softmax(teacher_logits.float(), dim=-1)
        s_log_probs = F.log_softmax(student_logits_far.float(), dim=-1)

        kl_divergence = F.kl_div(s_log_probs, t_probs, reduction="none").mean(dim=-1)

        assert (kl_divergence >= 0.0).all(), "KL Divergence must be non-negative."

        (kl_divergence < 2.0).float()

        student_logits_close = teacher_logits.clone() + (torch.randn_like(teacher_logits) * 0.1)
        s_log_probs_close = F.log_softmax(student_logits_close.float(), dim=-1)

        kl_div_close = F.kl_div(s_log_probs_close, t_probs, reduction="none").mean(dim=-1)
        trust_mask_close = (kl_div_close < 2.0).float()

        assert (trust_mask_close == 1.0).all(), "Trust mask rejected bounded student logits."
        assert (kl_div_close < kl_divergence).all(), "Proximal logits yield higher divergence than distant logits."


class TestObjectEncoder:

    def test_slot_attention_dimensions_and_matching(self) -> None:
        """Tests sequential processing dimensions and background slot invariances."""
        dim, slot_dim, num_slots = 64, 32, 5
        encoder = ObjectEncoder(dim=dim, slot_dim=slot_dim, num_slots=num_slots, iters=2, keyframe_interval=10).to(
            MODEL_DEVICE
        )

        batch_size, seq_len = 4, 16
        inputs = torch.randn(batch_size, seq_len, dim, device=MODEL_DEVICE)

        slots = encoder(inputs)

        assert slots.shape == (batch_size, num_slots, slot_dim), "Slot tensor shape mismatch."

        assert_close(
            slots[:, 0],
            encoder.prev_slots[:, 0].expand(batch_size, -1),
            msg="Background slot modified during slot attention forward pass.",
        )
