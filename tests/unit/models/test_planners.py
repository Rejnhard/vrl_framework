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

"""Unit tests for latent planning algorithms, MCTS, and epistemic tracking."""

import sys
from pathlib import Path
from typing import Tuple

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

from vrl_framework.core.contracts import PlannerDecisionTrace, PlanningBudget  # noqa: E402
from vrl_framework.core.settings import MODEL_DEVICE  # noqa: E402
from vrl_framework.models.planners import ActionMasker, MCTSPlanner, StabilityTracker  # noqa: E402


def patch_create_preview_budget(stability_score: float = 1.0):
    return PlanningBudget(
        health_score=stability_score,
        health_band=2,
        max_depth=0,
        num_samples=0,
        distill_enabled=False,
        teacher_ttl=0,
        allow_actor_lookahead=False,
        allow_teacher_write=False,
        allow_distillation=False,
        max_branch_survivors=0,
        min_survivor_floor=0,
        max_ood_risk=0.0,
        max_critic_divergence=0.0,
        max_planner_calls_per_env_step=0,
    )


class DummyDynamicsPredictor(nn.Module):
    """Deterministic transition model mock for isolating MCTS tree logic."""

    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.call_count = 0

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        self.call_count += state.size(0)
        return state + 0.1


class DummyActorCritic(nn.Module):
    """Mocks policy evaluation with static support bounds for reproducible selection."""

    def __init__(self, num_actions: int):
        super().__init__()
        self.num_actions = num_actions
        self.register_buffer("value_support", torch.linspace(-10, 10, 51))

    def forward(self, state: torch.Tensor, *args, **kwargs):
        batch_size = state.size(0)

        class MockOutput:
            def __init__(self, b, n, support_len, device):
                self.policy_logits = torch.arange(n, dtype=torch.float32).unsqueeze(0).expand(b, n).to(device)
                self.value_logits_1 = torch.randn(b, support_len, device=device)
                self.value_logits_2 = torch.randn(b, support_len, device=device)
                self.cost_value = torch.zeros(b, 1, device=device)
                self.pessimistic_value = -torch.norm(state, dim=-1, keepdim=True)

        return MockOutput(batch_size, self.num_actions, len(self.value_support), state.device)


class DummyCausalReasoner(nn.Module):
    def __init__(self, num_actions=8):
        super().__init__()
        self.num_actions = num_actions

    def forward(self, state_cat: torch.Tensor) -> torch.Tensor:
        return torch.zeros(state_cat.size(0), self.num_actions, device=state_cat.device)

    def inverse_dynamics(self, state_cat: torch.Tensor) -> torch.Tensor:
        return torch.zeros(state_cat.size(0), self.num_actions, device=state_cat.device)


class TestStabilityTracker:

    def test_epistemic_gate_tripping(self) -> None:
        """Tests parameter clamping based on epistemic uncertainty thresholds."""
        tracker = StabilityTracker().to(MODEL_DEVICE)
        tracker.dynamics_ready.fill_(True)
        tracker.epistemic_threshold = 5.0

        trust_high = tracker.evaluate_epistemic_gate(current_surprisal=1.0)
        assert trust_high is True, "Tracker erroneously restricted trust under low surprisal conditions."

        trust_low = tracker.evaluate_epistemic_gate(current_surprisal=20.0)
        assert trust_low is False, "Epistemic gate failed to clamp trust during extreme dynamic divergence."


class TestActionMasker:

    def test_gradient_safe_masking(self) -> None:
        """Validates gradient stability during backpropagation through masked Softmax."""
        masker = ActionMasker(num_actions=8, dim=256).to(MODEL_DEVICE)
        state = torch.randn(4, 256, requires_grad=True, device=MODEL_DEVICE)
        logits = torch.randn(4, 8, requires_grad=True, device=MODEL_DEVICE)

        masked_logits = masker(state, logits)
        probs = torch.nn.functional.softmax(masked_logits, dim=-1)
        loss = probs.sum()
        loss.backward()

        assert not torch.isnan(logits.grad).any(), "Masking induced NaN gradients via Softmax underflow."
        assert not torch.isnan(state.grad).any(), "Masking induced NaN gradients in state representation."


class TestControlBarrierFunctions:

    def test_cbf_safety_projection_and_autograd(self) -> None:
        """Tests Lagrangian constraint projection mapping and autograd preservation."""
        u_nom = torch.tensor([[1.0, 1.0, 0.0, 0.0], [2.0, 2.0, 0.0, 0.0]], requires_grad=True, device=MODEL_DEVICE)

        A_cbf = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]], device=MODEL_DEVICE)

        b_cbf = torch.tensor([[0.5], [0.5]], device=MODEL_DEVICE)

        constraint_violation = torch.bmm(A_cbf.unsqueeze(1), u_nom.unsqueeze(2)).squeeze(-1) - b_cbf

        A_norm_raw = torch.sum(A_cbf**2, dim=-1, keepdim=True)
        # Shift Softplus bounds by epsilon to prevent zero-division in Lagrange multipliers.
        A_norm_sq_safe = torch.nn.functional.softplus(A_norm_raw - 1e-3) + 1e-3

        raw_lambda = torch.nn.functional.relu(constraint_violation / A_norm_sq_safe)
        lambda_lagrange = torch.where(raw_lambda < 20.0, raw_lambda, 20.0 + torch.log1p(raw_lambda - 20.0))

        u_safe = u_nom - lambda_lagrange * A_cbf
        new_violation = torch.bmm(A_cbf.unsqueeze(1), u_safe.unsqueeze(2)).squeeze(-1) - b_cbf

        assert (
            new_violation <= 0.25
        ).all(), f"CBF failed to project action into the safe set. Violation magnitude: {new_violation}"

        loss = u_safe.sum()
        loss.backward()

        assert u_nom.grad is not None, "Lagrange projection severed the computational graph."
        assert not torch.isnan(u_nom.grad).any(), "NaN gradients induced by CBF projection."


class TestMCTSPlanner:

    @pytest.fixture
    def environment_mocks(self) -> Tuple[DummyDynamicsPredictor, DummyActorCritic, DummyCausalReasoner]:
        return (
            DummyDynamicsPredictor(state_dim=128).to(MODEL_DEVICE),
            DummyActorCritic(num_actions=8).to(MODEL_DEVICE),
            DummyCausalReasoner().to(MODEL_DEVICE),
        )

    def test_budget_hard_limits(self, environment_mocks: Tuple) -> None:
        """Ensures MCTS traversal respects computational budget."""
        dyn_model, policy, reasoner = environment_mocks
        planner = MCTSPlanner(num_actions=8, latent_dim=128).to(MODEL_DEVICE)

        batch_size = 4
        latent_state = torch.randn(batch_size, 128, device=MODEL_DEVICE)
        causal_context = torch.zeros(batch_size, 768, device=MODEL_DEVICE)
        halting_budget = torch.ones(batch_size, device=MODEL_DEVICE)

        budget = patch_create_preview_budget(stability_score=1.0)
        budget.max_depth = 2
        budget.num_samples = 4
        budget.allow_actor_lookahead = True

        class MockValidator:
            @staticmethod
            def assert_planner_budget_contract(b):
                pass

            @staticmethod
            def assert_policy_only_integrity(c, mode):
                pass

        import vrl_framework.models.planners

        vrl_framework.models.planners.PlannerValidator = MockValidator

        out = planner(
            initial_latent=latent_state,
            jepa_predictor=dyn_model,
            actor_critic=policy,
            critic_context=causal_context,
            planning_budget=budget,
            halting_budget=halting_budget,
            causal_engine=reasoner,
        )

        assert isinstance(
            out.decision_trace, PlannerDecisionTrace
        ), "MCTS failed to return the standardized trace object."
        assert dyn_model.call_count > 0, "Dynamics model was bypassed."

        expected_evaluations = batch_size * budget.max_depth * budget.num_samples
        assert (
            dyn_model.call_count <= expected_evaluations * 2
        ), "Planner vastly exceeded computational budget constraints."

    def test_ucb_determinism_under_seed(self, environment_mocks: Tuple) -> None:
        """Tests deterministic resolution of UCB tree policy under static RNG seeds."""
        dyn_model, policy, reasoner = environment_mocks
        planner = MCTSPlanner(num_actions=8, latent_dim=128).to(MODEL_DEVICE)

        batch_size = 2
        latent_state = torch.randn(batch_size, 128, device=MODEL_DEVICE)
        causal_context = torch.zeros(batch_size, 768, device=MODEL_DEVICE)
        halting_budget = torch.ones(batch_size, device=MODEL_DEVICE)

        budget = patch_create_preview_budget(stability_score=1.0)
        budget.max_depth = 3
        budget.num_samples = 5
        budget.allow_actor_lookahead = True

        torch.manual_seed(1337)
        out_1 = planner(latent_state, dyn_model, policy, causal_context, budget, halting_budget, reasoner)

        torch.manual_seed(1337)
        out_2 = planner(latent_state, dyn_model, policy, causal_context, budget, halting_budget, reasoner)

        assert_close(
            out_1.final_blended_logits,
            out_2.final_blended_logits,
            msg="MCTS tree expansion violated determinism under identical RNG seeds.",
        )
