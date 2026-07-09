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

import collections
import math
import time
from enum import Enum
from typing import Dict, Optional, Tuple

import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.nn.functional as F
import warp as wp

from vrl_framework.core.contracts import (
    ActionLogits,
    LatentMCTSOutput,
    LatentState,
    MetricsDict,
    PlannerDecisionTrace,
    PlannerOutput,
    PlannerRegime,
    PlannerValidator,
    PlanningBudget,
    TrainStepMetrics,
    beartype,
    jaxtyped,
)
from vrl_framework.core.settings import PLAN_CFG
from vrl_framework.math_ops.geometry import LorentzGeometry, compute_eff_dim
from vrl_framework.models.components import (
    FrequencyDomainBinding,
    SelectiveStateSpaceModel,
    ThresholdAttentionOptimized,
    layer_init,
)


class StabilityTracker(nn.Module):
    """Computes moving averages of critic, dynamics, and quantization errors to govern MCTS compute budget."""

    def __init__(self):
        super().__init__()
        self.register_buffer("ema_stats", torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.long))
        self.register_buffer("representation_ready", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("dynamics_ready", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("planner_ready", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("teacher_ready", torch.tensor(False, dtype=torch.bool))

        self.warmup_steps = 100
        self.epistemic_threshold = 1.5
        self.rep_heavy_steps = 200
        self.current_band = 2

    def evaluate_epistemic_gate(self, current_surprisal: float) -> bool:
        if not self.dynamics_ready.item():
            return torch.rand(1).item() < 0.15
        if current_surprisal > self.epistemic_threshold:
            return False
        return True

    def update_and_evaluate(self, metrics: PlannerOutput, planner_regret: float) -> PlanningBudget:
        self.step_counter.add_(1)
        with torch.no_grad():
            c_val = metrics.critic_divergence
            d_val = metrics.dynamics_error
            q_val = metrics.quantization_error
            r_val = planner_regret
            current_vals = self.ema_stats.new_tensor([c_val, 0.0, d_val, 0.0, q_val, 0.0, r_val, 0.0])
            self.ema_stats.lerp_(current_vals, weight=0.001)
            self.ema_stats[2::2].lerp_(current_vals[2::2], weight=0.01)

            abs_diffs = torch.abs(current_vals[0::2] - self.ema_stats[0::2])
            self.ema_stats[1].lerp_(abs_diffs[0], weight=0.001)
            self.ema_stats[3::2].lerp_(abs_diffs[1:], weight=0.01)

            means = self.ema_stats[[0, 2, 4]]
            mads = self.ema_stats[[1, 3, 5]]
            c_d_q_vals = torch.tensor([c_val, d_val, q_val], device=means.device, dtype=means.dtype)

            z_scores = torch.abs(c_d_q_vals - means).div_(mads.add(1e-4)).clamp_(min=0.0, max=15.0).tolist()
            z_critic, z_dyn, z_quant = z_scores

            z_critic = 0.0 if math.isnan(z_critic) else z_critic
            z_dyn = 0.0 if math.isnan(z_dyn) else z_dyn
            z_quant = 0.0 if math.isnan(z_quant) else z_quant

            normalized_stability_score = 0.4 * z_critic + 0.3 * z_dyn + 0.3 * z_quant

            steps = self.step_counter.item()

            if math.isnan(normalized_stability_score) or math.isinf(normalized_stability_score):
                normalized_stability_score = 3.0

            if steps > self.warmup_steps:
                self.representation_ready.fill_(bool(z_quant < 5.0))
            if steps > self.rep_heavy_steps:
                self.dynamics_ready.fill_(bool(self.representation_ready.item() and z_dyn < 5.0))
            self.planner_ready.fill_(bool(self.dynamics_ready.item() and z_critic < 5.0))
            self.teacher_ready.fill_(
                bool(self.planner_ready.item() and (r_val < 50.0 if not math.isnan(r_val) else False))
            )

            if not self.representation_ready.item() and steps <= self.warmup_steps * 2:
                self.current_band = 2
                adjusted_stability_score = normalized_stability_score * 0.5
            elif not self.planner_ready.item():
                if self.current_band == 0 and normalized_stability_score > 2.0:
                    self.current_band = 1
                elif self.current_band == 1 and normalized_stability_score < 1.0:
                    self.current_band = 0
                elif self.current_band == 2 and normalized_stability_score < 1.5:
                    self.current_band = 1
                adjusted_stability_score = normalized_stability_score * 1.0
            else:
                if self.current_band == 0 and normalized_stability_score > 2.5:
                    self.current_band = 1
                elif self.current_band == 1:
                    if normalized_stability_score > 4.0:
                        self.current_band = 2
                    elif normalized_stability_score < 1.5:
                        self.current_band = 0
                elif self.current_band == 2 and normalized_stability_score < 2.5:
                    self.current_band = 1

                if self.current_band == 2 and normalized_stability_score > 10.0:
                    self.current_band = 1
                    normalized_stability_score = 3.0

                adjusted_stability_score = normalized_stability_score * 1.5

            band = self.current_band

            normalized_stability_score = adjusted_stability_score

            max_depth = PLAN_CFG.max_depth if self.planner_ready.item() else 0
            num_samples = PLAN_CFG.num_samples if self.planner_ready.item() else 0
            distill = self.teacher_ready.item()
            ttl = 3 if self.teacher_ready.item() else 1
            lookahead = self.dynamics_ready.item()

            if band == 2:
                max_depth, num_samples, distill, ttl, lookahead = 0, 0, False, 0, False
            elif band == 1:
                max_depth = max(1, max_depth // 2)
                num_samples = max(4, num_samples // 2)
                distill = False

            allow_t_write = (band == 0) and self.teacher_ready.item()
            allow_distill = (band == 0) and distill
            max_survivors = max(2, num_samples // 2) if band < 2 else 0
            min_floor = max(1, max_survivors // 4)
            ood_risk_tolerance = 1.5 if band == 0 else (0.5 if band == 1 else 0.0)
            _ = 0.5 if band == 0 else (0.2 if band == 1 else 0.0)
            max_calls = 5 if band == 0 else (2 if band == 1 else 0)

            return PlanningBudget(
                health_score=normalized_stability_score,
                health_band=band,
                max_depth=max_depth,
                num_samples=num_samples,
                distill_enabled=distill,
                teacher_ttl=ttl,
                allow_actor_lookahead=lookahead,
                allow_teacher_write=allow_t_write,
                allow_distillation=allow_distill,
                max_branch_survivors=max_survivors,
                min_survivor_floor=min_floor,
                max_ood_risk=ood_risk_tolerance,
                max_critic_divergence=5.0,
                max_planner_calls_per_env_step=max_calls,
            )


class BudgetController(nn.Module):
    """Maps representational stability metrics to computation limits for trajectory rollouts."""

    def __init__(self, stability_threshold: float = 2.0):
        super().__init__()
        self.stability_threshold = stability_threshold
        self.tracker = StabilityTracker()
        self.register_buffer("planner_active_state", torch.tensor(False, dtype=torch.bool))

    def compute_budget(
        self, gate_output: PlannerOutput, planner_regret: float, planner_gain: float, halting_probability: float = 1.0
    ) -> PlanningBudget:
        """Determines resource allocation for MCTS rollouts based on model stability."""
        budget = self.tracker.update_and_evaluate(gate_output, planner_regret)

        bool_deactivate = (planner_regret > 0.1) and (planner_gain < -0.01)
        bool_activate = (planner_gain > 0.01) or (budget.health_score < 1.5)

        new_state = not bool_deactivate if self.planner_active_state.item() else bool_activate
        self.planner_active_state.fill_(new_state)

        if not self.planner_active_state.item():
            budget.num_samples = max(2, budget.num_samples // 4)
            budget.max_depth = max(1, budget.max_depth // 2)
            budget.allow_teacher_write = False

        if halting_probability < 0.1:
            budget.max_depth = max(1, budget.max_depth)
            budget.num_samples = max(2, budget.num_samples)

        return budget


class GatingHead(nn.Module):
    """Computes dynamic mixing weights between base policy and MCTS categorical distributions.

    Args:
        num_actions: Dimensionality of the discrete action space.
    """

    def __init__(self, num_actions: int):
        super().__init__()
        # Context projection concatenates policy priors, MCTS distributions, and 5 scalar metrics.
        # Shape: [batch_size, 2 * num_actions + 5]
        self.routing_net = nn.Sequential(nn.Linear(num_actions * 2 + 5, 64), nn.Mish(), nn.Linear(64, 2))

    def forward(
        self,
        base_logits: torch.Tensor,
        plan_logits: torch.Tensor,
        confidence_margin: torch.Tensor,
        critic_divergence: torch.Tensor,
        control_cost: torch.Tensor,
        stability_score: torch.Tensor,
        halting_budget: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:

        base_logits.size(0)
        context = torch.cat(
            [
                base_logits,
                plan_logits,
                confidence_margin.unsqueeze(-1),
                critic_divergence.unsqueeze(-1),
                control_cost.unsqueeze(-1),
                stability_score.unsqueeze(-1),
                halting_budget.unsqueeze(-1),
            ],
            dim=-1,
        )

        routing_logits = self.routing_net(context)

        horizon_decay = max(0.01, 0.5 - (0.4 * getattr(self, "global_train_step", 0) / 100000.0))
        dynamic_tau = torch.clamp(
            torch.tensor(horizon_decay - (0.1 * (1.0 - stability_score.mean().item() / 3.0))), min=0.1, max=1.0
        ).item()
        blend_decisions = F.gumbel_softmax(routing_logits, tau=dynamic_tau, hard=False)
        blend_weight = blend_decisions[:, 1].unsqueeze(-1)

        metrics_mean = torch.stack(
            [
                stability_score.mean(),
                critic_divergence.mean(),
                confidence_margin.mean(),
                control_cost.mean(),
                halting_budget.mean(),
            ]
        )

        cond_observe = (
            (metrics_mean[0] > 1.5) | (metrics_mean[1] > 5.0) | (metrics_mean[2] < -0.5) | (metrics_mean[4] < 0.1)
        )
        cond_advice = (metrics_mean[0] > 0.5) | (metrics_mean[3] > 2.0) | (metrics_mean[2] < 0.0)

        blend_weight = torch.where(
            cond_observe,
            torch.zeros_like(blend_weight),
            torch.where(cond_advice, torch.clamp(blend_weight, max=0.4), blend_weight),
        )

        if cond_observe.item():
            regime = PlannerRegime.OBSERVE_ONLY
        elif cond_advice.item():
            regime = PlannerRegime.ADVICE_ONLY
        else:
            regime = PlannerRegime.TEACHER_AUTHORITY

        return blend_weight, regime


class MCTSPlanner(nn.Module):
    """Latent Monte Carlo Tree Search (MCTS)."""

    def __init__(self, num_actions: int, latent_dim: int = 256):
        super().__init__()
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        self.lambda_temperature = PLAN_CFG.lambda_temperature
        self.noise_sigma = PLAN_CFG.noise_sigma

        self.action_projector = nn.Linear(num_actions, latent_dim, bias=False)
        torch.nn.init.orthogonal_(self.action_projector.weight)

        self.budget_controller = BudgetController(stability_threshold=2.0)
        self.routing_head = GatingHead(num_actions)

    def forward(
        self,
        initial_latent: torch.Tensor,
        jepa_predictor: nn.Module,
        actor_critic: nn.Module,
        critic_context: torch.Tensor,
        planning_budget: PlanningBudget,
        halting_budget: torch.Tensor,
        causal_engine: nn.Module,
        sae_module: Optional[nn.Module] = None,
        relational_memory: Optional[nn.Module] = None,
    ) -> LatentMCTSOutput:
        """
        Args:
            initial_latent (torch.Tensor): Root node representations [B, D].
            jepa_predictor (nn.Module): Forward dynamics model.
            actor_critic (nn.Module): Base policy and value network.
            critic_context (torch.Tensor): Auxiliary conditioning vector.
            planning_budget (PlanningBudget): Constraints for depth and sample count.
            halting_budget (torch.Tensor): ACT halting thresholds [B].
            causal_engine (nn.Module): Inverse dynamics mapping.
            sae_module (Optional[nn.Module]): Sparse autoencoder constraint module.
            relational_memory (Optional[nn.Module]): External episodic buffer.

        Returns:
            LatentMCTSOutput: Contains final blended logits [B, num_actions] and the decision trace.
        """
        PlannerValidator.assert_planner_budget_contract(planning_budget)
        horizon = planning_budget.max_depth
        num_samples = planning_budget.num_samples
        batch_size = initial_latent.size(0)
        device = initial_latent.device

        with torch.no_grad():
            PlannerValidator.assert_policy_only_integrity(critic_context, mode="policy_only")
            ac_out = actor_critic(initial_latent, critic_context)

            if ac_out.policy_logits.dim() > 2:
                base_logits = ac_out.policy_logits[:, 0, :]
            else:
                base_logits = ac_out.policy_logits

            base_logits_clean = base_logits[:, : self.num_actions]

            # -10 * ln(2) shift to penalize base unconditioned logits
            constant_penalty = -6.9314718056
            base_logits_clean = torch.nan_to_num(
                base_logits_clean + constant_penalty, nan=-50.0, posinf=50.0, neginf=-50.0
            )

            if horizon == 0 or num_samples == 0 or not planning_budget.allow_actor_lookahead:
                trace = PlannerDecisionTrace(
                    baseline_logits=base_logits_clean,
                    planner_logits_preblend=base_logits_clean,
                    executed_planner_logits=base_logits_clean,
                    final_action_logits=base_logits_clean,
                    truth_margin=0.0,
                    critic_divergence=0.0,
                    control_cost=0.0,
                    selected_depth=0,
                    selected_samples=0,
                    divergence_from_base=0.0,
                    predicted_gain=0.0,
                    batch_adv=0.0,
                    conf_score=0.0,
                    ood_risk=0.0,
                    planner_confidence=0.0,
                    planner_blend_weight=0.0,
                    reject_ratio=0.0,
                    survivor_ratio=0.0,
                    same_batch_advantage=0.0,
                    teacher_admitted=False,
                    teacher_confirmed=False,
                    halting_budget_used=halting_budget.mean().item(),
                    planner_regime=PlannerRegime.OBSERVE_ONLY,
                    planner_temperature=1.0,
                    baselogits_temperature=1.0,
                    prefilter_time_ms=0.0,
                    rollout_time_ms=0.0,
                )
                return LatentMCTSOutput(final_blended_logits=base_logits_clean, decision_trace=trace)

            base_action_probs = F.softmax(base_logits_clean.float(), dim=-1).to(base_logits_clean.dtype)
            mu_seq = base_action_probs.unsqueeze(1).expand(-1, horizon, -1).clone()

            noise = torch.randn(batch_size, num_samples, horizon, self.num_actions, device=device) * self.noise_sigma

            # Dirichlet alpha scaling inversely proportional to effective dimensionality
            eff_dim = compute_eff_dim(initial_latent)
            target_dim = initial_latent.size(-1)
            entropy_bonus = torch.clamp(
                (1.0 - (eff_dim / (target_dim / 2.0))).clone().detach().to(device), min=0.1, max=1.0
            )

            dirichlet_alpha = torch.full((self.num_actions,), 0.3, device=device).div_(entropy_bonus)

            dirichlet_dist = torch.distributions.Dirichlet(dirichlet_alpha)
            root_noise = dirichlet_dist.sample((batch_size, num_samples))

            action_samples_raw = mu_seq.unsqueeze(1).add(noise)
            action_samples = F.softmax(action_samples_raw.float(), dim=-1).type_as(mu_seq)

            exploration_fraction = entropy_bonus.mul(0.25)
            action_samples[:, :, 0, :].mul_(1.0 - exploration_fraction).add_(root_noise.mul(exploration_fraction))

            prefilter_start = time.perf_counter()
            base_u_initial = mu_seq[:, 0, :].unsqueeze(1).expand(-1, num_samples, -1)
            kl_dist_pre = torch.sum((action_samples[:, :, 0, :] - base_u_initial) ** 2, dim=-1)

            beam_width = max(
                1, max(planning_budget.min_survivor_floor, min(num_samples, planning_budget.max_branch_survivors))
            )
            _, top_survivor_indices = torch.topk(kl_dist_pre, beam_width, dim=1, largest=False)
            top_k_actions = action_samples.gather(
                1, top_survivor_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, horizon, self.num_actions)
            )

            prefilter_time = (time.perf_counter() - prefilter_start) * 1000.0

            current_state = initial_latent.repeat_interleave(beam_width, dim=0)
            expanded_critic_ctx = critic_context.repeat_interleave(beam_width, dim=0)

            trajectory_returns = torch.zeros(batch_size * beam_width, device=device)
            control_costs = torch.zeros(batch_size * beam_width, device=device)

            active_mask = torch.ones(batch_size * beam_width, dtype=torch.bool, device=device)
            confidence_sum = torch.zeros(batch_size * beam_width, device=device)
            accumulated_divergence = torch.zeros(batch_size * beam_width, device=device)
            uncertainty_penalty_sum = torch.zeros(batch_size * beam_width, device=device)

            selected_depth = 0
            rollout_start = time.perf_counter()

            for t in range(horizon):
                selected_depth = t + 1

                current_action = top_k_actions[:, :, t, :].reshape(batch_size * beam_width, self.num_actions)
                act_latent = self.action_projector(current_action)

                ambient_shift = current_state + act_latent * 0.01

                if current_state.requires_grad:
                    r_next_state = torch.utils.checkpoint.checkpoint(
                        jepa_predictor, ambient_shift, use_reentrant=False
                    )
                else:
                    r_next_state = jepa_predictor(ambient_shift)

                r_next_state = 10.0 * torch.tanh(r_next_state / 10.0)
                next_state = LorentzGeometry.project(r_next_state)

                inverse_engine = getattr(causal_engine, "inverse_dynamics", None)
                if inverse_engine is not None:
                    state_next_cat = torch.cat([current_state, next_state], dim=-1)
                    pred_inv_logits = inverse_engine(state_next_cat)
                    inverse_loss = F.cross_entropy(
                        pred_inv_logits, torch.argmax(current_action, dim=-1), reduction="none"
                    )
                else:
                    inverse_loss = torch.zeros(batch_size * beam_width, device=device)

                if sae_module is not None and relational_memory is not None:
                    confidence_margin = relational_memory.evaluate_truth_gate_differentiable(next_state)
                else:
                    confidence_margin = torch.zeros(batch_size * beam_width, device=device)

                confidence_sum = confidence_sum + (confidence_margin * active_mask.float())
                confidence_penalty = F.relu(0.25 - confidence_margin) * (1.0 + 0.2 * t) * 2.0

                confidence_penalty > planning_budget.max_ood_risk
                uncertainty_penalty_sum = uncertainty_penalty_sum + (confidence_penalty * active_mask.float())

                step_ac_out = actor_critic(next_state, expanded_critic_ctx)
                v1_probs = F.softmax(step_ac_out.value_logits_1.float(), dim=-1).to(step_ac_out.value_logits_1.dtype)
                v2_probs = F.softmax(step_ac_out.value_logits_2.float(), dim=-1).to(step_ac_out.value_logits_2.dtype)
                v1_val = torch.sum(v1_probs * actor_critic.value_support, dim=-1)
                v2_val = torch.sum(v2_probs * actor_critic.value_support, dim=-1)

                critic_divergence = torch.clamp(torch.abs(v1_val - v2_val), min=0.0, max=5.0)
                accumulated_divergence = accumulated_divergence + (critic_divergence * active_mask.float())

                div_mask = critic_divergence > (planning_budget.max_critic_divergence * 10.0)
                div_mask = div_mask.to(torch.bool)

                penalty_sum = torch.clamp(inverse_loss + torch.abs(confidence_penalty) + critic_divergence, max=100.0)
                soft_penalty = torch.clamp(-10.0 * F.softplus(penalty_sum), min=-1000.0, max=0.0)

                trajectory_returns = trajectory_returns + (soft_penalty * active_mask.float())

                invalid_mask = (penalty_sum > 50.0) & active_mask
                trajectory_returns = trajectory_returns.masked_fill(invalid_mask, -50.0)
                active_mask = active_mask & (~invalid_mask)

                step_cost_critic = step_ac_out.cost_value.squeeze(-1)
                state_value = step_ac_out.pessimistic_value.squeeze(-1)

                uncertainty_penalty = critic_divergence * (1.0 + 0.2 * t) * 0.5
                kinematic_cost_penalty = step_cost_critic * (1.0 + 0.2 * t) * 0.2

                step_returns = (0.95**t) * (
                    state_value - confidence_penalty - uncertainty_penalty - kinematic_cost_penalty
                )
                step_returns = torch.clamp(step_returns, min=-15.0, max=15.0)
                trajectory_returns = trajectory_returns + (step_returns * active_mask.float())

                base_u = (
                    mu_seq[:, t, :]
                    .unsqueeze(1)
                    .expand(-1, beam_width, -1)
                    .reshape(batch_size * beam_width, self.num_actions)
                )
                kl_control_cost = (
                    self.lambda_temperature * torch.sum((current_action - base_u) ** 2, dim=-1) / (self.noise_sigma**2)
                )
                control_costs = control_costs + (kl_control_cost * active_mask.float())

                current_state = torch.where(active_mask.unsqueeze(-1), next_state, current_state)

            rollout_time = (time.perf_counter() - rollout_start) * 1000.0

            trajectory_returns = trajectory_returns.view(batch_size, beam_width)
            control_costs = control_costs.view(batch_size, beam_width)
            active_mask_2d = active_mask.view(batch_size, beam_width)
            confidence_sum = confidence_sum.view(batch_size, beam_width)
            aaccum_divergence = accumulated_divergence.view(batch_size, beam_width)
            divergence_penalty_2d = uncertainty_penalty_sum.view(batch_size, beam_width)

            effective_survivors = active_mask_2d.sum(dim=-1).float()
            effective_survivors_clamped = torch.clamp(effective_survivors, min=1.0)

            final_control_cost = (control_costs * active_mask_2d.float()).sum(dim=-1) / effective_survivors_clamped
            final_control_cost_scalar = final_control_cost.mean().item()

            total_utility = trajectory_returns - control_costs

            torch.finfo(total_utility.dtype).min / 2.0
            total_utility = torch.where(
                active_mask_2d, total_utility, total_utility - F.softplus(divergence_penalty_2d) * 10.0
            )

            all_terminated = ~active_mask_2d.any(dim=1, keepdim=True)
            uniform_exploration = torch.randn_like(total_utility) * 5.0
            total_utility = torch.where(all_terminated, uniform_exploration, total_utility)

            log_weights = F.log_softmax((total_utility / self.lambda_temperature).float(), dim=1).to(
                trajectory_returns.dtype
            )  # [B, BeamWidth]
            log_action_probs = torch.log(top_k_actions[:, :, 0, :] + 1e-10)  # [B, BeamWidth, num_actions]
            planner_logits_preblend = torch.logsumexp(
                log_weights.unsqueeze(-1) + log_action_probs, dim=1
            )  # [B, num_actions]

            avg_truth_per_sample = (
                (confidence_sum * active_mask_2d.float()).sum(dim=-1)
                / effective_survivors_clamped
                / max(1, selected_depth)
            )
            avg_div_per_sample = (
                (aaccum_divergence * active_mask_2d.float()).sum(dim=-1)
                / effective_survivors_clamped
                / max(1, selected_depth)
            )

            ood_risk_per_sample = (divergence_penalty_2d * active_mask_2d.float()).sum(
                dim=-1
            ) / effective_survivors_clamped

            metrics_stack = torch.stack(
                [
                    avg_truth_per_sample.mean(),
                    avg_div_per_sample.mean(),
                    ood_risk_per_sample.max() if active_mask_2d.any() else torch.tensor(0.0, device=device),
                    effective_survivors.sum() / (batch_size * num_samples),
                ]
            ).tolist()
            avg_truth, avg_div, ood_risk, survivor_ratio = metrics_stack

            blend_w, regime = self.routing_head(
                base_logits_clean,
                planner_logits_preblend,
                avg_truth_per_sample,
                avg_div_per_sample,
                final_control_cost,
                torch.tensor(planning_budget.health_score, device=device).expand(batch_size),
                halting_budget,
            )

            if not planning_budget.allow_teacher_write and regime == PlannerRegime.TEACHER_AUTHORITY:
                regime = PlannerRegime.DISTILLATION_MODE

            if not planning_budget.allow_distillation and regime == PlannerRegime.TEACHER_AUTHORITY:
                regime = PlannerRegime.DISTILLATION_MODE

            uncertainty_score = max(0.1, 1.0 - (avg_div * 0.1) - max(0.0, -avg_truth)) * max(0.1, survivor_ratio)
            base_temp_scale = 1.0 + (1.0 - uncertainty_score)
            plan_temp_scale = (
                torch.clamp(torch.tensor(planning_budget.health_score * 0.5, device=device), min=0.1, max=2.0)
                / uncertainty_score
            )

            if regime == PlannerRegime.TEACHER_AUTHORITY:
                plan_temp_scale = plan_temp_scale * 0.5
                base_temp_scale = base_temp_scale * 1.5
            elif regime == PlannerRegime.OBSERVE_ONLY:
                plan_temp_scale = plan_temp_scale * 2.0
                base_temp_scale = 1.0

            if survivor_ratio < 0.2 or ood_risk > 1.0:
                plan_temp_scale = plan_temp_scale * 5.0
                base_temp_scale = 1.0
                regime = PlannerRegime.OBSERVE_ONLY
                blend_w = blend_w * 0.0

            executed_planner_logits = planner_logits_preblend / plan_temp_scale
            base_logits_temp = base_logits_clean / base_temp_scale

            final_blended_logits = base_logits_temp * (1.0 - blend_w) + executed_planner_logits * blend_w

            base_probs = F.softmax(base_logits_clean.float(), dim=-1)
            planned_log_probs = F.log_softmax(executed_planner_logits.float(), dim=-1)
            divergence_from_base = F.kl_div(planned_log_probs, base_probs, reduction="batchmean").item()

            penalty_factors = (
                torch.clamp(torch.tensor(avg_div), max=1.0).item()
                + max(0.0, -avg_truth)
                + (ood_risk * 0.1)
                + (final_control_cost_scalar * 0.01)
                + (1.0 - survivor_ratio)
            )
            regime_multiplier = (
                1.0
                if regime in [PlannerRegime.TEACHER_AUTHORITY, getattr(PlannerRegime, "DISTILLATION_MODE", None)]
                else (0.5 if regime == PlannerRegime.ADVICE_ONLY else 0.1)
            )
            planner_confidence = max(0.0, 1.0 - penalty_factors) * regime_multiplier

            trace = PlannerDecisionTrace(
                baseline_logits=base_logits_clean,
                planner_logits_preblend=planner_logits_preblend,
                executed_planner_logits=executed_planner_logits,
                final_action_logits=final_blended_logits,
                truth_margin=avg_truth,
                critic_divergence=avg_div,
                control_cost=final_control_cost_scalar,
                selected_depth=selected_depth,
                selected_samples=beam_width,
                divergence_from_base=divergence_from_base,
                predicted_gain=(
                    torch.clamp(total_utility[total_utility > -50.0], min=-10.0, max=10.0).mean().item()
                    if (active_mask_2d.any() and total_utility[total_utility > -50.0].numel() > 0)
                    else 0.0
                ),
                batch_adv=0.0,
                conf_score=0.0,
                ood_risk=ood_risk,
                planner_confidence=planner_confidence,
                planner_blend_weight=blend_w.mean().item(),
                reject_ratio=1.0 - (beam_width / max(1, num_samples)),
                survivor_ratio=survivor_ratio,
                same_batch_advantage=0.0,
                teacher_admitted=(regime == PlannerRegime.TEACHER_AUTHORITY),
                teacher_confirmed=False,
                halting_budget_used=halting_budget.mean().item(),
                planner_regime=regime,
                planner_temperature=float(plan_temp_scale),
                baselogits_temperature=float(base_temp_scale),
                prefilter_time_ms=prefilter_time,
                rollout_time_ms=rollout_time,
            )

        return LatentMCTSOutput(final_blended_logits=final_blended_logits, decision_trace=trace)


class HaltingHead(nn.Module):
    """
    Adaptive Computation Time (ACT).
    Halts when accumulated p_t exceed 1 - epsilon. Requires strict masking on padded sequences.
    """

    def __init__(self, dim=256, max_ponder_steps=5, epsilon=0.01):
        super().__init__()
        self.dim = dim
        self.max_ponder_steps = max_ponder_steps
        self.epsilon = epsilon

        self.halt_evaluator = nn.Sequential(
            layer_init(nn.Linear(dim, 128)), nn.Mish(), layer_init(nn.Linear(128, 1)), nn.Sigmoid()
        )

        self.rnn_cell = nn.GRUCell(dim, dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Returns: halt_probs [B], mean_ponder_cost, final_state [B, D]
        batch_size = x.size(0)
        device = x.device

        # Accumulators for ACT formulation [B]
        halting_probabilities = torch.zeros(batch_size, device=device)
        ponder_costs = torch.zeros(batch_size, device=device)
        remainders = torch.ones(batch_size, device=device)

        # State buffers
        updates = torch.zeros_like(x)  # [B, D]
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        current_state = x.clone()

        zero_context = torch.zeros_like(current_state)

        for step in range(self.max_ponder_steps):
            next_state = self.rnn_cell(zero_context, current_state)  # [B, Dim]
            current_state = torch.where(active_mask.unsqueeze(-1), next_state, current_state)

            raw_halt_prob = self.halt_evaluator(current_state).squeeze(-1)  # [B]
            halt_prob = torch.clamp(raw_halt_prob, min=0.05, max=0.95)

            halt_condition = (halting_probabilities + halt_prob) >= (1.0 - self.epsilon)
            is_last_step = step == self.max_ponder_steps - 1

            stop_now = (halt_condition | is_last_step) & active_mask  # [B]
            continue_now = (~stop_now) & active_mask  # [B]

            p_t = torch.where(stop_now, remainders, halt_prob)  # [B]

            halting_probabilities = halting_probabilities + p_t
            ponder_costs = ponder_costs + active_mask.float()
            remainders = remainders - p_t

            updates = updates + (current_state * p_t.unsqueeze(-1))  # [B, D]

            active_mask = continue_now

        final_state = updates + (current_state * remainders.unsqueeze(-1))  # [B, Dim]

        return halting_probabilities, ponder_costs.mean(), final_state


class PlateauDetector:
    """Detects multivariable stagnation in training metrics using linear regression slopes over a sliding window.

    Args:
        window_size (int): Number of historical steps to evaluate for slope calculation.
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.history_return: collections.deque[float] = collections.deque(maxlen=window_size)
        self.history_dynamics: collections.deque[float] = collections.deque(maxlen=window_size)
        self.history_latent: collections.deque[float] = collections.deque(maxlen=window_size)
        self.history_planning: collections.deque[float] = collections.deque(maxlen=window_size)
        self.history_hit_rate: collections.deque[float] = collections.deque(maxlen=window_size)

        self.return_slope = 0.0
        self.dynamics_slope = 0.0
        self.latent_slope = 0.0
        self.planning_slope = 0.0
        self.hit_rate_slope = 0.0

    def update_and_check(self, metrics: TrainStepMetrics) -> bool:
        self.history_return.append(metrics.sample_efficiency)
        self.history_dynamics.append(metrics.dynamics_mse)
        self.history_latent.append(metrics.latent_rank)
        self.history_planning.append(metrics.planning_gain)
        self.history_hit_rate.append(metrics.retrieval_hit_rate)

        n = len(self.history_return)
        if n < self.window_size:
            return False

        x = torch.arange(n, dtype=torch.float32)
        x_centered = x - x.mean()
        denominator = torch.sum(x_centered**2)

        if denominator == 0:
            return False

        y_matrix = torch.tensor(
            [
                list(self.history_return),
                list(self.history_dynamics),
                list(self.history_latent),
                list(self.history_planning),
                list(self.history_hit_rate),
            ],
            dtype=torch.float32,
        )

        y_centered = y_matrix - y_matrix.mean(dim=1, keepdim=True)
        slopes = torch.matmul(y_centered, x_centered) / denominator

        self.return_slope, self.dynamics_slope, self.latent_slope, self.planning_slope, self.hit_rate_slope = (
            slopes.tolist()
        )

        stagnant_count = torch.sum(torch.abs(slopes) < 1e-4).item()
        return stagnant_count >= 3


class ModuleSurvivalRule:
    MODULE_METRIC_MAP = {
        "planner": "planning_gain",
        "memory": ["retrieval_hit_rate", "sample_efficiency"],
        "world_model": "dynamics_mse",
    }

    @staticmethod
    def evaluate_module(baseline: MetricsDict, current: MetricsDict) -> bool:
        """Validates if a module improves primary metrics without regressions."""
        primary_improvement = (current.planning_gain > baseline.planning_gain) or (
            current.sample_efficiency > baseline.sample_efficiency
        )
        no_regression = (
            (current.robustness >= baseline.robustness * 0.95)
            and (current.dynamics_mse <= baseline.dynamics_mse * 1.05)
            and (current.compute_efficiency >= baseline.compute_efficiency * 0.95)
        )
        return primary_improvement and no_regression


def make_critic_context(device: torch.device, hidden_dim: int = 768) -> torch.Tensor:
    """Allocates empty evaluation context."""
    return torch.zeros(1, hidden_dim, device=device)


def create_planning_budget(
    band: int, stability_score: float, planner_ready: bool, teacher_ready: bool, dynamics_ready: bool
):
    max_depth = PLAN_CFG.max_depth if planner_ready else 0
    num_samples = PLAN_CFG.num_samples if planner_ready else 0
    distill = teacher_ready
    ttl = 3 if teacher_ready else 1
    lookahead = dynamics_ready

    if band == 2:
        max_depth, num_samples, distill, ttl, lookahead = 0, 0, False, 0, False
    elif band == 1:
        max_depth = max(1, max_depth // 2)
        num_samples = max(4, num_samples // 2)
        distill = False

    allow_t_write = (band == 0) and teacher_ready
    allow_distill = (band == 0) and distill
    max_survivors = max(2, num_samples // 2) if band < 2 else 0
    min_floor = max(1, max_survivors // 4)
    ood_tolerance = 1.5 if band == 0 else (0.5 if band == 1 else 0.0)
    critic_agree_floor = 0.5 if band == 0 else (0.2 if band == 1 else 0.0)
    max_calls = 5 if band == 0 else (2 if band == 1 else 0)

    return PlanningBudget(
        health_score=stability_score,
        health_band=band,
        max_depth=max_depth,
        num_samples=num_samples,
        distill_enabled=distill,
        retention_steps=ttl,
        allow_actor_lookahead=lookahead,
        allow_teacher_write=allow_t_write,
        allow_distillation=allow_distill,
        max_branch_survivors=max_survivors,
        min_survivor_floor=min_floor,
        max_ood_risk=ood_tolerance,
        max_critic_divergence=critic_agree_floor,
        max_planner_calls_per_env_step=max_calls,
    )


def create_preview_budget(stability_score: float = 1.0):
    return PlanningBudget(
        health_score=stability_score,
        health_band=2,
        max_depth=0,
        num_samples=0,
        distill_enabled=False,
        retention_steps=0,
        allow_actor_lookahead=False,
        allow_teacher_write=False,
        allow_distillation=False,
        max_branch_survivors=0,
        min_survivor_floor=0,
        max_ood_risk=0.0,
        max_critic_divergence=0.0,
        max_planner_calls_per_env_step=0,
    )


class PhaseController:
    def __init__(self, kl_threshold: float = 0.02):
        self.exploration_steps = 0
        self.consolidation_steps = 0
        self.is_consolidation_phase = False
        self.kl_threshold = kl_threshold
        self.vram_capacity_threshold = 0.85
        self.stagnation_counter = 0

    def evaluate_phase_transition(self, vram_usage_pct: float, surprisal_rate: float, current_kl_drift: float) -> bool:
        if self.is_consolidation_phase:
            self.consolidation_steps += 1
            if current_kl_drift > self.kl_threshold or self.consolidation_steps > 2000:
                self.is_consolidation_phase, self.exploration_steps, self.consolidation_steps = False, 0, 0
                return True
            return False

        self.exploration_steps += 1

        if vram_usage_pct > self.vram_capacity_threshold:
            self.is_consolidation_phase, self.stagnation_counter = True, 0
            return True

        self.stagnation_counter = (
            self.stagnation_counter + 1 if (abs(surprisal_rate) < 1e-4 and self.exploration_steps > 500) else 0
        )

        if self.stagnation_counter > 5:
            self.is_consolidation_phase, self.stagnation_counter = True, 0
            return True

        return False


class CapabilityStage(Enum):
    STAGE_0_REPRESENTATION = 0
    STAGE_1_DYNAMICS = 1
    STAGE_2_CONTROL = 2
    STAGE_3_PLANNING = 3
    STAGE_4_MEMORY = 4


wp.config.verify_cuda = True
wp.config.use_mempool = True  # type: ignore[attr-defined]


class ActionDecoder(nn.Module):
    """Decodes latent state into action probabilities via SSM."""

    def __init__(self, dim=256, num_actions=256):
        super().__init__()
        self.dim = dim
        self.num_actions = num_actions

        self.decoder_ssm = SelectiveStateSpaceModel(dim)
        self.action_head = nn.Linear(dim, num_actions)
        self.state_projector = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.Mish())

    def encode(self, context: torch.Tensor) -> torch.Tensor:
        projected = self.state_projector(context)
        return F.gumbel_softmax(projected, tau=1.0, hard=True, dim=-1)

    class TemporalVarianceScorer(nn.Module):
        def __init__(self, dim=256, heads=4):
            super().__init__()
            self.self_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
            self.variance_head = nn.Sequential(
                layer_init(nn.Linear(dim, 64)), nn.Mish(), layer_init(nn.Linear(64, 1)), nn.Sigmoid()
            )

        def forward(self, stm_sequence):
            attn_out, _ = self.self_attention(stm_sequence, stm_sequence, stm_sequence, need_weights=False)
            context_vector = attn_out.mean(dim=1)
            volatility_signal = self.variance_head(context_vector)
            return volatility_signal


class OpponentModel(nn.Module):
    """Forward dynamics model for opponent prediction."""

    def __init__(self, dim=256, heads=4):
        super().__init__()
        self.dim = dim
        self.fusion_layer = nn.Linear(dim, dim)
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.dynamics = nn.Sequential(nn.Linear(dim, 512), nn.LayerNorm(512), nn.Mish(), nn.Linear(512, dim))

    def predict_agent_action(self, visual_perception: torch.Tensor, communication_msg: torch.Tensor) -> torch.Tensor:

        # [B, Seq, D] projection requirement for nn.MultiheadAttention
        visual_perception = visual_perception.unsqueeze(1) if visual_perception.dim() < 3 else visual_perception
        communication_msg = communication_msg.unsqueeze(1) if communication_msg.dim() < 3 else communication_msg

        attn_out, _ = self.attention(query=visual_perception, key=communication_msg, value=communication_msg)
        return self.dynamics(attn_out.squeeze(1))

    def compute_belief_loss(
        self, actual_intent: torch.Tensor, predicted_intent: torch.Tensor, temperature: float = 0.1
    ) -> torch.Tensor:
        """
        Args:
            actual_intent (torch.Tensor): Ground truth embeddings [B, D].
            predicted_intent (torch.Tensor): Predicted embeddings [B, D].
            temperature (float): InfoNCE temperature scale.

        Returns:
            torch.Tensor: Scalar contrastive loss.
        """
        actual_norm = F.normalize(actual_intent.view(actual_intent.size(0), -1), p=2, dim=-1)  # [B, D]
        predicted_norm = F.normalize(predicted_intent.view(predicted_intent.size(0), -1), p=2, dim=-1)  # [B, D]

        logits = torch.matmul(predicted_norm, actual_norm.t()) / temperature  # [Batch, Batch]
        labels = torch.arange(logits.size(0), device=logits.device)  # [Batch]
        return F.cross_entropy(logits, labels)

    def state_fusion(self, fused_state: torch.Tensor) -> torch.Tensor:
        """Projects aggregated opponent states into the local agent's latent space."""
        return self.fusion_layer(fused_state)


class HyperparameterScheduler(nn.Module):
    """Schedules exploration and temperature hyperparameters based on an exponential moving average of performance.

    Uses discrete regimes with hysteresis to prevent scheduling oscillations.
    """

    def __init__(self):
        super().__init__()
        self.exploration_factor = nn.Parameter(torch.tensor(0.3))
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.causal_weight = nn.Parameter(torch.tensor(1.0))

        self.register_buffer("training_phase", torch.tensor(0, dtype=torch.long))
        self.register_buffer("performance_ema", torch.tensor(0.0))

    def evaluate_regime(self, current_performance: float):
        self.performance_ema.mul_(0.95).add_(current_performance * 0.05)
        perf = self.performance_ema.item()

        current_regime = self.training_phase.item()
        if current_regime == 0 and perf > 0.5:
            self.training_phase.fill_(1)
        elif current_regime == 1:
            if perf > 1.2:
                self.training_phase.fill_(2)
            elif perf < 0.3:
                self.training_phase.fill_(0)
        elif current_regime == 2 and perf < 0.9:
            self.training_phase.fill_(1)

    def forward(self) -> Dict[str, torch.Tensor]:
        perf = self.performance_ema.item()

        target_temp = max(0.05, 2.0 * math.exp(-perf * 1.5))
        target_exp = max(0.1, 0.8 * math.exp(-perf * 2.0))

        return {
            "exploration": torch.tensor(target_exp, device=self.temperature.device),
            "temperature": torch.tensor(target_temp, device=self.temperature.device),
            "causal_weight": F.softplus(self.causal_weight),
        }


class QuantizedLinear8bit(nn.Module):
    """8-bit linear layer wrapper using bitsandbytes.

    Reduces memory footprint during training by keeping weights in INT8.

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        # Threshold set to 6.0 as recommended for outlier extraction in LLM.int8() paper (Dettmers et al., 2022).
        self.layer = bnb.nn.Linear8bitLt(in_features, out_features, bias=True, has_fp16_weights=False, threshold=6.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast input to FP16 to match the expected compute type of bnb.nn.Linear8bitLt.
        return self.layer(x.to(torch.float16))


class StochasticPolicy(nn.Module):
    def __init__(self, num_actions: int, target_entropy: Optional[float] = None):
        super().__init__()
        self.num_actions = num_actions

        if target_entropy is None:
            self.target_entropy = -math.log(1.0 / num_actions) * 0.98
        else:
            self.target_entropy = target_entropy

        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.entropy_ema: torch.Tensor
        self.register_buffer("entropy_ema", torch.tensor(self.target_entropy))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_alpha.detach().float()).clamp(min=0.01, max=10.0)

    def step_temperature(self, current_fitness_variance: float = 1.0) -> bool:
        """Checks if policy entropy dropped below target."""
        return bool(self.entropy_ema.item() < self.target_entropy)

    def forward(
        self,
        action_logits: torch.Tensor,
        master_temperature: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        immediate_logits = action_logits[..., : self.num_actions]

        if deterministic:
            return torch.argmax(immediate_logits, dim=-1), None

        max_logits = torch.max(immediate_logits, dim=-1, keepdim=True)[0]
        stable_logits = immediate_logits - max_logits.detach()

        lsh_density_inv = 0.0
        if hasattr(self, "lsh_match_density"):
            safe_density = torch.clamp(self.lsh_match_density, min=1.0)
            lsh_density_inv = torch.log1p(1.0 / safe_density).mean().item()

        local_temperature = torch.exp(self.log_alpha.detach().float()).clamp(min=0.01, max=10.0) + lsh_density_inv

        is_master_controlled = master_temperature is not None
        if master_temperature is not None:
            effective_temperature = master_temperature.float()
        else:
            effective_temperature = local_temperature

        effective_temperature = torch.clamp(effective_temperature.to(self.log_alpha.dtype), min=0.05, max=10.0)
        temperature_scaled_logits = stable_logits / effective_temperature

        safe_logits = torch.clamp(temperature_scaled_logits.float(), min=-80.0, max=80.0)

        dist = torch.distributions.Categorical(logits=safe_logits)
        action = dist.sample()

        raw_log_prob = dist.log_prob(action)
        log_prob = torch.clamp(raw_log_prob, min=-20.0, max=0.0)

        if self.training and not is_master_controlled:
            current_entropy = dist.entropy().mean()
            self.entropy_ema = 0.99 * self.entropy_ema + 0.01 * current_entropy.detach()
            self._cached_entropy_diff = current_entropy.detach() - self.target_entropy
        else:
            self._cached_entropy_diff = torch.tensor(0.0, device=self.log_alpha.device)

        return action, log_prob

    @property
    def last_alpha_loss(self) -> torch.Tensor:
        if hasattr(self, "_cached_entropy_diff"):
            return -(self.log_alpha * self._cached_entropy_diff).mean()
        return (self.log_alpha * 0.0).sum()


class UncertaintyEstimator(nn.Module):
    def __init__(self, dim=256, heads=4):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.uncertainty_scorer = nn.Sequential(
            layer_init(nn.Linear(dim, 64)), nn.Mish(), layer_init(nn.Linear(64, 1)), nn.Sigmoid()
        )

    def forward(self, stm_sequence):
        attn_out, _ = self.self_attention(stm_sequence, stm_sequence, stm_sequence)
        context_vector = attn_out.mean(dim=1)
        uncertainty_signal = self.uncertainty_scorer(context_vector)
        return uncertainty_signal


class CrossModalAttention(nn.Module):
    def __init__(
        self,
        input_dims={"sensory": 256, "audio": 256, "proprioception": 32, "text": 256},
        latent_dim=256,
        num_latents=1,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.modality_grns = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(dim, dim), nn.Mish(), nn.LayerNorm(dim))
                for name, dim in input_dims.items()
            }
        )

        self.modality_projections = nn.ModuleDict(
            {name: nn.Linear(dim, latent_dim) for name, dim in input_dims.items()}
        )

        self.query_proj = nn.Linear(latent_dim, latent_dim)
        self.query_rnn = nn.GRUCell(latent_dim, latent_dim)
        self.register_buffer("previous_query", torch.zeros(1, latent_dim))
        self.workspace_attention = ThresholdAttentionOptimized(dim=latent_dim, heads=4)

        self.align_net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.Mish(), nn.LayerNorm(latent_dim), nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuses modalities into latent bottleneck."""
        processed_inputs = []
        batch_size = None

        for key, tensor in inputs.items():
            if key not in self.modality_projections:
                continue
            if tensor.dim() == 3:
                tensor = tensor.mean(dim=1)
            if batch_size is None:
                batch_size = tensor.size(0)

            grn_tensor = self.modality_grns[key](tensor)
            proj_tensor = self.modality_projections[key](grn_tensor)
            processed_inputs.append(FrequencyDomainBinding.normalize_spherical(proj_tensor))

        stacked_modalities = torch.stack(processed_inputs, dim=1)  # [B, num_modalities, latent_dim]
        batch_size = stacked_modalities.size(0)

        prior_expanded = self.previous_query.expand(batch_size, -1)  # [B, latent_dim]
        projected_state = self.query_proj(prior_expanded)
        batched_queries = projected_state.unsqueeze(1).expand(-1, 16, -1)

        workspace_tokens = self.workspace_attention(
            query=batched_queries, key=stacked_modalities, value=stacked_modalities
        )
        fused_context = workspace_tokens.mean(dim=1)  # [B, latent_dim]
        aligned_context = self.align_net(fused_context)

        return FrequencyDomainBinding.normalize_spherical(aligned_context)


class ActionMasker(nn.Module):
    """Projects policy logits onto a safe manifold defined by Control Barrier Functions (CBF)."""

    def __init__(self, num_actions: int = 8, dim: int = 256):
        super().__init__()
        self.num_actions = num_actions
        self.dim = dim

        self.f_net = nn.Sequential(nn.Linear(dim, 256), nn.Mish(), nn.Linear(256, dim))
        self.g_net = nn.Linear(dim, dim * num_actions)

        from torch.nn.utils.parametrizations import spectral_norm

        self.constraint_net = nn.Sequential(
            spectral_norm(nn.Linear(dim, 128)), nn.Mish(), spectral_norm(nn.Linear(128, 1))
        )

        self.logit_alpha = nn.Parameter(torch.zeros(1))

    @jaxtyped(typechecker=beartype)
    def forward(self, state_latent: LatentState, logits: ActionLogits) -> ActionLogits:
        batch_size = state_latent.size(0)
        u_nom = F.softmax(logits, dim=-1)

        f_x = self.f_net(state_latent)  # [B, D]
        g_x = self.g_net(state_latent).view(batch_size, self.dim, self.num_actions)  # [B, D, num_actions]

        with torch.enable_grad():
            state_attached = state_latent if state_latent.requires_grad else state_latent.clone().requires_grad_(True)
            h_val = self.constraint_net(state_attached)  # [B, 1]
            grad_outputs = torch.ones_like(h_val)

            # Compute Jacobian of constraint boundary function w.r.t state.
            dh_dx = torch.autograd.grad(
                outputs=h_val, inputs=state_attached, grad_outputs=grad_outputs, create_graph=True, retain_graph=True
            )[0].contiguous()

        Lf_h = torch.sum(dh_dx * f_x, dim=-1, keepdim=True)  # [B, 1]
        Lg_h = torch.sum(dh_dx.unsqueeze(-1) * g_x, dim=1)  # [B, num_actions]

        # Learnable slack variable determining Control Barrier Function boundary strictness
        alpha = 0.05 + 0.95 * torch.sigmoid(self.logit_alpha)
        h_x = self.constraint_net(state_latent)

        b_cbf = Lf_h + alpha * h_x
        A_cbf = -Lg_h

        constraint_violation = torch.bmm(A_cbf.unsqueeze(1), u_nom.unsqueeze(2)).squeeze(-1) - b_cbf

        A_norm_raw = torch.sum(A_cbf**2, dim=-1, keepdim=True)
        A_norm_sq_safe = F.softplus(A_norm_raw - 1e-3) + 1e-3
        lambda_lagrange = F.relu(constraint_violation / A_norm_sq_safe)

        _ = torch.std(logits, dim=-1, keepdim=True) + 1e-5

        with torch.autocast(device_type=state_latent.device.type, enabled=False):
            state_f32 = state_latent.float()
            t_comp = state_f32[:, 0:1]
            s_comp = state_f32[:, 1:]
            metric_norm = torch.clamp(
                torch.sqrt(torch.abs(torch.sum(s_comp * s_comp, dim=-1, keepdim=True) - t_comp * t_comp) + 1e-5),
                min=1.0,
            )
        metric_norm = metric_norm.to(state_latent.dtype)
        max_penalty = torch.clamp(
            torch.max(logits.detach(), dim=-1, keepdim=True)[0] - torch.min(logits.detach(), dim=-1, keepdim=True)[0],
            max=15.0,
        )

        lagrange_penalty = torch.tanh(lambda_lagrange * A_cbf) * (max_penalty * metric_norm)
        lagrange_penalty = torch.clamp(lagrange_penalty, min=-max_penalty, max=max_penalty)

        penalty_scale = torch.sigmoid(-lagrange_penalty)

        safe_logits = logits * penalty_scale
        logits = F.log_softmax(safe_logits, dim=-1)

        return logits if state_latent.requires_grad else logits.detach()
