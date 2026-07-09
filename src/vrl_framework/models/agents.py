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

from __future__ import annotations

import copy
import gc
import logging
import math
import os
import random
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from safetensors.torch import save_file
from torch.utils.checkpoint import checkpoint

from vrl_framework.core.contracts import (
    PlanningBudget,
    PolicySequenceBatch,
    RecurrentStateSnapshot,
    SlotState,
    beartype,
    jaxtyped,
)
from vrl_framework.core.settings import BATCH_SIZE, CFG, MODEL_DEVICE, MOE_CFG
from vrl_framework.environment.replay_buffer import (
    EpisodicReplayBuffer,
    MemoryEntityDescriptor,
    MemoryQualityBand,
    MemoryTierState,
)
from vrl_framework.math_ops.geometry import LorentzGeometry, compute_traj_entropy
from vrl_framework.models.components import (
    ActorCriticModule,
    CommunicationModule,
    CPUOffloadedEMA,
    DenseAssociativeMemory,
    DualPathSparseAutoencoder,
    DualStreamJEPACore,
    FrequencyDomainBinding,
    GradientShortTermMemory,
    HebbianLinear,
    IdentityModule,
    InterventionalCausalEngine,
    IntrinsicMotivationModule,
    LatentQuantizer,
    ModelPredictivePlanner,
    MultimodalSensoryHub,
    RepresentationGate,
    SharedWorkspaceBottleneck,
    StabilityGate,
    StochasticPolicy,
    ThresholdAttentionOptimized,
    TopKActivation,
    UnifiedGatedLinearBackbone,
    custom_load_state_dict,
    layer_init,
)
from vrl_framework.models.planners import (
    ActionMasker,
    CrossModalAttention,
    HaltingHead,
    MCTSPlanner,
    OpponentModel,
    UncertaintyEstimator,
)

logger = logging.getLogger(__name__)


class HighLevelPolicy(nn.Module):
    """HIRO high-level manager policy with off-policy target correction."""

    def __init__(
        self,
        input_dim: int,
        goal_dim: int = 256,
        update_freq: int = 16,
        goal_scale: float = 5.0,
        internal_ticks: int = 5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.base_update_freq = update_freq
        self.internal_ticks = internal_ticks
        self.workspace = SharedWorkspaceBottleneck(goal_dim, 4)
        self.register_buffer("internal_state", torch.zeros(1, goal_dim))
        self._update_freq = update_freq
        self.goal_dim = goal_dim
        self.goal_scale = goal_scale

        self.manager_actor = nn.Sequential(
            layer_init(nn.Linear(input_dim, 512)),
            nn.Mish(),
            nn.LayerNorm(512),
            layer_init(nn.Linear(512, 512)),
            nn.Mish(),
            layer_init(nn.Linear(512, self.goal_dim), std=0.01),
        )

        self.worker_state_proj = HebbianLinear(input_dim, 256, sparsity=0.3)
        self.worker_goal_proj = HebbianLinear(self.goal_dim, 256, sparsity=0.3)

        self.worker_fusion = nn.Sequential(
            HebbianLinear(512, 512, sparsity=0.5),
            nn.Mish(),
            nn.LayerNorm(512),
            TopKActivation(sparsity_k=128),
            nn.Linear(512, input_dim),
        )

        self.register_buffer("active_subgoal", torch.zeros(1, self.goal_dim))
        self.register_buffer("tick_counter", torch.tensor(0, dtype=torch.long))

    @property
    def update_freq(self):
        gen = (
            getattr(wandb.run.summary, "get", lambda k, d: d)("generation", 0)
            if getattr(wandb, "run", None) is not None
            else 0
        )
        if gen > 50000:
            return self.base_update_freq * 16
        elif gen > 10000:
            return self.base_update_freq * 4
        return self.base_update_freq

    def get_manager_goal(self, state: torch.Tensor, add_noise: bool = False) -> torch.Tensor:
        raw_goal = self.manager_actor(state)

        if add_noise:
            raw_goal = raw_goal + torch.randn_like(raw_goal) * 0.1

        standard_goal = F.normalize(raw_goal, p=2, dim=-1) * self.goal_scale
        return standard_goal

    def forward(self, state: torch.Tensor, external_goal: Optional[torch.Tensor] = None) -> torch.Tensor:
        if external_goal is not None:
            self.active_subgoal = external_goal
        else:
            self.active_subgoal = self.get_manager_goal(state, add_noise=self.training)

        s_proj = F.mish(self.worker_state_proj(state))

        g_expanded = self.active_subgoal.expand(state.size(0), -1)

        g_proj = F.mish(self.worker_goal_proj(g_expanded))

        worker_input = torch.cat([s_proj, g_proj], dim=-1)
        return self.worker_fusion(worker_input)

    def calculate_intrinsic_reward(
        self, state: torch.Tensor, next_state: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            target_state_ambient = state + goal
            target_state_ambient = 10.0 * torch.tanh(target_state_ambient / 10.0)
            next_state_safe = 10.0 * torch.tanh(next_state / 10.0)
            target_hyp = LorentzGeometry.project(target_state_ambient)
            next_state_hyp = LorentzGeometry.project(next_state_safe)

            dist_penalty = -LorentzGeometry.distance(target_hyp, next_state_hyp)
            return dist_penalty

    def hiro_off_policy_correction(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        actions: torch.Tensor,
        worker_actor_critic: nn.Module,
        num_candidates: int = 8,
        causal_masker: Optional[Any] = None,
    ) -> torch.Tensor:
        """HIRO off-policy target correction on the Lorentz manifold."""
        batch_size, dim = states.shape

        with torch.no_grad():
            # Map Euclidean delta to Riemannian tangent space
            m_dot = -LorentzGeometry.minkowski_dot(states, next_states, keepdim=True)
            safe_dot = torch.clamp(m_dot, min=1.005, max=65000.0)
            geodesic_dist = torch.acosh(safe_dot)
            tangent_direction = next_states - safe_dot * states

            tangent_norm = (
                LorentzGeometry.minkowski_dot(tangent_direction, tangent_direction, keepdim=True)
                .clamp_min(1e-4)
                .sqrt()
                .clamp_min(1e-4)
            )
            achieved_goals = (geodesic_dist * tangent_direction) / tangent_norm
            base_goals = F.normalize(achieved_goals, p=2, dim=-1) * self.goal_scale

            candidates = base_goals.unsqueeze(1).expand(-1, num_candidates, -1).clone()  # [B, K, D]

            noise = torch.randn_like(candidates) * (self.goal_scale * 0.1)
            candidates[:, 1:, :] += noise[:, 1:, :]
            candidates = F.normalize(candidates, p=2, dim=-1) * self.goal_scale

            flat_states = states.repeat_interleave(num_candidates, dim=0)  # [B * K, D]
            flat_candidates = candidates.reshape(batch_size * num_candidates, -1)  # [B * K, D]
            flat_actions = actions.repeat_interleave(num_candidates, dim=0)

            s_proj = F.mish(self.worker_state_proj(flat_states.float()))
            g_proj = F.mish(self.worker_goal_proj(flat_candidates.float()))

            worker_input = self.worker_fusion(torch.cat([s_proj, g_proj], dim=-1))

            fuzzy_kb = getattr(worker_actor_critic, "fuzzy_kb", None)
            if fuzzy_kb is not None:
                truth_scores = fuzzy_kb.evaluate_truth_gate_differentiable(flat_candidates)
                invalid_goals_mask = truth_scores < 0.0
            else:
                invalid_goals_mask = states.new_zeros(batch_size * num_candidates, dtype=torch.bool)

            active_critic_ctx = getattr(
                self, "last_critic_context", torch.zeros(batch_size, 768, device=states.device)
            )
            active_intent_ctx = getattr(
                self, "last_intent_context", torch.zeros(batch_size, 256, device=states.device)
            )

            expanded_critic_ctx = active_critic_ctx.repeat_interleave(num_candidates, dim=0)
            expanded_intent_ctx = active_intent_ctx.repeat_interleave(num_candidates, dim=0)

            ac_out = worker_actor_critic(worker_input, expanded_critic_ctx, intent_context=expanded_intent_ctx)
            policy_logits = ac_out.policy_logits
            q_values = ac_out.pessimistic_value.view(batch_size, num_candidates)

            q_values.masked_fill_(invalid_goals_mask.view(batch_size, num_candidates), -1e4)

            immediate_logits = (
                policy_logits[:, 0, : worker_actor_critic.num_actions]
                if policy_logits.dim() == 3
                else policy_logits[..., : worker_actor_critic.num_actions]
            )

            if causal_masker is not None:
                immediate_logits = causal_masker(worker_input, immediate_logits)

            dist = torch.distributions.Categorical(logits=immediate_logits)

            safe_flat_actions = torch.clamp(flat_actions.long(), 0, immediate_logits.size(-1) - 1)
            log_probs = dist.log_prob(safe_flat_actions).view(batch_size, num_candidates)

            combined_score = log_probs + 0.1 * q_values
            best_candidate_idx = torch.argmax(combined_score, dim=-1)

            best_goals = candidates[torch.arange(batch_size, device=states.device), best_candidate_idx]

        return best_goals


class AdversarialModulationRNN(nn.Module):
    """Generates norm-constrained FiLM parameters (gamma, beta) for regularization."""

    adversary_state: torch.Tensor

    def __init__(self, dim=256, hidden_dim=64):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.adversary_cell = nn.GRUCell(dim, hidden_dim)
        self.film_generator = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, 128)), nn.Mish(), layer_init(nn.Linear(128, dim * 2))
        )

        self.register_buffer("adversary_state", torch.zeros(1, hidden_dim))

    def forward(self, agent_representation: torch.Tensor) -> torch.Tensor:
        batch_size = agent_representation.size(0)
        if self.adversary_state.size(0) != batch_size:
            self.adversary_state = agent_representation.new_zeros(batch_size, self.hidden_dim)

        self.adversary_state = self.adversary_cell(agent_representation.detach(), self.adversary_state.detach())
        film_params = self.film_generator(self.adversary_state)

        gamma, beta = film_params.chunk(2, dim=-1)

        orthogonal_noise = torch.randn_like(gamma) * 0.02
        safe_gamma = 1.0 + torch.tanh(gamma + orthogonal_noise) * 0.15
        safe_beta = torch.tanh(beta + orthogonal_noise) * 0.15

        self.last_gamma = safe_gamma
        self.last_beta = safe_beta

        perturbed_vector = (agent_representation * safe_gamma) + safe_beta
        delta_vector = perturbed_vector - agent_representation

        l2_norm = torch.sqrt(torch.sum(delta_vector.float() ** 2, dim=-1, keepdim=True) + 1e-8).to(delta_vector.dtype)
        smooth_barrier_scale = torch.where(
            l2_norm > 1.5, 1.5 / l2_norm, torch.tensor(1.0, dtype=l2_norm.dtype, device=l2_norm.device)
        )
        delta_vector = delta_vector * smooth_barrier_scale

        perturbed_vector = agent_representation + delta_vector

        cos_sim = F.cosine_similarity(agent_representation, perturbed_vector, dim=-1)
        collapse_mask = cos_sim < 0.85

        perturbed_vector = torch.where(collapse_mask.unsqueeze(-1), perturbed_vector.detach(), perturbed_vector)
        self.collapse_penalty = collapse_mask.float().mean() * 10.0

        return perturbed_vector

    def reset_adversary_memory(self):
        self.adversary_state.fill_(0.0)


class AdversarialOptimizationController:
    def __init__(self, dim=256, pop_size=512):
        self.lambda_penalty = 5.0
        self.eta_sparsity = 0.1
        self.optimizer = None

    def apply_budgeted_warp(
        self,
        adversary_module,
        latent_context,
        jepa_module,
        actor_critic,
        consolidation_weight=None,
        critic_context=None,
    ):

        if consolidation_weight is None or consolidation_weight.mean().item() <= 0.05:
            return latent_context

        batch_size = latent_context.size(0)
        num_perturbations = 100

        expanded_context = latent_context.repeat_interleave(num_perturbations, dim=0)
        noise = torch.randn_like(expanded_context) * 0.1

        with torch.enable_grad():
            perturbed_context = adversary_module(expanded_context.detach() + noise)

        with torch.no_grad():
            if critic_context is not None:
                active_critic = critic_context.repeat_interleave(num_perturbations, dim=0)
            else:
                active_critic = perturbed_context.new_zeros(batch_size * num_perturbations, 768)

            ac_out = actor_critic(perturbed_context, critic_context=active_critic)
            value_preds = ac_out.pessimistic_value
            value_preds = value_preds.view(batch_size, num_perturbations)

            best_indices = torch.argmin(value_preds, dim=-1)
            batch_indexer = torch.arange(batch_size, device=perturbed_context.device)

        best_perturbed = perturbed_context.view(batch_size, num_perturbations, -1)[batch_indexer, best_indices]

        return best_perturbed

    def evaluate_and_evolve(self, adversary_module, actual_agent_loss, host_latent):
        """Computes regularization terms and executes an isolated optimization step for the adversary module.

        Evaluated Loss: L_adv = TD_host - lambda*(||gamma-1||_2 + ||beta||_2) - eta*(sparsity)
        """
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(adversary_module.parameters(), lr=3e-4)

        if not hasattr(adversary_module, "last_gamma") or not hasattr(adversary_module, "last_beta"):
            return

        if adversary_module.last_gamma is None or adversary_module.last_beta is None:
            return

        gamma_dev = torch.sqrt(torch.sum((adversary_module.last_gamma - 1.0) ** 2, dim=-1) + 1e-4).mean()
        beta_dev = torch.sqrt(torch.sum(adversary_module.last_beta**2, dim=-1) + 1e-4).mean()

        sparsity_penalty = torch.sum(torch.abs(adversary_module.last_beta), dim=-1).mean()
        collapse_pen = getattr(adversary_module, "collapse_penalty", torch.tensor(0.0, device=host_latent.device))

        structural_penalty = (
            (self.lambda_penalty * (gamma_dev + beta_dev)) + (self.eta_sparsity * sparsity_penalty) + collapse_pen
        )

        structural_penalty.backward()

        valid_grads = True
        for p in adversary_module.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                valid_grads = False
                break

        if valid_grads:
            torch.nn.utils.clip_grad_norm_(adversary_module.parameters(), 1.0)
            self.optimizer.step()
        else:
            self.optimizer.zero_grad()


class RLAgent(nn.Module):
    """Integrates the multimodal sensory pipeline with actor-critic heads via a latent replay buffer.

    Provides capabilities for predictive modeling, intrinsic motivation, and latent action masking.
    """

    def get_compute_penalty(self) -> float:
        return 0.1

    @property
    def mcts_teacher_buffer_logits(self) -> torch.Tensor:
        return self.mcts_memory_bank[:, : self.num_actions]

    @property
    def expiration_step(self) -> torch.Tensor:
        return self.mcts_memory_bank[:, self.num_actions]

    @property
    def mcts_teacher_buffer_gain(self) -> torch.Tensor:
        return self.mcts_memory_bank[:, self.num_actions + 1]

    @property
    def mcts_teacher_buffer_health(self) -> torch.Tensor:
        return self.mcts_memory_bank[:, self.num_actions + 2]

    @property
    def mcts_teacher_buffer_critic_gap(self) -> torch.Tensor:
        return self.mcts_memory_bank[:, self.num_actions + 3]

    @property
    def mcts_teacher_buffer_truth_margin(self) -> torch.Tensor:
        return self.mcts_memory_bank[:, self.num_actions + 4]

    def __init__(
        self,
        sensory_input_shape: Tuple[int, ...] = (4, 7, 7, 7),
        num_actions: int = 16,
        runtime_context: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.opt_representation = None
        self.opt_policy = None
        self.opt_causal = None
        self.num_actions = num_actions
        self.sensory_input_shape = sensory_input_shape
        self.runtime_context = runtime_context

        self.sensory = UnifiedGatedLinearBackbone(embed_dim=128, output_dim=256).to(MODEL_DEVICE)

        if CFG.ENABLE_MULTIMODAL:
            self.multimodal = MultimodalSensoryHub(latent_dim=256, sample_rate=44100).to(MODEL_DEVICE)
            self.bottleneck_attention = CrossModalAttention(
                input_dims={"sensory": 256, "audio": 256, "proprioception": 32, "text": 256}
            ).to(MODEL_DEVICE)
        else:
            self.multimodal = IdentityModule().to(MODEL_DEVICE)
            self.bottleneck_attention = lambda inputs: inputs["sensory"]

        self.jepa = DualStreamJEPACore(latent_dim=256).to(MODEL_DEVICE)
        self.latent_dynamics = SuccessorLatentDynamicsModel(latent_dim=256, action_dim=self.num_actions).to(
            MODEL_DEVICE
        )

        self.memory = EpisodicReplayBuffer(dim=256, runtime_context=self.runtime_context).to(MODEL_DEVICE)
        self.gradient_stm = GradientShortTermMemory(input_dim=256, hidden_dim=256).to(MODEL_DEVICE)

        self.scratchpad = nn.Parameter(torch.zeros(1, 8, 256))
        self.scratchpad_attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True).to(
            MODEL_DEVICE
        )
        self.scratchpad_write_gate = nn.Linear(256, 8).to(MODEL_DEVICE)
        self.scratchpad_write_val = nn.Linear(256, 256).to(MODEL_DEVICE)

        class LiquidTimeCell(nn.Module):
            """Liquid time constant RNN cell."""

            def __init__(self, in_dim: int, out_dim: int) -> None:
                super().__init__()
                self.decay_net = nn.Sequential(nn.Linear(in_dim + out_dim, out_dim), nn.Sigmoid())
                self.state_net = nn.Sequential(nn.Linear(in_dim + out_dim, out_dim), nn.Tanh())
                self.tau_min = 0.01

            def forward(self, x: torch.Tensor, h: torch.Tensor, delta_t: float = 1.0) -> torch.Tensor:
                hx = torch.cat([x, h], dim=-1)
                time_constant = self.decay_net(hx) * 2.0 + self.tau_min
                target_state = self.state_net(hx)

                # Integration: h(t) = target + (h(0) - target) * exp(-delta_t / tau)
                decay_factor = torch.exp(-delta_t / time_constant)
                next_h = target_state + (h - target_state) * decay_factor
                return next_h

        self.meta_gru = LiquidTimeCell(in_dim=256 + self.num_actions + 1, out_dim=256).to(MODEL_DEVICE)
        self.ponder_gru = LiquidTimeCell(in_dim=256, out_dim=256).to(MODEL_DEVICE)
        self.ponder_norm = nn.LayerNorm(256).to(MODEL_DEVICE)

        self.fwp_q = nn.Linear(256, 64).to(MODEL_DEVICE)
        self.fwp_k = nn.Linear(256, 64).to(MODEL_DEVICE)
        self.fwp_v = nn.Linear(256, 256).to(MODEL_DEVICE)

        self.stress_to_film = nn.Linear(1, 512).to(MODEL_DEVICE)

        self.register_buffer("global_gamma", torch.ones(1, 256))
        self.register_buffer("global_beta", torch.zeros(1, 256))

        class CapacityMoE(nn.Module):
            def __init__(self, dim: int, num_experts: int = 8, capacity_factor: float = 1.25) -> None:
                super().__init__()
                self.num_experts = num_experts
                self.capacity_factor = capacity_factor

                self.expert_centroids = nn.Parameter(torch.randn(num_experts, dim))
                nn.init.orthogonal_(self.expert_centroids)
                self.temperature = nn.Parameter(torch.tensor(0.1))

                self.experts = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(dim, dim * 4), nn.LayerNorm(dim * 4), nn.Mish(), nn.Linear(dim * 4, dim)
                        )
                        for _ in range(num_experts)
                    ]
                )
                self.register_buffer("step_counter", torch.tensor(0.0))
                self.register_buffer("expert_usage_ema", torch.ones(num_experts) / num_experts)
                self.expert_orthogonality = 0.0

            def forward(self, x: torch.Tensor, error_signal: Optional[torch.Tensor] = None) -> torch.Tensor:
                orig_dtype = x.dtype
                x = x.to(self.expert_centroids.dtype)

                batch_size = x.size(0)
                capacity = max(1, int(round((batch_size / self.num_experts) * self.capacity_factor)))

                x_norm = F.normalize(x, p=2, dim=-1)

                if self.training and self.expert_centroids.requires_grad:
                    noisy_centroids = self.expert_centroids + torch.randn_like(self.expert_centroids) * 0.001
                    centroids_norm = F.normalize(noisy_centroids, p=2, dim=-1)
                else:
                    centroids_norm = F.normalize(self.expert_centroids, p=2, dim=-1)

                router_logits = F.linear(x_norm, centroids_norm) / torch.clamp(self.temperature, min=0.2)
                if self.training:
                    router_logits = router_logits + torch.randn_like(router_logits) * 0.15

                scores, indices = torch.topk(router_logits, k=2, dim=-1)
                weights = F.softmax(scores, dim=-1)

                if self.training:
                    full_probs = F.softmax(router_logits, dim=-1)
                    expert_usage = full_probs.mean(dim=0)
                    if self.training:
                        self.expert_usage_ema.lerp_(expert_usage.detach(), 0.01)

                    if wandb.run is not None:
                        try:
                            from vrl_framework.trainer.ppo_engine import metrics_aggregator

                            metrics_aggregator.log(
                                {
                                    "moe/routing_entropy": -(full_probs * torch.log(full_probs + 1e-4))
                                    .sum(dim=-1)
                                    .mean(),
                                    "moe/expert_usage_variance": self.expert_usage_ema.var(),
                                    "moe/expert_orthogonality": self.expert_orthogonality,
                                }
                            )
                        except Exception:
                            pass

                    if x.requires_grad and error_signal is not None:
                        norm_error = (error_signal - error_signal.mean()) / (error_signal.std() + 1e-4)
                        norm_error = norm_error.to(full_probs.dtype)
                        credit_assignment = (full_probs * norm_error).mean(dim=0).detach()
                        balance_penalty = (expert_usage - 1.0 / self.num_experts).detach()

                        def router_hook(grad: torch.Tensor) -> torch.Tensor:
                            safe_g = torch.nan_to_num(grad, nan=0.0)
                            return safe_g + (
                                (balance_penalty.unsqueeze(0) * 0.01)
                                + (credit_assignment.unsqueeze(0) * 0.15)
                                + torch.randn_like(safe_g) * 0.05
                            ) * safe_g.abs().mean().clamp(min=1e-4)

                        router_logits.register_hook(router_hook)
                    elif x.requires_grad:
                        balance_penalty = (expert_usage - 1.0 / self.num_experts).detach()

                        def router_hook(grad: torch.Tensor) -> torch.Tensor:
                            safe_g = torch.nan_to_num(grad, nan=0.0)
                            return safe_g + (
                                (balance_penalty.unsqueeze(0) * 0.02) + torch.randn_like(safe_g) * 0.05
                            ) * safe_g.abs().mean().clamp(min=1e-4)

                        router_logits.register_hook(router_hook)

                out = torch.zeros_like(x)
                for k in range(2):
                    k_idx = indices[:, k]
                    k_weights = weights[:, k].unsqueeze(-1)
                    curr_cap = min(capacity, x.size(0)) if self.training else x.size(0)

                    for i, expert in enumerate(self.experts):
                        mask = k_idx == i
                        priority = torch.where(
                            mask, scores[:, k], torch.tensor(-float("inf"), device=scores.device, dtype=scores.dtype)
                        )

                        top_scores, sampled_indices = torch.topk(priority, curr_cap)
                        valid_sample_mask = (top_scores > -float("inf")).to(x.dtype).unsqueeze(-1)

                        sampled_inputs = x[sampled_indices]
                        expert_out = expert(sampled_inputs)

                        valid_out = (expert_out * k_weights[sampled_indices] * valid_sample_mask).to(out.dtype)
                        out.scatter_add_(0, sampled_indices.unsqueeze(-1).expand(-1, out.size(-1)), valid_out)

                return out.to(orig_dtype)

        self.moe: nn.Module
        if CFG.USE_MOE:
            self.moe = CapacityMoE(dim=256).to(MODEL_DEVICE)
        else:
            self.moe = nn.Sequential(layer_init(nn.Linear(256, 1024)), nn.Mish(), layer_init(nn.Linear(1024, 256))).to(
                MODEL_DEVICE
            )

        self.meta_inference = UncertaintyEstimator(dim=256).to(MODEL_DEVICE)
        self.halting_head = HaltingHead(dim=256).to(MODEL_DEVICE)
        self.dnc_query_generator = nn.Linear(256, 256).to(MODEL_DEVICE)
        self.dnc_cross_attention = ThresholdAttentionOptimized(dim=256, heads=4).to(MODEL_DEVICE)
        self.hypothesis_generator = ModelPredictivePlanner(
            num_actions=self.num_actions, latent_dim=256, k_samples=5
        ).to(MODEL_DEVICE)
        self.causal_masker = ActionMasker(self.num_actions).to(MODEL_DEVICE)
        self.interventional_causal_engine = InterventionalCausalEngine(dim=256).to(MODEL_DEVICE)
        from vrl_framework.trainer.ppo_engine import CausalIntegrityValidator

        self.causal_validator = CausalIntegrityValidator(causal_reasoner_ref=None).to(MODEL_DEVICE)

        self.fuzzy_kb = DenseAssociativeMemory(dim=256, num_predicates=4).to(MODEL_DEVICE)

        self.causal_symbolic_reasoner = GatedCausalReasoner(num_slots=8, slot_dim=32, action_dim=self.num_actions).to(
            MODEL_DEVICE
        )

        self.sae = DualPathSparseAutoencoder(d_model=256, dict_size=4096, k=15, residual_dim=16).to(MODEL_DEVICE)
        self.latent_mcts = MCTSPlanner(num_actions=self.num_actions, latent_dim=256).to(MODEL_DEVICE)
        self.action_think_idx = self.num_actions - 1
        self.register_buffer("mcts_buffer_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("mcts_teacher_buffer_states", torch.zeros(1024, 256))
        self.register_buffer("mcts_memory_bank", torch.zeros(1024, self.num_actions + 6))
        self.register_buffer("cumulative_surprisal", torch.zeros(1))

        self.lora_registry: dict[torch.Tensor, torch.Tensor] = {}
        from vrl_framework.core.settings import AGENTS_DIR as LORA_NVME_DIR

        self.lora_nvme_dir = LORA_NVME_DIR

        self._knowledge_streamer = None

        self.communication = CommunicationModule(input_dim=256, comm_dim=256).to(MODEL_DEVICE)
        self.lpm_module = IntrinsicMotivationModule(input_dim=256, action_dim=self.num_actions, hidden_dim=512).to(
            MODEL_DEVICE
        )
        self.register_buffer("last_jepa_error", torch.zeros(1))

        if CFG.ENABLE_MULTIMODAL:
            self.opponent_model = OpponentModel(dim=256, heads=4).to(MODEL_DEVICE)
        else:
            self.opponent_model = IdentityModule().to(MODEL_DEVICE)

        self.adversary_module = AdversarialModulationRNN(dim=256).to(MODEL_DEVICE)
        self.adversary_controller = AdversarialOptimizationController(dim=256, pop_size=1024)

        self.actor_critic = ActorCriticModule(input_dim=256, num_actions=self.num_actions).to(MODEL_DEVICE)

        if hasattr(torch, "compile"):
            if hasattr(self.sensory, "pcn_layer"):
                self.sensory.pcn_layer = torch.compile(self.sensory.pcn_layer, mode="reduce-overhead")
            if hasattr(self.jepa, "fp16_encoder"):
                self.jepa.fp16_encoder = torch.compile(self.jepa.fp16_encoder, mode="reduce-overhead")
            if hasattr(self.jepa, "target_encoder"):
                self.jepa.target_encoder = torch.compile(self.jepa.target_encoder, mode="reduce-overhead")

        from torch.nn.utils import spectral_norm

        def _apply_sn_safely(m: nn.Module) -> None:
            try:
                if hasattr(m, "parametrizations") and "weight" in m.parametrizations:
                    return
                if hasattr(m, "weight_orig"):
                    return
                spectral_norm(m)
            except Exception:
                import logging

                logging.getLogger(__name__).exception("Failed to apply spectral_norm")

        for name, module in self.latent_dynamics.named_modules():
            if isinstance(module, nn.Linear):
                _apply_sn_safely(module)

        if hasattr(self.actor_critic, "value_head"):
            if isinstance(self.actor_critic.value_head, nn.Linear):
                _apply_sn_safely(self.actor_critic.value_head)
        elif hasattr(self.actor_critic, "critic_1") and isinstance(self.actor_critic.critic_1, nn.Sequential):
            _apply_sn_safely(self.actor_critic.critic_1[-1])
            _apply_sn_safely(self.actor_critic.critic_2[-1])

        self.primary_objective_grad: torch.Tensor

        def _soft_cosine_gating_hook(module: nn.Module, grad_input: Any, grad_output: Any) -> Any:
            """Projects conflicting gradients via soft cosine gating."""
            if not grad_input or grad_input[0] is None:
                return None

            current_grad: torch.Tensor = grad_input[0]
            if not torch.isfinite(current_grad).all():
                return None

            orig_dtype = current_grad.dtype

            if (
                getattr(self, "primary_objective_grad", None) is None
                or getattr(self.primary_objective_grad, "numel", lambda: 0)() < 1
            ):
                self.primary_objective_grad = current_grad.detach().clone()
                return None

            if current_grad.shape == self.primary_objective_grad.shape:
                curr_g_f32 = current_grad.float()
                prim_g_f32 = self.primary_objective_grad.float()

                if not torch.isfinite(prim_g_f32).all():
                    self.primary_objective_grad = current_grad.detach().clone()
                    return None

                sim = F.cosine_similarity(curr_g_f32.view(-1), prim_g_f32.view(-1), dim=0)
                mask = (sim < 0).float()
                dot_product = torch.dot(curr_g_f32.view(-1), prim_g_f32.view(-1))
                norm_sq = torch.dot(prim_g_f32.view(-1), prim_g_f32.view(-1)) + 1e-4
                projection = (dot_product / norm_sq) * prim_g_f32
                curr_g_f32 = curr_g_f32 - (projection * mask)

                self.primary_objective_grad.mul_(0.9).add_(current_grad.detach().float(), alpha=0.1)

                new_grad = torch.clamp(curr_g_f32, min=-65000.0, max=65000.0).to(orig_dtype)
                return (new_grad,) + grad_input[1:]

            return None

        for head in self.jepa.predictor:
            head[-1].register_full_backward_hook(_soft_cosine_gating_hook)
        self.num_perturbations = 512
        self.register_buffer(
            "perturbation_vectors",
            torch.randn(self.num_perturbations, 256, dtype=torch.float16, device=MODEL_DEVICE) * 0.02,
        )
        self.register_buffer(
            "perturbation_scores", torch.zeros(self.num_perturbations, dtype=torch.float16, device=MODEL_DEVICE)
        )
        self.register_buffer("target_network_drift", torch.zeros(1, 256, dtype=torch.float16, device=MODEL_DEVICE))

        self._actor_critic_base_forward = self.actor_critic.forward

        def _forward_with_es_drift(
            actor_context: torch.Tensor,
            critic_context: Optional[torch.Tensor] = None,
            intent_context: Optional[torch.Tensor] = None,
            dynamics_model: Optional[Any] = None,
        ) -> Any:
            return self._actor_critic_base_forward(
                actor_context + self.target_network_drift, critic_context, intent_context, dynamics_model
            )

        self.actor_critic.forward_with_perturbations = _forward_with_es_drift

        self.actor_critic_ema = CPUOffloadedEMA(self.actor_critic, decay=0.999)

        self.hierarchical_planner = HighLevelPolicy(input_dim=256)
        self.exploration_layer = StochasticPolicy(num_actions=self.num_actions)

        self.cluster_layer = LatentQuantizer(dict_size=4096, latent_dim=256).to(MODEL_DEVICE)

        self.manager_goal: torch.Tensor
        self.previous_memory_context: torch.Tensor
        self.step_counter: torch.Tensor
        self.register_buffer("manager_goal", torch.zeros(1, 256))
        self.register_buffer("previous_memory_context", torch.zeros(1, 256))
        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.long))

        self.ponder_action_idx = self.num_actions - 1
        self.rollout_generator = ModelPredictivePlanner(num_actions=self.num_actions, latent_dim=256).to(MODEL_DEVICE)

        self.hidden_units = [self.sensory.pcn_layer] if getattr(self.sensory, "pcn_layer", None) is not None else []
        self.weights: list[Any] = []
        from vrl_framework.trainer.ppo_engine import PPOTrainer

        self.trainer = PPOTrainer(self, runtime_context=None)

        # Pre-allocate STM tensors and linear attention traces
        # to prevent reallocation and graph recompilation overhead.
        self.stm_buffer: list[torch.Tensor] = []
        self.stm_tensor: torch.Tensor
        self.stm_ptr: torch.Tensor
        self.lsh_hash_planes: torch.Tensor
        self.register_buffer("stm_tensor", torch.zeros(128, 1024, 256))
        self.register_buffer("stm_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("lsh_hash_planes", torch.randn(256, 8, dtype=torch.float16))
        self.register_buffer("primary_objective_grad", torch.zeros(256))

        self.s_hebb_trace: torch.Tensor
        self.z_hebb_trace: torch.Tensor
        self.continuous_stream: torch.Tensor
        self.register_buffer("s_hebb_trace", torch.zeros(128, 64, 256))
        self.register_buffer("z_hebb_trace", torch.zeros(128, 64))
        self.register_buffer("continuous_stream", torch.zeros(1, 256))

        self.prev_hold_out_loss = 0.0
        self.text_embedding = None
        self.deq_convergence_steps = torch.zeros(1, device=MODEL_DEVICE)
        self.cognitive_fatigue: torch.Tensor
        self.last_meta_state: torch.Tensor
        self.last_action_one_hot: torch.Tensor
        self.last_reward: torch.Tensor
        self.register_buffer("cognitive_fatigue", torch.zeros(1))

    def update_noise(
        self,
        actor_context: torch.Tensor,
        critic_context: Optional[torch.Tensor] = None,
        intent_context: Optional[torch.Tensor] = None,
    ) -> None:
        """Executes evolutionary mutation on exploration perturbation vectors."""
        with torch.no_grad():
            base_latent = actor_context.mean(dim=1, keepdim=False) if actor_context.dim() > 2 else actor_context
            expanded_latents = base_latent.expand(self.num_perturbations, -1)
            perturbed_latents = expanded_latents.add(self.perturbation_vectors)

            target_device = actor_context.device
            safe_critic = (
                critic_context
                if critic_context is not None
                else torch.zeros(1, 1024, device=target_device, dtype=actor_context.dtype)
            )
            safe_intent = (
                intent_context
                if intent_context is not None
                else torch.zeros(1, 256, device=target_device, dtype=actor_context.dtype)
            )

            base_critic = safe_critic.mean(dim=0, keepdim=True) if safe_critic.dim() > 1 else safe_critic
            base_intent = safe_intent.mean(dim=0, keepdim=True) if safe_intent.dim() > 1 else safe_intent

            expanded_critic = base_critic.expand(self.num_perturbations, -1)
            expanded_intent = base_intent.expand(self.num_perturbations, -1)

            perturbed_ctx = torch.cat([perturbed_latents, expanded_critic[:, 256:768], expanded_intent], dim=-1)
            s_val = self.actor_critic.intrinsic_critic(perturbed_ctx).squeeze(-1)
            s_val = torch.nan_to_num(s_val, nan=0.0, posinf=1.0, neginf=-1.0)

            self.perturbation_scores.mul_(0.9).add_(s_val * 0.1)

            sorted_idx = torch.argsort(self.perturbation_scores, descending=True)
            elite_count = max(1, self.num_perturbations // 4)
            elites = self.perturbation_vectors[sorted_idx[:elite_count]]

            p1 = torch.randint(0, elite_count, (self.num_perturbations,), device=target_device)
            p2 = torch.randint(0, elite_count, (self.num_perturbations,), device=target_device)

            mask = torch.rand(self.num_perturbations, 256, device=target_device) > 0.5
            mutated_batch = torch.where(mask, elites[p1], elites[p2])

            mut_mask = torch.rand(self.num_perturbations, 256, device=target_device) < 0.05
            mut_noise = torch.randn(self.num_perturbations, 256, device=target_device, dtype=torch.float16) * 0.02

            mutated_batch.add_(mut_mask.to(torch.float16) * mut_noise)
            mutated_batch = torch.clamp(mutated_batch, min=-1.0, max=1.0)

            mutated_batch[0].copy_(torch.clamp(elites[0], min=-1.0, max=1.0))
            self.perturbation_vectors.copy_(mutated_batch)
            self.target_network_drift.copy_(mutated_batch[0].unsqueeze(0) * 0.05)

    def evaluate_ood_adaptation(
        self, ood_text: str, host_latent: torch.Tensor, temperature_shift: float = 50.0
    ) -> None:
        """Evaluates zero-shot adaptation on provided text prompts.

        Args:
            ood_text: Semantic instruction string.
            host_latent: Baseline latent representation [Batch, Dim].
            temperature_shift: Scalar for distribution shift.
        """
        self.eval()

        # Access unwrapped module directly to bypass DDP overhead.
        base_module = self.actor_critic.module if hasattr(self.actor_critic, "module") else self.actor_critic

        with torch.no_grad():
            raw_bytes = self.TextTokenizer.transform(ood_text, host_latent.size(0), host_latent.device)
            text_tensor = self.multimodal.process_text(raw_bytes)

            initial_ood_pred = self.jepa(text_tensor)
            ood_latent = initial_ood_pred[0] if isinstance(initial_ood_pred, tuple) else initial_ood_pred
            ood_target = self.jepa.target_encoder(text_tensor)

            hold_out_loss = F.mse_loss(ood_latent, ood_target).item()

            if not hasattr(self, "prev_hold_out_loss"):
                self.prev_hold_out_loss = hold_out_loss

            loss_diff = self.prev_hold_out_loss - hold_out_loss
            self.prev_hold_out_loss = hold_out_loss

            payload = {"diagnostics/ZeroShot_Adaptation_Velocity": loss_diff}
            from vrl_framework.trainer.ppo_engine import metrics_aggregator

            metrics_aggregator.log(payload)

            torch.manual_seed(42)

            if hasattr(self, "actor_critic_ema"):
                eval_target = self.actor_critic_ema.materialize_shadow_copy(base_module)
            else:
                eval_target = base_module

            _ = torch.full((host_latent.size(0), 1), temperature_shift, device=host_latent.device)

            if hasattr(self, "adversary_module"):
                adversarial_attack = self.adversary_module(host_latent)
                eval_context = host_latent + (adversarial_attack * 0.1)
            else:
                eval_context = host_latent

            eval_out = eval_target(eval_context)
            zero_shot_logits = (
                eval_out.policy_logits
                if hasattr(eval_out, "policy_logits")
                else (eval_out[0] if isinstance(eval_out, tuple) else eval_out)
            )

            if wandb.run is not None:
                valid_latent_trace = host_latent.detach().clone()

                if valid_latent_trace.size(0) > 1:
                    traj_entropy = compute_traj_entropy(valid_latent_trace).item()
                else:
                    traj_entropy = 0.0

                eff_dim_val = (
                    self.evaluate_activation_diversity(valid_latent_trace)
                    if hasattr(self, "evaluate_activation_diversity")
                    else 0.0
                )
                metrics = {
                    "diagnostics/ZeroShot_Adaptation_Velocity": loss_diff,
                    "diagnostics/traj_entropy": traj_entropy,
                    "diagnostics/ZeroShot_Action_Variance": zero_shot_logits.var().item(),
                    "diagnostics/latent_effective_dimensionality": eff_dim_val,
                }
                from vrl_framework.trainer.ppo_engine import metrics_aggregator

                metrics_aggregator.log(metrics)

        self.train()

    def _consolidate_memory(self) -> None:
        if len(self.stm_buffer) >= BATCH_SIZE:
            batch = torch.cat([t.detach() for t in self.stm_buffer[:BATCH_SIZE]])
            with torch.no_grad():
                _, _, vq_indices = self.jepa.quantizer(batch)
                vq_indices = vq_indices.to(torch.int8)
            self.memory.store(vq_indices)
            self.stm_buffer = self.stm_buffer[BATCH_SIZE:]

    def consolidate_old_stm(self, max_stm_length: int = 100) -> None:
        """Consolidates the oldest transitions in the gradient STM buffer."""
        if len(self.stm_buffer) > max_stm_length:
            consolidated = self.gradient_stm.consolidate_memory(self.stm_buffer[:-max_stm_length])
            self.stm_buffer = self.stm_buffer[-max_stm_length:] + [consolidated.detach()]

    def _ensure_stm_buffer(self, batch_size: int, device: torch.device) -> None:
        """Resizes the Short-Term Memory tensor if the current batch size exceeds the allocated capacity."""
        current_capacity = self.stm_tensor.size(0)

        if batch_size > current_capacity:
            new_capacity = max(batch_size, current_capacity * 2)
            expanded_tensor = torch.zeros(new_capacity, 1024, 256, device=device)

            expanded_tensor[:current_capacity] = self.stm_tensor

            self.register_buffer("stm_tensor", expanded_tensor)
            self.stm_ptr = self.stm_ptr.to(device)

    class TextTokenizer:
        """Encodes strings into fixed-size byte tensors."""

        @staticmethod
        def transform(text_str: str, batch_size: int, device: torch.device, max_bytes: int = 256) -> torch.Tensor:
            """
            Args:
                text_str: Raw input string.
                batch_size: Target batch dimension.
                device: Target CUDA device.
                max_bytes: Maximum sequence truncation limit.

            Returns:
                Padded ASCII tensor, shape [B, 1, max_bytes].
            """
            if not text_str or not isinstance(text_str, str) or not text_str.strip():
                text_str = "<SILENCE>"

            encoded = bytearray(text_str.encode("utf-8", "ignore"))[:max_bytes]
            payload = torch.zeros(max_bytes, dtype=torch.float32, device=device)
            if encoded:
                payload[: len(encoded)] = torch.frombuffer(encoded, dtype=torch.uint8).to(
                    dtype=torch.float32, device=device
                )

            return payload.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)  # [B, 1, seq_len]

    def step(
        self,
        obs: torch.Tensor,
        external_signal: Optional[str] = None,
        audio: Optional[torch.Tensor] = None,
        proprioceptive_state: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        self.step_counter.add_(1)
        return self.forward(obs, external_signal, audio, proprioceptive_state)

    def forward(
        self,
        obs: torch.Tensor,
        external_signal: Optional[str] = None,
        audio: Optional[torch.Tensor] = None,
        proprioceptive_state: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Computes policy logits from multimodal inputs.

        Args:
            obs: Visual observation tensor, shape [B, C, H, W, D].
            external_signal: Textual instruction string.
            audio: Audio waveform, shape [B, 1, 44100].
            proprioceptive_state: Joint state representation, shape [B, 32].

        Returns:
            policy_logits_chunked: Action distributions, shape [B, num_actions].
            sparse_concepts: Extracted latent states, shape [B, latent_dim].
        """
        batch_size = obs.size(0)
        visual_features = self.sensory(obs)

        if self.training and hasattr(self.sensory, "pcn_layer"):
            self.sensory.pcn_layer.learning_phase()

        self._ensure_stm_buffer(batch_size, visual_features.device)
        ptr = int(self.stm_ptr.item())

        self.stm_tensor[:batch_size, ptr, :] = visual_features.detach()
        self.stm_ptr.fill_((ptr + 1) % 1024)
        valid_stm = self.stm_tensor[:batch_size, : max(1, ptr + 1), :]

        consolidated_stm = self.gradient_stm(valid_stm)
        visual_integrated = FrequencyDomainBinding.bundle(visual_features, consolidated_stm)

        if audio is None:
            audio = torch.zeros(batch_size, 1, 44100, device=MODEL_DEVICE)
        audio_features = self.multimodal.process_audio(audio)

        if proprioceptive_state is None:
            proprioceptive_state = torch.zeros(batch_size, 32, device=MODEL_DEVICE)

        modalities = {"sensory": visual_integrated, "audio": audio_features, "proprioception": proprioceptive_state}

        if not hasattr(self, "text_embedding"):
            self.text_embedding = None

        if self.manager_goal.size(0) != batch_size:
            self.register_buffer("manager_goal", torch.zeros(batch_size, 256, device=MODEL_DEVICE))

        from einops import repeat

        raw_bytes = self.TextTokenizer.transform(external_signal or "", batch_size, MODEL_DEVICE)
        text = self.multimodal.process_text(raw_bytes, goal_context=self.text_embedding)

        self.text_embedding = text.detach()

        if text.size(0) == 1 and batch_size > 1:
            text = repeat(text, "1 d -> b d", b=batch_size)
        elif text.size(0) != batch_size:
            raise RuntimeError(f"Dimensionality mismatch: expected batch size {batch_size}, got {text.size(0)}")

        current_step_tensor = self.step_counter

        warmup_limit = getattr(CFG, "WARMUP_STEPS", 1000)

        decay_factor = torch.clamp(1.0 - (current_step_tensor.float() / warmup_limit), min=0.0)
        noise_scale = getattr(CFG, "TEXT_NOISE_ALPHA", 0.1) * decay_factor

        noise = torch.randn_like(text) * noise_scale
        mask = (current_step_tensor < warmup_limit).float()

        text = text + noise * mask if text.dim() <= 2 else text + noise * mask.view(1, 1, 1)

        modalities["text"] = text

        if self.training:
            self.aux_text_loss = getattr(self.multimodal, "last_text_loss", torch.tensor(0.0, device=MODEL_DEVICE))

        if hasattr(self.sensory, "pcn_layer") and hasattr(self.sensory.pcn_layer, "last_convergence_steps"):
            self.deq_convergence_steps = self.sensory.pcn_layer.last_convergence_steps.detach()
        else:
            self.deq_convergence_steps = torch.full((obs.size(0),), 5.0, device=MODEL_DEVICE)

        integrated_context = self.bottleneck_attention(modalities)

        ptr = int(self.stm_ptr.item())
        batch_size = integrated_context.size(0)

        self._ensure_stm_buffer(batch_size, integrated_context.device)

        self.stm_tensor[:batch_size, ptr, :] = integrated_context.detach()
        self.stm_ptr.fill_((ptr + 1) % 1024)

        if self.continuous_stream.size(0) != batch_size:
            self.continuous_stream = torch.zeros_like(integrated_context)

        self.continuous_stream = self.ponder_gru(integrated_context, self.continuous_stream.detach(), delta_t=0.4)

        jepa_out = self.jepa(self.continuous_stream)
        latent_context = jepa_out[0] if isinstance(jepa_out, tuple) else jepa_out

        if self.training and hasattr(self, "adversary_module"):
            latent_context = self.adversary_controller.apply_budgeted_warp(
                self.adversary_module,
                latent_context,
                self.jepa,
                self.actor_critic,
                consolidation_weight=torch.tensor([1.0], device=MODEL_DEVICE),
            )

        pred_error_signal = torch.zeros(integrated_context.size(0), 1, device=MODEL_DEVICE)
        if isinstance(jepa_out, tuple) and len(jepa_out) == 4:
            _, online_pred, target_proj, _ = jepa_out
            pred_error_signal = (
                F.mse_loss(online_pred, target_proj.detach(), reduction="none").mean(dim=-1, keepdim=True).detach()
            )

        self.cognitive_fatigue += pred_error_signal.mean() * 0.1
        if self.cognitive_fatigue.item() > 100.0:
            self._offline_consolidation(latent_context.mean(dim=0, keepdim=True))

        if self.training and isinstance(jepa_out, tuple):
            self.last_vq_loss = (
                jepa_out[-1].detach()
                if isinstance(jepa_out[-1], torch.Tensor)
                else torch.tensor(jepa_out[-1], device=MODEL_DEVICE)
            )
            if len(jepa_out) == 4:
                self.last_byol_loss = F.mse_loss(
                    F.normalize(jepa_out[1], dim=-1), F.normalize(jepa_out[2], dim=-1)
                ).detach()
        else:
            self.last_vq_loss = torch.tensor(0.0, device=MODEL_DEVICE)

        if hasattr(self.jepa.quantizer, "cluster_usage"):
            quantized_context, _, _ = self.jepa.quantizer(latent_context)
            if self.training and self.step_counter.item() % 100 == 0:
                inactive_mask = self.jepa.quantizer.cluster_usage == 0
                if inactive_mask.any():
                    if hasattr(self, "_momentum_clear_callback"):
                        self._momentum_clear_callback(torch.where(inactive_mask)[0].tolist())
                    elif getattr(getattr(self, "jepa", None), "quantizer", None) is not None and hasattr(
                        self.jepa.quantizer, "_momentum_clear_callback"
                    ):
                        self.jepa.quantizer._momentum_clear_callback(torch.where(inactive_mask)[0].tolist())
        else:
            with torch.no_grad():
                quant_out = self.jepa.quantizer(latent_context)
                quantized_context = quant_out[0] if isinstance(quant_out, tuple) else quant_out

        z_detached = quantized_context.detach()

        # Clip feature variance for numerical stability.
        with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
            z_float = z_detached.float()
            variance = z_float.pow(2).mean(dim=-1, keepdim=True)

            batch_variance_mean = variance.mean(dim=0, keepdim=True)
            batch_variance_std = torch.sqrt(variance.var(dim=0, unbiased=False, keepdim=True) + 1e-6) + 1e-6

            stability_threshold = batch_variance_mean + (3.0 * batch_variance_std)

            penalty_scale = torch.where(
                variance > stability_threshold,
                torch.sqrt(stability_threshold / (variance + 1e-6)),
                torch.ones_like(variance),
            )

            z_detached = (z_float * penalty_scale).to(z_detached.dtype)

        batch_size = z_detached.size(0)

        max_steps = 10
        _ = torch.zeros(batch_size, 1, device=MODEL_DEVICE)
        ponder_cost = torch.zeros(batch_size, 1, device=MODEL_DEVICE)
        remain_prob = torch.ones(batch_size, 1, device=MODEL_DEVICE)
        final_context = torch.zeros_like(z_detached)

        current_state = z_detached

        if not hasattr(self, "s_hebb_trace") or self.s_hebb_trace.size(0) != batch_size:
            self.s_hebb_trace = torch.zeros(batch_size, 64, 256, device=MODEL_DEVICE)
            self.z_hebb_trace = torch.zeros(batch_size, 64, device=MODEL_DEVICE)

        lambda_decay = torch.sigmoid(-pred_error_signal).unsqueeze(-1)
        s_hebb = self.s_hebb_trace * lambda_decay
        z_hebb = self.z_hebb_trace * lambda_decay.squeeze(-1)

        film_params = self.stress_to_film(pred_error_signal)
        gamma, beta = film_params.chunk(2, dim=-1)

        sp_expanded = self.scratchpad.expand(batch_size, -1, -1)
        ring_ptr = self.memory.ring_ptr
        is_full = getattr(self.memory, "memory_full", False)

        cached_raw_keys = None
        cached_key_chunk_fp16 = None
        cached_key_chunk_hash = None
        valid_limit = 0

        if ring_ptr > 0 or is_full:
            valid_limit = self.memory.ram_ring_buffer_ep.size(0) if is_full else ring_ptr
            with torch.no_grad():
                cached_raw_keys = self.memory.ram_ring_buffer_ep[:valid_limit].detach().float()
                keys_norm_cached = F.normalize(cached_raw_keys, p=2, dim=-1)
                cached_key_chunk_fp16 = keys_norm_cached[:valid_limit].to(torch.float16)
                cached_key_chunk_hash = (torch.matmul(cached_key_chunk_fp16, self.lsh_hash_planes) > 0).to(
                    torch.float16
                ) * 2.0 - 1.0

        for step in range(max_steps):
            if hasattr(self.jepa, "proxy_surprisal"):
                self.halting_head.proxy_surprisal_reference = self.jepa.proxy_surprisal.detach()

            h_probs, p_cost, h_state = self.halting_head(current_state)
            h_t = h_probs.unsqueeze(-1) if h_probs.dim() == 1 else h_probs

            if step == max_steps - 1:
                h_t = torch.ones_like(h_t)

            p_t = h_t * remain_prob
            remain_prob = remain_prob - p_t

            if (
                getattr(CFG, "AGENT_MODE", "default") == "memory_augmented"
                and hasattr(self, "episodic_memory")
                and hasattr(self, "causal_predictor")
            ):
                logic_features = self.episodic_memory.query(current_state)
                causal_embed = self.causal_predictor(logic_features)
            else:
                logic_features = torch.zeros_like(current_state)
                causal_embed = torch.zeros_like(current_state)

            if not hasattr(self, "last_meta_state") or self.last_meta_state.size(0) != batch_size:
                self.last_meta_state = torch.zeros(batch_size, 256, device=MODEL_DEVICE)
                self.last_action_one_hot = torch.zeros(batch_size, self.num_actions, device=MODEL_DEVICE)
                self.last_reward = torch.zeros(batch_size, 1, device=MODEL_DEVICE)

            gru_input = torch.cat([causal_embed, self.last_action_one_hot, self.last_reward], dim=-1)
            h_meta = self.meta_gru(gru_input, self.last_meta_state)
            self.last_meta_state = h_meta.detach()

            sp_out, _ = self.scratchpad_attention(query=h_meta.unsqueeze(1), key=sp_expanded, value=sp_expanded)
            scratchpad_out = h_meta + sp_out.squeeze(1)

            modulated_state = (scratchpad_out * (1.0 + gamma)) + beta

            if self.training:
                mod_var = modulated_state.var(dim=0)
                std_loss = torch.mean(F.relu(1.0 - torch.sqrt(mod_var + 1e-4)))
                mod_centered = modulated_state - modulated_state.mean(dim=0, keepdim=True)
                mod_centered_f32 = mod_centered.float()
                cov_matrix = (mod_centered_f32.t() @ mod_centered_f32) / (batch_size - 1 + 1e-4)
                off_diag_mod = cov_matrix - torch.diag(torch.diag(cov_matrix))
                cov_loss = torch.sum(
                    F.smooth_l1_loss(off_diag_mod, torch.zeros_like(off_diag_mod), reduction="none")
                ) / cov_matrix.size(0)
                inv_loss = F.mse_loss(modulated_state, scratchpad_out)
                self.vicreg_loss = std_loss + 0.04 * cov_loss.to(modulated_state.dtype) + inv_loss

            if CFG.USE_MOE:
                moe_out = self.moe(modulated_state, error_signal=pred_error_signal)
            else:
                moe_out = modulated_state + self.moe(modulated_state)

            queries = self.dnc_query_generator(moe_out)

            if ring_ptr > 0 or is_full:
                assert cached_raw_keys is not None
                raw_keys = cached_raw_keys

                query_norm = F.normalize(queries, p=2, dim=-1)

                top_k = min(16, valid_limit)

                best_scores = torch.full((batch_size, top_k), -float("inf"), device=MODEL_DEVICE, dtype=torch.float16)
                best_indices = torch.zeros((batch_size, top_k), dtype=torch.long, device=MODEL_DEVICE)

                query_norm_fp16 = query_norm.to(torch.float16)

                with torch.no_grad():
                    query_hash = (torch.matmul(query_norm_fp16, self.lsh_hash_planes) > 0).to(
                        torch.float16
                    ) * 2.0 - 1.0

                    assert cached_key_chunk_fp16 is not None
                    assert cached_key_chunk_hash is not None

                    key_chunk_fp16 = cached_key_chunk_fp16
                    key_chunk_hash = cached_key_chunk_hash

                    hash_matches = torch.matmul(query_hash, key_chunk_hash.t())

                    dynamic_threshold = 4.0
                    if getattr(self.trainer, "ablation_metrics", None) is not None:
                        eff_dim = self.trainer.ablation_metrics.latent_rank
                        dynamic_threshold = max(2.0, 8.0 - (eff_dim / 30.0))

                    valid_mask = (hash_matches > dynamic_threshold).any(dim=0)

                    if valid_mask.any():
                        filtered_keys = key_chunk_fp16[valid_mask]
                        sim_chunk = torch.matmul(query_norm_fp16, filtered_keys.t())

                        original_indices = torch.where(valid_mask)[0]
                        k_chunk = min(top_k, sim_chunk.size(1))

                        chunk_top_scores, chunk_top_idx = torch.topk(sim_chunk, k_chunk, dim=-1)
                        global_top_idx = original_indices[chunk_top_idx]

                        combined_scores = torch.cat([best_scores, chunk_top_scores], dim=-1)
                        combined_indices = torch.cat([best_indices, global_top_idx], dim=-1)

                        best_scores, top_k_relative_idx = torch.topk(combined_scores, top_k, dim=-1)
                        best_indices = torch.gather(combined_indices, -1, top_k_relative_idx)

                keys_values = raw_keys[best_indices]
            else:
                keys_values = self.memory.core.detach().unsqueeze(0).expand(batch_size, -1, -1)

            retrieved_context_raw, _ = self.dnc_cross_attention(
                query=queries.unsqueeze(1), key=keys_values, value=keys_values
            )
            retrieved_context = (
                retrieved_context_raw.squeeze(1) if retrieved_context_raw.dim() == 3 else retrieved_context_raw
            )

            refined_context = moe_out + retrieved_context

            q_hebb = F.elu(self.fwp_q(refined_context)) + 1.0
            k_hebb = F.elu(self.fwp_k(refined_context)) + 1.0
            v_hebb = self.fwp_v(refined_context)

            q_f32, k_f32, v_f32 = q_hebb.float(), k_hebb.float(), v_hebb.float()
            s_hebb_f32 = s_hebb.float() + torch.bmm(k_f32.unsqueeze(2), v_f32.unsqueeze(1))
            z_hebb_f32 = z_hebb.float() + k_f32

            denominator = (q_f32 * z_hebb_f32).sum(dim=-1, keepdim=True)
            safe_denominator = torch.clamp(denominator, min=1e-4)
            out_hebb = (torch.bmm(q_f32.unsqueeze(1), s_hebb_f32).squeeze(1) / safe_denominator).to(
                refined_context.dtype
            )

            s_hebb = s_hebb_f32.to(s_hebb.dtype)
            z_hebb = z_hebb_f32.to(z_hebb.dtype)

            refined_context = refined_context + out_hebb

            current_state = self.ponder_gru(refined_context, current_state)
            final_context = final_context + (p_t * current_state)

            lambda_penalty = 0.2
            geom_prior = lambda_penalty * ((1.0 - lambda_penalty) ** step)
            safe_geom_prior = max(geom_prior, 1e-7)

            if hasattr(self, "proxy_surprisal_reference"):
                surprisal_factor = 1.0 / torch.clamp(self.proxy_surprisal_reference, min=0.1, max=10.0)
            else:
                surprisal_factor = torch.ones_like(p_t)

            safe_p_t = torch.clamp(p_t, min=1e-7, max=1.0)
            kl_divergence_term = safe_p_t * (torch.log(safe_p_t) - math.log(safe_geom_prior))

            ponder_cost = ponder_cost + kl_divergence_term * surprisal_factor.squeeze(-1)

            current_state = refined_context

        self.s_hebb_trace = s_hebb.detach()
        self.z_hebb_trace = z_hebb.detach()
        self.last_ponder_loss = ponder_cost.mean()

        final_context = self.ponder_norm(final_context)

        self.memory.store(final_context.detach())
        memory_context = final_context

        batch_size = obs.size(0)
        if self.manager_goal.size(0) != batch_size:
            self.register_buffer("manager_goal", torch.zeros(batch_size, 256, device=obs.device))
            self.register_buffer("previous_memory_context", torch.zeros(batch_size, 256, device=obs.device))

        current_epoch = getattr(self.trainer, "global_train_step", 0) if hasattr(self, "trainer") else 0
        dynamic_freq = self.hierarchical_planner.update_freq
        if current_epoch > 50000:
            dynamic_freq = min(512, self.hierarchical_planner.update_freq + int(current_epoch / 1000))

        if self.step_counter.item() % dynamic_freq == 0:
            raw_manager_goal = self.hierarchical_planner.get_manager_goal(memory_context, add_noise=self.training)

            dummy_action = torch.ones(batch_size, self.num_actions, device=MODEL_DEVICE) / self.num_actions
            anticipated_future = self.latent_dynamics(memory_context, dummy_action)

            lpm_reward, _ = self.lpm_module(memory_context, anticipated_future, dummy_action)
            self.manager_goal = raw_manager_goal * (1.0 + lpm_reward.unsqueeze(-1))

            self.previous_memory_context = memory_context.clone().detach()
        else:
            # HIRO goal transition: g_t = s_{t-c} + g_{t-c} - s_t
            raw_goal_shift = self.previous_memory_context - memory_context.detach()
            self.manager_goal = F.normalize((self.manager_goal * 0.98) + raw_goal_shift, p=2, dim=-1) * 5.0
            self.previous_memory_context = memory_context.clone().detach()

        self.step_counter += 1

        ptr = int(self.stm_ptr.item())
        if ptr >= 8:
            recent_history = self.stm_tensor[:, ptr - 8 : ptr, :]
        else:
            recent_history = self.stm_tensor[:, : ptr + 1, :]

        loss_signal = self.meta_inference(recent_history)

        if self.training:
            hypo_tau = 0.1 + loss_signal.mean().item()

            with torch.no_grad():
                if hasattr(self, "adversary_module"):
                    noise_delta = torch.randn_like(memory_context) * 0.05
                    adv_ctx = self.adversary_module(memory_context + noise_delta)
                else:
                    adv_ctx = memory_context

                optimal_actions = self.hypothesis_generator(
                    adv_ctx,
                    dynamics_model=self.latent_dynamics,
                    actor_critic=self.actor_critic,
                    fuzzy_kb=self.fuzzy_kb,
                    causal_reasoner=self.causal_symbolic_reasoner,
                    tau=hypo_tau,
                )

            base_worker_context = self.hierarchical_planner(memory_context, self.manager_goal)

            if not hasattr(self, "_action_to_worker_proj"):
                self._action_to_worker_proj = torch.randn(optimal_actions.size(-1), 256, device=optimal_actions.device)

            plan_embedding_raw = torch.matmul(optimal_actions, self._action_to_worker_proj)
            plan_embedding = plan_embedding_raw.mean(dim=1) if plan_embedding_raw.dim() == 3 else plan_embedding_raw

            worker_context = base_worker_context + (plan_embedding * 0.1)
        else:
            worker_context = self.hierarchical_planner(memory_context, self.manager_goal)
            with torch.no_grad():
                jepa_surprisal = self.jepa.proxy_surprisal.mean()

                if not hasattr(self, "jepa_surprisal_ema"):
                    self.jepa_surprisal_ema = jepa_surprisal
                else:
                    self.jepa_surprisal_ema = 0.95 * self.jepa_surprisal_ema + 0.05 * jepa_surprisal

                surprisal_hook = torch.tanh(self.jepa_surprisal_ema) * 0.25

                safe_cognitive_stress = loss_signal.mean()
                dynamic_temp = 0.1 + (safe_cognitive_stress * 0.5) + surprisal_hook

                target_log_alpha = torch.log(torch.clamp(dynamic_temp, min=0.05))
                self.exploration_layer.log_alpha.copy_(target_log_alpha)

        write_probs = torch.sigmoid(self.scratchpad_write_gate(worker_context))
        write_vals = self.scratchpad_write_val(worker_context)

        expanded_sp = self.scratchpad.expand(batch_size, -1, -1).clone()
        write_updates = write_probs.unsqueeze(-1) * write_vals.unsqueeze(1)
        new_scratchpad = expanded_sp * (1.0 - write_probs.unsqueeze(-1)) + write_updates

        sp_out_write, _ = self.scratchpad_attention(
            query=worker_context.unsqueeze(1), key=new_scratchpad, value=new_scratchpad
        )
        worker_context = worker_context + sp_out_write.squeeze(1)

        actor_out = self.actor_critic(worker_context)
        policy_logits = getattr(
            actor_out, "policy_logits", actor_out[0] if isinstance(actor_out, tuple) else actor_out
        )
        assert isinstance(policy_logits, torch.Tensor)

        if policy_logits.dim() == 3:
            immediate_logits = policy_logits[:, 0, : self.num_actions]
        else:
            immediate_logits = policy_logits[..., : self.num_actions]

        with torch.no_grad():
            jepa_variance = self.jepa.target_variance_ema.item() if hasattr(self.jepa, "target_variance_ema") else 1.0
            high_pred_error = pred_error_signal.squeeze(-1) > (jepa_variance * 1.5)

            action_one_hot_preview = F.one_hot(
                torch.argmax(immediate_logits, dim=-1), num_classes=self.num_actions
            ).float()
            predicted_future = self.latent_dynamics(z_detached, action_one_hot_preview)

            dummy_critic_preview = torch.zeros(batch_size, 768, device=MODEL_DEVICE)
            future_ac_out = self.actor_critic(predicted_future, dummy_critic_preview)
            critic_divergence = torch.abs(future_ac_out.value_logits_1 - future_ac_out.value_logits_2).mean(dim=-1)

            high_critic_risk = critic_divergence > 2.0

            if high_pred_error.any() or high_critic_risk.any():
                self.action_think_idx = self.num_actions - 1

            think_decisions = (
                (torch.argmax(immediate_logits, dim=-1) == self.action_think_idx) | high_pred_error | high_critic_risk
            )

            if think_decisions.any():
                logical_context = self.fuzzy_kb.reason(z_detached)
                causal_context = self.causal_symbolic_reasoner(logical_context)

                critic_ctx_proxy = torch.cat([z_detached, logical_context, causal_context], dim=-1)

                with torch.no_grad():
                    ac_out_eval = self.actor_critic(worker_context[think_decisions], critic_ctx_proxy[think_decisions])
                    v1, v2 = ac_out_eval.value_logits_1, ac_out_eval.value_logits_2
                    epistemic_uncert = min(torch.abs(v1 - v2).mean().item(), 100.0)

                    dynamic_error = getattr(self, "dynamics_mse_ema", torch.tensor(0.5)).item()
                    true_quant_error = getattr(self, "true_quant_error", 0.1)
                    representation_health_score = epistemic_uncert + (0.5 * dynamic_error) + (0.5 * true_quant_error)

                    dyn_err_val = getattr(self, "dynamics_mse_ema", torch.tensor(100.0)).item()
                    current_step = getattr(self, "global_train_step", 0)
                    anneal_factor = math.exp(-current_step / 5000.0)

                    gating_factor = math.exp(-0.1 * max(0.0, dyn_err_val - 5.0)) * anneal_factor
                    dynamic_depth = max(1, int(3 * gating_factor))
                    dynamic_samples = max(4, int(16 * gating_factor))

                    calc_health_band = (
                        2 if representation_health_score < 1.0 else (1 if representation_health_score < 3.0 else 0)
                    )
                    dummy_budget = PlanningBudget(
                        stability_score=representation_health_score,
                        validity_band=calc_health_band,
                        max_depth=dynamic_depth,
                        num_samples=dynamic_samples,
                        distill_enabled=True,
                        teacher_ttl=5,
                        allow_actor_lookahead=True,
                        allow_teacher_write=True,
                        allow_distillation=True,
                        max_branch_survivors=max(1, dynamic_samples // 2),
                        min_survivor_floor=1,
                        max_ood_risk=1.5,
                        max_critic_divergence=5.0,
                        max_planner_calls_per_env_step=1,
                    )

                    h_prev = z_detached[think_decisions][:, self.latent_dynamics.stoch_dim :]
                    ensemble_preds = torch.stack([head(h_prev) for head in self.latent_dynamics.ensemble_heads], dim=0)
                    epistemic_variance = ensemble_preds.float().var(dim=0).mean(dim=-1).to(ensemble_preds.dtype)

                    h_probs, _, _ = self.halting_head(z_detached[think_decisions])
                    epistemic_threshold = 2.0
                    dummy_halting = torch.where(
                        epistemic_variance > epistemic_threshold,
                        torch.ones_like(h_probs.squeeze(-1)),
                        h_probs.squeeze(-1),
                    )

                mcts_response = self.latent_mcts(
                    initial_latent=worker_context[think_decisions],
                    jepa_predictor=self.jepa.predictor,
                    actor_critic=self.actor_critic,
                    critic_context=critic_ctx_proxy[think_decisions],
                    planning_budget=dummy_budget,
                    halting_budget=dummy_halting,
                    causal_engine=self.causal_symbolic_reasoner,
                    sae_module=self.sae,
                    fuzzy_kb=self.fuzzy_kb,
                )
                immediate_logits[think_decisions, : self.num_actions - 1] = mcts_response.final_blended_logits[
                    :, : self.num_actions - 1
                ]

                num_think = int(think_decisions.sum().item())
                ptr = int(self.mcts_buffer_ptr.item())
                end_ptr = min(ptr + num_think, 1024)
                valid_inserts = end_ptr - ptr
                if valid_inserts > 0:
                    self.mcts_teacher_buffer_states[ptr:end_ptr] = worker_context[think_decisions][
                        :valid_inserts
                    ].detach()
                    self.mcts_teacher_buffer_logits[ptr:end_ptr] = mcts_response.final_blended_logits[
                        :valid_inserts
                    ].detach()
                    self.expiration_step[ptr:end_ptr] = 100
                    self.mcts_buffer_ptr.copy_(torch.tensor(end_ptr % 1024, dtype=torch.long, device=MODEL_DEVICE))

        safe_logits = self.causal_masker(worker_context, immediate_logits)

        policy_logits_chunked = policy_logits.clone()
        if policy_logits_chunked.dim() == 3:
            policy_logits_chunked[:, 0, : self.num_actions] = safe_logits
        else:
            policy_logits_chunked[..., : self.num_actions] = safe_logits

        sparse_concepts, _ = self.sae(worker_context)

        if getattr(self, "return_latents", False):
            return policy_logits_chunked, z_detached, sparse_concepts

        return policy_logits_chunked, sparse_concepts

    class PolicyTrajectoryBuffer:
        """Ring buffer for recurrent policy trajectories and BPTT snapshots."""

        def __init__(self, capacity: int, dim: int) -> None:
            self.capacity = capacity
            self.dim = dim
            self.ptr = 0
            self.size = 0

            self.states = torch.zeros(capacity, dim)
            self.actions = torch.zeros(capacity, dtype=torch.long)
            self.log_probs = torch.zeros(capacity)
            self.returns = torch.zeros(capacity)
            self.advantages = torch.zeros(capacity)
            self.next_states = torch.zeros(capacity, dim)
            self.hidden_states = torch.zeros(capacity, dim)
            self.costs = torch.zeros(capacity)

        def store_batch(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            log_probs: torch.Tensor,
            rets: torch.Tensor,
            advs: torch.Tensor,
            next_states: torch.Tensor,
            hidden_states: torch.Tensor,
            costs: torch.Tensor,
        ) -> None:
            batch_size = states.size(0)
            end_ptr = self.ptr + batch_size

            if end_ptr <= self.capacity:
                self.states[self.ptr : end_ptr] = states
                self.actions[self.ptr : end_ptr] = actions
                self.log_probs[self.ptr : end_ptr] = log_probs
                self.returns[self.ptr : end_ptr] = rets
                self.advantages[self.ptr : end_ptr] = advs
                self.next_states[self.ptr : end_ptr] = next_states
                self.hidden_states[self.ptr : end_ptr] = hidden_states
                self.costs[self.ptr : end_ptr] = costs
            else:
                overflow = end_ptr - self.capacity
                first_part = batch_size - overflow

                self.states[self.ptr :] = states[:first_part]
                self.actions[self.ptr :] = actions[:first_part]
                self.log_probs[self.ptr :] = log_probs[:first_part]
                self.returns[self.ptr :] = rets[:first_part]
                self.advantages[self.ptr :] = advs[:first_part]
                self.next_states[self.ptr :] = next_states[:first_part]
                self.hidden_states[self.ptr :] = hidden_states[:first_part]
                self.costs[self.ptr :] = costs[:first_part]

                self.states[:overflow] = states[first_part:]
                self.actions[:overflow] = actions[first_part:]
                self.log_probs[:overflow] = log_probs[first_part:]
                self.returns[:overflow] = rets[first_part:]
                self.advantages[:overflow] = advs[first_part:]
                self.next_states[:overflow] = next_states[first_part:]
                self.hidden_states[:overflow] = hidden_states[first_part:]
                self.costs[:overflow] = costs[first_part:]

            self.ptr = end_ptr % self.capacity
            self.size = min(self.size + batch_size, self.capacity)

        def sample_sequences(self, batch_size: int, chunk_size: int = 16, burn_in: int = 4) -> Optional[
            Tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]
        ]:
            seq_len = chunk_size + burn_in
            if self.size < seq_len:
                return None

            valid_starts = self.size - seq_len
            start_idx = torch.randint(0, valid_starts, (batch_size,))

            if self.size == self.capacity:
                cross_mask = (start_idx < self.ptr) & ((start_idx + seq_len) > self.ptr)
                start_idx = torch.where(cross_mask, (start_idx + seq_len) % valid_starts, start_idx)

            offsets = torch.arange(seq_len).unsqueeze(0)
            indices = start_idx.unsqueeze(1) + offsets

            return (
                self.states[indices],
                self.actions[indices],
                self.log_probs[indices],
                self.returns[indices],
                self.advantages[indices],
                self.next_states[indices],
                self.hidden_states[start_idx],
                self.costs[indices],
            )

    def ppo_update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        next_states: torch.Tensor,
        costs: Optional[torch.Tensor] = None,
        hidden_states_snapshot: Optional[torch.Tensor] = None,
    ) -> float:
        with torch.no_grad():
            states = torch.nan_to_num(states, nan=0.0, posinf=10.0, neginf=-10.0)
            next_states = torch.nan_to_num(next_states, nan=0.0, posinf=10.0, neginf=-10.0)
            if states.dtype in [torch.int8, torch.uint8]:
                corrected_basis = self.jepa.quantizer.basis_correction(states.float())
                states = self.jepa.quantizer.basis_unproj(corrected_basis)
            if next_states.dtype in [torch.int8, torch.uint8]:
                corrected_basis_ns = self.jepa.quantizer.basis_correction(next_states.float())
                next_states = self.jepa.quantizer.basis_unproj(corrected_basis_ns)

        if not hasattr(self, "policy_buffer"):
            self.policy_buffer = self.PolicyTrajectoryBuffer(capacity=16384, dim=states.size(-1))

        if hidden_states_snapshot is not None:
            h_snaps = torch.empty(
                hidden_states_snapshot.shape, dtype=hidden_states_snapshot.dtype, device="cpu", pin_memory=True
            )
            h_snaps.copy_(hidden_states_snapshot.detach(), non_blocking=True)
        else:
            h_snaps = torch.zeros(states.size(0), 256, dtype=torch.float32, device="cpu", pin_memory=True)

        c_vals = (
            costs.detach().to("cpu", non_blocking=True)
            if costs is not None
            else torch.zeros(states.size(0), device="cpu", pin_memory=True)
        )

        self.policy_buffer.store_batch(
            states.detach().to("cpu", non_blocking=True),
            actions.detach().to("cpu", non_blocking=True),
            old_log_probs.detach().to("cpu", non_blocking=True),
            returns.detach().to("cpu", non_blocking=True),
            advantages.detach().to("cpu", non_blocking=True),
            next_states.detach().to("cpu", non_blocking=True),
            h_snaps,
            c_vals,
        )

        seq_batch = self.policy_buffer.sample_sequences(batch_size=CFG.BATCH_SIZE, chunk_size=16, burn_in=4)
        if seq_batch is None:
            return 0.0

        s_seq, a_seq, lp_seq, ret_seq, adv_seq, ns_seq, h_snap, c_seq = seq_batch

        B, T, D = s_seq.shape
        s_seq.view(B * T, D).to(states.device)
        a_seq.view(B * T).to(actions.device)
        lp_seq.view(B * T).to(old_log_probs.device)
        ret_seq.view(B * T).to(returns.device)
        adv_seq.view(B * T).to(advantages.device)
        ns_seq.view(B * T, D).to(next_states.device)
        c_seq.view(B * T).to(costs.device if costs is not None else states.device)

        with torch.no_grad():
            burn_s = s_seq[:, :4, :].reshape(B * 4, D).to(MODEL_DEVICE)
            if hasattr(self.trainer.agent_core, "meta_gru"):
                dummy_action = torch.zeros(B * 4, self.trainer.agent_core.num_actions, device=MODEL_DEVICE)
                dummy_reward = torch.zeros(B * 4, 1, device=MODEL_DEVICE)
                burn_log = self.trainer.agent_core.fuzzy_kb.reason(burn_s)
                gru_input = torch.cat([burn_log, dummy_action, dummy_reward], dim=-1)
                gru_input = F.pad(gru_input, (0, 256 + self.trainer.agent_core.num_actions + 1 - gru_input.size(-1)))
                burn_h = h_snap.repeat_interleave(4, dim=0).to(MODEL_DEVICE)
                _ = self.trainer.agent_core.meta_gru(gru_input, burn_h)

        dummy_snapshots = RecurrentStateSnapshot(
            metagru_h=torch.zeros(B, 256, device=MODEL_DEVICE, dtype=torch.float16),
            pondergru_h=torch.zeros(B, 256, device=MODEL_DEVICE, dtype=torch.float16),
            gradientstm_h=torch.zeros(B, 256, device=MODEL_DEVICE, dtype=torch.float16),
            stmptr=torch.zeros(B, dtype=torch.long, device=MODEL_DEVICE),
            stmtensor_k=torch.zeros(B, 8, 256, device=MODEL_DEVICE, dtype=torch.float16),
        )

        batch = PolicySequenceBatch(
            states=s_seq.to(MODEL_DEVICE),
            actions=a_seq.to(MODEL_DEVICE),
            old_logprobs=lp_seq.to(MODEL_DEVICE),
            returns=ret_seq.to(MODEL_DEVICE),
            advantages=adv_seq.to(MODEL_DEVICE),
            next_states=ns_seq.to(MODEL_DEVICE),
            costs=c_seq.to(MODEL_DEVICE),
            dones=torch.zeros(B, s_seq.size(1), dtype=torch.bool, device=MODEL_DEVICE),
            episode_ids=torch.zeros(B, s_seq.size(1), dtype=torch.long, device=MODEL_DEVICE),
            recurrent_snapshots=dummy_snapshots,
            valid_mask=torch.ones(B, s_seq.size(1), dtype=torch.bool, device=MODEL_DEVICE),
            burnin=4,
            learn_length=16,
        )

        metrics = self.trainer.trainstep(batch)
        return metrics.policy_loss if hasattr(metrics, "policy_loss") else 0.0

    def clone(self) -> "RLAgent":
        new_agent_core = RLAgent(sensory_input_shape=self.sensory_input_shape, num_actions=self.num_actions)

        custom_load_state_dict(new_agent_core, self.state_dict())
        new_agent_core.to(MODEL_DEVICE)

        def sync_optimizer_state(src_opt: Any, dst_opt: Any) -> None:
            if src_opt is None or dst_opt is None:
                return
            for group, new_group in zip(src_opt.param_groups, dst_opt.param_groups):
                new_group["lr"] = group["lr"]

            for src_p, dst_p in zip(
                (p for group in src_opt.param_groups for p in group["params"]),
                (p for group in dst_opt.param_groups for p in group["params"]),
            ):
                if src_p in src_opt.state:
                    dst_opt.state[dst_p] = {}
                    for key, val in src_opt.state[src_p].items():
                        if isinstance(val, torch.Tensor):
                            dst_opt.state[dst_p][key] = val.clone().to(dst_p.device)
                        else:
                            dst_opt.state[dst_p][key] = val

        if hasattr(self, "trainer"):
            try:
                sync_optimizer_state(self.trainer.opt_policy, new_agent_core.trainer.opt_policy)
                if hasattr(self.trainer, "opt_policy_fp32"):
                    sync_optimizer_state(
                        self.trainer.opt_policy_fp32, getattr(new_agent_core.trainer, "opt_policy_fp32", None)
                    )
                sync_optimizer_state(self.trainer.opt_representation, new_agent_core.trainer.opt_representation)
                sync_optimizer_state(
                    getattr(self.trainer, "opt_causal", None), getattr(new_agent_core.trainer, "opt_causal", None)
                )
            except Exception as e:
                logger.warning(f"Optimizer momentum mapping bypassed: {e}")

        return new_agent_core

    @jaxtyped(typechecker=beartype)
    @jaxtyped(typechecker=beartype)
    def save_checkpoint(self, checkpoint_dir: str, epoch: int) -> None:
        is_master = True

        if is_master:
            os.makedirs(checkpoint_dir, exist_ok=True)
            temp_path_st = os.path.join(checkpoint_dir, f"agent_core_weights_ep{epoch}_tmp.safetensors")
            final_path_st = os.path.join(checkpoint_dir, f"agent_core_weights_ep{epoch}.safetensors")
            temp_path_meta = os.path.join(checkpoint_dir, f"agent_core_meta_ep{epoch}_tmp.pt")
            final_path_meta = os.path.join(checkpoint_dir, f"agent_core_meta_ep{epoch}.pt")

            state_dict = self.state_dict()
            contiguous_dict = {k: v.contiguous() for k, v in state_dict.items()}

            meta_state = {
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy_rng_state": np.random.get_state(),
                "python_rng_state": random.getstate(),
            }

            def _async_extract_and_save(st_dict_ref, tmp_st, fin_st, tmp_meta, fin_meta, base_meta_state):
                meta_state_clone = copy.deepcopy(base_meta_state)

                if hasattr(self, "trainer"):
                    meta_state_clone["opt_policy"] = (
                        self.trainer.opt_policy.state_dict() if self.trainer.opt_policy else None
                    )
                    meta_state_clone["opt_representation"] = (
                        self.trainer.opt_representation.state_dict() if self.trainer.opt_representation else None
                    )
                    meta_state_clone["opt_causal"] = (
                        self.trainer.opt_causal.state_dict()
                        if hasattr(self.trainer, "opt_causal") and self.trainer.opt_causal
                        else None
                    )
                    if hasattr(self.trainer, "opt_policy_fp32") and self.trainer.opt_policy_fp32:
                        meta_state_clone["opt_policy_fp32"] = self.trainer.opt_policy_fp32.state_dict()
                    meta_state_clone["scaler"] = self.trainer.scaler.state_dict() if self.trainer.scaler else None

                save_file(st_dict_ref, tmp_st)
                torch.save(meta_state_clone, tmp_meta)
                os.replace(tmp_st, fin_st)
                os.replace(tmp_meta, fin_meta)

            if torch.cuda.is_available():
                torch.cuda.current_stream().synchronize()

            import concurrent.futures

            if not hasattr(self, "_io_thread_pool"):
                self._io_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)

            self._io_thread_pool.submit(
                _async_extract_and_save,
                contiguous_dict,
                temp_path_st,
                final_path_st,
                temp_path_meta,
                final_path_meta,
                meta_state,
            )

    @jaxtyped(typechecker=beartype)
    def load_checkpoint(self, checkpoint_dir: str, epoch: int) -> None:
        from safetensors.torch import load_file

        final_path_st = os.path.join(checkpoint_dir, f"agent_core_weights_ep{epoch}.safetensors")
        final_path_meta = os.path.join(checkpoint_dir, f"agent_core_meta_ep{epoch}.pt")

        state_dict = load_file(final_path_st, device=str(next(self.parameters()).device))
        custom_load_state_dict(self, state_dict)

        if os.path.exists(final_path_meta):
            meta_state = torch.load(final_path_meta, map_location="cpu")
            if "torch_rng_state" in meta_state:
                torch.set_rng_state(meta_state["torch_rng_state"])
            if "cuda_rng_state" in meta_state and meta_state["cuda_rng_state"] is not None:
                torch.cuda.set_rng_state_all(meta_state["cuda_rng_state"])
            if "numpy_rng_state" in meta_state:
                np.random.set_state(meta_state["numpy_rng_state"])
            if "python_rng_state" in meta_state:
                random.setstate(meta_state["python_rng_state"])

            if hasattr(self, "trainer"):
                if meta_state.get("opt_policy") and self.trainer.opt_policy:
                    self.trainer.opt_policy.load_state_dict(meta_state["opt_policy"])
                if meta_state.get("opt_representation") and self.trainer.opt_representation:
                    self.trainer.opt_representation.load_state_dict(meta_state["opt_representation"])
                if meta_state.get("opt_causal") and hasattr(self.trainer, "opt_causal") and self.trainer.opt_causal:
                    self.trainer.opt_causal.load_state_dict(meta_state["opt_causal"])
                if (
                    meta_state.get("opt_policy_fp32")
                    and hasattr(self.trainer, "opt_policy_fp32")
                    and self.trainer.opt_policy_fp32
                ):
                    self.trainer.opt_policy_fp32.load_state_dict(meta_state["opt_policy_fp32"])
                if meta_state.get("scaler") and self.trainer.scaler:
                    self.trainer.scaler.load_state_dict(meta_state["scaler"])

    def evaluate_activation_diversity(self, sensory_input):
        """Computes representation diversity via SVD Entropy.

        Calculates the effective rank of the latent manifold.
        """
        with torch.no_grad():
            features = self.sensory(sensory_input)
            if features.dim() > 2:
                features = features.view(features.size(0), -1)
                if features.size(-1) != 256:
                    if not hasattr(self, "_diversity_proj") or self._diversity_proj.in_features != features.size(-1):
                        self._diversity_proj = nn.Linear(
                            features.size(-1), 256, device=features.device, dtype=features.dtype
                        )
                    features = self._diversity_proj(features)

            jepa_out = self.jepa(features)
            latent = jepa_out[0] if isinstance(jepa_out, tuple) else jepa_out

            if latent.size(0) > 1:
                latent_centered = latent - latent.mean(dim=0, keepdim=True)
                cov_matrix = (latent_centered.T @ latent_centered) / (latent.size(0) - 1)

                jitter = torch.eye(cov_matrix.size(0), device=cov_matrix.device) * 1e-5
                stable_matrix = (cov_matrix + jitter).float()

                try:
                    eigenvalues = torch.linalg.svdvals(stable_matrix)
                except RuntimeError:
                    eigenvalues = torch.linalg.svdvals(stable_matrix.cpu()).to(cov_matrix.device)

                eigenvalues = torch.clamp(eigenvalues - 1e-5, min=1e-8)
                norm_eigenvalues = eigenvalues / eigenvalues.sum()
                entropy = -torch.sum(norm_eigenvalues * torch.log(norm_eigenvalues))

                return torch.exp(entropy).item()

            return latent.norm().item() / latent.size(-1)

    def adaptive_store(self, data, modality="text"):
        if isinstance(data, str):
            tensor_data = self._process_text(data)
        else:
            tensor_data = data

        if tensor_data.dim() == 1:
            tensor_data = tensor_data.unsqueeze(0)

        self.memory.store(tensor_data.detach())

    def metabolic_cost(self):
        base_cost = 0.05

        if hasattr(self.moe, "router"):
            with torch.no_grad():
                l1_router = self.moe.router.weight.abs().mean().item()
                dynamic_cost = l1_router * 0.1
        else:
            dynamic_cost = 0.02

        return base_cost + dynamic_cost

    def prune_low_utility_experts(self, sensory_input, utility_threshold=0.01):
        with torch.no_grad():
            x_flat = self.sensory(sensory_input).view(-1, 256)

            if hasattr(self.moe, "router"):
                routing_probs_dense = F.softmax(self.moe.router(x_flat), dim=-1)

                top_k = int(
                    max(1, min(routing_probs_dense.size(-1), getattr(self.moe, "top_k", getattr(self.moe, "k", 1))))
                )

                routing_selected = torch.topk(routing_probs_dense, k=top_k, dim=-1).indices.reshape(-1).to(torch.int64)
                routing_hist = (
                    torch.bincount(routing_selected, minlength=routing_probs_dense.size(-1))
                    .to(routing_probs_dense.device)
                    .float()
                )

                routing_probs = routing_hist / routing_hist.sum().clamp_min(1.0)

                inactive_experts_mask_mask = routing_probs < utility_threshold
                if inactive_experts_mask_mask.any():
                    num_inactive = inactive_experts_mask_mask.sum().item()
                    k_samples = min(num_inactive, x_flat.size(0))
                    selected_states = x_flat[torch.randperm(x_flat.size(0))[:k_samples]]
                    normalized_anchors = F.normalize(selected_states, p=2, dim=-1)

                    if k_samples < num_inactive:
                        pad_size = num_inactive - k_samples
                        fallback = torch.randn(pad_size, 256, device=x_flat.device)
                        fallback = F.normalize(fallback, p=2, dim=-1)
                        normalized_anchors = torch.cat([normalized_anchors, fallback], dim=0)

                    with torch.no_grad():
                        self.moe.router.weight[inactive_experts_mask_mask, :] = normalized_anchors

            networks_to_prune = []
            if hasattr(self.bottleneck_attention, "modality_projections"):
                networks_to_prune.append(self.bottleneck_attention.modality_projections)
            networks_to_prune.append(self.actor_critic.actor_core)
            networks_to_prune.append(self.actor_critic.critic_1)
            networks_to_prune.append(self.actor_critic.critic_2)
            networks_to_prune.append(self.actor_critic.cost_critic)
            networks_to_prune.append(self.actor_critic.intrinsic_critic)

            for net in networks_to_prune:
                if isinstance(net, nn.ModuleDict):
                    layers = [layer for layer in net.values() if hasattr(layer, "weight")]
                else:
                    layers = [
                        layer for layer in net.modules() if hasattr(layer, "weight") and isinstance(layer, nn.Linear)
                    ]

                for layer in layers:
                    weight_vars = layer.weight.var(dim=1)
                    inactive_units = weight_vars < 1e-5

                    if inactive_units.any():
                        num_inactive = inactive_units.sum().item()
                        fan_in = layer.weight.size(1)

                        if fan_in == x_flat.size(1):
                            k_samples = min(num_inactive, x_flat.size(0))
                            selected_states = x_flat[torch.randperm(x_flat.size(0))[:k_samples]]
                            reborn_weights = F.normalize(selected_states, p=2, dim=-1) * math.sqrt(2.0 / fan_in)

                            if k_samples < num_inactive:
                                pad_size = num_inactive - k_samples
                                fallback = torch.randn(pad_size, fan_in, device=layer.weight.device)
                                fallback = F.normalize(fallback, p=2, dim=-1) * math.sqrt(2.0 / fan_in)
                                reborn_weights = torch.cat([reborn_weights, fallback], dim=0)
                        else:
                            reborn_weights = torch.randn(num_inactive, fan_in, device=layer.weight.device)
                            reborn_weights = F.normalize(reborn_weights, p=2, dim=-1) * math.sqrt(2.0 / fan_in)

                        with torch.no_grad():
                            layer.weight[inactive_units, :] = reborn_weights

    def expand_topology(self, capacity_request=None):
        if hasattr(self.moe, "update_topology"):
            self.moe.update_topology(drop_fraction=0.05)

        if hasattr(self, "hierarchical_planner"):
            for module in self.hierarchical_planner.modules():
                if isinstance(module, HebbianLinear):
                    module.update_topology(drop_fraction=0.05)

    def evaluate_topology_expansion(self, sensory_input):
        with torch.no_grad():
            features = self.sensory(sensory_input)
            feature_variance = features.var().item()
            if feature_variance > 1.5:
                self.expand_topology()

    def temporal_reasoning(self, query, max_steps=15, disable_early_exit=False):
        """Executes latent beam search utilizing the dynamics model and value network."""
        current_thought = query

        if current_thought.dim() == 3:
            current_thought = current_thought[:, -1, :]
        elif current_thought.dim() == 1:
            current_thought = current_thought.unsqueeze(0)

        thought_trajectory = []
        beam_width = 3

        with torch.no_grad():
            for step in range(max_steps):
                actor_out_policy = self.actor_critic(current_thought)
                policy_logits = (
                    actor_out_policy.policy_logits
                    if hasattr(actor_out_policy, "policy_logits")
                    else (actor_out_policy[0] if isinstance(actor_out_policy, tuple) else actor_out_policy)
                )

                if policy_logits.dim() == 3:
                    immediate_logits = policy_logits[:, 0, : self.num_actions]
                else:
                    immediate_logits = policy_logits[..., : self.num_actions]

                _, topk_actions = torch.topk(immediate_logits, beam_width, dim=-1)

                best_value = -float("inf")
                best_next_thought = None

                for k in range(beam_width):
                    candidate_action = topk_actions[:, k]
                    action_one_hot = F.one_hot(candidate_action, num_classes=self.num_actions).float()

                    predicted_next = self.latent_dynamics(current_thought, action_one_hot)
                    if predicted_next.dim() == 2:
                        predicted_next = predicted_next.squeeze(1)

                    candidate_thought = self.causal_symbolic_reasoner(predicted_next)

                    logical_ctx_eval = self.fuzzy_kb.reason(candidate_thought)
                    causal_ctx_eval = self.causal_symbolic_reasoner(logical_ctx_eval)
                    critic_ctx_eval = torch.cat([candidate_thought, logical_ctx_eval, causal_ctx_eval], dim=-1)
                    actor_out_val = self.actor_critic(candidate_thought, critic_context=critic_ctx_eval)
                    val_logits = (
                        actor_out_val.pessimistic_value
                        if hasattr(actor_out_val, "pessimistic_value")
                        else (
                            actor_out_val[1]
                            if isinstance(actor_out_val, tuple)
                            else torch.zeros(1, device=candidate_thought.device)
                        )
                    )
                    state_value = val_logits.mean().item()

                    if state_value > best_value:
                        best_value = state_value
                        best_next_thought = candidate_thought

                current_thought = best_next_thought
                thought_trajectory.append(current_thought)

                if not disable_early_exit and len(thought_trajectory) > 1:
                    delta = F.mse_loss(thought_trajectory[-1], thought_trajectory[-2])
                    if delta < 1e-3:
                        break

        if not thought_trajectory:
            return query

        return torch.stack(thought_trajectory)

    def _process_text(self, text: str, hidden_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Processes byte-encoded text sequences and triggers offline consolidation if idle."""
        if hidden_state is None:
            current_state = torch.zeros(1, 256, device=MODEL_DEVICE, dtype=torch.float32)
        else:
            current_state = hidden_state.detach()

        if self.cognitive_fatigue.item() > 100.0:
            self._offline_consolidation(current_state)
            return current_state

        if text is None or text.strip() == "":
            self._offline_consolidation(current_state)
            return current_state

        trainer_uncertainty = (
            getattr(self.trainer, "uncertainty_signal", torch.tensor(0.0)).item() if hasattr(self, "trainer") else 0.0
        )
        if trainer_uncertainty < 0.5:
            return current_state

        if hasattr(self, "wiki_sub"):
            import zmq

            try:
                raw_bytes = self.wiki_sub.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                raw_bytes = text.encode("utf-8", errors="ignore")
        else:
            raw_bytes = text.encode("utf-8", errors="ignore")

        total_length = len(raw_bytes)
        if total_length == 0:
            self._offline_consolidation(current_state)
            return current_state

        chunk_size = 1024
        byte_tensor = torch.frombuffer(raw_bytes, dtype=torch.uint8)
        num_chunks = math.ceil(total_length / chunk_size)
        padded_length = num_chunks * chunk_size

        if padded_length > total_length:
            pad_size = padded_length - total_length
            byte_tensor = F.pad(byte_tensor, (0, pad_size), value=0)

        pinned_tensor = byte_tensor.pin_memory()

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            chunk_tensor = pinned_tensor[start_idx:end_idx].to(
                device=MODEL_DEVICE, dtype=torch.float32, non_blocking=True
            )
            chunk_tensor = chunk_tensor.unsqueeze(0).unsqueeze(-1)

            prior_mu = current_state.clone()
            prior_logvar = torch.full_like(current_state, -2.0)

            if self.lora_registry:
                with torch.no_grad():
                    signatures = torch.stack(list(self.lora_registry.keys()))
                    distances = torch.norm(signatures - current_state, dim=-1)
                    best_idx = torch.argmin(distances)
                    if distances[best_idx] < 5.0:
                        lora_weights = list(self.lora_registry.values())[best_idx]
                        try:
                            modulation = torch.matmul(current_state, lora_weights.float().to(current_state.device))
                            current_state = FrequencyDomainBinding.normalize_spherical(current_state + modulation)
                        except Exception as e:
                            logger.error(f"Failed to apply O-LoRA episodic skill matrix: {e}")

            current_state = self.multimodal.process_text(chunk_tensor, goal_context=current_state)
            surprisal = getattr(self.multimodal, "last_text_loss", torch.tensor(0.0)).item()

            post_mu = current_state.clone()
            post_logvar = torch.full_like(current_state, -2.5)

            if surprisal > 1.5 and hasattr(self, "fuzzy_kb") and hasattr(self, "memory") and not CFG.STRICT_EX_NIHILO:
                with torch.no_grad():
                    epistemic_query = self.dnc_query_generator(post_mu.detach())
                    episodic_ctx, alarm_ctx, proc_ctx = self.memory.retrieve_triple_head(epistemic_query, top_k=3)

                    if episodic_ctx is not None:
                        grounded_concept = FrequencyDomainBinding.bind(
                            post_mu.detach(), episodic_ctx.mean(dim=0, keepdim=True)
                        )
                    else:
                        grounded_concept = post_mu.detach()

                    abstract_pred = torch.randint(0, len(self.fuzzy_kb.predicates), (1,)).item()
                    self.fuzzy_kb.add_fact(prior_mu.detach(), pred_id=abstract_pred, obj=grounded_concept)

            self.pending_vib_updates.append(
                (prior_mu.detach(), prior_logvar.detach(), post_mu.detach(), post_logvar.detach(), surprisal)
            )

            if len(self.pending_vib_updates) > 1000:
                self.pending_vib_updates.clear()

            self.cognitive_fatigue += surprisal * 0.1

        return current_state.detach()

    def _offline_consolidation(self, anchor_state):
        """Executes offline maintenance routines (SAE resampling, KB pruning, LoRA serialization)."""
        adapter_signature = anchor_state.detach().clone().squeeze(0)

        temp_cpu_cache = []
        if (
            hasattr(self, "trainer")
            and hasattr(self.trainer, "opt_representation")
            and self.trainer.opt_representation is not None
        ):
            for state in self.trainer.opt_representation.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cpu()
                        temp_cpu_cache.append((state, k))

        torch.cuda.empty_cache()
        gc.collect()

        with torch.no_grad():
            if hasattr(self, "sae") and hasattr(self.sae, "execute_reservoir_resampling"):
                self.sae.execute_reservoir_resampling()

            if hasattr(self, "fuzzy_kb"):
                valid_mask = torch.ones(self.fuzzy_kb.max_facts, dtype=torch.bool, device=MODEL_DEVICE)
                for i in range(self.fuzzy_kb.kb_ptr.item()):
                    if not self.fuzzy_kb.evaluate_truth_gate(self.fuzzy_kb.kb_subjects[i]):
                        valid_mask[i] = False
                        self.fuzzy_kb.utility_scores[i] = 0.0

                if not valid_mask.all():
                    valid_indices = torch.where(valid_mask)[0]
                    self.fuzzy_kb.kb_subjects[: len(valid_indices)] = self.fuzzy_kb.kb_subjects[valid_indices]
                    self.fuzzy_kb.kb_objects[: len(valid_indices)] = self.fuzzy_kb.kb_objects[valid_indices]
                    self.fuzzy_kb.kb_ptr.fill_(len(valid_indices))

            covariance_proxy = (
                torch.matmul(anchor_state.transpose(0, 1), anchor_state) + torch.eye(256, device=MODEL_DEVICE) * 1e-5
            )
            try:
                U, S, V = torch.linalg.svd(covariance_proxy)
            except RuntimeError:
                U, S, V = torch.linalg.svd(covariance_proxy.cpu())
                U, S, V = U.to(MODEL_DEVICE), S.to(MODEL_DEVICE), V.to(MODEL_DEVICE)

            cumulative_variance = torch.cumsum(S.cpu(), dim=0).to(S.device) / S.sum()
            dynamic_rank = torch.searchsorted(cumulative_variance, 0.90).item() + 1
            dynamic_rank = max(4, min(dynamic_rank, 64))

            lora_A = U[:, :dynamic_rank]
            lora_B = V[:dynamic_rank, :]
            adapter_payload = torch.matmul(lora_A, lora_B).to(torch.float16)

            if len(self.lora_registry) > 20:
                oldest_key = list(self.lora_registry.keys())[0]
                del self.lora_registry[oldest_key]

            self.lora_registry[adapter_signature] = adapter_payload
            self.cognitive_fatigue.fill_(0.0)

        for state, k in temp_cpu_cache:
            state[k] = state[k].to(MODEL_DEVICE, non_blocking=True)

    def _process_audio(self, audio):
        return self.multimodal.process_audio(audio)

    def mutate(self):
        """Applies Gaussian noise and topology rewiring for evolutionary updates."""
        with torch.no_grad():
            ac_module = self.actor_critic._orig_mod if hasattr(self.actor_critic, "_orig_mod") else self.actor_critic
            actor_params = (
                list(ac_module.actor_core.parameters())
                + list(ac_module.actor_head_continuous.parameters())
                + list(ac_module.actor_head_discrete.parameters())
            )
            for param in actor_params:
                noise = torch.randn_like(param) * 0.02
                param.add_(noise)

            if hasattr(self.moe, "router"):
                for param in self.moe.router.parameters():
                    noise = torch.randn_like(param) * 0.05
                    param.add_(noise)

            if hasattr(self.moe, "update_topology"):
                self.moe.update_topology(drop_fraction=0.1)

            if hasattr(self, "hierarchical_planner"):
                for module in self.hierarchical_planner.modules():
                    if isinstance(module, HebbianLinear):
                        module.update_topology(drop_fraction=0.1)

            if hasattr(self, "experience_buffer"):
                self.experience_buffer.clear()


class ObjectEncoder(nn.Module):
    """Slot attention mechanism for invariant object representations."""

    def __init__(self, dim=256, slot_dim=32, num_slots=8, iters=3, keyframe_interval=10):
        super().__init__()
        self.dim = dim
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.iters = iters
        self.keyframe_interval = keyframe_interval

        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_pre_ff = nn.LayerNorm(slot_dim)

        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_k = nn.Linear(dim, slot_dim, bias=False)
        self.project_v = nn.Linear(dim, slot_dim, bias=False)

        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(nn.Linear(slot_dim, slot_dim * 2), nn.Mish(), nn.Linear(slot_dim * 2, slot_dim))

        self.register_buffer("slot_mu", torch.randn(1, num_slots, slot_dim))
        self.register_buffer("slot_logsigma", torch.randn(1, num_slots, slot_dim))
        self.register_buffer("prev_slots", torch.zeros(1, num_slots, slot_dim))
        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.long))

    def forward(self, inputs):
        """Applies slot attention and periodic bipartite matching."""
        b = inputs.size(0)
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)

        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)
        v = self.project_v(inputs)

        mu = self.slot_mu.expand(b, self.num_slots, -1)
        sigma = self.slot_logsigma.exp().expand(b, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)

        for _ in range(self.iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            q = self.project_q(slots_norm)

            attn_logits = torch.bmm(q, k.transpose(1, 2)) * (self.slot_dim**-0.5)
            attn = torch.softmax(attn_logits.float(), dim=-1).to(attn_logits.dtype)
            attn_sum = attn.sum(dim=-1, keepdim=True) + 1e-5
            updates = torch.bmm(attn, v) / attn_sum

            slots_flat = slots_prev.reshape(-1, self.slot_dim)
            updates_flat = updates.reshape(-1, self.slot_dim)

            new_slots_flat = self.gru(updates_flat, slots_flat)
            slots = new_slots_flat.view(b, self.num_slots, self.slot_dim)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        if self.prev_slots.size(0) != b:
            self.prev_slots = torch.zeros(b, self.num_slots, self.slot_dim, device=inputs.device)

        self.step_counter += 1
        is_keyframe = self.step_counter.item() % self.keyframe_interval == 0

        with torch.no_grad():
            matched_slots = torch.zeros_like(slots)
            matched_slots[:, 0] = slots[:, 0]

            if is_keyframe:
                import scipy.optimize

                cost_matrix_gpu_full = torch.cdist(slots.detach(), self.prev_slots.detach(), p=2.0)
                cost_matrix_np = cost_matrix_gpu_full.cpu().numpy()

                for i in range(b):
                    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix_np[i])
                    for r, c in zip(row_ind, col_ind):
                        if r != 0 and c != 0:
                            matched_slots[i, c] = slots[i, r]
            else:
                curr_slots_gpu = slots.detach()
                prev_slots_gpu = self.prev_slots.detach()
                diff = curr_slots_gpu.unsqueeze(2) - prev_slots_gpu.unsqueeze(1)
                cost_matrix_gpu = torch.norm(diff, dim=-1)

                cost_matrix_gpu[:, 0, :] = float("inf")
                cost_matrix_gpu[:, :, 0] = float("inf")

                batch_idx = torch.arange(b, device=slots.device)
                for _ in range(1, self.num_slots):
                    min_vals, row_indices = torch.min(cost_matrix_gpu, dim=2)
                    _, r = torch.min(min_vals, dim=1)

                    c = row_indices[batch_idx, r]
                    valid_mask = cost_matrix_gpu[batch_idx, r, c] != float("inf")

                    valid_b = batch_idx[valid_mask]
                    valid_r = r[valid_mask]
                    valid_c = c[valid_mask]

                    matched_slots[valid_b, valid_c] = slots[valid_b, valid_r]
                    cost_matrix_gpu[valid_b, valid_r, :] = float("inf")
                    cost_matrix_gpu[valid_b, :, valid_c] = float("inf")

            self.prev_slots = matched_slots.clone()

        return slots + (matched_slots.to(slots.device) - slots).detach()


class GatedCausalReasoner(nn.Module):
    """Predicts causal slot dynamics using gated error masks."""

    def __init__(self, num_slots: int = 8, slot_dim: int = 32, action_dim: int = 16):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        self.action_proj = nn.Linear(action_dim, slot_dim, bias=False)
        self.dynamics_net = nn.Sequential(
            layer_init(nn.Linear(slot_dim * 2, 128)),
            nn.Mish(),
            nn.LayerNorm(128),
            layer_init(nn.Linear(128, slot_dim)),
        )

    @jaxtyped(typechecker=beartype)
    def forward(self, current_slots: Union[SlotState, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        """Predicts causal slot dynamics conditionally on the provided actions.

        Args:
            current_slots: Current slot representations, shape [B, N, D] or [B, N*D].
            action: Action vector, shape [B, A].

        Returns:
            Predicted next slot representations, shape [B, N*D].
        """
        if current_slots.dim() == 2:
            current_slots = current_slots.view(-1, self.num_slots, self.slot_dim)

        b, n, d = current_slots.shape
        action_vec = self.action_proj(action.float()).unsqueeze(1).expand(-1, n, -1)

        slot_action_pair = torch.cat([current_slots, action_vec], dim=-1)
        predicted_delta = self.dynamics_net(slot_action_pair)

        predicted_slots = current_slots + predicted_delta
        return predicted_slots.view(b, -1)

    def __call__(self, current_slots, action=None, use_fast=False):
        if action is None:
            action = torch.zeros(current_slots.size(0), self.action_proj.in_features, device=current_slots.device)
        return self.forward(current_slots, action)

    def compute_gated_loss(self, predicted_slots, target_slots):
        """Computes MSE exclusively for active slots, masking static backgrounds."""
        if predicted_slots.dim() == 2:
            predicted_slots = predicted_slots.view(-1, self.num_slots, self.slot_dim)
        if target_slots.dim() == 2:
            target_slots = target_slots.view(-1, self.num_slots, self.slot_dim)

        slot_activity = torch.norm(target_slots, dim=-1, keepdim=True)
        active_mask = (slot_activity > 0.1).float()

        mse = F.mse_loss(predicted_slots, target_slots, reduction="none")
        gated_mse = mse * active_mask + mse * (~active_mask.bool()).float() * 0.01

        return gated_mse.sum() / (active_mask.sum() + 1e-8)

    def prune_old_connections(self):
        with torch.no_grad():
            for name, param in self.dynamics_net.named_parameters():
                if "weight" in name and param.dim() > 1:
                    weight_mag = param.abs()
                    threshold = torch.quantile(weight_mag, 0.05)
                    mask = weight_mag > threshold
                    param.data.mul_(mask.float())


def clear_optimizer_momentum(optimizer, param_tensor, indices_to_clear):
    if optimizer is None:
        return
    state = optimizer.state.get(param_tensor, None)
    if state is not None:
        if "exp_avg" in state:
            state["exp_avg"][indices_to_clear] = 0.0
        if "exp_avg_sq" in state:
            state["exp_avg_sq"][indices_to_clear] = 0.0


class BlockSparseMoE(nn.Module):
    """Asynchronous Block-Sparse Mixture of Experts utilizing NVMe-VRAM streaming."""

    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim
        self.num_experts = MOE_CFG.num_experts
        self.sparsity = MOE_CFG.sparsity
        self.capacity_factor = MOE_CFG.capacity_factor
        self.routed_dim = int(dim * 1.5)

        self.shared_w1 = nn.Parameter(torch.randn(dim, self.routed_dim) / math.sqrt(dim))
        self.shared_w2 = nn.Parameter(torch.randn(self.routed_dim, dim) / math.sqrt(self.routed_dim))

        self.router = nn.Linear(dim, self.num_experts, bias=False)
        self.stress_to_router = nn.Linear(1, self.num_experts, bias=False)

        self.register_buffer("expert_bias", torch.zeros(self.num_experts))
        self.bias_update_rate = MOE_CFG.bias_update_rate
        self.shared_gate_proj = nn.Linear(dim, 1)
        self.expert_gate_proj = nn.Parameter(torch.randn(self.num_experts, dim, 1) / math.sqrt(dim))
        self.register_buffer("expert_usage_ema", torch.ones(self.num_experts))
        self.register_buffer("is_offloaded", torch.ones(self.num_experts, dtype=torch.bool))

        self.descriptors = {
            i: MemoryEntityDescriptor(
                object_id=f"expert_{i}",
                tier_state=MemoryTierState.COLD_ON_SSD,
                segment_id=i,
                byte_offset=0,
                length=0,
                dtype=torch.float16,
                quant_scheme=0.0,
                quality_score=1.0,
                last_access_step=0,
                predicted_next_use=0.0,
                true_quant_error=0.0,
                quality_band=MemoryQualityBand.GREEN,
            )
            for i in range(self.num_experts)
        }

        self.nvme_dir = os.path.join(".", "moe_nvme_fp16_cache")
        os.makedirs(self.nvme_dir, exist_ok=True)
        self.w1_mmap_path = os.path.join(self.nvme_dir, "w1_expert.bin")
        self.w2_mmap_path = os.path.join(self.nvme_dir, "w2_expert.bin")

        packed_size_w1 = dim * self.routed_dim
        packed_size_w2 = self.routed_dim * dim

        def _map_nvme_tensor(path: str, size: int) -> torch.Tensor:
            import os

            if not os.path.exists(path) or os.path.getsize(path) < size * 2:
                with open(path, "wb") as f:
                    f.truncate(size * 2)
            storage = torch.UntypedStorage.from_file(path, True, size * 2)
            return torch.Tensor(storage).view(torch.float16)[:size]

        self.w1_ssd = _map_nvme_tensor(self.w1_mmap_path, self.num_experts * packed_size_w1).view(
            self.num_experts, packed_size_w1
        )
        self.w2_ssd = _map_nvme_tensor(self.w2_mmap_path, self.num_experts * packed_size_w2).view(
            self.num_experts, packed_size_w2
        )

        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.ram_resident_size = min(self.num_experts, max(4, int((vram_gb - 2.0) / 0.46)))
        else:
            self.ram_resident_size = min(4, self.num_experts)
        self.w1_pinned_staging = torch.zeros(self.ram_resident_size, packed_size_w1, dtype=torch.float16, device="cpu")
        self.w2_pinned_staging = torch.zeros(self.ram_resident_size, packed_size_w2, dtype=torch.float16, device="cpu")
        self.vram_buffer_A_w1 = nn.Parameter(torch.zeros(self.ram_resident_size, dim, self.routed_dim))
        self.vram_buffer_A_w2 = nn.Parameter(torch.zeros(self.ram_resident_size, self.routed_dim, dim))
        self.vram_buffer_B_w1 = torch.zeros(self.ram_resident_size, dim, self.routed_dim, device=MODEL_DEVICE)
        self.vram_buffer_B_w2 = torch.zeros(self.ram_resident_size, self.routed_dim, dim, device=MODEL_DEVICE)
        self.active_buffer = "A"
        self.current_hot_experts = []
        self.io_stream = torch.cuda.Stream()
        self.step_counter = 0

    def prepare_memory_state(
        self, top_k_current: torch.Tensor, top_k_ema: torch.Tensor, top_k_predicted: torch.Tensor
    ):
        """Allocates hardware residency limits for specified active experts."""
        self.step_counter += 1

        if not hasattr(self, "static_experts_initialized"):
            self.current_hot_experts = list(range(self.ram_resident_size))

            for v_idx, e_idx in enumerate(self.current_hot_experts):
                ram_b1 = self.w1_ssd[e_idx].clone()
                ram_b2 = self.w2_ssd[e_idx].clone()
                with torch.no_grad():
                    self.vram_buffer_A_w1[v_idx].copy_(ram_b1.view(self.dim, self.routed_dim), non_blocking=True)
                    self.vram_buffer_A_w2[v_idx].copy_(ram_b2.view(self.routed_dim, self.dim), non_blocking=True)
                self.descriptors[e_idx].tier_state = MemoryTierState.HOT_IN_VRAM

            self.static_experts_initialized = True

        for i in self.current_hot_experts:
            self.descriptors[i].last_access_step = self.step_counter

    def update_topology(self, jepa_error_signal=0.0, optimizer=None, drop_fraction=None):
        """Executes targeted weight sparsification within the active VRAM buffer."""
        if drop_fraction is None:
            drop_fraction = MOE_CFG.drop_fraction

        with torch.no_grad():
            active_w1 = self.vram_buffer_A_w1 if self.active_buffer == "A" else self.vram_buffer_B_w1
            active_w2 = self.vram_buffer_A_w2 if self.active_buffer == "A" else self.vram_buffer_B_w2

            for v_idx, e_idx in enumerate(self.current_hot_experts):
                if self.expert_usage_ema[e_idx] < 1e-4:
                    w1_mag = torch.abs(active_w1[v_idx])
                    w2_mag = torch.abs(active_w2[v_idx])
                    with torch.no_grad():
                        active_w1[v_idx][w1_mag < torch.quantile(w1_mag, drop_fraction)] = 0.0
                        active_w2[v_idx][w2_mag < torch.quantile(w2_mag, drop_fraction)] = 0.0

    def forward(self, x, error_signal=None, inactive_mask=None):
        original_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(1)

        B, S, D = x.shape
        num_tokens = B * S
        x_flat = x.view(num_tokens, D)

        raw_routing_logits = self.router(x_flat)

        if self.training:
            routing_noise = torch.randn_like(raw_routing_logits.float()) * 0.5
            gate_probs = F.softmax(raw_routing_logits.float() + routing_noise, dim=-1)
            expert_usage = gate_probs.mean(dim=0)
            orthogonality_loss = torch.sum(expert_usage * torch.log(expert_usage * self.num_experts + 1e-9))

            active_w1 = self.vram_buffer_A_w1 if self.active_buffer == "A" else self.vram_buffer_B_w1
            flat_w1 = active_w1.view(active_w1.size(0), -1)
            norm_w1 = F.normalize(flat_w1, p=2, dim=1)
            gram_matrix = torch.mm(norm_w1, norm_w1.t())

            gram_matrix.fill_diagonal_(0.0)
            weight_ortho_loss = (
                torch.sum(F.relu(torch.abs(gram_matrix) - 0.2))
                / (active_w1.size(0) * max(1, active_w1.size(0) - 1))
                * 0.01
            )

            if not hasattr(self, "aux_loss"):
                self.aux_loss = 0.0
            self.aux_loss = self.aux_loss + (orthogonality_loss * 0.05) + weight_ortho_loss

        with torch.no_grad():
            expert_probs_sched = F.softmax(raw_routing_logits.float(), dim=-1)
            current_demand = expert_probs_sched.sum(dim=0)
            _, top_k_current = torch.topk(current_demand, self.ram_resident_size)
            _, top_k_ema = torch.topk(self.expert_usage_ema, self.ram_resident_size)

            self.prepare_memory_state(top_k_current, top_k_ema, top_k_current)

            self.is_offloaded.fill_(True)
            for e in self.current_hot_experts:
                self.is_offloaded[e] = False

        if self.training:
            raw_routing_logits = raw_routing_logits + self.expert_bias.unsqueeze(0)

        if error_signal is not None:
            stress_flat = error_signal.view(num_tokens, 1)
            raw_routing_logits = raw_routing_logits + self.stress_to_router(stress_flat)

        safe_routing_logits = 30.0 * torch.tanh(raw_routing_logits / 30.0)

        absent_mask = self.is_offloaded.clone()

        penalty_mask = torch.where(
            absent_mask.unsqueeze(0),
            torch.tensor(-float("inf"), device=x.device, dtype=safe_routing_logits.dtype),
            torch.tensor(0.0, device=x.device, dtype=safe_routing_logits.dtype),
        )
        safe_routing_logits = safe_routing_logits + penalty_mask

        cost_matrix = safe_routing_logits.float()
        transport_matrix = F.softmax(cost_matrix, dim=-1)
        for _ in range(3):
            transport_matrix = transport_matrix / (transport_matrix.sum(dim=0, keepdim=True) + 1e-8)
            transport_matrix = transport_matrix / (transport_matrix.sum(dim=1, keepdim=True) + 1e-8)
        transport_matrix = transport_matrix * cost_matrix.size(0) / self.num_experts
        expert_probs = transport_matrix.t().to(safe_routing_logits.dtype)

        expert_capacity = max(1, int((num_tokens / self.num_experts) * self.capacity_factor))
        topk_weights, topk_indices = torch.topk(expert_probs, k=expert_capacity, dim=1)

        if self.training:
            with torch.no_grad():
                token_allocations = topk_weights.sum(dim=1).float()
                allocation_fraction = token_allocations / (token_allocations.sum() + 1e-8)
                bias_adjustment = ((1.0 / self.num_experts) - allocation_fraction) * self.bias_update_rate
                self.expert_bias.add_(bias_adjustment)

                raw_gate_probs = F.softmax(raw_routing_logits.float(), dim=-1)
                true_expert_usage = raw_gate_probs.mean(dim=0)
                self.expert_usage_ema = 0.99 * self.expert_usage_ema + 0.01 * true_expert_usage

        shared_h = F.gelu(F.linear(x_flat, self.shared_w1.t()))
        shared_out = F.linear(shared_h, self.shared_w2.t())
        shared_alpha = torch.clamp(torch.sigmoid(self.shared_gate_proj(x_flat)), min=0.05, max=1.0)

        routed_out = torch.zeros_like(x_flat)
        active_w1 = self.vram_buffer_A_w1 if self.active_buffer == "A" else self.vram_buffer_B_w1
        active_w2 = self.vram_buffer_A_w2 if self.active_buffer == "A" else self.vram_buffer_B_w2

        for v_idx, e_idx in enumerate(self.current_hot_experts):
            e_tokens_idx = topk_indices[e_idx]
            e_weights = topk_weights[e_idx].unsqueeze(-1)

            x_expert = x_flat[e_tokens_idx]
            a_gate = self.expert_gate_proj[e_idx]

            h_act = F.gelu(F.linear(x_expert, active_w1[v_idx].t()))
            y_act = F.linear(h_act, active_w2[v_idx].t())
            e_alpha = torch.clamp(torch.sigmoid(F.linear(x_expert, a_gate.t())), min=0.05, max=1.0)

            final_expert_out = ((y_act * e_alpha) * e_weights).to(routed_out.dtype)
            routed_out.scatter_add_(0, e_tokens_idx.unsqueeze(-1).expand(-1, D), final_expert_out)

        out = x_flat + (shared_out * shared_alpha) + routed_out
        if not hasattr(self, "aux_loss"):
            self.aux_loss = torch.tensor(0.0, device=x.device)
        return out.view(original_shape)


class ECMoE(BlockSparseMoE):
    """Capacity-bound MoE utilizing Fisher Information approximations for dynamic network scaling."""

    def __init__(self, dim=256):
        # Pre-allocate full memory map inherently inside BlockSparseMoE.
        super().__init__(dim=dim)

        self.stability_gate = StabilityGate(max_norm=15.0)
        self.representation_gate = RepresentationGate(target_variance=1.0, hinge_margin=0.2)

        # Pre-allocate hardware budget and initialize state tensors for Net2Net expansion and distillation.
        self.max_experts = MOE_CFG.num_experts
        self.active_experts_count = max(2, MOE_CFG.num_experts // 2)
        self.register_buffer("active_mask", torch.zeros(self.max_experts, dtype=torch.bool))
        self.active_mask[: self.active_experts_count] = True

        self.register_buffer("expert_routing_load", torch.zeros(self.max_experts, dtype=torch.float32))
        self.register_buffer("loss_ema", torch.tensor(10.0, dtype=torch.float32))
        self.register_buffer("learning_progress", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("expert_correlation_matrix", torch.eye(self.max_experts, dtype=torch.float32))
        self.register_buffer("maturity_matrix", torch.zeros(self.max_experts, dtype=torch.bool))
        self.register_buffer("expert_epochs", torch.zeros(self.max_experts, dtype=torch.long))

    def forward(self, x, error_signal=None, inactive_mask=None):
        x_stable = self.stability_gate(x)
        x_healthy = self.representation_gate(x_stable)

        _ = torch.where(
            self.active_mask,
            torch.tensor(0.0, device=x.device, dtype=x.dtype),
            torch.tensor(-float("inf"), device=x.device, dtype=x.dtype),
        )

        expert_mask = self.active_mask if hasattr(self, "active_mask") else ~self.is_offloaded

        if inactive_mask is None:
            final_mask = ~expert_mask
        else:
            final_mask = inactive_mask | ~expert_mask

        base_out = super().forward(x_healthy, error_signal, inactive_mask=final_mask)

        if self.training:
            self.aux_loss = self.aux_loss + self.representation_gate.latent_health_loss

            if not hasattr(self, "fisher_trace_ema"):
                self.register_buffer("fisher_trace_ema", torch.ones(self.max_experts, device=x.device))
                self.register_buffer("fisher_accumulation_steps", torch.zeros(1, device=x.device))

            if self.router.weight.grad is not None:
                with torch.no_grad():
                    current_fisher = (self.router.weight.grad**2).mean(dim=-1)

                    self.fisher_trace_ema = 0.99 * self.fisher_trace_ema + 0.01 * current_fisher
                    self.fisher_accumulation_steps += 1

                    if hasattr(self, "routing_weights") and self.routing_weights is not None:
                        batch_load = self.routing_weights.mean(dim=0)
                        self.expert_routing_load = 0.99 * self.expert_routing_load + 0.01 * batch_load

                        routing_centered_f32 = (self.routing_weights - batch_load.unsqueeze(0)).float()
                        cov = torch.mm(routing_centered_f32.t(), routing_centered_f32) / (
                            routing_centered_f32.size(0) - 1 + 1e-5
                        )
                        std = torch.sqrt(torch.diag(cov) + 1e-5)
                        corr = (cov / torch.ger(std, std)).to(self.routing_weights.dtype)
                        self.expert_correlation_matrix = 0.99 * self.expert_correlation_matrix + 0.01 * corr

                    if error_signal is not None:
                        current_loss = error_signal.mean().item()
                        self.learning_progress.fill_(
                            0.9 * self.learning_progress.item() + 0.1 * abs(self.loss_ema.item() - current_loss)
                        )
                        self.loss_ema.fill_(0.9 * self.loss_ema.item() + 0.1 * current_loss)

                    active_fisher_mean = self.fisher_trace_ema[self.active_mask].mean().item()

                    self.expert_epochs[self.active_mask] += 1
                    mature_candidates = (
                        (self.expert_epochs > 1000) & (self.expert_routing_load > 0.1) & (self.loss_ema < 1.0)
                    )
                    newly_mature = mature_candidates & ~self.maturity_matrix
                    if newly_mature.any():
                        self.maturity_matrix |= newly_mature
                        for idx in newly_mature.nonzero(as_tuple=True)[0]:
                            if idx < self.vram_buffer_A_w1.size(0):
                                self.vram_buffer_A_w1[idx].requires_grad = False
                                self.vram_buffer_A_w2[idx].requires_grad = False
                                self.vram_buffer_B_w1[idx].requires_grad = False
                                self.vram_buffer_B_w2[idx].requires_grad = False

                    if (
                        self.active_experts_count == self.max_experts
                        and self.maturity_matrix.sum().item() == self.max_experts
                    ):
                        import concurrent.futures

                        def _sleep_consolidation():
                            with torch.no_grad():
                                oldest_idx = torch.topk(self.expert_epochs.float(), 4).indices
                                exp_a = oldest_idx[0]
                                for exp_b in oldest_idx[1:]:
                                    self.w1_ssd[exp_a].copy_((self.w1_ssd[exp_a] + self.w1_ssd[exp_b]) / 2.0)
                                    self.w2_ssd[exp_a].copy_((self.w2_ssd[exp_a] + self.w2_ssd[exp_b]) / 2.0)
                                    self.active_mask[exp_b] = False
                                    self.maturity_matrix[exp_b] = False
                                    self.expert_epochs[exp_b] = 0
                                    self.expert_routing_load[exp_b] = 0.0

                        if not hasattr(self, "_sleep_pool"):
                            self._sleep_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                        self._sleep_pool.submit(_sleep_consolidation)
                        self.active_experts_count -= 3
                    elif self.active_experts_count == self.max_experts:
                        triu_indices = torch.triu_indices(self.max_experts, self.max_experts, offset=1)
                        active_corr = self.expert_correlation_matrix[triu_indices[0], triu_indices[1]]
                        active_corr_mask = self.active_mask[triu_indices[0]] & self.active_mask[triu_indices[1]]

                        if active_corr_mask.any():
                            max_corr_idx = torch.argmax(active_corr * active_corr_mask.float())
                            if active_corr[max_corr_idx] > 0.85:
                                exp_a, exp_b = triu_indices[0, max_corr_idx], triu_indices[1, max_corr_idx]

                                self.w1_ssd[exp_a].copy_((self.w1_ssd[exp_a] + self.w1_ssd[exp_b]) / 2.0)
                                self.w2_ssd[exp_a].copy_((self.w2_ssd[exp_a] + self.w2_ssd[exp_b]) / 2.0)
                                self.active_mask[exp_b] = False
                                self.expert_routing_load[exp_b] = 0.0
                                self.expert_correlation_matrix[exp_b, :] = 0.0
                                self.expert_correlation_matrix[:, exp_b] = 0.0
                                self.expert_correlation_matrix[exp_b, exp_b] = 1.0
                                self.active_experts_count -= 1

                    is_plateau = self.learning_progress.item() < 1e-4
                    high_residual = active_fisher_mean > 1e-4

                    if is_plateau and high_residual and self.active_experts_count < self.max_experts:
                        inactive_indices = (~self.active_mask).nonzero(as_tuple=True)[0]
                        if len(inactive_indices) > 0:
                            new_exp_idx = inactive_indices[0]

                            active_loads = self.expert_routing_load.clone()
                            active_loads[~self.active_mask] = -1.0
                            overloaded_exp_idx = torch.argmax(active_loads)

                            self.w1_ssd[new_exp_idx].copy_(self.w1_ssd[overloaded_exp_idx])
                            self.w2_ssd[new_exp_idx] = torch.randn_like(self.w2_ssd[new_exp_idx]) * 0.01

                            self.active_mask[new_exp_idx] = True
                            self.active_experts_count += 1
                            self.learning_progress.fill_(1.0)
                            self.fisher_accumulation_steps.fill_(0.0)

                            if hasattr(self, "_momentum_clear_callback"):
                                self._momentum_clear_callback(new_exp_idx)
        return base_out


class SuccessorLatentDynamicsModel(nn.Module):
    """State Space architecture combining unrolled recurrence with Successor Representations."""

    def __init__(self, latent_dim=256, action_dim=8, stoch_dim=32, det_dim=224):
        super().__init__()
        self.opt_representation = None
        self.opt_policy = None
        self.opt_causal = None
        self.stoch_dim = stoch_dim
        self.det_dim = det_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.cell = nn.GRUCell(stoch_dim + action_dim, det_dim)

        self.prior_net = nn.Sequential(nn.Linear(det_dim, det_dim), nn.Mish(), nn.Linear(det_dim, stoch_dim * 2))

        self.posterior_net = nn.Sequential(
            nn.Linear(det_dim + latent_dim, det_dim), nn.Mish(), nn.Linear(det_dim, stoch_dim * 2)
        )

        self.successor_projector = nn.Sequential(
            nn.Linear(det_dim, 256),
            nn.LayerNorm(256),
            nn.Linear(256, 256),  # Predicts the cumulative expected future state representation
        )

        self.ensemble_heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(det_dim, 128), nn.Mish(), nn.Linear(128, 256)) for _ in range(3)]
        )

    def compute_kl_balancing_loss(self, prev_state, action_one_hot, current_observation):
        """Computes the KL balancing loss between the prior and posterior state distributions."""
        if prev_state.size(-1) == (self.stoch_dim + self.det_dim):
            z_prev = prev_state[:, : self.stoch_dim]
            h_prev = prev_state[:, self.stoch_dim :]
        else:
            z_prev = torch.zeros(prev_state.size(0), self.stoch_dim, device=prev_state.device, dtype=prev_state.dtype)
            h_prev = torch.zeros(prev_state.size(0), self.det_dim, device=prev_state.device, dtype=prev_state.dtype)

        x = torch.cat([z_prev, action_one_hot.to(prev_state.dtype)], dim=-1).to(self.cell.weight_ih.dtype)
        h_prev = h_prev.to(self.cell.weight_ih.dtype)
        h_next = self.cell(x, h_prev)

        prior_params = self.prior_net(h_next)
        prior_mu, prior_logvar = prior_params.chunk(2, dim=-1)

        post_input = torch.cat([h_next, current_observation], dim=-1)
        post_params = self.posterior_net(post_input)
        post_mu, post_logvar = post_params.chunk(2, dim=-1)

        prior_logvar = -20.0 + F.softplus(5.0 - F.softplus(5.0 - prior_logvar) + 20.0)
        post_logvar = -20.0 + F.softplus(5.0 - F.softplus(5.0 - post_logvar) + 20.0)

        prior_dist = torch.distributions.Normal(prior_mu, torch.exp(0.5 * prior_logvar))
        post_dist = torch.distributions.Normal(post_mu, torch.exp(0.5 * post_logvar))

        post_dist_sg = torch.distributions.Normal(post_mu.detach(), torch.exp(0.5 * post_logvar).detach())
        prior_dist_sg = torch.distributions.Normal(prior_mu.detach(), torch.exp(0.5 * prior_logvar).detach())

        # KL Balancing logic from DreamerV3
        kl_value = (
            0.8 * torch.distributions.kl.kl_divergence(post_dist_sg, prior_dist).mean()
            + 0.2 * torch.distributions.kl.kl_divergence(post_dist, prior_dist_sg).mean()
        )

        z_next = post_dist.rsample()
        state_repr = torch.cat([z_next, h_next], dim=-1)

        return kl_value, state_repr

    def forward(self, prev_state, action_one_hot, return_successor=False):
        """Unrolls prior state predictions conditionally on the provided actions."""
        if prev_state.size(-1) == (self.stoch_dim + self.det_dim):
            z_prev = prev_state[:, : self.stoch_dim]
            h_prev = prev_state[:, self.stoch_dim :]
        else:
            z_prev = torch.zeros(prev_state.size(0), self.stoch_dim, device=prev_state.device, dtype=prev_state.dtype)
            h_prev = torch.zeros(prev_state.size(0), self.det_dim, device=prev_state.device, dtype=prev_state.dtype)

        x = torch.cat([z_prev, action_one_hot.to(prev_state.dtype)], dim=-1).to(self.cell.weight_ih.dtype)
        h_prev = h_prev.to(self.cell.weight_ih.dtype)
        h_next = self.cell(x, h_prev)

        prior_params = self.prior_net(h_next)
        mu, logvar = prior_params.chunk(2, dim=-1)

        # Softplus variance clipping
        safe_logvar = 5.0 - F.softplus(5.0 - logvar.float())
        safe_logvar = -20.0 + F.softplus(safe_logvar + 20.0)
        std = torch.exp(0.5 * safe_logvar).to(logvar.dtype)
        eps = torch.randn_like(std)
        z_next = mu + eps * std

        state_repr = torch.cat([z_next, h_next], dim=-1)

        if return_successor:
            successor_features = self.successor_projector(h_next)
            return state_repr, successor_features

        return state_repr

    def imagine_rollout(
        self, initial_state, policy_net, horizon=16, health_band=0, critic_divergence=0.0, planner_regime=0
    ):
        """Generates internal state trajectories via the latent dynamics model."""
        max_diagnostic_horizon = 3
        max_advice_horizon = 8
        max_teacher_horizon = 16

        # Bound the rollout horizon based on the planner regime:
        # 0: Diagnostic, 1: External Advice, 2: MCTS Teacher.
        if planner_regime == 0:
            safe_horizon = min(horizon, max_diagnostic_horizon)
        elif planner_regime == 1:
            safe_horizon = min(horizon, max_advice_horizon)
        else:
            safe_horizon = min(horizon, max_teacher_horizon)

        states = [initial_state]
        current_state = initial_state

        def _checkpoint_step(state: torch.Tensor) -> torch.Tensor:
            """Evaluates a single unrolled step using gradient checkpointing."""
            ac_out = policy_net(state)
            immediate_logits = ac_out.policy_logits[..., : self.action_dim]

            clamped_logits = torch.clamp(immediate_logits, min=-20.0, max=20.0)
            current_step = 0.0
            if hasattr(policy_net, "global_step"):
                current_step = policy_net.global_step.item()
            elif (self_ref := locals().get("self")) is not None and hasattr(self_ref, "global_step"):
                current_step = self.global_step.item()

            base_tau = 1.0 * (0.999**current_step)
            current_tau = max(0.05, float(base_tau))

            action_one_hot = F.gumbel_softmax(clamped_logits, tau=current_tau, hard=True, dim=-1)

            if action_one_hot.dim() == 3:
                action_one_hot = action_one_hot.squeeze(1)

            return self.forward(state, action_one_hot)

        surprisal_err = 0.0
        surprisal_velocity = 0.0
        getattr(policy_net, "jepa", getattr(self, "jepa", None))

        if not hasattr(self, "_live_state_memory"):
            self._live_state_memory = current_state.detach()

        if current_state is not None:
            live_shift = F.mse_loss(current_state.detach(), self._live_state_memory).item()
            self._live_state_memory = torch.nan_to_num(current_state.detach(), nan=0.0, posinf=1.0, neginf=-1.0)

            current_surprisal = 0.0 if math.isnan(live_shift) else live_shift
            prev_surprisal = getattr(self, "_prev_surprisal_err", current_surprisal)
            if math.isnan(prev_surprisal):
                prev_surprisal = 0.0

            surprisal_velocity = current_surprisal - prev_surprisal
            self._prev_surprisal_err = 0.9 * prev_surprisal + 0.1 * current_surprisal
            surprisal_err = current_surprisal

        _safe_vel = 0.0 if math.isnan(surprisal_velocity) else surprisal_velocity
        _safe_err = 0.0 if math.isnan(surprisal_err) else surprisal_err

        learning_progress_reward = max(0.0, -_safe_vel)

        _exponent = max(-2.0, min(2.0, -0.1 * _safe_err + 0.5 * learning_progress_reward))
        base_lambda = safe_horizon * math.exp(_exponent)

        if math.isnan(base_lambda) or math.isinf(base_lambda):
            base_lambda = float(safe_horizon)

        dynamic_horizon = max(1, min(safe_horizon * 2, int(base_lambda + torch.empty(1).exponential_().item())))

        for _ in range(dynamic_horizon):
            if current_state.requires_grad:
                current_state = checkpoint(
                    _checkpoint_step, current_state, use_reentrant=False, preserve_rng_state=False
                )
            else:
                current_state = _checkpoint_step(current_state)
                states.append(current_state)

        return torch.stack(states, dim=1)
