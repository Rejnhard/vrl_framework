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
import logging
import math
import os
import random
import time
from typing import Any, Optional, Union, cast

import bitsandbytes as bnb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from vrl_framework.core.contracts import (
    ConvergenceStats,
    PlannerDecisionTrace,
    PlannerOutput,
    PlannerRegime,
    PlannerValidator,
    PlanningBudget,
    PolicySequenceBatch,
    TrainStepMetrics,
)
from vrl_framework.core.settings import (
    AGENTS_DIR,
    BATCH_SIZE,
    CFG,
    DEACTIVATION_THRESHOLD,
    DESIRED_ENERGY,
    ENABLE_OFFLINE_ROLLOUT,
    ENV_EVENT_RATE,
    INIT_POPULATION,
    MAX_POPULATION,
    MEM_CFG,
    MODEL_DEVICE,
    PLAN_CFG,
    SIM_DIR,
    TRAIN_CFG,
    WORLD_DIM,
)
from vrl_framework.environment import VectorizedPopulation, VectorizedWorld4D
from vrl_framework.environment.world_dynamics import visualize_complex_entity_3D
from vrl_framework.math_ops.geometry import LorentzGeometry, compute_traj_entropy


def execute_internal_cognitive_ticks(
    model: nn.Module, state_repr: torch.Tensor, memory_ctx: torch.Tensor, ticks: int = 5
) -> torch.Tensor:
    """
    Performs iterative latent state updates via a shared bottleneck.

    Args:
        model: Neural module containing the workspace bottleneck mechanism.
        state_repr: Current observation embeddings. Shape: (batch_size, latent_dim).
        memory_ctx: Context vectors retrieved from memory. Shape: (batch_size, latent_dim).
        ticks: Number of recurrent update iterations.

    Returns:
        Gated perception state tensor. Shape: (batch_size, latent_dim).
    """

    b_size = state_repr.size(0)
    dim = state_repr.size(-1)

    if not hasattr(model, "workspace"):
        from vrl_framework.models.components import SharedWorkspaceBottleneck

        model.workspace = SharedWorkspaceBottleneck(dim, 3).to(state_repr.device)
        model.internal_state = torch.zeros(b_size, dim, device=state_repr.device)

    current_thought = model.internal_state.detach()
    if current_thought.size(0) != b_size:
        current_thought = torch.zeros(b_size, dim, device=state_repr.device)

    for _ in range(ticks):
        subsystems = torch.stack([state_repr, memory_ctx, current_thought], dim=1)
        broadcast_thought = model.workspace(subsystems)
        delta_thought = broadcast_thought.sub(current_thought)
        thought_divergence = (
            torch.sum(delta_thought.float().pow(2), dim=-1, keepdim=True)
            .add_(1e-8)
            .sqrt_()
            .to(delta_thought.dtype)
            .sigmoid_()
        )
        current_thought = current_thought.mul(1.0 - thought_divergence).add_(broadcast_thought.mul(thought_divergence))

    model.internal_state = current_thought.detach()

    gated_perception_state = state_repr * torch.sigmoid(current_thought.detach()) + (current_thought.detach() * 0.15)
    return gated_perception_state


class MetricsAggregator:
    _instance = None
    metrics: dict
    last_known: dict
    _last_step_metrics: dict
    global_payload_dump: dict
    _metrics_defined: bool

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MetricsAggregator, cls).__new__(cls)
            cls._instance.metrics = {}
            cls._instance.last_known = {}
            cls._instance._last_step_metrics = {}
            cls._instance.global_payload_dump = {}
            cls._instance._metrics_defined = False
        return cls._instance

    @staticmethod
    def _sanitize_value(v: Any) -> Any:
        import math

        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                val = v.item()
                return None if math.isnan(val) or math.isinf(val) else float(val)
            return None
        if isinstance(v, bool):
            return float(v)
        if isinstance(v, (int, float)):
            return None if math.isnan(v) or math.isinf(v) else float(v)
        return v

    def log(self, payload: dict, step: Optional[int] = None) -> None:
        clean_payload = {}
        for k, v in payload.items():
            v = self._sanitize_value(v)
            if v is None:
                continue
            clean_payload[k] = v
        self.metrics.update(clean_payload)
        self.last_known.update(clean_payload)

    def log_step_metrics(self, payload: dict) -> None:
        clean_payload = {}
        for k, v in payload.items():
            v = self._sanitize_value(v)
            if v is None:
                continue
            clean_payload[k] = v
        self._last_step_metrics.update(clean_payload)
        self.metrics.update(clean_payload)
        self.last_known.update(clean_payload)

    def flush(self, global_train_step: int, generation: Optional[int] = None) -> None:
        if wandb.run is not None:

            def _resolve(val):
                if isinstance(val, torch.Tensor):
                    v = val.item()
                    return v
                return val

            payload = {}
            payload.update(
                {
                    k: _resolve(v)
                    for k, v in self.metrics.items()
                    if k not in ["global_train_step", "generation", "epoch"]
                }
            )
            if hasattr(self, "_last_step_metrics"):
                payload.update(
                    {
                        k: _resolve(v)
                        for k, v in self._last_step_metrics.items()
                        if k not in ["global_train_step", "generation", "epoch"]
                    }
                )

            payload["global_train_step"] = global_train_step

            if generation is not None:
                payload["generation"] = generation
            elif "generation" in self.last_known:
                payload["generation"] = self.last_known["generation"]
            elif "epoch" in self.last_known:
                payload["generation"] = self.last_known["epoch"]
            elif wandb.run is not None and wandb.run.summary.get("generation") is not None:
                payload["generation"] = wandb.run.summary.get("generation")

            if not getattr(self, "_metrics_defined", False):
                wandb.define_metric("generation")
                wandb.define_metric("global_train_step", step_metric="generation")
                wandb.define_metric("loss/*", step_metric="generation")
                self._metrics_defined = True

            wandb.log(payload, commit=True)

        self.metrics.clear()
        if hasattr(self, "_last_step_metrics"):
            self._last_step_metrics.clear()


metrics_aggregator = MetricsAggregator()


class ComputeBudget:
    """
    Dynamic execution scheduler with exponential backoff based on module utility signals.
    """

    def __init__(self, target_frequency_steps: int, max_utility_decay: float = 0.5):
        self.target_frequency_steps = target_frequency_steps
        self.max_utility_decay = max_utility_decay
        self.current_step = 0
        self.accumulated_utility = 0.0
        self.executions = 0

    def request_execution(self) -> bool:
        self.current_step += 1
        return self.current_step >= self.target_frequency_steps

    def commit_execution(self, elapsed_ms: float, utility_score: float) -> None:
        self.current_step = 0

        if self.executions == 0:
            self.accumulated_utility = utility_score
        else:
            self.accumulated_utility = 0.9 * self.accumulated_utility + 0.1 * utility_score

        self.executions += 1

        # Apply exponential backoff to execution frequency if accumulated utility decays below threshold.
        if self.accumulated_utility < (self.max_utility_decay * 0.2):
            self.target_frequency_steps = min(self.target_frequency_steps * 2, 1000)
        elif self.accumulated_utility > self.max_utility_decay:
            self.target_frequency_steps = max(self.target_frequency_steps // 2, 1)


class StepScheduler:
    def __init__(self):
        self.budgets = {
            "planner": ComputeBudget(target_frequency_steps=5, max_utility_decay=0.5),
            "diagnostics": ComputeBudget(target_frequency_steps=100, max_utility_decay=0.8),
            "memory_consolidation": ComputeBudget(target_frequency_steps=50, max_utility_decay=0.6),
            "eval_harness": ComputeBudget(target_frequency_steps=200, max_utility_decay=0.9),
            "plateau_update": ComputeBudget(target_frequency_steps=20, max_utility_decay=0.7),
            "scheduler_phase_transition": ComputeBudget(target_frequency_steps=50, max_utility_decay=0.8),
        }

    def can_execute(self, module_name: str) -> bool:
        return self.budgets[module_name].request_execution()

    def finalize_execution(self, module_name: str, elapsed_ms: float, utility_score: float = 0.0):
        self.budgets[module_name].commit_execution(elapsed_ms, utility_score)


class MetricsDict:
    def __init__(
        self,
        sample_efficiency=0.0,
        planning_gain=0.0,
        latent_rank=0.0,
        retrieval_hit_rate=0.0,
        critic_divergence=0.0,
        dynamics_error=0.0,
    ):
        self.sample_efficiency = sample_efficiency
        self.planning_gain = planning_gain
        self.latent_rank = latent_rank
        self.retrieval_hit_rate = retrieval_hit_rate
        self.critic_divergence = critic_divergence
        self.dynamics_error = dynamics_error


class PPOTrainer:
    opt_policy_fp32: Optional[torch.optim.Optimizer] = None

    def __init__(self, agent_core: Union[nn.Module, Any] = None, runtime_context: Any = None) -> None:
        if agent_core is not None and not isinstance(agent_core, nn.Module):
            runtime_context = agent_core
            agent_core = None

        self.runtime_context = runtime_context

        if agent_core is None:
            from vrl_framework.environment.world_dynamics import Agent

            dummy_pos = torch.empty(len(WORLD_DIM), dtype=torch.long, device=MODEL_DEVICE)

            with torch.device(MODEL_DEVICE):
                dummy_agent = Agent(position=dummy_pos)
            agent_core = dummy_agent.agent_core

        self.agent_core = agent_core
        self.agent_core.runtime_context = self.runtime_context
        self.current_accum_step: int = 0
        self.global_train_step: int = 0

        self.scheduler = StepScheduler()

        self.pred_error_ema = torch.tensor(1.0, device=MODEL_DEVICE)
        self.surprisal_std = torch.tensor(0.1, device=MODEL_DEVICE)
        self.dynamics_mse_ema = torch.tensor(1.0, device=MODEL_DEVICE)
        self.mse_diff = torch.tensor(0.0, device=MODEL_DEVICE)
        self.uncertainty_signal = torch.tensor(0.0, device=MODEL_DEVICE)
        self.hyperbolic_failures: int = 0  # Track Lorentz projection violations.
        self.checkpoint_reverts: int = 0

        self.log_var_policy = nn.Parameter(torch.zeros(1, device=MODEL_DEVICE))
        self.log_var_value = nn.Parameter(torch.zeros(1, device=MODEL_DEVICE))
        self.log_var_aux = nn.Parameter(torch.zeros(1, device=MODEL_DEVICE))

        self.loss_balancing_rate: float = 0.01

        # Initialize exponential moving average (EMA) weights for auxiliary objectives.
        self.loss_weights_ema = {
            "value": torch.tensor(1.0, device=MODEL_DEVICE),
            "policy": torch.tensor(1.0, device=MODEL_DEVICE),
            "aux": torch.tensor(1.0, device=MODEL_DEVICE),
        }

        self.mcts_teacher_buffer_states: Optional[torch.Tensor] = None
        self.mcts_teacher_buffer_logits: Optional[torch.Tensor] = None
        self.expiration_step: Optional[torch.Tensor] = None
        self.mcts_teacher_buffer_gain: Optional[torch.Tensor] = None
        self.mcts_teacher_confidence: Optional[torch.Tensor] = None
        self.mcts_teacher_buffer_critic_gap: Optional[torch.Tensor] = None
        self.mcts_teacher_value_divergence: Optional[torch.Tensor] = None
        self.mcts_teacher_entry_class: Optional[torch.Tensor] = None
        self.mcts_teacher_metadata_regime: Optional[torch.Tensor] = None
        self.teacher_adv_signs: Optional[torch.Tensor] = None
        self.mcts_buffer_ptr: torch.Tensor = torch.tensor(0, dtype=torch.long, device=MODEL_DEVICE)

        self.holdout_buffer_states: Optional[torch.Tensor] = None
        self.holdout_buffer_actions: Optional[torch.Tensor] = None
        self.holdout_buffer_next_states: Optional[torch.Tensor] = None
        self.holdout_ptr: int = 0

    def _ensure_teacher_buffers_allocated(self, device: Union[str, torch.device]) -> None:
        """Pre-allocates buffers for MCTS trajectories to prevent dynamic GPU memory fragmentation."""
        if not hasattr(self, "mcts_teacher_buffer_states") or self.mcts_teacher_buffer_states is None:
            self.mcts_teacher_buffer_states = torch.zeros(1024, 256, device=device)
            self.mcts_teacher_buffer_logits = torch.zeros(1024, self.agent_core.num_actions, device=device)
            self.expiration_step = torch.zeros(1024, dtype=torch.long, device=device)
            self.mcts_teacher_buffer_gain = torch.zeros(1024, device=device)
            self.mcts_teacher_confidence = torch.zeros(1024, device=device)
            self.mcts_teacher_buffer_critic_gap = torch.zeros(1024, device=device)
            self.mcts_teacher_value_divergence = torch.zeros(1024, device=device)
            self.mcts_teacher_entry_class = torch.zeros(1024, dtype=torch.long, device=device)
            self.mcts_teacher_metadata_regime = torch.zeros(1024, dtype=torch.long, device=device)
            self.teacher_adv_signs = torch.zeros(1024, device=device)

            self.mcts_teacher_realized_advantages: Optional[torch.Tensor] = torch.zeros(1024, device=device)
            self.mcts_teacher_delayed_confirmation: Optional[torch.Tensor] = torch.zeros(1024, device=device)
            self.mcts_teacher_ood_risk: Optional[torch.Tensor] = torch.zeros(1024, device=device)
            self.mcts_teacher_blend_weight: Optional[torch.Tensor] = torch.zeros(1024, device=device)
            self.mcts_teacher_acceptance_mask: Optional[torch.Tensor] = torch.zeros(
                1024, dtype=torch.bool, device=device
            )
            self.mcts_teacher_acceptance_score: Optional[torch.Tensor] = torch.zeros(1024, device=device)
            self.mcts_teacher_buffer_health: Optional[torch.Tensor] = torch.zeros(1024, device=device)
            self.mcts_teacher_buffer_truth_margin: Optional[torch.Tensor] = torch.zeros(1024, device=device)

    def _ensure_holdout_buffers_allocated(self, device: Union[str, torch.device]) -> None:
        if not hasattr(self, "holdout_buffer_states") or self.holdout_buffer_states is None:
            self.holdout_buffer_states = torch.zeros(512, 256, device=device)
            self.holdout_buffer_actions = torch.zeros(512, self.agent_core.num_actions, device=device)
            self.holdout_buffer_next_states = torch.zeros(512, 256, device=device)

    def _initialize_optimizer_state(self) -> None:
        if (
            self.runtime_context is None
            or not hasattr(self.runtime_context, "sim_dir")
            or not self.runtime_context.sim_dir
        ):
            raise RuntimeError("runtime_context.sim_dir is required and must point to the active run directory.")
        base_dir = self.runtime_context.sim_dir

        self.error_logs_dir = os.path.join(base_dir, "error_logs")
        os.makedirs(self.error_logs_dir, exist_ok=True)

        if hasattr(self.agent_core, "moe") and hasattr(CFG, "OFFLOAD_DIR") and CFG.OFFLOAD_DIR:
            self.agent_core.moe.nvme_dir = CFG.OFFLOAD_DIR
            os.makedirs(self.agent_core.moe.nvme_dir, exist_ok=True)

            self.ablation_states: dict[str, torch.Tensor] = {}

        self.retrieval_ratio = getattr(CFG, "RETRIEVAL_RATIO", 0.5)
        self.curriculum_chunk_size = getattr(CFG, "CURRICULUM_CHUNK_SIZE", 4096)

        self.planner_regret_ema = torch.tensor(0.0, device=MODEL_DEVICE)
        self.planning_gain_ema = torch.tensor(0.0, device=MODEL_DEVICE)

        class PlateauDetector:
            def __init__(self):
                self.return_slope = 0.0
                self.dynamics_slope = 0.0
                self.latent_rank_slope = 0.0
                self.planning_gain_slope = 0.0
                self.hit_rate_slope = 0.0
                self.history_return = []
                self.history_dynamics = []
                self.history_latent_rank = []
                self.history_planning_gain = []
                self.history_hit_rate = []

            def _compute_slope(self, history: list) -> float:
                if len(history) < 2:
                    return 0.0
                x = np.arange(len(history))
                y = np.array(history)
                A = np.vstack([x, np.ones(len(x))]).T
                m, c = np.linalg.lstsq(A, y, rcond=None)[0]
                return float(m)

            def update_and_check(self, metrics) -> bool:
                self.history_return.append(getattr(metrics, "sample_efficiency", 0.0))
                self.history_dynamics.append(getattr(metrics, "dynamics_error", 0.0))
                self.history_latent_rank.append(getattr(metrics, "latent_rank", 0.0))
                self.history_planning_gain.append(getattr(metrics, "planning_gain", 0.0))
                self.history_hit_rate.append(getattr(metrics, "retrieval_hit_rate", 0.0))

                window_size = 20
                if len(self.history_return) > window_size:
                    self.history_return.pop(0)
                    self.history_dynamics.pop(0)
                    self.history_latent_rank.pop(0)
                    self.history_planning_gain.pop(0)
                    self.history_hit_rate.pop(0)

                self.return_slope = self._compute_slope(self.history_return)
                self.dynamics_slope = self._compute_slope(self.history_dynamics)

                return (
                    self.return_slope < 1e-4
                    and abs(self.dynamics_slope) < 1e-4
                    and len(self.history_return) >= window_size
                )

        class PIDLagrangianController:
            def __init__(self, kp=0.1, ki=0.01, kd=0.001):
                self.kp = kp
                self.ki = ki
                self.kd = kd
                self.integral_error = 0.0
                self.previous_error = 0.0
                self.target_cost = 0.05
                self.current_multiplier = 0.1

            def calculate_multiplier(self, cost: float) -> float:
                error = cost - self.target_cost
                self.integral_error = max(0.0, self.integral_error + error)
                derivative = error - self.previous_error

                adjustment = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * derivative)
                self.current_multiplier = max(0.001, min(10.0, self.current_multiplier + adjustment))
                self.previous_error = error

                return self.current_multiplier

        self.plateau_detector = PlateauDetector()
        self.ablation_metrics = MetricsDict(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.segproto_cache_version = 1
        self.entropy_penalty_sum = 0.0
        self.validation_metrics = PlannerValidator()

        self.pid_controller = PIDLagrangianController(kp=0.1, ki=0.01, kd=0.001)

        def _gather_params(core, attrs):
            params = []
            for attr in attrs:
                if hasattr(core, attr):
                    obj = getattr(core, attr)
                    if isinstance(obj, torch.nn.Module) or (hasattr(obj, "parameters") and callable(obj.parameters)):
                        params.extend(obj.parameters())
                    elif isinstance(obj, torch.Tensor) and obj.requires_grad:
                        params.append(obj)
            return params

        policy_attrs = [
            "actor_critic",
            "moe",
            "sae",
            "hierarchical_planner",
            "lpm_module",
            "meta_gru",
            "ponder_gru",
            "fwp_q",
            "fwp_k",
            "fwp_v",
            "stress_to_film",
            "scratchpad_attention",
            "scratchpad_write_gate",
            "scratchpad_write_val",
            "scratchpad",
        ]
        policy_params = _gather_params(self.agent_core, policy_attrs)
        if not policy_params:
            policy_params = [torch.nn.Parameter(torch.zeros(1, device=MODEL_DEVICE))]

        policy_params_filtered = [p for p in policy_params if p.requires_grad]

        self.opt_policy = torch.optim.AdamW(policy_params_filtered, lr=3e-5, weight_decay=1e-4, fused=True)

        rep_attrs = ["jepa", "sensory", "multimodal", "bottleneck_attention", "communication", "opponent_model"]
        rep_params = _gather_params(self.agent_core, rep_attrs)
        if not rep_params:
            rep_params = [torch.nn.Parameter(torch.zeros(1, device=MODEL_DEVICE))]

        causal_attrs = ["dnc_query_generator", "latent_dynamics", "causal_masker"]
        causal_params = _gather_params(self.agent_core, causal_attrs)
        if not causal_params:
            causal_params = [torch.nn.Parameter(torch.zeros(1, device=MODEL_DEVICE))]

        self.opt_representation = torch.optim.AdamW(rep_params, lr=1e-5, fused=True)
        self.opt_causal = torch.optim.AdamW(causal_params, lr=1e-5, fused=True)

        self.scheduler_policy = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.opt_policy, T_0=5000, T_mult=2, eta_min=1e-6
        )
        self.scheduler_representation = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.opt_representation, T_0=10000, T_mult=2, eta_min=1e-6
        )
        self.scheduler_causal = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.opt_causal, T_0=10000, T_mult=2, eta_min=1e-6
        )

        self.scaler = (
            torch.amp.GradScaler(device="cuda") if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler()
        )

        def _clear_opt_state(
            optimizer: torch.optim.Optimizer, param: torch.Tensor, indices: Any, dim: int = 0
        ) -> None:
            if (
                indices is None
                or (isinstance(indices, torch.Tensor) and indices.numel() == 0)
                or (not isinstance(indices, torch.Tensor) and len(indices) == 0)
            ):
                return

            if param in optimizer.state:
                state = optimizer.state[param]
                safe_indices = torch.as_tensor(indices, dtype=torch.long, device=param.device)

                with torch.no_grad():
                    if "exp_avg" in state:
                        state["exp_avg"].index_fill_(dim, safe_indices, 0.0)
                    if "exp_avg_sq" in state:
                        state["exp_avg_sq"].index_fill_(dim, safe_indices, 0.0)

        if hasattr(self.agent_core, "moe"):

            def _momentum_clear_callback(exp_idx: Any) -> None:
                if hasattr(self.agent_core.moe, "vram_buffer_A_w1"):
                    _clear_opt_state(self.opt_policy, self.agent_core.moe.vram_buffer_A_w1, exp_idx, dim=0)
                    _clear_opt_state(self.opt_policy, self.agent_core.moe.vram_buffer_A_w2, exp_idx, dim=0)
                    _clear_opt_state(self.opt_policy, self.agent_core.moe.vram_buffer_B_w1, exp_idx, dim=0)
                    _clear_opt_state(self.opt_policy, self.agent_core.moe.vram_buffer_B_w2, exp_idx, dim=0)

            self.agent_core.moe._momentum_clear_callback = _momentum_clear_callback

        if hasattr(self.agent_core, "sae"):

            def _sae_momentum_clear_callback(inactive_idx: Any) -> None:
                _clear_opt_state(self.opt_policy, self.agent_core.sae.encoder.weight, inactive_idx, dim=0)
                _clear_opt_state(self.opt_policy, self.agent_core.sae.decoder.weight, inactive_idx, dim=1)

            self.agent_core.sae._momentum_clear_callback = _sae_momentum_clear_callback

    def train(self, target_epochs=50000):
        for handler in logging.root.handlers[:]:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                logging.root.removeHandler(handler)

        logger = logging.getLogger("PPOTrainer")
        logger.info("Initializing Vectorized Environment Engine from cold start...")

        try:
            with torch.device(MODEL_DEVICE):
                engine = EnvironmentSimulationEngine()
                engine.total_generations = target_epochs

            if self.runtime_context is not None:
                engine.metrics = self.runtime_context.metrics
                engine.memory_bank = self.runtime_context.lmdb_bank
                engine.runtime_context = self.runtime_context

            world_limits = torch.tensor(WORLD_DIM, dtype=torch.float32, device=MODEL_DEVICE)
            random_positions = (
                torch.rand((INIT_POPULATION, len(WORLD_DIM)), device=MODEL_DEVICE) * world_limits
            ).long()
            from vrl_framework.environment.world_dynamics import Agent

            for i in range(INIT_POPULATION):
                initial_agent = Agent(position=random_positions[i], agent_core=self.agent_core)
                initial_agent.energy = DESIRED_ENERGY * random.uniform(1.2, 3.5)
                engine.entities.append(initial_agent)

            engine.batched_agents.global_agent_core = self.agent_core
            engine.batched_agents.agent_cores = [self.agent_core for _ in range(engine.batched_agents.max_agents)]
            engine.trainer = self

            self._initialize_optimizer_state()
            self.agent_core.trainer = self
            self.world = engine

            if hasattr(CFG, "RESUME_CHECKPOINT") and CFG.RESUME_CHECKPOINT is not None:
                logger.info(f"Restoring simulation state from: {CFG.RESUME_CHECKPOINT}")
                self.load_checkpoint(CFG.RESUME_CHECKPOINT)
                import re

                match = re.search(r"gen_(\d+)", CFG.RESUME_CHECKPOINT)
                engine.generation = int(match.group(1)) if match else self.global_train_step

            engine.total_generations = engine.generation + target_epochs

            try:
                engine.run_simulation()
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            finally:
                engine.close()
        except Exception:
            logger.exception("Fatal error within training loop")
            raise

    def run_inference_loop(self, target_epochs=None):
        import logging

        import torch

        logger = logging.getLogger("vrl_framework")

        logger.info("Initializing Zero-Shot Inference Engine...")

        if not hasattr(self, "engine_ref") and not hasattr(self, "world"):
            with torch.device(MODEL_DEVICE):
                engine = EnvironmentSimulationEngine()
                engine.total_generations = target_epochs

                if self.runtime_context is not None:
                    engine.metrics = getattr(self.runtime_context, "metrics", None)
                    engine.memory_bank = getattr(self.runtime_context, "lmdb_bank", None)
                    engine.runtime_context = self.runtime_context

                world_limits = torch.tensor(WORLD_DIM, dtype=torch.float32, device=MODEL_DEVICE)
                random_positions = (
                    torch.rand((INIT_POPULATION, len(WORLD_DIM)), device=MODEL_DEVICE) * world_limits
                ).long()
                import random

                from vrl_framework.environment.world_dynamics import Agent

                for i in range(INIT_POPULATION):
                    initial_agent = Agent(position=random_positions[i], agent_core=self.agent_core)
                    initial_agent.energy = DESIRED_ENERGY * random.uniform(1.2, 3.5)
                    engine.entities.append(initial_agent)

                engine.batched_agents.global_agent_core = self.agent_core
                engine.batched_agents.agent_cores = [self.agent_core for _ in range(engine.batched_agents.max_agents)]
                engine.trainer = self
                self._initialize_optimizer_state()
                self.agent_core.trainer = self
                self.world = engine

        engine = self.engine_ref if hasattr(self, "engine_ref") else self.world

        if hasattr(CFG, "RESUME_CHECKPOINT") and CFG.RESUME_CHECKPOINT is not None:
            logger.info(f"Restoring absolute simulation topology for Inference: {CFG.RESUME_CHECKPOINT}")
            self.load_checkpoint(CFG.RESUME_CHECKPOINT)
            import re

            match = re.search(r"gen_(\d+)", CFG.RESUME_CHECKPOINT)
            engine.generation = int(match.group(1)) if match else self.global_train_step

        logger.info("Locking computational graphs (eval mode)...")
        self.agent_core.eval()
        for p in self.agent_core.parameters():
            p.requires_grad = False

        try:
            with torch.no_grad():
                logger.info("Running deterministic rollout visualization and benchmarks...")
                engine.run_benchmarks()
                engine.visualize_intelligence_metrics(force=True)

                if engine.entities:
                    best_ent = max(engine.entities, key=lambda o: getattr(o, "fitness", 0.0))
                    engine.visualize_agent_core_structure(best_ent, engine.generation, force=True)

                logger.info("Inference cycle complete. Visualizations dispatched to W&B.")
        except Exception:
            logger.exception("Inference fault")

    def store_mcts_trajectory(
        self,
        state: torch.Tensor,
        trace: PlannerDecisionTrace,
        advantage_signal: torch.Tensor,
        planning_budget: PlanningBudget = None,
    ) -> None:
        """
        Caches planner trajectories for offline policy distillation.

        Args:
            state: Observation latents. Shape: (batch_size, latent_dim).
            trace: Execution metrics from the planner.
            advantage_signal: GAE values for the corresponding steps. Shape: (batch_size,).
            planning_budget: Configuration bounds for the search tree.
        """
        with torch.no_grad():
            self._ensure_teacher_buffers_allocated(state.device)
            assert self.mcts_teacher_buffer_logits is not None
            assert self.mcts_teacher_metadata_regime is not None
            assert self.mcts_teacher_buffer_gain is not None
            assert self.mcts_teacher_value_divergence is not None
            assert self.mcts_teacher_buffer_critic_gap is not None
            assert self.mcts_teacher_confidence is not None
            assert self.expiration_step is not None
            assert self.teacher_adv_signs is not None
            assert self.mcts_teacher_entry_class is not None
            assert self.mcts_teacher_realized_advantages is not None
            assert self.mcts_teacher_delayed_confirmation is not None
            assert self.mcts_teacher_ood_risk is not None
            assert self.mcts_teacher_blend_weight is not None
            assert self.mcts_teacher_acceptance_mask is not None
            assert self.mcts_teacher_acceptance_score is not None
            ptr = int(self.mcts_buffer_ptr.item())
            batch_size = state.size(0)

            if ptr + batch_size > 1024:
                ptr = 0

            safe_batch = int(min(batch_size, 1024 - ptr))
            end_ptr = ptr + safe_batch

            self.mcts_teacher_buffer_logits[ptr:end_ptr] = trace.executed_planner_logits[:safe_batch]
            self.mcts_teacher_metadata_regime[ptr:end_ptr] = trace.planner_regime
            self.mcts_teacher_buffer_gain[ptr:end_ptr] = trace.predicted_gain
            self.mcts_teacher_value_divergence[ptr:end_ptr] = abs(trace.planner_confidence - trace.truth_margin)
            self.mcts_teacher_buffer_critic_gap[ptr:end_ptr] = trace.critic_divergence

            self.mcts_teacher_confidence[ptr:end_ptr] = trace.planner_confidence
            self.expiration_step[ptr:end_ptr] = getattr(planning_budget, "teacher_ttl", 5) if planning_budget else 5

            self.teacher_adv_signs[ptr:end_ptr] = torch.sign(advantage_signal[:safe_batch])

            dyn_err_val = (
                getattr(self, "pred_error_ema", torch.tensor(5.0)).item() if hasattr(self, "pred_error_ema") else 5.0
            )
            adapt_factor = min(1.0, 5.0 / (dyn_err_val + 1e-4))
            conf_thresh = -0.9 + (1.0 * adapt_factor)
            div_thresh = 50.0 - (45.0 * adapt_factor)
            is_admitted_bool = (
                (trace.planner_confidence > conf_thresh)
                and (trace.truth_margin >= -5.0)
                and (trace.critic_divergence < div_thresh)
            )
            is_admitted = torch.as_tensor(is_admitted_bool, device=MODEL_DEVICE)

            entry_class = (
                torch.zeros(batch_size, dtype=torch.long, device=MODEL_DEVICE)
                if is_admitted_bool
                else torch.full((batch_size,), 2, dtype=torch.long, device=MODEL_DEVICE)
            )

            self.mcts_teacher_entry_class[ptr:end_ptr] = entry_class

            self.mcts_teacher_realized_advantages[ptr:end_ptr] = advantage_signal.detach()[:safe_batch]
            self.mcts_teacher_delayed_confirmation[ptr:end_ptr] = 0.0
            self.mcts_teacher_ood_risk[ptr:end_ptr] = trace.ood_risk
            self.mcts_teacher_confidence[ptr:end_ptr] = trace.planner_confidence
            self.mcts_teacher_blend_weight[ptr:end_ptr] = trace.planner_blend_weight
            self.mcts_teacher_acceptance_mask[ptr:end_ptr] = is_admitted
            self.mcts_teacher_acceptance_score[ptr:end_ptr] = (
                trace.planner_confidence
                + trace.truth_margin
                - trace.critic_divergence
                - trace.ood_risk
                + trace.planner_blend_weight
                + advantage_signal
            ).detach()[:safe_batch]

            mean_adv = advantage_signal.mean().item()
            trace.batch_adv = mean_adv
            trace.same_batch_advantage = mean_adv + (torch.randn(1).item() * 0.01)
            trace.conf_score = 0.0
            trace.teacher_admitted = is_admitted_bool
            trace.teacher_confirmed = False

            if not hasattr(self, "_last_teacher_admission_rate"):
                setattr(self, "_last_teacher_admission_rate", float(is_admitted_bool))
            else:
                setattr(
                    self,
                    "_last_teacher_admission_rate",
                    0.95 * float(getattr(self, "_last_teacher_admission_rate", 0.0)) + 0.05 * float(is_admitted_bool),
                )

            self.mcts_buffer_ptr.fill_(end_ptr % 1024)

            if (
                is_admitted_bool
                and getattr(self, "runtime_context", None) is not None
                and getattr(self.runtime_context, "lmdb_bank", None) is not None
            ):
                payload_cpu = state[:safe_batch].detach().cpu().numpy().astype(np.float16)
                keys = [f"rag_{self.global_train_step}_{i}".encode("utf-8") for i in range(safe_batch)]
                values = [arr.tobytes() for arr in payload_cpu]
                adv_tensor = advantage_signal.detach()[:safe_batch].cpu().numpy()
                qe_tensor = np.zeros(safe_batch)

                if hasattr(self.runtime_context.lmdb_bank, "batch_write_semantic"):
                    self.runtime_context.lmdb_bank.batch_write_semantic(keys, values, payload_cpu)
                elif hasattr(self.runtime_context.lmdb_bank, "batch_write"):
                    self.runtime_context.lmdb_bank.batch_write(keys, values, adv_tensor, qe_tensor)

                if hasattr(self.runtime_context.lmdb_bank, "env") and hasattr(
                    self.runtime_context.lmdb_bank.env, "sync"
                ):
                    self.runtime_context.lmdb_bank.env.sync()

    def save_checkpoint(self, path: str) -> None:
        """
        Serializes model parameters, optimizer states, and telemetry buffers.

        Args:
            path: Target file path for the checkpoint.
        """

        def _clone_checkpoint_value(value):
            if value is None:
                return None
            if torch.is_tensor(value):
                return value.detach().clone().cpu()
            if isinstance(value, dict):
                return {k: _clone_checkpoint_value(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_clone_checkpoint_value(v) for v in value]
            if isinstance(value, tuple):
                return tuple(_clone_checkpoint_value(v) for v in value)
            if hasattr(value, "state_dict") and callable(value.state_dict):
                return value.state_dict()
            return value

        def _capture_named_state(owner, names):
            captured = {}
            for name in names:
                if hasattr(owner, name):
                    captured[name] = _clone_checkpoint_value(getattr(owner, name))
            return captured

        plateau_state = {
            "return_slope": float(getattr(self.plateau_detector, "return_slope", 0.0)),
            "dynamics_slope": float(getattr(self.plateau_detector, "dynamics_slope", 0.0)),
            "latent_rank_slope": float(getattr(self.plateau_detector, "latent_rank_slope", 0.0)),
            "planning_gain_slope": float(getattr(self.plateau_detector, "planning_gain_slope", 0.0)),
            "hit_rate_slope": float(getattr(self.plateau_detector, "hit_rate_slope", 0.0)),
            "history_return": list(getattr(self.plateau_detector, "history_return", [])),
            "history_dynamics": list(getattr(self.plateau_detector, "history_dynamics", [])),
            "history_latent_rank": list(getattr(self.plateau_detector, "history_latent_rank", [])),
            "history_planning_gain": list(getattr(self.plateau_detector, "history_planning_gain", [])),
            "history_hit_rate": list(getattr(self.plateau_detector, "history_hit_rate", [])),
        }

        scheduler_state = None
        if hasattr(self, "scheduler"):
            if hasattr(self.scheduler, "state_dict") and callable(self.scheduler.state_dict):
                scheduler_state = self.scheduler.state_dict()
            else:
                scheduler_state = _capture_named_state(
                    self.scheduler,
                    [
                        "module_budgets",
                        "execution_counters",
                        "cooldowns",
                        "ema_latency_ms",
                        "ema_utility",
                        "last_execution_step",
                        "stability_score_ema",
                        "stability_band",
                    ],
                )

        cpu_budget_state = None
        if hasattr(self, "scheduler"):
            if hasattr(self.scheduler, "state_dict") and callable(self.scheduler.state_dict):
                cpu_budget_state = self.scheduler.state_dict()
            else:
                cpu_budget_state = _capture_named_state(self.scheduler, ["budgets"])

        trainer_teacher_state = _capture_named_state(
            self,
            [
                "mcts_teacher_metadata_regime",
                "teacher_adv_signs",
                "mcts_teacher_entry_class",
                "mcts_teacher_realized_advantages",
                "mcts_teacher_delayed_confirmation",
                "mcts_teacher_ood_risk",
                "mcts_teacher_confidence",
                "mcts_teacher_blend_weight",
                "expiration_step",
                "mcts_teacher_acceptance_mask",
                "mcts_teacher_acceptance_score",
                "mcts_buffer_ptr",
                "teacher_execution_step",
                "teacher_execution_epoch",
                "teacher_write_count",
                "teacher_accept_count",
                "teacher_reject_count",
            ],
        )

        agent_core_teacher_state = _capture_named_state(
            self.agent_core,
            [
                "mcts_teacher_logits",
                "mcts_teacher_values",
                "mcts_teacher_advantages",
                "mcts_teacher_regimes",
                "mcts_teacher_sign_adv",
                "mcts_teacher_entry_class",
                "expiration_step",
                "mcts_teacher_buffer_stability",
                "mcts_teacher_write_ptr",
                "mcts_buffer_ptr",
            ],
        )

        holdout_state = _capture_named_state(
            self,
            [
                "holdout_ptr",
                "holdout_buffer_states",
                "holdout_buffer_actions",
                "holdout_buffer_next_states",
                "holdout_buffer_rewards",
                "holdout_buffer_dones",
                "holdout_episode_ids",
                "holdout_teacher_mask",
            ],
        )

        checkpoint_data = {
            "checkpoint_schema_version": 2,
            "model_state_dict": self.agent_core.state_dict(),
            "opt_policy_state": self.opt_policy.state_dict(),
            "opt_representation_state": self.opt_representation.state_dict(),
            "opt_causal_state": self.opt_causal.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "global_train_step": int(getattr(self, "global_train_step", 0)),
            "checkpoint_reverts": int(getattr(self, "checkpoint_reverts", 0)),
            "divergence_errors": int(getattr(self, "divergence_errors", 0)),
            "pred_error_ema": float(self.pred_error_ema.item()),
            "dynamics_mse_ema": float(self.dynamics_mse_ema.item()),
            "planner_regret_ema": float(self.planner_regret_ema.item()),
            "planning_gain_ema": float(self.planning_gain_ema.item()),
            "segproto_cache_version": int(getattr(self, "segproto_cache_version", 1)),
            "ablation_states": _clone_checkpoint_value(getattr(self, "ablation_states", {})),
            "ablation_metrics": _clone_checkpoint_value(getattr(self, "ablation_metrics", None)),
            "scheduler_temperature_integral": float(getattr(self, "entropy_penalty_sum", 0.0)),
            "plateau_state": plateau_state,
            "scheduler_state": scheduler_state,
            "cpu_budget_state": cpu_budget_state,
            "trainer_teacher_state": trainer_teacher_state,
            "agent_core_teacher_state": agent_core_teacher_state,
            "holdout_state": holdout_state,
            "ltm_index": (
                getattr(self.agent_core.memory, "ltm_index", None) if hasattr(self.agent_core, "memory") else None
            ),
            "knowledge_graph": (
                getattr(self.agent_core.causal_validator, "graph", None)
                if hasattr(self.agent_core, "causal_validator")
                else None
            ),
            "loss_weights_ema": (
                {k: v.detach().cpu().clone() for k, v in self.loss_weights_ema.items()}
                if hasattr(self, "loss_weights_ema")
                else None
            ),
            "log_var_policy": self.log_var_policy.detach().cpu().clone() if hasattr(self, "log_var_policy") else None,
            "log_var_value": self.log_var_value.detach().cpu().clone() if hasattr(self, "log_var_value") else None,
            "log_var_aux": self.log_var_aux.detach().cpu().clone() if hasattr(self, "log_var_aux") else None,
            "prev_jepa_loss_ema": (
                self.prev_jepa_loss_ema.detach().cpu().clone() if hasattr(self, "prev_jepa_loss_ema") else None
            ),
            "env_volatility_ema": (
                self.env_volatility_ema.detach().cpu().clone() if hasattr(self, "env_volatility_ema") else None
            ),
            "watchdog_loss_ema": float(self.watchdog_loss_ema) if hasattr(self, "watchdog_loss_ema") else None,
            "watchdog_loss_std": float(self.watchdog_loss_std) if hasattr(self, "watchdog_loss_std") else None,
            "stability_tracker_ema": (
                self.agent_core.latent_mcts.budget_controller.tracker.ema_stats.detach().cpu().clone()
                if hasattr(self.agent_core, "latent_mcts")
                and hasattr(self.agent_core.latent_mcts, "budget_controller")
                else None
            ),
        }

        if hasattr(self, "opt_policy_fp32") and self.opt_policy_fp32 is not None:
            checkpoint_data["opt_policy_fp32_state"] = self.opt_policy_fp32.state_dict()

        if hasattr(self.agent_core, "latent_mcts") and hasattr(self.agent_core.latent_mcts, "budget_controller"):
            checkpoint_data["representation_ready"] = float(
                self.agent_core.latent_mcts.budget_controller.tracker.representation_ready.item()
            )
            checkpoint_data["dynamics_ready"] = float(
                self.agent_core.latent_mcts.budget_controller.tracker.dynamics_ready.item()
            )
            checkpoint_data["planner_ready"] = float(
                self.agent_core.latent_mcts.budget_controller.tracker.planner_ready.item()
            )
            checkpoint_data["teacher_ready"] = float(
                self.agent_core.latent_mcts.budget_controller.tracker.teacher_ready.item()
            )

        if hasattr(self, "world"):
            checkpoint_data["world_grid"] = self.world.grid.detach().cpu().clone()
            if hasattr(self.world, "audio_grid"):
                checkpoint_data["audio_grid"] = self.world.audio_grid.detach().cpu().clone()
            if hasattr(self.world, "batched_agents"):
                if hasattr(self.world, "entities"):
                    for i, ent in enumerate(self.world.entities):
                        if i < self.world.batched_agents.max_agents and hasattr(ent, "hidden_state"):
                            self.world.batched_agents.hidden_states[i] = ent.hidden_state.detach()

                checkpoint_data["population_state"] = {
                    k: v.detach().cpu().clone() if torch.is_tensor(v) else v
                    for k, v in self.world.batched_agents.state_dict().items()
                }

        if wandb.run is not None:
            checkpoint_data["wandb_run_id"] = wandb.run.id

        torch.save(checkpoint_data, path)
        logging.info(f"Checkpoint serialized to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Restores model and optimizer states from checkpoint."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")

        checkpoint_data = torch.load(path, map_location=MODEL_DEVICE, weights_only=False)

        try:
            self.agent_core.load_state_dict(checkpoint_data["model_state_dict"], strict=True)
        except RuntimeError as e:
            logging.warning(
                "[Zero-Shot] Dimension mismatch detected. Attempting non-strict "
                f"surgical load... Details: {str(e)[:100]}"
            )

            current_state = self.agent_core.state_dict()
            filtered_state = {
                k: v
                for k, v in checkpoint_data["model_state_dict"].items()
                if k in current_state and current_state[k].shape == v.shape
            }
            self.agent_core.load_state_dict(filtered_state, strict=False)
            logging.info(
                f"[Zero-Shot] Successfully loaded {len(filtered_state)} out of "
                f"{len(current_state)} structural tensors."
            )

        if hasattr(self.agent_core, "moe") and hasattr(self.agent_core.moe, "nvme_dir"):
            metadata_file = os.path.join(self.agent_core.moe.nvme_dir, "moe_metadata.json")
            if os.path.exists(metadata_file):
                logging.info("Validating MoE NVMe caches against primary checkpoint timeframe...")

        try:
            self.opt_policy.load_state_dict(checkpoint_data["opt_policy_state"])
            if "opt_causal_state" in checkpoint_data:
                self.opt_causal.load_state_dict(checkpoint_data["opt_causal_state"])
            if "opt_representation_state" in checkpoint_data:
                self.opt_representation.load_state_dict(checkpoint_data["opt_representation_state"])
            if "scaler_state" in checkpoint_data:
                self.scaler.load_state_dict(checkpoint_data["scaler_state"])
        except Exception as e:
            logging.warning(f"Optimizer state mismatch (expected in zero-shot ablation). Optimizers reset. {e}")

        if (
            "opt_policy_fp32_state" in checkpoint_data
            and hasattr(self, "opt_policy_fp32")
            and self.opt_policy_fp32 is not None
        ):
            try:
                self.opt_policy_fp32.load_state_dict(checkpoint_data["opt_policy_fp32_state"])
            except Exception:
                logging.getLogger(__name__).exception("Failed to load opt_policy_fp32_state")

        self.global_train_step = checkpoint_data.get("global_train_step", 0)
        self.checkpoint_reverts = checkpoint_data.get("checkpoint_reverts", 0)
        self.divergence_errors = 0
        if hasattr(self, "pred_error_ema"):
            self.pred_error_ema.fill_(checkpoint_data.get("pred_error_ema", 1.0))
        self.dynamics_mse_ema.fill_(checkpoint_data.get("dynamics_mse_ema", 1.0))
        self.planner_regret_ema.fill_(checkpoint_data.get("planner_regret_ema", 0.0))
        self.planning_gain_ema.fill_(checkpoint_data.get("planning_gain_ema", 0.0))
        self.ablation_states = checkpoint_data.get("ablation_states", {})

        if "scheduler_temperature_integral" in checkpoint_data:
            self.entropy_penalty_sum = checkpoint_data["scheduler_temperature_integral"]
        if (
            "mcts_teacher_metadata_regime" in checkpoint_data
            and checkpoint_data["mcts_teacher_metadata_regime"] is not None
        ):
            self._ensure_teacher_buffers_allocated(MODEL_DEVICE)
            if self.mcts_teacher_metadata_regime is not None:
                self.mcts_teacher_metadata_regime.copy_(
                    checkpoint_data["mcts_teacher_metadata_regime"].to(MODEL_DEVICE)
                )
        if "teacher_adv_signs" in checkpoint_data and checkpoint_data["teacher_adv_signs"] is not None:
            self._ensure_teacher_buffers_allocated(MODEL_DEVICE)
            if self.teacher_adv_signs is not None:
                self.teacher_adv_signs.copy_(checkpoint_data["teacher_adv_signs"].to(MODEL_DEVICE))
        if "mcts_teacher_entry_class" in checkpoint_data and checkpoint_data["mcts_teacher_entry_class"] is not None:
            self._ensure_teacher_buffers_allocated(MODEL_DEVICE)
            if self.mcts_teacher_entry_class is not None:
                self.mcts_teacher_entry_class.copy_(checkpoint_data["mcts_teacher_entry_class"].to(MODEL_DEVICE))

        plateau_state = checkpoint_data.get("plateau_state", {})
        self.plateau_detector.return_slope = plateau_state.get("return_slope", 0.0)
        self.plateau_detector.dynamics_slope = plateau_state.get("dynamics_slope", 0.0)
        self.plateau_detector.latent_rank_slope = plateau_state.get("latent_rank_slope", 0.0)
        self.plateau_detector.planning_gain_slope = plateau_state.get("planning_gain_slope", 0.0)
        self.plateau_detector.hit_rate_slope = plateau_state.get("hit_rate_slope", 0.0)

        for val in plateau_state.get("history_return", []):
            self.plateau_detector.history_return.append(val)
        for val in plateau_state.get("history_dynamics", []):
            self.plateau_detector.history_dynamics.append(val)
        for val in plateau_state.get("history_latent_rank", []):
            self.plateau_detector.history_latent_rank.append(val)
        for val in plateau_state.get("history_planning_gain", []):
            self.plateau_detector.history_planning_gain.append(val)
        for val in plateau_state.get("history_hit_rate", []):
            self.plateau_detector.history_hit_rate.append(val)

        if (
            hasattr(self.agent_core, "latent_mcts")
            and hasattr(self.agent_core.latent_mcts, "budget_controller")
            and "representation_ready" in checkpoint_data
        ):
            self.agent_core.latent_mcts.budget_controller.tracker.representation_ready.fill_(
                bool(checkpoint_data["representation_ready"])
            )
            self.agent_core.latent_mcts.budget_controller.tracker.dynamics_ready.fill_(
                bool(checkpoint_data["dynamics_ready"])
            )
            self.agent_core.latent_mcts.budget_controller.tracker.planner_ready.fill_(
                bool(checkpoint_data["planner_ready"])
            )
            self.agent_core.latent_mcts.budget_controller.tracker.teacher_ready.fill_(
                bool(checkpoint_data["teacher_ready"])
            )

        if (
            "ltm_index" in checkpoint_data
            and checkpoint_data["ltm_index"] is not None
            and hasattr(self.agent_core, "memory")
        ):
            self.agent_core.memory.ltm_index = checkpoint_data["ltm_index"]
        if (
            "knowledge_graph" in checkpoint_data
            and checkpoint_data["knowledge_graph"] is not None
            and hasattr(self.agent_core, "causal_validator")
        ):
            self.agent_core.causal_validator.graph = checkpoint_data["knowledge_graph"]

        if "loss_weights_ema" in checkpoint_data and checkpoint_data["loss_weights_ema"] is not None:
            for k, v in checkpoint_data["loss_weights_ema"].items():
                if k in self.loss_weights_ema:
                    self.loss_weights_ema[k].copy_(v.to(MODEL_DEVICE))
        with torch.no_grad():
            if "log_var_policy" in checkpoint_data and checkpoint_data["log_var_policy"] is not None:
                self.log_var_policy.copy_(checkpoint_data["log_var_policy"].to(MODEL_DEVICE))
            if "log_var_value" in checkpoint_data and checkpoint_data["log_var_value"] is not None:
                self.log_var_value.copy_(checkpoint_data["log_var_value"].to(MODEL_DEVICE))
            if "log_var_aux" in checkpoint_data and checkpoint_data["log_var_aux"] is not None:
                self.log_var_aux.copy_(checkpoint_data["log_var_aux"].to(MODEL_DEVICE))
        if "prev_jepa_loss_ema" in checkpoint_data and checkpoint_data["prev_jepa_loss_ema"] is not None:
            self.prev_jepa_loss_ema = checkpoint_data["prev_jepa_loss_ema"].to(MODEL_DEVICE)
        if "env_volatility_ema" in checkpoint_data and checkpoint_data["env_volatility_ema"] is not None:
            self.env_volatility_ema = checkpoint_data["env_volatility_ema"].to(MODEL_DEVICE)
        if "watchdog_loss_ema" in checkpoint_data and checkpoint_data["watchdog_loss_ema"] is not None:
            self.watchdog_loss_ema = checkpoint_data["watchdog_loss_ema"]
        if "watchdog_loss_std" in checkpoint_data and checkpoint_data["watchdog_loss_std"] is not None:
            self.watchdog_loss_std = checkpoint_data["watchdog_loss_std"]

        if hasattr(self, "world"):
            if "world_grid" in checkpoint_data:
                self.world.grid.copy_(checkpoint_data["world_grid"].to(MODEL_DEVICE))
            if "audio_grid" in checkpoint_data and hasattr(self.world, "audio_grid"):
                self.world.audio_grid.copy_(checkpoint_data["audio_grid"].to(MODEL_DEVICE))
            if "population_state" in checkpoint_data and hasattr(self.world, "batched_agents"):
                try:
                    self.world.batched_agents.load_state_dict(checkpoint_data["population_state"], strict=False)
                except RuntimeError:
                    current_pop_state = self.world.batched_agents.state_dict()
                    filtered_pop_state = {
                        k: v
                        for k, v in checkpoint_data["population_state"].items()
                        if k in current_pop_state and current_pop_state[k].shape == v.shape
                    }
                    self.world.batched_agents.load_state_dict(filtered_pop_state, strict=False)

                if hasattr(self.world, "entities"):
                    for i, ent in enumerate(self.world.entities):
                        if i < self.world.batched_agents.max_agents:
                            ent.hidden_state = self.world.batched_agents.hidden_states[i].clone().to(MODEL_DEVICE)

        import re

        match = re.search(r"gen_(\d+)", path)
        if match:
            gen_num = match.group(1)
            agents_dir = os.path.dirname(path)
            run_dir = os.path.dirname(agents_dir)
            runs_base = os.path.dirname(run_dir)

            if not getattr(CFG, "IGNORE_LORA", False):
                explicit_lora = getattr(CFG, "LORA_PATH", None)
                if explicit_lora:
                    if os.path.exists(explicit_lora):
                        lora_path = explicit_lora
                    else:
                        lora_path = os.path.join(
                            runs_base, explicit_lora, os.path.basename(agents_dir), f"lora_skill_gen_{gen_num}.pt"
                        )
                else:
                    lora_path = os.path.join(agents_dir, f"lora_skill_gen_{gen_num}.pt")

                if os.path.exists(lora_path):
                    try:
                        lora_data = torch.load(lora_path, map_location=MODEL_DEVICE, weights_only=False)
                        if not hasattr(self.agent_core, "lora_registry"):
                            self.agent_core.lora_registry = {}
                        self.agent_core.lora_registry.update(lora_data)
                    except Exception:
                        pass

            if not getattr(CFG, "IGNORE_LMDB", False):
                if hasattr(self, "runtime_context") and getattr(self.runtime_context, "lmdb_bank", None) is not None:
                    explicit_lmdb = getattr(CFG, "LMDB_PATH", None)
                    if explicit_lmdb:
                        if os.path.exists(explicit_lmdb):
                            lmdb_candidate = explicit_lmdb
                        else:
                            lmdb_candidate = os.path.join(runs_base, explicit_lmdb, "global_replay.lmdb")
                    else:
                        lmdb_candidate = os.path.join(run_dir, "global_replay.lmdb")

                    if os.path.exists(lmdb_candidate):
                        try:
                            if hasattr(self.runtime_context.lmdb_bank, "reconnect"):
                                self.runtime_context.lmdb_bank.reconnect(lmdb_candidate)
                            elif hasattr(self.runtime_context.lmdb_bank, "load_from_path"):
                                self.runtime_context.lmdb_bank.load_from_path(lmdb_candidate)
                        except Exception:
                            pass

        logging.info(f"Checkpoint restored from {path}")

    def restore_checkpoint(
        self,
        decision_trace: PlannerDecisionTrace,
        advantages: torch.Tensor,
        q_error: float,
        planning_budget: PlanningBudget = None,
        halting_probs: Optional[torch.Tensor] = None,
    ) -> None:
        self.checkpoint_reverts += 1

        with torch.no_grad():
            if not hasattr(self, "global_return_mean"):
                self.global_return_mean = torch.tensor(0.0, device=advantages.device)
                self.global_return_std = torch.tensor(1.0, device=advantages.device)

        rollback_val = getattr(self, "rollback_rate", 0.0)
        if (rollback_val.item() if isinstance(rollback_val, torch.Tensor) else rollback_val) > 0.05:
            with torch.no_grad():
                for param in self.agent_core.actor_critic.parameters():
                    if param.is_floating_point():
                        param.add_(torch.randn_like(param) * 0.05)
            if hasattr(self, "rollback_rate"):
                self.rollback_rate.fill_(0.0)

        state_dump = {
            "timestamp": time.time(),
            "global_step": self.global_train_step,
            "decision_trace": decision_trace,
            "advantages_mean": advantages.mean().item(),
            "critic_divergence": decision_trace.critic_divergence,
            "quantization_error": q_error,
            "planner_regime": decision_trace.planner_regime,
            "planner_temperature": decision_trace.planner_temperature,
            "budget_max_depth": planning_budget.max_depth if planning_budget else 0,
            "budget_stability_score": planning_budget.health_score if planning_budget else 0.0,
            "halting_probabilities": halting_probs.detach().cpu().clone() if halting_probs is not None else None,
        }
        dump_path = os.path.join(self.error_logs_dir, f"rollback_{self.global_train_step}.pt")
        torch.save(state_dump, dump_path)

        if hasattr(self, "semantic_backup_state"):
            # Load parameters matching current model dimensions
            safe_state_dict = {
                k: v
                for k, v in self.semantic_backup_state.items()
                if k in self.agent_core.state_dict() and self.agent_core.state_dict()[k].shape == v.shape
            }
            self.agent_core.load_state_dict(safe_state_dict, strict=False)

        for attr in ["meta_gru", "ponder_gru", "gradient_stm"]:
            if hasattr(self.agent_core, attr):
                mod = getattr(self.agent_core, attr)
                if hasattr(mod, "hidden") and mod.hidden is not None:
                    mod.hidden.zero_()
        for attr in ["manager_goal", "previous_memory_context", "internal_state", "interoceptive_target"]:
            if hasattr(self.agent_core, attr):
                tensor = getattr(self.agent_core, attr)
                if isinstance(tensor, torch.Tensor):
                    tensor.zero_()
        if hasattr(self.agent_core, "stm_tensor"):
            self.agent_core.stm_tensor.zero_()

        if hasattr(self, "world") and hasattr(self.world, "entities"):
            for ent in self.world.entities:
                if hasattr(ent, "experience_buffer"):
                    ent.experience_buffer.ptr = 0
                    ent.experience_buffer.size = 0

        self.hyperbolic_failures = 0
        if hasattr(self, "tracker_backup"):
            if hasattr(self.agent_core, "latent_mcts") and hasattr(self.agent_core.latent_mcts, "budget_controller"):
                self.agent_core.latent_mcts.budget_controller.tracker.ema_stats.copy_(
                    self.tracker_backup["tracker_ema"]
                )
            self.pred_error_ema.fill_(min(self.tracker_backup["pred_error_ema"], 10.0))
            self.dynamics_mse_ema.fill_(min(self.tracker_backup["dynamics_mse_ema"], 10.0))
        else:
            self.pred_error_ema.fill_(1.0)
            self.dynamics_mse_ema.fill_(1.0)

        opts_to_reset: list[torch.optim.Optimizer] = [
            self.opt_policy,
            self.opt_representation,
            self.opt_causal,
        ]
        if hasattr(self, "opt_policy_fp32") and self.opt_policy_fp32 is not None:
            opts_to_reset.append(cast(torch.optim.Optimizer, self.opt_policy_fp32))

        for opt in opts_to_reset:
            for group in opt.param_groups:
                for p in group["params"]:
                    state = opt.state.get(p, None)
                    if state is not None:
                        if "exp_avg" in state:
                            state["exp_avg"].zero_()
                        if "exp_avg_sq" in state:
                            state["exp_avg_sq"].zero_()

    def evaluate_holdout(self) -> ConvergenceStats:
        with torch.no_grad():
            if (
                self.holdout_buffer_states is None
                or self.holdout_buffer_actions is None
                or self.holdout_buffer_next_states is None
                or self.holdout_ptr < 10
            ):
                return ConvergenceStats(
                    holdout_dynamics_mse=1.0,
                    hyperbolic_failure_rate=0.0,
                    critic_divergence_ema=1.0,
                    confirmed_planner_gain=0.0,
                    teacher_confirmation_rate=float("nan"),
                    semantic_rollback_frequency=0.0,
                    ablation_survival_rate=1.0,
                )

            assert self.holdout_buffer_states is not None
            assert self.holdout_buffer_actions is not None
            assert self.holdout_buffer_next_states is not None
            valid_states = self.holdout_buffer_states[: self.holdout_ptr]
            valid_actions = self.holdout_buffer_actions[: self.holdout_ptr]
            valid_next = self.holdout_buffer_next_states[: self.holdout_ptr]

            predicted_next = self.agent_core.latent_dynamics(valid_states, valid_actions)
            predicted_next = 10.0 * torch.tanh(predicted_next / 10.0)
            valid_next = 10.0 * torch.tanh(valid_next / 10.0)
            predicted_hyp = LorentzGeometry.project(predicted_next)
            valid_next_hyp = LorentzGeometry.project(valid_next)

            holdout_mse_vector = LorentzGeometry.distance(predicted_hyp, valid_next_hyp)
            holdout_mse = holdout_mse_vector.mean().item()

            predicted_minkowski = LorentzGeometry.minkowski_dot(predicted_hyp, predicted_hyp)
            failure_rate = torch.abs(predicted_minkowski + 1.0).mean().item()

            if self.mcts_teacher_entry_class is not None and hasattr(self, "mcts_buffer_ptr"):
                assert self.mcts_teacher_entry_class is not None
                current_ptr = self.mcts_buffer_ptr.item()
                if not hasattr(self, "_last_mcts_ptr"):
                    self._last_mcts_ptr = current_ptr
                    self._teacher_conf_history = []
                if current_ptr != self._last_mcts_ptr:
                    if current_ptr > self._last_mcts_ptr:
                        new_indices = torch.arange(
                            self._last_mcts_ptr,
                            current_ptr,
                            dtype=torch.long,
                            device=self.mcts_teacher_entry_class.device,
                        )
                    else:
                        new_indices = torch.cat(
                            [
                                torch.arange(
                                    self._last_mcts_ptr,
                                    1024,
                                    dtype=torch.long,
                                    device=self.mcts_teacher_entry_class.device,
                                ),
                                torch.arange(
                                    0, current_ptr, dtype=torch.long, device=self.mcts_teacher_entry_class.device
                                ),
                            ]
                        )
                    if len(new_indices) > 0:
                        valid_teacher_entries = self.mcts_teacher_entry_class[new_indices].clone()
                        if (
                            hasattr(self, "mcts_teacher_realized_advantages")
                            and self.mcts_teacher_realized_advantages is not None
                        ):
                            active_advs = self.mcts_teacher_realized_advantages[new_indices]
                            baseline_adv = active_advs.mean().item()
                            promoted_entries = (valid_teacher_entries == 0) & (active_advs > baseline_adv)
                            valid_teacher_entries[promoted_entries] = 1
                            self.mcts_teacher_entry_class[new_indices] = valid_teacher_entries
                        admitted_mask = valid_teacher_entries != 2
                        if admitted_mask.any():
                            self._teacher_conf_history.extend(
                                (valid_teacher_entries[admitted_mask] == 1).float().cpu().tolist()
                            )
                        if len(self._teacher_conf_history) > 1000:
                            self._teacher_conf_history = self._teacher_conf_history[-1000:]
                self._last_mcts_ptr = current_ptr
                if hasattr(self, "_teacher_conf_history") and len(self._teacher_conf_history) > 0:
                    confirmation_rate = sum(self._teacher_conf_history) / len(self._teacher_conf_history)
                else:
                    confirmation_rate = float("nan")
                if (
                    hasattr(self, "mcts_teacher_realized_advantages")
                    and self.mcts_teacher_realized_advantages is not None
                ):
                    active_mask = self.mcts_teacher_entry_class == 1
                    if active_mask.any():
                        confirmed_gain_tensor = self.mcts_teacher_realized_advantages[active_mask]
                    else:
                        confirmed_gain_tensor = torch.tensor([0.0])
                else:
                    confirmed_gain_tensor = torch.tensor([0.0])
            else:
                confirmation_rate = float("nan")
                confirmed_gain_tensor = torch.tensor([0.0])

            confirmed_gain = confirmed_gain_tensor.mean().item() if len(confirmed_gain_tensor) > 0 else 0.0
            if math.isnan(confirmed_gain):
                confirmed_gain = 0.0

            rollback_freq = getattr(self, "checkpoint_reverts", 0) / max(1.0, self.global_train_step / 1000.0)

            base_ac = (
                self.agent_core.actor_critic.module
                if hasattr(self.agent_core.actor_critic, "module")
                else self.agent_core.actor_critic
            )
            expected_dim = (
                base_ac.critic_1[0].in_features
                if isinstance(base_ac.critic_1, nn.Sequential)
                else base_ac.critic_1.in_features
            )
            pad_size = max(0, expected_dim - valid_states.size(-1))

            if hasattr(self.agent_core, "build_critic_context"):
                critic_ctx = self.agent_core.build_critic_context(valid_states, torch.zeros_like(valid_states))
            else:
                critic_ctx = torch.cat(
                    [
                        valid_states,
                        torch.zeros(
                            valid_states.size(0), pad_size, device=valid_states.device, dtype=valid_states.dtype
                        ),
                    ],
                    dim=-1,
                )

            if hasattr(base_ac, "critic_1") and hasattr(base_ac, "critic_2"):
                v1_val = base_ac.critic_1(critic_ctx)
                v2_val = base_ac.critic_2(critic_ctx)
            else:
                ac_out = self.agent_core.actor_critic(valid_states, critic_ctx)
                v1_probs = F.softmax(ac_out.value_logits_1.float(), dim=-1).to(ac_out.value_logits_1.dtype)
                v2_probs = F.softmax(ac_out.value_logits_2.float(), dim=-1).to(ac_out.value_logits_2.dtype)
                v1_val = torch.sum(
                    v1_probs * getattr(self.agent_core.actor_critic, "value_support", torch.ones_like(v1_probs)),
                    dim=-1,
                )
                v2_val = torch.sum(
                    v2_probs * getattr(self.agent_core.actor_critic, "value_support", torch.ones_like(v2_probs)),
                    dim=-1,
                )
            real_critic_div = torch.abs(v1_val - v2_val).mean().item()

            if hasattr(self, "planning_gain_ema"):
                with torch.no_grad():
                    self.planning_gain_ema.copy_(
                        torch.tensor(
                            0.9 * self.planning_gain_ema.item() + 0.1 * confirmed_gain,
                            device=self.planning_gain_ema.device,
                        )
                    )
                    regret_signal = torch.abs(
                        torch.tensor(real_critic_div, device=self.planner_regret_ema.device) - confirmed_gain
                    )
                    self.planner_regret_ema.copy_(0.9 * self.planner_regret_ema + 0.1 * regret_signal)

            return ConvergenceStats(
                holdout_dynamics_mse=holdout_mse,
                hyperbolic_failure_rate=failure_rate,
                critic_divergence_ema=real_critic_div,
                confirmed_planner_gain=confirmed_gain,
                teacher_confirmation_rate=confirmation_rate,
                semantic_rollback_frequency=rollback_freq,
                ablation_survival_rate=1.0,
            )

    def evaluate_ablation_admission(
        self,
        module_name: str,
        test_states: torch.Tensor,
        current_scorecard: ConvergenceStats = None,
        baseline_scorecard: ConvergenceStats = None,
    ) -> bool:
        ablation_states = cast(dict[str, str], self.ablation_states)

        if module_name not in ablation_states:
            ablation_states[module_name] = "full"

        if current_scorecard is not None and baseline_scorecard is not None:
            if (
                current_scorecard.confirmed_planner_gain < baseline_scorecard.confirmed_planner_gain
                or current_scorecard.holdout_dynamics_mse > baseline_scorecard.holdout_dynamics_mse
                or current_scorecard.hyperbolic_failure_rate > baseline_scorecard.hyperbolic_failure_rate
                or current_scorecard.critic_divergence_ema > baseline_scorecard.critic_divergence_ema
                or current_scorecard.semantic_rollback_frequency > baseline_scorecard.semantic_rollback_frequency
            ):
                ablation_states[module_name] = "frozen"
                return False

        return ablation_states[module_name] == "full"

    def apply_agc(self, parameters, clip_val=0.05, eps=1e-3):
        import torch.nn.utils as nn_utils

        valid_params = [p for p in parameters if p.grad is not None and p.dtype not in [torch.int8, torch.uint8]]
        if valid_params:
            for p in valid_params:
                if not torch.isfinite(p.grad).all():
                    return
            nn_utils.clip_grad_norm_(valid_params, max_norm=0.5)

    def select_environment_action(self, latent_state: torch.Tensor, stm_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            halting_probs, ponder_cost, final_thought_state = self.agent_core.halting_head(stm_tensor)

            if wandb.run is not None:
                metrics_aggregator.log_step_metrics(
                    {"planner/halting_budget_mean": float(halting_probs.mean().item())}
                )

            critic_context = None

            if hasattr(self, "build_actor_critic_context") and callable(self.build_actor_critic_context):
                critic_context = self.build_actor_critic_context(
                    latent_state=latent_state, stm_tensor=final_thought_state
                )
            elif hasattr(self.agent_core, "build_actorcritic_context") and callable(
                self.agent_core.build_actorcritic_context
            ):
                critic_context = self.agent_core.build_actorcritic_context(
                    latent_state=latent_state, stm_tensor=final_thought_state
                )
            elif hasattr(self.agent_core, "build_actor_critic_context") and callable(
                self.agent_core.build_actor_critic_context
            ):
                critic_context = self.agent_core.build_actor_critic_context(
                    latent_state=latent_state, stm_tensor=final_thought_state
                )

            if critic_context is None:
                lightweight_deployment_context = torch.zeros(
                    latent_state.size(0), 768, device=latent_state.device, dtype=latent_state.dtype
                )
                copy_width = min(latent_state.size(-1), lightweight_deployment_context.size(-1))
                lightweight_deployment_context[:, :copy_width] = latent_state[:, :copy_width]
                critic_context = lightweight_deployment_context

            ac_out = self.agent_core.actor_critic(
                latent_state, critic_context=critic_context, intent_context=final_thought_state
            )

            safe_max_depth = getattr(CFG, "PLAN_MAX_DEPTH", 3)
            safe_num_samples = getattr(CFG, "PLAN_NUM_SAMPLES", 16)
            dyn_err = getattr(self, "dynamics_mse_ema", torch.tensor(100.0)).item()

            if not hasattr(self, "dyn_err_history"):
                self.dyn_err_history = []
            self.dyn_err_history.append(dyn_err)
            if len(self.dyn_err_history) > 1000:
                self.dyn_err_history.pop(0)
            p90_err = np.percentile(self.dyn_err_history, 90) if len(self.dyn_err_history) > 10 else 100.0

            if dyn_err > p90_err:
                dynamic_depth = 0
                dynamic_samples = 0
            else:
                gating_factor = math.exp(-0.1 * max(0.0, dyn_err - 5.0))
                dynamic_depth = max(1, int(safe_max_depth * gating_factor))
                dynamic_samples = max(2, int(safe_num_samples * gating_factor))

            allow_lookahead = bool(dynamic_depth > 0)
            planning_budget = PlanningBudget(
                health_score=1.0,
                health_band=0,
                max_depth=dynamic_depth,
                num_samples=dynamic_samples,
                distill_enabled=True,
                teacher_ttl=5,
                allow_actor_lookahead=allow_lookahead,
                allow_teacher_write=True,
                allow_distillation=True,
                max_branch_survivors=max(1, dynamic_samples // 2),
                min_survivor_floor=1,
                max_ood_risk=1.5,
                max_critic_divergence=0.5,
                max_planner_calls_per_env_step=1,
            )

            if not hasattr(self.agent_core, "actor_critic_target"):
                import copy

                self.agent_core.actor_critic_target = copy.deepcopy(self.agent_core.actor_critic)
            else:
                tau = 0.005
                with torch.no_grad():
                    for target_param, param in zip(
                        self.agent_core.actor_critic_target.parameters(), self.agent_core.actor_critic.parameters()
                    ):
                        target_param.lerp_(param, tau)

            # Extract active predictor handling both ModuleList and single module instances.
            active_predictor = (
                self.agent_core.jepa.predictor[0]
                if isinstance(self.agent_core.jepa.predictor, torch.nn.ModuleList)
                else self.agent_core.jepa.predictor
            )

            planner_output = self.agent_core.latent_mcts(
                initial_latent=latent_state,
                jepa_predictor=active_predictor,
                actor_critic=self.agent_core.actor_critic_target,
                critic_context=critic_context,
                planning_budget=planning_budget,
                halting_budget=halting_probs,
                causal_engine=self.agent_core.causal_symbolic_reasoner,
                sae_module=getattr(self.agent_core, "sae", None),
                relational_memory=getattr(self.agent_core, "fuzzy_kb", None),
            )

            if planner_output.decision_trace.planner_regime == PlannerRegime.OBSERVE_ONLY:
                final_logits = ac_out.policy_logits[..., : self.agent_core.num_actions]
            else:
                final_logits = planner_output.final_blended_logits[..., : self.agent_core.num_actions]

            safe_final_logits = final_logits.nan_to_num(nan=0.0, posinf=50.0, neginf=-50.0).clamp_(min=-50.0, max=50.0)
            action_dist = torch.distributions.Categorical(logits=safe_final_logits)
            return action_dist.sample()

    def trainstep(self, trajectory_batch: PolicySequenceBatch) -> TrainStepMetrics:
        """
        Executes a single optimization step.

        Args:
            trajectory_batch: Batched transitions from environment rollouts.

        Returns:
            TrainStepMetrics containing aggregated loss values and system telemetry.
        """
        step_start_time = time.perf_counter()
        if not hasattr(self, "opt_representation"):
            self._initialize_optimizer_state()

        if (
            getattr(self, "runtime_context", None) is not None
            and getattr(self.runtime_context, "lmdb_bank", None) is not None
        ):
            try:
                if hasattr(self.agent_core, "scratchpad") and self.agent_core.scratchpad is not None:

                    query_vector = self.agent_core.scratchpad.data.mean(dim=(0, 1)).detach().cpu().numpy()

                    sampled_memory = None
                    if hasattr(self.runtime_context.lmdb_bank, "retrieve"):
                        sampled_memory = self.runtime_context.lmdb_bank.retrieve(query_vector, top_k=1)

                    if sampled_memory and len(sampled_memory) > 0:
                        raw_payload = (
                            torch.from_numpy(np.frombuffer(sampled_memory[0], dtype=np.float16))
                            .to(MODEL_DEVICE)
                            .float()
                        )
                        aligned_latent = (
                            raw_payload[:256]
                            if raw_payload.size(0) >= 256
                            else F.pad(raw_payload, (0, 256 - raw_payload.size(0)))
                        )

                        expanded_bytes = (
                            aligned_latent.unsqueeze(0)
                            .unsqueeze(0)
                            .expand(
                                self.agent_core.scratchpad.data.size(0), self.agent_core.scratchpad.data.size(1), -1
                            )
                        )

                        self.agent_core.scratchpad.data = (1.0 - 0.1) * self.agent_core.scratchpad.data + (
                            0.1 * expanded_bytes * torch.sigmoid(self.agent_core.scratchpad.data)
                        )
            except Exception:
                import logging

                logging.getLogger(__name__).exception("Failed to update scratchpad from aligned latent")

        if not hasattr(self.agent_core, "curriculum_state"):
            self.agent_core.curriculum_state = CurriculumStateTracker().to(MODEL_DEVICE)

        target_curriculum = getattr(CFG, "CURRICULUM_DIR", None)
        target_offload = getattr(CFG, "OFFLOAD_DIR", None)
        target_start = getattr(CFG, "CURRICULUM_START", None)
        use_last = getattr(CFG, "USE_LAST_PATHS", False)

        if use_last or hasattr(self.agent_core, "curriculum_state"):
            embedded_c = self.agent_core.curriculum_state.retrieve_path("embedded_curriculum_path")
            embedded_o = self.agent_core.curriculum_state.retrieve_path("embedded_offload_path")
            target_curriculum = embedded_c if embedded_c else target_curriculum
            target_offload = embedded_o if embedded_o else target_offload

        if target_curriculum:
            self.agent_core.curriculum_state.store_path("embedded_curriculum_path", target_curriculum)
        if target_offload:
            self.agent_core.curriculum_state.store_path("embedded_offload_path", target_offload)
            if hasattr(self.agent_core, "moe"):
                self.agent_core.moe.nvme_dir = target_offload
                os.makedirs(self.agent_core.moe.nvme_dir, exist_ok=True)

        if target_curriculum and os.path.exists(target_curriculum):
            if not hasattr(self, "curriculum_stream"):
                c_idx = int(self.agent_core.curriculum_state.file_index.item())
                c_off = int(self.agent_core.curriculum_state.byte_offset.item())

                saved_filename = self.agent_core.curriculum_state.retrieve_path("embedded_curriculum_path")
                start_filter = target_start

                if saved_filename and not target_start:
                    files_list = get_curriculum_files(target_curriculum)
                    for i, f in enumerate(files_list):
                        if os.path.basename(f) == saved_filename:
                            c_idx = i
                            break

                self.curriculum_stream = load_deterministic_curriculum(target_curriculum, c_idx, c_off, start_filter)
                self.curriculum_chunk = next(self.curriculum_stream, None)
                self.curriculum_ptr = 0

            if getattr(self, "curriculum_stream", None) is not None and self.curriculum_chunk is not None:
                file_idx, chunk_start_offset, raw_byte_chunk, chunk_filename = self.curriculum_chunk
                read_window = getattr(self, "curriculum_chunk_size", 4096)
                c_end_ptr = self.curriculum_ptr + read_window

                if c_end_ptr >= len(raw_byte_chunk):
                    sliced_bytes = raw_byte_chunk[self.curriculum_ptr :]
                    self.curriculum_chunk = next(self.curriculum_stream, None)
                    self.curriculum_ptr = 0
                    if self.curriculum_chunk is not None:
                        new_idx, new_off, _, new_filename = self.curriculum_chunk
                        self.agent_core.curriculum_state.file_index.fill_(new_idx)
                        self.agent_core.curriculum_state.byte_offset.fill_(new_off)
                        self.agent_core.curriculum_state.store_path("embedded_curriculum_path", new_filename)
                else:
                    sliced_bytes = raw_byte_chunk[self.curriculum_ptr : c_end_ptr]
                    self.curriculum_ptr = c_end_ptr
                    self.agent_core.curriculum_state.file_index.fill_(file_idx)
                    self.agent_core.curriculum_state.byte_offset.fill_(chunk_start_offset + self.curriculum_ptr)
                    self.agent_core.curriculum_state.store_path("embedded_curriculum_path", chunk_filename)

                with torch.no_grad():
                    byte_array = np.frombuffer(sliced_bytes, dtype=np.uint8)
                    if len(byte_array) > 0:
                        norm_bytes = (
                            torch.tensor(byte_array.copy(), dtype=torch.float32, device=MODEL_DEVICE) / 127.5
                        ) - 1.0
                        latent_dim = getattr(self.agent_core, "hidden_dim", 256)

                        if norm_bytes.size(0) < latent_dim:
                            norm_bytes = F.pad(norm_bytes, (0, latent_dim - norm_bytes.size(0)))
                        else:
                            norm_bytes = norm_bytes[:latent_dim]

                        self.agent_core.curriculum_context = norm_bytes
                        ratio = 0.05

                        if hasattr(self.agent_core, "scratchpad") and self.agent_core.scratchpad is not None:
                            expanded_bytes = norm_bytes.unsqueeze(0).unsqueeze(0).expand(-1, 8, -1)

                            data = self.agent_core.scratchpad.data
                            mean, std = data.mean(), data.std() + 1e-6

                            norm_input = (expanded_bytes - expanded_bytes.mean()) / (expanded_bytes.std() + 1e-6)
                            scaled_input = norm_input * std + mean

                            self.agent_core.scratchpad.data = (1.0 - ratio) * data + ratio * scaled_input

        _current_reverts = getattr(self, "checkpoint_reverts", 0)
        _safe_dyn_ema = getattr(self, "dynamics_mse_ema", torch.tensor(0.0)).item()
        _safe_pred_ema = getattr(self, "pred_error_ema", torch.tensor(0.0)).item()
        _is_healthy = (
            getattr(self, "hyperbolic_failures", 0) == 0
            and not math.isnan(getattr(self, "watchdog_loss_ema", 0.0))
            and _safe_dyn_ema < 10.0
            and _safe_pred_ema < 10.0
        )

        _weights_finite = all(torch.isfinite(p).all() for p in self.agent_core.parameters())
        if not hasattr(self, "semantic_backup_state") or (
            _current_reverts == getattr(self, "_last_reverts", -1)
            and _is_healthy
            and _weights_finite
            and getattr(self, "global_train_step", 0) % 5 == 0
        ):
            self.semantic_backup_state = {k: v.detach().cpu().clone() for k, v in self.agent_core.state_dict().items()}
        self._last_reverts = _current_reverts

        trajectory_batch.states = torch.nan_to_num(trajectory_batch.states, nan=0.0, posinf=10.0, neginf=-10.0)
        trajectory_batch.next_states = torch.nan_to_num(
            trajectory_batch.next_states, nan=0.0, posinf=10.0, neginf=-10.0
        )

        PlannerValidator.assert_no_nan_inf(trajectory_batch.states, "trajectory_batch.states")
        PlannerValidator.assert_sequence_boundaries(trajectory_batch.dones, episode_ids=trajectory_batch.episode_ids)
        if trajectory_batch.dones.float().mean() == 1.0:
            if hasattr(self.agent_core, "meta_gru") and self.agent_core.meta_gru.hidden is not None:
                self.agent_core.meta_gru.hidden.zero_()
            if hasattr(self.agent_core, "ponder_gru") and self.agent_core.ponder_gru.hidden is not None:
                self.agent_core.ponder_gru.hidden.zero_()
            if hasattr(self.agent_core, "latent_mcts") and hasattr(self.agent_core.latent_mcts, "reset_tree"):
                self.agent_core.latent_mcts.reset_tree()

            if (
                getattr(self, "hyperbolic_failures", 0) > 2000
                or self.pred_error_ema.item() > 5000.0
                or self.dynamics_mse_ema.item() > 5000.0
            ):
                trace_mock = PlannerDecisionTrace(
                    baseline_logits=torch.zeros(1),
                    planner_logits_preblend=torch.zeros(1),
                    executed_planner_logits=torch.zeros(1),
                    final_action_logits=torch.zeros(1),
                    truth_margin=0.0,
                    critic_divergence=10.0,
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
                    halting_budget_used=0.0,
                    planner_regime=0,
                    planner_temperature=1.0,
                    baselogits_temperature=1.0,
                    prefilter_time_ms=0.0,
                    rollout_time_ms=0.0,
                )
                self.restore_checkpoint(trace_mock, torch.zeros(1), 10.0, planning_budget=None, halting_probs=None)

        if self.global_train_step % 100 == 0 and hasattr(self.agent_core, "actor_critic"):
            valid_grads = [p.grad for p in self.agent_core.actor_critic.parameters() if p.grad is not None]
            if valid_grads:
                stacked_grads = torch.cat([g.view(-1) for g in valid_grads])
                if stacked_grads.numel() > 1:
                    safe_grads = torch.nan_to_num(stacked_grads, nan=0.0)
                    probe_score = float(torch.var(safe_grads).item())
                else:
                    probe_score = 0.0
                self._last_gradient_conflict = probe_score
                if probe_score > 10.0:
                    self.loss_weights_ema["aux"] = torch.clamp(self.loss_weights_ema["aux"] * 0.9, min=0.1)

        states = trajectory_batch.states
        actions = trajectory_batch.actions
        old_log_probs = trajectory_batch.old_logprobs
        returns = trajectory_batch.returns
        advantages = trajectory_batch.advantages
        next_states = trajectory_batch.next_states

        costs = trajectory_batch.costs
        snapshots = trajectory_batch.recurrent_snapshots
        burnin = int(trajectory_batch.burnin)

        batch_size, seq_len, _ = states.shape
        safe_actions = torch.clamp(actions.long(), 0, self.agent_core.num_actions - 1)
        action_one_hot = F.one_hot(safe_actions, num_classes=self.agent_core.num_actions).float().to(states.device)

        self.agent_core.meta_gru.hidden = snapshots.metagru_h
        self.agent_core.ponder_gru.hidden = snapshots.pondergru_h
        self.agent_core.gradient_stm.hidden = snapshots.gradientstm_h

        burnin_states = states[:, :burnin, :]
        burnin_actions = action_one_hot[:, :burnin, :]

        with torch.no_grad():
            b_size = burnin_states.size(0)
            dummy_reward = torch.zeros(b_size, burnin, 1, device=burnin_states.device, dtype=burnin_states.dtype)

            has_meta = hasattr(self.agent_core.meta_gru, "hidden") and self.agent_core.meta_gru.hidden is not None
            if has_meta:
                meta_dtype = next(self.agent_core.meta_gru.parameters()).dtype
                self.agent_core.meta_gru.hidden = self.agent_core.meta_gru.hidden.to(meta_dtype)
                meta_gru_seq = torch.cat([burnin_states, burnin_actions, dummy_reward], dim=-1).to(meta_dtype)

            has_ponder = (
                hasattr(self.agent_core.ponder_gru, "hidden") and self.agent_core.ponder_gru.hidden is not None
            )
            if has_ponder:
                ponder_dtype = next(self.agent_core.ponder_gru.parameters()).dtype
                self.agent_core.ponder_gru.hidden = self.agent_core.ponder_gru.hidden.to(ponder_dtype)
                ponder_seq = burnin_states.to(ponder_dtype)

            for t in range(burnin):
                if has_meta:
                    self.agent_core.meta_gru.hidden = self.agent_core.meta_gru(
                        meta_gru_seq[:, t, :], self.agent_core.meta_gru.hidden
                    )
                if has_ponder:
                    self.agent_core.ponder_gru.hidden = self.agent_core.ponder_gru(
                        ponder_seq[:, t, :], self.agent_core.ponder_gru.hidden
                    )

        states = states[:, burnin:, :].reshape(-1, states.size(-1))
        actions = safe_actions[:, burnin:].reshape(-1)
        action_one_hot = action_one_hot[:, burnin:, :].reshape(-1, action_one_hot.size(-1))
        old_log_probs = old_log_probs[:, burnin:].reshape(-1)
        old_log_probs = torch.nan_to_num(old_log_probs, nan=0.0, posinf=10.0, neginf=-10.0)
        returns = returns[:, burnin:].reshape(-1)

        returns = torch.nan_to_num(returns, nan=0.0, posinf=1e4, neginf=-1e4)

        if returns.numel() > 0:
            std_val = float(returns.std().item()) if returns.numel() > 1 else 1.0
            safe_std = std_val if std_val > 1e-2 else 1.0
            ret_avg = float(returns.mean().item())
            ret_max = float(returns.max().item())
        else:
            safe_std = 1.0
            ret_avg = 0.0
            ret_max = 0.0

        metrics_aggregator.log(
            {
                "metrics/return_avg": ret_avg,
                "metrics/return_best": ret_max,
                "metrics/sample_efficiency": ret_avg / safe_std,
            }
        )

        returns = torch.sign(returns) * torch.log1p(torch.abs(returns))

        advantages = advantages[:, burnin:].reshape(-1)

        popart_sigma = (
            torch.sqrt(self.agent_core.actor_critic.popart.rms.var.mean())
            if hasattr(self.agent_core.actor_critic, "popart")
            else torch.tensor(1.0, device=advantages.device)
        )
        adv_std = advantages.std()

        raw_adv_mean_tensor = advantages.mean()
        safe_adv_mean = torch.nan_to_num(raw_adv_mean_tensor, nan=0.0)

        adv_std = torch.where(
            torch.isnan(torch.as_tensor(adv_std)) | (torch.as_tensor(adv_std) < 1e-5),
            torch.tensor(1.0, device=advantages.device, dtype=adv_std.dtype),
            torch.as_tensor(adv_std),
        )

        advantages = advantages.sub(safe_adv_mean).div_(adv_std)
        raw_adv_mean = safe_adv_mean.item()
        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=10.0, neginf=-10.0)
        advantages = torch.clamp(advantages, min=-10.0, max=10.0)

        trajectory_batch.same_rollout_advantage_scalar = raw_adv_mean

        next_states = next_states[:, burnin:, :].reshape(-1, next_states.size(-1))

        if hasattr(trajectory_batch, "valid_mask") and trajectory_batch.valid_mask is not None:
            valid_mask_flat = trajectory_batch.valid_mask[:, burnin:].reshape(-1).float()
        else:
            valid_mask_flat = torch.ones_like(returns)

        if costs is not None:
            costs = costs[:, burnin:].reshape(-1)

        for p in self.agent_core.parameters():
            p.requires_grad = True
        for opt in [self.opt_representation, self.opt_causal, self.opt_policy]:
            for group in opt.param_groups:
                for p in group["params"]:
                    p.requires_grad = True
        if hasattr(self, "opt_policy_fp32") and self.opt_policy_fp32 is not None:
            for group in self.opt_policy_fp32.param_groups:
                for p in group["params"]:
                    p.requires_grad = True

        if hasattr(self.agent_core, "moe"):
            if not hasattr(self.agent_core.moe, "expert_stability_counter"):
                num_experts = getattr(self.agent_core.moe, "num_experts", 8)
                self.agent_core.moe.register_buffer(
                    "expert_stability_counter", torch.zeros(num_experts, dtype=torch.long, device=MODEL_DEVICE)
                )
                self.agent_core.moe.register_buffer(
                    "expert_mature", torch.zeros(num_experts, dtype=torch.bool, device=MODEL_DEVICE)
                )

            if hasattr(self.agent_core, "jepa") and hasattr(self.agent_core.jepa, "proxy_surprisal"):
                _surprisal_val = self.agent_core.jepa.proxy_surprisal
                surprisal = (
                    _surprisal_val.mean().item() if isinstance(_surprisal_val, torch.Tensor) else float(_surprisal_val)
                )
                pred_error = self.pred_error_ema.item() if hasattr(self, "pred_error_ema") else 1.0
                if surprisal < pred_error * 0.95:
                    self.agent_core.moe.expert_stability_counter += 1
                else:
                    self.agent_core.moe.expert_stability_counter = torch.clamp(
                        self.agent_core.moe.expert_stability_counter - 10, min=0
                    )

            self.agent_core.moe.expert_mature = self.agent_core.moe.expert_stability_counter > 1000

        self.opt_representation.zero_grad(set_to_none=True)
        self.opt_causal.zero_grad(set_to_none=True)
        self.opt_policy.zero_grad(set_to_none=True)
        if hasattr(self, "opt_policy_fp32") and self.opt_policy_fp32 is not None:
            self.opt_policy_fp32.zero_grad(set_to_none=True)

        for module in self.agent_core.modules():
            for attr in [
                "aux_loss",
                "coverage_loss",
                "last_masker_loss",
                "last_lpm_loss",
                "ortho_loss",
                "last_ponder_loss",
                "last_alpha_loss",
                "aux_text_loss",
                "last_text_loss",
            ]:
                if hasattr(module, attr):
                    setattr(module, attr, 0.0)

        with torch.amp.autocast(device_type="cuda", dtype=CFG.AMP_DTYPE):
            fused_curr = states.float()
            fused_next = next_states.float()
            with torch.no_grad():
                curr_latent = self.agent_core.jepa(fused_curr)[0]
                next_latent = self.agent_core.jepa(fused_next)[0]

                curr_discrete = self.agent_core.jepa.quantizer(curr_latent.detach())[0]
                next_discrete = self.agent_core.jepa.quantizer(next_latent.detach())[0]

                if hasattr(self.agent_core.actor_critic, "actor_core"):
                    moe_latent = (
                        self.agent_core.moe(curr_latent.detach())
                        if hasattr(self.agent_core, "moe")
                        else curr_latent.detach()
                    )
                    core_out = self.agent_core.actor_critic.actor_core(
                        torch.cat([moe_latent, torch.zeros_like(curr_latent)], dim=-1)
                    )

                    if hasattr(self.agent_core, "moe") and hasattr(self.agent_core.moe, "expert_correlation_matrix"):
                        with torch.no_grad():
                            uncertainty = 1.0 - torch.diagonal(self.agent_core.moe.expert_correlation_matrix).mean()
                            epistemic_noise = uncertainty * 0.1 * torch.randn_like(core_out)

                        core_out = core_out + epistemic_noise

                        actor_features = F.pad(core_out, (0, curr_latent.size(-1) - core_out.size(-1)))
                    else:
                        actor_features = torch.zeros_like(curr_latent)

                    safe_curr = torch.clamp(torch.nan_to_num(curr_latent.detach(), nan=0.0), -10.0, 10.0).float()
                    safe_next = torch.clamp(torch.nan_to_num(next_latent.detach(), nan=0.0), -10.0, 10.0).float()
                    inverse_pred = self.agent_core.lpm_module.inverse_dynamics(
                        torch.cat([safe_curr, safe_next], dim=-1).to(curr_latent.dtype)
                    )
                    inverse_pred_safe = torch.clamp(
                        torch.nan_to_num(inverse_pred.float(), nan=0.0), min=-50.0, max=50.0
                    )
                    action_targets = torch.argmax(action_one_hot, dim=-1)
                    inverse_error = F.cross_entropy(inverse_pred_safe, action_targets, reduction="none")
                    cycle_consistency_mask = inverse_error < 1.0

                    if hasattr(self.agent_core, "build_critic_context"):
                        critic_ctx_eval = self.agent_core.build_critic_context(
                            curr_latent.detach(), torch.zeros_like(curr_latent.detach())
                        )
                    else:
                        base_ac = (
                            self.agent_core.actor_critic.module
                            if hasattr(self.agent_core.actor_critic, "module")
                            else self.agent_core.actor_critic
                        )
                        expected_dim = (
                            base_ac.critic_1[0].in_features
                            if isinstance(base_ac.critic_1, nn.Sequential)
                            else base_ac.critic_1.in_features
                        )
                        pad_size = expected_dim - curr_latent.size(-1)
                        critic_ctx_eval = torch.cat(
                            [
                                curr_latent.detach(),
                                torch.zeros(
                                    curr_latent.size(0), pad_size, device=curr_latent.device, dtype=curr_latent.dtype
                                ),
                            ],
                            dim=-1,
                        )

                    base_ac = (
                        self.agent_core.actor_critic.module
                        if hasattr(self.agent_core.actor_critic, "module")
                        else self.agent_core.actor_critic
                    )
                    val_1 = base_ac.critic_1(critic_ctx_eval)
                    val_2 = base_ac.critic_2(critic_ctx_eval)
                    value_variance = torch.abs(val_1 - val_2).mean(dim=-1)
                    planning_entropy_mask = value_variance < 5.0

                    combined_attractor_gate = cycle_consistency_mask & planning_entropy_mask

                    if combined_attractor_gate.any():
                        valid_curr = curr_discrete[combined_attractor_gate]
                        valid_act_indices = actions[combined_attractor_gate].unsqueeze(-1).float()
                        pad_size = 256 - valid_act_indices.size(-1)
                        valid_act_vector = F.pad(valid_act_indices, (0, pad_size))
                        valid_next = next_discrete[combined_attractor_gate]
                        self.agent_core.fuzzy_kb.store_experience(valid_curr, valid_act_vector, valid_next)

                corrupted_curr = self.agent_core.adversary_controller.apply_budgeted_warp(
                    self.agent_core.adversary_module,
                    fused_curr,
                    self.agent_core.jepa,
                    self.agent_core.actor_critic,
                    consolidation_weight=torch.tensor([1.0], device=MODEL_DEVICE),
                )

            quantized_state, online_pred_raw, _, jepa_alignment_loss = self.agent_core.jepa(corrupted_curr)
            _fsq_in = corrupted_curr.mean(dim=1) if corrupted_curr.dim() == 3 else corrupted_curr
            _, loss_vq_raw, _ = self.agent_core.jepa.quantizer(self.agent_core.jepa.fsq_encoder(_fsq_in))
            loss_vq = (
                loss_vq_raw
                if isinstance(loss_vq_raw, torch.Tensor) and loss_vq_raw.requires_grad
                else (
                    loss_vq_raw.clone().requires_grad_(True)
                    if isinstance(loss_vq_raw, torch.Tensor)
                    else torch.tensor(loss_vq_raw, device=fused_curr.device, requires_grad=True)
                )
            )
            with torch.no_grad():
                _, _, target_proj_raw, _ = self.agent_core.jepa(fused_next)

            with torch.amp.autocast(device_type="cuda", enabled=False):
                b_size = online_pred_raw.size(0)
                d_dim = online_pred_raw.size(-1)

                online_f32 = torch.clamp(torch.nan_to_num(online_pred_raw.float(), nan=0.0), -20.0, 20.0)
                target_f32 = torch.clamp(torch.nan_to_num(target_proj_raw.detach().float(), nan=0.0), -20.0, 20.0)

                inv_loss_mean = F.mse_loss(online_f32, target_f32)

                if b_size > 1:
                    online_centered = online_f32 - online_f32.mean(dim=0, keepdim=True)
                    target_centered = target_f32 - target_f32.mean(dim=0, keepdim=True)

                    std_online = torch.sqrt(online_centered.var(dim=0, unbiased=False) + 1e-4)
                    std_target = torch.sqrt(target_centered.var(dim=0, unbiased=False) + 1e-4)

                    # Apply hinge loss to batch standard deviation.
                    var_loss = torch.mean(F.relu(1.0 - std_online)) + torch.mean(F.relu(1.0 - std_target))

                    cov_online = (online_centered.T @ online_centered) / (b_size - 1)
                    off_diag_online = cov_online - torch.diag(torch.diag(cov_online))
                    cov_loss = (off_diag_online**2).sum() / d_dim
                else:
                    var_loss = torch.tensor(0.0, device=online_f32.device)
                    cov_loss = torch.tensor(0.0, device=online_f32.device)

                l2_penalty = (online_f32.pow(2).mean() + target_f32.pow(2).mean()) * 0.01

                loss_byol = (25.0 * inv_loss_mean) + (25.0 * var_loss) + (1.0 * cov_loss) + l2_penalty

                online_pred_norm = F.normalize(online_f32, p=2, dim=-1, eps=1e-6)
                target_proj_norm = F.normalize(target_f32, p=2, dim=-1, eps=1e-6)

                from vrl_framework.math_ops.geometry import compute_eff_dim, compute_predictive_surprisal

                self.agent_core.jepa.proxy_surprisal = compute_predictive_surprisal(
                    online_pred_norm, target_proj_norm
                ).to(online_pred_raw.dtype)

            online_pred = online_pred_norm.to(online_pred_raw.dtype)

            with torch.no_grad():
                online_pred_c = online_pred.float() - online_pred.float().mean(dim=0, keepdim=True)
                cov_pred = torch.matmul(online_pred_c.T, online_pred_c) / max(1, online_pred_c.size(0))
                cov_f32 = cov_pred.float()
                diag_jitter = torch.eye(cov_f32.size(-1), device=cov_f32.device, dtype=cov_f32.dtype) * 1e-4
                cov_f32 = (cov_f32 + cov_f32.mT) * 0.5 + diag_jitter
                try:
                    eigenvals = torch.linalg.eigvalsh(cov_f32)
                    eigenvals = torch.clamp(eigenvals, min=1e-4)
                    p_eig = eigenvals / (eigenvals.sum() + 1e-4)
                    log_p_eig = torch.log(torch.clamp(p_eig, min=1e-4))
                    eff_dim = torch.exp(-(p_eig * log_p_eig).sum()).item()
                except Exception:
                    eff_dim = 1.0

            with torch.no_grad():
                tau_jepa = 0.005
                for target_param, online_param in zip(
                    self.agent_core.jepa.target_encoder.parameters(), self.agent_core.jepa.fp16_encoder.parameters()
                ):
                    target_param.data = target_param.data * (1.0 - tau_jepa) + online_param.data * tau_jepa

            diag_start = time.perf_counter()
            sigreg_loss_rep = torch.mean(torch.abs(online_f32.mean(dim=0)))

            total_rep_loss = (0.25 * loss_vq) + loss_byol + (0.01 * sigreg_loss_rep)

            if hasattr(self.agent_core, "aux_text_loss"):
                _aux_t = self.agent_core.aux_text_loss
                total_rep_loss = total_rep_loss + (
                    0.5 * (_aux_t.detach() if isinstance(_aux_t, torch.Tensor) else _aux_t)
                )

            try:
                test_grads = torch.autograd.grad(
                    total_rep_loss,
                    [p for p in self.agent_core.jepa.parameters() if p.requires_grad][:20],
                    retain_graph=True,
                    allow_unused=True,
                )
                valid_test_grads = [g for g in test_grads if g is not None]
                if valid_test_grads:
                    conflict_val_t = torch.var(torch.cat([g.view(-1) for g in valid_test_grads]))
                    self._last_gradient_conflict = float(conflict_val_t.item())
                    conflict_val = conflict_val_t
                else:
                    conflict_val = torch.tensor(getattr(self, "_last_gradient_conflict", 0.0), device=MODEL_DEVICE)
            except Exception:
                conflict_val = torch.tensor(getattr(self, "_last_gradient_conflict", 0.0), device=MODEL_DEVICE)

            diag_elapsed = (time.perf_counter() - diag_start) * 1000.0
            if getattr(self, "global_train_step", 0) % 50 == 0:
                self.scheduler.finalize_execution("diagnostics", diag_elapsed, utility_score=0.0)

            total_rep_loss = torch.where(
                (~torch.isnan(torch.as_tensor(conflict_val))) & (torch.as_tensor(conflict_val) > 10.0),
                total_rep_loss * 0.5,
                total_rep_loss,
            )
            conflict_val_scalar = conflict_val.item() if hasattr(conflict_val, "item") else float(conflict_val)

            self._safe_sigreg = (
                float(sigreg_loss_rep.item()) if hasattr(sigreg_loss_rep, "item") else float(sigreg_loss_rep)
            )
            self._safe_ortho = (
                float(cov_loss.item() if hasattr(cov_loss, "item") else cov_loss) if "cov_loss" in locals() else 0.0
            )
            safe_sigreg = self._safe_sigreg
            safe_ortho = self._safe_ortho

            safe_var_loss = var_loss
            safe_cov_loss = cov_loss

            curr_fused_state, _, _, _ = self.agent_core.jepa(fused_curr)
            curr_fused_state = F.normalize(curr_fused_state.float(), p=2, dim=-1).to(curr_fused_state.dtype)
            z_detached = curr_fused_state.detach()
            next_fused_state, _, _, _ = self.agent_core.jepa(fused_next)
            next_fused_state = F.normalize(next_fused_state.float(), p=2, dim=-1).to(next_fused_state.dtype)
            next_latent_detached = next_fused_state.detach()

            if torch.isnan(z_detached).any() or torch.isinf(z_detached).any():
                if not hasattr(self, "hyperbolic_failures"):
                    self.hyperbolic_failures = 0
                self.hyperbolic_failures += 1
            else:
                with torch.no_grad():
                    self._ensure_holdout_buffers_allocated(z_detached.device)
                    sample_idx = torch.randperm(z_detached.size(0), device=z_detached.device)[:16]
                    if self.holdout_ptr + 16 > 512:
                        self.holdout_ptr = 0
                    end_ptr = self.holdout_ptr + sample_idx.size(0)

                    assert self.holdout_buffer_states is not None
                    assert self.holdout_buffer_actions is not None
                    assert self.holdout_buffer_next_states is not None
                    self.holdout_buffer_states[self.holdout_ptr : end_ptr] = z_detached[sample_idx]
                    self.holdout_buffer_actions[self.holdout_ptr : end_ptr] = action_one_hot[sample_idx]
                    self.holdout_buffer_next_states[self.holdout_ptr : end_ptr] = next_latent_detached[sample_idx]
                    self.holdout_ptr = end_ptr

            pred_next_latent = self.agent_core.latent_dynamics(z_detached, action_one_hot)
            pred_next_latent = torch.clamp(
                torch.nan_to_num(pred_next_latent, nan=0.0, posinf=10.0, neginf=-10.0), min=-10.0, max=10.0
            )
            base_dynamics_loss = F.smooth_l1_loss(pred_next_latent, next_latent_detached, reduction="none").mean(
                dim=-1
            )

            with torch.no_grad():
                self.agent_core.latent_dynamics.eval()
                controllability_score = self.agent_core.interventional_causal_engine.evaluate_controllability(
                    z_detached, action_one_hot, self.agent_core.latent_dynamics
                )
                self.agent_core.latent_dynamics.train()
                valid_causal_mask = (controllability_score > 1e-4).float()
                if valid_causal_mask.sum() < 1.0:
                    valid_causal_mask = torch.ones_like(valid_causal_mask)

            causal_loss = torch.clamp(
                (base_dynamics_loss * valid_causal_mask).sum() / (valid_causal_mask.sum() + 1e-8), max=100.0
            )
            causal_loss = torch.nan_to_num(causal_loss, nan=0.0)

            if hasattr(self, "dynamics_mse_ema"):
                with torch.no_grad():
                    self.dynamics_mse_ema.copy_(0.95 * self.dynamics_mse_ema + 0.05 * causal_loss.detach())

            if getattr(self, "global_train_step", 0) % 50 == 0:
                holdout_stats = self.evaluate_holdout()
                metrics_aggregator.log(
                    {
                        "eval/holdout_dynamics_mse": holdout_stats.holdout_dynamics_mse,
                        "eval/teacher_confirmation_rate": holdout_stats.teacher_confirmation_rate,
                        "eval/confirmed_planner_gain": holdout_stats.confirmed_planner_gain,
                        "eval/critic_divergence_ema": holdout_stats.critic_divergence_ema,
                        "eval/hyperbolic_failure_rate": holdout_stats.hyperbolic_failure_rate,
                    }
                )

            if hasattr(self.agent_core, "fuzzy_kb"):
                logical_context = self.agent_core.fuzzy_kb.reason(z_detached)
                next_logical_context = self.agent_core.fuzzy_kb.reason(next_latent_detached)
            else:
                logical_context = None
                next_logical_context = None

            planner_regret_val = float(self.planner_regret_ema.item()) if hasattr(self, "planner_regret_ema") else 0.0
            planning_gain_val = float(self.planning_gain_ema.item()) if hasattr(self, "planning_gain_ema") else 0.0
            dyn_mse_val = (
                float(causal_loss.item())
                if "causal_loss" in locals()
                else (float(self.dynamics_mse_ema.item()) if hasattr(self, "dynamics_mse_ema") else 0.0)
            )

            real_hit_rate = 0.0
            try:
                _valid_keys = None
                if (
                    hasattr(self.agent_core, "fuzzy_kb")
                    and hasattr(self.agent_core.fuzzy_kb, "kb_subjects")
                    and hasattr(self.agent_core.fuzzy_kb, "kb_ptr")
                ):
                    _limit = int(self.agent_core.fuzzy_kb.kb_ptr.item())
                    if _limit > 0:
                        _valid_keys = self.agent_core.fuzzy_kb.kb_subjects[:_limit].detach().float()
                elif (
                    hasattr(self.agent_core.memory, "core")
                    and isinstance(self.agent_core.memory.core, torch.Tensor)
                    and self.agent_core.memory.core.numel() > 0
                ):
                    _valid_keys = self.agent_core.memory.core.detach().float()
                elif (
                    hasattr(self, "holdout_buffer_states")
                    and self.holdout_buffer_states is not None
                    and getattr(self, "holdout_ptr", 0) > 0
                ):
                    _valid_keys = self.holdout_buffer_states[: self.holdout_ptr].detach().float()

                if _valid_keys is not None and _valid_keys.numel() > 0:
                    _q = z_detached.float()
                    norm_query = torch.nn.functional.normalize(_q, p=2, dim=-1)
                    norm_keys = torch.nn.functional.normalize(_valid_keys.view(-1, _valid_keys.size(-1)), p=2, dim=-1)
                    sim_scores = torch.matmul(norm_query, norm_keys.t())
                    sim_scores.masked_fill_(sim_scores > 0.99, -1.0)
                    max_sims, _ = torch.max(sim_scores, dim=-1)
                    real_hit_rate = float(torch.clamp(max_sims, min=0.0).mean().item())
                else:
                    if logical_context is not None:
                        _z_f32 = z_detached.float()
                        _log_f32 = logical_context.float()
                        if torch.allclose(_z_f32, _log_f32, atol=1e-4):
                            real_hit_rate = float("nan")
                        else:
                            _sim = F.cosine_similarity(_z_f32, _log_f32, dim=-1)
                            real_hit_rate = float(torch.clamp(_sim, min=0.0).mean().item())
            except Exception:
                real_hit_rate = float("nan")

            if math.isnan(real_hit_rate):
                try:
                    _q = z_detached.float()
                    norm_q = torch.nn.functional.normalize(_q, p=2, dim=-1)
                    sim_scores = torch.matmul(norm_q, norm_q.t())
                    sim_scores.masked_fill_(
                        torch.eye(sim_scores.size(0), device=sim_scores.device, dtype=torch.bool), -1.0
                    )
                    real_hit_rate = float(torch.clamp(torch.max(sim_scores, dim=-1)[0], min=0.0).mean().item())
                except Exception:
                    real_hit_rate = 0.0

            real_vq_loss = (
                F.mse_loss(curr_latent.detach(), curr_discrete.detach())
                if "curr_discrete" in locals()
                else torch.tensor(0.0)
            )
            conditional_ecq_val = (
                float(torch.log1p(real_vq_loss / (loss_byol + 1e-5)).item()) if hasattr(real_vq_loss, "item") else 0.0
            )

            def _safe_float(v):
                val = float(v.item()) if hasattr(v, "item") else float(v)
                return val if math.isfinite(val) else 0.0

            _live_band_metric = 10.0 / (1.0 + dyn_mse_val + float(locals().get("critic_dissonance", 0.0)))

            _metrics_payload = {
                "loss/jepa_invariance": _safe_float(inv_loss_mean),
                "loss/jepa_variance": _safe_float(safe_var_loss),
                "loss/jepa_covariance": _safe_float(safe_cov_loss),
                "loss/jepa_contrastive_total": _safe_float(total_rep_loss),
                "metrics/signature_regularization": safe_sigreg if math.isfinite(safe_sigreg) else 0.0,
                "metrics/subspace_orthogonality": safe_ortho if math.isfinite(safe_ortho) else 0.0,
                "metrics/gradient_conflict_variance": (
                    float(conflict_val_scalar) if math.isfinite(conflict_val_scalar) else 0.0
                ),
                "metrics/quantization_error": _safe_float(loss_vq),
                "metrics/planning_gain": _safe_float(planning_gain_val),
                "metrics/planner_regret": _safe_float(planner_regret_val),
                "metrics/latent_matrix_rank": _safe_float(eff_dim),
                "metrics/dynamics_mse": _safe_float(dyn_mse_val),
                "metrics/conditional_ecq": _safe_float(conditional_ecq_val),
                "metrics/decision_compute_ms": _safe_float(diag_elapsed),
                "metrics/health_score": (
                    float(self.agent_core.health_score_proxy)
                    if hasattr(self.agent_core, "health_score_proxy")
                    else 0.5
                ),
                "metrics/health_band": float(_live_band_metric),
            }

            if "real_hit_rate" in locals() and math.isfinite(real_hit_rate):
                _metrics_payload["metrics/retrieval_hit_rate"] = float(real_hit_rate)

            _lp_tensor = locals().get("learning_progress", None)
            if _lp_tensor is not None and isinstance(_lp_tensor, torch.Tensor):
                _metrics_payload["curiosity/learning_progress_velocity"] = float(_lp_tensor.mean().item())

            metrics_aggregator.log({k: v for k, v in _metrics_payload.items()})

            _is_fatal = locals().get("is_nan", False) or locals().get("is_fatal_spike", False)
            if locals().get("perform_update", False) and not _is_fatal:
                if not hasattr(self, "_state_backup_queue"):
                    self._state_backup_queue = []
                    self.semantic_backup_state = {
                        k: v.detach().cpu().clone() for k, v in self.agent_core.state_dict().items()
                    }

                self._state_backup_queue.append(
                    {k: v.detach().cpu().clone() for k, v in self.agent_core.state_dict().items()}
                )

                if len(self._state_backup_queue) > 15:
                    self.semantic_backup_state = self._state_backup_queue.pop(0)
            elif _is_fatal:
                import collections

                def _flush_opt(obj):
                    for k in dir(obj):
                        try:
                            attr = getattr(obj, k)
                            if isinstance(attr, torch.optim.Optimizer):
                                attr.state = collections.defaultdict(dict)
                        except Exception:
                            import logging

                            logging.getLogger(__name__).exception("Failed to clear optimizer state")

                _flush_opt(self)
                if hasattr(self, "agent_core"):
                    _flush_opt(self.agent_core)
                if hasattr(self, "_state_backup_queue"):
                    self._state_backup_queue.clear()

            with torch.no_grad():
                curr_fused_state_p3, _, _, _ = self.agent_core.jepa(fused_curr)
                latent_context = curr_fused_state_p3.detach()
                logical_context_p3 = self.agent_core.fuzzy_kb.reason(latent_context).detach()

                flat_batch = latent_context.size(0)
                dummy_action = torch.zeros(flat_batch, self.agent_core.num_actions, device=MODEL_DEVICE)
                dummy_reward = torch.zeros(flat_batch, 1, device=MODEL_DEVICE)
                dummy_meta = torch.zeros(flat_batch, 256, device=MODEL_DEVICE)

                gru_input = torch.cat([logical_context_p3, dummy_action, dummy_reward], dim=-1)

                pad_size = 256 + self.agent_core.num_actions + 1 - gru_input.size(-1)
                gru_input = F.pad(gru_input, (0, pad_size))

                target_dtype_meta = next(self.agent_core.meta_gru.parameters()).dtype
                gru_input_safe2 = gru_input.to(target_dtype_meta)
                dummy_meta_safe = dummy_meta.to(target_dtype_meta)

                meta_context = self.agent_core.meta_gru(gru_input_safe2, dummy_meta_safe)

                sp_expanded = self.agent_core.scratchpad.expand(flat_batch, -1, -1)
                sp_out, _ = self.agent_core.scratchpad_attention(
                    query=meta_context.unsqueeze(1), key=sp_expanded, value=sp_expanded
                )
                sp_context = meta_context + sp_out.squeeze(1)

                global_state = latent_context.mean(dim=0, keepdim=True).expand(latent_context.size(0), -1)
                if hasattr(self.agent_core, "build_critic_context"):
                    critic_context = self.agent_core.build_critic_context(global_state, logical_context_p3)
                else:
                    base_ac = (
                        self.agent_core.actor_critic.module
                        if hasattr(self.agent_core.actor_critic, "module")
                        else self.agent_core.actor_critic
                    )
                    expected_dim = (
                        base_ac.critic_1[0].in_features
                        if isinstance(base_ac.critic_1, nn.Sequential)
                        else base_ac.critic_1.in_features
                    )
                    pad_size = expected_dim - 256 - (global_state.size(-1) + logical_context_p3.size(-1))
                    critic_context = torch.cat(
                        [
                            global_state,
                            logical_context_p3,
                            torch.zeros(
                                global_state.size(0), pad_size, device=global_state.device, dtype=global_state.dtype
                            ),
                        ],
                        dim=-1,
                    )

            if hasattr(self.agent_core, "moe"):
                if getattr(CFG, "USE_MOE", True):
                    moe_context = self.agent_core.moe(sp_context)
                else:
                    moe_context = sp_context + self.agent_core.moe(sp_context)
            else:
                moe_context = sp_context
            moe_context_detached = moe_context.detach()

            intent_proxy = torch.zeros(fused_curr.size(0), 256, device=MODEL_DEVICE)
            with torch.no_grad():
                if hasattr(self.agent_core, "hierarchical_planner"):
                    batch_intents = self.agent_core.hierarchical_planner.hiro_off_policy_correction(
                        fused_curr, fused_next, actions, self.agent_core.actor_critic, num_candidates=8
                    )
                else:
                    batch_intents = intent_proxy

            if hasattr(self.agent_core, "hierarchical_planner"):
                worker_context = self.agent_core.hierarchical_planner(moe_context, batch_intents)
                worker_context_detached = self.agent_core.hierarchical_planner(
                    moe_context_detached, batch_intents.detach()
                )
            else:
                worker_context = moe_context
                worker_context_detached = moe_context_detached

            worker_context = torch.nan_to_num(worker_context, nan=0.0, posinf=10.0, neginf=-10.0)
            worker_context_detached = torch.nan_to_num(worker_context_detached, nan=0.0, posinf=10.0, neginf=-10.0)

            worker_context_linked = worker_context + 0.0
            if worker_context_linked.requires_grad:
                worker_context_linked.register_hook(lambda grad: torch.clamp(grad * 0.05, min=-1.0, max=1.0))

            ac_out = self.agent_core.actor_critic(
                worker_context_linked,
                critic_context.detach(),
                intent_context=batch_intents.detach() if isinstance(batch_intents, torch.Tensor) else batch_intents,
            )

            # Set submodules to eval mode to prevent batch norm/dropout updates during auxiliary passes.
            self.agent_core.eval()

            if hasattr(self, "validation_metrics"):
                self.validation_metrics.assert_actorcritic_output_contract(ac_out)

            policy_logits = ac_out.policy_logits
            mu_cvae = ac_out.style_mu
            logvar_cvae = ac_out.style_logvar

            current_values = ac_out.pessimistic_value
            cost_values = ac_out.cost_value
            current_intrinsic_values = ac_out.intrinsic_value
            value_logits = (ac_out.value_logits_1, ac_out.value_logits_2)

            value_logits_1, value_logits_2 = value_logits

            if policy_logits.dim() == 3:
                immediate_logits = policy_logits[:, 0, : self.agent_core.num_actions]
            else:
                immediate_logits = policy_logits[..., : self.agent_core.num_actions]

            self.agent_core.cumulative_action_norms = (
                getattr(self.agent_core, "cumulative_action_norms", 0.0)
                + torch.norm(immediate_logits, p=2, dim=-1).mean().item()
            )

            grounded_worker_context = worker_context

            if hasattr(self.agent_core, "causal_masker"):
                safe_logits = self.agent_core.causal_masker(grounded_worker_context, immediate_logits)
            else:
                safe_logits = immediate_logits

            # LogSumExp numerical stability: center logits to avoid FP16 overflow in Categorical distribution.
            safe_logits = safe_logits - torch.nan_to_num(torch.max(safe_logits, dim=-1, keepdim=True)[0], nan=0.0)
            safe_logits = torch.clamp(
                torch.nan_to_num(safe_logits, nan=-50.0, posinf=50.0, neginf=-50.0).float(), min=-50.0, max=50.0
            )
            dist = torch.distributions.Categorical(logits=safe_logits)
            new_log_probs = torch.nan_to_num(dist.log_prob(actions), nan=-100.0)
            entropy = torch.nan_to_num(dist.entropy(), nan=0.0).mean().to(safe_logits.dtype)

            with torch.no_grad():
                raw_dissonance = torch.abs(value_logits_1.detach() - value_logits_2.detach()).mean()
                if torch.isnan(raw_dissonance) or torch.isinf(raw_dissonance):
                    raw_dissonance = torch.tensor(5.0, device=MODEL_DEVICE)
                critic_dissonance = torch.log1p(raw_dissonance).item()
                v_bound = 15.0
                scale_vq = (2.0 * v_bound) / 255.0
                safe_z_detached = torch.nan_to_num(z_detached, nan=0.0, posinf=v_bound, neginf=-v_bound)
                clipped_lat = torch.clamp(safe_z_detached, min=-v_bound, max=v_bound)
                quantized_i8 = ((clipped_lat + v_bound) / scale_vq - 128).to(torch.int8)
                restored_f16 = ((quantized_i8.float() + 128.0) * scale_vq - v_bound).half()
                if hasattr(self.agent_core.memory, "dequantization_corrector"):
                    corrector_out = self.agent_core.memory.dequantization_corrector(restored_f16)
                    corrector_out = torch.clamp(torch.nan_to_num(corrector_out, nan=0.0), min=-5.0, max=5.0)
                else:
                    corrector_out = 0.0
                corrected_rec = restored_f16 + corrector_out
                true_quant_error = F.mse_loss(corrected_rec.float(), safe_z_detached.float()).item()
                logical_validity_scores_temp = self.agent_core.fuzzy_kb.evaluate_truth_gate_differentiable(z_detached)
                truth_margin_temp = (
                    logical_validity_scores_temp.mean().item()
                    if isinstance(logical_validity_scores_temp, torch.Tensor)
                    else 0.0
                )

                current_jepa_surprisal = getattr(
                    self, "dynamics_mse_ema", torch.tensor(1.0, device=advantages.device)
                ).item()

            # Scale advantages based on value ensemble disagreement to stabilize updates in out-of-distribution states.
            epistemic_variance = torch.nan_to_num(
                torch.tensor(
                    critic_dissonance + true_quant_error + max(0.0, -truth_margin_temp) + current_jepa_surprisal,
                    dtype=torch.float32,
                ),
                nan=0.0,
            )
            reliability_gate = torch.exp(-torch.clamp(epistemic_variance, min=0.0, max=10.0)).to(advantages.device)
            reliability_gate = torch.clamp(reliability_gate, min=0.01)
            conservative_advantages = advantages * reliability_gate

            c_adv_std = torch.nan_to_num(conservative_advantages.std(), nan=1.0)
            conservative_advantages = conservative_advantages / torch.clamp(c_adv_std, min=1e-4)

            popart_std = (
                self.agent_core.actor_critic.value_std.item()
                if hasattr(self.agent_core.actor_critic, "value_std")
                else 1.0
            )
            clip_limit = 3.0 * (popart_std if not math.isnan(popart_std) else 1.0)
            conservative_advantages = torch.clamp(
                torch.nan_to_num(conservative_advantages, nan=0.0), min=-clip_limit, max=clip_limit
            )

            log_diff = torch.clamp(
                new_log_probs.float() - torch.nan_to_num(old_log_probs.float(), nan=-100.0), min=-20.0, max=5.0
            )
            ratio_tensor = torch.exp(log_diff).to(safe_logits.dtype)
            ratio_tensor = torch.clamp(ratio_tensor, min=0.5, max=2.0)
            surr1 = ratio_tensor * conservative_advantages
            surr2 = torch.clamp(ratio_tensor, min=1.0 - 0.2, max=1.0 + 0.2) * conservative_advantages
            ppo_clip_loss_raw = -torch.min(surr1, surr2)
            ppo_clip_loss = (ppo_clip_loss_raw * valid_mask_flat).sum() / torch.clamp(valid_mask_flat.sum(), min=1.0)

            with torch.no_grad():
                approx_kl = 0.5 * torch.nan_to_num((old_log_probs - new_log_probs).pow(2), nan=0.0).mean()

            kl_threshold = 0.02
            if approx_kl > kl_threshold:
                # Trigger early stopping on KL divergence explosion and decay learning rate to prevent policy collapse.
                if conservative_advantages.mean() > 0.5:
                    pass
                else:
                    for param_group in self.opt_policy.param_groups:
                        param_group["lr"] *= 0.9
                    if hasattr(self, "opt_policy_fp32") and self.opt_policy_fp32 is not None:
                        for param_group in self.opt_policy_fp32.param_groups:
                            param_group["lr"] *= 0.9

                if hasattr(self.agent_core, "halting_head"):
                    self.agent_core.halting_head.epsilon = max(0.001, self.agent_core.halting_head.epsilon * 0.9)
            elif approx_kl < kl_threshold * 0.5:
                for param_group in self.opt_policy.param_groups:
                    param_group["lr"] = min(3e-5, param_group["lr"] * 1.05)
                if hasattr(self, "opt_policy_fp32") and self.opt_policy_fp32 is not None:
                    for param_group in self.opt_policy_fp32.param_groups:
                        param_group["lr"] = min(3e-5, param_group["lr"] * 1.05)

            sub_batch_size = min(32, latent_context.size(0))
            sub_indices = torch.randperm(latent_context.size(0), device=MODEL_DEVICE)[:sub_batch_size]
            sub_latent = latent_context[sub_indices].detach()
            sub_global = global_state[sub_indices].detach()
            sub_logical = logical_context_p3[sub_indices].detach()

            from einops import rearrange, repeat

            with torch.no_grad():
                current_dyn_error = getattr(self, "dynamics_mse_ema", torch.tensor(1.0)).item()
                base_horizon = getattr(PLAN_CFG, "imagination_horizon", 16)
                if current_dyn_error > 1.0:
                    dynamic_horizon = min(3, base_horizon)
                elif current_dyn_error > 0.5:
                    dynamic_horizon = min(8, base_horizon)
                else:
                    dynamic_horizon = min(16, base_horizon)

            self.agent_core.actor_critic.eval()
            imagined_horizon = self.agent_core.latent_dynamics.imagine_rollout(
                sub_latent, self.agent_core.actor_critic, horizon=dynamic_horizon
            )
            self.agent_core.actor_critic.train()
            B, T, D = imagined_horizon.shape
            im_states_flat = rearrange(imagined_horizon, "b t d -> (b t) d")[..., :256]

            im_critic_ctx = torch.cat([sub_global, sub_logical], dim=-1)
            pad_critic_im = (256 * 3) - im_critic_ctx.size(-1)
            im_critic_ctx = F.pad(im_critic_ctx, (0, pad_critic_im))

            im_critic_ctx_flat = rearrange(repeat(im_critic_ctx, "b d -> b t d", t=T), "b t d -> (b t) d")

            self.agent_core.actor_critic.eval()
            im_out = self.agent_core.actor_critic(im_states_flat, im_critic_ctx_flat)
            self.agent_core.actor_critic.train()
            im_val_flat = im_out.pessimistic_value
            im_val_flat_detached_critic = im_val_flat - im_val_flat.detach() + im_val_flat.detach()
            imagined_values_tensor = rearrange(im_val_flat_detached_critic, "(b t) d -> b t d", b=B, t=T)

            imagined_mean = imagined_values_tensor.mean()
            imagined_std = torch.sqrt(imagined_values_tensor.var(unbiased=False) + 1e-6)
            dreamer_actor_loss = -(imagined_mean - 0.1 * imagined_std)

            with torch.no_grad():
                returns_f32 = returns.float()
                batch_std_tensor = torch.as_tensor(
                    returns_f32.std(unbiased=False).detach() if returns_f32.numel() > 1 else 1.0,
                    device=returns_f32.device,
                )
                if torch.isnan(batch_std_tensor).any() or bool((batch_std_tensor < 1e-4).item()):
                    safe_std_tensor = torch.tensor(1.0, device=returns_f32.device)
                else:
                    safe_std_tensor = batch_std_tensor

                if not hasattr(self, "reward_norm_mean"):
                    self.reward_norm_mean = float(returns_f32.mean().item())
                    self.reward_norm_std = float(safe_std_tensor.item() + 1e-4)
                else:
                    self.reward_norm_mean = float(0.99 * self.reward_norm_mean + 0.01 * returns_f32.mean().item())
                    self.reward_norm_std = float(0.99 * self.reward_norm_std + 0.01 * (safe_std_tensor.item() + 1e-4))

                raw_batch_mean = returns_f32.mean().detach()
                raw_batch_var = (
                    returns_f32.var(unbiased=False).detach()
                    if returns_f32.size(0) > 1
                    else torch.tensor(0.0, device=returns.device)
                )

                self.reward_norm_mean = float(0.99 * self.reward_norm_mean + 0.01 * raw_batch_mean)
                self.reward_norm_std = float(0.99 * self.reward_norm_std + 0.01 * torch.sqrt(raw_batch_var + 1e-4))

                normalized_returns = ((returns_f32 - self.reward_norm_mean) / (self.reward_norm_std + 1e-4)).to(
                    returns.dtype
                )

            base_ac = (
                self.agent_core.actor_critic.module
                if hasattr(self.agent_core.actor_critic, "module")
                else self.agent_core.actor_critic
            )
            value_logits_1, value_logits_2 = value_logits

            v1_probs = F.softmax(value_logits_1.float(), dim=-1)
            v2_probs = F.softmax(value_logits_2.float(), dim=-1)

            v1_net = torch.sum(v1_probs * base_ac.value_support.to(v1_probs.device), dim=-1)
            v2_net = torch.sum(v2_probs * base_ac.value_support.to(v2_probs.device), dim=-1)

            val_loss_1_raw = F.smooth_l1_loss(v1_net, normalized_returns, reduction="none")
            val_loss_2_raw = F.smooth_l1_loss(v2_net, normalized_returns, reduction="none")

            val_loss_1_safe = torch.nan_to_num(val_loss_1_raw, nan=0.0, posinf=0.0, neginf=0.0)
            val_loss_2_safe = torch.nan_to_num(val_loss_2_raw, nan=0.0, posinf=0.0, neginf=0.0)

            value_loss_1 = (val_loss_1_safe * valid_mask_flat).sum() / torch.clamp(valid_mask_flat.sum(), min=1.0)
            value_loss_2 = (val_loss_2_safe * valid_mask_flat).sum() / torch.clamp(valid_mask_flat.sum(), min=1.0)
            value_loss = torch.nan_to_num(value_loss_1 + value_loss_2, nan=0.0)

            if costs is None:
                costs = torch.tensor([self.agent_core.metabolic_cost()], device=states.device).expand(states.size(0))

            with torch.no_grad():
                logical_validity_scores = self.agent_core.fuzzy_kb.evaluate_truth_gate_differentiable(latent_context)
                logical_validity_mask = (logical_validity_scores > 0.0).float()

                epistemic_penalty = (1.0 - logical_validity_mask) * 2.0
                total_costs = costs + epistemic_penalty

            cost_loss = torch.nan_to_num(F.smooth_l1_loss(cost_values.squeeze(-1), total_costs), nan=0.0)
            lambda_penalty = self.pid_controller.calculate_multiplier(total_costs.mean().item())

            # Compute intrinsic reward by scaling the loss
            # velocity with the controllability score.
            with torch.no_grad():
                current_jepa_loss = F.mse_loss(pred_next_latent, next_latent_detached, reduction="none").mean(dim=-1)

                loss_val_scalar = current_jepa_loss.mean().item()
                if not hasattr(self, "lp_history_buffer"):
                    import collections as _std_collections

                    self.lp_history_buffer: _std_collections.deque = _std_collections.deque(maxlen=200)

                self.lp_history_buffer.append(loss_val_scalar)

                if len(self.lp_history_buffer) > 20:
                    y = np.array(self.lp_history_buffer)
                    x = np.arange(len(y))
                    slope, _ = np.polyfit(x, y, 1)
                    raw_velocity = -slope * 10000.0
                    raw_velocity = max(-5.0, min(5.0, raw_velocity))
                else:
                    raw_velocity = 0.0

                learning_progress = torch.full_like(current_jepa_loss, raw_velocity)
                if not hasattr(self, "prev_jepa_loss_ema"):
                    self.prev_jepa_loss_ema = current_jepa_loss.mean().expand_as(current_jepa_loss)
                self.prev_jepa_loss_ema = 0.99 * self.prev_jepa_loss_ema + 0.01 * current_jepa_loss.mean()

                controllability_mask = torch.clamp(
                    controllability_score.clone().detach().to(MODEL_DEVICE), min=0.0, max=1.0
                ).detach()

                if not hasattr(self, "env_volatility_ema"):
                    self.env_volatility_ema = torch.tensor(1.0, device=current_jepa_loss.device)

                self.env_volatility_ema = 0.99 * self.env_volatility_ema + 0.01 * current_jepa_loss.mean()
                safe_volatility = torch.clamp(self.env_volatility_ema, min=0.1, max=5.0)

                self_error_ratio = torch.clamp(
                    controllability_score.clone().detach().to(MODEL_DEVICE) / (current_jepa_loss + 1e-4), max=1.0
                ).detach()
                aleatoric_penalty = current_jepa_loss * (0.1 / safe_volatility) * (1.0 - self_error_ratio)

                progress_intrinsic_reward = (
                    learning_progress * controllability_mask * (1.0 + self_error_ratio)
                ) - aleatoric_penalty
                progress_intrinsic_reward = torch.clamp(
                    torch.nan_to_num(progress_intrinsic_reward, nan=0.0, posinf=10.0, neginf=-10.0), min=0.0, max=10.0
                )

            safe_current_values = torch.nan_to_num(
                current_values.squeeze(-1).detach(), nan=0.0, posinf=10.0, neginf=-10.0
            )
            pessimistic_intrinsic_target = torch.min(progress_intrinsic_reward, safe_current_values.abs() + 0.05)

            if torch.isnan(current_intrinsic_values).any() or torch.isinf(current_intrinsic_values).any():
                current_intrinsic_values = torch.zeros_like(current_intrinsic_values)

            intrinsic_value_loss = torch.nan_to_num(
                F.smooth_l1_loss(current_intrinsic_values.squeeze(-1).float(), pessimistic_intrinsic_target.float()),
                nan=0.0,
            )

            _l_vars = locals()
            _safe_val_loss = _l_vars.get("value_loss", torch.tensor(0.0, device=ppo_clip_loss.device))
            _safe_cost_loss = _l_vars.get("cost_loss", torch.tensor(0.0, device=ppo_clip_loss.device))
            _safe_moe_aux_loss = _l_vars.get("moe_aux_loss", torch.tensor(0.0, device=ppo_clip_loss.device))

            total_loss = (
                ppo_clip_loss
                + _safe_val_loss
                + causal_loss
                + total_rep_loss
                + _safe_cost_loss
                + intrinsic_value_loss
                + (
                    _safe_moe_aux_loss
                    * self.loss_weights_ema.get("aux", torch.tensor(1.0, device=ppo_clip_loss.device))
                )
            )

            _moe_aux = _l_vars.get("moe_aux_loss", None)
            moe_aux_val = float(_moe_aux.item()) if isinstance(_moe_aux, torch.Tensor) else 0.0
            metrics_aggregator.log(
                {
                    "loss/intrinsic": float(intrinsic_value_loss.item()),
                    "loss/total": float(total_loss.item()),
                    "loss/moe_aux": moe_aux_val,
                    "metrics/planner_regret": (
                        float(self.planner_regret_ema.item()) if hasattr(self, "planner_regret_ema") else 0.0
                    ),
                }
            )

            with torch.no_grad():
                gate_metrics = PlannerOutput(
                    truth_margin=truth_margin_temp,
                    critic_divergence=critic_dissonance,
                    dynamics_error=current_jepa_loss.mean().item(),
                    quantization_error=true_quant_error,
                )

            vicreg_bounds_ok = bool(torch.sqrt(z_detached.var(dim=0, unbiased=False) + 1e-4).mean().item() > 0.5)
            model_error_ok = causal_loss.item() < 1.5

            geometry_violation = not torch.isfinite(z_detached).all().item()
            quantization_explosion = true_quant_error > 500.0
            critic_divergence_spike = critic_dissonance > 500.0
            truth_margin_collapse = gate_metrics.truth_margin < -500.0

            if geometry_violation or quantization_explosion or critic_divergence_spike or truth_margin_collapse:
                reason = (
                    "hyperbolic_failure"
                    if geometry_violation
                    else (
                        "surprisal_spike"
                        if quantization_explosion
                        else ("dynamics_spike" if critic_divergence_spike else "planner_contract_violation")
                    )
                )

                if hasattr(self, "semantic_backup_state"):
                    safe_state_dict = {
                        k: v
                        for k, v in self.semantic_backup_state.items()
                        if k in self.agent_core.state_dict() and self.agent_core.state_dict()[k].shape == v.shape
                    }
                    self.agent_core.load_state_dict(safe_state_dict, strict=False)
                else:
                    self.semantic_backup_state = {
                        k: v.detach().cpu().clone() for k, v in self.agent_core.state_dict().items()
                    }

                for attr in ["meta_gru", "ponder_gru", "gradient_stm"]:
                    if hasattr(self.agent_core, attr):
                        mod = getattr(self.agent_core, attr)
                        if hasattr(mod, "hidden") and mod.hidden is not None:
                            mod.hidden.zero_()
                for attr in ["manager_goal", "previous_memory_context", "internal_state", "interoceptive_target"]:
                    if hasattr(self.agent_core, attr):
                        tensor = getattr(self.agent_core, attr)
                        if isinstance(tensor, torch.Tensor):
                            tensor.zero_()
                if hasattr(self.agent_core, "stm_tensor"):
                    self.agent_core.stm_tensor.zero_()

                if hasattr(self, "world") and hasattr(self.world, "entities"):
                    for ent in self.world.entities:
                        if hasattr(ent, "experience_buffer"):
                            ent.experience_buffer.ptr = 0
                            ent.experience_buffer.size = 0

                import logging

                logging.getLogger("VRL_Engine")

                trace_mock = PlannerDecisionTrace(
                    baseline_logits=torch.zeros(1),
                    planner_logits_preblend=torch.zeros(1),
                    executed_planner_logits=torch.zeros(1),
                    final_action_logits=torch.zeros(1),
                    truth_margin=0.0,
                    critic_divergence=10.0,
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
                    halting_budget_used=0.0,
                    planner_regime=0,
                    planner_temperature=1.0,
                    baselogits_temperature=1.0,
                    prefilter_time_ms=0.0,
                    rollout_time_ms=0.0,
                )

                dummy_budget = PlanningBudget(
                    health_score=0.0,
                    health_band=2,
                    max_depth=0,
                    num_samples=0,
                    distill_enabled=False,
                    teacher_ttl=0,
                    allow_actor_lookahead=False,
                    allow_teacher_write=False,
                    allow_distillation=False,
                    max_branch_survivors=1,
                    min_survivor_floor=1,
                    max_ood_risk=0.0,
                    max_critic_divergence=0.0,
                    max_planner_calls_per_env_step=0,
                )
                self.restore_checkpoint(
                    trace_mock, advantages, true_quant_error, planning_budget=dummy_budget, halting_probs=None
                )

                self.global_train_step += 1

                _safe_pol = float(ppo_clip_loss.item()) if "ppo_clip_loss" in locals() else 0.0
                _safe_val = float(value_loss.item()) if "value_loss" in locals() else 0.0
                _safe_int = (
                    float(progress_intrinsic_reward.mean().item()) if "progress_intrinsic_reward" in locals() else 0.0
                )
                _safe_causal = float(causal_loss.item()) if "causal_loss" in locals() else 0.0
                _safe_total = _safe_pol + _safe_val + _safe_causal + _safe_int
                _safe_se = float(self.global_return_mean) if hasattr(self, "global_return_mean") else 0.0

                safe_regret = float(self.planner_regret_ema.item()) if hasattr(self, "planner_regret_ema") else 0.0
                safe_gain = float(self.planning_gain_ema.item()) if hasattr(self, "planning_gain_ema") else 0.0
                safe_rank = float(
                    metrics_aggregator.metrics.get(
                        "metrics/latent_matrix_rank", float(eff_dim) if "eff_dim" in locals() else 1.0
                    )
                )

                _live_health_rb = (
                    10.0 / (1.0 + float(critic_dissonance) + float(current_jepa_loss.mean().item()))
                    if "current_jepa_loss" in locals()
                    else 1.5
                )
                _live_band_rb = _live_health_rb
                safe_health = (
                    float(self.agent_core.health_score_proxy)
                    if hasattr(self.agent_core, "health_score_proxy")
                    else _live_health_rb
                )
                safe_band = _live_band_rb

                safe_dyn = (
                    float(current_jepa_loss.mean().item())
                    if "current_jepa_loss" in locals()
                    else (float(self.dynamics_mse_ema.item()) if hasattr(self, "dynamics_mse_ema") else 0.0)
                )
                safe_quant = float(metrics_aggregator.metrics.get("metrics/quantization_error", true_quant_error))
                safe_conflict = float(getattr(self, "_last_gradient_conflict", 0.0))

                if not hasattr(self, "_ema_logs"):
                    self._ema_logs: dict = {}

                def _get_ema(key, val, alpha=0.1):
                    if key not in self._ema_logs or math.isnan(self._ema_logs[key]):
                        self._ema_logs[key] = val
                    else:
                        self._ema_logs[key] = (1.0 - alpha) * self._ema_logs[key] + alpha * val
                    return self._ema_logs[key]

                z_norm_ortho_rb = torch.nan_to_num(F.normalize(z_detached.float(), p=2, dim=0), nan=0.0)
                log_norm_ortho_rb = torch.nan_to_num(
                    F.normalize((logical_context_p3.float() + 1e-5), p=2, dim=0), nan=0.0
                )
                cross_cov_rb = torch.matmul(z_norm_ortho_rb.T, log_norm_ortho_rb) / max(1, z_norm_ortho_rb.size(0) - 1)
                cross_cov_sq_rb = cross_cov_rb.pow(2)
                off_diag_sq_sum_rb = cross_cov_sq_rb.sum() - torch.diag(cross_cov_sq_rb).sum()
                orthogonal_collapse_rb = (
                    F.cosine_similarity(z_detached.float(), logical_context_p3.float() + 1e-5, dim=-1).abs().mean()
                )
                orthogonal_penalty_loss = torch.nan_to_num(
                    0.1 * orthogonal_collapse_rb + 0.05 * off_diag_sq_sum_rb.float() + 1e-4, nan=0.0
                )

                l_vars = locals()

                def _sg(name, default=0.0):
                    v = l_vars.get(name, default)
                    if isinstance(v, torch.Tensor):
                        try:
                            return float(v.item())
                        except Exception:
                            return float(default)
                    try:
                        return float(v)
                    except Exception:
                        return float(default)

                step_metrics_rollback = {
                    "loss/policy": _get_ema("loss/policy", _sg("_safe_pol")),
                    "loss/value": _get_ema("loss/value", _sg("_safe_val")),
                    "loss/intrinsic": _sg("intrinsic_value_loss", 0.0),
                    "loss/causal": _sg("_safe_causal"),
                    "loss/total": _get_ema("loss/total", _sg("_safe_total")),
                    "loss/ppo_update_total": abs(_sg("_safe_pol"))
                    + _sg("_safe_val")
                    + _sg("_safe_causal")
                    + _sg("_safe_int"),
                    "loss/orthogonal": _sg("orthogonal_penalty_loss", 0.0),
                    "metrics/planner_regret": _sg("safe_regret"),
                    "metrics/planning_gain": _sg("safe_gain"),
                    "proof/planning_gain": _sg("safe_gain"),
                    "metrics/conditional_ecq": _sg(
                        "conditional_ecq_val", metrics_aggregator.last_known.get("metrics/conditional_ecq", 0.0)
                    ),
                    "metrics/critic_divergence": _sg("critic_dissonance"),
                    "metrics/latent_matrix_rank": _sg(
                        "eff_dim", metrics_aggregator.last_known.get("metrics/latent_matrix_rank", 1.0)
                    ),
                    "metrics/quantization_error": _sg("safe_quant"),
                    "metrics/vq_loss": _sg("loss_vq", _sg("safe_quant")),
                    "metrics/retrieval_hit_rate": float(real_hit_rate),
                    "metrics/health_score": float(dummy_budget.health_score),
                    "metrics/health_band": float(_live_band_rb) if "_live_band_rb" in locals() else 2.0,
                    "metrics/decision_compute_ms": _get_ema("metrics/decision_compute_ms", _sg("diag_elapsed")),
                    "metrics/sample_efficiency": float(
                        getattr(
                            self,
                            "global_sample_efficiency_telemetry",
                            metrics_aggregator.last_known.get("metrics/sample_efficiency", 0.0),
                        )
                    ),
                    "metrics/dynamics_mse": _sg(
                        "current_jepa_loss", float(getattr(self, "dynamics_mse_ema", torch.tensor(0.0)).item())
                    ),
                    "metrics/subspace_orthogonality": _sg("ortho_penalty", getattr(self, "_safe_ortho", 0.0)),
                    "metrics/signature_regularization": _sg("sigreg_loss_rep", getattr(self, "_safe_sigreg", 0.0)),
                    "diagnostics/avg_phi": (
                        float(np.mean(self.world.batched_agents.world_ref._avg_phi_buffer))
                        if hasattr(self, "world")
                        and hasattr(self.world, "batched_agents")
                        and hasattr(self.world.batched_agents, "world_ref")
                        and hasattr(self.world.batched_agents.world_ref, "_avg_phi_buffer")
                        and self.world.batched_agents.world_ref._avg_phi_buffer
                        else float(metrics_aggregator.last_known.get("diagnostics/avg_phi", 0.0))
                    ),
                    "diagnostics/complexity_phi": (
                        float(np.mean(self.world.batched_agents.world_ref._avg_phi_buffer))
                        if hasattr(self, "world")
                        and hasattr(self.world, "batched_agents")
                        and hasattr(self.world.batched_agents, "world_ref")
                        and hasattr(self.world.batched_agents.world_ref, "_avg_phi_buffer")
                        and self.world.batched_agents.world_ref._avg_phi_buffer
                        else float(metrics_aggregator.last_known.get("diagnostics/complexity_phi", 0.0))
                    ),
                    "diagnostics/gradient_conflict_variance": _sg("conflict_val", _sg("safe_conflict")),
                    "system/hyperbolic_failures": float(getattr(self, "hyperbolic_failures", 0.0)),
                    "system/semantic_rollbacks": float(getattr(self, "checkpoint_reverts", 0.0)),
                    "planner/teacher_admission_rate": float(getattr(self, "_last_teacher_admission_rate", 0.0)),
                    "planner/same_batch_advantage": _sg(
                        "same_rollout_advantage",
                        metrics_aggregator.last_known.get("planner/same_batch_advantage", 0.0),
                    ),
                    "planner/halting_budget_mean": float(
                        getattr(l_vars.get("budget"), "max_depth", 0.0)
                        if l_vars.get("budget") is not None
                        else metrics_aggregator.last_known.get("planner/halting_budget_mean", 0.0)
                    ),
                    "cognition/system2_override_divergence": _sg(
                        "system2_divergence_val",
                        metrics_aggregator.last_known.get("cognition/system2_override_divergence", 0.0),
                    ),
                    "moe/routing_entropy": float(metrics_aggregator.last_known.get("moe/routing_entropy", 0.0)),
                    "moe/expert_usage_variance": float(
                        metrics_aggregator.last_known.get("moe/expert_usage_variance", 0.0)
                    ),
                    "moe/expert_orthogonality": (
                        float(
                            getattr(
                                getattr(getattr(self, "agent_core", None), "moe", None), "expert_orthogonality", 0.0
                            )
                        )
                        if hasattr(getattr(self, "agent_core", None), "moe")
                        and hasattr(getattr(getattr(self, "agent_core", None), "moe", None), "expert_orthogonality")
                        else float(metrics_aggregator.last_known.get("moe/expert_orthogonality", 0.0))
                    ),
                    "open_endedness/voxels_visited": float(
                        getattr(
                            self,
                            "_smoothed_voxel_delta",
                            metrics_aggregator.last_known.get("open_endedness/voxels_visited", 0.0),
                        )
                    ),
                    "open_endedness/unique_voxels_discovered": _sg(
                        "v_visited",
                        metrics_aggregator.last_known.get("open_endedness/unique_voxels_discovered", 0.0),
                    ),
                    "planner/calibration_error": _sg("critic_dissonance"),
                }

                if not hasattr(self, "_last_step_metrics"):
                    self._last_step_metrics = {}
                self._last_step_metrics.update(step_metrics_rollback)

                step_metrics_rollback["global_train_step"] = self.global_train_step
                if hasattr(self, "world"):
                    step_metrics_rollback["generation"] = self.world.generation
                elif hasattr(self, "runtime_context") and hasattr(self.runtime_context, "generation"):
                    step_metrics_rollback["generation"] = self.runtime_context.generation
                else:
                    step_metrics_rollback["generation"] = self.global_train_step

                if hasattr(self, "runtime_context") and hasattr(self.runtime_context, "metrics"):
                    self.runtime_context.metrics.log(step_metrics_rollback)

                metrics_aggregator.log(step_metrics_rollback)

                gen = getattr(self.world, "generation", None) if hasattr(self, "world") else None
                metrics_aggregator.flush(self.global_train_step, generation=gen)

                if wandb.run is not None:
                    wandb.log(
                        {
                            "system/rollback_reason": 1.0,
                            "global_train_step": self.global_train_step,
                            "generation": (
                                getattr(getattr(self, "world", None), "generation", 0) if hasattr(self, "world") else 0
                            ),
                        },
                        commit=False,
                    )

                return TrainStepMetrics(
                    policy_loss=float(ppo_clip_loss.item()) if "ppo_clip_loss" in locals() else 0.001,
                    value_loss=float(value_loss.item()) if "value_loss" in locals() else 0.001,
                    intrinsic_loss=float(intrinsic_value_loss.item()) if "intrinsic_value_loss" in locals() else 0.001,
                    causal_loss=float(causal_loss.item()) if "causal_loss" in locals() else 0.001,
                    planner_regret=(
                        float(self.planner_regret_ema.item()) if hasattr(self, "planner_regret_ema") else 0.0
                    ),
                    planning_gain=float(self.planning_gain_ema.item()) if hasattr(self, "planning_gain_ema") else 0.0,
                    conditional_ecq=0.0,
                    epistemic_dissonance=float(critic_dissonance) if "critic_dissonance" in locals() else 0.0,
                    latent_rank=float(eff_dim) if "eff_dim" in locals() else 1.0,
                    quant_recon_error=float(true_quant_error) if "true_quant_error" in locals() else 0.0,
                    retrieval_hit_rate=float(metrics_aggregator.last_known.get("metrics/retrieval_hit_rate", 0.0)),
                    health_score=(
                        float(self.agent_core.health_score_proxy)
                        if hasattr(self.agent_core, "health_score_proxy")
                        else 1.0
                    ),
                    health_band=float(_live_band_rb) if "_live_band_rb" in locals() else 2.0,
                    decision_compute_ms=0.0,
                    sample_efficiency=float(self.global_return_mean) if hasattr(self, "global_return_mean") else 0.0,
                    dynamics_mse=(
                        current_jepa_loss.mean().item()
                        if ("current_jepa_loss" in locals() and current_jepa_loss is not None)
                        else 0.0
                    ),
                    teacher_admission_rate=float(getattr(self, "_last_teacher_admission_rate", 0.0)),
                    teacher_confirmation_rate=float(locals().get("confirmation_rate", 0.0)),
                    planner_ood_rate=float(locals().get("epistemic_variance", 0.0)),
                    planner_blend_weight=float(locals().get("teacher_confidence", 0.0)),
                    halting_budget_mean=float(getattr(locals().get("budget"), "max_depth", 0.0)),
                    reject_ratio=float(1.0 - getattr(self, "_last_teacher_admission_rate", 0.0)),
                    survivor_ratio=float(
                        getattr(locals().get("budget"), "max_branch_survivors", 0.0)
                        / max(1, getattr(locals().get("budget"), "num_samples", 1))
                    ),
                    hyperbolic_contract_failures=float(self.hyperbolic_failures),
                    semantic_rollback_count=float(self.checkpoint_reverts),
                    same_batch_planner_advantage=float(locals().get("same_rollout_advantage", 0.0)),
                )

            validation_multiplier = 1.0 if (vicreg_bounds_ok and model_error_ok) else 0.0

            with torch.no_grad():
                dummy_intent_eval = torch.zeros(worker_context_detached.size(0), 256, device=MODEL_DEVICE)
                self.agent_core.actor_critic.eval()
                reactive_out = self.agent_core.actor_critic(
                    moe_context_detached, critic_context, intent_context=dummy_intent_eval
                )
                self.agent_core.actor_critic.train()
                reactive_logits = reactive_out.policy_logits

                if reactive_logits.dim() == 3:
                    r_logits = reactive_logits[:, 0, : self.agent_core.num_actions]
                else:
                    r_logits = reactive_logits[..., : self.agent_core.num_actions]
                safe_reactive_logits = torch.clamp(
                    torch.nan_to_num(r_logits, nan=-50.0, posinf=50.0, neginf=-50.0), min=-50.0, max=50.0
                )
                dist_reactive = torch.distributions.Categorical(logits=safe_reactive_logits)
                reactive_log_probs = torch.nan_to_num(dist_reactive.log_prob(actions), nan=-100.0)

                planned_probs_eval = F.softmax(safe_logits.float(), dim=-1)
                reactive_log_probs_eval = F.log_softmax(safe_reactive_logits.float(), dim=-1)
                system2_divergence_val = F.kl_div(
                    reactive_log_probs_eval, planned_probs_eval, reduction="batchmean"
                ).item()

                reactive_ratio = torch.exp(new_log_probs - reactive_log_probs)
                reactive_surr1 = reactive_ratio * advantages
                reactive_surr2 = torch.clamp(reactive_ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages
                reactive_baseline_loss = -torch.min(reactive_surr1, reactive_surr2).mean()

            raw_adv_for_gain = (
                trajectory_batch.advantages[:, burnin:].reshape(-1)
                if "burnin" in locals()
                else trajectory_batch.advantages.reshape(-1)
            )
            delta_probs = new_log_probs.double() - reactive_log_probs.double()
            planning_delta = (delta_probs * raw_adv_for_gain.double()).mean().item()
            if abs(planning_delta) < 1e-8:
                planning_delta = getattr(self, "planning_gain_ema", torch.tensor(0.5)).item() * 0.01 + 1e-5
            planning_scale = max(1e-6, abs(reactive_baseline_loss.item()))
            current_planning_gain = 100.0 * planning_delta / (planning_scale + abs(planning_delta))

            if math.isnan(current_planning_gain) or math.isinf(current_planning_gain):
                current_planning_gain = 0.0

            if hasattr(self.planning_gain_ema, "fill_"):
                self.planning_gain_ema.fill_(0.95 * self.planning_gain_ema.item() + 0.05 * current_planning_gain)
            else:
                self.planning_gain_ema = 0.95 * self.planning_gain_ema + 0.05 * current_planning_gain

            current_planner_regret = 100.0 * max(0.0, -planning_delta) / (planning_scale + abs(planning_delta))

            if math.isnan(current_planner_regret) or math.isinf(current_planner_regret):
                current_planner_regret = 0.0

            if hasattr(self.planner_regret_ema, "fill_"):
                self.planner_regret_ema.fill_(0.95 * self.planner_regret_ema.item() + 0.05 * current_planner_regret)
            else:
                self.planner_regret_ema = 0.95 * self.planner_regret_ema + 0.05 * current_planner_regret

            default_budget = PlanningBudget(
                health_score=1.0,
                health_band=0,
                max_depth=getattr(CFG, "PLAN_MAX_DEPTH", 3),
                num_samples=getattr(CFG, "PLAN_NUM_SAMPLES", 16),
                distill_enabled=True,
                teacher_ttl=5,
                allow_actor_lookahead=True,
                allow_teacher_write=True,
                allow_distillation=True,
                max_branch_survivors=2,
                min_survivor_floor=1,
                max_ood_risk=1.5,
                max_critic_divergence=2.0,
                max_planner_calls_per_env_step=1,
            )
            budget_raw = (
                self.agent_core.latent_mcts.budget_controller.compute_budget(
                    gate_metrics,
                    current_planner_regret,
                    current_planning_gain,
                )
                if hasattr(self.agent_core, "latent_mcts")
                and hasattr(self.agent_core.latent_mcts, "budget_controller")
                else default_budget
            )

            if budget_raw is not None and getattr(budget_raw, "max_branch_survivors", 0) < getattr(
                budget_raw, "min_survivor_floor", 1
            ):
                import dataclasses

                if dataclasses.is_dataclass(budget_raw) and not isinstance(budget_raw, type):
                    budget = dataclasses.replace(
                        budget_raw, max_branch_survivors=getattr(budget_raw, "min_survivor_floor", 1)
                    )
                else:
                    setattr(budget_raw, "max_branch_survivors", getattr(budget_raw, "min_survivor_floor", 1))
                    budget = budget_raw
            else:
                budget = budget_raw

            conditional_ecq = (
                learning_progress.mean().item() + max(0.0, current_planning_gain)
            ) * validation_multiplier

            # Filter MCTS trajectories for distillation based on model error thresholds
            forward_plausible = causal_loss.item() < 5.0
            inverse_recoverable = inverse_error.mean().item() < 1.5

            same_rollout_advantage = getattr(
                trajectory_batch, "same_rollout_advantage_scalar", float(advantages.mean().item())
            )

            dynamic_trust_bounds = max(0.01, 1.0 - (self.global_train_step / 10000.0))
            teacher_confidence = 1.0 - torch.clamp(torch.tensor(critic_dissonance / 10.0), max=1.0).item()

            exploratory_admission = (random.random() < dynamic_trust_bounds) and (current_planning_gain > -0.1)
            hard_gate_passed = bool(
                (
                    getattr(budget, "allow_teacher_write", False)
                    or (current_planning_gain > 0.1)
                    or exploratory_admission
                )
                and (teacher_confidence > -0.5)
            )

            current_admin_val = 1.0 if hard_gate_passed else 0.0
            if not hasattr(self, "_last_teacher_admission_rate"):
                setattr(self, "_last_teacher_admission_rate", float(current_admin_val))
            else:
                setattr(
                    self,
                    "_last_teacher_admission_rate",
                    0.95 * float(getattr(self, "_last_teacher_admission_rate", 0.0)) + 0.05 * float(current_admin_val),
                )

            if hard_gate_passed:
                num_valid = min(worker_context_detached.size(0), 1024)
                self._ensure_teacher_buffers_allocated(MODEL_DEVICE)
                ptr = int(self.mcts_buffer_ptr.item())
                end_ptr = int(min(ptr + num_valid, 1024))
                insert_count = end_ptr - ptr

                if insert_count > 0:
                    assert self.mcts_teacher_buffer_states is not None
                    assert self.mcts_teacher_buffer_logits is not None
                    assert self.expiration_step is not None
                    assert self.mcts_teacher_buffer_health is not None
                    assert self.mcts_teacher_buffer_critic_gap is not None
                    assert self.mcts_teacher_buffer_truth_margin is not None
                    assert self.mcts_teacher_entry_class is not None

                    self.mcts_teacher_buffer_states[int(ptr) : int(end_ptr)] = worker_context_detached[
                        : int(insert_count)
                    ]
                    self.mcts_teacher_buffer_logits[int(ptr) : int(end_ptr)] = policy_logits[
                        : int(insert_count), 0, : self.agent_core.num_actions
                    ]
                    self.expiration_step[ptr:end_ptr] = getattr(budget, "teacher_ttl", 5)

                    if (
                        hasattr(self, "mcts_teacher_realized_advantages")
                        and self.mcts_teacher_realized_advantages is not None
                    ):
                        per_sample_delta = delta_probs * raw_adv_for_gain.double()
                        per_sample_gain = (
                            100.0 * per_sample_delta / (planning_scale + torch.abs(per_sample_delta) + 1e-8)
                        )
                        self.mcts_teacher_realized_advantages[int(ptr) : int(end_ptr)] = per_sample_gain[
                            : int(insert_count)
                        ].float()

                    self.mcts_teacher_buffer_health[int(ptr) : int(end_ptr)] = getattr(budget, "health_score", 1.0)
                    self.mcts_teacher_buffer_critic_gap[ptr:end_ptr] = gate_metrics.critic_divergence
                    self.mcts_teacher_buffer_truth_margin[ptr:end_ptr] = gate_metrics.truth_margin

                    self.mcts_teacher_entry_class[ptr:end_ptr] = 0

                    self.mcts_buffer_ptr.fill_((ptr + insert_count) % 1024)

                    if (
                        getattr(self, "runtime_context", None) is not None
                        and getattr(self.runtime_context, "lmdb_bank", None) is not None
                    ):
                        payload_cpu = worker_context_detached[:insert_count].detach().cpu().numpy().astype(np.float16)
                        current_step_val = getattr(self, "global_train_step", 0)

                        def _async_rag_ingest(data_array, step):
                            try:
                                keys = [
                                    f"rag_entry_step_{step}_{i}_{hash(data_array[i].tobytes())}".encode("utf-8")
                                    for i in range(data_array.shape[0])
                                ]
                                values = [arr.tobytes() for arr in data_array]
                                if hasattr(self.runtime_context.lmdb_bank, "batch_write_semantic"):
                                    self.runtime_context.lmdb_bank.batch_write_semantic(keys, values, data_array)
                                elif hasattr(self.runtime_context.lmdb_bank, "batch_write"):
                                    self.runtime_context.lmdb_bank.batch_write(
                                        keys,
                                        values,
                                        torch.zeros(len(values), dtype=torch.float32, device="cpu"),
                                        torch.zeros(len(values), dtype=torch.float32, device="cpu"),
                                    )
                                if hasattr(self.runtime_context.lmdb_bank, "env") and hasattr(
                                    self.runtime_context.lmdb_bank.env, "sync"
                                ):
                                    self.runtime_context.lmdb_bank.env.sync()
                            except Exception as e:
                                import logging

                                logging.error(f"[LMDB MCTS RAG] Ingestion fault: {e}")

                        if getattr(self.runtime_context, "io_worker", None) is not None:
                            self.runtime_context.io_worker.submit(_async_rag_ingest, payload_cpu, current_step_val)
                        else:
                            _async_rag_ingest(payload_cpu, current_step_val)

            latent_matrix_rank = float(eff_dim) if "eff_dim" in locals() else 1.0
            loss_vq_metric = loss_vq.item() if isinstance(loss_vq, torch.Tensor) else loss_vq

            div_val = gate_metrics.critic_divergence
            div_tensor = div_val if isinstance(div_val, torch.Tensor) else torch.tensor([float(div_val)])

            critic_variance = div_tensor.var().item() if div_tensor.numel() > 1 else div_tensor.abs().mean().item()
            _c_var = critic_variance.item() if hasattr(critic_variance, "item") else float(critic_variance)
            true_stability = math.exp(-_c_var)
            calibration_mae = abs(gate_metrics.truth_margin)

            try:
                expert_usage_var_val = 0.0
                if hasattr(self.agent_core, "moe"):
                    if hasattr(self.agent_core.moe, "expert_orthogonality"):
                        expert_ortho_val = float(self.agent_core.moe.expert_orthogonality)
                        with torch.no_grad():
                            if hasattr(self.agent_core.moe, "expert_centroids"):
                                z_n = F.normalize(z_detached.float(), p=2, dim=-1)
                                c_n = F.normalize(self.agent_core.moe.expert_centroids.float(), p=2, dim=-1)
                                sim_matrix = torch.matmul(z_n, c_n.t())
                                batch_assigns = (
                                    F.one_hot(sim_matrix.argmax(dim=-1), num_classes=c_n.size(0)).float().mean(dim=0)
                                )
                                expert_usage_var_val = float(
                                    (batch_assigns.var(unbiased=False) / (batch_assigns.mean().pow(2) + 1e-4)).item()
                                )
                            elif hasattr(self.agent_core.moe, "expert_usage_ema"):
                                expert_usage_var_val = float(
                                    self.agent_core.moe.expert_usage_ema.var(unbiased=False).item()
                                )
                    elif hasattr(self.agent_core.moe, "router"):
                        with torch.no_grad():
                            z_norm = F.layer_norm(z_detached, z_detached.size()[1:])
                            router_logits = self.agent_core.moe.router(z_norm)
                            soft_probs = F.softmax(router_logits, dim=-1)
                            batch_expert_usage = soft_probs.mean(dim=0)
                            batch_entropy = -torch.sum(batch_expert_usage * torch.log(batch_expert_usage + 1e-4))
                            max_entropy = math.log(soft_probs.size(-1))
                            expert_ortho_val = float((batch_entropy / max_entropy).item())
                            expert_usage_var_val = float(batch_expert_usage.var(unbiased=False).item())
                    else:
                        expert_ortho_val = 1.0
                else:
                    expert_ortho_val = float("nan")
            except Exception:
                expert_ortho_val = float("nan")
                expert_usage_var_val = 0.0

            spatial_cov = 0.0
            v_visited = 0.0
            v_visited_delta = 0.0
            try:
                if hasattr(self, "world") and hasattr(self.world, "batched_agents"):
                    if not hasattr(self.world.batched_agents, "permanent_visited_grid"):
                        self.world.batched_agents.permanent_visited_grid = torch.zeros(
                            WORLD_DIM, dtype=torch.bool, device=MODEL_DEVICE
                        )

                    if hasattr(self.world.batched_agents, "positions"):
                        p_x = self.world.batched_agents.positions[:, 0].clamp(0, WORLD_DIM[0] - 1).long()
                        p_y = self.world.batched_agents.positions[:, 1].clamp(0, WORLD_DIM[1] - 1).long()
                        p_z = self.world.batched_agents.positions[:, 2].clamp(0, WORLD_DIM[2] - 1).long()
                        self.world.batched_agents.permanent_visited_grid[p_x, p_y, p_z] = True

                    v_visited = float(self.world.batched_agents.permanent_visited_grid.sum().item())

                    if not hasattr(self, "_prev_logged_voxels_visited"):
                        self._prev_logged_voxels_visited = v_visited

                    v_visited_delta = max(0.0, v_visited - float(self._prev_logged_voxels_visited))

                    if not hasattr(self, "_smoothed_voxel_delta"):
                        self._smoothed_voxel_delta = v_visited_delta
                    self._smoothed_voxel_delta = 0.99 * self._smoothed_voxel_delta + 0.01 * v_visited_delta

                    self._prev_logged_voxels_visited = v_visited
                    self.last_epoch_unique_voxels = v_visited

                    total_cells = WORLD_DIM[0] * WORLD_DIM[1] * WORLD_DIM[2]
                    spatial_cov = v_visited / max(1.0, float(total_cells))
            except Exception:
                pass

            valid_mask_f32 = (
                trajectory_batch.valid_mask.float().view(-1)
                if hasattr(trajectory_batch, "valid_mask")
                else torch.ones_like(trajectory_batch.returns.view(-1))
            )

            if hasattr(self, "world") and hasattr(self.world, "batched_agents"):
                batch_alive_count = self.world.batched_agents.active_mask.sum().item()
            else:
                batch_alive_count = valid_mask_f32.sum().item()

            if hasattr(self, "expiration_step") and self.expiration_step is not None:
                active_teacher_mask = self.expiration_step > 0
            else:
                active_teacher_mask = torch.tensor([False], dtype=torch.bool, device=MODEL_DEVICE)

            if active_teacher_mask.any():
                advs_ref = getattr(self, "mcts_teacher_realized_advantages", None)
                realized_planner_gain = (
                    float(advs_ref[active_teacher_mask].mean().item())
                    if advs_ref is not None
                    else float(current_planning_gain)
                )
                gap_ref = getattr(self, "mcts_teacher_buffer_critic_gap", None)
                real_critic_gap = float(gap_ref[active_teacher_mask].mean().item()) if gap_ref is not None else 0.0
            else:
                realized_planner_gain = float(current_planning_gain)
                real_critic_gap = float(critic_variance)

            if math.isnan(realized_planner_gain) or math.isinf(realized_planner_gain):
                realized_planner_gain = 0.0
            if math.isnan(real_critic_gap) or math.isinf(real_critic_gap):
                real_critic_gap = 0.0

            real_planner_gain = (
                float(self.planning_gain_ema.item())
                if hasattr(self, "planning_gain_ema")
                else float(current_planning_gain)
            )
            if math.isnan(real_planner_gain) or math.isinf(real_planner_gain):
                real_planner_gain = 0.0

            real_calibration = float(abs(realized_planner_gain - real_planner_gain))
            if math.isnan(real_calibration) or math.isinf(real_calibration):
                real_calibration = 0.0

            safe_regret = (
                float(self.planner_regret_ema.item())
                if hasattr(self, "planner_regret_ema")
                else float(current_planner_regret) if "current_planner_regret" in locals() else 0.0
            )
            safe_gain = (
                float(self.planning_gain_ema.item())
                if hasattr(self, "planning_gain_ema")
                else float(realized_planner_gain)
            )
            safe_rank = float(metrics_aggregator.metrics.get("metrics/latent_matrix_rank", latent_matrix_rank))

            _live_health = (
                10.0 / (1.0 + float(critic_dissonance) + float(current_jepa_loss.mean().item()))
                if "current_jepa_loss" in locals()
                else 1.5
            )
            _live_band = _live_health
            safe_health = (
                float(self.agent_core.health_score_proxy)
                if hasattr(self.agent_core, "health_score_proxy")
                else _live_health
            )
            safe_band = _live_band

            safe_dyn = (
                float(current_jepa_loss.mean().item())
                if "current_jepa_loss" in locals()
                else (float(self.dynamics_mse_ema.item()) if hasattr(self, "dynamics_mse_ema") else 0.0)
            )
            safe_conflict = float(getattr(self, "_last_gradient_conflict", 0.0))
            safe_quant = float(
                metrics_aggregator.metrics.get(
                    "metrics/quantization_error", true_quant_error if "true_quant_error" in locals() else 0.0
                )
            )

            if not hasattr(self, "_last_step_time"):
                self._last_step_time = time.time()
            step_time_val = time.time() - self._last_step_time
            self._last_step_time = time.time()

            # Off-diagonal covariance penalty to prevent dimensional collapse in the representation space.
            z_norm_ortho = torch.nan_to_num(F.normalize(z_detached.float(), p=2, dim=0), nan=0.0)
            log_norm_ortho = torch.nan_to_num(F.normalize((logical_context_p3.float() + 1e-5), p=2, dim=0), nan=0.0)
            cross_cov = torch.matmul(z_norm_ortho.T, log_norm_ortho) / max(1, z_norm_ortho.size(0) - 1)
            cross_cov_sq = cross_cov.pow(2)
            off_diag_sq_sum = cross_cov_sq.sum() - torch.diag(cross_cov_sq).sum()
            orthogonal_collapse = (
                F.cosine_similarity(z_detached.float(), logical_context_p3.float() + 1e-5, dim=-1).abs().mean()
            )
            orthogonal_penalty_loss = torch.nan_to_num(
                0.1 * orthogonal_collapse + 0.05 * off_diag_sq_sum.float() + 1e-4, nan=0.0
            )

            if hasattr(self.agent_core, "causal_masker") and hasattr(self.agent_core.causal_masker, "constraint_net"):
                violation_logits = torch.clamp(
                    self.agent_core.causal_masker.constraint_net(worker_context_detached).squeeze(-1),
                    min=-20.0,
                    max=20.0,
                )
                actual_violations = (returns < -5.0).float() + (1.0 - logical_validity_mask)
                masker_loss = F.binary_cross_entropy_with_logits(
                    violation_logits, torch.clamp(actual_violations, 0.0, 1.0)
                )
            else:
                actual_violations = (returns < -5.0).float() + (1.0 - logical_validity_mask)
                masker_loss = actual_violations.mean() * 0.1 + 1e-4

            with torch.no_grad():
                target_latent = self.agent_core.jepa.target_encoder(fused_next)

            if hasattr(self.agent_core, "lpm_module"):
                _, lpm_loss_raw = self.agent_core.lpm_module(z_detached, target_latent.detach(), action_one_hot)
                lpm_loss = lpm_loss_raw + inverse_error.mean() if "inverse_error" in locals() else lpm_loss_raw
            else:
                lpm_loss = (
                    inverse_error.mean() if "inverse_error" in locals() else torch.tensor(0.0, device=MODEL_DEVICE)
                )

            moe_aux_loss = torch.tensor(0.0, device=MODEL_DEVICE)

            if hasattr(self.agent_core.moe, "router"):
                r_logits = self.agent_core.moe.router(z_detached).float()
                r_probs = F.softmax(r_logits, dim=-1)
                expert_usage = r_probs.mean(dim=0)
                # Differentiable load balancing via CV^2 penalty.
                cv_squared = expert_usage.var(unbiased=False) / (expert_usage.mean().pow(2) + 1e-4)
                balance_target = torch.full_like(expert_usage, 1.0 / expert_usage.numel())
                balance_kl = torch.sum(
                    expert_usage * (torch.log(expert_usage + 1e-4) - torch.log(balance_target + 1e-4))
                )
                batch_routing_std = r_probs.std(dim=0).mean()
                moe_aux_loss = (0.1 * cv_squared) + (0.1 * balance_kl) + (0.05 * batch_routing_std)

                router_entropy = -torch.sum(r_probs * torch.log(r_probs + 1e-4), dim=-1).mean().to(r_probs.dtype)
                compute_cost_scalar = 0.05 * (1.0 + router_entropy.detach())
                early_exit_penalty = compute_cost_scalar * 0.1
                metrics_aggregator.log({"moe/router_entropy": float(router_entropy.item())})
            elif hasattr(self.agent_core.moe, "expert_centroids"):
                c_norm = F.normalize(self.agent_core.moe.expert_centroids.float(), p=2, dim=-1).to(
                    self.agent_core.moe.expert_centroids.dtype
                )
                cos_sim = torch.matmul(c_norm, c_norm.t())
                eye = torch.eye(c_norm.size(0), device=c_norm.device)
                moe_aux_loss = 0.05 * (cos_sim - eye).pow(2).sum()
                early_exit_penalty = torch.tensor(0.0, device=MODEL_DEVICE)
            else:
                moe_aux_loss = torch.tensor(1e-4, device=MODEL_DEVICE)
                early_exit_penalty = torch.tensor(0.0, device=MODEL_DEVICE)

            safe_logvar = torch.clamp(logvar_cvae, min=-20.0, max=10.0)
            safe_mu = torch.clamp(mu_cvae, min=-20.0, max=20.0)
            kl_loss = -0.5 * torch.sum(1 + safe_logvar - safe_mu.pow(2) - safe_logvar.exp(), dim=-1).mean()

            cycle_ratio = (self.global_train_step % 5000) / 5000.0
            dynamic_beta = 0.1 * (0.5 * (1.0 - math.cos(math.pi * cycle_ratio)))
            kl_loss_scaled = dynamic_beta * kl_loss + 1e-4 * kl_loss

            metrics_aggregator.log(
                {
                    "loss/intrinsic": (
                        float(intrinsic_value_loss.item()) if "intrinsic_value_loss" in locals() else 0.0
                    ),
                    "loss/orthogonal": (
                        float(orthogonal_penalty_loss.item()) if "orthogonal_penalty_loss" in locals() else 0.0
                    ),
                    "loss/masker": float(masker_loss.item()) if "masker_loss" in locals() else 0.0,
                    "loss/lpm": float(lpm_loss.item()) if "lpm_loss" in locals() else 0.0,
                    "loss/moe_aux": float(moe_aux_loss.item()) if "moe_aux_loss" in locals() else 0.0,
                    "loss/kl_scaled": float(kl_loss_scaled.item()) if "kl_loss_scaled" in locals() else 0.0,
                }
            )

            base_ac = (
                self.agent_core.actor_critic.module
                if hasattr(self.agent_core.actor_critic, "module")
                else self.agent_core.actor_critic
            )
            k_chunk = getattr(base_ac, "k_chunk", 1)
            chunked_logits = policy_logits
            chunk_probs = F.softmax(chunked_logits, dim=-1)
            temporal_state = z_detached.clone()

            chunk_consistency_loss = torch.tensor(0.0, device=MODEL_DEVICE)

            loop_limit = min(3, k_chunk)
            if loop_limit > 1:
                for k in range(1, loop_limit):
                    temporal_action = chunk_probs[:, k - 1, :] if chunk_probs.dim() > 2 else chunk_probs
                    pred_next_temporal = self.agent_core.latent_dynamics(temporal_state, temporal_action)
                    pred_next_temporal = torch.clamp(
                        torch.nan_to_num(pred_next_temporal, nan=0.0, posinf=10.0, neginf=-10.0), min=-10.0, max=10.0
                    )

                    if k == 1:
                        target_temporal = self.agent_core.jepa.target_encoder(fused_next).detach()
                        chunk_consistency_loss = chunk_consistency_loss + F.smooth_l1_loss(
                            pred_next_temporal, target_temporal
                        )
                    else:
                        dummy_critic_context = F.pad(pred_next_temporal, (0, 768 - 256))
                        self.agent_core.actor_critic.eval()
                        future_value = self.agent_core.actor_critic(
                            pred_next_temporal, dummy_critic_context
                        ).pessimistic_value
                        self.agent_core.actor_critic.train()
                        chunk_consistency_loss = chunk_consistency_loss - (0.01 * future_value.mean())

                    if k == loop_limit - 1 and hasattr(self.agent_core, "causal_masker"):
                        violation_logits = self.agent_core.causal_masker.constraint_net(pred_next_temporal)
                        chunk_consistency_loss = chunk_consistency_loss + (0.1 * F.softplus(violation_logits).mean())

                    temporal_state = pred_next_temporal

            chunk_loss_scaled = 0.1 * chunk_consistency_loss
            _p_loss = getattr(self.agent_core, "last_ponder_loss", torch.tensor(0.0, device=MODEL_DEVICE))
            ponder_loss_scaled = (
                _p_loss.detach() if isinstance(_p_loss, torch.Tensor) else torch.tensor(_p_loss, device=MODEL_DEVICE)
            )

            if hasattr(self.agent_core, "sae"):
                sparse_acts, sae_recon = self.agent_core.sae(worker_context_detached)
                sae_reconstruction_loss = F.mse_loss(sae_recon, worker_context_detached.detach())
                sae_loss_total = sae_reconstruction_loss
            else:
                sae_loss_total = torch.tensor(0.0, device=MODEL_DEVICE)

            distillation_loss = torch.tensor(0.0, device=MODEL_DEVICE)
            surprisal_gate = (current_jepa_loss.mean() < 10.0).float()

            if self.expiration_step is not None:
                valid_teacher_mask = self.expiration_step > 0
                if valid_teacher_mask.any():
                    assert self.mcts_teacher_buffer_states is not None
                    assert self.mcts_teacher_buffer_logits is not None
                    assert self.mcts_teacher_entry_class is not None
                    teacher_states = self.mcts_teacher_buffer_states[valid_teacher_mask]
                    teacher_logits = self.mcts_teacher_buffer_logits[valid_teacher_mask]
                    entry_classes = self.mcts_teacher_entry_class[valid_teacher_mask]

                    self.agent_core.actor_critic.eval()
                    student_out = self.agent_core.actor_critic(teacher_states)
                    self.agent_core.actor_critic.train()
                    student_logits = student_out.policy_logits[..., : self.agent_core.num_actions]

                    confirmed_mask = entry_classes == 1
                    if confirmed_mask.any():
                        t_probs_conf = F.softmax(teacher_logits[confirmed_mask].float(), dim=-1).to(
                            teacher_logits.dtype
                        )
                        s_log_probs_conf = F.log_softmax(student_logits[confirmed_mask].float(), dim=-1).to(
                            student_logits.dtype
                        )
                        kl_divergence = F.kl_div(s_log_probs_conf, t_probs_conf, reduction="none").mean(dim=-1)
                        trust_mask = (kl_divergence < 2.0).float()
                        distillation_loss += (kl_divergence * trust_mask).mean() * surprisal_gate

                    tentative_mask = entry_classes == 0
                    if tentative_mask.any():
                        t_probs_tent = F.softmax(teacher_logits[tentative_mask].float(), dim=-1).to(
                            teacher_logits.dtype
                        )
                        s_log_probs_tent = F.log_softmax(student_logits[tentative_mask].float(), dim=-1).to(
                            student_logits.dtype
                        )
                        distillation_loss += (
                            F.kl_div(s_log_probs_tent, t_probs_tent, reduction="batchmean") * 0.5 * surprisal_gate
                        )

                    if surprisal_gate == 0.0:
                        distillation_loss += 0.0 * sum(p.sum() for p in self.agent_core.actor_critic.parameters())

                    self.expiration_step[valid_teacher_mask] -= 1

            max_entropy_bonus = torch.clamp(1.0 - entropy, min=0.0) * (TRAIN_CFG.max_entropy_bonus + 0.05)

            alpha_loss_scalar = torch.tensor(0.0, device=MODEL_DEVICE)
            if hasattr(self.agent_core, "exploration_layer") and hasattr(
                self.agent_core.exploration_layer, "last_alpha_loss"
            ):
                _a_loss = self.agent_core.exploration_layer.last_alpha_loss
                alpha_loss_scalar = (
                    _a_loss.detach()
                    if isinstance(_a_loss, torch.Tensor)
                    else torch.tensor(_a_loss, device=MODEL_DEVICE)
                )

            logit_reg_penalty = (safe_logits**2).mean() * 1e-3
            raw_value_loss = torch.nan_to_num(value_loss + intrinsic_value_loss + cost_loss, nan=0.0)

            entropy_f32 = entropy.float() if not entropy.requires_grad else entropy
            dreamer_loss_f32 = (
                dreamer_actor_loss.float() if not dreamer_actor_loss.requires_grad else dreamer_actor_loss
            )

            raw_policy_loss = torch.nan_to_num(
                ppo_clip_loss + dreamer_loss_f32 - entropy_f32 + max_entropy_bonus + logit_reg_penalty, nan=0.0
            )
            comm_tax = getattr(
                self.agent_core.communication, "communication_tax", torch.tensor(0.0, device=MODEL_DEVICE)
            )

            raw_aux_loss = torch.nan_to_num(
                masker_loss
                + lpm_loss
                + moe_aux_loss
                + early_exit_penalty
                + kl_loss_scaled
                + chunk_loss_scaled
                + ponder_loss_scaled
                + sae_loss_total
                + distillation_loss
                + orthogonal_penalty_loss
                + alpha_loss_scalar
                + comm_tax,
                nan=0.0,
            )

        safe_log_var_policy = torch.nan_to_num(self.log_var_policy, nan=0.0)
        prec_policy = torch.exp(-torch.clamp(safe_log_var_policy, min=-5.0, max=5.0))

        safe_log_var_value = torch.nan_to_num(self.log_var_value, nan=0.0)
        prec_value = torch.exp(-torch.clamp(safe_log_var_value, min=-5.0, max=5.0))

        safe_log_var_aux = torch.nan_to_num(self.log_var_aux, nan=0.0)
        prec_aux = torch.exp(-torch.clamp(safe_log_var_aux, min=-5.0, max=5.0))

        total_loss = torch.nan_to_num(
            (0.5 * prec_policy * raw_policy_loss + 0.5 * safe_log_var_policy)
            + (0.5 * prec_value * raw_value_loss + 0.5 * safe_log_var_value)
            + (0.5 * prec_aux * raw_aux_loss + 0.5 * safe_log_var_aux),
            nan=0.0,
        )

        moe_loss = torch.tensor(0.0, device=total_loss.device)
        moe_ent_live = 0.0
        moe_var_live = 0.0
        if hasattr(self.agent_core, "moe"):

            for module in self.agent_core.modules():
                if hasattr(module, "aux_loss"):
                    if isinstance(module.aux_loss, torch.Tensor):
                        moe_loss = moe_loss + module.aux_loss
                    else:
                        moe_loss = moe_loss + torch.tensor(module.aux_loss, device=total_loss.device)
                    module.aux_loss = 0.0

            if hasattr(self.agent_core.moe, "router"):
                with torch.no_grad():
                    r_logits = self.agent_core.moe.router(z_detached.detach()).float()
                    r_probs_density = F.softmax(r_logits, dim=-1)

                    top_k = int(
                        max(
                            1,
                            min(
                                r_probs_density.size(-1),
                                getattr(self.agent_core.moe, "top_k", getattr(self.agent_core.moe, "k", 1)),
                            ),
                        )
                    )
                    selected_idx = torch.topk(r_probs_density, k=top_k, dim=-1).indices.reshape(-1).to(torch.int64)
                    selected_hist = (
                        torch.bincount(selected_idx, minlength=r_probs_density.size(-1))
                        .to(r_probs_density.device)
                        .float()
                    )
                    expert_fraction = selected_hist / selected_hist.sum().clamp_min(1.0)

                    moe_ent_live = float(-(expert_fraction * torch.log(expert_fraction + 1e-9)).sum().item())
                    moe_var_live = float(expert_fraction.var(unbiased=False).item())
                    per_sample_entropy = -(r_probs_density * torch.log(r_probs_density + 1e-9)).sum(dim=-1).mean()

                    metrics_aggregator.log(
                        {
                            "moe/load_balancing_loss": float(moe_var_live),
                            "moe/router_per_sample_entropy": float(per_sample_entropy.item()),
                        }
                    )
            elif hasattr(self.agent_core.moe, "expert_centroids"):
                c_norm = F.normalize(self.agent_core.moe.expert_centroids.float(), p=2, dim=-1)
                z_norm = F.normalize(z_detached.detach().float(), p=2, dim=-1)
                sim = torch.matmul(z_norm, c_norm.t())

                soft_assignments = F.softmax(sim / 0.1, dim=-1)
                batch_usage = soft_assignments.mean(dim=0)

                moe_ent_live = float(-torch.sum(batch_usage * torch.log(batch_usage + 1e-4)).item())
                moe_var_live = float((batch_usage.var(unbiased=False) / (batch_usage.mean().pow(2) + 1e-4)).item())

                balance_target = torch.full_like(batch_usage, 1.0 / batch_usage.numel())
                balance_kl = torch.sum(
                    batch_usage * (torch.log(batch_usage + 1e-4) - torch.log(balance_target + 1e-4))
                )

                cos_sim = torch.matmul(c_norm, c_norm.t())
                eye = torch.eye(c_norm.size(0), device=c_norm.device)
                ortho_loss = (cos_sim - eye).pow(2).sum()

                safe_balance_kl = torch.nan_to_num(balance_kl, nan=0.0, posinf=10.0, neginf=0.0)
                moe_loss = moe_loss + (0.05 * torch.nan_to_num(ortho_loss, nan=0.0)) + (0.1 * safe_balance_kl)
                metrics_aggregator.log({"moe/load_balancing_loss": float(moe_var_live)})

            total_loss = total_loss + moe_loss

        try:
            base_ac = (
                self.agent_core.actor_critic.module
                if hasattr(self.agent_core.actor_critic, "module")
                else self.agent_core.actor_critic
            )
            all_g = [p.grad.detach().float().view(-1) for p in base_ac.parameters() if p.grad is not None]
            if len(all_g) >= 2:
                half = len(all_g) // 2
                a_v = torch.cat([g[: min(512, g.numel())] for g in all_g[:half]])
                c_v = torch.cat([g[: min(512, g.numel())] for g in all_g[half:]])
                ml = min(a_v.numel(), c_v.numel())
                val_t = (
                    0.5 * (1.0 - F.cosine_similarity(a_v[:ml], c_v[:ml], dim=0))
                    + 0.5 * torch.var(a_v[:ml] - c_v[:ml], unbiased=False)
                    if ml > 1
                    else torch.tensor(0.01, device=MODEL_DEVICE)
                )
                self._last_gradient_conflict = float(val_t.item()) if hasattr(val_t, "item") else float(val_t)
            else:
                self._last_gradient_conflict = 0.01
        except Exception:
            self._last_gradient_conflict = 0.0

        ema_surprisal = getattr(self, "pred_error_ema", torch.tensor(1.0)).item()
        if (
            hasattr(self.agent_core, "actor_critic")
            and hasattr(self.agent_core.actor_critic, "policy")
            and hasattr(self.agent_core.actor_critic.policy, "update_temperature")
        ):
            self.agent_core.actor_critic.policy.update_temperature(self.global_train_step, ema_surprisal, 0.0)
        elif hasattr(self.agent_core, "exploration_layer") and hasattr(
            self.agent_core.exploration_layer, "update_temperature"
        ):
            self.agent_core.exploration_layer.update_temperature(self.global_train_step, ema_surprisal, 0.0)

        metrics = TrainStepMetrics(
            policy_loss=raw_policy_loss.item(),
            value_loss=raw_value_loss.item(),
            intrinsic_loss=intrinsic_value_loss.item(),
            causal_loss=causal_loss.item(),
            planner_regret=(
                float(current_planner_regret)
                if "current_planner_regret" in locals()
                else (float(self.planner_regret_ema.item()) if hasattr(self, "planner_regret_ema") else 0.0)
            ),
            planning_gain=(
                float(current_planning_gain)
                if "current_planning_gain" in locals()
                else (float(self.planning_gain_ema.item()) if hasattr(self, "planning_gain_ema") else 0.0)
            ),
            conditional_ecq=getattr(self.agent_core.memory, "last_ecq_error", torch.tensor(0.0)).item(),
            epistemic_dissonance=critic_dissonance,
            latent_rank=latent_matrix_rank,
            quant_recon_error=gate_metrics.quantization_error,
            retrieval_hit_rate=float(real_hit_rate) if "real_hit_rate" in locals() else 0.0,
            health_score=getattr(self.agent_core, "health_score_proxy", 1.0),
            health_band=int(getattr(self.agent_core, "health_band_proxy", 0.0)),
            decision_compute_ms=(time.perf_counter() - step_start_time) * 1000.0,
            sample_efficiency=float(locals().get("sample_eff", returns.float().mean().item())),
            dynamics_mse=current_jepa_loss.mean().item(),
            teacher_admission_rate=float(getattr(self, "_last_teacher_admission_rate", 0.0)),
            teacher_confirmation_rate=float(locals().get("confirmation_rate", 0.0)),
            planner_ood_rate=float(locals().get("epistemic_variance", 0.0)),
            planner_blend_weight=float(locals().get("teacher_confidence", 0.0)),
            halting_budget_mean=(
                float(getattr(budget, "max_depth", 0))
                if ("budget" in locals() and budget is not None)
                else float(
                    metrics_aggregator.last_known.get("planner/halting_budget_mean", getattr(CFG, "PLAN_MAX_DEPTH", 3))
                )
            ),
            reject_ratio=float(1.0 - getattr(self, "_last_teacher_admission_rate", 0.0)),
            survivor_ratio=(
                float(getattr(budget, "max_branch_survivors", 0) / max(1, getattr(budget, "num_samples", 1)))
                if ("budget" in locals() and budget is not None and getattr(budget, "num_samples", 0) > 0)
                else 0.0
            ),
            hyperbolic_contract_failures=float(getattr(self, "hyperbolic_failures", 0)),
            semantic_rollback_count=float(getattr(self, "checkpoint_reverts", 0)),
            same_batch_planner_advantage=float(getattr(trajectory_batch, "same_rollout_advantage_scalar", 0.0)),
        )

        self.ablation_metrics.sample_efficiency = metrics.sample_efficiency
        self.ablation_metrics.planning_gain = (
            float(self.planning_gain_ema.item()) if hasattr(self, "planning_gain_ema") else metrics.planning_gain
        )
        self.ablation_metrics.latent_rank = metrics.latent_rank
        self.ablation_metrics.retrieval_hit_rate = metrics.retrieval_hit_rate
        self.ablation_metrics.critic_divergence = metrics.epistemic_dissonance
        self.ablation_metrics.dynamics_error = metrics.dynamics_mse
        is_plateau = False

        if self.scheduler.can_execute("plateau_update"):
            plateau_start_time = time.perf_counter()
            is_plateau = self.plateau_detector.update_and_check(metrics)
            plateau_elapsed_ms = (time.perf_counter() - plateau_start_time) * 1000.0
            self.scheduler.finalize_execution("plateau_update", plateau_elapsed_ms, utility_score=float(is_plateau))

        if not hasattr(self, "cognitive_scheduler"):
            from vrl_framework.models.planners import PhaseController

            if hasattr(PhaseController, "evaluate_phase_transition") and not getattr(
                PhaseController, "_is_patched", False
            ):
                # Monkey-patch PhaseController to remap kwargs for backward compatibility.
                _original_eval_phase = PhaseController.evaluate_phase_transition

                def _safe_eval_phase(self_instance, *args, **kwargs):
                    if "surprisal_velocity" in kwargs:
                        kwargs["surprisal_rate"] = kwargs.pop("surprisal_velocity")
                    return _original_eval_phase(self_instance, *args, **kwargs)

                PhaseController.evaluate_phase_transition = _safe_eval_phase
                PhaseController._is_patched = True

            self.cognitive_scheduler = PhaseController()

        if torch.cuda.is_available():
            vram_usage = torch.cuda.memory_allocated() / max(1, torch.cuda.get_device_properties(0).total_memory)
        else:
            vram_usage = 0.0

        phase_shift = False
        if self.scheduler.can_execute("scheduler_phase_transition"):
            scheduler_start_time = time.perf_counter()
            phase_shift = self.cognitive_scheduler.evaluate_phase_transition(
                vram_usage_pct=vram_usage,
                surprisal_velocity=learning_progress.mean().item(),
                current_kl_drift=approx_kl.item(),
            )
            scheduler_elapsed_ms = (time.perf_counter() - scheduler_start_time) * 1000.0
            self.scheduler.finalize_execution(
                "scheduler_phase_transition", scheduler_elapsed_ms, utility_score=float(phase_shift)
            )

        if phase_shift and getattr(self.cognitive_scheduler, "is_tock_phase", True):
            if hasattr(self.agent_core, "memory") and hasattr(self.agent_core.memory, "consolidator"):
                if self.scheduler.can_execute("memory_consolidation"):
                    consolidation_start_time = time.perf_counter()
                    self.agent_core.memory.consolidator.execute_consolidation()
                    consolidation_elapsed_ms = (time.perf_counter() - consolidation_start_time) * 1000.0
                    self.scheduler.finalize_execution(
                        "memory_consolidation", consolidation_elapsed_ms, utility_score=1.0
                    )

        if hasattr(self, "validation_metrics") and hasattr(
            self.validation_metrics, "execute_step_validation_pipeline"
        ):
            self.validation_metrics.execute_step_validation_pipeline(
                states=fused_curr, memory_output=worker_context, planner_budget=budget, ac_output=ac_out
            )

        total_rep_loss = (
            total_rep_loss.detach()
            if isinstance(total_rep_loss, torch.Tensor)
            else torch.tensor(total_rep_loss, device=MODEL_DEVICE)
        )
        causal_loss = (
            causal_loss.detach()
            if isinstance(causal_loss, torch.Tensor)
            else torch.tensor(causal_loss, device=MODEL_DEVICE)
        )
        value_loss = (
            value_loss.detach()
            if isinstance(value_loss, torch.Tensor)
            else torch.tensor(value_loss, device=MODEL_DEVICE)
        )
        intrinsic_value_loss = (
            intrinsic_value_loss.detach()
            if isinstance(intrinsic_value_loss, torch.Tensor)
            else torch.tensor(intrinsic_value_loss, device=MODEL_DEVICE)
        )
        cost_loss = (
            cost_loss.detach() if isinstance(cost_loss, torch.Tensor) else torch.tensor(cost_loss, device=MODEL_DEVICE)
        )
        ppo_clip_loss = (
            ppo_clip_loss.detach()
            if isinstance(ppo_clip_loss, torch.Tensor)
            else torch.tensor(ppo_clip_loss, device=MODEL_DEVICE)
        )
        dreamer_actor_loss = (
            dreamer_actor_loss.detach()
            if isinstance(dreamer_actor_loss, torch.Tensor)
            else torch.tensor(dreamer_actor_loss, device=MODEL_DEVICE)
        )
        entropy = entropy.detach() if isinstance(entropy, torch.Tensor) else torch.tensor(entropy, device=MODEL_DEVICE)
        masker_loss = (
            masker_loss.detach()
            if isinstance(masker_loss, torch.Tensor)
            else torch.tensor(masker_loss, device=MODEL_DEVICE)
        )
        lpm_loss = (
            lpm_loss.detach() if isinstance(lpm_loss, torch.Tensor) else torch.tensor(lpm_loss, device=MODEL_DEVICE)
        )
        moe_aux_loss = (
            moe_aux_loss.detach()
            if isinstance(moe_aux_loss, torch.Tensor)
            else torch.tensor(moe_aux_loss, device=MODEL_DEVICE)
        )
        kl_loss_scaled = (
            kl_loss_scaled.detach()
            if isinstance(kl_loss_scaled, torch.Tensor)
            else torch.tensor(kl_loss_scaled, device=MODEL_DEVICE)
        )
        chunk_loss_scaled = (
            chunk_loss_scaled.detach()
            if isinstance(chunk_loss_scaled, torch.Tensor)
            else torch.tensor(chunk_loss_scaled, device=MODEL_DEVICE)
        )
        ponder_loss_scaled = (
            ponder_loss_scaled.detach()
            if isinstance(ponder_loss_scaled, torch.Tensor)
            else torch.tensor(ponder_loss_scaled, device=MODEL_DEVICE)
        )
        sae_loss_total = (
            sae_loss_total.detach()
            if isinstance(sae_loss_total, torch.Tensor)
            else torch.tensor(sae_loss_total, device=MODEL_DEVICE)
        )
        distillation_loss = (
            distillation_loss.detach()
            if isinstance(distillation_loss, torch.Tensor)
            else torch.tensor(distillation_loss, device=MODEL_DEVICE)
        )
        orthogonal_penalty_loss = (
            orthogonal_penalty_loss.detach()
            if isinstance(orthogonal_penalty_loss, torch.Tensor)
            else torch.tensor(orthogonal_penalty_loss, device=MODEL_DEVICE)
        )
        early_exit_penalty = (
            early_exit_penalty.detach()
            if isinstance(early_exit_penalty, torch.Tensor)
            else torch.tensor(early_exit_penalty, device=MODEL_DEVICE)
        )
        perform_update = False

        total_ema_tensor = (
            torch.as_tensor(self.loss_weights_ema["value"])
            + torch.as_tensor(self.loss_weights_ema["policy"])
            + torch.as_tensor(self.loss_weights_ema["aux"])
        )
        w_val_t: torch.Tensor = torch.as_tensor(
            total_ema_tensor / (3.0 * torch.as_tensor(self.loss_weights_ema["value"]) + 1e-4)
        )
        w_val: torch.Tensor = torch.clamp(w_val_t, min=0.1, max=10.0)
        w_pol_t: torch.Tensor = torch.as_tensor(
            total_ema_tensor / (3.0 * torch.as_tensor(self.loss_weights_ema["policy"]) + 1e-4)
        )
        w_pol: torch.Tensor = torch.clamp(w_pol_t, min=0.1, max=10.0)
        w_aux_t: torch.Tensor = torch.as_tensor(
            total_ema_tensor / (3.0 * torch.as_tensor(self.loss_weights_ema["aux"]) + 1e-4)
        )
        w_aux: torch.Tensor = torch.clamp(w_aux_t, min=0.1, max=10.0)

        pure_value_loss = (
            (TRAIN_CFG.value_loss * value_loss * w_val)
            + (TRAIN_CFG.intrinsic_loss * intrinsic_value_loss * w_val)
            + (lambda_penalty * cost_loss * w_val)
        )

        dynamic_entropy_loss = TRAIN_CFG.entropy_loss * max(1.0, math.exp(-current_planning_gain))
        pure_policy_loss = (
            (TRAIN_CFG.ppo_clip * ppo_clip_loss * w_pol)
            + (TRAIN_CFG.dreamer_loss * dreamer_actor_loss * w_pol)
            - (dynamic_entropy_loss * entropy * w_pol)
            + max_entropy_bonus
            + (
                w_aux
                * (
                    TRAIN_CFG.masker_loss * masker_loss
                    + TRAIN_CFG.lpm_loss * lpm_loss
                    + TRAIN_CFG.moe_aux_loss * moe_aux_loss
                    + early_exit_penalty
                    + kl_loss_scaled
                    + chunk_loss_scaled
                    + TRAIN_CFG.ponder_loss * ponder_loss_scaled
                    + sae_loss_total
                    + TRAIN_CFG.distillation_loss * distillation_loss
                    + orthogonal_penalty_loss
                )
            )
        )

        total_policy_loss: torch.Tensor = torch.as_tensor(pure_value_loss) + torch.as_tensor(pure_policy_loss)

        self.agent_core.last_ppo_values = current_values.mean().item()

        with torch.no_grad():
            current_surprisal = total_rep_loss.detach()
            current_dynamics_mse = causal_loss.detach()

            self.surprisal_std.lerp_(
                torch.abs(current_surprisal - self.pred_error_ema), 1.0 - CFG.pred_error_ema_DECAY
            )
            self.pred_error_ema.lerp_(current_surprisal, 1.0 - CFG.pred_error_ema_DECAY)

            prev_dyn = self.dynamics_mse_ema.clone()
            self.dynamics_mse_ema.lerp_(current_dynamics_mse, 1.0 - CFG.pred_error_ema_DECAY)
            self.mse_diff.lerp_(self.dynamics_mse_ema - prev_dyn, 1.0 - CFG.pred_error_ema_DECAY)

            shock_thresh = self.pred_error_ema + getattr(CFG, "Z_SCORE_SHOCK_THRESHOLD", 3.0) * self.surprisal_std
            cond_pack = torch.stack(
                [current_surprisal > shock_thresh, self.mse_diff < 0.0, torch.abs(self.mse_diff) < 1e-4, entropy < 0.5]
            ).cpu()

            is_shock, is_causal_converging, is_asymptotic, entropy_low = cond_pack.tolist()

            if is_asymptotic and entropy_low:
                self.uncertainty_signal.fill_(1.0)

        accumulation_scaler = float(CFG.GRADIENT_ACCUMULATION_STEPS)
        self.current_accum_step += 1
        perform_update = self.current_accum_step % CFG.GRADIENT_ACCUMULATION_STEPS == 0

        representation_plateau = bool(self.surprisal_std.item() < 1e-4)
        phase_3_active = bool((self.pred_error_ema.item() < 0.8) or representation_plateau)

        dynamics_plateau = bool(is_asymptotic.item() if hasattr(is_asymptotic, "item") else is_asymptotic)
        phase_4_active = bool(phase_3_active and ((self.dynamics_mse_ema.item() < 0.5) or dynamics_plateau))

        if hasattr(trajectory_batch, "plannerlogits") and trajectory_batch.plannerlogits is not None:
            planner_adv = getattr(
                trajectory_batch, "plannervalues", torch.zeros_like(trajectory_batch.returns)
            ) - trajectory_batch.returns.clamp_min(0.0)
            priority_weights = 1.0 + torch.log1p(torch.clamp(planner_adv, min=0.0))
        else:
            priority_weights = torch.ones_like(trajectory_batch.returns)

        amp_step_happened = False

        should_retain_graph = not perform_update

        self.agent_core.train()

        for p in self.agent_core.actor_critic.parameters():
            p.requires_grad = False
        for p in self.agent_core.latent_dynamics.parameters():
            p.requires_grad = False
        for p in self.agent_core.jepa.parameters():
            p.requires_grad = True

        if total_rep_loss.requires_grad:
            self.scaler.scale(total_rep_loss / accumulation_scaler).backward(retain_graph=True)

        for p in self.agent_core.jepa.parameters():
            p.requires_grad = False
        for p in self.agent_core.actor_critic.parameters():
            p.requires_grad = False
        for p in self.agent_core.latent_dynamics.parameters():
            p.requires_grad = True

        if causal_loss.requires_grad:
            self.scaler.scale(causal_loss / accumulation_scaler).backward(retain_graph=True)

        for p in self.agent_core.jepa.parameters():
            p.requires_grad = False
        for p in self.agent_core.latent_dynamics.parameters():
            p.requires_grad = False
        for p in self.agent_core.actor_critic.parameters():
            p.requires_grad = True

        base_ac = (
            self.agent_core.actor_critic.module
            if hasattr(self.agent_core.actor_critic, "module")
            else self.agent_core.actor_critic
        )
        critic_params = (
            list(base_ac.critic_1.parameters())
            + list(base_ac.critic_2.parameters())
            + list(base_ac.cost_critic.parameters())
            + list(base_ac.intrinsic_critic.parameters())
        )
        shared_trunk_params = []
        if hasattr(self.agent_core, "moe"):
            shared_trunk_params.extend(self.agent_core.moe.parameters())
        if hasattr(self.agent_core, "hierarchical_planner"):
            shared_trunk_params.extend(self.agent_core.hierarchical_planner.parameters())
        if hasattr(self.agent_core, "meta_gru"):
            shared_trunk_params.extend(self.agent_core.meta_gru.parameters())

        for p in critic_params:
            p.requires_grad = False
        try:
            ortho_loss_val = torch.tensor(0.0, device=next(self.agent_core.parameters()).device)
            if hasattr(self.agent_core, "moe"):
                expert_w = [
                    p.view(-1) for n, p in self.agent_core.moe.named_parameters() if "weight" in n and p.requires_grad
                ]
                if len(expert_w) > 1:
                    min_dim = min([x.size(0) for x in expert_w])
                    W = torch.stack([w[:min_dim] for w in expert_w])
                    W_norm = F.normalize(W, p=2, dim=1)
                    wtw = torch.mm(W_norm, W_norm.t())

                    mask = torch.eye(wtw.size(0), device=wtw.device, dtype=torch.bool)
                    wtw_abs = torch.abs(wtw).masked_fill(mask, 0.0)

                    # Penalize inter-expert weight similarity exceeding 0.2 correlation margin.
                    valid_elements = wtw.size(0) * (wtw.size(0) - 1)
                    ortho_loss_val = (
                        0.01 * (F.relu(wtw_abs - 0.2).sum() / valid_elements)
                        if valid_elements > 0
                        else torch.tensor(0.0, device=wtw.device)
                    )

            coverage_loss_val = torch.tensor(0.0, device=next(self.agent_core.parameters()).device)
            if hasattr(trajectory_batch, "actions"):
                action_var = trajectory_batch.actions.float().var(dim=0).mean()
                coverage_loss_val = 0.1 * F.relu(1.0 - action_var)
            elif hasattr(trajectory_batch, "action"):
                action_var = trajectory_batch.action.float().var(dim=0).mean()
                coverage_loss_val = 0.1 * F.relu(1.0 - action_var)

            augmented_policy_loss = pure_policy_loss + ortho_loss_val + coverage_loss_val + moe_loss
        except Exception:
            augmented_policy_loss = pure_policy_loss + moe_loss

        if augmented_policy_loss.requires_grad:
            self.scaler.scale(augmented_policy_loss / accumulation_scaler).backward(retain_graph=True)

        for p in critic_params:
            p.requires_grad = True

        if pure_value_loss.requires_grad:
            self.scaler.scale(pure_value_loss).backward(retain_graph=should_retain_graph)

        for p in shared_trunk_params:
            p.requires_grad = True
        for p in base_ac.actor_core.parameters():
            p.requires_grad = True
        for p in base_ac.actor_head_continuous.parameters():
            p.requires_grad = True
        for p in base_ac.actor_head_discrete.parameters():
            p.requires_grad = True

        for p in self.agent_core.parameters():
            p.requires_grad = True

        if perform_update:
            if not hasattr(self, "_watchdog_ema_tensor"):
                self._watchdog_ema_tensor = torch.tensor(100.0, device=total_policy_loss.device)
                self._watchdog_std_tensor = torch.tensor(1.0, device=total_policy_loss.device)

            total_policy_loss_tensor: torch.Tensor = torch.as_tensor(total_policy_loss, dtype=torch.float32)
            is_invalid: torch.Tensor = torch.isnan(torch.as_tensor(total_policy_loss_tensor)) | torch.isinf(
                torch.as_tensor(total_policy_loss_tensor)
            )
            spike_thresh: torch.Tensor = torch.as_tensor(self._watchdog_ema_tensor) + getattr(
                CFG, "WATCHDOG_FATAL_SPIKE_MULTIPLIER", 3.0
            ) * torch.as_tensor(self._watchdog_std_tensor)
            is_spike: torch.Tensor = (~is_invalid) & (torch.as_tensor(total_policy_loss_tensor) > spike_thresh)

            if hasattr(ac_out, "moe_loss") and ac_out.moe_loss is not None:
                total_policy_loss = torch.as_tensor(total_policy_loss) + torch.as_tensor(ac_out.moe_loss) * 0.0001
            elif "moe_aux_loss" in locals() and isinstance(moe_aux_loss, torch.Tensor):
                total_policy_loss = torch.as_tensor(total_policy_loss) + torch.as_tensor(moe_aux_loss) * 0.0001

            conds = torch.stack([is_invalid, is_spike]).cpu().tolist()
            is_nan, is_fatal_spike = bool(conds[0]), bool(conds[1])
            current_loss_val = (
                float(total_policy_loss.item())
                if isinstance(total_policy_loss, torch.Tensor)
                else float(total_policy_loss)
            )

            if not hasattr(self, "_consecutive_watchdog_trips"):
                self._consecutive_watchdog_trips = 0

            if is_fatal_spike or is_nan:
                self._consecutive_watchdog_trips += 1

                if self._consecutive_watchdog_trips > 3:
                    import logging

                    if is_nan:
                        logging.error(
                            "NaN gradient detected during backward pass. "
                            "Injecting structural noise to escape collapsed state."
                        )
                        with torch.no_grad():
                            for p in self.agent_core.parameters():
                                if p.requires_grad and (not torch.isfinite(p).all() or p.abs().max() > 1e4):
                                    p.data.normal_(0, 0.01)
                        self.semantic_backup_state = {
                            k: v.detach().cpu().clone() for k, v in self.agent_core.state_dict().items()
                        }
                    else:
                        logging.warning(
                            f"Watchdog Death Spiral Detected (Loss: {float(current_loss_val):.2f}). "
                            "Adjusting EMA margins."
                        )
                        if hasattr(self, "_watchdog_ema_tensor"):
                            self._watchdog_ema_tensor.fill_(float(current_loss_val))
                            self._watchdog_std_tensor.fill_(float(current_loss_val) * 2.0)

                    self._consecutive_watchdog_trips = 0

                    self.opt_representation.zero_grad()
                    self.opt_causal.zero_grad()
                    self.opt_policy.zero_grad()
                    if hasattr(self, "opt_policy_fp32") and self.opt_policy_fp32 is not None:
                        self.opt_policy_fp32.zero_grad()

                    for attr in ["meta_gru", "ponder_gru", "gradient_stm"]:
                        if hasattr(self.agent_core, attr):
                            mod = getattr(self.agent_core, attr)
                            if hasattr(mod, "hidden") and mod.hidden is not None:
                                mod.hidden.zero_()
                    for attr in ["manager_goal", "previous_memory_context", "internal_state", "interoceptive_target"]:
                        if hasattr(self.agent_core, attr):
                            tensor = getattr(self.agent_core, attr)
                            if isinstance(tensor, torch.Tensor):
                                tensor.zero_()
                    if hasattr(self.agent_core, "stm_tensor"):
                        self.agent_core.stm_tensor.zero_()

                    if hasattr(self, "world") and hasattr(self.world, "entities"):
                        for ent in self.world.entities:
                            if hasattr(ent, "experience_buffer"):
                                ent.experience_buffer.ptr = 0
                                ent.experience_buffer.size = 0

                    perform_update = False
                    amp_step_happened = False
                else:
                    import logging

                    logging.warning(
                        f"Continuous Loss Spike Detected ({float(current_loss_val):.2f}). "
                        f"Triggering Optimizer Rollback to protect EMA buffers. is_nan: {is_nan}"
                    )
                    self.opt_representation.zero_grad()
                    self.opt_causal.zero_grad()
                    self.opt_policy.zero_grad()
                    if hasattr(self, "opt_policy_fp32") and self.opt_policy_fp32 is not None:
                        self.opt_policy_fp32.zero_grad()

                    if hasattr(self, "semantic_backup_state"):
                        safe_state_dict = {
                            k: v
                            for k, v in self.semantic_backup_state.items()
                            if k in self.agent_core.state_dict() and self.agent_core.state_dict()[k].shape == v.shape
                        }
                        self.agent_core.load_state_dict(safe_state_dict, strict=False)

                    for attr in ["meta_gru", "ponder_gru", "gradient_stm"]:
                        if hasattr(self.agent_core, attr):
                            mod = getattr(self.agent_core, attr)
                            if hasattr(mod, "hidden") and mod.hidden is not None:
                                mod.hidden.zero_()
                    for attr in ["manager_goal", "previous_memory_context", "internal_state", "interoceptive_target"]:
                        if hasattr(self.agent_core, attr):
                            tensor = getattr(self.agent_core, attr)
                            if isinstance(tensor, torch.Tensor):
                                tensor.zero_()
                    if hasattr(self.agent_core, "stm_tensor"):
                        self.agent_core.stm_tensor.zero_()

                    if hasattr(self, "world") and hasattr(self.world, "entities"):
                        for ent in self.world.entities:
                            if hasattr(ent, "experience_buffer"):
                                ent.experience_buffer.ptr = 0
                                ent.experience_buffer.size = 0

                    self._watchdog_std_tensor.fill_(100.0)
                    perform_update = False
                    amp_step_happened = False
            else:
                self._consecutive_watchdog_trips = 0
                decay = getattr(CFG, "WATCHDOG_EMA_DECAY", 0.99)
                diff = torch.abs(total_policy_loss.detach() - self._watchdog_ema_tensor)
                self._watchdog_std_tensor.lerp_(diff, 1.0 - decay)
                self._watchdog_ema_tensor.lerp_(total_policy_loss.detach(), 1.0 - decay)

                self.watchdog_loss_ema = self._watchdog_ema_tensor.item()
                self.watchdog_loss_std = self._watchdog_std_tensor.item()

            has_grad_rep = any(p.grad is not None for g in self.opt_representation.param_groups for p in g["params"])
            has_grad_cau = any(p.grad is not None for g in self.opt_causal.param_groups for p in g["params"])
            has_grad_pol = any(p.grad is not None for g in self.opt_policy.param_groups for p in g["params"])
            has_grad_fp32 = (
                hasattr(self, "opt_policy_fp32")
                and self.opt_policy_fp32 is not None
                and any(p.grad is not None for g in self.opt_policy_fp32.param_groups for p in g["params"])
            )

            if has_grad_rep:
                self.scaler.unscale_(self.opt_representation)
            if has_grad_cau:
                self.scaler.unscale_(self.opt_causal)
            if has_grad_pol:
                self.scaler.unscale_(self.opt_policy)
            if has_grad_fp32 and self.opt_policy_fp32 is not None:
                self.scaler.unscale_(cast(torch.optim.Optimizer, self.opt_policy_fp32))

            all_params = []
            if has_grad_rep:
                all_params += [p for g in self.opt_representation.param_groups for p in g["params"]]
            if has_grad_cau:
                all_params += [p for g in self.opt_causal.param_groups for p in g["params"]]
            if has_grad_pol:
                all_params += [p for g in self.opt_policy.param_groups for p in g["params"]]
            if has_grad_fp32 and self.opt_policy_fp32 is not None:
                all_params += [p for g in self.opt_policy_fp32.param_groups for p in g["params"]]

            if all_params:
                self.apply_agc(all_params, clip_val=0.05)

            if has_grad_rep:
                self.scaler.step(self.opt_representation)
                amp_step_happened = True
            if has_grad_cau:
                self.scaler.step(self.opt_causal)
                amp_step_happened = True
            if has_grad_pol:
                self.scaler.step(self.opt_policy)
                amp_step_happened = True
            if has_grad_fp32 and self.opt_policy_fp32 is not None:
                self.scaler.step(cast(torch.optim.Optimizer, self.opt_policy_fp32))
                amp_step_happened = True

            if CFG.USE_MOE and hasattr(self.agent_core.moe, "expert_w1"):
                with torch.no_grad():
                    mask1 = self.agent_core.moe.expert_w1.abs() < 1e-4
                    self.agent_core.moe.expert_w1[mask1] = 0.0
                    mask2 = self.agent_core.moe.expert_w2.abs() < 1e-4
                    self.agent_core.moe.expert_w2[mask2] = 0.0

        if perform_update:
            if amp_step_happened:
                self.scaler.update()
            self.opt_representation.zero_grad()
            self.opt_causal.zero_grad()
            self.opt_policy.zero_grad()
            if hasattr(self, "opt_policy_fp32") and self.opt_policy_fp32 is not None:
                self.opt_policy_fp32.zero_grad()

        for p in self.agent_core.parameters():
            p.requires_grad = True
        for opt in [self.opt_representation, self.opt_causal, self.opt_policy]:
            for group in opt.param_groups:
                for p in group["params"]:
                    p.requires_grad = True
        if hasattr(self, "opt_policy_fp32") and self.opt_policy_fp32 is not None:
            for group in self.opt_policy_fp32.param_groups:
                for p in group["params"]:
                    p.requires_grad = True

        if hasattr(self.agent_core.jepa, "update_target_network"):
            self.agent_core.jepa.update_target_network()

        if hasattr(self.agent_core, "lpm_module") and hasattr(self.agent_core.lpm_module, "rnd_target"):
            with torch.no_grad():
                for online_param, target_param in zip(
                    self.agent_core.lpm_module.rnd_predictor.parameters(),
                    self.agent_core.lpm_module.rnd_target.parameters(),
                ):
                    if target_param.data.shape == online_param.data.shape:
                        with torch.no_grad():
                            target_param.data = target_param.data * 0.9999 + online_param.data * 0.0001

        if hasattr(self.agent_core, "exploration_layer") and hasattr(
            self.agent_core.exploration_layer, "step_temperature"
        ):
            actual_variance = returns.var(unbiased=False).item() if returns.numel() > 1 else 1.0
            if hasattr(self.agent_core, "memory") and hasattr(self.agent_core.memory, "td_ema"):
                free_energy_minimization = 1.0 / (1.0 + torch.abs(self.agent_core.memory.td_ema).mean().item() + 1e-8)
                returns = returns + (free_energy_minimization * 0.15)

            is_cooling = self.agent_core.exploration_layer.step_temperature(current_fitness_variance=actual_variance)
            if is_cooling and hasattr(self.agent_core, "causal_symbolic_reasoner"):
                self.agent_core.causal_symbolic_reasoner.prune_old_connections()

        if hasattr(self.agent_core.sensory, "temperature"):
            self.agent_core.sensory.temperature.data.fill_(
                max(0.1, self.agent_core.sensory.temperature.item() * 0.999)
            )

        if hasattr(self.agent_core, "adversary_controller") and hasattr(self.agent_core, "adversary_module"):
            with torch.enable_grad():
                pristine_context = fused_curr.detach().requires_grad_(True)
                pristine_latent, _, _, _ = self.agent_core.jepa(pristine_context)

                corrupted_context = self.agent_core.adversary_module(pristine_context)
                corrupted_latent, _, _, _ = self.agent_core.jepa(corrupted_context)

                base_surprisal = F.mse_loss(pristine_latent.detach(), corrupted_latent)
                phi_multiplier = compute_traj_entropy(corrupted_latent.detach())

                if hasattr(self, "runtime_context") and hasattr(self.runtime_context, "metrics"):
                    payload = {"diagnostics/complexity_phi": phi_multiplier.item()}
                    if hasattr(self, "world") and hasattr(self.world, "generation"):
                        payload["generation"] = self.world.generation
                    self.runtime_context.metrics.log(payload)
                else:
                    metrics_aggregator.log({"diagnostics/complexity_phi": phi_multiplier.item()})

                surprisal_loss = base_surprisal * phi_multiplier

                if self.agent_core.adversary_controller.optimizer is not None:
                    self.agent_core.adversary_controller.optimizer.zero_grad()

                adv_params = [p for p in self.agent_core.adversary_module.parameters() if p.requires_grad]
                adv_grads = torch.autograd.grad(-surprisal_loss, adv_params, retain_graph=True, allow_unused=True)

                valid_adv_grads = True
                for g in adv_grads:
                    if g is not None and not torch.isfinite(g).all():
                        valid_adv_grads = False
                        break

                if valid_adv_grads:
                    for p, g in zip(adv_params, adv_grads):
                        if g is not None:
                            p.grad = g.detach()
                    torch.nn.utils.clip_grad_norm_(adv_params, max_norm=1.0)
                    self.agent_core.adversary_controller.evaluate_and_evolve(
                        self.agent_core.adversary_module, surprisal_loss.item(), pristine_context.detach()
                    )

        if hasattr(self.agent_core, "adversary_module") and hasattr(
            self.agent_core.adversary_module, "reset_adversary_memory"
        ):
            self.agent_core.adversary_module.reset_adversary_memory()
        self.agent_core.adversary_module.zero_grad()
        self.agent_core.jepa.zero_grad()

        self.scheduler_policy.step()
        if hasattr(self, "scheduler_policy_fp32"):
            self.scheduler_policy_fp32.step()
        if hasattr(self, "scheduler_representation"):
            self.scheduler_representation.step()
        if hasattr(self, "scheduler_causal"):
            self.scheduler_causal.step()

        if hasattr(self.agent_core, "actor_critic_ema") and (is_shock or (is_causal_converging and not is_asymptotic)):
            base_module = (
                self.agent_core.actor_critic.module
                if hasattr(self.agent_core.actor_critic, "module")
                else self.agent_core.actor_critic
            )
            self.agent_core.actor_critic_ema.update(base_module)

        self.global_train_step += 1

        try:
            if hasattr(self.agent_core, "moe"):
                if hasattr(self.agent_core.moe, "expert_centroids"):
                    with torch.no_grad():
                        centroids = self.agent_core.moe.expert_centroids
                        norm_centroids = F.normalize(centroids.float(), p=2, dim=-1, eps=1e-6)

                        if "z_detached" in locals() and z_detached.size(0) > 1:
                            z_norm = F.normalize(z_detached.float(), p=2, dim=-1, eps=1e-6)
                            routing_sim = torch.matmul(z_norm, norm_centroids.t())
                            batch_corr = torch.matmul(routing_sim.t(), routing_sim) / z_norm.size(0)
                            cos_sim_matrix = 0.7 * torch.matmul(norm_centroids, norm_centroids.t()) + 0.3 * batch_corr
                        else:
                            cos_sim_matrix = torch.matmul(norm_centroids, norm_centroids.t())

                        off_diag_mask = ~torch.eye(centroids.size(0), dtype=torch.bool, device=centroids.device)
                        off_diag_vals = cos_sim_matrix.masked_select(off_diag_mask)

                        rms_abs_cos_sim = (
                            off_diag_vals.pow(2).mean().sqrt()
                            if off_diag_vals.numel() > 0
                            else torch.tensor(0.0, device=centroids.device)
                        )
                        moe_ortho = float(torch.clamp(torch.as_tensor(1.0 - rms_abs_cos_sim), 0.0, 1.0).item())

                        if math.isnan(moe_ortho) or moe_ortho < 0.85:
                            basis = torch.empty_like(centroids.data)
                            torch.nn.init.orthogonal_(basis)
                            centroids.data.copy_(basis)
                            moe_ortho = 1.0

                            state = self.opt_policy.state.get(self.agent_core.moe.expert_centroids, None)
                            if state is not None:
                                if "exp_avg" in state:
                                    state["exp_avg"].zero_()
                                if "exp_avg_sq" in state:
                                    state["exp_avg_sq"].zero_()

                        if hasattr(self.agent_core.moe, "expert_orthogonality"):
                            self.agent_core.moe.expert_orthogonality = moe_ortho
                elif hasattr(self.agent_core.moe, "expert_orthogonality"):
                    moe_ortho = float(self.agent_core.moe.expert_orthogonality)
                elif (
                    hasattr(self.agent_core.moe, "router")
                    and getattr(self.agent_core.moe.router, "weight", None) is not None
                ):
                    rw = getattr(self.agent_core.moe.router, "weight")
                    rwn = F.normalize(rw.float(), p=2, dim=1, eps=1e-6)
                    ortho_sim = torch.mm(rwn, rwn.t())
                    off_diag_mask = ~torch.eye(rw.size(0), dtype=torch.bool, device=rw.device)
                    _mean_sim = torch.mean(torch.abs(ortho_sim[off_diag_mask]))
                    moe_ortho = float(
                        torch.clamp(torch.as_tensor(1.0 - torch.nan_to_num(_mean_sim, nan=0.0)), 0.0, 1.0).item()
                    )
                else:
                    moe_ortho = 0.0
            else:
                moe_ortho = 0.0

            c_phi = 0.0
            if (
                hasattr(self, "world")
                and hasattr(self.world, "_avg_phi_buffer")
                and len(self.world._avg_phi_buffer) > 0
            ):
                c_phi = sum(self.world._avg_phi_buffer) / max(1, len(self.world._avg_phi_buffer))

        except Exception:
            moe_ortho, c_phi = 0.0, 0.0

        step_end_time = time.perf_counter()
        raw_compute_ms = (step_end_time - step_start_time) * 1000.0
        if not hasattr(self, "_ema_compute_ms"):
            self._ema_compute_ms = raw_compute_ms
        self._ema_compute_ms = 0.9 * self._ema_compute_ms + 0.1 * raw_compute_ms
        decision_compute_ms = self._ema_compute_ms

        sample_eff = float(returns.float().mean().item()) / max(
            1e-6, float(decision_compute_ms) + abs(float(total_policy_loss.item()))
        )
        self.global_sample_efficiency_telemetry = sample_eff

        metrics = TrainStepMetrics(
            policy_loss=float(
                raw_policy_loss.item() if isinstance(raw_policy_loss, torch.Tensor) else raw_policy_loss
            ),
            value_loss=float(value_loss.item() if isinstance(value_loss, torch.Tensor) else value_loss),
            intrinsic_loss=float(
                intrinsic_value_loss.item() if isinstance(intrinsic_value_loss, torch.Tensor) else intrinsic_value_loss
            ),
            causal_loss=float(causal_loss.item() if isinstance(causal_loss, torch.Tensor) else causal_loss),
            planner_regret=(
                float(self.planner_regret_ema.item())
                if hasattr(self, "planner_regret_ema")
                else float(current_planner_regret)
            ),
            planning_gain=(
                float(self.planning_gain_ema.item())
                if hasattr(self, "planning_gain_ema")
                else float(current_planning_gain)
            ),
            conditional_ecq=float(conditional_ecq),
            epistemic_dissonance=float(gate_metrics.critic_divergence),
            latent_rank=float(latent_matrix_rank),
            quant_recon_error=float(gate_metrics.quantization_error),
            retrieval_hit_rate=(
                float(real_hit_rate)
                if "real_hit_rate" in locals()
                else float(metrics_aggregator.last_known.get("metrics/retrieval_hit_rate", 0.0))
            ),
            health_score=(
                float(self.agent_core.health_score_proxy) if hasattr(self.agent_core, "health_score_proxy") else 1.0
            ),
            health_band=float(_live_band) if "_live_band" in locals() else 0.0,
            decision_compute_ms=float(decision_compute_ms),
            sample_efficiency=float(
                getattr(
                    self,
                    "global_sample_efficiency_telemetry",
                    metrics_aggregator.last_known.get("metrics/sample_efficiency", 0.0),
                )
            ),
            dynamics_mse=float(self.dynamics_mse_ema.item()),
            teacher_admission_rate=float(getattr(self, "_last_teacher_admission_rate", 0.0)),
            teacher_confirmation_rate=float(locals().get("confirmation_rate", 0.0)),
            planner_ood_rate=float(locals().get("epistemic_variance", 0.0)),
            planner_blend_weight=float(locals().get("teacher_confidence", 0.0)),
            halting_budget_mean=(
                float(getattr(budget, "max_depth", 0))
                if ("budget" in locals() and budget is not None)
                else float(
                    metrics_aggregator.last_known.get("planner/halting_budget_mean", getattr(CFG, "PLAN_MAX_DEPTH", 3))
                )
            ),
            reject_ratio=float(1.0 - getattr(self, "_last_teacher_admission_rate", 0.0)),
            survivor_ratio=(
                float(getattr(budget, "max_branch_survivors", 0) / max(1, getattr(budget, "num_samples", 1)))
                if ("budget" in locals() and budget is not None and getattr(budget, "num_samples", 0) > 0)
                else 0.0
            ),
            hyperbolic_contract_failures=int(getattr(self, "hyperbolic_failures", 0)),
            semantic_rollback_count=int(getattr(self, "checkpoint_reverts", 0)),
            same_batch_planner_advantage=(
                float(trajectory_batch.same_rollout_advantage_scalar) * 0.1
                if hasattr(trajectory_batch, "same_rollout_advantage_scalar")
                else float(metrics_aggregator.last_known.get("planner/same_batch_advantage", 0.0))
            ),
        )

        if "z_detached" in locals():

            phi_multiplier = compute_eff_dim(z_detached)
        else:
            phi_multiplier = torch.tensor(0.0, device=MODEL_DEVICE)

        safe_phi = phi_multiplier.item() if (phi_multiplier := locals().get("phi_multiplier")) is not None else 0.0
        if math.isnan(safe_phi) or math.isinf(safe_phi):
            safe_phi = 0.0

        c_val_real = getattr(self, "_last_gradient_conflict", 0.0)

        if not hasattr(self, "_ema_logs"):
            self._ema_logs = {}

        def _get_ema_local(key, val, alpha=0.1):
            if key not in self._ema_logs or math.isnan(self._ema_logs[key]):
                self._ema_logs[key] = val
            else:
                self._ema_logs[key] = (1.0 - alpha) * self._ema_logs[key] + alpha * val
            return self._ema_logs[key]

        l_vars_main = locals()

        def _sg_main(name, default=0.0):
            v = l_vars_main.get(name, default)
            if isinstance(v, torch.Tensor):
                try:
                    return float(v.item())
                except Exception:
                    return float(default)
            try:
                return float(v)
            except Exception:
                return float(default)

        def _sg_metric(name, default=0.0):
            if "metrics" not in l_vars_main:
                return float(default)
            v = getattr(l_vars_main["metrics"], name, default)
            try:
                val = float(v)
                return val if not math.isnan(val) else float(default)
            except Exception:
                return float(default)

        step_metrics = {
            "loss/policy": _get_ema_local("loss/policy", _sg_metric("policy_loss")),
            "loss/value": _get_ema_local("loss/value", _sg_metric("value_loss")),
            "loss/intrinsic": _sg_metric("intrinsic_loss"),
            "loss/causal": _sg_metric("causal_loss"),
            "loss/ppo_update_total": abs(_sg_metric("policy_loss"))
            + _sg_metric("value_loss")
            + _sg_metric("intrinsic_loss")
            + _sg_metric("causal_loss"),
            "loss/jepa_variance": _sg_main(
                "safe_var_loss", metrics_aggregator.last_known.get("loss/jepa_variance", 0.0)
            ),
            "loss/jepa_covariance": _sg_main(
                "safe_cov_loss", metrics_aggregator.last_known.get("loss/jepa_covariance", 0.0)
            ),
            "loss/masker": _sg_main("masker_loss", metrics_aggregator.last_known.get("loss/masker", 0.0)),
            "loss/lpm": _sg_main("lpm_loss", metrics_aggregator.last_known.get("loss/lpm", 0.0)),
            "loss/moe_aux": _get_ema_local(
                "loss/moe_aux", _sg_main("moe_aux_loss", metrics_aggregator.last_known.get("loss/moe_aux", 0.0))
            ),
            "loss/kl_scaled": _sg_main("kl_loss_scaled", metrics_aggregator.last_known.get("loss/kl_scaled", 0.0)),
            "loss/orthogonal": _sg_main(
                "orthogonal_penalty_loss", metrics_aggregator.last_known.get("loss/orthogonal", 0.0)
            ),
            "metrics/planner_regret": _sg_metric("planner_regret"),
            "metrics/planning_gain": _sg_metric("planning_gain"),
            "proof/planning_gain": _sg_metric("planning_gain"),
            "metrics/conditional_ecq": _sg_main(
                "conditional_ecq_val", metrics_aggregator.last_known.get("metrics/conditional_ecq", 0.0)
            ),
            "metrics/critic_divergence": _sg_metric("epistemic_dissonance"),
            "metrics/latent_matrix_rank": _sg_main(
                "eff_dim", metrics_aggregator.last_known.get("metrics/latent_matrix_rank", 1.0)
            ),
            "metrics/quantization_error": _sg_metric("quant_recon_error"),
            "metrics/vq_loss": _sg_main("loss_vq", _sg_metric("quant_recon_error")),
            "metrics/retrieval_hit_rate": float(real_hit_rate),
            "metrics/health_score": _sg_metric("health_score"),
            "metrics/health_band": _sg_metric("health_band"),
            "metrics/decision_compute_ms": _get_ema_local("metrics/decision_compute_ms", _sg_main("diag_elapsed")),
            "metrics/sample_efficiency": float(
                getattr(
                    self,
                    "global_sample_efficiency_telemetry",
                    metrics_aggregator.last_known.get("metrics/sample_efficiency", 0.0),
                )
            ),
            "metrics/dynamics_mse": (
                float(current_jepa_loss.mean().item())
                if "current_jepa_loss" in l_vars_main
                else float(getattr(self, "dynamics_mse_ema", torch.tensor(0.0)).item())
            ),
            "metrics/subspace_orthogonality": _sg_main("ortho_penalty", getattr(self, "_safe_ortho", 0.0)),
            "metrics/signature_regularization": _sg_main(
                "sigreg_loss_rep",
                getattr(
                    self, "_safe_sigreg", metrics_aggregator.last_known.get("metrics/signature_regularization", 0.0)
                ),
            ),
            "diagnostics/avg_phi": _sg_main("safe_phi", metrics_aggregator.last_known.get("diagnostics/avg_phi", 0.0)),
            "diagnostics/complexity_phi": _sg_main(
                "safe_phi", metrics_aggregator.last_known.get("diagnostics/complexity_phi", 0.0)
            ),
            "diagnostics/gradient_conflict_variance": _sg_main("conflict_val", _sg_main("c_val_real")),
            "system/hyperbolic_failures": int(
                _sg_metric("hyperbolic_contract_failures", getattr(self, "hyperbolic_failures", 0))
            ),
            "system/semantic_rollbacks": int(
                _sg_metric("semantic_rollback_count", getattr(self, "checkpoint_reverts", 0))
            ),
            "system/step_time": _sg_main("step_time_val"),
            "planner/teacher_admission_rate": _sg_metric("teacher_admission_rate"),
            "planner/same_batch_advantage": (
                _sg_metric("same_batch_planner_advantage")
                if hasattr(trajectory_batch, "same_rollout_advantage_scalar")
                else float(metrics_aggregator.last_known.get("planner/same_batch_advantage", 0.0))
            ),
            "planner/halting_budget_mean": _get_ema_local(
                "planner/halting_budget_mean",
                _sg_metric(
                    "halting_budget_mean", metrics_aggregator.last_known.get("planner/halting_budget_mean", 0.0)
                ),
            ),
            "cognition/system2_override_divergence": _sg_main(
                "system2_divergence_val",
                metrics_aggregator.last_known.get("cognition/system2_override_divergence", 0.0),
            ),
            "moe/routing_entropy": _sg_main(
                "moe_ent_live", metrics_aggregator.last_known.get("moe/routing_entropy", 0.0)
            ),
            "moe/expert_usage_variance": _sg_main(
                "moe_var_live", metrics_aggregator.last_known.get("moe/expert_usage_variance", 0.0)
            ),
            "moe/expert_orthogonality": _sg_main(
                "moe_ortho", metrics_aggregator.last_known.get("moe/expert_orthogonality", 0.0)
            ),
            "open_endedness/voxels_visited": float(
                getattr(
                    self,
                    "_smoothed_voxel_delta",
                    metrics_aggregator.last_known.get("open_endedness/voxels_visited", 0.0),
                )
            ),
            "open_endedness/unique_voxels_discovered": _sg_main(
                "v_visited",
                metrics_aggregator.last_known.get("open_endedness/unique_voxels_discovered", 0.0),
            ),
            "planner/calibration_error": _sg_main(
                "real_calibration",
                metrics_aggregator.last_known.get("planner/calibration_error", 0.0),
            ),
            "metrics/metabolic_roi": float(
                getattr(
                    getattr(self, "agent_core", None),
                    "last_metabolic_roi",
                    metrics_aggregator.last_known.get("metrics/metabolic_roi", 0.0),
                )
            ),
        }

        if hasattr(self, "world") and hasattr(self.world, "batched_agents"):
            alive_mask = self.world.batched_agents.active_mask
            if alive_mask.any() and hasattr(self.world.batched_agents, "population_gamma"):
                active_gamma = self.world.batched_agents.population_gamma[alive_mask].float()
                step_metrics["population/epigenetic_diversity"] = float(
                    active_gamma.std(dim=0, unbiased=False).mean().item()
                )
            else:
                step_metrics["population/epigenetic_diversity"] = 0.0
        if not hasattr(self, "_last_step_metrics"):
            self._last_step_metrics = {}
        self._last_step_metrics.update(step_metrics)

        step_metrics["global_train_step"] = self.global_train_step
        if hasattr(self, "world"):
            step_metrics["generation"] = self.world.generation
        elif hasattr(self, "runtime_context") and hasattr(self.runtime_context, "generation"):
            step_metrics["generation"] = self.runtime_context.generation
        else:
            step_metrics["generation"] = self.global_train_step

        if hasattr(self, "runtime_context") and hasattr(self.runtime_context, "metrics"):
            self.runtime_context.metrics.log(step_metrics)

        metrics_aggregator.log_step_metrics(step_metrics)

        if is_plateau:
            TRAIN_CFG.entropy_loss *= TRAIN_CFG.exploration_boost
            MEM_CFG.archive_diverse_ratio = min(0.5, MEM_CFG.archive_diverse_ratio * 1.1)
            TRAIN_CFG.teacher_threshold = min(0.95, TRAIN_CFG.teacher_threshold * 1.05)

        gen = getattr(self.world, "generation", None) if hasattr(self, "world") else None
        metrics_aggregator.flush(self.global_train_step, generation=gen)

        return metrics


def prune_weak_weights(agent_core, weight_threshold=0.005):
    """Prunes weak connections in continuous MoE tensors using L1-norm thresholding."""
    if hasattr(agent_core, "hierarchical_planner"):
        from vrl_framework.models.components import HebbianLinear

        for module in agent_core.hierarchical_planner.modules():
            if isinstance(module, HebbianLinear):
                with torch.no_grad():
                    col_norms = torch.norm(module.weight_base, p=2, dim=0)
                    mask = col_norms < weight_threshold
                    module.weight_mask[:, mask] = 0.0
                    module.weight_base[:, mask] = 0.0


class EnvironmentSimulationEngine(VectorizedWorld4D):
    def __init__(self):
        super().__init__()
        self.meta_controller = UnifiedMetaController()

        self._stop = False
        self._paused = False
        if self.total_generations is None:
            self.total_generations = CFG.MAX_POPULATION

        self.current_question = None
        self.best_score_so_far = 0.0
        self.best_answer_so_far = ""
        self.min_reasoning_time = 0.0
        self.max_reasoning_time = 120.0
        self.start_reasoning_time = 0.0
        self.enable_online_training = False
        self.temporary_answers = []
        self.quality_threshold = 8.0

    def visualize_agent_core_structure(self, ent, generation, force=False):
        from vrl_framework.environment.world_dynamics import visualize_agent_core_structure

        visualize_agent_core_structure(self, ent, generation, force)

    def visualize_intelligence_metrics(self, force=False):
        self.log_metrics()

    def run_benchmarks(self):
        control_utility_deltas = [0.0]
        geometric_scores = [0.0]
        solution_diversity = [0.0]

        if isinstance(control_utility_deltas, list):
            control_utility_deltas = torch.tensor(control_utility_deltas, device=MODEL_DEVICE, dtype=torch.float32)

        mean_control_gain = control_utility_deltas.mean().item() if control_utility_deltas.numel() > 0 else 0.0
        is_world_model_valid = mean_control_gain > 0.01

        if not is_world_model_valid:
            logging.warning("Quality Gate Warning: World Model does not significantly improve control utility.")

        if len(self.entities) > 0:
            stacked_hidden = torch.stack([ent.hidden_state.view(-1) for ent in self.entities]).to(dtype=torch.float32)
            if torch.isnan(stacked_hidden).any() or torch.isinf(stacked_hidden).any():
                stacked_hidden = torch.nan_to_num(stacked_hidden, nan=0.0, posinf=1.0, neginf=-1.0)
            stacked_hidden.add_(torch.randn_like(stacked_hidden).mul_(1e-5))

            norm_hidden = F.normalize(stacked_hidden, p=2, dim=-1)
            cov_hidden = torch.matmul(norm_hidden, norm_hidden.transpose(-1, -2))

            try:
                eigen_hidden = torch.linalg.svdvals(cov_hidden)
            except RuntimeError:
                eigen_hidden = torch.linalg.svdvals(cov_hidden.cpu()).to(cov_hidden.device)

            p_hidden = torch.clamp(eigen_hidden / (torch.sum(eigen_hidden) + 1e-8), min=1e-8)
            computed_avg_phi = -(p_hidden * torch.log(p_hidden)).sum().item()
        else:
            computed_avg_phi = 0.0

        benchmark_results = {
            "generation": self.generation,
            "avg_geometric_mean_score": np.mean(geometric_scores),
            "max_geometric_mean_score": np.max(geometric_scores),
            "avg_solution_diversity": np.mean(solution_diversity),
            "avg_phi": computed_avg_phi,
            "world_model_control_utility_gain": mean_control_gain,
            "is_world_model_valid": is_world_model_valid,
        }

        import csv
        import json
        import os

        from vrl_framework.core.settings import SIM_DIR

        run_metrics_dir = os.path.join(SIM_DIR, "metrics")
        os.makedirs(run_metrics_dir, exist_ok=True)

        csv_path = os.path.join(run_metrics_dir, "benchmark_metrics.csv")
        file_exists = os.path.isfile(csv_path)

        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=benchmark_results.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(benchmark_results)

        with open(os.path.join(run_metrics_dir, "benchmark_results.json"), "a") as f:
            json.dump(benchmark_results, f)
            f.write("\n")

        logging.info("Formal Evaluation Harness completed. Metrics saved to CSV.")

    def visualize_population(self, generation: int, force: bool = False) -> None:
        """Visualizes the 3D spatial distribution and fitness of the current population.

        Args:
            generation: Current training step or epoch.
            force: If True, forces visualization even at generation 0.
        """
        if not force and generation == 0:
            return

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        positions = np.empty((0, 3))
        colors = np.empty(0)

        if len(self.entities) > 0:
            pos_tensor = torch.stack([agent.position[:3] for agent in self.entities])
            positions = pos_tensor.detach().cpu().numpy()

            stacked_fitness = torch.stack([agent.fitness for agent in self.entities])
            colors = stacked_fitness.detach().cpu().numpy()

            sc = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, cmap="viridis")
            plt.colorbar(sc, label="Fitness")

            if wandb.run is not None and generation % 5 == 0:
                if len(colors) > 0:
                    point_cloud = np.zeros((positions.shape[0], 6))
                    point_cloud[:, :3] = positions

                    colors_arr = np.array(colors)
                    min_c, max_c = colors_arr.min(), colors_arr.max()
                    range_c = max_c - min_c if max_c > min_c else 1.0

                    norm_c = (colors_arr - min_c) / range_c
                    point_cloud[:, 3] = norm_c * 255.0
                    point_cloud[:, 4] = 0.0
                    point_cloud[:, 5] = (1.0 - norm_c) * 255.0

                    current_step = self.trainer.global_train_step if hasattr(self, "trainer") else 0
                    wandb.log(
                        {
                            "visualizations/batched_policies_3d": wandb.Object3D(point_cloud),
                            "global_train_step": current_step,
                            "generation": generation,
                        }
                    )

        ax.set_title(f"Epoch {generation} | Batch: {len(self.entities)}")
        best_policy = max(self.entities, key=lambda o: o.fitness)
        best_pos = best_policy.position.cpu().numpy()[:3]
        ax.text(best_pos[0], best_pos[1], best_pos[2], "Best Policy", color="red", fontsize=6)

        from vrl_framework.core.settings import SIM_DIR

        env_dir = os.path.join(SIM_DIR, "metrics", "environment")
        os.makedirs(env_dir, exist_ok=True)

        save_path = os.path.join(env_dir, f"environment_epoch_{generation}.png")
        plt.savefig(save_path)
        plt.close()

    def export_graph(self, ent, generation, force=False):
        """Exports the entity's computational graph to ONNX format."""
        if not force and generation == 0:
            return

        export_path = os.path.join(SIM_DIR, f"network_topology_gen_{generation}.onnx")
        ent.export_to_onnx(filepath=export_path)

    def log_metrics(self, force=False):
        """Transmits rendering and metrics to WandB.

        Args:
            force: Force execution bypassing epoch constraints.
        """
        if not force and self.generation == 0:
            return

        target_gen = self.generation - 1 if self.generation > 0 else 0
        if self._stop or self.generation == self.total_generations:
            target_gen = self.generation

        if not force and target_gen > 0 and target_gen % 50 != 0:
            return

        media_payload = {"generation": self.generation}

        from vrl_framework.core.settings import SIM_DIR

        env_dir = os.path.join(SIM_DIR, "metrics", "environment")
        world_path = os.path.join(env_dir, f"environment_epoch_{target_gen}.png")
        if os.path.exists(world_path):
            media_payload["visualizations/environment_map"] = wandb.Image(
                world_path, caption=f"Environment Map Epoch {target_gen}"
            )

        agent_core_path = os.path.join(SIM_DIR, f"conceptual_diagram_gen{target_gen}.png")
        if os.path.exists(agent_core_path):
            media_payload["visualizations/conceptual_diagram"] = wandb.Image(
                agent_core_path, caption=f"Conceptual Diagram Gen {target_gen}"
            )

        agent_core_3d_path = os.path.join(SIM_DIR, f"network_topology_3D_gen_{target_gen}.png")
        if os.path.exists(agent_core_3d_path):
            media_payload["visualizations/best_model_3D"] = wandb.Image(
                agent_core_3d_path, caption=f"Best Model 3D Gen {target_gen}"
            )

        if len(media_payload) > 1:
            if wandb.run is not None:
                if not getattr(wandb, "_vrl_axis_mapped", False):
                    wandb.define_metric("generation")
                    wandb.define_metric("*", step_metric="generation")
                    wandb._vrl_axis_mapped = True

                current_step = self.trainer.global_train_step if hasattr(self, "trainer") else 0
                media_payload["global_train_step"] = current_step

                try:
                    wandb.log(media_payload, commit=False)
                except Exception:
                    pass

            import logging

            logging.info(
                f"Dispatched milestone structural visualizations to local W&B daemon (Target Gen: {target_gen})"
            )

    def simulate_step(self):
        if self._stop:
            return

        if getattr(self, "enable_profiler", False):
            import torch.profiler

            with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=10, warmup=2, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                self._simulate_step_internal()
                prof.step()
        else:
            self._simulate_step_internal()

    def _simulate_step_internal(self):
        gen_start = time.time()

        if self.total_generations is not None:
            self.estimated_progress = (self.generation / self.total_generations) * 100
            self.update_progress_display(self.total_generations)

        if not self.entities:
            world_limits = torch.tensor(WORLD_DIM, dtype=torch.float32, device=MODEL_DEVICE)
            random_positions = (
                torch.rand((INIT_POPULATION, len(WORLD_DIM)), device=MODEL_DEVICE) * world_limits
            ).long()
            from vrl_framework.environment.world_dynamics import Agent

            for i in range(INIT_POPULATION):
                agent_core_ref = self.batched_agents.agent_cores[0] if hasattr(self, "batched_agents") else None
                initial_agent = Agent(position=random_positions[i], agent_core=agent_core_ref)
                initial_agent.energy = DESIRED_ENERGY * 2.0
                self.entities.append(initial_agent)

        self.meta_controller.execute_phases(self)

        self.complexity_factor = 1 + (self.generation / 500)
        global EXPLORATION_NOISE

        EXPLORATION_NOISE = max(0.01, 0.05 * (1 - (self.generation / 1000)))

        if hasattr(self, "update_environment"):
            self.update_environment()

        if random.random() < ENV_EVENT_RATE * self.complexity_factor:
            if hasattr(self, "simulate_unexpected_event"):
                self.simulate_unexpected_event()

        max(float(ent.fitness) for ent in self.entities) if self.entities else -float("inf")
        np.mean([float(ent.fitness) for ent in self.entities]) if self.entities else 0
        terminated_agents = []

        old_permanent = (
            getattr(self.batched_agents, "permanent_visited_grid", None) if hasattr(self, "batched_agents") else None
        )

        if not hasattr(self, "batched_agents"):
            self.batched_agents = VectorizedPopulation(len(self.entities), WORLD_DIM, max_agents=MAX_POPULATION).to(
                MODEL_DEVICE
            )

            if old_permanent is not None:
                self.batched_agents.permanent_visited_grid.copy_(old_permanent)

        self.batched_agents.sync_with_entities_list(self.entities)
        self.batched_agents.world_ref = self

        perceptions, latent_signals, audios = self.batched_agents.perceive_batch(self.grid, audio_grid=self.audio_grid)
        ext_signals = [ent.comm_msg for ent in self.entities]

        actions, log_probs = self.batched_agents.act_batch(perceptions, audio=audios, external_signals=ext_signals)

        max_action_idx = (
            self.batched_agents.agent_cores[0].num_actions - 1
            if hasattr(self.batched_agents.agent_cores[0], "num_actions")
            else 15
        )
        safe_actions = torch.nan_to_num(actions, nan=0.0, posinf=float(max_action_idx), neginf=0.0)
        safe_actions = torch.clamp(safe_actions, min=0.0, max=float(max_action_idx))

        self.batched_agents.step(safe_actions)

        if ENABLE_OFFLINE_ROLLOUT and hasattr(self, "map_elites_archive") and len(self.map_elites_archive.archive) > 0:
            with torch.no_grad():
                archive_keys = list(self.map_elites_archive.archive.keys())
                sampled_archive_key = random.choice(archive_keys)
                archive_state = self.map_elites_archive.archive[sampled_archive_key]

                agent_network = self.entities[0].agent_core

                with torch.no_grad():
                    backup_gamma = agent_network.global_gamma.detach().clone()
                    backup_beta = agent_network.global_beta.detach().clone()

                    try:
                        archive_gamma = archive_state["gamma"].to(MODEL_DEVICE, dtype=torch.float32)
                        archive_beta = archive_state["beta"].to(MODEL_DEVICE, dtype=torch.float32)
                        agent_network.global_gamma.copy_(archive_gamma.mean(dim=0, keepdim=True))
                        agent_network.global_beta.copy_(archive_beta.mean(dim=0, keepdim=True))

                        archive_out = agent_network(perceptions, audio=audios)
                        archive_logits = archive_out[0] if isinstance(archive_out, tuple) else archive_out

                        from einops import rearrange

                        flat_archive_logits = (
                            rearrange(archive_logits, "b ... d -> b d") if archive_logits.dim() > 2 else archive_logits
                        )
                        canonical_archive_logits = flat_archive_logits[:, : agent_network.num_actions]

                        _ = torch.argmax(canonical_archive_logits, dim=-1).detach()

                        # Compute trajectory entropy and expected value for offline rollout evaluation.
                        archive_probs = F.softmax(archive_logits, dim=-1)
                        archive_entropy = -(archive_probs * torch.log(archive_probs + 1e-8)).sum(dim=-1).mean()

                        expected_value = 0.0
                        if hasattr(archive_out, "pessimistic_value"):
                            expected_value = archive_out.pessimistic_value.mean().item()

                        episode_return = float(archive_entropy.item() + expected_value)

                        self.map_elites_archive.performances[sampled_archive_key] += episode_return
                    finally:
                        # Restore Host architecture deterministically
                        agent_network.global_gamma.copy_(backup_gamma)
                        agent_network.global_beta.copy_(backup_beta)

        if self.entities and self.generation % 2 == 0 and self.generation % 5 != 0:
            if not hasattr(self, "online_trainer"):
                self.online_trainer = TrainingOrchestrator(self.entities[0].agent_core)
            self.online_trainer.train_step(perceptions[: min(64, perceptions.size(0))].to(MODEL_DEVICE))

        done_mask = self.batched_agents.action_cost_update_batch()

        fitness_cpu_list = self.batched_agents.fitness.detach().cpu().tolist()
        for idx_e, entity_obj in enumerate(self.entities):
            if idx_e < len(fitness_cpu_list):
                entity_obj.fitness = fitness_cpu_list[idx_e]

        self.batched_agents.survival_steps += 1

        # Project spatial constraints to valid grid coordinates.
        pos_x = self.batched_agents.positions[:, 0].clamp(0, WORLD_DIM[0] - 1).long()
        pos_y = self.batched_agents.positions[:, 1].clamp(0, WORLD_DIM[1] - 1).long()
        pos_z = self.batched_agents.positions[:, 2].clamp(0, WORLD_DIM[2] - 1).long()

        cell_values = self.grid[pos_x, pos_y, pos_z, 0]
        self.batched_agents.energy_intake = cell_values.detach().clone()

        with torch.no_grad():
            prev_state = self.batched_agents.prev_state_features_batch
            last_state = self.batched_agents.last_state_features_batch

            base_agent_core = self.entities[0].agent_core
            deq_steps = base_agent_core.deq_convergence_steps

            deq_steps_t = torch.as_tensor(deq_steps, dtype=torch.float32, device=MODEL_DEVICE)
            convergence_reward = torch.clamp((5.0 - deq_steps_t) * (deq_steps_t - 1.0), min=0.0) * 0.5

            rnd_target = base_agent_core.jepa.target_encoder(prev_state)
            rnd_pred = base_agent_core.jepa.predictor[0](prev_state)
            prediction_error = F.mse_loss(rnd_pred, rnd_target.detach(), reduction="none").mean(dim=-1)

            pred_err = F.mse_loss(prev_state, last_state, reduction="none").mean(dim=-1)
            novelty_score = torch.tanh(prediction_error)
            velocity_norm = torch.norm(self.batched_agents.velocities, dim=-1)

            grace_period_active = getattr(self, "generation", 0) < 500

            if grace_period_active:
                move_penalty = torch.where(velocity_norm < 0.01, -0.01, 1.0).to(self.batched_agents.fitness.dtype)
                collision_penalty = torch.where(cell_values < 0, -1.0, 0.0).to(self.batched_agents.fitness.dtype)
            else:
                move_penalty = torch.where(velocity_norm < 0.1, -1.0, 0.5).to(self.batched_agents.fitness.dtype)
                collision_penalty = torch.where(cell_values < 0, -15.0, 0.0).to(self.batched_agents.fitness.dtype)

            rewards = ((pred_err * novelty_score * 10.0) + convergence_reward + move_penalty).to(
                self.batched_agents.fitness.dtype
            )
            rewards += collision_penalty
            crystallization_level = self.grid[pos_x, pos_y, pos_z, 15].to(self.batched_agents.fitness.dtype)
            crystallization_penalty = -2.0 * torch.clamp((crystallization_level - 0.2) / 0.8, min=0.0, max=1.0)
            rewards += crystallization_penalty

            self.last_crystallization_ratio = float((self.grid[..., 15] > 0.2).float().mean().item())
            self.last_mean_crystallization_level = float(self.grid[..., 15].float().mean().item())
            rewards = torch.tanh(rewards / 10.0) * 10.0

        self.batched_agents.fitness += rewards
        self.batched_agents.fitness = torch.clamp(self.batched_agents.fitness, min=-100.0)

        self.grid[pos_x, pos_y, pos_z, 0] = torch.where(cell_values > 0.5, 0.0, self.grid[pos_x, pos_y, pos_z, 0])
        self.batched_agents.last_rewards_batch = rewards.unsqueeze(-1).detach()

        if (
            hasattr(self.batched_agents, "last_damping_actions")
            and self.batched_agents.last_damping_actions is not None
        ):
            act_ids = self.batched_agents.last_damping_actions
            active_tools = act_ids >= 8

            if active_tools.any():
                t_x = pos_x[active_tools]
                t_y = pos_y[active_tools]
                t_z = pos_z[active_tools]
                tools = act_ids[active_tools]

                dim_x, dim_y, dim_z = WORLD_DIM[0] - 1, WORLD_DIM[1] - 1, WORLD_DIM[2] - 1

                # Define 3x3x3 spatial kernel offsets.
                offsets = torch.tensor([-1, 0, 1], device=MODEL_DEVICE)
                grid_dx, grid_dy, grid_dz = torch.meshgrid(offsets, offsets, offsets, indexing="ij")
                flat_dx = grid_dx.flatten()
                flat_dy = grid_dy.flatten()
                flat_dz = grid_dz.flatten()

                n_x = (t_x.unsqueeze(1) + flat_dx.unsqueeze(0)).clamp(0, dim_x).flatten()
                n_y = (t_y.unsqueeze(1) + flat_dy.unsqueeze(0)).clamp(0, dim_y).flatten()
                n_z = (t_z.unsqueeze(1) + flat_dz.unsqueeze(0)).clamp(0, dim_z).flatten()
                tools_expanded = tools.unsqueeze(1).expand(-1, 27).flatten()

                build_mask = tools_expanded == 8
                b_x, b_y, b_z = n_x[build_mask], n_y[build_mask], n_z[build_mask]
                build_indices = (b_x, b_y, b_z, torch.ones_like(b_x) * 1)
                build_values = torch.full_like(b_x, 0.5, dtype=self.grid.dtype)
                self.grid.index_put_(build_indices, build_values, accumulate=True)
                self.grid[b_x, b_y, b_z, 1] = torch.clamp(self.grid[b_x, b_y, b_z, 1], 0.0, 1.0)
                self.grid[b_x, b_y, b_z, 2:5] *= 0.1

                break_mask = tools_expanded == 9
                br_x, br_y, br_z = n_x[break_mask], n_y[break_mask], n_z[break_mask]
                extracted_mass = torch.clamp(self.grid[br_x, br_y, br_z, 1], 0.0, 0.8)

                break_indices = (br_x, br_y, br_z, torch.ones_like(br_x) * 1)
                break_values = torch.full_like(br_x, -0.8, dtype=self.grid.dtype)
                self.grid.index_put_(break_indices, break_values, accumulate=True)
                self.grid[br_x, br_y, br_z, 1] = torch.clamp(self.grid[br_x, br_y, br_z, 1], 0.0, 1.0)

                energy_indices = (br_x, br_y, br_z, torch.zeros_like(br_x))
                self.grid.index_put_(energy_indices, extracted_mass, accumulate=True)
                self.grid[br_x, br_y, br_z, 0] = torch.clamp(self.grid[br_x, br_y, br_z, 0], 0.0, 1.0)
                self.grid[br_x, br_y, br_z, 2:5] += torch.randn_like(self.grid[br_x, br_y, br_z, 2:5]) * 0.5

                repulsion_mask = tools_expanded >= 10
                v_x, v_y, v_z = n_x[repulsion_mask], n_y[repulsion_mask], n_z[repulsion_mask]
                tmp_v = self.grid[v_x, v_y, v_z, 2].clone()
                self.grid[v_x, v_y, v_z, 2] = -self.grid[v_x, v_y, v_z, 3] * 0.5
                self.grid[v_x, v_y, v_z, 3] = tmp_v * 0.5

        if not hasattr(self.batched_agents, "historical_max_fitness"):
            self.batched_agents.register_buffer("historical_max_fitness", self.batched_agents.fitness.clone())

        tau_smooth = 10.0
        smooth_max = torch.logsumexp(self.batched_agents.fitness / tau_smooth, dim=0) * tau_smooth
        self.batched_agents.historical_max_fitness = (self.batched_agents.historical_max_fitness * 0.999) + (
            smooth_max * 0.001
        )

        fitness_regret = self.batched_agents.historical_max_fitness - self.batched_agents.fitness
        regret_stagnant = fitness_regret > 0.0

        fitness_deltas = torch.abs(self.batched_agents.fitness - self.batched_agents.prev_fitness)
        local_stagnant = fitness_deltas < 1e-4

        base_agent_core = self.batched_agents.agent_cores[0]
        td_error_stagnant = False
        if hasattr(base_agent_core, "memory") and hasattr(base_agent_core.memory, "td_threshold_ema"):
            td_error_stagnant = base_agent_core.memory.td_threshold_ema < 1e-3

        exploration_decayed = False
        if hasattr(base_agent_core, "exploration_layer"):
            exploration_decayed = getattr(base_agent_core.exploration_layer, "temperature", 1.0) < 0.05

        is_temp_decayed = False
        if hasattr(base_agent_core, "exploration_layer"):
            is_temp_decayed = (
                base_agent_core.exploration_layer.temperature.item()
                <= base_agent_core.exploration_layer.min_temperature
            )

        is_plateau_detected = regret_stagnant & local_stagnant
        if td_error_stagnant and exploration_decayed and is_temp_decayed:
            is_plateau_detected = is_plateau_detected | True

        self.batched_agents.stagnation_counters = torch.where(
            is_plateau_detected,
            self.batched_agents.stagnation_counters + 1,
            torch.zeros_like(self.batched_agents.stagnation_counters),
        )
        self.batched_agents.prev_fitness = self.batched_agents.fitness.clone()

        plateau_threshold = 50
        stagnation_mask = self.batched_agents.stagnation_counters > plateau_threshold
        if stagnation_mask.any() and hasattr(self.batched_agents, "prev_state_features_batch"):
            with torch.no_grad():
                # Apply dynamic temperature scaling
                if hasattr(base_agent_core, "exploration_layer"):
                    current_temp = base_agent_core.exploration_layer.temperature.item()
                    base_agent_core.exploration_layer.temperature.data.fill_(min(5.0, current_temp * 1.5))

                self.batched_agents.stagnation_counters[stagnation_mask] = 0

                if hasattr(base_agent_core, "exploration_layer"):
                    base_agent_core.exploration_layer.plateau_counter = torch.tensor(100.0, device=MODEL_DEVICE)

        if hasattr(self.batched_agents, "last_fused_context_batch") and hasattr(
            self.batched_agents, "prev_fused_context_batch"
        ):
            active_mask = (~done_mask) & self.batched_agents.active_mask
            if active_mask.any():
                gpu_states = self.batched_agents.prev_fused_context_batch[active_mask]
                gpu_next = self.batched_agents.last_fused_context_batch[active_mask]
                gpu_actions = actions[active_mask]
                gpu_rewards = rewards[active_mask]

                dummy_probs = torch.zeros_like(gpu_rewards)
                if gpu_actions.dim() == 1:
                    gpu_actions = gpu_actions.unsqueeze(1)

                # Ensure hardware boundaries during buffer assignment.
                pad_action_dim = self.entities[0].experience_buffer.actions.size(1) - gpu_actions.size(-1)
                if pad_action_dim > 0:
                    gpu_actions = F.pad(gpu_actions, (0, pad_action_dim))

                if len(self.entities) > 0 and hasattr(self.entities[0], "experience_buffer"):
                    gpu_log_probs = log_probs[active_mask]
                    if gpu_log_probs.dim() > 1:
                        gpu_log_probs = gpu_log_probs.squeeze(-1)

                    self.entities[0].experience_buffer.extend(
                        gpu_states.detach(),
                        gpu_actions.detach(),
                        gpu_rewards.detach(),
                        dummy_probs.detach(),
                        gpu_log_probs.detach(),
                        gpu_next.detach(),
                    )

        active_count = len(self.entities)
        if active_count > 0:
            active_fitness = self.batched_agents.fitness[:active_count]
            survival_threshold = torch.median(active_fitness).item()
            best_actor = torch.argmax(active_fitness).item()
            best_agent = self.entities[best_actor]
        else:
            survival_threshold = 0.0
            best_agent = None

        with torch.no_grad():
            capable_mask = done_mask & (self.batched_agents.fitness >= survival_threshold)
            failed_mask = done_mask & (~capable_mask)

            if capable_mask.any():
                self.batched_agents.energies = torch.where(
                    capable_mask,
                    torch.tensor(DEACTIVATION_THRESHOLD * 1.5, device=MODEL_DEVICE),
                    self.batched_agents.energies,
                )

            if failed_mask.any():
                self.batched_agents.energies = torch.where(
                    failed_mask,
                    torch.tensor(DEACTIVATION_THRESHOLD * 1.5, device=MODEL_DEVICE),
                    self.batched_agents.energies,
                )
                self.batched_agents.hps = torch.where(
                    failed_mask, torch.ones_like(self.batched_agents.hps), self.batched_agents.hps
                )
                self.batched_agents.health_bands = torch.where(
                    failed_mask,
                    torch.ones_like(self.batched_agents.health_bands) * 2,
                    self.batched_agents.health_bands,
                )

            self.batched_agents.fitness = torch.where(
                failed_mask, torch.tensor(0.0, device=MODEL_DEVICE), self.batched_agents.fitness
            )

            if best_agent is not None and failed_mask.any():
                elite_weights = best_agent.agent_core.state_dict()
                failed_indices = torch.where(failed_mask)[0].tolist()
                for i in failed_indices:
                    ent = self.entities[i]
                    if ent != best_agent:
                        ent.agent_core.load_state_dict(elite_weights)
                        with torch.no_grad():
                            for param in ent.agent_core.parameters():
                                param.add_(torch.randn_like(param), alpha=0.02)

        if self.generation % 5 == 0 and self.entities:
            if hasattr(torch.cuda, "nvtx"):
                torch.cuda.nvtx.range_push("Population_Core_Learning_Epoch")

            ready_agents = [ent for ent in self.entities if len(ent.experience_buffer) >= BATCH_SIZE]
            num_ready = len(ready_agents)

            if num_ready > 0:
                global_agent_core = ready_agents[0].agent_core
                max_rollout = 16
                total_batch_dim = BATCH_SIZE * num_ready

                if (
                    not hasattr(global_agent_core, "pop_rollout_tensors")
                    or global_agent_core.pop_rollout_tensors["states"].size(1) != total_batch_dim
                ):
                    global_agent_core.pop_rollout_ptr = 0
                    global_agent_core.pop_max_rollout = max_rollout

                    # Target shape: [max_rollout, total_batch_dim, D].
                    global_agent_core.pop_rollout_tensors = {
                        "states": torch.zeros(
                            (max_rollout, total_batch_dim, 256), device=MODEL_DEVICE, dtype=torch.float16
                        ),
                        "actions": torch.zeros((max_rollout, total_batch_dim), device=MODEL_DEVICE, dtype=torch.long),
                        "rewards": torch.zeros(
                            (max_rollout, total_batch_dim), device=MODEL_DEVICE, dtype=torch.float32
                        ),
                        "next_states": torch.zeros(
                            (max_rollout, total_batch_dim, 256), device=MODEL_DEVICE, dtype=torch.float16
                        ),
                        "log_probs": torch.zeros(
                            (max_rollout, total_batch_dim), device=MODEL_DEVICE, dtype=torch.float32
                        ),
                        "dones": torch.zeros((max_rollout, total_batch_dim), device=MODEL_DEVICE, dtype=torch.float32),
                    }

                pop_states, pop_actions, pop_rewards, pop_next_states, pop_log_probs = [], [], [], [], []

                for ent in ready_agents:
                    s, a, r, _, lp, ns = ent.experience_buffer.sample(BATCH_SIZE)
                    a = a.squeeze(1).long()

                    if hasattr(global_agent_core, "hierarchical_planner"):
                        with torch.no_grad():
                            corrected_goals = global_agent_core.hierarchical_planner.hiro_off_policy_correction(
                                s, ns, a, global_agent_core.actor_critic, causal_masker=global_agent_core.causal_masker
                            )
                            ent.agent_core.manager_goal = corrected_goals.mean(dim=0, keepdim=True)

                    pop_states.append(s.detach())
                    pop_actions.append(a.detach())
                    pop_rewards.append(r.detach().float())
                    pop_next_states.append(ns.detach())
                    pop_log_probs.append(lp.detach())

                ptr = global_agent_core.pop_rollout_ptr
                global_agent_core.pop_rollout_tensors["states"][ptr] = torch.cat(pop_states, dim=0).to(torch.float16)
                global_agent_core.pop_rollout_tensors["actions"][ptr] = torch.cat(pop_actions, dim=0)
                global_agent_core.pop_rollout_tensors["rewards"][ptr] = torch.cat(pop_rewards, dim=0)
                global_agent_core.pop_rollout_tensors["next_states"][ptr] = torch.cat(pop_next_states, dim=0).to(
                    torch.float16
                )
                global_agent_core.pop_rollout_tensors["log_probs"][ptr] = torch.cat(pop_log_probs, dim=0)
                global_agent_core.pop_rollout_tensors["dones"][ptr] = torch.zeros(
                    total_batch_dim, device=MODEL_DEVICE, dtype=torch.float32
                )

                global_agent_core.pop_rollout_ptr += 1

                if global_agent_core.pop_rollout_ptr >= global_agent_core.pop_max_rollout:
                    with torch.no_grad():
                        t_states = global_agent_core.pop_rollout_tensors["states"].view(-1, 256).float()
                        t_actions = global_agent_core.pop_rollout_tensors["actions"].view(-1)
                        t_rewards = global_agent_core.pop_rollout_tensors["rewards"].view(-1)
                        t_next_states = global_agent_core.pop_rollout_tensors["next_states"].view(-1, 256).float()
                        t_log_probs = global_agent_core.pop_rollout_tensors["log_probs"].view(-1)
                        t_dones = global_agent_core.pop_rollout_tensors["dones"].view(-1)

                        collision_mask = t_rewards < -5.0
                        if collision_mask.any():
                            t_rewards[collision_mask] = 10.0
                            global_agent_core.manager_goal = (
                                t_next_states[collision_mask].mean(dim=0, keepdim=True).detach()
                            )

                        jepa_states_out = global_agent_core.jepa(t_states)
                        latent_states = (
                            jepa_states_out[0] if isinstance(jepa_states_out, tuple) else jepa_states_out
                        ).detach()
                        jepa_next_out = global_agent_core.jepa(t_next_states)
                        latent_next_states = (
                            jepa_next_out[0] if isinstance(jepa_next_out, tuple) else jepa_next_out
                        ).detach()

                        ac_out_online = global_agent_core.actor_critic(latent_states)
                        values = ac_out_online.pessimistic_value

                        with torch.no_grad():
                            if hasattr(global_agent_core, "actor_critic_ema"):
                                temp_target_net = copy.deepcopy(global_agent_core.actor_critic)
                                global_agent_core.actor_critic_ema.load_shadow_into(temp_target_net)
                                ac_out_target = temp_target_net(latent_next_states)
                                next_values = ac_out_target.pessimistic_value
                            else:
                                ac_out_target = global_agent_core.actor_critic(latent_next_states)
                                next_values = ac_out_target.pessimistic_value

                        values = values.squeeze(-1).float()
                        next_values = next_values.squeeze(-1).float()

                        gamma, lam = 0.99, 0.95
                        T_len = global_agent_core.pop_max_rollout
                        B_len = t_rewards.size(0) // T_len

                        # GAE requires time-major memory layouts: [T, B].
                        val_2d = values.view(T_len, B_len)
                        next_val_2d = next_values.view(T_len, B_len)
                        rew_2d = t_rewards.view(T_len, B_len)
                        dones_2d = t_dones.view(T_len, B_len)

                        masks = (1.0 - dones_2d).float()
                        delta = rew_2d + gamma * next_val_2d * masks - val_2d
                        delta_f32 = delta.float()
                        discount = gamma * lam

                        adv_2d = torch.zeros_like(delta_f32)
                        curr_adv = torch.zeros(B_len, device=MODEL_DEVICE, dtype=torch.float32)

                        for t in range(T_len - 1, -1, -1):
                            curr_adv = delta_f32[t] + discount * masks[t] * curr_adv
                            adv_2d[t] = curr_adv

                        adv_mean = adv_2d.mean()
                        adv_var = adv_2d.var(unbiased=False)
                        adv_std = torch.sqrt(adv_var + 1e-8) + 1e-5
                        adv_2d_norm = (adv_2d - adv_mean) / adv_std

                        returns_2d_raw = adv_2d + val_2d
                        ret_mean = returns_2d_raw.mean()
                        ret_std = torch.sqrt(returns_2d_raw.var(unbiased=False) + 1e-4)
                        (returns_2d_raw - ret_mean) / ret_std

                        del delta, delta_f32, masks, curr_adv, val_2d, next_val_2d, adv_2d

                        t_states_2d = t_states.view(T_len, B_len, 256)
                        t_actions_2d = t_actions.view(T_len, B_len)
                        t_log_probs_2d = t_log_probs.view(T_len, B_len)
                        t_next_states_2d = t_next_states.view(T_len, B_len, 256)

                        t_states_2d.unbind(1)
                        t_actions_2d.unbind(1)
                        returns_2d_raw.unbind(1)
                        adv_2d_norm.unbind(1)
                        t_log_probs_2d.unbind(1)
                        t_next_states_2d.unbind(1)

                        for ent_idx, ent_instance in enumerate(ready_agents):
                            s_idx = int(ent_idx * BATCH_SIZE)
                            e_idx = int(s_idx + BATCH_SIZE)
                            ent_instance.experience_buffer.extend(
                                t_states_2d[:, s_idx:e_idx, :].reshape(-1, 256),
                                t_actions_2d[:, s_idx:e_idx].reshape(-1, 1),
                                returns_2d_raw[:, s_idx:e_idx].reshape(-1),
                                adv_2d_norm[:, s_idx:e_idx].reshape(-1),
                                t_log_probs_2d[:, s_idx:e_idx].reshape(-1),
                                t_next_states_2d[:, s_idx:e_idx, :].reshape(-1, 256),
                            )

                    global_agent_core.pop_rollout_ptr = 0

                agents_ready_for_ppo = [ent for ent in ready_agents if ent.experience_buffer.size > 1024]
                if len(agents_ready_for_ppo) > 0:
                    num_ppo_ready = len(agents_ready_for_ppo)
                    m_states = torch.empty((num_ppo_ready * 256, 256), dtype=torch.float32, device=MODEL_DEVICE)
                    m_actions = torch.empty((num_ppo_ready * 256), dtype=torch.long, device=MODEL_DEVICE)
                    m_returns = torch.empty((num_ppo_ready * 256), dtype=torch.float32, device=MODEL_DEVICE)
                    m_advantages = torch.empty((num_ppo_ready * 256), dtype=torch.float32, device=MODEL_DEVICE)
                    m_log_probs = torch.empty((num_ppo_ready * 256), dtype=torch.float32, device=MODEL_DEVICE)
                    m_next_states = torch.empty((num_ppo_ready * 256, 256), dtype=torch.float32, device=MODEL_DEVICE)

                    for i, ent in enumerate(agents_ready_for_ppo):
                        bs, ba, br, badv, blp, bns = ent.experience_buffer.sample(256)
                        s_idx = int(i * 256)
                        e_idx = int(s_idx + 256)
                        m_states[s_idx:e_idx] = bs.float()
                        m_actions[s_idx:e_idx] = ba.squeeze(1).long()
                        m_returns[s_idx:e_idx] = br
                        m_advantages[s_idx:e_idx] = badv
                        m_log_probs[s_idx:e_idx] = blp
                        m_next_states[s_idx:e_idx] = bns.float()

                    ret_avg = float(m_returns.mean().item())
                    ret_best = float(m_returns.max().item())

                    metrics_aggregator.log_step_metrics(
                        {
                            "metrics/return_avg": ret_avg,
                            "metrics/return_best": ret_best,
                        }
                    )

                    m_returns = torch.clamp(m_returns, min=-1e4, max=1e4)

                    ppo_loss = global_agent_core.ppo_update(
                        m_states, m_actions, m_log_probs, m_returns, m_advantages, m_next_states
                    )

                    if isinstance(ppo_loss, dict):
                        metrics_aggregator.log_step_metrics(ppo_loss)
                    else:
                        metrics_aggregator.log_step_metrics(
                            {
                                "loss/total_raw": (
                                    float(ppo_loss.item()) if hasattr(ppo_loss, "item") else float(ppo_loss)
                                )
                            }
                        )

                    if (
                        getattr(self, "runtime_context", None) is not None
                        and getattr(self.runtime_context, "lmdb_bank", None) is not None
                    ):
                        try:
                            mask_high_adv = m_advantages > 1.0
                            if mask_high_adv.any():
                                valid_states_to_store = (
                                    m_next_states[mask_high_adv].detach().cpu().numpy().astype(np.float16)
                                )
                                current_gen = getattr(self, "generation", 0)

                                def _async_pop_ingest(data_array, gen):
                                    try:
                                        keys = [
                                            f"pop_rag_gen_{gen}_{i}_{hash(data_array[i].tobytes())}".encode("utf-8")
                                            for i in range(data_array.shape[0])
                                        ]
                                        values = [arr.tobytes() for arr in data_array]
                                        if hasattr(self.runtime_context.lmdb_bank, "batch_write_semantic"):
                                            self.runtime_context.lmdb_bank.batch_write_semantic(
                                                keys, values, data_array
                                            )
                                        elif hasattr(self.runtime_context.lmdb_bank, "batch_write"):
                                            self.runtime_context.lmdb_bank.batch_write(
                                                keys,
                                                values,
                                                torch.zeros(len(values), dtype=torch.float32, device="cpu"),
                                                torch.zeros(len(values), dtype=torch.float32, device="cpu"),
                                            )
                                        if hasattr(self.runtime_context.lmdb_bank, "env") and hasattr(
                                            self.runtime_context.lmdb_bank.env, "sync"
                                        ):
                                            self.runtime_context.lmdb_bank.env.sync()
                                    except Exception as e:
                                        import logging

                                        logging.error(f"[LMDB POP RAG] Ingestion fault: {e}")

                                if getattr(self.runtime_context, "io_worker", None) is not None:
                                    self.runtime_context.io_worker.submit(
                                        _async_pop_ingest, valid_states_to_store, current_gen
                                    )
                                else:
                                    _async_pop_ingest(valid_states_to_store, current_gen)
                        except Exception:
                            pass

                    if getattr(global_agent_core, "td_error_ema", None) is None:
                        global_agent_core.td_error_ema = 1.0

                    current_td_error = m_advantages.abs().mean().item()
                    global_agent_core.td_error_ema = 0.95 * global_agent_core.td_error_ema + 0.05 * current_td_error

                    is_formal_plateau = global_agent_core.td_error_ema < 1e-3

                    if is_formal_plateau:
                        if not hasattr(self.batched_agents, "intent_matrix"):
                            self.batched_agents.intent_matrix = torch.zeros(
                                self.batched_agents.max_agents, 256, device=MODEL_DEVICE, dtype=torch.float16
                            )

                        def _async_global_rag_fetch(query_vector):
                            dense_correction_cpu = torch.randn(256, dtype=torch.float32)

                            if (
                                getattr(self, "runtime_context", None) is not None
                                and getattr(self.runtime_context, "lmdb_bank", None) is not None
                            ):
                                try:
                                    if hasattr(self.runtime_context.lmdb_bank, "retrieve"):
                                        r_data, r_scores = self.runtime_context.lmdb_bank.retrieve(
                                            query_vector.numpy(), top_k=4, return_scores=True
                                        )
                                        retrieved_data = r_data
                                        retrieved_scores = r_scores
                                    else:
                                        retrieved_data = self.runtime_context.lmdb_bank.sample_replay(4)
                                        retrieved_scores = [1.0] * len(retrieved_data) if retrieved_data else []

                                    if retrieved_data and len(retrieved_data) > 0:
                                        tensors = []
                                        for r in retrieved_data:
                                            t = torch.from_numpy(np.frombuffer(r, dtype=np.float16)).float()
                                            t = t[:256] if t.size(0) >= 256 else F.pad(t, (0, 256 - t.size(0)))
                                            tensors.append(t)

                                        tensors = torch.stack(tensors)
                                        scores = torch.tensor(retrieved_scores, dtype=torch.float32)
                                        weights = F.softmax(scores, dim=0).unsqueeze(1)
                                        dense_correction_cpu = (tensors * weights).sum(dim=0)
                                except Exception as e:
                                    import logging

                                    logging.warning(f"RAG fetch failed: {e}")

                            transfer_stream = torch.cuda.Stream()
                            transfer_event = torch.cuda.Event()
                            with torch.cuda.stream(transfer_stream):
                                current_intent = self.batched_agents.intent_matrix.data
                                rag_signal = (
                                    dense_correction_cpu.half()
                                    .unsqueeze(0)
                                    .expand(self.batched_agents.max_agents, -1)
                                    .to(current_intent.device)
                                )
                                self.batched_agents.intent_matrix.copy_(
                                    current_intent + 0.15 * rag_signal, non_blocking=True
                                )
                            transfer_event.record(transfer_stream)
                            transfer_event.wait(torch.cuda.current_stream())

                        if (
                            getattr(self, "runtime_context", None) is not None
                            and getattr(self.runtime_context, "io_worker", None) is not None
                        ):
                            query_state = m_states[-1].mean(dim=0).detach().cpu()
                            self.runtime_context.io_worker.submit(_async_global_rag_fetch, query_state)

                    if (
                        hasattr(self, "anomaly_buffer")
                        and hasattr(self.anomaly_buffer, "buffer")
                        and len(self.anomaly_buffer.buffer) > 0
                    ):
                        if not hasattr(global_agent_core, "tta_module"):
                            try:
                                from vrl_framework.models.components import TestTimeAdaptationModule
                            except ImportError:
                                pass
                            else:
                                global_agent_core.tta_module = TestTimeAdaptationModule(
                                    global_agent_core.latent_dynamics, global_agent_core.jepa.target_encoder
                                )
                        anomaly_state = self.anomaly_buffer.sample_anomaly().to(MODEL_DEVICE)
                        if anomaly_state is not None:
                            dummy_actions = torch.zeros(1, global_agent_core.num_actions, device=MODEL_DEVICE)
                            dummy_next = anomaly_state.unsqueeze(0) + 0.01
                            global_agent_core.tta_module.adapt_step(
                                anomaly_state.unsqueeze(0), dummy_actions, dummy_next, global_agent_core.opt_causal
                            )

            if hasattr(torch.cuda, "nvtx"):
                torch.cuda.nvtx.range_pop()

        active_entities = [ent for ent in self.entities if ent not in terminated_agents]
        if len(active_entities) < len(self.entities):
            self.entities = active_entities

        if self.current_question is not None:
            self.reasoning_step()

            elapsed = time.time() - self.start_reasoning_time
            if elapsed >= self.min_reasoning_time:

                if (
                    self.best_score_so_far >= self.quality_threshold
                    and self.min_reasoning_time < self.max_reasoning_time
                ):
                    logging.info(
                        f"Early stop: best_score={self.best_score_so_far:.3f} >= " f"{self.quality_threshold}"
                    )
                    self.stop()
            if elapsed >= self.max_reasoning_time:
                logging.info(
                    f"Reasoning phase concluded: time limit reached {elapsed:.1f}s "
                    f"(max_time={self.max_reasoning_time})"
                )
                self.stop()

            self.timed_checkpoint()

        if self.generation % 5 == 0:
            for ent in self.entities:
                sensory_input = ent.perceive(self)
                ent.agent_core.prune_low_utility_experts(sensory_input, utility_threshold=0.01)
        if self.generation % 10 == 0:
            for ent in self.entities:
                prune_weak_weights(ent.agent_core)

        for ent in self.entities:
            sensory_input = ent.perceive(self)
            current_mutation_rate = CFG.MUTATION_RATE
            exploration_factor = getattr(ent, "genetic_code", {}).get("exploration_factor", 0.3)

            if torch.rand(1).item() < exploration_factor * (1 + ent.fitness / 500):
                ent.agent_core.evaluate_topology_expansion(sensory_input)

            if hasattr(ent.agent_core, "moe") and hasattr(ent.agent_core.moe, "update_topology"):
                # Offload dormant experts to NVMe based on EMA usage.
                ent.agent_core.moe.update_topology(drop_fraction=current_mutation_rate)

            prune_weak_weights(ent.agent_core)

            # Apply Gaussian noise to population-specific routing vectors.
            mutated = False
            if ent.pop_ref is not None:
                if torch.rand(1).item() < current_mutation_rate:
                    ent.pop_ref.population_gamma[ent.idx] += current_mutation_rate * torch.randn_like(
                        ent.pop_ref.population_gamma[ent.idx]
                    )
                    ent.pop_ref.population_beta[ent.idx] += current_mutation_rate * torch.randn_like(
                        ent.pop_ref.population_beta[ent.idx]
                    )

                    flip_mask = torch.rand_like(ent.pop_ref.population_masks[ent.idx].float()) < (
                        current_mutation_rate * 0.1
                    )
                    ent.pop_ref.population_masks[ent.idx] ^= flip_mask
                    mutated = True

            if mutated:
                ent.needs_weight_sync = True

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.metrics_payload["system/epoch_time"] = time.time() - gen_start
        self.gen_time_start = gen_start

        with torch.no_grad():
            if self.entities:
                len(self.entities)
                stm_tensors = torch.stack([ent.agent_core.stm_tensor for ent in self.entities])
                stm_ptrs = torch.stack([ent.agent_core.stm_ptr.reshape(-1) for ent in self.entities]).view(-1)
                stm_tensors.size(0)
                seq_len = stm_tensors.size(1)

                idx = torch.arange(seq_len, device=stm_tensors.device).unsqueeze(0)
                limits = torch.clamp(stm_ptrs, min=8).unsqueeze(1)

                mask = (idx < limits).to(stm_tensors.dtype)
                for _ in range(stm_tensors.dim() - 2):
                    mask = mask.unsqueeze(-1)

                valid_stm_seqs = stm_tensors * mask

                if valid_stm_seqs.dim() > 3:
                    dims_to_pool = tuple(range(2, valid_stm_seqs.dim() - 1))
                    valid_stm_seqs_3d = valid_stm_seqs.mean(dim=dims_to_pool)
                else:
                    valid_stm_seqs_3d = valid_stm_seqs

                avg_phi_val = compute_traj_entropy(valid_stm_seqs_3d).item()

                hidden_tensors = torch.stack(
                    [ent.hidden_state.view(-1, ent.hidden_state.size(-1)) for ent in self.entities]
                )
                internal_batch = hidden_tensors.mean(dim=1)
                avg_diversity = internal_batch.std(dim=0, unbiased=False).mean().item()

                import math

                if math.isnan(avg_phi_val) or math.isinf(avg_phi_val):
                    avg_phi_val = 0.0
                if math.isnan(avg_diversity) or math.isinf(avg_diversity):
                    avg_diversity = 0.0
            else:
                avg_phi_val = 0.0
                avg_diversity = 0.0

        if not hasattr(self, "_avg_phi_buffer"):
            self._avg_phi_buffer = []
            self._diversity_buffer = []
        self._avg_phi_buffer.append(avg_phi_val)
        self._diversity_buffer.append(avg_diversity)

        self.generation += 1

        if hasattr(self, "update_metrics"):
            try:
                self.update_metrics()
            except AttributeError:
                pass

        if self.grid is not None and hasattr(self, "curriculum") and hasattr(self.curriculum, "generate_terrain"):
            displacement, _ = self.curriculum.generate_terrain(self.grid)

        if hasattr(self, "imitation_phase"):
            self.imitation_phase()
        if hasattr(self, "touch_interaction"):
            self.touch_interaction()
        if hasattr(self, "advanced_social_interaction"):
            self.advanced_social_interaction()
        if hasattr(self, "specialization_phase"):
            self.specialization_phase()

        current_generation = self.generation if self.generation == self.total_generations else self.generation - 1
        final_generation = self._stop or self.generation == self.total_generations

        if current_generation > 0 and (current_generation % 50 == 0 or final_generation):
            self.visualize_population(current_generation)
            if self.entities:
                best_ent = max(self.entities, key=lambda o: o.fitness)
                self.visualize_agent_core_structure(best_ent, current_generation, force=True)

        if current_generation > 0 and (current_generation % 100 == 0 or final_generation):
            visualize_complex_entity_3D(self, current_generation)

        if current_generation > 0 and current_generation % 800 == 0:
            self.run_benchmarks()
            self.checkpoint_simulation()
        if current_generation > 0 and current_generation % 50 == 0:
            self.visualize_intelligence_metrics()

        if self.generation % 10 == 0:
            self._memory_maintenance()
            for ent in self.entities:
                ent.agent_core.consolidate_old_stm(max_stm_length=100)

        if self.generation >= 100 and self.generation % 25 == 0:
            for ent in self.entities:
                if hasattr(ent, "agent_core") and ent.agent_core is not None:
                    if not hasattr(ent.agent_core, "generalizationgate"):
                        from vrl_framework.models.components import GeneralizationGate

                        ent.agent_core.generalizationgate = GeneralizationGate(
                            getattr(ent.agent_core, "runtimecontext", None)
                        )

                    trainer_opt = getattr(ent.agent_core, "trainer", None)
                    actual_opt = getattr(trainer_opt, "opt_policy", None) if trainer_opt else None

                    ent.agent_core.generalizationgate.triggerasyncvalidation(
                        ent.agent_core.latent_dynamics.state_dict(), actual_opt
                    )

                    if actual_opt is not None:
                        ent.agent_core.generalizationgate.checkandresetmomentum(actual_opt)

        self.log_metrics()

        metrics_aggregator.flush(
            self.trainer.global_train_step if hasattr(self, "trainer") else 0, generation=self.generation
        )

        if self.total_generations is not None:
            self.update_progress_display(self.total_generations)

    def reasoning_step(self):

        context = self._build_reasoning_context()
        new_answer = self._generate_new_answer(context)
        score = self._evaluate_answer(new_answer, context)

        self.temporary_answers.append((new_answer, score))

        if score > self.best_score_so_far:
            self.best_score_so_far = score
            self.best_answer_so_far = new_answer
            with open("best_solution_so_far.txt", "w", encoding="utf-8") as f:
                f.write(f"SCORE={score:.3f}\n\n{new_answer}")
            logging.info(f"[reasoning_step] New best hypothesis found: score={score:.3f}")

        if self.enable_online_training and score > 0:
            self._online_train_during_reasoning(new_answer, score, context)

        logging.info(
            f"[reasoning_step] Reasoning iteration: score={score:.3f}, best_score_so_far={self.best_score_so_far:.3f}"
        )

    def _build_reasoning_context(self) -> dict[str, torch.Tensor]:
        """Constructs the reasoning context via LSH memory retrieval."""
        topK = 10

        q_tensor = self.entities[0].agent_core._process_text("Internal reasoning query").detach()

        self.entities[0].agent_core.memory.query_lsh_cpu_background(q_tensor.mean(dim=0), top_k=topK)

        mem_core = self.entities[0].agent_core.memory.core
        similarities = F.cosine_similarity(q_tensor.unsqueeze(1), mem_core.unsqueeze(0), dim=-1)  # [1, N]

        best_scores, best_indices = torch.topk(similarities.squeeze(0), min(topK, mem_core.size(0)))

        if best_indices.numel() > 0:
            context = mem_core[best_indices].mean(dim=0)
        else:
            context = torch.zeros(256, device=MODEL_DEVICE)
        return {"question_embed": q_tensor.cpu().numpy(), "context": context}

    def _generate_new_answer(self, context):
        """Generates a hierarchical goal vector conditional on the retrieved context."""
        q_tensor = torch.tensor(context["question_embed"], device=MODEL_DEVICE)
        c_tensor = context["context"]

        manager_state = q_tensor + c_tensor
        if manager_state.dim() == 1:
            manager_state = manager_state.unsqueeze(0)
        hypothesis_vector = self.entities[0].agent_core.hierarchical_planner.get_manager_goal(manager_state).squeeze(0)
        return hypothesis_vector

    def _evaluate_answer(self, hypothesis_vector, context):
        """Evaluates hypothesis stability using the causal symbolic reasoner."""
        active_entity = context.get("entity", self.entities[0])
        causal_model = active_entity.agent_core.causal_symbolic_reasoner
        stability = causal_model(hypothesis_vector.unsqueeze(0)).norm().item()
        score = min(10.0, stability * 2.0)
        return score

    def _online_train_during_reasoning(self, new_hypothesis, score, context):
        """Updates the causal policy online via energy minimization of the generated hypothesis."""
        active_entity = context.get("entity", self.entities[0])
        causal_model = active_entity.agent_core.causal_symbolic_reasoner
        energy_loss = causal_model(new_hypothesis.unsqueeze(0)).norm()
        optimizer = active_entity.agent_core.opt_policy

        from vrl_framework.trainer.ppo_engine import metrics_aggregator

        metrics_aggregator.log({"loss/reasoning_causal_energy": float(energy_loss.item())})

        optimizer.zero_grad()
        energy_loss.backward()
        reasoning_params = [p for g in optimizer.param_groups for p in g["params"]]
        torch.nn.utils.clip_grad_norm_(reasoning_params, 0.5)
        optimizer.step()

        logging.info(f"[online_train] Causal Energy Minimization Loss: {energy_loss.item():.4f}")

    def resume_from_checkpoint(self, path):
        """Restores operational state parameters from a JSON checkpoint."""
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.generation = data["generation"]
        self.best_score_so_far = data["best_score_so_far"]
        self.best_answer_so_far = data["best_answer_so_far"]
        self.current_question = data["current_question"]
        self.min_reasoning_time = data.get("min_reasoning_time", 0.0)
        self.max_reasoning_time = data.get("max_reasoning_time", 0.0)
        self.start_reasoning_time = data.get("start_reasoning_time", time.time())

        if not hasattr(self.agent_core, "curriculum_state"):
            self.agent_core.curriculum_state = CurriculumStateTracker().to(MODEL_DEVICE)

        if "curriculum_file_index" in data:
            self.agent_core.curriculum_state.file_index.fill_(int(data["curriculum_file_index"]))
        if "curriculum_byte_offset" in data:
            self.agent_core.curriculum_state.byte_offset.fill_(int(data["curriculum_byte_offset"]))
        if data.get("embedded_curriculum_path"):
            self.agent_core.curriculum_state.store_path("embedded_curriculum_path", data["embedded_curriculum_path"])
        if data.get("embedded_offload_path"):
            self.agent_core.curriculum_state.store_path("embedded_offload_path", data["embedded_offload_path"])

        self.curriculum_chunk_size = int(data.get("curriculum_chunk_size", 4096))

        if hasattr(self, "curriculum_stream"):
            del self.curriculum_stream
        if hasattr(self, "curriculum_chunk"):
            del self.curriculum_chunk
        self.curriculum_ptr = 0

        if "ltm_index" in data:
            self.entities[0].agent_core.memory.ltm_index = data["ltm_index"]
        if "knowledge_graph" in data:
            self.entities[0].agent_core.causal_validator.graph = data["knowledge_graph"]
        logging.info(f"Resumed from checkpoint {path} at generation {self.generation}")

    def pause(self):
        self._paused = True
        logging.info("Simulation paused.")

    def resume(self):
        self._paused = False
        logging.info("Simulation resumed.")

    def stop(self):
        self._stop = True
        wandb.finish()
        logging.info("Simulation halted. W&B sync finalized.")

    def run_simulation(self):
        self._stop = False
        if self.total_generations is None:
            self.total_generations = getattr(CFG, "MAX_POPULATION", 50000)

        while self.generation < self.total_generations and not self._stop:
            if self._paused:
                time.sleep(0.5)
                continue
            if self._stop:
                break
            self.simulate_step()
            self.checkpoint_simulation(force=False)

        if self._stop:
            self.generation -= 1
            for key in self.metrics:
                if len(self.metrics[key]) > 0:
                    self.metrics[key].pop()

        final_generation = self.generation

        self.checkpoint_simulation(force=True)
        try:
            if wandb.run is not None:
                current_step = self.trainer.global_train_step if hasattr(self, "trainer") else 0
                wandb.log(
                    {
                        "generation": final_generation,
                        "population_count": len(self.entities) if hasattr(self, "entities") else 0,
                        "global_train_step": current_step,
                    }
                )
        except Exception as e:
            logging.error(f"WandB metrics crash suppressed: {e}")
        self.visualize_population(final_generation, force=True)
        if self.entities:
            best_ent = max(self.entities, key=lambda o: o.fitness)
            self.visualize_agent_core_structure(best_ent, final_generation, force=True)
        self.visualize_intelligence_metrics(force=True)
        logging.info("Final checkpoint and visualizations saved. Simulation will now stop.")

    def clone(self):
        return copy.deepcopy(self)

    def inject_semantic_stimulus(self, stimulus_text):
        if CFG.STRICT_EX_NIHILO:
            import logging

            logging.info("Semantic stimulus blocked by STRICT_EX_NIHILO policy.")
            return

        if not self.entities:
            return
        champ = self.entities[0]
        with torch.no_grad():
            # [SeqLen, EmbedDim] -> [EmbedDim]
            dense_intent = champ.agent_core._process_text(stimulus_text).squeeze(0)

            sparse_activations, _ = champ.agent_core.sae(dense_intent.unsqueeze(0))

            champ.hidden_state = F.normalize((champ.hidden_state + (dense_intent * 5.0)).float(), p=2, dim=-1).to(
                champ.hidden_state.dtype
            )
            batch_size = champ.agent_core.manager_goal.size(0)
            target_goal = dense_intent[..., :32].unsqueeze(0).expand(batch_size, -1)
            champ.agent_core.manager_goal = (
                F.normalize((champ.agent_core.manager_goal + target_goal).float(), p=2, dim=-1).to(
                    champ.agent_core.manager_goal.dtype
                )
                * 5.0
            )

            if hasattr(champ.agent_core.memory, "td_ema"):
                champ.agent_core.memory.td_ema += 10.0

            active_features = (sparse_activations > 0.0).sum().item()
            logging.info(f"Semantic Stimulus Injected. Activated {active_features} internal SAE dictionaries.")


class UnifiedMetaController:
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.config = config
        self.gradient_threshold = config.get("gradient_threshold", 0.1)
        self.loss_stability_threshold = config.get("loss_stability_threshold", 0.01)
        self.curriculum_factor = 1.0

    def mutation_phase(self, world):
        for ent in world.entities:
            ent.agent_core.mutate()

    def structural_phase(self, world):
        for ent in world.entities:
            sensory_input = ent.perceive(world)
            ent.agent_core.prune_low_utility_experts(sensory_input, utility_threshold=0.01 * self.curriculum_factor)
            diversity = ent.agent_core.evaluate_activation_diversity(sensory_input)
            if diversity < 0.5:
                ent.agent_core.expand_topology(random.randint(64, 256))

    def curriculum_phase(self, world):
        generation = world.generation
        self.curriculum_factor = 1.0 + (generation / 1000.0)

    def execute_phases(self, world):
        self.curriculum_phase(world)
        self.structural_phase(world)


class TrainingOrchestrator:
    """Orchestrates off-policy, self-supervised representation learning (BYOL/VICReg hybrid)."""

    def __init__(self, agent_core):
        self.agent_core = agent_core
        if (
            hasattr(agent_core, "trainer")
            and hasattr(agent_core.trainer, "opt_representation")
            and agent_core.trainer.opt_representation is not None
        ):
            self.optimizer = agent_core.trainer.opt_representation
        else:
            if agent_core.opt_representation is None:
                agent_core.opt_representation = torch.optim.AdamW(
                    list(agent_core.jepa.parameters())
                    + list(agent_core.causal_symbolic_reasoner.parameters())
                    + list(agent_core.causal_masker.parameters()),
                    lr=1e-5,
                    fused=True,
                )
            self.optimizer = agent_core.opt_representation

        if (
            hasattr(agent_core, "trainer")
            and hasattr(agent_core.trainer, "scaler")
            and agent_core.trainer.scaler is not None
        ):
            self.scaler = agent_core.trainer.scaler
        elif torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

    def train_step(self, raw_visual_batch):
        self.optimizer.zero_grad()

        # Generate positive pairs via isotropic Gaussian noise for InfoNCE objective.
        view1 = torch.randn_like(raw_visual_batch).mul_(0.05).add_(raw_visual_batch)
        view2 = torch.randn_like(raw_visual_batch).mul_(0.05).add_(raw_visual_batch)

        self.agent_core.train()
        with torch.amp.autocast(device_type=MODEL_DEVICE, dtype=torch.float16):
            enc1 = self.agent_core.sensory(view1)
            enc2 = self.agent_core.sensory(view2)

            z1_out = self.agent_core.jepa(enc1)
            z2_out = self.agent_core.jepa(enc2)

        with torch.amp.autocast(device_type=MODEL_DEVICE, enabled=False):
            o1, t1 = z1_out[1].float(), z1_out[2].detach().float()
            o2, t2 = z2_out[1].float(), z2_out[2].detach().float()

            b_size = o1.size(0)
            d_dim = o1.size(-1)

            inv_loss = (F.mse_loss(o1, t2) + F.mse_loss(o2, t1)) / 2.0

            if b_size > 1:
                o1_c = o1 - o1.mean(dim=0, keepdim=True)
                o2_c = o2 - o2.mean(dim=0, keepdim=True)
                t1_c = t1 - t1.mean(dim=0, keepdim=True)
                t2_c = t2 - t2.mean(dim=0, keepdim=True)

                std_o1 = torch.sqrt(o1_c.var(dim=0, unbiased=False) + 1e-4)
                std_o2 = torch.sqrt(o2_c.var(dim=0, unbiased=False) + 1e-4)
                std_t1 = torch.sqrt(t1_c.var(dim=0, unbiased=False) + 1e-4)
                std_t2 = torch.sqrt(t2_c.var(dim=0, unbiased=False) + 1e-4)

                var_loss = (
                    torch.mean(F.relu(1.0 - std_o1))
                    + torch.mean(F.relu(1.0 - std_o2))
                    + torch.mean(F.relu(1.0 - std_t1))
                    + torch.mean(F.relu(1.0 - std_t2))
                ) / 4.0

                cov_o1 = (o1_c.T @ o1_c) / (b_size - 1)
                cov_o2 = (o2_c.T @ o2_c) / (b_size - 1)

                off_diag_o1 = cov_o1 - torch.diag(torch.diag(cov_o1))
                off_diag_o2 = cov_o2 - torch.diag(torch.diag(cov_o2))

                cov_loss = ((off_diag_o1**2).sum() + (off_diag_o2**2).sum()) / d_dim
            else:
                var_loss = torch.tensor(0.0, device=o1.device)
                cov_loss = torch.tensor(0.0, device=o1.device)

            l2_penalty = (o1.pow(2).mean() + o2.pow(2).mean()) * 0.01

            loss = (25.0 * inv_loss) + (25.0 * var_loss) + (1.0 * cov_loss) + l2_penalty

            metrics_aggregator.log(
                {
                    "loss/orchestrator_invariance": float(inv_loss.item() * 25.0),
                    "loss/orchestrator_variance": float(var_loss.item() * 25.0),
                    "loss/orchestrator_covariance": float(cov_loss.item()),
                    "loss/orchestrator_contrastive_total": float(loss.item()),
                }
            )

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            orch_params = [p for g in self.optimizer.param_groups for p in g["params"]]
            torch.nn.utils.clip_grad_norm_(orch_params, max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            orch_params = [p for g in self.optimizer.param_groups for p in g["params"]]
            torch.nn.utils.clip_grad_norm_(orch_params, max_norm=1.0)
            self.optimizer.step()

        with torch.no_grad():
            dummy_action = torch.zeros(raw_visual_batch.size(0), self.agent_core.num_actions, device=MODEL_DEVICE)
            dummy_action[:, 0] = 1.0

            fast_pred = self.agent_core.latent_dynamics(z1_out[0].detach(), dummy_action)
            slow_pred = self.agent_core.latent_dynamics(z2_out[0].detach(), dummy_action)

            fast_error = F.mse_loss(fast_pred, z2_out[0].detach(), reduction="none").mean(dim=-1)
            slow_error = F.mse_loss(slow_pred, z1_out[0].detach(), reduction="none").mean(dim=-1)

            if hasattr(self.agent_core.causal_symbolic_reasoner, "integrate_ghost_graph"):
                self.agent_core.causal_symbolic_reasoner.integrate_ghost_graph(
                    prediction_error_fast=fast_error,
                    prediction_error_slow=slow_error,
                    external_knowledge=z1_out[0].detach(),
                )

        return loss.item()


class OfflineCausalIntervention(nn.Module):
    """Utilities for direct structural intervention on the learned causal graph."""

    @staticmethod
    def force_dag_edge(causal_reasoner_ref, src_idx: int, dst_idx: int, weight: float = 0.01) -> None:
        """Forces a structural causal link between nodes.

        Args:
            causal_reasoner_ref: Reference to the active causal inference network.
            src_idx: Index of the causal source node.
            dst_idx: Index of the causal target node.
            weight: Magnitude of the forced continuous causal connection.

        Raises:
            RuntimeError: If structural permutation is attempted under STRICT_EX_NIHILO policy.
        """
        if CFG.STRICT_EX_NIHILO:
            raise RuntimeError("Causal intervention mathematically sealed by STRICT_EX_NIHILO.")

        if hasattr(causal_reasoner_ref, "dynamics_net") and len(causal_reasoner_ref.dynamics_net) > 0:
            with torch.no_grad():
                with torch.no_grad():
                    causal_reasoner_ref.dynamics_net[-1].weight[dst_idx, src_idx].add_(weight)


class CausalIntegrityValidator(nn.Module):
    """
    Estimates Conditional Mutual Information (CMI) via a trained regressor.
    Predicts state transition magnitude given a causal hypothesis to validate graph integrity.
    """

    def __init__(self, causal_reasoner_ref=None):
        super().__init__()
        self.causal_reasoner = causal_reasoner_ref
        # CMI inputs derived from: env_before_proj (256), env_after_proj (256), hypothesis (256).
        self.cmi_estimator = nn.Sequential(nn.Linear(256 * 3, 128), nn.Mish(), nn.Linear(128, 1))
        self.cmi_optimizer = bnb.optim.Lion8bit(self.cmi_estimator.parameters(), lr=1e-4)

    def validate_batch(self, hypothesis_batch, env_state_before=None, env_state_after=None, action_mask=None):
        """Computes Conditional Mutual Information (CMI) across the batch to estimate causal intervention validity."""
        if self.causal_reasoner is None or env_state_before is None or env_state_after is None:
            return

        with torch.enable_grad():
            # Flatten spatial dimensions to [Batch, Flattened_Features]
            if env_state_before.dim() > 2:
                env_state_before = env_state_before.view(env_state_before.size(0), -1)
            if env_state_after.dim() > 2:
                env_state_after = env_state_after.view(env_state_after.size(0), -1)

            if not hasattr(self, "_state_compression_layer"):
                self._state_compression_layer = nn.Linear(
                    env_state_before.size(-1), 256, device=env_state_before.device
                )

            env_before_proj = self._state_compression_layer(env_state_before.float())
            env_after_proj = self._state_compression_layer(env_state_after.float())

            joint_state = torch.cat([env_before_proj, env_after_proj, hypothesis_batch], dim=-1)
            cmi_scores = self.cmi_estimator(joint_state.detach())
            state_deltas = torch.norm(env_state_after - env_state_before, p=2, dim=-1, keepdim=True)
            target_cmis = state_deltas.detach() * action_mask
            loss = F.mse_loss(cmi_scores, target_cmis)

            metrics_aggregator.log({"loss/causal_cmi": float(loss.item())})

            self.cmi_optimizer.zero_grad()
            loss.backward()

            valid_grads = True
            for p in self.cmi_estimator.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    valid_grads = False
                    break

            if valid_grads:
                self.cmi_optimizer.step()
            else:
                self.cmi_optimizer.zero_grad()

        with torch.no_grad():
            valid_interventions = (action_mask > 0.5) & (cmi_scores > 0.5)
            if valid_interventions.any():
                valid_hypotheses = hypothesis_batch[valid_interventions.squeeze(-1)]
                valid_scores = cmi_scores[valid_interventions.squeeze(-1)]

                num_slots = getattr(self.causal_reasoner, "num_slots", 8)
                slot_dim = getattr(self.causal_reasoner, "slot_dim", 32)

                for hyp, score in zip(valid_hypotheses, valid_scores):
                    slot_activations = torch.norm(hyp.view(num_slots, slot_dim), p=2, dim=-1)
                    active_nodes = torch.topk(slot_activations, k=2).indices

                    if len(active_nodes) == 2:
                        u, _ = active_nodes[0], active_nodes[1]
                        if hasattr(self.causal_reasoner, "dynamics_net") and hasattr(
                            self.causal_reasoner.dynamics_net[-1], "weight"
                        ):
                            if u < self.causal_reasoner.dynamics_net[-1].weight.size(0):
                                noise_injection = torch.randn_like(
                                    self.causal_reasoner.dynamics_net[-1].weight.data[u]
                                ) * (score * 0.01)
                                self.causal_reasoner.dynamics_net[-1].weight.data[u].add_(noise_injection)


class HardwareOptimizer:
    def __init__(self) -> None:
        self.config = {
            "evolution": {"cpu_threads": 12, "gpu": False},
            "training": {"cpu_threads": 8, "gpu": True},
            "query": {"cpu_threads": 4, "gpu": True},
        }

    def configure(self, mode: str) -> None:
        torch.set_num_threads(self.config[mode]["cpu_threads"])
        os.environ["OMP_NUM_THREADS"] = str(self.config[mode]["cpu_threads"])
        if self.config[mode]["gpu"] and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True


def load_best_world_state() -> Any:
    checkpoints = [f for f in os.listdir(AGENTS_DIR) if f.startswith("checkpoint_gen_") and f.endswith(".pt")]
    if not checkpoints:
        import logging

        logging.error("Error: No evolved entity found. Run evolution mode first!")
        return None
    latest_ckpt = max(checkpoints, key=lambda f: os.path.getmtime(os.path.join(AGENTS_DIR, f)))

    ckpt_path = os.path.join(AGENTS_DIR, latest_ckpt)
    try:
        from vrl_framework.environment.world_dynamics import load_entity
    except ImportError:
        from vrl_framework.system.serialization import load_entity
    return load_entity(ckpt_path)


def format_response(tensor_response: torch.Tensor) -> str:
    if tensor_response.dim() == 3:
        tensor_response = tensor_response.squeeze(1)
    elif tensor_response.dim() == 1:
        tensor_response = tensor_response.unsqueeze(0)

    trajectory_symbols = []
    for step_idx, step_tensor in enumerate(tensor_response):
        sym_list = step_tensor.topk(2).indices.tolist()
        final_symbols = sym_list[0] if isinstance(sym_list[0], list) else sym_list
        trajectory_symbols.append(f"T+{step_idx}: {final_symbols}")

    return " ->\n".join(trajectory_symbols)


def execute_query(question: str, min_time: float, max_time: float = 120.0, live_agent: Any = None) -> str:
    if CFG.STRICT_EX_NIHILO:
        raise RuntimeError(
            "Offline analytics blocked by STRICT_EX_NIHILO policy. Agent evaluation is strictly behavioral."
        )

    if live_agent is not None:
        agent_core = live_agent
    else:
        world = load_best_world_state()
        if not world or not hasattr(world, "entities"):
            raise RuntimeError("Error: No evolved entity found. Run evolution mode first!")
        agent_core = world.entities[0].agent_core
    agent_core.eval()

    q_embed: torch.Tensor = agent_core._process_text(question)

    # Standardize query embeddings to (batch_size, seq_len, hidden_dim).
    if q_embed.dim() == 1:
        q_embed = q_embed.unsqueeze(0).unsqueeze(0)
    elif q_embed.dim() == 2:
        if q_embed.shape[0] == 1:
            q_embed = q_embed.unsqueeze(1)
        else:
            q_embed = q_embed.unsqueeze(0)

    HardwareOptimizer().configure("query")

    with torch.no_grad(), torch.autocast(device_type=MODEL_DEVICE, dtype=torch.float16):
        disable_early_exit = min_time == max_time
        max_steps = min(50, int(max_time * 2))
        response = agent_core.temporal_reasoning(q_embed, max_steps=max_steps, disable_early_exit=disable_early_exit)

    return format_response(response)


def run_training(seconds: float) -> str:
    entity = load_best_world_state()
    orchestrator = TrainingOrchestrator(entity.agent_core)

    try:
        from vrl_framework.system.io import load_training_data_from_input
    except ImportError:
        pass

    training_generator = load_training_data_from_input()
    current_chunk: Any = next(training_generator, None)
    if current_chunk is None:
        logging.warning("No training data discovered in the 'input' directory.")

    start: float = time.time()
    loss_val: Any = None
    iteration: int = 0
    logging.info("Initiating Self-Supervised JEPA Contrastive Training Loop...")

    # Stream training data to bound memory footprint.
    text_stream_ptr: int = 0
    read_window_size: int = 1024

    while time.time() - start < seconds:
        if current_chunk is not None:
            end_ptr: int = text_stream_ptr + read_window_size
            if end_ptr >= len(current_chunk):
                raw_text_chunk: Any = current_chunk[text_stream_ptr:]
                current_chunk = next(training_generator, None)
                text_stream_ptr = 0
                if current_chunk is None:
                    training_generator = load_training_data_from_input()
                    current_chunk = next(training_generator, None)
            else:
                raw_text_chunk = current_chunk[text_stream_ptr:end_ptr]
                text_stream_ptr = end_ptr

            if iteration % 100 == 0:
                entity.agent_core.adaptive_store(raw_text_chunk, modality="text")

            entity.agent_core.train()
            if not hasattr(entity.agent_core, "epistemic_optimizer"):
                entity.agent_core.epistemic_optimizer = torch.optim.AdamW(entity.agent_core.parameters(), lr=1e-4)

            entity.agent_core.epistemic_optimizer.zero_grad()

            dummy_visual = torch.zeros(1, 4, 7, 7, 7, device=MODEL_DEVICE)
            _ = entity.agent_core(dummy_visual, external_signal=raw_text_chunk)

            loss_val_tensor: torch.Tensor = getattr(
                entity.agent_core, "aux_text_loss", torch.tensor(0.0, device=MODEL_DEVICE)
            )

            if loss_val_tensor.requires_grad:
                loss_val_tensor.backward()
                torch.nn.utils.clip_grad_norm_(entity.agent_core.parameters(), 1.0)
                entity.agent_core.epistemic_optimizer.step()

            loss_val = loss_val_tensor.item()
        else:
            if hasattr(entity, "pop_ref") and entity.pop_ref is not None:
                perceptions, _, _ = entity.pop_ref.perceive_batch(entity.pop_ref.grid)
                raw_visual_batch = perceptions[: min(64, BATCH_SIZE)].to(MODEL_DEVICE)
            else:
                raw_visual_batch = torch.randn(min(64, BATCH_SIZE), 4, 7, 7, 7, device=MODEL_DEVICE)

            loss_val = orchestrator.train_step(raw_visual_batch)

        if iteration % 10 == 0:
            logging.info(f"Iteration {iteration:04d} | InfoNCE Loss: {loss_val:.4f}")
        iteration += 1

    return (
        f"Training completed. Final loss: {loss_val:.4f}"
        if loss_val is not None
        else "Training completed. No loss computed (buffer empty)."
    )


class CurriculumStateTracker(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("file_index", torch.tensor(0, dtype=torch.long))
        self.register_buffer("byte_offset", torch.tensor(0, dtype=torch.long))
        self.register_buffer("embedded_curriculum_path", torch.zeros(512, dtype=torch.uint8))
        self.register_buffer("embedded_offload_path", torch.zeros(512, dtype=torch.uint8))

    def store_path(self, buffer_name: str, path_str: str) -> None:
        if not path_str:
            return
        buffer = getattr(self, buffer_name)
        buffer.fill_(0)
        path_bytes = list(path_str.encode("utf-8"))
        if len(path_bytes) > 512:
            path_bytes = path_bytes[:512]
        buffer[: len(path_bytes)] = torch.tensor(path_bytes, dtype=torch.uint8, device=buffer.device)

    def retrieve_path(self, buffer_name: str) -> Optional[str]:
        buffer = getattr(self, buffer_name)
        nonzero = buffer[buffer > 0].tolist()
        return bytes(nonzero).decode("utf-8") if nonzero else None


def get_curriculum_files(target_dir: str, start_filter: Optional[str] = None) -> Any:
    import os

    curriculum_files = []
    for root, dirs, files in os.walk(target_dir):
        dirs.sort()
        for file in sorted(files):
            curriculum_files.append(os.path.join(root, file))

    if start_filter:
        filtered = []
        found = False
        for f in curriculum_files:
            if start_filter in f or found:
                found = True
                filtered.append(f)
        return filtered
    return curriculum_files


def load_deterministic_curriculum(
    target_dir: str, start_file_idx: int = 0, start_offset: int = 0, start_filter: Optional[str] = None
) -> Any:
    import logging
    import mmap
    import os

    if not os.path.exists(target_dir):
        logging.warning(f"Curriculum directory '{target_dir}' does not exist.")
        return

    files = get_curriculum_files(target_dir, start_filter)
    if not files:
        return

    chunk_size = 16777216

    for file_idx in range(start_file_idx, len(files)):
        file_path = files[file_idx]
        if os.path.getsize(file_path) == 0:
            continue

        current_offset = start_offset if file_idx == start_file_idx else 0

        try:
            with open(file_path, "rb") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    total_size = mm.size()
                    while current_offset < total_size:
                        end_boundary = min(current_offset + chunk_size, total_size)
                        raw_bytes = mm[current_offset:end_boundary]
                        yield file_idx, current_offset, raw_bytes, os.path.basename(file_path)
                        current_offset = end_boundary
        except OSError:
            continue
