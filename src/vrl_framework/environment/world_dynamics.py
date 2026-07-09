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

"""
Vectorized environment dynamics and physics engine.
Implements sparse tensor updates and PID-controlled curriculum generation.
"""

import collections
import multiprocessing
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import warp as wp

if multiprocessing.current_process().name == "MainProcess":
    wp.init()

import gc
import json
import logging
import math
import os
import threading
import time
import uuid
from typing import Any, Tuple

import bitsandbytes as bnb
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch.multiprocessing as mp
import wandb
from matplotlib.patches import Patch

from vrl_framework.core.contracts import PlannerValidator, PlanningBudget
from vrl_framework.core.settings import (
    AGENTS_DIR,
    CFG,
    DEACTIVATION_THRESHOLD,
    DESIRED_ENERGY,
    INIT_POPULATION,
    LEARNING_RATE,
    MAX_POPULATION,
    MIN_POPULATION,
    MODEL_DEVICE,
    PENALTY_GAMMA,
    RECOVERY_MARGIN,
    USE_3D_VOXELS,
    WORLD_DIM,
)
from vrl_framework.math_ops.geometry import compute_eff_dim, compute_expert_orthogonality, compute_traj_entropy


@wp.kernel
def step_sparse_substrate_kernel(
    state_grid: wp.array3d[wp.uint8],
    energy_grid: wp.array3d[wp.uint8],
    active_mask: wp.array3d[wp.uint8],
    dim_x: int,
    dim_y: int,
    dim_z: int,
):
    i, j, k = wp.tid()  # type: ignore[misc]
    if active_mask[i, j, k] == wp.uint8(0):  # type: ignore[index]
        return

    current_energy = energy_grid[i, j, k]  # type: ignore[index]
    if current_energy > wp.uint8(128):
        state_grid[i, j, k] = wp.uint8(1)  # type: ignore[index]
        energy_grid[i, j, k] = current_energy - wp.uint8(10)  # type: ignore[index]
    elif current_energy > wp.uint8(10):
        energy_grid[i, j, k] = current_energy - wp.uint8(1)  # type: ignore[index]
    else:
        state_grid[i, j, k] = wp.uint8(0)  # type: ignore[index]


class LearningProgressCurriculum:
    """
    PID-based curriculum manager tracking prediction error equilibrium.
    """

    penalty_grid: torch.Tensor
    target_penalty_grid: torch.Tensor
    integral_error_grid: torch.Tensor
    prev_error_grid: torch.Tensor
    kp: float
    ki: float
    kd: float

    def __init__(self, target_survival_rate: float = 0.6, learning_rate: float = 0.05, tau: float = 0.99):
        self.target_survival = target_survival_rate
        self.lr = learning_rate
        self.tau = tau
        self.optimal_homeostasis = DESIRED_ENERGY * 0.8

        self.surprisal_ema = 0.0
        self.surprisal_velocity_ema = 0.0

        self._pt_substrate_state = torch.zeros(
            (WORLD_DIM[0], WORLD_DIM[1], WORLD_DIM[2]), dtype=torch.uint8, device=MODEL_DEVICE
        )
        self._pt_substrate_energy = torch.zeros(
            (WORLD_DIM[0], WORLD_DIM[1], WORLD_DIM[2]), dtype=torch.uint8, device=MODEL_DEVICE
        )
        self._pt_active_mask = torch.zeros(
            (WORLD_DIM[0], WORLD_DIM[1], WORLD_DIM[2]), dtype=torch.uint8, device=MODEL_DEVICE
        )

        self.substrate_state: Any = None
        self.substrate_energy: Any = None
        self.active_mask: Any = None

    def step_physics_substrate(self, sub_steps: int = 5):
        if self.substrate_state is None:
            self.substrate_state = wp.from_dlpack(torch.utils.dlpack.to_dlpack(self._pt_substrate_state))
            self.substrate_energy = wp.from_dlpack(torch.utils.dlpack.to_dlpack(self._pt_substrate_energy))
            self.active_mask = wp.from_dlpack(torch.utils.dlpack.to_dlpack(self._pt_active_mask))

        for _ in range(sub_steps):
            wp.launch(
                kernel=step_sparse_substrate_kernel,
                dim=(WORLD_DIM[0], WORLD_DIM[1], WORLD_DIM[2]),
                inputs=[
                    self.substrate_state,
                    self.substrate_energy,
                    self.active_mask,
                    WORLD_DIM[0],
                    WORLD_DIM[1],
                    WORLD_DIM[2],
                ],
            )

    def compute_homeostatic_divergence(self, current_energy: torch.Tensor) -> torch.Tensor:
        return torch.abs(current_energy - self.optimal_homeostasis) / self.optimal_homeostasis

    def apply_somatic_penalty(self, external_reward: torch.Tensor, current_energy: torch.Tensor) -> torch.Tensor:
        """Scales external reward by subtracting weighted divergence from target energy state."""
        somatic_error = self.compute_homeostatic_divergence(current_energy)
        return external_reward - (somatic_error * 0.5)

    @property
    def penalty_multiplier(self):
        active = self.penalty_grid > 1e-4
        if active.any():
            return self.penalty_grid[active].mean().item()
        return self.penalty_grid.mean().item()

    def compute_multiplier(self, death_count: int, total_population: int, current_coverage: float = 16.0) -> float:
        return self.penalty_multiplier

    def compute_spatial_multiplier(self, deaths_grid: torch.Tensor, population_grid: torch.Tensor) -> torch.Tensor:
        active_mask = population_grid > 0
        survival_rate = 1.0 - (deaths_grid / torch.clamp(population_grid, min=1.0))

        error = (survival_rate - self.target_survival) * active_mask.type_as(survival_rate)

        self.integral_error_grid.mul_(0.99).add_(error)
        self.integral_error_grid.clamp_(min=0.0)
        derivative = error - self.prev_error_grid

        pid_adjustment = error * self.kp
        pid_adjustment.add_(self.integral_error_grid, alpha=self.ki)
        pid_adjustment.add_(derivative, alpha=self.kd)

        raw_penalty = torch.clamp(self.target_penalty_grid + pid_adjustment, min=1e-8, max=5.0)
        mask_f32 = active_mask.type_as(self.target_penalty_grid)
        self.target_penalty_grid.lerp_(raw_penalty, mask_f32)
        self.penalty_grid = self.tau * self.penalty_grid + (1.0 - self.tau) * self.target_penalty_grid
        self.prev_error_grid.copy_(error)

        return self.penalty_grid


# Global singleton for stateful curriculum tracking across generations
metabolic_curriculum = LearningProgressCurriculum()
metabolic_curriculum.penalty_grid = torch.ones(
    (WORLD_DIM[0], WORLD_DIM[1], WORLD_DIM[2]), dtype=torch.float32, device=MODEL_DEVICE
)
metabolic_curriculum.target_penalty_grid = torch.ones(
    (WORLD_DIM[0], WORLD_DIM[1], WORLD_DIM[2]), dtype=torch.float32, device=MODEL_DEVICE
)
metabolic_curriculum.integral_error_grid = torch.zeros(
    (WORLD_DIM[0], WORLD_DIM[1], WORLD_DIM[2]), dtype=torch.float32, device=MODEL_DEVICE
)
metabolic_curriculum.prev_error_grid = torch.zeros(
    (WORLD_DIM[0], WORLD_DIM[1], WORLD_DIM[2]), dtype=torch.float32, device=MODEL_DEVICE
)
metabolic_curriculum.kp = 0.1
metabolic_curriculum.ki = 0.01
metabolic_curriculum.kd = 0.05


def compute_survival_curriculum(generation: int, grace_period: int = 500) -> float:
    if generation < grace_period:
        return 0.0
    return min(1.0, (generation - grace_period) / 500.0)


wp.config.verify_cuda = False
wp.config.mode = "release"


@wp.kernel
def cost_kernel(
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    actions: wp.array[wp.vec3],
    energies: wp.array[wp.float32],
    hps: wp.array[wp.float32],
    anomalies: wp.array[wp.vec3],
    anomaly_states: wp.array[wp.int32],
    bounds: wp.vec3,
    dt: wp.float32,
    physics_multiplier: wp.float32,
    tax_multiplier: wp.float32,
):
    """Semi-implicit Euler integration with boundary repulsion."""
    tid = wp.tid()

    pos = positions[tid]
    vel = velocities[tid]
    act = actions[tid]

    friction_coef = 0.05 * physics_multiplier
    gravity_shift = wp.vec3(0.0, -9.81 * (physics_multiplier - 1.0), 0.0)

    repulsion_force = wp.vec3(0.0, 0.0, 0.0)
    wall_margin = 2.0
    k_repel = 50.0

    center = wp.vec3(bounds[0] * 0.5, bounds[1] * 0.5, bounds[2] * 0.5)
    dist_to_center = wp.length(pos - center)
    if dist_to_center > 0.1:
        repulsion_force = repulsion_force - ((pos - center) / dist_to_center) * (dist_to_center * 0.1) * (
            1.0 - (energies[tid] / 100.0)
        )

    if pos[0] < wall_margin:
        repulsion_force = repulsion_force + wp.vec3(k_repel / (pos[0] + 0.1), 0.0, 0.0)
    elif pos[0] > bounds[0] - wall_margin:
        repulsion_force = repulsion_force - wp.vec3(k_repel / (bounds[0] - pos[0] + 0.1), 0.0, 0.0)

    if pos[1] < wall_margin:
        repulsion_force = repulsion_force + wp.vec3(0.0, k_repel / (pos[1] + 0.1), 0.0)
    elif pos[1] > bounds[1] - wall_margin:
        repulsion_force = repulsion_force - wp.vec3(0.0, k_repel / (bounds[1] - pos[1] + 0.1), 0.0)

    if pos[2] < wall_margin:
        repulsion_force = repulsion_force + wp.vec3(0.0, 0.0, k_repel / (pos[2] + 0.1))
    elif pos[2] > bounds[2] - wall_margin:
        repulsion_force = repulsion_force - wp.vec3(0.0, 0.0, k_repel / (bounds[2] - pos[2] + 0.1))

    total_force = act + gravity_shift + repulsion_force

    vel_half = vel + total_force * (dt * 0.5)
    new_pos = pos + vel_half * dt

    vel_full = vel_half + total_force * (dt * 0.5)
    vel = vel_full * wp.exp(-friction_coef * dt)

    bounce_cost = 0.0
    if new_pos[0] < 0.0 or new_pos[0] > bounds[0]:
        new_pos = wp.vec3(wp.clamp(new_pos[0], 0.0, bounds[0]), new_pos[1], new_pos[2])
        vel = wp.vec3(-vel[0] * 0.5, vel[1], vel[2])
        bounce_cost += 1.5

    if new_pos[1] < 0.0 or new_pos[1] > bounds[1]:
        new_pos = wp.vec3(new_pos[0], wp.clamp(new_pos[1], 0.0, bounds[1]), new_pos[2])
        vel = wp.vec3(vel[0], -vel[1] * 0.5, vel[2])
        bounce_cost += 1.5

    if new_pos[2] < 0.0 or new_pos[2] > bounds[2]:
        new_pos = wp.vec3(new_pos[0], new_pos[1], wp.clamp(new_pos[2], 0.0, bounds[2]))
        vel = wp.vec3(vel[0], vel[1], -vel[2] * 0.5)
        bounce_cost += 1.5

    positions[tid] = new_pos  # type: ignore[index]
    velocities[tid] = vel  # type: ignore[index]

    kinetic_energy = wp.min(0.5 * wp.dot(vel, vel), 1000.0)
    action_magnitude = wp.min(wp.length(act), 50.0)
    cost_drain = (
        (0.05 + action_magnitude * 0.02 + kinetic_energy * 0.01 + bounce_cost) * dt * physics_multiplier
    ) * tax_multiplier

    energy_replenish = float(0.0)
    for i in range(anomalies.shape[0]):
        if anomaly_states[i] == 1:
            dist = wp.length(new_pos - anomalies[i])
            if dist < 2.0:
                energy_replenish += 25.0
                anomaly_states[i] = 0  # type: ignore[index]

    base_new_energy = energies[tid] - cost_drain

    if energy_replenish > 0.0:
        lambda_s = 0.05
        max_e = 100.0
        base_new_energy = max_e * (1.0 - wp.exp(-lambda_s * energy_replenish)) + base_new_energy * wp.exp(
            -lambda_s * energy_replenish
        )

    if base_new_energy < 0.0:
        hps[tid] = hps[tid] + base_new_energy  # type: ignore[index]
        base_new_energy = 0.0

    energies[tid] = wp.min(base_new_energy, 100.0)  # type: ignore[index]


class ProceduralEnvironmentGenerator(nn.Module):
    """
    Maps global spatial states to localized geometric displacements
    and friction modulations via a 3D CNN.
    """

    def __init__(self, num_anomalies: int = 10, text_buffer_capacity: int = 10000) -> None:
        super().__init__()
        self.num_anomalies = num_anomalies
        hidden_dim = getattr(CFG, "HIDDEN_DIM", 256)

        self.architect_conv = nn.Sequential(
            nn.Conv3d(4, 16, kernel_size=3, stride=2, padding=1),
            nn.Mish(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.Mish(),
            nn.Flatten(),
            nn.Linear(32 * 2 * 2 * 2, hidden_dim),
            nn.Mish(),
        )
        self.temporal_memory = nn.GRUCell(hidden_dim, hidden_dim)
        self.terrain_projector = nn.Linear(hidden_dim, num_anomalies * 3 + 1)

        self.optimizer = torch.optim.AdamW(
            list(self.architect_conv.parameters())
            + list(self.temporal_memory.parameters())
            + list(self.terrain_projector.parameters()),
            lr=1e-4,
        )
        self.register_buffer("solvability_fifo", torch.zeros(100, dtype=torch.float32))
        self.register_buffer("fifo_ptr", torch.tensor(0, dtype=torch.long))
        self.architect_state: torch.Tensor
        self.register_buffer("architect_state", torch.zeros(1, hidden_dim))

        self.text_buffer_capacity = text_buffer_capacity
        self.text_buffer: list[str] = []
        self.register_buffer("text_compression_history", torch.zeros(text_buffer_capacity, dtype=torch.float32))
        self.register_buffer("text_learning_progress", torch.zeros(text_buffer_capacity, dtype=torch.float32))
        self.text_ptr = 0

    def generate_terrain(self, spatial_volume: torch.Tensor, uncertainty_signal: float = 0.0):
        """
        Computes geometric displacement and friction modulation maps.

        Args:
            spatial_volume: [Batch, Channels, X, Y, Z]
            uncertainty_signal: Scalar float
        """
        global_state = spatial_volume.mean(dim=0, keepdim=True)
        conv_features = self.architect_conv(global_state)

        self.architect_state = self.temporal_memory(conv_features, self.architect_state)
        architect_logits = self.terrain_projector(self.architect_state)

        displacement_logits = architect_logits[:, :-1]
        physics_logits = architect_logits[:, -1:]

        base_displacement = torch.tanh(displacement_logits.view(self.num_anomalies, 3)) * 10.0
        structured_walls = torch.sign(base_displacement) * torch.clamp(torch.abs(base_displacement), min=5.0, max=15.0)

        final_displacement = (1.0 - uncertainty_signal) * base_displacement + uncertainty_signal * structured_walls
        physics_multiplier = 1.0 + torch.tanh(physics_logits.squeeze(-1)) * 0.5

        return final_displacement, physics_multiplier

    def add_text_challenge(self, text_chunk: str, initial_compression: float):
        if len(self.text_buffer) < self.text_buffer_capacity:
            self.text_buffer.append(text_chunk)
            idx = len(self.text_buffer) - 1
        else:
            idx = self.text_ptr
            self.text_buffer[idx] = text_chunk
            self.text_ptr = (self.text_ptr + 1) % self.text_buffer_capacity

        self.text_compression_history[idx] = initial_compression
        self.text_learning_progress[idx] = 1.0

    def sample_text_curriculum(self, batch_size=1):
        """
        Samples text curricula proportionally to their historical compression progress.
        """
        if not self.text_buffer:
            return None

        valid_size = len(self.text_buffer)
        valid_progress = self.text_learning_progress[:valid_size]
        probs = F.softmax(valid_progress / 0.1, dim=0)
        selected_indices_tensor = torch.multinomial(probs, num_samples=batch_size, replacement=True)
        selected_indices = selected_indices_tensor.cpu().tolist()
        return [self.text_buffer[i] for i in selected_indices], selected_indices

    def update_text_progress(
        self,
        indices: list,
        prior_mu: torch.Tensor,
        prior_logvar: torch.Tensor,
        post_mu: torch.Tensor,
        post_logvar: torch.Tensor,
        new_surprisal: torch.Tensor,
    ) -> None:
        """
        Evaluates compression via KL divergence.

        Args:
            prior_mu, prior_logvar, post_mu, post_logvar: [Batch, Latent_Dim]
        """
        if not indices:
            return
        with torch.no_grad():
            kl_div = -0.5 * torch.sum(
                1
                + post_logvar
                - prior_logvar
                - ((post_mu - prior_mu).pow(2) / prior_logvar.exp())
                - torch.exp(post_logvar - prior_logvar),
                dim=-1,
            )

        compression_score = new_surprisal + (0.1 * kl_div.mean())
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=self.text_compression_history.device)

        old_compression = self.text_compression_history[idx_tensor]
        progress = old_compression - compression_score

        self.text_learning_progress.data[idx_tensor] = (
            0.8 * self.text_learning_progress.data[idx_tensor] + 0.2 * progress
        )
        self.text_compression_history[idx_tensor] = compression_score.squeeze()

    def optimize_regret(
        self, agent_fitnesses: torch.Tensor, spatial_volume: torch.Tensor, jepa_error: Any = None
    ) -> None:
        if agent_fitnesses.size(0) < 2:
            return
        self.optimizer.zero_grad(set_to_none=True)

        ptr = self.fifo_ptr.item()
        raw_surprise = jepa_error.mean() if jepa_error is not None else agent_fitnesses.var()

        if not hasattr(self, "leaky_surprise_signal"):
            self.register_buffer("leaky_surprise_signal", raw_surprise.detach().clone())
        self.leaky_surprise_signal.lerp_(raw_surprise.detach(), 0.001)
        surprise_signal = self.leaky_surprise_signal

        self.solvability_fifo[ptr] = surprise_signal.detach()
        self.fifo_ptr.fill_((ptr + 1) % 100)
        recent_history = torch.cat([self.solvability_fifo[ptr:], self.solvability_fifo[:ptr]])[-10:]

        valid_history_mask = (self.solvability_fifo > 0.0).sum() >= 10
        _ = torch.where(
            valid_history_mask,
            recent_history[5:].mean() - recent_history[:5].mean(),
            torch.tensor(-1.0, device=surprise_signal.device),
        )

        global_state = spatial_volume.mean(dim=0, keepdim=True)
        conv_features = self.architect_conv(global_state)

        architect_state_eval = self.temporal_memory(conv_features, self.architect_state.detach())
        architect_logits = self.terrain_projector(architect_state_eval)

        displacement_logits = architect_logits[:, :-1]
        physics_logits = architect_logits[:, -1:]

        base_displacement = torch.tanh(displacement_logits) * 10.0
        env_complexity_norm = torch.norm(base_displacement, p=2) + torch.abs(physics_logits.squeeze(-1)).sum()

        wobble = torch.sin(torch.tensor(ptr, device=surprise_signal.device).float() * 0.1) * 2.0
        dynamic_target_complexity = torch.clamp(
            (0.4 - surprise_signal) * 25.0 + wobble + torch.randn_like(surprise_signal), min=-5.0, max=15.0
        ).detach()

        surrogate_loss = F.mse_loss(env_complexity_norm, dynamic_target_complexity)

        loss = surrogate_loss + (0.01 * torch.linalg.vector_norm(architect_logits, ord=2, dim=-1).mean())

        from vrl_framework.trainer.ppo_engine import metrics_aggregator

        metrics_aggregator.log(
            {"loss/curriculum_surrogate": float(surrogate_loss.item()), "loss/curriculum_total": float(loss.item())}
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()


class VectorizedPopulation(nn.Module):
    """Batched agent tensor buffers."""

    active_mask: torch.Tensor
    signal_tensor: torch.Tensor
    ids: torch.Tensor
    ages: torch.Tensor
    survival_steps: torch.Tensor
    energies: torch.Tensor
    hps: torch.Tensor
    health_bands: torch.Tensor
    fitness: torch.Tensor
    prev_fitness: torch.Tensor
    stagnation_counters: torch.Tensor
    ts_mu: torch.Tensor
    ts_sigma: torch.Tensor
    population_gamma: torch.Tensor
    population_beta: torch.Tensor
    population_masks: torch.Tensor
    replicated_flags: torch.Tensor
    visited_spatial_grid: torch.Tensor
    permanent_visited_grid: torch.Tensor
    energy_intake: torch.Tensor
    positions: torch.Tensor
    hidden_states: torch.Tensor
    proprioceptive_states: torch.Tensor
    velocities: torch.Tensor
    genetic_codes: torch.Tensor
    batched_actor_w1: torch.Tensor
    batched_actor_b1: torch.Tensor
    batched_actor_w2: torch.Tensor
    batched_actor_b2: torch.Tensor
    batched_actor_w3: torch.Tensor
    batched_actor_b3: torch.Tensor
    move_vectors: torch.Tensor
    smoothed_intake_ema: torch.Tensor
    smoothed_penalty_ema: torch.Tensor
    prev_state_features_batch: Any
    last_state_features_batch: Any

    def __init__(self, initial_agents: int, world_dim: Tuple[int, ...], max_agents: int = 128) -> None:
        super().__init__()
        self.max_agents = max_agents
        self.world_dim = world_dim

        self.register_buffer("active_mask", torch.zeros(max_agents, dtype=torch.bool))
        self.active_mask[:initial_agents] = True

        self.register_buffer(
            "signal_tensor", torch.zeros(world_dim[0], world_dim[1], world_dim[2], 128, dtype=torch.float16)
        )

        self.register_buffer("ids", torch.arange(max_agents, dtype=torch.long))
        self.register_buffer("ages", torch.zeros(max_agents, dtype=torch.long))
        self.register_buffer("survival_steps", torch.zeros(max_agents, dtype=torch.long))
        self.register_buffer("energies", torch.full((max_agents,), 1.5))
        self.register_buffer("hps", torch.ones(max_agents, dtype=torch.float32))
        self.register_buffer("health_bands", torch.zeros(max_agents, dtype=torch.long))
        self.register_buffer("fitness", torch.zeros(max_agents))
        self.register_buffer("prev_fitness", torch.zeros(max_agents))
        self.register_buffer("stagnation_counters", torch.zeros(max_agents, dtype=torch.long))
        self.register_buffer("ts_mu", torch.full((max_agents,), 25.0))
        self.register_buffer("ts_sigma", torch.full((max_agents,), 25.0 / 3.0))
        self.register_buffer("population_gamma", torch.ones(max_agents, 256, dtype=torch.float16))
        self.register_buffer("population_beta", torch.zeros(max_agents, 256, dtype=torch.float16))
        self.register_buffer("population_masks", torch.ones(max_agents, 256, dtype=torch.bool))
        self.register_buffer("replicated_flags", torch.zeros(max_agents, dtype=torch.bool))
        self.register_buffer(
            "visited_spatial_grid", torch.zeros((world_dim[0], world_dim[1], world_dim[2]), dtype=torch.float32)
        )
        self.register_buffer(
            "permanent_visited_grid", torch.zeros((world_dim[0], world_dim[1], world_dim[2]), dtype=torch.bool)
        )
        self.register_buffer("energy_intake", torch.zeros(max_agents, dtype=torch.float32))

        positions_init = torch.rand((max_agents, 4), dtype=torch.float32)
        positions_init[:, 0] *= float(self.world_dim[0])
        positions_init[:, 1] *= float(self.world_dim[1])
        positions_init[:, 2] *= float(self.world_dim[2])
        positions_init[:, 3] *= float(self.world_dim[3] if len(self.world_dim) > 3 else 1.0)
        self.register_buffer("positions", positions_init)
        self.register_buffer("hidden_states", torch.randn(max_agents, 256))
        self.register_buffer("proprioceptive_states", torch.zeros(max_agents, 32))
        self.register_buffer("velocities", torch.zeros(max_agents, 3))

        base_genetics = torch.tensor([[0.03, 0.2, 1.0]])
        genetic_noise = torch.randn(max_agents, 3) * 0.01
        self.register_buffer("genetic_codes", (base_genetics + genetic_noise).clamp(0.01, 1.5))

        # Instantiate shared policy architecture to enforce parameter tying across the population.
        from vrl_framework.models.agents import RLAgent

        self.global_agent_core = RLAgent(sensory_input_shape=(4, 7, 7, 7), num_actions=17).to(MODEL_DEVICE)
        self.agent_cores = [self.global_agent_core for _ in range(self.max_agents)]

        self.register_buffer("batched_actor_w1", torch.zeros(self.max_agents, 256, 512, dtype=torch.float32))
        self.register_buffer("batched_actor_b1", torch.zeros(self.max_agents, 1, 256, dtype=torch.float32))
        self.register_buffer("batched_actor_w2", torch.zeros(self.max_agents, 128, 256, dtype=torch.float32))
        self.register_buffer("batched_actor_b2", torch.zeros(self.max_agents, 1, 128, dtype=torch.float32))
        self.register_buffer("batched_actor_w3", torch.zeros(self.max_agents, 8, 128, dtype=torch.float32))
        self.register_buffer("batched_actor_b3", torch.zeros(self.max_agents, 1, 8, dtype=torch.float32))

        self.experience_states: collections.deque = collections.deque(maxlen=10000)
        self.experience_actions: collections.deque = collections.deque(maxlen=10000)
        self.experience_next_states: collections.deque = collections.deque(maxlen=10000)
        self.last_state_features_batch = None

        self.register_buffer(
            "move_vectors",
            torch.tensor(
                [
                    [-1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 0.0, 1.0],
                    [-1.0, -1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
                device=MODEL_DEVICE,
            ),
        )

    def sync_with_entities_list(self, entities_list: list) -> None:
        current_count = len(entities_list)
        self.entities = entities_list

        with torch.no_grad():
            gamma_buffer = torch.zeros_like(self.population_gamma)
            beta_buffer = torch.zeros_like(self.population_beta)
            mask_buffer = torch.zeros_like(self.population_masks)

            for new_idx, ent in enumerate(entities_list[: self.max_agents]):
                if (
                    hasattr(ent, "idx")
                    and ent.idx is not None
                    and not getattr(ent, "needs_epigenetic_inheritance", False)
                ):
                    gamma_buffer[new_idx] = self.population_gamma[ent.idx]
                    beta_buffer[new_idx] = self.population_beta[ent.idx]
                    mask_buffer[new_idx] = self.population_masks[ent.idx]

            self.population_gamma.copy_(gamma_buffer)
            self.population_beta.copy_(beta_buffer)
            self.population_masks.copy_(mask_buffer)

            self.active_mask.fill_(False)
            self.active_mask[: min(current_count, self.max_agents)] = True

            for idx, ent in enumerate(entities_list[: self.max_agents]):
                if hasattr(ent, "_pos_fallback") and ent._pos_fallback is not None:
                    fallback_val = ent._pos_fallback[:3]
                    self.positions[idx][: len(fallback_val)] = torch.as_tensor(
                        fallback_val, device=self.positions.device, dtype=torch.float32
                    )
                ent.idx = idx
                ent.pop_ref = self
                ent.agent_core = self.global_agent_core
                if getattr(ent, "needs_epigenetic_inheritance", False) and hasattr(ent, "origin_idx"):
                    p_idx = ent.origin_idx
                    self.population_gamma[idx] = self.population_gamma[p_idx].clone()
                    self.population_beta[idx] = self.population_beta[p_idx].clone()
                    self.population_masks[idx] = self.population_masks[p_idx].clone()

                    mut_rate = ent.genetic_code.get("mutation_rate", 0.02)
                    noise_gamma = torch.randn_like(self.population_gamma[idx])
                    noise_beta = torch.randn_like(self.population_beta[idx])

                    norm_sq_gamma = torch.norm(self.population_gamma[idx]) ** 2 + 1e-8
                    norm_sq_beta = torch.norm(self.population_beta[idx]) ** 2 + 1e-8
                    noise_gamma -= (
                        torch.dot(noise_gamma.flatten(), self.population_gamma[idx].flatten()) / norm_sq_gamma
                    ) * self.population_gamma[idx]
                    noise_beta -= (
                        torch.dot(noise_beta.flatten(), self.population_beta[idx].flatten()) / norm_sq_beta
                    ) * self.population_beta[idx]

                    self.population_gamma[idx].add_(noise_gamma, alpha=mut_rate)
                    self.population_beta[idx].add_(noise_beta, alpha=mut_rate)

                    flip_mask = torch.rand_like(self.population_masks[idx].float()) < (mut_rate * 0.1)
                    self.population_masks[idx] ^= flip_mask

                    ent.needs_epigenetic_inheritance = False

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = actions.size(0)

        with torch.no_grad():
            act_idx = torch.clamp(actions.view(-1).long(), 0, 16)
            active_mask_slice = self.active_mask[:batch_size]

            movement = self.move_vectors[act_idx] * active_mask_slice.unsqueeze(1).to(self.move_vectors.dtype)

            self.positions[:batch_size, :3] += movement

            if not hasattr(self, "_cached_world_dim_tensor"):
                self._cached_world_dim_tensor = torch.tensor(
                    self.world_dim[:3], device=self.positions.device, dtype=self.positions.dtype
                )
            self.positions[:batch_size, :3].remainder_(self._cached_world_dim_tensor)

            pos_x = self.positions[:batch_size, 0].long()
            pos_y = self.positions[:batch_size, 1].long()
            pos_z = self.positions[:batch_size, 2].long()
            tax_mult = metabolic_curriculum.penalty_grid[pos_x, pos_y, pos_z]

            movement_magnitude = torch.norm(movement, p=2, dim=-1)

            metabolic_burn = (0.05 * tax_mult) + (movement_magnitude * 0.02)
            energy_discovery = torch.where(
                (movement_magnitude > 1e-3) & (torch.rand_like(movement_magnitude) < 0.05), 1.5, 0.0
            )

            self.energy_intake[:batch_size] += energy_discovery * (1.0 + torch.exp(-movement_magnitude))

            safe_burn = torch.nan_to_num(metabolic_burn, nan=0.05)
            self.energies[:batch_size].sub_(safe_burn * active_mask_slice.type_as(safe_burn)).clamp_(
                min=-1.0, max=float(DESIRED_ENERGY * 10.0)
            )

            inactive_mask = (self.energies[:batch_size] <= 0) & active_mask_slice

            survival_reward = 0.05 - 1.05 * inactive_mask.type_as(self.energies)
            discovery_reward = torch.nan_to_num(energy_discovery * 0.5, nan=0.0)

            rewards = (survival_reward + discovery_reward) * active_mask_slice.type_as(survival_reward)
            rewards = torch.nan_to_num(rewards, nan=0.0, posinf=5.0, neginf=-5.0)

            self.fitness[:batch_size].add_(rewards)
            self.fitness[:batch_size] = torch.nan_to_num(self.fitness[:batch_size], nan=0.0)

        next_states = self.hidden_states[:batch_size].clone()
        return next_states, rewards, inactive_mask

    def sample_opponent_trueskill(self, agent_idx: int) -> int:
        """
        Samples an opponent index using epsilon-greedy over TrueSkill lower bounds.
        """
        with torch.no_grad():
            active_count = int(self.active_mask.sum().item())
            if active_count <= 1:
                return agent_idx

            valid_pool = torch.where(self.active_mask)[0]
            if random.random() < 0.05:
                return int(valid_pool[torch.randint(0, active_count, (1,))].item())

            pessimistic_scores = self.ts_mu - (3.0 * self.ts_sigma)

            mask = self.active_mask.clone()
            mask[agent_idx] = False

            valid_pool_masked = torch.where(mask)[0]
            pool_size = valid_pool_masked.numel()
            if pool_size == 0:
                return agent_idx

            valid_scores = torch.clamp(pessimistic_scores[valid_pool_masked], min=-100.0, max=100.0)
            probs = F.softmax(valid_scores / 5.0, dim=0)

            probs = torch.nan_to_num(probs, nan=0.0)
            if probs.sum().item() <= 0:
                return int(valid_pool_masked[torch.randint(0, pool_size, (1,))].item())

            sampled_valid_idx = int(torch.multinomial(probs, 1).item())
            return int(valid_pool_masked[sampled_valid_idx].item())

    def _init_random_positions(self) -> torch.Tensor:
        pos = torch.zeros(self.max_agents, 4, dtype=torch.long)
        for i, dim in enumerate(self.world_dim):
            pos[:, i] = torch.randint(0, dim, (self.max_agents,))
        return pos

    def _ensure_capacity_buffers(self, batch_size: int, device: torch.device) -> None:
        if not hasattr(self, "_reusable_inactive_mask") or self._reusable_inactive_mask.size(0) != batch_size:
            self.register_buffer("_reusable_inactive_mask", torch.zeros(batch_size, dtype=torch.bool, device=device))

    def perceive_batch(
        self, environment_grid: torch.Tensor, audio_grid: Any = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = self.max_agents
        x = self.positions[:, 0].clamp(3, self.world_dim[0] - 4).long()
        y = self.positions[:, 1].clamp(3, self.world_dim[1] - 4).long()
        z = self.positions[:, 2].clamp(3, self.world_dim[2] - 4).long()
        _ = self.positions[:, 3].clamp(0, self.world_dim[3] - 1).long()

        if not hasattr(self, "_dx"):
            self._dx = torch.arange(-3, 4, device=self.positions.device).view(7, 1, 1)
            self._dy = torch.arange(-3, 4, device=self.positions.device).view(1, 7, 1)
            self._dz = torch.arange(-3, 4, device=self.positions.device).view(1, 1, 7)

        X_idx = (x.view(N, 1, 1, 1) + self._dx).clamp(0, self.world_dim[0] - 1)
        Y_idx = (y.view(N, 1, 1, 1) + self._dy).clamp(0, self.world_dim[1] - 1)
        Z_idx = (z.view(N, 1, 1, 1) + self._dz).clamp(0, self.world_dim[2] - 1)

        perceptions = environment_grid[X_idx, Y_idx, Z_idx, :]
        perceptions = perceptions.permute(0, 4, 1, 2, 3)

        latent_signals = self.signal_tensor[x, y, z].float()

        if audio_grid is not None:
            audio = audio_grid.expand(N, 1, -1)
        else:
            audio = torch.zeros(N, 1, 44100, device=self.positions.device)

        return perceptions.float(), latent_signals, audio

    def act_batch(
        self, perceptions: torch.Tensor, audio: Any = None, external_signals: Any = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = perceptions.size(0)
        device = MODEL_DEVICE

        actions = torch.zeros(batch_size, dtype=torch.long, device=device)
        log_probs = torch.zeros(batch_size, dtype=torch.float, device=device)

        with torch.no_grad():
            base_agent_core = self.agent_cores[0]
            base_agent_core.eval()

            if hasattr(base_agent_core, "step_counter"):
                base_agent_core.step_counter.add_(1)

            if not self.active_mask[:batch_size].any():
                return actions, log_probs

            active_indices = torch.nonzero(self.active_mask[:batch_size], as_tuple=True)[0]
            active_perceptions = perceptions[active_indices]

            if audio is None:
                active_audio = torch.zeros(active_indices.size(0), 1, 44100, device=device)
            else:
                active_audio = audio[active_indices]

            visual_features_active = base_agent_core.sensory(active_perceptions)
            audio_features_active = base_agent_core.multimodal.process_audio(active_audio)

            visual_features = visual_features_active.new_zeros((batch_size,) + visual_features_active.shape[1:])
            audio_features = audio_features_active.new_zeros((batch_size,) + audio_features_active.shape[1:])

            visual_features[active_indices] = visual_features_active
            audio_features[active_indices] = audio_features_active

            modalities = {
                "sensory": visual_features,
                "audio": audio_features,
                "proprioception": self.proprioceptive_states,
            }

            fused_context = base_agent_core.bottleneck_attention(modalities)
            for _ in range(2):
                top_down_prior = torch.tanh(fused_context)
                prediction_residual = visual_features - top_down_prior
                modalities["sensory"] = visual_features + prediction_residual

                fused_context_updated = base_agent_core.bottleneck_attention(modalities)
                fused_context = (fused_context + fused_context_updated) * 0.5

            if hasattr(self, "last_state_features_batch") and self.last_state_features_batch is not None:
                self.prev_state_features_batch = self.last_state_features_batch.clone()
            else:
                _init_jepa = base_agent_core.jepa(fused_context)
                self.prev_state_features_batch = (
                    _init_jepa[0] if isinstance(_init_jepa, tuple) else _init_jepa
                ).detach()

            self.prev_fused_context_batch = fused_context.detach()

            metabolic_stamina = self.energies[:batch_size].unsqueeze(-1).to(torch.float16)
            valence_entropy_scale = torch.clamp(1.0 - (metabolic_stamina / float(DESIRED_ENERGY)), 0.0, 1.0)

            if hasattr(base_agent_core, "causal_symbolic_reasoner") and hasattr(
                base_agent_core.causal_symbolic_reasoner.dynamics_net[-1], "weight"
            ):
                dag_uncertainty = torch.var(base_agent_core.causal_symbolic_reasoner.dynamics_net[-1].weight)
            else:
                dag_uncertainty = torch.tensor(0.0, device=device)
            safe_dag = dag_uncertainty.unsqueeze(0).expand(batch_size, 1).to(torch.float16)
            b_size = metabolic_stamina.size(0)
            interoceptive_state = torch.cat(
                [
                    metabolic_stamina.view(b_size, -1).float(),
                    valence_entropy_scale.view(b_size, -1).float(),
                    safe_dag.view(b_size, -1).float(),
                ],
                dim=-1,
            )

            # Late-binding of interoceptive modules bypasses base architectural constraints.
            # Warning: Causes graph breaks in torch._dynamo; targeted for refactor into AgentCore.__init__.
            if not hasattr(base_agent_core, "interoceptive_jepa"):
                base_agent_core.interoceptive_jepa = nn.Sequential(
                    nn.Linear(3, 128), nn.Mish(), nn.Linear(128, 64)
                ).to(device)
                base_agent_core.interoceptive_predictor = nn.Sequential(
                    nn.Linear(64, 64), nn.Mish(), nn.Linear(64, 3)
                ).to(device)
                base_agent_core.interoceptive_target = torch.zeros_like(interoceptive_state)

            int_latent = base_agent_core.interoceptive_jepa(interoceptive_state)
            int_pred = base_agent_core.interoceptive_predictor(int_latent)

            somatic_surprisal = F.mse_loss(int_pred, base_agent_core.interoceptive_target, reduction="none").mean(
                dim=-1, keepdim=True
            )
            base_agent_core.interoceptive_target = interoceptive_state.detach()

            somatic_stress_global = somatic_surprisal.mean()
            structural_noise = torch.randn_like(fused_context).mul_(valence_entropy_scale).mul_(0.1)

            circadian_wave = torch.sin(base_agent_core.step_counter.float() / 100.0)
            exhaustion_gate = torch.clamp_(1.0 - (metabolic_stamina / 2.0), min=0.0, max=1.0)
            consolidation_weight = exhaustion_gate.mul_(0.5 + 0.5 * circadian_wave)

            fused_context = fused_context.add_(structural_noise).mul_(1.0 - (consolidation_weight * 0.1))

            jepa_out = base_agent_core.jepa(fused_context)
            latent_context = jepa_out[0] if isinstance(jepa_out, tuple) else jepa_out

            base_agent_core.cognitive_fatigue += somatic_stress_global * 0.1
            if base_agent_core.cognitive_fatigue.item() > 100.0:
                base_agent_core._offline_consolidation(latent_context.mean(dim=0, keepdim=True))

            proprioceptive_anchor = F.pad(
                self.proprioceptive_states[:batch_size],
                (0, latent_context.size(-1) - self.proprioceptive_states.size(-1)),
                "constant",
                0.0,
            )
            latent_context = latent_context - proprioceptive_anchor

            latent_context = F.normalize(latent_context, p=2, dim=-1)
            latent_context = latent_context + torch.randn_like(latent_context) * 0.01
            latent_context = F.normalize(latent_context, p=2, dim=-1)
            latent_context = latent_context + (proprioceptive_anchor * 0.1)

            # Apply Feature-wise Linear Modulation (FiLM) using population-specific affine transformations.
            if hasattr(self, "population_gamma"):
                agent_gamma = self.population_gamma[:batch_size] * base_agent_core.global_gamma
                agent_beta = self.population_beta[:batch_size] + base_agent_core.global_beta
                latent_context = (latent_context * agent_gamma) + agent_beta

            latent_context = latent_context.detach()

            if not hasattr(self, "last_actions_batch"):
                self.last_actions_batch = torch.zeros(batch_size, base_agent_core.num_actions, device=device)
                self.last_rewards_batch = torch.zeros(batch_size, 1, device=device)

            base_agent_core.last_action_one_hot = self.last_actions_batch
            base_agent_core.last_reward = self.last_rewards_batch

            moe_context = base_agent_core.moe(latent_context)

            if not hasattr(self, "intent_matrix"):
                self.intent_matrix = torch.zeros(self.max_agents, 256, device=device)

            if hasattr(base_agent_core, "manager_goal") and base_agent_core.manager_goal.size(0) == batch_size:
                self.intent_matrix[:batch_size] = base_agent_core.manager_goal.detach()

            batch_intents = self.intent_matrix[:batch_size]
            base_agent_core.eval()

            cognitive_stress = getattr(base_agent_core.jepa, "proxy_surprisal", 0.0)
            if not isinstance(cognitive_stress, torch.Tensor):
                cognitive_stress = torch.tensor(float(cognitive_stress), device=device)

            avg_hp = self.energies[:batch_size].mean()

            if cognitive_stress.numel() > 1:
                c_mean = cognitive_stress.mean()
                c_var = cognitive_stress.var(unbiased=False)
                ontological_shock = (cognitive_stress > (c_mean + torch.sqrt(c_var) * 0.5)).view(-1, 1)
            else:
                ontological_shock = (cognitive_stress > 0.1).view(-1, 1)
            proactive_shock = avg_hp < (DESIRED_ENERGY * 0.3)
            stress_mask = ontological_shock | proactive_shock

            planned_context = base_agent_core.hierarchical_planner(moe_context, batch_intents)
            worker_context = torch.where(stress_mask, planned_context, moe_context)

            if hasattr(base_agent_core, "system2_override_divergence") or True:
                cos_sim = F.cosine_similarity(planned_context.float(), moe_context.float(), dim=-1)
                divergence_from_base = 1.0 - torch.nan_to_num(cos_sim.mean(), nan=1.0)
                base_agent_core.system2_override_divergence = torch.clamp(
                    divergence_from_base + (torch.randn(1, device=device).squeeze() * 0.01), min=0.01, max=22.0
                )

            actor_out = base_agent_core.actor_critic(worker_context, intent_context=batch_intents)
            policy_logits = (
                actor_out.policy_logits
                if hasattr(actor_out, "policy_logits")
                else (actor_out[0] if isinstance(actor_out, tuple) else actor_out)
            )

            if not hasattr(self, "_local_py_step"):
                self._local_py_step = 0
            self._local_py_step += 1
            local_step = self._local_py_step

            if local_step % 10 == 0 and hasattr(base_agent_core.trainer, "select_environment_action"):
                with torch.no_grad():
                    elite_worker_context = worker_context[0:1]
                    elite_intent = batch_intents[0:1] if batch_intents is not None else None
                    mcts_sampled_action = base_agent_core.trainer.select_environment_action(
                        elite_worker_context, elite_intent
                    )
                    if mcts_sampled_action is not None and policy_logits.dim() >= 2:
                        override_val = 50.0
                        action_idx = (
                            mcts_sampled_action.view(1)
                            if mcts_sampled_action.numel() == 1
                            else mcts_sampled_action[0:1]
                        )
                        if policy_logits.dim() == 3:
                            policy_logits[0, 0, :].fill_(-50.0)
                            policy_logits[0, 0].scatter_(0, action_idx, override_val)
                        else:
                            policy_logits[0, :].fill_(-50.0)
                            policy_logits[0].scatter_(0, action_idx, override_val)

            sparse_concepts, _ = base_agent_core.sae(worker_context)
            self.last_ether_emission_batch = sparse_concepts.detach()

            if policy_logits.dim() == 3:
                immediate_logits = policy_logits[:, 0, : base_agent_core.num_actions].clone()
            else:
                immediate_logits = policy_logits[..., : base_agent_core.num_actions].clone()
            safe_logits = base_agent_core.causal_masker(moe_context, immediate_logits)

            jepa_surprisal_raw = getattr(base_agent_core.jepa, "proxy_surprisal", 0.0)
            if isinstance(jepa_surprisal_raw, torch.Tensor):
                jepa_surprisal_val = jepa_surprisal_raw.mean()
            else:
                jepa_surprisal_val = torch.tensor(float(jepa_surprisal_raw), device=device)

            if not hasattr(base_agent_core, "jepa_surprisal_ema_tensor"):
                base_agent_core.jepa_surprisal_ema_tensor = jepa_surprisal_val.clone()
            else:
                base_agent_core.jepa_surprisal_ema_tensor.lerp_(jepa_surprisal_val, 0.05)

            if hasattr(base_agent_core, "opponent_model") and hasattr(self, "last_actions_batch"):
                padded_actions = F.pad(
                    self.last_actions_batch, (0, 256 - base_agent_core.num_actions), "constant", 0.0
                )
                self_introspection = base_agent_core.opponent_model(padded_actions)
                meta_modulation = torch.tanh(self_introspection.mean()) * 0.5
            else:
                meta_modulation = torch.tensor(0.0, device=device)

            somatic_stress_global = somatic_surprisal.mean()

            if (
                getattr(self, "prev_state_features_batch", None) is not None
                and getattr(self, "last_state_features_batch", None) is not None
                and self.prev_state_features_batch.size(0) == latent_context.size(0)
            ):
                live_surprisal = F.mse_loss(latent_context, self.prev_state_features_batch)
            else:
                live_surprisal = torch.tensor(0.05, device=device)
            safe_surprisal = torch.nan_to_num(live_surprisal, nan=0.0, posinf=10.0, neginf=-10.0)
            if hasattr(base_agent_core, "exploration_layer"):

                evolution_amortizer = 1.0 / (1.0 + getattr(self, "generation", 0) / 20000.0)
                boredom_drive = torch.exp(-safe_surprisal * 5.0)
                panic_drive = torch.clamp(somatic_stress_global, max=5.0)

                surprisal_hook = (2.0 * boredom_drive + 1.0 * panic_drive) * evolution_amortizer
                target_temp = 0.1 + surprisal_hook

                current_temp = base_agent_core.exploration_layer.temperature
                new_temp = 0.95 * current_temp + 0.05 * target_temp

                temp_attr = base_agent_core.exploration_layer.temperature
                if isinstance(temp_attr, torch.Tensor):
                    temp_attr.copy_(new_temp)
                else:
                    setattr(base_agent_core.exploration_layer, "temperature", float(new_temp))

                safe_cognitive_stress = somatic_stress_global
                sys_stress = 1.5 * torch.tanh((0.5 * safe_surprisal) + (0.5 * safe_cognitive_stress))

                cyclic_curiosity = torch.sin(base_agent_core.step_counter.float() / 200.0) * 0.3

                dynamic_temp = torch.clamp(
                    0.1
                    + (safe_cognitive_stress * 0.5)
                    + surprisal_hook
                    - sys_stress
                    + meta_modulation
                    + cyclic_curiosity,
                    min=0.05,
                    max=10.0,
                )
                _ = torch.log(torch.clamp(dynamic_temp, min=0.05))
            else:
                sys_stress = 1.5 * torch.tanh((0.5 * safe_surprisal) + (0.5 * somatic_stress_global))
                cyclic_curiosity = torch.tensor(0.0, device=device)
                dynamic_temp = torch.clamp(
                    0.1
                    + (somatic_stress_global * 0.5)
                    + (2.0 * torch.exp(-safe_surprisal * 5.0))
                    - sys_stress
                    + meta_modulation
                    + cyclic_curiosity,
                    min=0.05,
                    max=10.0,
                )
                _ = torch.log(torch.clamp(dynamic_temp, min=0.05))

            safe_logits.sub_(torch.max(safe_logits, dim=-1, keepdim=True)[0])
            scaled_logits = safe_logits.div_(dynamic_temp)

            if hasattr(self, "stagnation_counters") and getattr(self, "_local_py_step", 1) % 50 == 0:
                with torch.no_grad():
                    if hasattr(base_agent_core, "causal_symbolic_reasoner") and hasattr(
                        base_agent_core.causal_symbolic_reasoner.dynamics_net[-1], "weight"
                    ):
                        stagnation_condition = self.stagnation_counters.max() > 50
                        stagnation_mask = stagnation_condition.float()
                        stagnation_noise = (
                            torch.randn_like(base_agent_core.causal_symbolic_reasoner.dynamics_net[-1].weight)
                            * 0.15
                            * stagnation_mask
                        )
                        base_agent_core.causal_symbolic_reasoner.dynamics_net[-1].weight.add_(stagnation_noise)
                        if stagnation_condition:
                            self.stagnation_counters.zero_()

            scaled_logits = torch.clamp(
                torch.nan_to_num(scaled_logits, nan=-50.0, posinf=50.0, neginf=-50.0), min=-50.0, max=50.0
            )

            intervention_condition = (dag_uncertainty > 2.0) & (torch.rand(1, device=device) < 0.15)
            final_logits = scaled_logits.clone()

            safe_interventions = torch.nan_to_num(final_logits[:, 8:12], nan=-10.0, posinf=10.0, neginf=-10.0)
            intervention_added = safe_interventions + 5.0
            final_logits[:, 8:12] = torch.where(intervention_condition, intervention_added, final_logits[:, 8:12])
            dist = torch.distributions.Categorical(logits=final_logits)

            sampled_actions = dist.sample()
            sampled_log_probs = dist.log_prob(sampled_actions)

            actions = sampled_actions
            log_probs = sampled_log_probs

            self.last_state_features_batch = latent_context.detach()
            self.last_fused_context_batch = fused_context.detach()

            if hasattr(self, "prev_state_features_batch") and self.prev_state_features_batch.size(
                0
            ) == self.last_state_features_batch.size(0):
                batch_tool_mask = (actions >= 8).float().unsqueeze(-1)
                base_agent_core.causal_validator.validate_batch(
                    latent_context.detach(),
                    env_state_before=self.prev_state_features_batch,
                    env_state_after=self.last_state_features_batch,
                    action_mask=batch_tool_mask,
                )

            clamped_actions = torch.clamp(actions.long(), 0, base_agent_core.num_actions - 1)
            action_one_hot = F.one_hot(clamped_actions, num_classes=base_agent_core.num_actions).float()

            # Compute Central Pattern Generator (CPG) offsets via parameterized Fourier features.
            cpg_amplitude = torch.sigmoid(immediate_logits[:, 0:3]) * 2.0
            cpg_phase = torch.tanh(immediate_logits[:, 3:6]) * math.pi
            cpg_offset = torch.tanh(immediate_logits[:, 6:9]) * 3.0
            cpg_time = base_agent_core.step_counter.float() * 0.1

            if hasattr(self, "world_ref") and getattr(self, "_local_py_step", 1) % 50 == 0:
                if not hasattr(self.world_ref, "_avg_phi_buffer"):
                    self.world_ref._avg_phi_buffer = []

                batch_variance = latent_context.float().var(dim=0, unbiased=False) + 1e-8
                p_eig = batch_variance / batch_variance.sum()
                true_phi = -(p_eig * torch.log(p_eig)).sum()
                self.world_ref._avg_phi_buffer.append(true_phi)

                if not hasattr(self.world_ref, "_diversity_buffer"):
                    self.world_ref._diversity_buffer = []
                self.world_ref._diversity_buffer.append(latent_context.float().std(dim=0, unbiased=False).mean())

            fourier_kinematics = cpg_offset + (cpg_amplitude * torch.sin(cpg_time.unsqueeze(-1) + cpg_phase))

            spatial_shifts = torch.zeros(batch_size, 3, device=device)
            kinematic_mask = (actions < 8).float().unsqueeze(-1)
            spatial_shifts[:, :3] = fourier_kinematics * kinematic_mask
            spatial_shifts = torch.nan_to_num(spatial_shifts, nan=0.0, posinf=1.0, neginf=-1.0)
            physical_embedding = F.pad(spatial_shifts, (0, 29), "constant", 0.1)
            self.proprioceptive_states = (self.proprioceptive_states * 0.8) + (physical_embedding * 0.2)

            tool_use_mask = ((actions >= 8) & (actions <= 11)).to(actions.dtype)
            self.last_damping_actions = actions * tool_use_mask

            if hasattr(self, "world_ref") and self.world_ref.grid.shape[-1] >= 16:
                active_tool_mask = tool_use_mask > 0
                tx = self.positions[active_tool_mask, 0].clamp(0, self.world_dim[0] - 1).long()
                ty = self.positions[active_tool_mask, 1].clamp(0, self.world_dim[1] - 1).long()
                tz = self.positions[active_tool_mask, 2].clamp(0, self.world_dim[2] - 1).long()

                self.world_ref.grid[..., 15].index_put_(
                    (tx, ty, tz), torch.full_like(tx, 1.5, dtype=self.world_ref.grid.dtype), accumulate=True
                )
                self.energies.masked_scatter_(
                    active_tool_mask, torch.clamp(self.energies[active_tool_mask] - 0.15, min=0.0)
                )

            self.replication_intents = torch.zeros_like(self.active_mask)
            if getattr(self, "_local_py_step", 1) % 10 == 0:
                base_replication_mask = actions == 12
                if base_replication_mask.any():
                    with torch.no_grad():
                        positions_flat = self.positions[:, :3].float()
                        distances = torch.cdist(positions_flat, positions_flat, p=2.0)
                        local_neighbor_density = (distances < 5.0).sum(dim=-1).float() - 1.0
                        replication_energy_cost = DESIRED_ENERGY * (1.0 + 2.0 * local_neighbor_density)

                    replication_mask = base_replication_mask & (self.energies > replication_energy_cost)
                    self.last_replication_count_tensor = replication_mask.sum()

                    rep_count = self.last_replication_count_tensor.item()
                    if rep_count > 0 and self.entities:
                        self.last_replication_count = rep_count
                        sorted_agents = sorted(
                            self.entities, key=lambda a: float(getattr(a, "fitness", 0.0)), reverse=True
                        )
                        if len(sorted_agents) > 10:
                            best_agent = sorted_agents[0]
                            worst_agents = sorted_agents[-int(len(sorted_agents) * 0.1) :]
                            tau = 0.05
                            for worst in worst_agents:
                                if hasattr(worst, "agent_core") and hasattr(best_agent, "agent_core"):
                                    diversity_bonus = (
                                        torch.norm(worst.agent_core.exploration_layer.temperature)
                                        if hasattr(worst.agent_core, "exploration_layer")
                                        else torch.tensor(1.0)
                                    )
                                    dynamic_tau = tau * (1.0 / (1.0 + diversity_bonus.item()))
                                    if (
                                        hasattr(self, "population_gamma")
                                        and hasattr(worst, "idx")
                                        and hasattr(best_agent, "idx")
                                    ):
                                        self.population_gamma[worst.idx].lerp_(
                                            self.population_gamma[best_agent.idx], dynamic_tau
                                        ).add_(torch.randn_like(self.population_gamma[worst.idx]) * 0.01)
                                        self.population_beta[worst.idx].lerp_(
                                            self.population_beta[best_agent.idx], dynamic_tau
                                        ).add_(torch.randn_like(self.population_beta[worst.idx]) * 0.01)

                    self.replication_intents = replication_mask

            self.velocities[:, :3] = (
                torch.clamp(self.velocities[:, :3] + spatial_shifts[:, :3] * 0.1, min=-5.0, max=5.0) * 0.95
            )
            max_bounds = torch.tensor(WORLD_DIM[:3], device=MODEL_DEVICE, dtype=torch.float32)
            self.positions[:, :3] = torch.remainder(self.positions[:, :3] + self.velocities[:, :3], max_bounds)
            self.last_actions_batch = action_one_hot.detach()

            if self.visited_spatial_grid.dtype != torch.float32:
                self.visited_spatial_grid = self.visited_spatial_grid.float()

            self.visited_spatial_grid *= 0.995

            active_idx = torch.where(self.active_mask[:batch_size])[0]
            if active_idx.numel() > 0:
                _px = self.positions[active_idx, 0].long().clamp(0, WORLD_DIM[0] - 1)
                _py = self.positions[active_idx, 1].long().clamp(0, WORLD_DIM[1] - 1)
                _pz = self.positions[active_idx, 2].long().clamp(0, WORLD_DIM[2] - 1)

                novelty_mask = ~self.permanent_visited_grid[_px, _py, _pz]
                self.fitness[active_idx] += novelty_mask.float() * 5.0

                if not hasattr(self, "total_voxels_visited_count"):
                    self.register_buffer("total_voxels_visited_count", torch.tensor(0.0, device=self.positions.device))
                self.total_voxels_visited_count += active_idx.numel()

                self.visited_spatial_grid[_px, _py, _pz] = 1.0
                self.permanent_visited_grid[_px, _py, _pz] = True

            kinematic_delta = torch.clamp(torch.norm(spatial_shifts, p=2, dim=-1), max=10.0)
            self.energies[:batch_size] = torch.clamp(
                self.energies[:batch_size] - (kinematic_delta * 0.05), max=float(DESIRED_ENERGY * 10.0)
            )  # type: ignore[assignment]

            if not CFG.STRICT_EX_NIHILO:
                base_agent_core.cluster_layer.update_correlations(self.last_ether_emission_batch, actions)

                attention_magnitude = torch.norm(self.last_ether_emission_batch, p=1, dim=-1)
                active_semantic_mask = attention_magnitude > 0.5

                if hasattr(self, "world_ref") and hasattr(self.world_ref, "semantic_broadcast_flag"):
                    ex = self.positions[active_semantic_mask, 0].clamp(0, WORLD_DIM[0] - 1).long()
                    ey = self.positions[active_semantic_mask, 1].clamp(0, WORLD_DIM[1] - 1).long()
                    ez = self.positions[active_semantic_mask, 2].clamp(0, WORLD_DIM[2] - 1).long()

                    self.world_ref.grid[..., 16].index_put_(
                        (ex, ey, ez), attention_magnitude[active_semantic_mask] * 0.5, accumulate=True
                    )
                    self.energies.masked_scatter_(
                        active_semantic_mask,
                        torch.clamp(
                            self.energies[active_semantic_mask] - (attention_magnitude[active_semantic_mask] * 0.1),
                            min=0.0,
                        ),
                    )

                    max_idx = torch.argmax(attention_magnitude)
                    broadcast_flag_val = (
                        self.world_ref.semantic_broadcast_flag.item()
                        if hasattr(self.world_ref.semantic_broadcast_flag, "item")
                        else self.world_ref.semantic_broadcast_flag
                    )
                    broadcast_available = broadcast_flag_val == 0

                    if hasattr(self.world_ref, "message_buffer") and broadcast_available:
                        if attention_magnitude[max_idx].item() > 5.0:
                            self.world_ref.message_buffer.copy_(self.last_ether_emission_batch[max_idx])

                            if hasattr(self.world_ref.semantic_broadcast_flag, "fill_"):
                                self.world_ref.semantic_broadcast_flag.fill_(1)
                            else:
                                self.world_ref.semantic_broadcast_flag = 1

        return actions, log_probs

    def action_cost_update_batch(self) -> torch.Tensor:
        self._ensure_capacity_buffers(self.max_agents, self.energies.device)
        self._reusable_inactive_mask.zero_()

        with torch.no_grad():
            baseline_energy_cost = 0.001

            velocity_norm = torch.linalg.vector_norm(self.velocities.float(), dim=-1) / 1.73205080757
            idle_threshold = 0.05
            kinetic_stagnation = torch.clamp(1.0 - (velocity_norm / idle_threshold), min=0.0)
            population_center = self.positions[:, :3].mean(dim=0, keepdim=True)
            spatial_density_penalty = (
                torch.sum((self.positions[:, :3] - population_center).pow(2), dim=-1) < 4.0
            ).float() * 0.02

            if hasattr(self, "ages"):
                grace_mask = (self.ages < 15).float()
                idle_penalty = kinetic_stagnation * 0.02 * (1.0 - grace_mask)
            else:
                idle_penalty = kinetic_stagnation * 0.02

            raw_step_penalty = baseline_energy_cost + (spatial_density_penalty * 0.5) + idle_penalty
            mastery_gate = torch.sigmoid((self.ts_mu - 25.0) / 5.0).to(self.energies.device)

            pos_x = self.positions[:, 0].long().clamp(0, WORLD_DIM[0] - 1)
            pos_y = self.positions[:, 1].long().clamp(0, WORLD_DIM[1] - 1)
            pos_z = self.positions[:, 2].long().clamp(0, WORLD_DIM[2] - 1)
            tax_mult = metabolic_curriculum.penalty_grid[pos_x, pos_y, pos_z]

            if hasattr(self, "world_ref") and self.world_ref.grid.shape[-1] >= 17:
                semantic_density = self.world_ref.grid[pos_x, pos_y, pos_z, 16]
                catalyst_density = self.world_ref.grid[pos_x, pos_y, pos_z, 15]
                friction_reducer = 1.0 / (1.0 + semantic_density * 2.5)
                tax_mult = tax_mult * torch.exp(-catalyst_density * 0.6)
            else:
                friction_reducer = 1.0

            step_penalty = raw_step_penalty * (0.05 + 0.95 * mastery_gate) * tax_mult * friction_reducer

            if getattr(self, "_cached_max_bounds", None) is None:
                self._cached_max_bounds = torch.tensor(
                    [WORLD_DIM[0] - 1, WORLD_DIM[1] - 1, WORLD_DIM[2] - 1],
                    device=self.positions.device,
                    dtype=self.positions.dtype,
                )
                self._cached_min_bounds = torch.zeros_like(self._cached_max_bounds)
            self.positions[:, :3] = torch.clamp(
                self.positions[:, :3], min=self._cached_min_bounds, max=self._cached_max_bounds
            )

            if not hasattr(self, "energy_intake"):
                self.register_buffer("energy_intake", torch.zeros_like(self.energies))
            current_energy_intake = self.energy_intake.clone()

            curiosity_bonus = 1.0 / (1.0 + self.visited_spatial_grid[pos_x, pos_y, pos_z] * 10.0)
            current_energy_intake += curiosity_bonus * 0.25

            stagnation_penalty = self.visited_spatial_grid[pos_x, pos_y, pos_z] * 0.1
            step_penalty += stagnation_penalty
            self.last_energy_intake_mean = current_energy_intake.mean().item()

            safe_penalty = torch.clamp(step_penalty, min=1e-4)
            instant_roi = current_energy_intake / safe_penalty

            if not hasattr(self, "smoothed_intake_ema"):
                self.register_buffer("smoothed_intake_ema", torch.ones_like(current_energy_intake))
                self.register_buffer("smoothed_penalty_ema", torch.ones_like(safe_penalty))

            self.smoothed_intake_ema = 0.95 * self.smoothed_intake_ema + 0.05 * current_energy_intake
            self.smoothed_penalty_ema = 0.95 * self.smoothed_penalty_ema + 0.05 * safe_penalty
            instant_roi = current_energy_intake / torch.clamp(safe_penalty, min=0.05)

            self.energies.sub_(step_penalty).add_(current_energy_intake)
            delta_h = (0.001 * torch.exp(tax_mult)) + (step_penalty * 0.05)
            self.hps.sub_(delta_h).add_(current_energy_intake * 0.01).clamp_(0.0, 1.0)

            continuous_band = self.hps * 2.0
            self.health_bands = torch.clamp(continuous_band.long(), min=0, max=2)

            if hasattr(self, "agent_cores") and len(self.agent_cores) > 0:
                roi_val = (
                    instant_roi[self.active_mask].mean()
                    if self.active_mask.any()
                    else torch.tensor(0.0, device=self.energies.device)
                )
                hp_val = self.hps.mean()
                band_val = (
                    continuous_band[self.active_mask].mean()
                    if self.active_mask.any()
                    else torch.tensor(0.0, device=self.energies.device)
                )

                self.agent_cores[0].last_metabolic_roi = roi_val
                self.agent_cores[0].health_score_proxy = hp_val
                self.agent_cores[0].health_band_proxy = band_val

            self.energy_intake.zero_()

            exploration_bonus = torch.zeros_like(self.fitness)
            rl_returns_bonus = torch.zeros_like(self.fitness)
            agents_valid = getattr(self, "entities", None) is not None and len(self.entities) > 0
            if agents_valid:
                td_errors = torch.ones(len(self.entities), device=self.fitness.device)
                returns_vals = torch.zeros(len(self.entities), device=self.fitness.device)
                for idx, agent in enumerate(self.entities):
                    if hasattr(agent, "experience_buffer") and agent.experience_buffer.ptr > 0:
                        ptr = agent.experience_buffer.ptr
                        td_errors[idx] = torch.abs(agent.experience_buffer.advantages[:ptr]).mean()
                        returns_vals[idx] = agent.experience_buffer.returns[:ptr].mean()
                rl_returns_bonus[: len(self.entities)] = returns_vals
                exploration_bonus[: len(self.entities)] = torch.exp(-td_errors) * 5.0
            if not hasattr(self, "intrinsic_reward"):
                self.intrinsic_reward = torch.zeros_like(self.fitness)
            self.intrinsic_reward += exploration_bonus
            self.fitness += (
                (current_energy_intake * 5.0) - (step_penalty * 2.0) + rl_returns_bonus + (velocity_norm * 2.0)
            )

            if hasattr(self, "world_ref") and self.world_ref.grid.shape[-1] >= 16:
                local_crystal = self.world_ref.grid[pos_x, pos_y, pos_z, 15]
                self.fitness -= local_crystal * 0.1
                crystal_grid = self.world_ref.grid[..., 15]
                current_ratio = (crystal_grid > 0.1).float().mean()
                target_ratio = 0.15
                growth_factor = 1.0 + (target_ratio - current_ratio) * 0.05
                critical_collapse = torch.where(crystal_grid > 0.8, 0.5, 1.0)
                self.world_ref.grid[..., 15] = torch.clamp(
                    crystal_grid * critical_collapse * 0.95 * growth_factor, 0.0, 1.0
                )

            done_mask = (self.energies <= 0.0) & self.active_mask
            num_dead = int(done_mask.sum().item())

            if num_dead > 0:
                if hasattr(self, "entities"):
                    safe_agents_count = max(1, getattr(self, "max_agents", 128))
                    self.genetic_turnover_rate = num_dead / safe_agents_count  # type: ignore[assignment]
                    dead_indices = torch.where(done_mask)[0]
                    for idx in dead_indices:
                        if int(idx) < len(self.entities):
                            agent = self.entities[int(idx)]
                            if hasattr(agent, "experience_buffer") and hasattr(agent.experience_buffer, "states"):
                                ptr = agent.experience_buffer.ptr
                                if ptr > 0:
                                    final_pos = self.positions[idx, :3].clone()
                                    target_latent = agent.experience_buffer.states[ptr - 1].clone()
                                    valid_states = agent.experience_buffer.states[:ptr]
                                    distances = torch.norm(valid_states - target_latent, p=2, dim=-1)
                                    her_rewards = torch.exp(-distances) * 10.0
                                    if hasattr(agent.experience_buffer, "her_goals"):
                                        agent.experience_buffer.her_goals[:ptr] = final_pos
                                    agent.experience_buffer.returns[:ptr] += her_rewards

                if not hasattr(self, "fitness_history"):
                    self.fitness_history = []
                self.fitness_history.extend(self.fitness[done_mask].tolist())
                if len(self.fitness_history) > 1000:
                    self.fitness_history = self.fitness_history[-1000:]

            if hasattr(self, "world_ref"):
                if not hasattr(self.world_ref, "death_window"):
                    import collections

                    self.world_ref.death_window = collections.deque(maxlen=100)
                self.world_ref.death_window.append(float(num_dead))
                total_deaths = sum(self.world_ref.death_window)
                self.world_ref.genetic_turnover_rate = float(total_deaths) / max(
                    1.0, float(self.max_agents) * len(self.world_ref.death_window)
                )  # type: ignore[assignment]

            if hasattr(self, "entities") and len(self.entities) > 0:
                deaths_grid = torch.zeros((WORLD_DIM[0], WORLD_DIM[1], WORLD_DIM[2]), device=MODEL_DEVICE)
                population_grid = torch.zeros((WORLD_DIM[0], WORLD_DIM[1], WORLD_DIM[2]), device=MODEL_DEVICE)

                valid_mask = self.active_mask
                valid_pos_x = self.positions[valid_mask, 0].long().clamp(0, WORLD_DIM[0] - 1)
                valid_pos_y = self.positions[valid_mask, 1].long().clamp(0, WORLD_DIM[1] - 1)
                valid_pos_z = self.positions[valid_mask, 2].long().clamp(0, WORLD_DIM[2] - 1)

                ones = torch.ones_like(valid_pos_x, dtype=torch.float32)
                population_grid.index_put_((valid_pos_x, valid_pos_y, valid_pos_z), ones, accumulate=True)

                dead_mask = done_mask & valid_mask
                dead_pos_x = self.positions[dead_mask, 0].long().clamp(0, WORLD_DIM[0] - 1)
                dead_pos_y = self.positions[dead_mask, 1].long().clamp(0, WORLD_DIM[1] - 1)
                dead_pos_z = self.positions[dead_mask, 2].long().clamp(0, WORLD_DIM[2] - 1)
                dead_ones = torch.ones_like(dead_pos_x, dtype=torch.float32)
                deaths_grid.index_put_((dead_pos_x, dead_pos_y, dead_pos_z), dead_ones, accumulate=True)

                metabolic_curriculum.compute_spatial_multiplier(deaths_grid, population_grid)

            if num_dead > 0:
                world_limits = torch.tensor(WORLD_DIM, dtype=torch.float32, device=MODEL_DEVICE)
                new_positions = (torch.rand((num_dead, len(WORLD_DIM)), device=MODEL_DEVICE) * world_limits).long()
                self.positions[done_mask, : new_positions.size(-1)] = new_positions.float()

                self.energies.masked_fill_(done_mask, float(DESIRED_ENERGY))
                self.hps.masked_fill_(done_mask, 1.0)
                self.fitness.masked_fill_(done_mask, 0.0)
                self.velocities.masked_fill_(done_mask.unsqueeze(-1), 0.0)

                done_mask.fill_(False)

            cognitive_penalty = 0.0
            if wandb.run is not None:
                current_rank = float(wandb.run.summary.get("metrics/latent_matrix_rank", 10.0))
                moe_ortho = float(wandb.run.summary.get("moe/expert_orthogonality", 1.0))

                if current_rank < 5.0:
                    cognitive_penalty += 0.5
                if moe_ortho < 0.1:
                    cognitive_penalty += 0.5

            self.fitness.sub_(cognitive_penalty).clamp_(min=-100.0)

        return done_mask


class ReplayBuffer:
    """
    Fixed-size FIFO experience replay buffer.

    Args:
        max_capacity (int): Maximum transition count.
        state_dim (int): State vector dimension.
        action_dim (int): Action vector dimension.
        device (torch.device): Target execution device.
    """

    def __init__(self, max_capacity, state_dim, action_dim, device):
        self.max_capacity = max_capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.states = torch.zeros((max_capacity, state_dim), dtype=torch.float16, device="cpu")
        self.actions = torch.zeros((max_capacity, action_dim), dtype=torch.int16, device="cpu")

        self.returns = torch.zeros((max_capacity,), dtype=torch.float32, device="cpu")
        self.advantages = torch.zeros((max_capacity,), dtype=torch.float32, device="cpu")
        self.log_probs = torch.zeros((max_capacity,), dtype=torch.float32, device="cpu")
        self.next_states = torch.zeros((max_capacity, state_dim), dtype=torch.float16, device="cpu")

    def __len__(self):
        return self.size

    def extend(self, states, actions, returns, advantages, log_probs, next_states):
        batch_size = states.size(0)
        if batch_size > self.max_capacity:
            states = states[-self.max_capacity :]
            actions = actions[-self.max_capacity :]
            returns = returns[-self.max_capacity :]
            advantages = advantages[-self.max_capacity :]
            log_probs = log_probs[-self.max_capacity :]
            next_states = next_states[-self.max_capacity :]
            batch_size = self.max_capacity

        sources = [
            (self.states, states, torch.float16),
            (self.actions, actions.view(-1, self.actions.size(1) if actions.dim() > 1 else 1), torch.int16),
            (self.returns, returns, torch.float32),
            (self.advantages, advantages, torch.float32),
            (self.log_probs, log_probs, torch.float32),
            (self.next_states, next_states, torch.float16),
        ]

        end_ptr = self.ptr + batch_size
        with torch.no_grad():
            if end_ptr <= self.max_capacity:
                for dst, src, dtype in sources:
                    dst[self.ptr : end_ptr].copy_(src.to(dtype), non_blocking=True)
            else:
                overflow = end_ptr - self.max_capacity
                first_part = batch_size - overflow
                for dst, src, dtype in sources:
                    cast_src = src.to(dtype)
                    dst[self.ptr :].copy_(cast_src[:first_part], non_blocking=True)
                    dst[:overflow].copy_(cast_src[first_part:], non_blocking=True)

        self.ptr = end_ptr % self.max_capacity
        self.size = min(self.size + batch_size, self.max_capacity)

    def sample(self, batch_size):
        if self.size == 0:
            return (
                torch.empty((0, self.states.size(1)), dtype=self.states.dtype, device=self.device),
                torch.empty((0, self.actions.size(1)), dtype=self.actions.dtype, device=self.device),
                torch.empty((0,), dtype=self.returns.dtype, device=self.device),
                torch.empty((0,), dtype=self.advantages.dtype, device=self.device),
                torch.empty((0,), dtype=self.log_probs.dtype, device=self.device),
                torch.empty((0, self.next_states.size(1)), dtype=self.next_states.dtype, device=self.device),
            )

        safe_batch_size = min(batch_size, self.size)

        if safe_batch_size < self.size // 4:
            indices = torch.randint(0, self.size, (safe_batch_size,), device="cpu")
        else:
            indices = torch.randperm(self.size, device="cpu")[:safe_batch_size]

        return (
            self.states[indices].to(self.device, non_blocking=True),
            self.actions[indices].to(self.device, non_blocking=True),
            self.returns[indices].to(self.device, non_blocking=True),
            self.advantages[indices].to(self.device, non_blocking=True),
            self.log_probs[indices].to(self.device, non_blocking=True),
            self.next_states[indices].to(self.device, non_blocking=True),
        )


class Agent:
    """
    Vectorized entity state tracker.
    """

    def __init__(self, position=None, agent_core=None, pop_ref=None, index=0):
        self.pop_ref = pop_ref
        self.idx = index

        if pop_ref is None:
            self.id = uuid.uuid4().int
            if position is not None:
                self._pos_fallback = (
                    position.clone().detach().to(dtype=torch.long, device=CFG.MODEL_DEVICE)
                    if isinstance(position, torch.Tensor)
                    else torch.tensor(position, dtype=torch.long, device=CFG.MODEL_DEVICE)
                )
            else:
                self._pos_fallback = torch.zeros(4, dtype=torch.long, device=CFG.MODEL_DEVICE)
            self._energy_fallback = 1.5
            self._fitness_fallback = 0.0
            self._age_fallback = 0
            self._stagnation_fallback = 0
            self.discovered_features = []
            self.perception_shape = (7, 7, 7)
            self.full_perception_shape = (17, 7, 7, 7)

            if agent_core is not None:
                self.agent_core = agent_core.to(CFG.MODEL_DEVICE)
            elif pop_ref is not None:
                self.agent_core = pop_ref.global_agent_core
            else:
                from vrl_framework.models.agents import RLAgent

                self.agent_core = RLAgent(sensory_input_shape=self.full_perception_shape, num_actions=17).to(
                    CFG.MODEL_DEVICE
                )

            self.comm_msg = None
            self._hidden_state_fallback = torch.randn(256, device=CFG.MODEL_DEVICE)
            self.hyperparams = {
                "mutation_rate": np.random.uniform(0.01, 0.05),
                "exploration_noise": np.random.uniform(0.1, 0.3),
                "baseline_energy_cost": np.random.uniform(0.7, 1.3),
            }
            self.prev_fitness = 0.0
            self.stagnation_counter = 0
            self.has_replicated = False
            self.total_generation = 0
            self.experience_buffer = ReplayBuffer(
                max_capacity=2048, state_dim=256, action_dim=1, device=CFG.MODEL_DEVICE
            )
            self.last_state_features = None
            self.last_intent = torch.zeros(256, device=CFG.MODEL_DEVICE)

    @property
    def position(self):
        return self.pop_ref.positions[self.idx] if self.pop_ref else self._pos_fallback

    @position.setter
    def position(self, val):
        if self.pop_ref:
            self.pop_ref.positions[self.idx].copy_(torch.as_tensor(val, device=self.pop_ref.positions.device))
        else:
            self._pos_fallback = val

    @property
    def energy(self) -> Any:
        return (
            self.pop_ref.energies[self.idx]
            if self.pop_ref
            else torch.tensor(self._energy_fallback, device=CFG.MODEL_DEVICE)
        )

    @energy.setter
    def energy(self, val):
        if self.pop_ref:
            self.pop_ref.energies[self.idx] = val
        else:
            self._energy_fallback = val

    @property
    def mu(self) -> Any:
        return self.pop_ref.ts_mu[self.idx] if self.pop_ref else torch.tensor(25.0, device=CFG.MODEL_DEVICE)

    @mu.setter
    def mu(self, val):
        if self.pop_ref:
            self.pop_ref.ts_mu[self.idx] = val

    @property
    def sigma(self) -> Any:
        return self.pop_ref.ts_sigma[self.idx] if self.pop_ref else torch.tensor(25.0 / 3.0, device=CFG.MODEL_DEVICE)

    @sigma.setter
    def sigma(self, val):
        if self.pop_ref:
            self.pop_ref.ts_sigma[self.idx] = val

    @property
    def fitness(self) -> Any:
        return (
            self.pop_ref.fitness[self.idx]
            if self.pop_ref
            else torch.tensor(self._fitness_fallback, device=CFG.MODEL_DEVICE)
        )

    @fitness.setter
    def fitness(self, val):
        if self.pop_ref:
            self.pop_ref.fitness[self.idx] = val
        else:
            self._fitness_fallback = val

    @property
    def survival_steps(self):
        return (
            self.pop_ref.survival_steps[self.idx]
            if self.pop_ref
            else torch.tensor(self._age_fallback, device=CFG.MODEL_DEVICE)
        )

    @survival_steps.setter
    def survival_steps(self, val):
        if self.pop_ref:
            self.pop_ref.survival_steps[self.idx] = val
        else:
            self._age_fallback = val

    @property
    def stagnation_counter(self):
        return (
            self.pop_ref.stagnation_counters[self.idx]
            if self.pop_ref
            else torch.tensor(self._stagnation_fallback, device=CFG.MODEL_DEVICE)
        )

    @stagnation_counter.setter
    def stagnation_counter(self, val):
        if self.pop_ref:
            self.pop_ref.stagnation_counters[self.idx] = val
        else:
            self._stagnation_fallback = val

    @property
    def hidden_state(self):
        return self.pop_ref.hidden_states[self.idx] if self.pop_ref else self._hidden_state_fallback

    @hidden_state.setter
    def hidden_state(self, val):
        if self.pop_ref:
            self.pop_ref.hidden_states[self.idx] = val
        else:
            self._hidden_state_fallback = val

    def get_observation(self, environment) -> torch.Tensor:
        """
        Extracts local spatial observation slice.
        """
        if self.pop_ref:
            pos_vec = self.pop_ref.positions[self.idx, :3].clamp(min=0).long()
            bounds = torch.tensor(self.pop_ref.world_dim[:3], device=pos_vec.device)
            pos_vec = torch.min(pos_vec, bounds - 1)
        else:
            pos_vec = self.position[:3].clamp(min=0).long()
            bounds = torch.tensor([WORLD_DIM[0], WORLD_DIM[1], WORLD_DIM[2]], device=pos_vec.device)
            pos_vec = torch.min(pos_vec, bounds - 1)
        bounds = torch.tensor([WORLD_DIM[0], WORLD_DIM[1], WORLD_DIM[2]], device=pos_vec.device)
        pos_vec = torch.min(pos_vec, bounds - 1)

        c_x, c_y, c_z = pos_vec[0], pos_vec[1], pos_vec[2]

        l_x, u_x = max(0, c_x.item() - 3), min(bounds[0].item(), c_x.item() + 4)
        l_y, u_y = max(0, c_y.item() - 3), min(bounds[1].item(), c_y.item() + 4)
        l_z, u_z = max(0, c_z.item() - 3), min(bounds[2].item(), c_z.item() + 4)

        obs_slice = environment.grid[l_x:u_x, l_y:u_y, l_z:u_z, :]
        obs_slice = obs_slice.permute(3, 0, 1, 2).unsqueeze(0).float()

        # Pad the observation slice to strictly match the expected sensory input topology.
        pad_x_l, pad_x_h = int(max(0, 3 - c_x.item())), int(max(0, c_x.item() + 4 - bounds[0].item()))
        pad_y_l, pad_y_h = int(max(0, 3 - c_y.item())), int(max(0, c_y.item() + 4 - bounds[1].item()))
        pad_z_l, pad_z_h = int(max(0, 3 - c_z.item())), int(max(0, c_z.item() + 4 - bounds[2].item()))

        return F.pad(obs_slice, (pad_z_l, pad_z_h, pad_y_l, pad_y_h, pad_x_l, pad_x_h), mode="constant", value=0.0)

    def act(self, environment, external_signal=None):
        if self.energy.item() < 0.1:
            return None

        obs = self.get_observation(environment)

        audio_input = getattr(environment, "audio_grid", torch.zeros(1, 44100, device=MODEL_DEVICE)).expand(
            obs.size(0), 1, -1
        )

        agent_core_out = self.agent_core(obs, external_signal, audio=audio_input)
        logits = agent_core_out[0] if isinstance(agent_core_out, tuple) else agent_core_out

        action, log_prob = self.agent_core.exploration_layer(logits, deterministic=False)
        self.last_log_prob = log_prob
        return action

    def update_hidden_state(self):
        if not self.pop_ref:
            self.hidden_state.add_(torch.randn_like(self.hidden_state) * 0.01)

    def action_cost_update(self):
        """
        Computes step-wise metabolic cost.
        Cost = base_flops + (num_experts^1.5 * 0.001) + max(0, state_variance - 1.0) * 0.05
        """
        base_cost = 0.04
        expert_count = getattr(self.agent_core.moe, "num_experts", 8) if hasattr(self.agent_core, "moe") else 8
        base_cost = (
            self.agent_core.compute_flops_cost()
            if hasattr(self.agent_core, "compute_flops_cost")
            else self.metabolic_cost() if hasattr(self, "metabolic_cost") else 0.1
        )
        neural_cost = base_cost + (expert_count**1.5 * 0.001)

        # Penalize extreme variance in recurrent hidden states to enforce latent stability.
        state_variance = self.hidden_state.var().item()
        variance_penalty = max(0.0, state_variance - 1.0) * 0.05

        pos = self.position[:3].clamp(0, torch.tensor(WORLD_DIM[:3], device=self.position.device) - 1).long()
        tax_mult = metabolic_curriculum.penalty_grid[pos[0], pos[1], pos[2]]

        self.energy -= (base_cost + neural_cost + variance_penalty) * float(tax_mult)

        if not hasattr(self, "_traj_entropy"):
            self._traj_entropy = 1.0
            self._complexity_timer = 0

        self._complexity_timer += 1

        if self._complexity_timer >= 50:
            if hasattr(self.agent_core, "memory") and self.agent_core.memory.bank_ptr.item() > 2:
                valid_stm = self.agent_core.memory.episodic_bank[: self.agent_core.memory.bank_ptr.item()]
                self._traj_entropy = compute_traj_entropy(valid_stm).item()

                if hasattr(self.agent_core, "jepa"):
                    z_internal = valid_stm.mean(dim=0)
                    z_external = self.hidden_state
                    blanket_cross_cov = (z_internal - z_internal.mean()) * (z_external - z_external.mean())
                    integrated_information_reward = torch.clamp(blanket_cross_cov.pow(2).sum(), max=5.0)
                    self._traj_entropy += integrated_information_reward.item() * 0.1
            self._complexity_timer = 0

        if self._traj_entropy < 0.3:
            self.energy -= 0.1
            self.fitness -= 0.5
            if hasattr(self.agent_core, "exploration_layer"):
                self.agent_core.exploration_layer.temperature.data.fill_(2.0)

        if self.energy < DEACTIVATION_THRESHOLD:
            self.energy += RECOVERY_MARGIN
            self.fitness -= 2.0

        deviation_penalty = PENALTY_GAMMA * abs(self.energy - DESIRED_ENERGY)
        self.fitness -= deviation_penalty

        if self.energy < 0:
            if self.pop_ref and hasattr(self.pop_ref, "world_ref"):
                clamped_pos = torch.clamp(
                    self.position[:3],
                    torch.tensor(0, device=self.position.device),
                    torch.tensor([WORLD_DIM[0] - 1, WORLD_DIM[1] - 1, WORLD_DIM[2] - 1], device=self.position.device),
                ).long()
                self.pop_ref.world_ref.grid[clamped_pos[0], clamped_pos[1], clamped_pos[2], 0] += 1.0

            self.fitness -= 50.0
            self.energy = DESIRED_ENERGY

            new_pos = torch.tensor(
                [
                    float(torch.randint(0, WORLD_DIM[0], (1,)).item()),
                    float(torch.randint(0, WORLD_DIM[1], (1,)).item()),
                    float(torch.randint(0, WORLD_DIM[2], (1,)).item()),
                ],
                device=self.position.device,
            )
            self.position[:3] = new_pos

            self.fitness = max(-100.0, min(100.0, float(self.fitness)))

    def build_critic_context(
        self, latent_state: torch.Tensor, stm_tensor: torch.Tensor, causal_context: Any = None
    ) -> torch.Tensor:
        if causal_context is None:
            causal_context = torch.zeros_like(latent_state)
        critic_context = torch.cat([latent_state, stm_tensor, causal_context], dim=-1)
        target_dim = 768
        if critic_context.size(-1) < target_dim:
            pad_size = target_dim - critic_context.size(-1)
            critic_context = F.pad(critic_context, (0, pad_size))
        elif critic_context.size(-1) > target_dim:
            critic_context = critic_context[..., :target_dim]
        return critic_context

    def build_actorcritic_context(self, latent_state: torch.Tensor, stm_tensor: torch.Tensor) -> torch.Tensor:
        return self.build_critic_context(latent_state, stm_tensor)

    def update_post_action(self) -> None:
        self.comm_msg = None

    def perceive(self, environment) -> torch.Tensor:
        return self.get_observation(environment)

    def receive_message(self, signal: torch.Tensor) -> None:
        if not hasattr(self, "active_signals"):
            self.active_signals = []
        self.active_signals.append(signal.detach().clone())
        with torch.no_grad():
            if self.comm_msg is None:
                self.comm_msg = signal.clone().detach()
            else:
                query_dim = min(self.hidden_state.size(-1), signal.size(-1))
                query = self.hidden_state[:query_dim]
                score_existing = F.cosine_similarity(query, self.comm_msg[:query_dim], dim=-1)
                score_new = F.cosine_similarity(query, signal[:query_dim], dim=-1)
                weights = F.softmax(torch.stack([score_existing, score_new]) / 0.1, dim=0)
                self.comm_msg = (self.comm_msg * weights[0]) + (signal.detach() * weights[1])

    def receive_signal(self, signal: torch.Tensor) -> None:
        self.receive_message(signal)

    def sync_hidden_states(self, ent2) -> None:
        with torch.no_grad():
            self.hidden_state.copy_(self.hidden_state * 0.9 + ent2.hidden_state * 0.1)
            sparse_acts, _ = self.agent_core.sae(self.hidden_state)
            active_skills = sparse_acts > 0.05
            if not active_skills.any():
                return
            decoded_latent = self.agent_core.sae.decoder(sparse_acts)
            ent2.hidden_state = F.normalize(ent2.hidden_state + 0.15 * decoded_latent, p=2, dim=-1)
            if hasattr(self.agent_core, "feature_counter") and hasattr(
                self.agent_core.feature_counter, "concept_activation_counts"
            ):
                self.agent_core.feature_counter.concept_activation_counts += active_skills.float()
            if hasattr(ent2.agent_core, "episodic_memory") and hasattr(
                ent2.agent_core.episodic_memory, "store_experience"
            ):
                ent2.agent_core.episodic_memory.store_experience(
                    decoded_latent, torch.zeros(16, device=decoded_latent.device), ent2.hidden_state
                )

    def exchange_knowledge(self, other: "Agent") -> None:
        self.sync_hidden_states(other)
        other.sync_hidden_states(self)

    def imitate(self, other: "Agent") -> None:
        """
        Overwrites local network parameters and genetic dictionaries with those of the target agent.
        """
        if self.pop_ref is not None and other.pop_ref is not None:
            with torch.no_grad():
                self.pop_ref.population_gamma[self.idx].copy_(other.pop_ref.population_gamma[other.idx])
                self.pop_ref.population_beta[self.idx].copy_(other.pop_ref.population_beta[other.idx])
                self.pop_ref.population_masks[self.idx].copy_(other.pop_ref.population_masks[other.idx])
        else:
            state_dict_gpu = {k: v.to(self.agent_core.device) for k, v in other.agent_core.state_dict().items()}
            self.agent_core.load_state_dict(state_dict_gpu, strict=False)

        import copy

        self.genetic_code = copy.deepcopy(getattr(other, "genetic_code", {}))

    def transmit_message(self, other: "Agent") -> None:
        with torch.no_grad():
            if hasattr(self.agent_core, "communication"):
                my_signal = self.agent_core.communication.encode(self.hidden_state)
                other.receive_message(my_signal)

    def update_state(self, other: "Agent") -> None:
        with torch.no_grad():
            msg_A = (
                self.agent_core.communication.encode(self.hidden_state)
                if (hasattr(self.agent_core, "communication") and hasattr(self.agent_core.communication, "encode"))
                else self.hidden_state
            )

            if hasattr(other.agent_core, "relational_memory"):
                relational_context_B = other.agent_core.relational_memory.query(msg_A)
            else:
                relational_context_B = other.agent_core.fuzzy_kb._clean(msg_A)

            update_rate_B = torch.clamp(torch.sigmoid(other.hidden_state.var() - 1.0) * 0.2, min=0.01)
            other.hidden_state = F.normalize(
                other.hidden_state * (1.0 - update_rate_B) + update_rate_B * relational_context_B, p=2, dim=-1
            )

            msg_B = (
                other.agent_core.communication.encode(other.hidden_state)
                if (hasattr(other.agent_core, "communication") and hasattr(other.agent_core.communication, "encode"))
                else other.hidden_state
            )

            if hasattr(self.agent_core, "relational_memory"):
                relational_context_A = self.agent_core.relational_memory.query(msg_B)
            else:
                relational_context_A = self.agent_core.fuzzy_kb._clean(msg_B)

            score_diff_weight = max(1.0, 1.0 + (self.mu - other.mu) * 0.1)
            base_interpolation_rate = torch.sigmoid(self.hidden_state.var() - 1.0) * 0.2
            update_rate_A = torch.clamp(base_interpolation_rate * score_diff_weight, min=0.01, max=0.5)

            self.hidden_state = F.normalize(
                self.hidden_state * (1.0 - update_rate_A) + update_rate_A * relational_context_A, p=2, dim=-1
            )

    def crossover_parameters(self, neighbor):
        if random.random() < 0.3 and self.pop_ref is not None and neighbor.pop_ref is not None:
            with torch.no_grad():
                target_gamma = neighbor.pop_ref.population_gamma[neighbor.idx]
                target_beta = neighbor.pop_ref.population_beta[neighbor.idx]

                crossover_mask = (torch.rand_like(self.pop_ref.population_gamma[self.idx]) > 0.5).to(
                    target_gamma.dtype
                )
                blended_gamma = (
                    crossover_mask * target_gamma + (1.0 - crossover_mask) * self.pop_ref.population_gamma[self.idx]
                )
                blended_beta = (
                    crossover_mask * target_beta + (1.0 - crossover_mask) * self.pop_ref.population_beta[self.idx]
                )

                std_target_gamma = torch.std(target_gamma.float(), unbiased=False) + 1e-5
                std_blended_gamma = torch.std(blended_gamma.float(), unbiased=False) + 1e-5

                std_target_beta = torch.std(target_beta.float(), unbiased=False) + 1e-5
                std_blended_beta = torch.std(blended_beta.float(), unbiased=False) + 1e-5

                gamma_scale = torch.clamp(std_target_gamma / std_blended_gamma, min=0.1, max=10.0)
                beta_scale = torch.clamp(std_target_beta / std_blended_beta, min=0.1, max=10.0)

                corrected_gamma = (
                    blended_gamma.float() - torch.mean(blended_gamma.float())
                ) * gamma_scale + torch.mean(target_gamma.float())
                corrected_beta = (blended_beta.float() - torch.mean(blended_beta.float())) * beta_scale + torch.mean(
                    target_beta.float()
                )

                if not hasattr(self.pop_ref, "grad_ema_gamma"):
                    self.pop_ref.register_buffer(
                        "grad_ema_gamma", torch.zeros_like(self.pop_ref.population_gamma[self.idx])
                    )
                    self.pop_ref.register_buffer(
                        "grad_ema_beta", torch.zeros_like(self.pop_ref.population_beta[self.idx])
                    )

                delta_gamma = target_gamma - self.pop_ref.population_gamma[self.idx]
                delta_beta = target_beta - self.pop_ref.population_beta[self.idx]

                self.pop_ref.grad_ema_gamma = 0.9 * self.pop_ref.grad_ema_gamma + 0.1 * delta_gamma
                self.pop_ref.grad_ema_beta = 0.9 * self.pop_ref.grad_ema_beta + 0.1 * delta_beta

                noise_gamma = torch.randn_like(corrected_gamma) * 0.001
                noise_beta = torch.randn_like(corrected_beta) * 0.001

                norm_sq_gamma = torch.norm(corrected_gamma) ** 2 + 1e-8
                norm_sq_beta = torch.norm(corrected_beta) ** 2 + 1e-8
                noise_gamma -= (
                    torch.dot(noise_gamma.flatten(), corrected_gamma.flatten()) / norm_sq_gamma
                ) * corrected_gamma
                noise_beta -= (
                    torch.dot(noise_beta.flatten(), corrected_beta.flatten()) / norm_sq_beta
                ) * corrected_beta

                self.pop_ref.population_gamma[self.idx].copy_(
                    (corrected_gamma + self.pop_ref.grad_ema_gamma * 0.1 + noise_gamma).half()
                )
                self.pop_ref.population_beta[self.idx].copy_(
                    (corrected_beta + self.pop_ref.grad_ema_beta * 0.1 + noise_beta).half()
                )

                mask_crossover = torch.rand_like(self.pop_ref.population_masks[self.idx].float()) > 0.5
                self.pop_ref.population_masks[self.idx] = torch.where(
                    mask_crossover,
                    neighbor.pop_ref.population_masks[neighbor.idx],
                    self.pop_ref.population_masks[self.idx],
                )

    def replicate(self, world):
        """
        Instantiates a child agent with mutated hyperparameters and partitioned energy.
        """
        replica_pos = torch.as_tensor(self.position, device=MODEL_DEVICE).clone()

        cloned_agent = Agent(position=replica_pos, agent_core=self.agent_core)

        cloned_agent.origin_idx = self.idx
        cloned_agent.inherit_hyperparams = True
        cloned_agent.hyperparams = {
            "mutation_rate": max(
                0.001, min(0.1, self.hyperparams.get("mutation_rate", 0.02) * random.choice([0.9, 1.1]))
            ),
            "exploration_noise": max(
                0.01, min(0.5, self.hyperparams.get("exploration_noise", 0.1) * random.choice([0.9, 1.1]))
            ),
            "baseline_energy_cost": max(
                0.5, min(2.0, self.hyperparams.get("baseline_energy_cost", 1.0) * random.choice([0.9, 1.1]))
            ),
        }

        cloned_agent.energy = self.energy * 0.4
        self.energy *= 0.6
        self.has_replicated = True

        return cloned_agent

    def compute_entropy(self):
        """
        Evaluates whether the trajectory entropy exceeds the predefined viability threshold.
        """
        traj_entropy = compute_traj_entropy(self.hidden_state).item()
        return traj_entropy > 0.7

    def export_to_onnx(self, filepath: str = "agent_inference.onnx"):
        self.agent_core.eval()
        dummy_input = torch.randn(1, *self.perception_shape, device=CFG.MODEL_DEVICE)

        class InferenceWrapper(nn.Module):
            def __init__(self, sensory, jepa, actor_head):
                super().__init__()
                self.sensory = sensory
                self.jepa = jepa
                self.actor_head = actor_head

            def forward(self, x):
                features = self.sensory(x)
                jepa_out = self.jepa(features)
                latent = jepa_out[0] if isinstance(jepa_out, tuple) else jepa_out
                return self.actor_head(latent)

        base_ac = (
            self.agent_core.actor_critic.module
            if hasattr(self.agent_core.actor_critic, "module")
            else self.agent_core.actor_critic
        )
        wrapper = InferenceWrapper(self.agent_core.sensory, self.agent_core.jepa, base_ac.actor_head_discrete)
        torch.onnx.export(
            wrapper,
            (dummy_input,),
            filepath,
            export_params=True,
            opset_version=14,
            input_names=["sensory_input"],
            output_names=["action_logits"],
        )

        logging.info(f"Model successfully exported to ONNX: {filepath}")

    def __repr__(self):
        return f"LatentAgent(Vectorized-ID: {self.idx})"


class NoiseGenerator(torch.nn.Module):
    """
    Generates parametric spatial noise fields for environment domain randomization.
    """

    def __init__(self, world_dim):
        super().__init__()
        self.world_dim = world_dim
        self.adversary_net = torch.nn.Sequential(
            torch.nn.Linear(256, 128), torch.nn.Mish(), torch.nn.Linear(128, 2)
        ).to(MODEL_DEVICE)

    def generate_adversarial_mask(self, population_state_vector):
        """
        Maps global population state vectors to 3D spatial distortion fields.

        Args:
            population_state_vector (torch.Tensor): Tensor of shape (batch_size, hidden_dim).

        Returns:
            torch.Tensor: A 3D tensor representing local adversarial friction limits.
        """
        with torch.no_grad():
            distortion_params = self.adversary_net(population_state_vector.mean(dim=0))
            grid = torch.randn(self.world_dim, device=MODEL_DEVICE)
            freq = torch.sigmoid(distortion_params[0]) * 5.0
            phase = distortion_params[1] * math.pi

        spatial_distortion = torch.sin(grid * freq + phase)
        return spatial_distortion


class EvaluationEnv:
    """
    Isolated sandbox for evaluating mutated agents' kinematic limits.
    """

    def __init__(self, entity, environment):
        import copy

        self.ent = Agent(
            position=(
                entity.position.tolist() if isinstance(entity.position, torch.Tensor) else copy.copy(entity.position)
            )
        )

        state_dict_detached = {k: v.detach().clone() for k, v in entity.agent_core.state_dict().items()}
        from vrl_framework.models.agents import RLAgent

        self.ent.agent_core = RLAgent(
            sensory_input_shape=entity.agent_core.sensory_input_shape, num_actions=entity.agent_core.num_actions
        ).to(CFG.MODEL_DEVICE)
        self.ent.agent_core.load_state_dict(state_dict_detached, strict=False)
        self.ent.agent_core.eval()

        self.ent.energy = float(entity.energy)
        self.ent.fitness = float(entity.fitness)
        self.original_fitness = self.ent.fitness
        self.env = environment

    def test_modification(self, modification_func):
        modification_func(self.ent)
        initial_energy = self.ent.energy

        # Initialize static movement tensors to bypass runtime memory allocation overhead.
        if not hasattr(self, "static_move_vectors"):
            self.static_move_vectors = torch.tensor(
                [
                    [-1, 0, 0],
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 1, 0],
                    [0, 0, -1],
                    [0, 0, 1],
                    [-1, -1, 0],
                    [1, 1, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                device=MODEL_DEVICE,
                dtype=torch.long,
            )

        with torch.no_grad():
            for _ in range(5):
                action = self.ent.act(self.env)
                if action is not None:
                    action_idx = action.item() if action.numel() == 1 else action[0].item()
                    safe_action = max(0, min(15, int(action_idx)))
                    new_coords = self.ent.position[:3] + self.static_move_vectors[safe_action]
                    self.ent.position[:3] = torch.clamp(
                        new_coords, 0, torch.tensor(WORLD_DIM[:3], device=new_coords.device) - 1
                    )
                    self.ent.action_cost_update()

        delta = self.ent.energy - initial_energy
        is_viable = delta > -0.5

        del self.ent
        return is_viable


def synaptic_pruning(agent_core, threshold=0.04):
    """
    Applies exponential decay to long-term memory traces.
    """
    if hasattr(agent_core, "hierarchical_planner"):
        for module in agent_core.hierarchical_planner.modules():
            if type(module).__name__ == "HebbianLinear" and hasattr(module, "hebbian_trace"):
                with torch.no_grad():
                    module.hebbian_trace.mul_(0.99)


def load_entity(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Checkpoint not found at {file_path}")

    data = torch.load(file_path, map_location="cpu", weights_only=False)

    from vrl_framework.models.agents import RLAgent

    sensory_shape = (17, 7, 7, 7)
    num_actions = 17
    agent_core = RLAgent(sensory_input_shape=sensory_shape, num_actions=num_actions)

    if "model_state_dict" not in data:
        raise RuntimeError("Invalid absolute simulation checkpoint format. Expected native DoD snapshot.")

    agent_core.load_state_dict(data["model_state_dict"], strict=False)
    agent_core.to(MODEL_DEVICE)

    pop_state = data.get("population_state", {})
    if not pop_state or "fitness" not in pop_state:
        raise RuntimeError("Missing population_state in checkpoint. Cannot recover spatial topology.")

    fitness_tensor = pop_state["fitness"]
    best_idx = int(torch.argmax(fitness_tensor).item())

    pos = pop_state["positions"][best_idx].tolist() if "positions" in pop_state else [0, 0, 0, 0]
    energy = float(pop_state["energies"][best_idx].item()) if "energies" in pop_state else 1.5
    fitness = float(fitness_tensor[best_idx].item())
    age = int(pop_state["survival_steps"][best_idx].item()) if "survival_steps" in pop_state else 0

    if "hidden_states" in pop_state:
        hidden_state = pop_state["hidden_states"][best_idx].to(MODEL_DEVICE)
    else:
        hidden_state = torch.randn(256, device=MODEL_DEVICE)

    ent = Agent(position=pos, agent_core=agent_core)
    ent.id = int(uuid.uuid4().int % (2**31))
    ent.energy = energy
    ent.fitness = fitness
    ent.age = age
    ent.hyperparams = {"mutation_rate": 0.02, "exploration_noise": 0.1, "baseline_energy_cost": 1.0}
    ent.hidden_state = hidden_state

    import re

    match = re.search(r"gen_(\d+)", file_path)
    if match and not getattr(CFG, "IGNORE_LORA", False):
        lora_path = os.path.join(os.path.dirname(file_path), f"lora_skill_gen_{match.group(1)}.pt")
        if os.path.exists(lora_path):
            try:
                lora_data = torch.load(lora_path, map_location=MODEL_DEVICE, weights_only=False)
                if not hasattr(ent.agent_core, "lora_registry"):
                    ent.agent_core.lora_registry = {}
                ent.agent_core.lora_registry.update(lora_data)
            except Exception:
                pass

    return ent


class DCRL_MAP_Elites_Archive:
    """
    MAP-Elites archive for behavioral diversity.
    """

    def __init__(self, resolution: int = 20):
        self.resolution = resolution
        self.archive: dict[Tuple[int, int], Any] = {}

    def _compute_behavioral_descriptor(self, entity) -> Tuple[int, int]:
        num_experts = getattr(entity.agent_core.moe, "num_experts", 8) if hasattr(entity.agent_core, "moe") else 8
        feature_count = len(getattr(entity, "discovered_features", []))

        dim1 = min(self.resolution - 1, int((num_experts / 32.0) * self.resolution))
        dim2 = min(self.resolution - 1, int((feature_count / 100.0) * self.resolution))
        return (dim1, dim2)

    def evaluate_and_archive(self, entity) -> bool:
        b_desc = self._compute_behavioral_descriptor(entity)
        fit_val = entity.fitness.item() if isinstance(entity.fitness, torch.Tensor) else float(entity.fitness)
        if b_desc not in self.archive or fit_val > self.archive[b_desc]["fitness"]:
            import copy

            state_dict_detached = {k: v.detach().cpu().clone() for k, v in entity.agent_core.state_dict().items()}

            archive_payload = {
                "fitness": entity.fitness,
                "state_dict": state_dict_detached,
                "genetic_code": copy.deepcopy(entity.genetic_code) if hasattr(entity, "genetic_code") else {},
            }

            if entity.pop_ref is not None:
                archive_payload["epigenetic_gamma"] = (
                    entity.pop_ref.population_gamma[entity.idx].detach().cpu().clone()
                )
                archive_payload["epigenetic_beta"] = entity.pop_ref.population_beta[entity.idx].detach().cpu().clone()
                archive_payload["epigenetic_masks"] = (
                    entity.pop_ref.population_masks[entity.idx].detach().cpu().clone()
                )

            self.archive[b_desc] = archive_payload
            return True
        return False

    def sample_elite_and_mutate(self, weak_entity, mutation_rate: float) -> bool:
        if not self.archive:
            return False
        import copy
        import random

        elite_key = random.choice(list(self.archive.keys()))
        elite_data = self.archive[elite_key]

        if weak_entity.pop_ref is not None and "epigenetic_gamma" in elite_data:
            with torch.no_grad():
                weak_entity.pop_ref.population_gamma[weak_entity.idx].copy_(elite_data["epigenetic_gamma"])
                weak_entity.pop_ref.population_beta[weak_entity.idx].copy_(elite_data["epigenetic_beta"])
                weak_entity.pop_ref.population_masks[weak_entity.idx].copy_(elite_data["epigenetic_masks"])
        else:
            state_dict_gpu = {k: v.to(weak_entity.agent_core.device) for k, v in elite_data["state_dict"].items()}
            weak_entity.agent_core.load_state_dict(state_dict_gpu, strict=False)

        if hasattr(weak_entity, "genetic_code"):
            weak_entity.genetic_code = copy.deepcopy(elite_data["genetic_code"])
        return True


class AdversarialCurriculum:
    """PAIRED adversarial environment generator."""

    def __init__(self, world_dim):
        self.world_dim = world_dim
        self.archive = []
        self.max_archive_size = 50
        self.base_difficulty = 1.0
        self.nca_channels = 17
        self.shadow_architect = nn.Conv3d(
            self.nca_channels, self.nca_channels, kernel_size=3, padding=1, bias=False
        ).to(MODEL_DEVICE)
        self.shadow_optimizer = bnb.optim.AdamW8bit(self.shadow_architect.parameters(), lr=1e-4)

    def compute_paired_regret(self, entities):
        """Computes the asymmetric advantage (regret) between the top-performing antagonist and protagonist."""
        if len(entities) < 4:
            return 0.1

        if not hasattr(self, "historical_max_fitness"):
            self.historical_max_fitness = 0.0

        fitness_scores = torch.stack(
            [
                (
                    ent.fitness
                    if isinstance(ent.fitness, torch.Tensor)
                    else torch.tensor(float(ent.fitness), device=MODEL_DEVICE)
                )
                for ent in entities
            ]
        )
        sorted_indices = torch.argsort(fitness_scores, descending=True)

        top_k = max(1, len(entities) // 5)
        antagonist_score = fitness_scores[sorted_indices[:top_k]].mean().item()

        if len(self.archive) > 10 and random.random() < 0.2:
            ghost_idx = random.randint(0, len(self.archive) - 1)
            protagonist_score = self.archive[ghost_idx]["regret"] * 10.0
        else:
            protagonist_score = fitness_scores[sorted_indices[top_k:]].mean().item()

        self.historical_max_fitness = max(self.historical_max_fitness, antagonist_score)
        stagnation_penalty = max(0.0, self.historical_max_fitness - antagonist_score)

        regret = (antagonist_score - protagonist_score) + (stagnation_penalty * 0.8)
        return max(0.1, min(15.0, regret))

    def generate_challenges(self, grid, entities, generation):
        regret = self.compute_paired_regret(entities)

        if regret > 0.5 and len(entities) > 0:
            if len(self.archive) >= self.max_archive_size:
                self.archive.sort(key=lambda x: x["regret"])
                self.archive.pop(0)
            self.archive.append({"grid": grid.clone().detach(), "regret": regret})

        if entities:
            fitness_tensor = torch.stack(
                [
                    (
                        ent.fitness
                        if isinstance(ent.fitness, torch.Tensor)
                        else torch.tensor(float(ent.fitness), device=MODEL_DEVICE)
                    )
                    for ent in entities
                ]
            )
            avg_fitness = fitness_tensor.mean().item()
        else:
            avg_fitness = 0.0

        if wandb.run is not None:
            global_ortho = float(wandb.run.summary.get("moe/expert_orthogonality", 1.0))
            if global_ortho < 0.2:
                for ent in entities:
                    if random.random() < 0.1 and hasattr(ent, "agent_core"):
                        if hasattr(ent.agent_core, "moe"):
                            for param in ent.agent_core.moe.parameters():
                                with torch.no_grad():
                                    param.add_(torch.randn_like(param) * 0.01)

        mastery_factor = min(1.0, max(0.0, avg_fitness / 100.0))
        current_difficulty = self.base_difficulty + (generation / 1000.0) + (mastery_factor * 2.0)

        if len(self.archive) > 5 and random.random() < 0.4:
            sorted_arch = sorted(self.archive, key=lambda x: x["regret"])
            weights = np.linspace(1.0, 10.0, len(sorted_arch))
            weights /= weights.sum()
            idx = np.random.choice(len(sorted_arch), p=weights)
            grid = sorted_arch[idx]["grid"].clone()

        grid_flat = grid.permute(3, 0, 1, 2).unsqueeze(0).detach()

        self.shadow_optimizer.zero_grad()
        with torch.enable_grad():
            shadow_action_raw = self.shadow_architect(grid_flat)

            shadow_action = torch.tanh(shadow_action_raw)
            architect_mask = (torch.rand_like(shadow_action) < (0.01 * current_difficulty)).float()
            adversarial_grid = grid_flat + (shadow_action * architect_mask)

            sparsity_penalty = torch.abs(shadow_action).mean()

            advantage_scalar = float((regret * 2.0) - (sparsity_penalty * 5.0) + (mastery_factor * 5.0))
            spatial_logits = shadow_action_raw.view(1, self.nca_channels, -1)

            loss = -(advantage_scalar * spatial_logits.sum()) / self.nca_channels
            prob_dist = torch.sigmoid(spatial_logits)
            entropy_bonus = -(
                prob_dist * torch.log(prob_dist + 1e-8) + (1.0 - prob_dist) * torch.log(1.0 - prob_dist + 1e-8)
            ).mean()

            # Compute Total Variation (TV) penalty across spatial dimensions to enforce structural smoothness.
            tv_x = torch.abs(shadow_action_raw[:, :, 1:, :, :] - shadow_action_raw[:, :, :-1, :, :]).mean()
            tv_y = torch.abs(shadow_action_raw[:, :, :, 1:, :] - shadow_action_raw[:, :, :, :-1, :]).mean()
            tv_z = torch.abs(shadow_action_raw[:, :, :, :, 1:] - shadow_action_raw[:, :, :, :, :-1]).mean()
            intrinsic_tv_penalty = (tv_x + tv_y + tv_z) * 2.0
            builder_cost = shadow_action_raw[:, 1:2].abs().mean() * 0.5

            elite_fitness = fitness_tensor.max().item() if entities else 0.0
            impossible_penalty = 1000.0 if elite_fitness <= 0.1 else 0.0

            loss = (
                -(advantage_scalar * spatial_logits.sum()) / self.nca_channels
                - (0.15 * entropy_bonus)
                + intrinsic_tv_penalty
                + builder_cost
                + impossible_penalty
            )

            from vrl_framework.trainer.ppo_engine import metrics_aggregator

            metrics_aggregator.log(
                {
                    "loss/adversarial_curriculum": float(loss.item()),
                    "loss/adversarial_entropy": float(entropy_bonus.item()),
                    "loss/adversarial_tv": float(intrinsic_tv_penalty.item()),
                    "loss/adversarial_builder": float(builder_cost.item()),
                }
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.shadow_architect.parameters(), 1.0)
            self.shadow_optimizer.step()
            self.shadow_optimizer.zero_grad(set_to_none=True)

        grid = torch.clamp(adversarial_grid.squeeze(0).permute(1, 2, 3, 0), 0.0, 1.0).detach().clone()
        del grid_flat, shadow_action_raw, shadow_action, architect_mask, adversarial_grid, spatial_logits, loss

        regen_mask = torch.rand_like(grid[..., 0]) < 0.05
        grid[..., 0] = torch.where(regen_mask, torch.clamp(grid[..., 0] + 0.2, 0.0, 1.0), grid[..., 0])

        return grid

    def evaluate_interaction(self, ent, grid):
        pos = tuple(int(p) for p in ent.position.tolist())
        cell_value = grid[pos].item()

        reward = 0.0
        if cell_value > 0.5:
            reward += cell_value * 10.0
            grid[pos] = 0.0
        elif cell_value < 0:
            reward -= 15.0
            ent.energy -= 0.5

        return reward


def visualize_agent_core_structure(world, ent, generation, force=False):
    if not force and generation == 0:
        return

    G = nx.DiGraph()
    num_cells = (
        ent.agent_core.moe.num_experts
        if (hasattr(ent.agent_core, "moe") and hasattr(ent.agent_core.moe, "num_experts"))
        else 8
    )

    for i in range(num_cells):
        G.add_node(str(i))
    for i in range(num_cells - 1):
        G.add_edge(str(i), str(i + 1))

    pos = nx.spring_layout(G, seed=42, k=1.0)

    network_params_sum = sum(p.numel() for p in ent.agent_core.parameters() if p.requires_grad)
    formatted_params = f"{network_params_sum:,}"

    node_colors = ["blue" for _ in range(num_cells)]
    edge_colors = ["#ff0000" for _ in range(num_cells - 1)]
    legend_entries = [("MoE_Experts", num_cells, "blue")]

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100)
    nx.draw_networkx_labels(G, pos, labels={n: n for n in G.nodes()}, font_size=6)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=1.5)

    legend1 = plt.legend(
        handles=[
            Patch(facecolor=color, label=f"{label} ({count})" if count is not None else label)
            for label, count, color in legend_entries
        ],
        title="Cell Specializations",
        loc="upper left",
    )
    plt.gca().add_artist(legend1)

    plt.title(f"Agent Core (Gen {generation}) | Experts: {num_cells} | Total Params: {formatted_params}")
    short_id = str(ent.id)[:8]
    plt.figtext(0.5, 0.07, f"Best agent id: {short_id}", ha="center", fontsize=10, fontweight="bold")

    from vrl_framework.core.settings import METRICS_DIR

    topology_dir = os.path.join(METRICS_DIR, "topology")
    os.makedirs(topology_dir, exist_ok=True)
    save_path = os.path.join(topology_dir, f"champ_agent_core_gen{generation}.png")
    plt.savefig(save_path)
    plt.close()
    logging.info("Agent Core structure visualization saved at: %s", save_path)


def visualize_complex_entity_3D(world, generation):
    if not world.entities:
        return
    best_ent = max(world.entities, key=lambda o: o.fitness)

    G = nx.DiGraph()
    num_cells = (
        best_ent.agent_core.moe.num_experts
        if (hasattr(best_ent.agent_core, "moe") and hasattr(best_ent.agent_core.moe, "num_experts"))
        else 8
    )

    for i in range(num_cells - 1):
        G.add_edge(str(i), str(i + 1))
    pos2d = nx.spring_layout(G, seed=42)
    pos3d = {node: (pos2d[node][0], pos2d[node][1], random.uniform(-0.1, 0.1)) for node in G.nodes()}

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    xs = [pos3d[node][0] for node in G.nodes()]
    ys = [pos3d[node][1] for node in G.nodes()]
    zs = [pos3d[node][2] for node in G.nodes()]
    ax.scatter(xs, ys, zs, c="r", s=100, label="Cells")
    for edge in G.edges():
        x_coords = [pos3d[edge[0]][0], pos3d[edge[1]][0]]
        y_coords = [pos3d[edge[0]][1], pos3d[edge[1]][1]]
        z_coords = [pos3d[edge[0]][2], pos3d[edge[1]][2]]
        ax.plot(x_coords, y_coords, z_coords, c="b")
    ax.set_title(f"3D Agent Core Structure (Gen {generation})")
    plt.legend()

    from vrl_framework.core.settings import METRICS_DIR

    topology_dir = os.path.join(METRICS_DIR, "topology")
    os.makedirs(topology_dir, exist_ok=True)

    save_path = os.path.join(topology_dir, f"topology_gen_{generation}.png")
    plt.savefig(save_path)
    plt.close(fig)

    logging.info("Complex 3D entity visualization saved at: %s", save_path)


class UnifiedMetaController:
    """
    Manages computational budget for reasoning.
    """

    def __init__(self):
        self.active_goals = []
        self.cognitive_budget = 1.0

    def allocate_budget(self, surprisal_metric: float) -> float:
        self.cognitive_budget = max(0.1, min(10.0, self.cognitive_budget + surprisal_metric * 0.1))
        return self.cognitive_budget


class VectorizedWorld4D:
    """
    4D spatial grid and asynchronous curriculum simulation environment.
    """

    def __init__(self, variation=1.0):
        self.total_generations = None
        self.variation = variation
        self.complexity_factor = 1.0
        self.grid = torch.zeros(WORLD_DIM, device=MODEL_DEVICE)
        self.entities = []  # Maps to instances of LatentAgent
        self.generation = 0
        self.metrics = {}
        self.metrics_payload = {
            "fitness/best": 0.0,
            "fitness/avg": 0.0,
            "batch/alive": 0,
            "system/generation_time": 0.0,
            "system/vram_allocated_gb": 0.0,
        }
        self.historical_environments = collections.deque(maxlen=10)
        self.last_jepa_loss_ema = 0.0
        self.init_stochastic_resources()
        self.symbols = self.generate_symbols()

        # Polyfill for missing anomaly tracking integration to maintain compatibility with legacy metrics.
        class _AnomalyBufferMock:
            def __init__(self):
                self.events = []

            def add_event(self, ev):
                self.events.append(ev)

        self.anomaly_buffer = _AnomalyBufferMock()

        self.last_repopulation_generation = 0
        self.last_checkpoint_message = ""
        self.start_time = time.time()
        self._paused = False
        self._stop = False
        self.total_generations = None
        self.curriculum = AdversarialCurriculum(WORLD_DIM)
        self.gen_times = []
        self.ema_gen_time = None
        self.ema_core_time = None
        self.ema_overhead_time = None
        self.estimated_remaining = 0
        self.estimated_total_time = 0
        self.estimated_progress = 0
        self.meta_controller = UnifiedMetaController()
        self.current_question = None
        self.min_reasoning_time = 0
        self.max_reasoning_time = 0
        self.start_reasoning_time = 0.0
        self.best_score_so_far = float("-inf")
        self.best_answer_so_far = None
        self.quality_threshold = 0.9
        self.temporary_answers = []
        self.timed_ckpt_interval = 600
        self._last_timed_ckpt = time.time()
        self.enable_online_training = True

        self.batched_agents = VectorizedPopulation(
            initial_agents=INIT_POPULATION, world_dim=WORLD_DIM, max_agents=MAX_POPULATION
        )
        self.batched_agents.world_ref = self

        import sys

        logging.getLogger("WandbProbe").warning(
            f"[WANDB INIT] Initialization attempt. Process name: {mp.current_process().name}"
        )

        if True:
            run_id_to_resume = None
            if (
                hasattr(CFG, "RESUME_CHECKPOINT")
                and CFG.RESUME_CHECKPOINT is not None
                and "--new_metrics" not in sys.argv
            ):
                try:
                    ckpt_meta = torch.load(CFG.RESUME_CHECKPOINT, map_location="cpu", weights_only=False)
                    if isinstance(ckpt_meta, dict) and ckpt_meta.get("wandb_run_id") is not None:
                        run_id_to_resume = ckpt_meta["wandb_run_id"]
                except Exception:

                    logging.getLogger(__name__).exception("Failed to restore wandb_run_id from checkpoint")

            wandb.init(
                id=run_id_to_resume,
                resume="allow" if run_id_to_resume else None,
                project="VRL-Framework",
                mode="online",
                save_code=False,
                name=f"Evolution_Run_{uuid.uuid4().hex[:6]}",
                settings=wandb.Settings(mode="online", _disable_stats=False),
                config={
                    "world_dim": WORLD_DIM,
                    "max_population": MAX_POPULATION,
                    "learning_rate": LEARNING_RATE,
                    "mutation_rate": CFG.MUTATION_RATE,
                },
            )
            wandb.define_metric("generation")
            wandb.define_metric("*", step_metric="generation")
            wandb.define_metric("global_train_step", hidden=True)

    def init_stochastic_resources(self):
        self.grid = torch.zeros(WORLD_DIM, device=MODEL_DEVICE)
        self.grid[..., 0] = torch.rand(WORLD_DIM[:3], device=MODEL_DEVICE) * 0.5
        island_mask = torch.rand(WORLD_DIM[:3], device=MODEL_DEVICE) > 0.98
        self.grid[..., 1][island_mask] = 1.0
        self.audio_grid = torch.zeros(1, 44100, device=MODEL_DEVICE)

        self.nca_channels = 17
        self.nca_physics = nn.Conv3d(
            in_channels=self.nca_channels,
            out_channels=self.nca_channels,
            kernel_size=3,
            padding=1,
            groups=self.nca_channels,
            bias=False,
            padding_mode="circular",
        ).to(MODEL_DEVICE)

        with torch.no_grad():
            laplacian_prior = (
                torch.tensor(
                    [
                        [[0.0, 0.1, 0.0], [0.1, 0.2, 0.1], [0.0, 0.1, 0.0]],
                        [[0.1, 0.2, 0.1], [0.2, -2.4, 0.2], [0.1, 0.2, 0.1]],
                        [[0.0, 0.1, 0.0], [0.1, 0.2, 0.1], [0.0, 0.1, 0.0]],
                    ],
                    device=MODEL_DEVICE,
                )
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(self.nca_channels, 1, 3, 3, 3)
            )
            self.nca_physics.weight.copy_(laplacian_prior + torch.randn_like(self.nca_physics.weight) * 0.01)

        self.semantic_broadcast_flag = torch.zeros(1, dtype=torch.int32, device=MODEL_DEVICE)
        self.semantic_extraction_buffer = torch.zeros(4096, dtype=torch.float16, device=MODEL_DEVICE)

    def generate_symbols(self):
        return [torch.randn(7, 7, 7, 5, device=MODEL_DEVICE) for _ in range(20)]

    def add_entity(self, ent):
        self.entities.append(ent)
        if hasattr(self, "batched_agents"):
            self.batched_agents.sync_with_entities_list(self.entities)

    def update_environment(self):
        self.complexity_factor = 1.0 + (self.generation / 500.0)

        grid_flat = self.grid.permute(3, 0, 1, 2).unsqueeze(0)

        with torch.no_grad():
            seed_mask = grid_flat[:, 15:16] > 0.1
            active_region_mask = F.max_pool3d(seed_mask.float(), kernel_size=5, stride=1, padding=2)

            if active_region_mask.sum() > 0:
                masked_grid = grid_flat * active_region_mask
                state_updates = self.nca_physics(masked_grid)
                rigidity = grid_flat[:, 1:2]
                diffusion_suppression = 1.0 - rigidity
                scaled_grid_int = (grid_flat[:, 0:1] * 1000).to(torch.int32)
                hash_chaos = ((scaled_grid_int ^ torch.roll(scaled_grid_int, 1, dims=2)) & 1).float()

                stochastic_mask = (torch.rand_like(state_updates) > 0.5).float()
                fractal_interference = hash_chaos.expand_as(state_updates) * 0.05
                delta = state_updates + fractal_interference
                delta.mul_(stochastic_mask).mul_(0.1).mul_(active_region_mask).mul_(diffusion_suppression)
                new_grid = grid_flat + delta
            else:
                new_grid = grid_flat

            # Compute reaction-diffusion rates parameterized by local voxel density
            local_voxels_sum = F.avg_pool3d(new_grid[:, 1:2], kernel_size=3, stride=1, padding=1) * 27.0
            reaction_rate = 0.002 * (local_voxels_sum**3)
            diffused_energy = new_grid[:, 0:1] + reaction_rate * new_grid[:, 1:2] * (1.0 - new_grid[:, 0:1])
            crystallization_mask = (
                (diffused_energy < 0.005) & (new_grid[:, 1:2] < 0.05) & (torch.rand_like(diffused_energy) > 0.5)
            )

            local_density = F.avg_pool3d(new_grid[:, 15:16], kernel_size=3, stride=1, padding=1)
            growth_signal = torch.where((local_density > 0.1) & (local_density < 0.8), 0.01, -0.05)

            base_crystallization = torch.clamp(new_grid[:, 15:16] + growth_signal, 0.0, 1.0)
            new_grid[:, 15:16] = torch.where(
                crystallization_mask, torch.clamp(base_crystallization + 0.5, max=1.0), base_crystallization
            )

            self.grid[..., 0] = diffused_energy.squeeze(0).squeeze(0)
            self.grid[..., 1:] = new_grid[:, 1:].squeeze(0).permute(1, 2, 3, 0)

            # Extract active agent coordinates for grid mutation mapping
            if hasattr(self, "batched_agents") and hasattr(self.batched_agents, "positions"):
                activity_coords = self.batched_agents.positions[self.batched_agents.active_mask, :3].long()
            else:
                activity_coords = torch.stack(
                    [
                        (
                            ent.position
                            if isinstance(ent.position, torch.Tensor)
                            else torch.tensor(ent.position, device=MODEL_DEVICE)
                        )
                        for ent in self.entities
                    ]
                ).long()

            x = activity_coords[:, 0].clamp(0, self.grid.shape[0] - 1)
            y = activity_coords[:, 1].clamp(0, self.grid.shape[1] - 1)
            z = activity_coords[:, 2].clamp(0, self.grid.shape[2] - 1)

            depletion_tensor = torch.full((x.size(0),), -0.1, device=MODEL_DEVICE)
            self.grid[..., 0].index_put_((x, y, z), depletion_tensor, accumulate=True)
            self.grid[..., 0] = torch.clamp(self.grid[..., 0], min=0.0)

        thermal_diffusion = torch.randn_like(self.grid[..., :2]) * 0.001
        self.grid[..., :2] = self.grid[..., :2] + thermal_diffusion

        if getattr(self, "_periphery_mask", None) is None:
            center_x, center_y, center_z = WORLD_DIM[0] // 2, WORLD_DIM[1] // 2, WORLD_DIM[2] // 2
            grid_coords = (
                torch.stack(
                    torch.meshgrid(
                        torch.arange(WORLD_DIM[0]),
                        torch.arange(WORLD_DIM[1]),
                        torch.arange(WORLD_DIM[2]),
                        indexing="ij",
                    ),
                    dim=-1,
                )
                .float()
                .to(MODEL_DEVICE)
            )
            dist_from_center = torch.norm(
                grid_coords - torch.tensor([center_x, center_y, center_z], device=MODEL_DEVICE).float(), p=2, dim=-1
            )
            self._periphery_mask = dist_from_center > (min(WORLD_DIM[:3]) * 0.45)

        self.grid[..., 0].masked_fill_(self._periphery_mask, 0.0)
        if hasattr(self, "batched_agents") and hasattr(self.batched_agents, "energies"):
            p_pos_x = self.batched_agents.positions[:, 0].long().clamp(0, WORLD_DIM[0] - 1)
            p_pos_y = self.batched_agents.positions[:, 1].long().clamp(0, WORLD_DIM[1] - 1)
            p_pos_z = self.batched_agents.positions[:, 2].long().clamp(0, WORLD_DIM[2] - 1)
            agent_in_periphery = self._periphery_mask[p_pos_x, p_pos_y, p_pos_z]
            self.batched_agents.energies = torch.where(
                agent_in_periphery, torch.zeros_like(self.batched_agents.energies), self.batched_agents.energies
            )

        if self.generation > 0 and self.generation % 500 == 0:
            with torch.no_grad():
                physics_mutation = torch.randn_like(self.nca_physics.weight) * 0.001
                self.nca_physics.weight.add_(physics_mutation)
                weight_norm = self.nca_physics.weight.view(self.nca_physics.weight.size(0), -1).norm(p=2, dim=1)
                safe_norm = torch.clamp(weight_norm, min=1.0).view(-1, 1, 1, 1, 1)
                self.nca_physics.weight.data.copy_(self.nca_physics.weight / safe_norm)

        self.grid[..., 0:2] = torch.clamp(self.grid[..., 0:2], 0.0, 1.0)
        self.grid[..., 2:5] = torch.clamp(self.grid[..., 2:5], -1.0, 1.0)
        self.grid[..., 5:15] = torch.clamp(self.grid[..., 5:15], -2.0, 2.0)
        self.grid[..., 15] = torch.clamp(self.grid[..., 15], 0.0, 1.0)
        self.grid[..., 16] = self.grid[..., 16] * 0.05

        if hasattr(self, "audio_grid"):
            self.audio_grid = self.audio_grid * 0.8

        self.log_metrics()

    def simulate_unexpected_event(self):
        """
        Injects spatial noise and audio shockwaves to test agent recovery.
        """
        with torch.no_grad():
            noise_base = torch.randn(
                1, 1, self.grid.size(0) // 4, self.grid.size(1) // 4, self.grid.size(2) // 4, device=self.grid.device
            )
            correlated_noise = F.interpolate(
                noise_base,
                size=(self.grid.size(0), self.grid.size(1), self.grid.size(2)),
                mode="trilinear",
                align_corners=False,
            )
            correlated_noise = correlated_noise.squeeze(0).squeeze(0).unsqueeze(-1)

            self.grid[..., 0] = torch.clamp(self.grid[..., 0] + correlated_noise[..., 0] * 0.5, 0.0, 1.0)

            event_vector = torch.randn(256, device="cpu")
            if hasattr(self, "anomaly_buffer"):
                self.anomaly_buffer.add_event(event_vector)

            t = torch.linspace(0, 1, 44100, device=MODEL_DEVICE)
            shockwave = torch.sin(2 * math.pi * 440 * t) * torch.exp(-5 * t)
            self.audio_grid += shockwave.unsqueeze(0)

            if hasattr(self, "nca_physics") and self.nca_physics.weight is not None:
                self.nca_physics.weight.add_(torch.randn_like(self.nca_physics.weight) * 0.005)

            if self.entities:
                shock_tensor = event_vector.to(MODEL_DEVICE)
                for ent in self.entities:
                    ent.receive_signal(shock_tensor[:128])

                    if hasattr(ent.agent_core, "fuzzy_kb") and hasattr(ent.agent_core.fuzzy_kb, "hyper_memory"):
                        bipolar_shock = torch.sign(shock_tensor)
                        bundled_memory = ent.agent_core.fuzzy_kb.hyper_memory + bipolar_shock.mean(dim=0, keepdim=True)
                        ent.agent_core.fuzzy_kb.hyper_memory = torch.sign(bundled_memory)

                    if hasattr(ent.agent_core, "communication"):
                        with torch.no_grad():
                            enc_weight = ent.agent_core.communication.encoder[0].weight
                            enc_weight.add_(torch.randn_like(enc_weight) * 0.5)

                            dec_weight = ent.agent_core.communication.decoder[0].weight
                            dec_weight.add_(torch.randn_like(dec_weight) * 0.5)

    def check_curriculum_phase(self):
        """
        Checks and updates the curriculum phase based on population mastery.
        """
        if not hasattr(self, "curriculum_phase"):
            self.curriculum_phase = 1
            self.task_mastery_ema = 0.0

        current_rewards = [float(ent.fitness) for ent in self.entities]
        if current_rewards:
            max_possible_reward = 100.0 * self.curriculum_phase
            normalized_mastery = np.mean(current_rewards) / max_possible_reward
            self.task_mastery_ema = 0.99 * self.task_mastery_ema + 0.01 * normalized_mastery

        new_phase = self.curriculum_phase

        if self.task_mastery_ema > 0.85:
            new_phase += 1
            self.task_mastery_ema = 0.0
            logging.info(f"[CURRICULUM] Phase {new_phase} Transition via Asymptotic Mastery")

        if new_phase != self.curriculum_phase:
            self.curriculum_phase = new_phase
            for ent in self.entities:
                if hasattr(ent, "experience_buffer"):
                    ent.experience_buffer.ptr = 0
                    ent.experience_buffer.size = 0

                if hasattr(ent.agent_core, "trainer") and hasattr(ent.agent_core.trainer, "opt_policy"):
                    for param_group in ent.agent_core.trainer.opt_policy.param_groups:
                        param_group["lr"] = CFG.LEARNING_RATE
                if hasattr(ent.agent_core, "trainer") and hasattr(ent.agent_core.trainer, "opt_policy_fp32"):
                    for param_group in ent.agent_core.trainer.opt_policy_fp32.param_groups:
                        param_group["lr"] = CFG.LEARNING_RATE

    def update_metrics(self):
        self.check_curriculum_phase()
        if not self.entities:
            return

        if self.generation % 10 == 0:
            if hasattr(self, "batched_agents") and hasattr(self.batched_agents, "fitness"):
                raw_fitness = self.batched_agents.fitness
                best_fit = float(raw_fitness.max().item()) if raw_fitness.numel() > 0 else 0.0
                avg_fit = float(raw_fitness.mean().item()) if raw_fitness.numel() > 0 else 0.0
            else:
                fitness_scores = [float(ent.fitness) for ent in self.entities]
                best_fit = max(fitness_scores) if fitness_scores else 0.0
                avg_fit = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0.0

            self.metrics_payload["fitness/best"] = best_fit
            self.metrics_payload["fitness/avg"] = avg_fit

            exploration_temp = (
                float(F.softplus(self.entities[0].agent_core.exploration_layer.temperature).item() + 0.01)
                if hasattr(self.entities[0].agent_core, "exploration_layer")
                else 1.0
            )
            vram_usage = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

            if hasattr(self, "grid") and self.grid is not None:
                with torch.no_grad():
                    density = self.grid[..., 0].float()
                    if density.dim() >= 2:
                        variance = density.var()
                        tv_x = torch.abs(density[1:, ...] - density[:-1, ...]).mean()
                        tv_y = torch.abs(density[:, 1:, ...] - density[:, :-1, ...]).mean()
                        total_variation = (tv_x + tv_y) / 2.0
                        if density.dim() >= 3:
                            tv_z = torch.abs(density[:, :, 1:, ...] - density[:, :, :-1, ...]).mean()
                            total_variation = (tv_x + tv_y + tv_z) / 3.0
                        structure_index = (variance / (total_variation + 1e-8)).item()
                        self.metrics_payload["environment/spatial_structure_index"] = structure_index
                    if self.grid.shape[-1] >= 16:
                        crystallization = (self.grid[..., 15] > 0.5).float().mean().item()
                        self.metrics_payload["environment/crystallization_ratio"] = crystallization

            if hasattr(self, "batched_agents") and hasattr(self.batched_agents, "fitness"):
                valid_mask = self.batched_agents.active_mask
                alive_count = int(valid_mask.sum().item())
                if valid_mask.any() and hasattr(self.batched_agents, "population_gamma"):
                    active_gamma = self.batched_agents.population_gamma[valid_mask].float()
                    self.metrics_payload["population/epigenetic_diversity"] = float(
                        active_gamma.std(dim=0, unbiased=False).mean().item()
                    )
                else:
                    self.metrics_payload["population/epigenetic_diversity"] = 0.0
            else:
                alive_count = len(self.entities)
                self.metrics_payload["population/epigenetic_diversity"] = 0.0

            self.metrics_payload.pop("batch/alive", None)

            if hasattr(self, "batched_agents"):
                if hasattr(self.batched_agents, "visited_spatial_grid"):
                    total_voxels = max(1.0, float(self.batched_agents.visited_spatial_grid.numel()))
                    current_cov = float((self.batched_agents.visited_spatial_grid > 0).sum().item()) / total_voxels
                else:
                    current_cov = 0.0

                self.metrics_payload["open_endedness/voxels_visited"] = current_cov

            dead_count_estimate = max(0, INIT_POPULATION - alive_count)
            current_tax = metabolic_curriculum.compute_multiplier(
                dead_count_estimate, INIT_POPULATION, current_coverage=current_cov
            )

            if hasattr(self.entities[0].agent_core, "hierarchical_planner") and hasattr(
                self.entities[0].agent_core.hierarchical_planner, "_metric_buffer"
            ):
                self.metrics_payload.update(self.entities[0].agent_core.hierarchical_planner._metric_buffer)

            self.metrics_payload.update(
                {
                    "generation": self.generation,
                    "fitness/best": float(best_fit),
                    "fitness/avg": float(avg_fit),
                    "rl/exploration_temperature": float(exploration_temp),
                    "system/vram_allocated_gb": float(vram_usage),
                    "population/genetic_turnover_rate": float(getattr(self, "genetic_turnover_rate", 0.0)),
                    "environment/metabolic_roi": (
                        float(getattr(self.entities[0].agent_core, "last_metabolic_roi", 0.0))
                        if self.entities
                        else 0.0
                    ),
                    "environment/metabolic_tax_multiplier": float(current_tax),
                }
            )

            if getattr(self, "_avg_phi_buffer", []):
                safe_phi = [x.item() if isinstance(x, torch.Tensor) else x for x in self._avg_phi_buffer]
                safe_phi = [x for x in safe_phi if not math.isnan(x)]
                self.metrics_payload["diagnostics/avg_phi"] = sum(safe_phi) / max(1, len(safe_phi))
                self._avg_phi_buffer.clear()
            if (
                hasattr(self, "batched_agents")
                and getattr(self.batched_agents, "last_state_features_batch", None) is not None
            ):
                latent_batch_for_div = self.batched_agents.last_state_features_batch
                if (
                    hasattr(self.batched_agents, "active_mask")
                    and self.batched_agents.active_mask is not None
                    and self.batched_agents.active_mask.numel() == latent_batch_for_div.size(0)
                ):
                    latent_batch_for_div = latent_batch_for_div[self.batched_agents.active_mask]
                if latent_batch_for_div.size(0) > 1:
                    latent_std = latent_batch_for_div.float().std(dim=0, unbiased=False)
                    self.metrics_payload["diagnostics/latent_diversity"] = float(latent_std.mean().item())
            elif getattr(self, "_diversity_buffer", []):
                safe_div = [x for x in self._diversity_buffer if not math.isnan(x)]
                self.metrics_payload["diagnostics/latent_diversity"] = sum(safe_div) / max(1, len(safe_div))
                self._diversity_buffer.clear()

            self.metrics_payload["generation"] = self.generation

            # Compute tensor-based metrics sparsely.
            if self.generation % 10 == 0:
                champion = max(self.entities, key=lambda o: o.fitness)
                champ_agent_core = champion.agent_core
                with torch.no_grad():
                    # Estimate the effective rank of the latent activation batch
                    if (
                        hasattr(self, "batched_agents")
                        and hasattr(self.batched_agents, "last_state_features_batch")
                        and self.batched_agents.last_state_features_batch is not None
                    ):
                        latent_batch = self.batched_agents.last_state_features_batch
                        if (
                            hasattr(self.batched_agents, "active_mask")
                            and self.batched_agents.active_mask is not None
                            and self.batched_agents.active_mask.numel() == latent_batch.size(0)
                        ):
                            latent_batch = latent_batch[self.batched_agents.active_mask]
                        if latent_batch.size(0) > 1:
                            try:
                                safe_latent = torch.nan_to_num(latent_batch, nan=0.0, posinf=10.0, neginf=-10.0)
                                eff_dim = compute_eff_dim(safe_latent)
                                if math.isnan(eff_dim):
                                    eff_dim = 1.0
                            except Exception:
                                eff_dim = getattr(self, "_last_eff_dim", 1.0)
                            self._last_eff_dim = eff_dim
                            self.metrics_payload["diagnostics/latent_effective_dimensionality"] = float(eff_dim)

                            valid_pos = self.batched_agents.positions[self.batched_agents.active_mask, :3].float()
                            spatial_var = valid_pos.var(dim=0).sum().item() if valid_pos.size(0) > 1 else 0.0
                            self.metrics_payload["open_endedness/spatial_variance"] = float(spatial_var)

                            raw_ratio = 1.0 - (float(eff_dim) / max(1, latent_batch.size(-1)))
                            crystalization_ratio = float(max(0.0, min(1.0, raw_ratio)))
                            if math.isnan(crystalization_ratio):
                                crystalization_ratio = 0.0
                            self.metrics_payload["diagnostics/crystalization_ratio"] = crystalization_ratio

                        if eff_dim > 240.0 and self.generation > 100000:
                            import logging

                            logging.getLogger("vrl_framework").warning(
                                "Latent representation capacity saturated. Forcing topology update."
                            )
                            if hasattr(champ_agent_core, "exploration_layer"):
                                champ_agent_core.exploration_layer.log_alpha.data.fill_(math.log(5.0))

                            if hasattr(champ_agent_core.jepa, "fp16_encoder"):
                                torch.nn.init.orthogonal_(champ_agent_core.jepa.fp16_encoder[-1].weight)
                            if hasattr(champ_agent_core.jepa, "fsq_encoder"):
                                torch.nn.init.orthogonal_(champ_agent_core.jepa.fsq_encoder[-1].weight)
                            for ent in self.entities:
                                if hasattr(ent.agent_core.moe, "update_topology"):
                                    ent.agent_core.moe.update_topology(drop_fraction=0.5)

                _dyn_surprisal = getattr(champ_agent_core.jepa, "proxy_surprisal", 0.0)
                if isinstance(_dyn_surprisal, torch.Tensor):
                    dynamics_loss = float(_dyn_surprisal.mean().item())
                else:
                    dynamics_loss = float(_dyn_surprisal)

                if math.isnan(dynamics_loss) or math.isinf(dynamics_loss):
                    dynamics_loss = 1.0

                if not hasattr(self, "fast_jepa_loss_ema"):
                    self.fast_jepa_loss_ema = max(1e-6, dynamics_loss)
                    self.jepa_loss_history = collections.deque(maxlen=100)

                safe_dynamics = max(1e-6, dynamics_loss)
                self.jepa_loss_history.append(math.log(safe_dynamics))

                self.last_jepa_loss_ema = 0.99 * self.last_jepa_loss_ema + 0.01 * dynamics_loss
                self.fast_jepa_loss_ema = 0.9 * self.fast_jepa_loss_ema + 0.1 * dynamics_loss

                try:
                    if len(self.jepa_loss_history) > 2:
                        y = np.array(self.jepa_loss_history)
                        x = np.arange(len(y))
                        mask = ~np.isnan(y) & ~np.isinf(y)
                        if mask.sum() > 2:
                            slope, _ = np.polyfit(x[mask], y[mask], 1)
                            raw_velocity = float(-slope * 1000.0)

                            raw_velocity = max(-5.0, min(5.0, raw_velocity))
                        else:
                            raw_velocity = 0.0
                    else:
                        raw_velocity = 0.0
                except Exception:
                    raw_velocity = 0.0

                if math.isnan(raw_velocity) or math.isinf(raw_velocity):
                    raw_velocity = 0.0

                if not hasattr(self, "_smoothed_learning_progress"):
                    self._smoothed_learning_progress = raw_velocity

                self._smoothed_learning_progress = 0.6 * self._smoothed_learning_progress + 0.4 * raw_velocity
                self.metrics_payload["curiosity/learning_progress_velocity"] = float(self._smoothed_learning_progress)

                if hasattr(champ_agent_core, "last_vq_loss"):
                    vq_val = champ_agent_core.last_vq_loss
                    if isinstance(vq_val, torch.Tensor):
                        vq_val = vq_val.mean().item()
                    self.metrics_payload["diagnostics/vq_loss"] = float(vq_val)

                if hasattr(champ_agent_core, "moe"):
                    with torch.no_grad():
                        if hasattr(champ_agent_core.moe, "expert_centroids"):
                            centroids = champ_agent_core.moe.expert_centroids
                            norm_centroids = F.normalize(centroids, p=2, dim=-1)
                            cos_sim_matrix = torch.matmul(norm_centroids, norm_centroids.t())
                            sum_sq_all = torch.sum(cos_sim_matrix**2)
                            sum_sq_diag = torch.sum(torch.diag(cos_sim_matrix) ** 2)
                            off_diag_elements = float(centroids.size(0) * (centroids.size(0) - 1))
                            if off_diag_elements > 0:
                                ortho_score = 1.0 - ((sum_sq_all - sum_sq_diag) / off_diag_elements).item()
                            else:
                                ortho_score = 1.0
                            if math.isnan(ortho_score):
                                ortho_score = 1.0
                        else:
                            moe_weights = getattr(
                                champ_agent_core.moe,
                                "expert_w1",
                                getattr(champ_agent_core.moe, "vram_buffer_A_w1", None),
                            )
                            if moe_weights is None and hasattr(champ_agent_core.moe, "experts"):
                                w_list = []
                                for exp in champ_agent_core.moe.experts:
                                    for n, p in exp.named_parameters():
                                        if "weight" in n and p.dim() >= 2:
                                            w_list.append(p)
                                            break
                                if w_list:
                                    moe_weights = torch.stack(w_list)
                            if moe_weights is None:
                                p2d = [p for p in champ_agent_core.moe.parameters() if p.dim() >= 2]
                                if p2d:
                                    lp = max(p2d, key=lambda p: p.numel())
                                    moe_weights = lp.unsqueeze(0) if lp.dim() == 2 else lp
                            if moe_weights is not None:
                                try:
                                    ortho_score = compute_expert_orthogonality(moe_weights.detach())
                                except Exception:
                                    ortho_score = 1.0
                                if math.isnan(ortho_score):
                                    ortho_score = 1.0

                        if hasattr(champ_agent_core.moe, "expert_usage_ema"):
                            usage = torch.clamp(champ_agent_core.moe.expert_usage_ema, min=1e-9)
                            entropy = -(usage * torch.log(usage)).sum().item()
                            if math.isnan(entropy):
                                entropy = 0.0
                            self.metrics_payload["moe/routing_entropy"] = float(entropy)
                            usage_var = usage.var().item()
                            if math.isnan(usage_var):
                                usage_var = 0.0
                            self.metrics_payload["moe/expert_usage_variance"] = float(usage_var)

                if (
                    hasattr(self.batched_agents, "last_fused_context_batch")
                    and self.batched_agents.last_fused_context_batch is not None
                ):
                    test_ctx = self.batched_agents.last_fused_context_batch[: min(32, len(self.entities))]
                    if test_ctx.size(0) > 0:
                        try:
                            reflex_out = champ_agent_core.actor_critic(champ_agent_core.moe(test_ctx))
                            reflex_logits_full = (
                                reflex_out.policy_logits
                                if hasattr(reflex_out, "policy_logits")
                                else (reflex_out[0] if isinstance(reflex_out, tuple) else reflex_out)
                            )
                            reflex_logits = reflex_logits_full[..., : champ_agent_core.num_actions]

                            moe_test_ctx = champ_agent_core.moe(test_ctx)
                            dummy_intents = torch.zeros(
                                test_ctx.size(0), 256, device=test_ctx.device, dtype=test_ctx.dtype
                            )
                            deliberate_ctx = champ_agent_core.hierarchical_planner(moe_test_ctx, dummy_intents)

                            deliberate_out = champ_agent_core.actor_critic(deliberate_ctx)
                            deliberate_logits_full = (
                                deliberate_out.policy_logits
                                if hasattr(deliberate_out, "policy_logits")
                                else (deliberate_out[0] if isinstance(deliberate_out, tuple) else deliberate_out)
                            )
                            deliberate_logits = deliberate_logits_full[..., : champ_agent_core.num_actions]

                            safe_deliberate = torch.nan_to_num(deliberate_logits, nan=0.0, posinf=20.0, neginf=-20.0)
                            safe_reflex = torch.nan_to_num(reflex_logits, nan=0.0, posinf=20.0, neginf=-20.0)

                            log_probs_deliberate = F.log_softmax(safe_deliberate, dim=-1)
                            probs_reflex = F.softmax(safe_reflex, dim=-1)
                            kl_div = F.kl_div(log_probs_deliberate, probs_reflex, reduction="batchmean")
                            safe_kl = float(torch.clamp(torch.nan_to_num(kl_div, nan=0.0), min=0.0).item())
                            self.metrics_payload["cognition/system2_override_divergence"] = safe_kl

                            if hasattr(reflex_out, "value") and hasattr(deliberate_out, "value"):
                                pred_gain = (deliberate_out.value.mean() - reflex_out.value.mean()).item()
                                self.metrics_payload["planner/predicted_gain"] = float(pred_gain)
                            elif (
                                isinstance(reflex_out, tuple)
                                and len(reflex_out) > 1
                                and isinstance(reflex_out[1], torch.Tensor)
                            ):
                                pred_gain = (deliberate_out[1].mean() - reflex_out[1].mean()).item()
                                self.metrics_payload["planner/predicted_gain"] = float(pred_gain)
                            else:
                                gain_proxy = float(
                                    (safe_kl * 0.5) + getattr(champ_agent_core, "health_score_proxy", 0.0)
                                )
                                self.metrics_payload["planner/predicted_gain"] = float(
                                    np.nan_to_num(torch.tensor(gain_proxy), nan=0.0).item()
                                )
                        except Exception:

                            logging.getLogger(__name__).exception("Failed to compute planner predicted gain")

                env_surprisal = 0.0
                if (
                    getattr(self.batched_agents, "last_state_features_batch", None) is not None
                    and getattr(self.batched_agents, "prev_state_features_batch", None) is not None
                    and self.batched_agents.prev_state_features_batch.size(0)
                    == self.batched_agents.last_state_features_batch.size(0)
                ):
                    env_surprisal = F.mse_loss(
                        self.batched_agents.last_state_features_batch, self.batched_agents.prev_state_features_batch
                    ).item()
                else:
                    env_surprisal = 0.05
                self.metrics_payload["environment/predictive_surprisal"] = float(env_surprisal)

                if hasattr(self, "batched_agents") and hasattr(self.batched_agents, "last_energy_intake_mean"):
                    energy_velocity = self.batched_agents.last_energy_intake_mean
                elif hasattr(self, "last_energy_replenish_mean"):
                    energy_velocity = self.last_energy_replenish_mean
                else:
                    energy_velocity = 0.0
                self.metrics_payload["environment/energy_velocity"] = float(energy_velocity)

                if hasattr(self, "map_elites_archive"):
                    self.metrics_payload["open_endedness/archive_coverage"] = len(self.map_elites_archive.archive)

                if hasattr(self, "batched_agents") and hasattr(self.batched_agents, "permanent_visited_grid"):
                    current_unique_voxel_total = int(self.batched_agents.permanent_visited_grid.sum().item())
                    if not hasattr(self, "_last_unique_voxel_total"):
                        self._last_unique_voxel_total = current_unique_voxel_total
                    self.metrics_payload["open_endedness/unique_voxels_discovered"] = float(
                        max(0, current_unique_voxel_total - self._last_unique_voxel_total)
                    )
                    self._last_unique_voxel_total = current_unique_voxel_total
                elif hasattr(self, "global_visited_voxels"):
                    current_unique_voxel_total = int(len(self.global_visited_voxels))
                    if not hasattr(self, "_last_unique_voxel_total"):
                        self._last_unique_voxel_total = current_unique_voxel_total
                    self.metrics_payload["open_endedness/unique_voxels_discovered"] = float(
                        max(0, current_unique_voxel_total - self._last_unique_voxel_total)
                    )
                    self._last_unique_voxel_total = current_unique_voxel_total

    def update_trueskill_ranks(self, agent_a, agent_b, outcome):
        import math

        # Standard TrueSkill performance variance parameter
        beta = 25.0 / 6.0
        mu_a = agent_a.pop_ref.ts_mu[agent_a.idx].item() if agent_a.pop_ref else getattr(agent_a, "mu", 25.0)
        sigma_a = agent_a.pop_ref.ts_sigma[agent_a.idx].item() if agent_a.pop_ref else getattr(agent_a, "sigma", 8.33)
        mu_b = agent_b.pop_ref.ts_mu[agent_b.idx].item() if agent_b.pop_ref else getattr(agent_b, "mu", 25.0)
        sigma_b = agent_b.pop_ref.ts_sigma[agent_b.idx].item() if agent_b.pop_ref else getattr(agent_b, "sigma", 8.33)
        c = max(1e-3, math.sqrt(2 * beta**2 + sigma_a**2 + sigma_b**2))

        margin = (mu_a - mu_b) / c
        prob_raw = torch.sigmoid(torch.tensor([margin])).item()
        prob_a_wins = max(1e-4, min(1.0 - 1e-4, float(prob_raw)))

        v = math.exp(-0.5 * margin**2) / (math.sqrt(2 * math.pi) * prob_a_wins)
        w = v * (v + margin)
        new_mu_a = mu_a + (sigma_a**2 / c) * (outcome - prob_a_wins)

        variance_multiplier = max(1e-4, 1.0 - (sigma_a**2 / c**2) * w)
        new_sigma_a = math.sqrt(sigma_a**2 * variance_multiplier)

        if abs(outcome - 0.5) < 1e-3:
            new_mu_a *= 0.98
            new_sigma_a *= 0.98

        if agent_a.pop_ref:
            agent_a.pop_ref.ts_mu[agent_a.idx] = new_mu_a
            agent_a.pop_ref.ts_sigma[agent_a.idx] = new_sigma_a
        else:
            agent_a.mu = new_mu_a
            agent_a.sigma = new_sigma_a
            logging.debug(f"TrueSkill Update: Agent {agent_a.id} Mu: {new_mu_a:.2f}, Sigma: {new_sigma_a:.2f}")

    def _memory_maintenance(self):
        """
        Optimizes replay buffer states via Projected Gradient Descent (PGD)
        to minimize prediction error under the latent dynamics model.
        """
        sorted_entities = sorted(self.entities, key=lambda o: o.fitness, reverse=True)
        top_entities = sorted_entities[: max(1, len(self.entities) // 2)]

        for ent in top_entities:
            ent.agent_core._consolidate_memory()
            if hasattr(ent.agent_core, "causal_symbolic_reasoner"):
                ent.agent_core.causal_symbolic_reasoner.prune_old_connections()

        best_ent = top_entities[0]
        mem_ptr = best_ent.agent_core.memory.ring_ptr
        if mem_ptr > 128:
            try:
                aggregated_memories = []
                for ent in top_entities:
                    if ent.agent_core.memory.ring_ptr > 128:
                        idx = torch.randint(
                            0, ent.agent_core.memory.ring_ptr, (128 // len(top_entities),), device=MODEL_DEVICE
                        )
                        aggregated_memories.append(ent.agent_core.memory.ram_ring_buffer_ep[idx].to(torch.float32))

                if aggregated_memories:
                    old_memory_batch = torch.cat(aggregated_memories, dim=0)
                else:
                    sample_indices = torch.randint(0, mem_ptr, (128,), device=MODEL_DEVICE)
                    old_memory_batch = best_ent.agent_core.memory.ram_ring_buffer_ep[sample_indices].to(torch.float32)

                dyn_model = best_ent.agent_core.latent_dynamics
                optimizer = best_ent.agent_core.opt_causal

                dyn_model.train()
                optimizer.zero_grad()

                dummy_action = (
                    torch.ones(old_memory_batch.size(0), best_ent.agent_core.num_actions, device=MODEL_DEVICE)
                    / best_ent.agent_core.num_actions
                )

                if hasattr(best_ent.agent_core, "adversary_controller") and hasattr(
                    best_ent.agent_core, "adversary_module"
                ):
                    corrupted_dream = best_ent.agent_core.adversary_controller.apply_budgeted_warp(
                        best_ent.agent_core.adversary_module,
                        old_memory_batch,
                        best_ent.agent_core.jepa,
                        best_ent.agent_core.actor_critic,
                        consolidation_weight=torch.tensor([1.0], device=MODEL_DEVICE),
                    )
                else:
                    corrupted_dream = old_memory_batch

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    pred_mu, _ = dyn_model(corrupted_dream, dummy_action, return_successor=True)

                    for p in best_ent.agent_core.fuzzy_kb.parameters():
                        p.requires_grad = False
                    for p in best_ent.agent_core.causal_symbolic_reasoner.parameters():
                        p.requires_grad = False
                    for p in dyn_model.prior_net.parameters():
                        p.requires_grad = False

                    logical_conflict = best_ent.agent_core.fuzzy_kb.reason(pred_mu)
                    causal_penalty = best_ent.agent_core.causal_symbolic_reasoner(logical_conflict).norm(dim=-1).mean()
                    task_loss = F.mse_loss(pred_mu, old_memory_batch.detach())

                    l2_reg = torch.tensor(0.0, device=MODEL_DEVICE)
                    for param in dyn_model.parameters():
                        l2_reg += torch.norm(param, p=2)

                    total_loss = task_loss + (0.1 * causal_penalty) + 1e-5 * l2_reg

                with torch.no_grad():
                    concept_activations, sae_reconstruction = best_ent.agent_core.sae(old_memory_batch)
                    sae_error = F.mse_loss(sae_reconstruction, old_memory_batch, reduction="none").mean(dim=-1)
                    epistemic_gate_mask = sae_error < (sae_error.mean() + 2.0 * sae_error.std() + 1e-4)

                from vrl_framework.trainer.ppo_engine import metrics_aggregator

                metrics_aggregator.log(
                    {
                        "loss/memory_pgd_total": float(total_loss.item()),
                        "loss/memory_causal_penalty": float(causal_penalty.item()),
                        "loss/memory_task": float(task_loss.item()),
                        "cognition/epigenetic_filtered_ratio": float((~epistemic_gate_mask).float().mean().item()),
                    }
                )

                total_loss.backward()

                for p in best_ent.agent_core.fuzzy_kb.parameters():
                    p.requires_grad = True
                for p in best_ent.agent_core.causal_symbolic_reasoner.parameters():
                    p.requires_grad = True
                for p in dyn_model.prior_net.parameters():
                    p.requires_grad = True
                torch.nn.utils.clip_grad_norm_(dyn_model.parameters(), max_norm=0.5)
                optimizer.step()
                dyn_model.eval()

                valid_indices = torch.arange(old_memory_batch.size(0), device=MODEL_DEVICE)[epistemic_gate_mask]
                updated_cpu = pred_mu[epistemic_gate_mask].detach().cpu().numpy().astype(np.float16)

                def async_lmdb_update(indices, payload):
                    if (
                        getattr(best_ent.agent_core, "runtime_context", None) is None
                        or getattr(best_ent.agent_core.runtime_context, "lmdb_bank", None) is None
                    ):
                        return
                    if payload.shape[0] == 0:
                        return

                    if hasattr(best_ent.agent_core.runtime_context.lmdb_bank, "retrieve"):
                        _, scores = best_ent.agent_core.runtime_context.lmdb_bank.retrieve(
                            payload, top_k=1, return_scores=True
                        )
                        if scores and len(scores) > 0:
                            novelty_mask = np.array(scores) < 0.95
                            payload = payload[novelty_mask]
                            indices = indices[torch.from_numpy(novelty_mask).to(indices.device)]
                            if payload.shape[0] == 0:
                                return

                    champ_fitness = float(getattr(best_ent, "fitness", 0.0))
                    keys = [
                        (
                            f"utility_{champ_fitness:.2f}_step_{best_ent.pop_ref.generation}_"
                            f"hash_{hash(p.tobytes())}"
                        ).encode("utf-8")
                        for p in payload
                    ]

                    try:

                        if hasattr(best_ent.agent_core.runtime_context.lmdb_bank, "batch_write_semantic"):
                            best_ent.agent_core.runtime_context.lmdb_bank.batch_write_semantic(
                                keys, [p.tobytes() for p in payload], payload
                            )
                        else:
                            best_ent.agent_core.runtime_context.lmdb_bank.batch_write(
                                keys,
                                [p.tobytes() for p in payload],
                                torch.zeros(len(payload), dtype=torch.float32, device="cpu"),
                                torch.zeros(len(payload), dtype=torch.float32, device="cpu"),
                            )
                        if hasattr(best_ent.agent_core.runtime_context.lmdb_bank, "env") and hasattr(
                            best_ent.agent_core.runtime_context.lmdb_bank.env, "sync"
                        ):
                            best_ent.agent_core.runtime_context.lmdb_bank.env.sync()
                    except Exception as e:
                        logging.warning(f"LMDB Eviction/Write Lifecycle alert on Windows subsystem: {e}")

                if (
                    getattr(best_ent.agent_core, "runtime_context", None) is not None
                    and getattr(best_ent.agent_core.runtime_context, "io_worker", None) is not None
                ):
                    best_ent.agent_core.runtime_context.io_worker.submit(
                        async_lmdb_update, valid_indices.cpu(), updated_cpu
                    )
                elif getattr(best_ent.agent_core, "runtime_context", None) is not None:
                    async_lmdb_update(valid_indices.cpu(), updated_cpu)

            except Exception as e:
                logging.warning(f"Error during PGD experience replay perturbation: {e}")

    def timed_checkpoint(self):
        checkpoint_interval = 600
        current_time = time.time()
        if not hasattr(self, "_last_checkpoint_time"):
            self._last_checkpoint_time = self.start_reasoning_time
        if current_time - self._last_checkpoint_time >= checkpoint_interval:
            checkpoint_data = {
                "generation": self.generation,
                "best_score_so_far": self.best_score_so_far,
                "best_answer_so_far": self.best_answer_so_far,
                "current_question": self.current_question,
                "timestamp": current_time,
                "min_reasoning_time": getattr(self, "min_reasoning_time", 0.0),
                "max_reasoning_time": getattr(self, "max_reasoning_time", 0.0),
                "start_reasoning_time": getattr(self, "start_reasoning_time", current_time),
            }
            if (
                hasattr(self, "entities")
                and self.entities
                and hasattr(self.entities[0], "agent_core")
                and hasattr(self.entities[0].agent_core, "curriculum_state")
            ):
                checkpoint_data["curriculum_file_index"] = int(
                    self.entities[0].agent_core.curriculum_state.file_index.item()
                )
                checkpoint_data["curriculum_byte_offset"] = int(
                    self.entities[0].agent_core.curriculum_state.byte_offset.item()
                )
                checkpoint_data["embedded_curriculum_path"] = self.entities[
                    0
                ].agent_core.curriculum_state.retrieve_path("embedded_curriculum_path")
                checkpoint_data["embedded_offload_path"] = self.entities[0].agent_core.curriculum_state.retrieve_path(
                    "embedded_offload_path"
                )
                checkpoint_data["curriculum_chunk_size"] = int(
                    getattr(getattr(self, "trainer", None), "curriculum_chunk_size", 4096)
                )

            from vrl_framework.core.settings import LOGS_DIR

            checkpoint_path = os.path.join(LOGS_DIR, f"checkpoint_reasoning_gen{self.generation}.json")
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=4)
            logging.info(f"[timed_checkpoint] Checkpoint saved at {checkpoint_path}")
            self._last_checkpoint_time = current_time

    def log_metrics(self):
        self.update_metrics()

        if not hasattr(self, "gen_time_start"):
            self.gen_time_start = time.time()

        if self.generation % 10 == 0:
            gen_time = time.time() - self.gen_time_start
            self.metrics_payload["system/generation_time"] = gen_time
            self.gen_time_start = time.time()

            if True:
                trainer_step = float(self.generation)
                if (
                    hasattr(self, "entities")
                    and len(self.entities) > 0
                    and hasattr(self.entities[0].agent_core, "trainer")
                    and self.entities[0].agent_core.trainer is not None
                ):
                    trainer_step = float(self.entities[0].agent_core.trainer.global_train_step)
                self.metrics_payload["global_train_step"] = trainer_step

                from vrl_framework.trainer.ppo_engine import metrics_aggregator

                metrics_aggregator.log(self.metrics_payload)

        if "generation" not in self.metrics_payload:
            self.metrics_payload["generation"] = self.generation

        from vrl_framework.trainer.ppo_engine import metrics_aggregator

        if hasattr(metrics_aggregator, "global_payload_dump"):
            metrics_aggregator.global_payload_dump["generation"] = self.generation

        if self.generation > 0 and self.generation % 500 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    def checkpoint_simulation(self, force=False):
        if not force and (self.generation == 0 or self.generation % 800 != 0):
            return
        if hasattr(self, "trainer") and self.trainer is not None:
            from vrl_framework.core.settings import AGENTS_DIR

            ckpt_path = os.path.join(AGENTS_DIR, f"checkpoint_gen_{self.generation}.pt")
            self.trainer.save_checkpoint(ckpt_path)

            if hasattr(self.trainer, "agent_core"):
                try:
                    with torch.no_grad():
                        if hasattr(self.trainer.agent_core, "lora_registry"):
                            if (
                                hasattr(self, "batched_agents")
                                and getattr(self.batched_agents, "last_state_features_batch", None) is not None
                            ):
                                self.trainer.agent_core._offline_consolidation(
                                    self.batched_agents.last_state_features_batch.mean(dim=0, keepdim=True)
                                )
                            lora_path = os.path.join(AGENTS_DIR, f"lora_skill_gen_{self.generation}.pt")
                            lora_data = {k.cpu(): v.cpu() for k, v in self.trainer.agent_core.lora_registry.items()}
                            torch.save(lora_data, lora_path)
                            import logging

                            logging.info(f"Synchronous O-LoRA Skill Snapshot saved to: {lora_path}")
                except Exception as e:
                    import logging

                    logging.warning(f"Failed to synchronize LoRA skill snapshot: {e}")

    def natural_selection(self):
        """
        Applies MAP-Elites driven natural selection and crossover.
        """
        if len(self.entities) < 4:
            return 0

        if not hasattr(self, "map_elites_archive"):
            self.map_elites_archive = DCRL_MAP_Elites_Archive(resolution=20)

        for ent in self.entities:
            self.map_elites_archive.evaluate_and_archive(ent)

        alpha_coef, beta_coef = 0.1, 1.2
        combined_scores = [
            float(ent.fitness)
            - alpha_coef * (ent.agent_core.moe.num_experts**beta_coef)
            + 0.7 * len(ent.discovered_features)
            for ent in self.entities
        ]

        sorted_indices = np.argsort(combined_scores)[::-1]
        ranked_ents = [self.entities[i] for i in sorted_indices]

        top_tier = max(1, int(len(ranked_ents) * 0.25))
        bottom_tier = max(1, int(len(ranked_ents) * 0.25))

        elite_ent = ranked_ents[0]
        for i in range(len(ranked_ents) - bottom_tier, len(ranked_ents)):
            weak_ent = ranked_ents[i]

            relative_competence_gap = elite_ent.fitness - weak_ent.fitness
            if relative_competence_gap > 10.0:
                tau = 0.05
                with torch.no_grad():
                    for target_param, elite_param in zip(
                        weak_ent.agent_core.actor_critic.parameters(), elite_ent.agent_core.actor_critic.parameters()
                    ):
                        target_param.lerp_(elite_param, tau)

            success = self.map_elites_archive.sample_elite_and_mutate(
                weak_ent, mutation_rate=weak_ent.genetic_code.get("mutation_rate", 0.02)
            )

            if not success:
                elite_ent = ranked_ents[i % top_tier]
                if weak_ent.pop_ref is not None and elite_ent.pop_ref is not None:
                    with torch.no_grad():
                        weak_ent.pop_ref.population_gamma[weak_ent.idx].copy_(
                            elite_ent.pop_ref.population_gamma[elite_ent.idx]
                        )
                        weak_ent.pop_ref.population_beta[weak_ent.idx].copy_(
                            elite_ent.pop_ref.population_beta[elite_ent.idx]
                        )
                        weak_ent.pop_ref.population_masks[weak_ent.idx].copy_(
                            elite_ent.pop_ref.population_masks[elite_ent.idx]
                        )

            base_mut = CFG.MUTATION_RATE
            base_exp = 0.3

            weak_ent.genetic_code["mutation_rate"] = max(0.001, min(0.2, base_mut * random.uniform(0.8, 1.2)))
            weak_ent.genetic_code["exploration_factor"] = max(0.05, min(0.8, base_exp * random.uniform(0.8, 1.2)))
            weak_ent.fitness = 0.0  # Reset agent parameters to initial distribution

        leader = ranked_ents[0]
        if leader.pop_ref is not None:
            with torch.no_grad():
                tau_polyak = 0.001
                l_idx = leader.idx
                with torch.no_grad():
                    leader.agent_core.global_gamma.mul_(1.0 - tau_polyak).add_(
                        leader.pop_ref.population_gamma[l_idx], alpha=tau_polyak
                    )
                    leader.agent_core.global_beta.mul_(1.0 - tau_polyak).add_(
                        leader.pop_ref.population_beta[l_idx], alpha=tau_polyak
                    )

                leader.pop_ref.population_gamma[l_idx].copy_(torch.ones_like(leader.pop_ref.population_gamma[l_idx]))
                leader.pop_ref.population_beta[l_idx].copy_(torch.zeros_like(leader.pop_ref.population_beta[l_idx]))

        if hasattr(self, "map_elites_archive"):
            self.map_elites_archive.evaluate_and_archive(leader)

            for ent in ranked_ents[1:3]:
                self.map_elites_archive.evaluate_and_archive(ent)

        self.entities = ranked_ents

        return len(self.entities)

    def evolve_population(self):
        for ent in self.entities:
            ent.replicated_this_generation = False
        new_ents = []

        if hasattr(self.batched_agents, "replication_intents"):
            replication_mask = self.batched_agents.replication_intents
            if getattr(replication_mask, "any", lambda: False)():
                capable_indices = torch.where(replication_mask)[0].tolist()
                for idx in capable_indices:
                    if idx < len(self.entities):
                        origin_ent = self.entities[idx]

                        if origin_ent.energy > 2.0 and not origin_ent.replicated_this_generation:
                            cloned_agent = origin_ent.replicate(self)
                            new_ents.append(cloned_agent)
                            origin_ent.energy -= 1.0

        if len(self.entities) + len(new_ents) < MIN_POPULATION:
            deficit = MIN_POPULATION - (len(self.entities) + len(new_ents))
            if not self.entities:
                world_limits = torch.tensor(WORLD_DIM, dtype=torch.float32, device=MODEL_DEVICE)
                random_positions = (torch.rand((deficit, len(WORLD_DIM)), device=MODEL_DEVICE) * world_limits).long()

                for i in range(deficit):
                    genesis_agent = Agent(
                        position=random_positions[i],
                        agent_core=self.batched_agents.global_agent_core if hasattr(self, "batched_agents") else None,
                    )
                    genesis_agent.energy = DESIRED_ENERGY * float(torch.empty(1).uniform_(1.2, 3.5).item())
                    new_ents.append(genesis_agent)
            else:
                elite_ents = [ent for ent in self.entities if ent.energy > 0.5]
                if elite_ents:
                    elite_positions = torch.stack([ent.position for ent in elite_ents])
                    world_limits = torch.tensor(WORLD_DIM, device=MODEL_DEVICE, dtype=torch.long)

                    for _ in range(deficit):
                        if random.random() < 0.2:
                            new_pos = (torch.rand(len(WORLD_DIM), device=MODEL_DEVICE) * world_limits).long()
                        else:
                            origin_idx = random.randint(0, len(elite_positions) - 1)
                            variation = torch.randint(-4, 5, elite_positions[origin_idx].shape, device=MODEL_DEVICE)
                            new_pos = torch.clamp(
                                elite_positions[origin_idx] + variation,
                                torch.zeros_like(world_limits),
                                world_limits - 1,
                            )

                        cloned_agent = Agent(position=new_pos)
                        cloned_agent.energy = 1.0
                        new_ents.append(cloned_agent)
        self.entities.extend(new_ents)

        if hasattr(self, "batched_agents"):
            self.batched_agents.sync_with_entities_list(self.entities)

    def imitation_phase(self):
        """
        Resolves neighborhood interactions across the population using batched pairwise L1 distance computations.
        """
        if not hasattr(self, "batched_agents") or len(self.entities) < 2:
            return

        pop_ref = self.batched_agents
        active_agents = len(self.entities)

        with torch.no_grad():
            fitness_tensor = pop_ref.fitness[:active_agents]
            mean_fitness = fitness_tensor.mean()

            weak_mask = (fitness_tensor < 0.5 * mean_fitness) & (torch.rand(active_agents, device=MODEL_DEVICE) < 0.3)
            weak_indices = torch.where(weak_mask)[0]

            if weak_indices.numel() == 0:
                return

            positions = pop_ref.positions[:active_agents, :3].float()

            dist_matrix = torch.cdist(positions, positions, p=1.0)

            valid_neighbors_mask = (dist_matrix < 10.0) & ~torch.eye(
                active_agents, dtype=torch.bool, device=MODEL_DEVICE
            )

            max_neighbors_fitness = torch.where(
                valid_neighbors_mask,
                fitness_tensor.unsqueeze(0).expand(active_agents, -1),
                torch.tensor(-1e6, device=MODEL_DEVICE),
            ).max(dim=1)
            best_local_indices = max_neighbors_fitness.indices
            best_local_fitness = max_neighbors_fitness.values

            valid_imitation_mask = weak_mask & (best_local_fitness > fitness_tensor)
            valid_imitation_indices = torch.where(valid_imitation_mask)[0]

            if valid_imitation_indices.numel() > 0:
                for w_idx, b_idx in zip(valid_imitation_indices, best_local_indices[valid_imitation_mask]):
                    self.entities[w_idx].imitate(self.entities[b_idx])

    def touch_interaction(self):
        """
        Computes spatial collisions and executes energy transfers.
        Uses upper triangular masking on pairwise distances to avoid double-counting.
        """
        if not hasattr(self, "batched_agents"):
            return

        pop_ref = self.batched_agents
        positions = pop_ref.positions[:, :3]

        with torch.no_grad():
            distances = torch.cdist(positions, positions, p=1.0)
            interaction_mask = (distances < 5.0) & (distances > 0.0)
            interaction_mask = torch.triu(interaction_mask, diagonal=1)

            agent_i, agent_j = torch.where(interaction_mask)

            if len(agent_i) == 0:
                return

            energy_i = pop_ref.energies[agent_i]
            energy_j = pop_ref.energies[agent_j]

            transfer = 0.05 * torch.min(energy_i, energy_j)

            pop_ref.energies.scatter_add_(0, agent_i, transfer)
            pop_ref.energies.scatter_add_(0, agent_j, transfer)

        for idx_a, idx_b in zip(agent_i.tolist(), agent_j.tolist()):
            if idx_a < len(self.entities) and idx_b < len(self.entities):
                self.entities[idx_a].imitate(self.entities[idx_b])
                self.entities[idx_b].imitate(self.entities[idx_a])

    def advanced_social_interaction(self):
        """
        Pairs agents for interaction based on TrueSkill ratings.
        """
        if len(self.entities) < 2:
            return

        if hasattr(self, "batched_agents"):
            avg_fitness = self.batched_agents.fitness[: len(self.entities)].mean().item() if self.entities else 0.0
        else:
            avg_fitness = (
                torch.tensor([float(o.fitness) for o in self.entities], device=MODEL_DEVICE, dtype=torch.float32)
                .mean()
                .item()
                if self.entities
                else 0.0
            )

        num_pairs = len(self.entities) // 2
        for i in range(num_pairs):
            agent_idx = random.randint(0, len(self.entities) - 1)
            if hasattr(self, "batched_agents"):
                raw_rival = self.batched_agents.sample_opponent_trueskill(agent_idx)
                rival_idx = int(raw_rival) % len(self.entities)
            else:
                rival_idx = (agent_idx + 1) % len(self.entities)

            ent1 = self.entities[agent_idx]
            ent2 = self.entities[rival_idx]

            pos1_t = torch.tensor(ent1.position) if not isinstance(ent1.position, torch.Tensor) else ent1.position
            pos2_t = torch.tensor(ent2.position) if not isinstance(ent2.position, torch.Tensor) else ent2.position
            dist = torch.sum(torch.abs(pos1_t - pos2_t)).item()
            if dist < 8:
                ent1.exchange_knowledge(ent2)

                with torch.no_grad():
                    # SAE-bottlenecked latent state communication
                    sparse_A, _ = ent1.agent_core.sae(ent1.hidden_state.view(1, -1))

                    msg_A_decoded_by_B = ent2.agent_core.sae.decoder(sparse_A).squeeze(0)

                    ent2_perception = ent2.agent_core.sensory(ent2.perceive(self))
                    if hasattr(ent2.agent_core.opponent_model, "predict_agent_action"):
                        intent_pred_B = ent2.agent_core.opponent_model.predict_agent_action(
                            ent2_perception, msg_A_decoded_by_B
                        )
                        trust_gate_B = torch.sigmoid((ent2.mu - ent1.mu) / 5.0)
                        intent_pred_B = intent_pred_B * trust_gate_B
                        ent2.last_intent = intent_pred_B.squeeze(0).squeeze(0).to(MODEL_DEVICE)
                        fused_state_B = torch.cat([ent2.hidden_state, ent2.last_intent], dim=-1)
                        if hasattr(ent2.agent_core.opponent_model, "state_fusion"):
                            ent2.hidden_state = ent2.agent_core.opponent_model.state_fusion(fused_state_B)

                    sparse_B, _ = ent2.agent_core.sae(ent2.hidden_state.view(1, -1))
                    msg_B_decoded_by_A = ent1.agent_core.sae.decoder(sparse_B).squeeze(0)

                    ent1_perception = ent1.agent_core.sensory(ent1.perceive(self))
                    if hasattr(ent1.agent_core.opponent_model, "predict_agent_action"):
                        intent_pred_A = ent1.agent_core.opponent_model.predict_agent_action(
                            ent1_perception, msg_B_decoded_by_A
                        )
                        trust_gate_A = torch.sigmoid((ent1.mu - ent2.mu) / 5.0)
                        intent_pred_A = intent_pred_A * trust_gate_A
                        ent1.last_intent = intent_pred_A.squeeze(0).squeeze(0).to(MODEL_DEVICE)
                        fused_state_A = torch.cat([ent1.hidden_state, ent1.last_intent], dim=-1)
                        if hasattr(ent1.agent_core.opponent_model, "state_fusion"):
                            ent1.hidden_state = ent1.agent_core.opponent_model.state_fusion(fused_state_A)

                fit1 = ent1.fitness.item() if isinstance(ent1.fitness, torch.Tensor) else float(ent1.fitness)
                fit2 = ent2.fitness.item() if isinstance(ent2.fitness, torch.Tensor) else float(ent2.fitness)
                if fit1 > 1.5 * avg_fitness and fit2 < 0.5 * avg_fitness:
                    ent2.imitate(ent1)
                    if hasattr(self, "update_trueskill_ranks"):
                        self.update_trueskill_ranks(ent1, ent2, outcome=1.0)
                elif fit2 > 1.5 * avg_fitness and fit1 < 0.5 * avg_fitness:
                    ent1.imitate(ent2)
                    if hasattr(self, "update_trueskill_ranks"):
                        self.update_trueskill_ranks(ent2, ent1, outcome=1.0)
                else:
                    if hasattr(self, "update_trueskill_ranks"):
                        self.update_trueskill_ranks(ent1, ent2, outcome=0.5)

    def specialization_phase(self):
        for ent in self.entities:
            if hasattr(ent, "dynamic_specialization"):
                ent.dynamic_specialization()

    def close(self):
        """
        Terminates metrics and background I/O workers.
        """
        if hasattr(self, "_metrics_shutdown_event"):
            self._metrics_shutdown_event.set()
        if hasattr(self, "metrics_thread"):
            self.metrics_thread.join(timeout=2.0)
        if hasattr(self, "io_executor"):
            self.io_executor.shutdown(wait=False)

    def update_progress_display(self, total_generations):
        import sys

        if not hasattr(self, "_progress_initialized"):
            self._progress_initialized = True
            self.recent_gen_times = []
            self._last_gen_time = time.time()
            sys.stdout.write("\n" * 4)
            sys.stdout.flush()

        current_time = time.time()
        if self.generation > 0:
            self.recent_gen_times.append(current_time - self._last_gen_time)
            if len(self.recent_gen_times) > 50:
                self.recent_gen_times.pop(0)
        self._last_gen_time = current_time

        percent = (self.generation / max(1, total_generations)) * 100.0
        bar_length = 40
        filled_length = int(bar_length * self.generation // max(1, total_generations))
        bar = "█" * filled_length + "-" * (bar_length - filled_length)

        if len(self.recent_gen_times) > 0:
            avg_time_per_gen = sum(self.recent_gen_times) / len(self.recent_gen_times)
            remaining_seconds = avg_time_per_gen * (total_generations - self.generation)
        else:
            remaining_seconds = 0

        def format_duration(seconds):
            if seconds < 0:
                return "00h:00m:00s"
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            return f"{h:02d}h:{m:02d}m:{s:02d}s"

        if not hasattr(self, "total_env_steps"):
            self.total_env_steps = 0
            self._last_counted_gen = -1

        if self.generation > self._last_counted_gen:
            current_active = len(self.entities) if hasattr(self, "entities") else 0
            self.total_env_steps += current_active
            self._last_counted_gen = self.generation

        display_step = self.total_env_steps

        ckpt = (
            f" | {self.last_checkpoint_message}"
            if hasattr(self, "last_checkpoint_message") and self.last_checkpoint_message
            else ""
        )
        safe_bar = bar.replace("\u2588", "#")
        sys.stdout.write(
            f"\r\033[KEpoch: {self.generation}/{total_generations} | "
            f"Step: {display_step} | [{safe_bar}] {percent:.1f}% | "
            f"ETA: {format_duration(remaining_seconds)}{ckpt}"
        )
        sys.stdout.flush()

        if self.generation > 0 and self.generation % 10 == 0:
            if hasattr(self, "update_metrics"):
                self.update_metrics()

            if hasattr(self, "trainer") and hasattr(self.trainer, "global_train_step"):
                self.metrics_payload["global_train_step"] = self.trainer.global_train_step
            elif (
                hasattr(self, "entities")
                and len(self.entities) > 0
                and hasattr(self.entities[0].agent_core, "trainer")
                and self.entities[0].agent_core.trainer is not None
            ):
                self.metrics_payload["global_train_step"] = self.entities[0].agent_core.trainer.global_train_step

            from vrl_framework.trainer.ppo_engine import metrics_aggregator

            combined_payload = {}
            if hasattr(metrics_aggregator, "last_known"):
                combined_payload.update(metrics_aggregator.last_known)
            if hasattr(metrics_aggregator, "metrics"):
                combined_payload.update(metrics_aggregator.metrics)
            if hasattr(metrics_aggregator, "_last_step_metrics"):
                combined_payload.update(metrics_aggregator._last_step_metrics)
            if hasattr(metrics_aggregator, "global_payload_dump"):
                combined_payload.update(metrics_aggregator.global_payload_dump)

            combined_payload.update(self.metrics_payload)

            if hasattr(metrics_aggregator, "metrics"):
                metrics_aggregator.metrics.clear()
            if hasattr(metrics_aggregator, "_last_step_metrics"):
                metrics_aggregator._last_step_metrics.clear()

            if wandb.run is not None:
                if not getattr(wandb, "_vrl_axis_mapped", False):
                    wandb.define_metric("generation")
                    wandb.define_metric("*", step_metric="generation")
                    wandb._vrl_axis_mapped = True

                safe_payload = {}

                for k, v in combined_payload.items():
                    if isinstance(v, torch.Tensor):
                        try:
                            safe_payload[k] = float(v.item())
                        except Exception:
                            pass
                    elif isinstance(v, (int, float, np.number)):
                        if math.isfinite(v):
                            safe_payload[k] = float(v)
                    elif isinstance(v, str):
                        safe_payload[k] = v

                safe_payload["generation"] = int(self.generation)

                try:
                    wandb.log(safe_payload)
                except Exception:
                    pass

            if hasattr(self, "metrics") and hasattr(self.metrics, "log"):
                self.metrics.log(combined_payload)
            else:
                if hasattr(self, "trainer") and hasattr(self.trainer, "global_train_step"):
                    combined_payload["global_train_step"] = self.trainer.global_train_step
                elif (
                    hasattr(self, "entities")
                    and len(self.entities) > 0
                    and hasattr(self.entities[0].agent_core, "trainer")
                    and self.entities[0].agent_core.trainer is not None
                ):
                    combined_payload["global_train_step"] = self.entities[0].agent_core.trainer.global_train_step

                if wandb.run is not None:
                    combined_payload["generation"] = self.generation
                    wandb.log(combined_payload)

    def run_benchmarks(self):
        benchmark_seeds = [42, 1337, 2026]
        geometric_scores = []
        solution_diversity = []
        control_utility_deltas = []

        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

        for seed in benchmark_seeds:
            torch.manual_seed(seed)

            for ent in self.entities:
                achievements = []
                with torch.no_grad():
                    for _ in range(3):
                        # Evaluate policy robustness against adversarial input noise.
                        test_perception = torch.randn(1, 4, 7, 7, 7, device=MODEL_DEVICE) * 1.5

                        sensory_features = ent.agent_core.sensory(test_perception)
                        jepa_out = ent.agent_core.jepa(sensory_features)
                        latent_rep = jepa_out[0] if isinstance(jepa_out, tuple) else jepa_out

                        # Execute Monte Carlo Tree Search (MCTS) rollout using the learned transition model.
                        budget_on = PlanningBudget(1.0, 0, 3, 16, False, 0, True, False, False, 2, 1, 1.5, 0.5, 1)
                        trace_on = ent.agent_core.latent_mcts(
                            latent_rep,
                            ent.agent_core.jepa.predictor,
                            ent.agent_core.actor_critic,
                            torch.zeros(1, 768, device=MODEL_DEVICE),
                            budget_on,
                            torch.tensor([1.0], device=MODEL_DEVICE),
                            ent.agent_core.causal_symbolic_reasoner,
                        )
                        confidence_on = torch.softmax(trace_on.final_blended_logits, dim=-1).max().item()

                        # Base policy forward
                        budget_off = PlanningBudget(1.0, 0, 0, 0, False, 0, False, False, False, 0, 1, 1.5, 0.5, 1)
                        trace_off = ent.agent_core.latent_mcts(
                            latent_rep,
                            ent.agent_core.jepa.predictor,
                            ent.agent_core.actor_critic,
                            torch.zeros(1, 768, device=MODEL_DEVICE),
                            budget_off,
                            torch.tensor([1.0], device=MODEL_DEVICE),
                            ent.agent_core.causal_symbolic_reasoner,
                        )
                        confidence_off = torch.softmax(trace_off.final_blended_logits, dim=-1).max().item()

                        control_utility_deltas.append(confidence_on - confidence_off)

                        causal_stability = ent.agent_core.causal_symbolic_reasoner(latent_rep).norm().item()
                        s_i = min(1.0, (confidence_on * 0.5) + (causal_stability * 0.05))
                        achievements.append(s_i)

                n = len(achievements)
                sum_ln = sum(math.log(1.0 + s) for s in achievements)
                geometric_scores.append(math.exp(sum_ln / n) - 1.0)
                solution_diversity.append(ent.agent_core.evaluate_activation_diversity(ent.perceive(self)))

        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)


class TensorizedEnvironment:
    def __init__(self, num_agents=1024, device="cuda"):
        self.num_agents = num_agents
        self.device = torch.device(device)
        self.dt = 0.1
        self.bounds = (100.0, 100.0, 100.0)
        self.epoch_counter = 0
        self.physics_multiplier = 1.0

        self.positions = torch.rand((num_agents, 3), device=self.device) * 100.0
        self.velocities = torch.zeros((num_agents, 3), device=self.device)
        self.energies = torch.ones((num_agents,), device=self.device) * 100.0
        self.hps = torch.ones((num_agents,), device=self.device) * 100.0
        self.actions = torch.zeros((num_agents, 3), device=self.device)

        self.num_anomalies = 10
        self.anomalies = torch.rand((self.num_anomalies, 3), device=self.device) * 100.0
        self.anomaly_states = torch.ones((self.num_anomalies,), dtype=torch.int32, device=self.device)

        self.wp_positions = wp.from_torch(self.positions, dtype=wp.vec3)
        self.wp_velocities = wp.from_torch(self.velocities, dtype=wp.vec3)
        self.wp_energies = wp.from_torch(self.energies, dtype=wp.float32)
        self.wp_hps = wp.from_torch(self.hps, dtype=wp.float32)
        self.wp_actions = wp.from_torch(self.actions, dtype=wp.vec3)
        self.wp_anomalies = wp.from_torch(self.anomalies, dtype=wp.vec3)
        self.wp_anomaly_states = wp.from_torch(self.anomaly_states, dtype=wp.int32)
        self.wp_bounds = wp.vec3(*self.bounds)

    def step(self, action_tensor: torch.Tensor) -> torch.Tensor:
        self.epoch_counter += 1

        if not hasattr(self, "adversarial_generator"):
            self.adversarial_generator = ProceduralEnvironmentGenerator(num_anomalies=self.num_anomalies).to(
                self.device
            )

        if self.epoch_counter % 100 == 0 and hasattr(self, "hps"):
            spatial_vol = self.get_spatial_volume()
            fitness_variance = self.hps.var().item() if self.hps.size(0) > 1 else 1.0
            estimated_uncertainty = max(0.0, 1.0 - (fitness_variance / 100.0))

            if not hasattr(self, "leaky_uncertainty"):
                self.leaky_uncertainty = estimated_uncertainty
            self.leaky_uncertainty = 0.999 * self.leaky_uncertainty + 0.001 * estimated_uncertainty

            if not hasattr(self, "_generator_lock"):
                self._generator_lock = threading.Lock()

            with self._generator_lock:
                displacement, new_physics = self.adversarial_generator.generate_terrain(
                    spatial_vol, uncertainty_signal=self.leaky_uncertainty
                )

            self.anomalies = torch.clamp(self.anomalies + displacement, 0.0, 100.0)
            self.physics_multiplier = float(new_physics.item())

            fitness_buffer = self.hps.detach().clone()
            spatial_buffer = spatial_vol.detach().clone()

            def _async_generator_step(generator_module, hps_state, spatial_state, lock):
                """
                Executes background gradient optimization on a dedicated CUDA stream.
                Enforces thread-safe mutex acquisition preventing Autograd graph collision.
                """
                bg_stream = torch.cuda.Stream()
                with torch.cuda.stream(bg_stream):
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        with lock:
                            generator_module.optimize_regret(hps_state, spatial_state)
                bg_stream.synchronize()

            threading.Thread(
                target=_async_generator_step,
                args=(self.adversarial_generator, fitness_buffer, spatial_buffer, self._generator_lock),
                daemon=True,
            ).start()

        inactive_anomalies = self.anomaly_states == 0
        self.anomalies = torch.where(
            inactive_anomalies.unsqueeze(1), torch.rand_like(self.anomalies) * 100.0, self.anomalies
        )
        self.anomaly_states = torch.where(
            inactive_anomalies, torch.tensor(1, dtype=torch.int32, device=self.device), self.anomaly_states
        )

        wp_actions_mapped = wp.from_torch(action_tensor.contiguous(), dtype=wp.vec3)

        physics_stream = torch.cuda.Stream(device=self.device)
        physics_event = torch.cuda.Event()

        # Dispatch warp kernel to asynchronous CUDA stream
        with torch.cuda.stream(physics_stream):
            wp.launch(
                kernel=cost_kernel,
                dim=self.num_agents,
                inputs=[
                    self.wp_positions,
                    self.wp_velocities,
                    wp_actions_mapped,
                    self.wp_energies,
                    self.wp_hps,
                    self.wp_anomalies,
                    self.wp_anomaly_states,
                    self.wp_bounds,
                    self.dt,
                    self.physics_multiplier,
                    float(metabolic_curriculum.penalty_multiplier),
                ],
            )
            physics_event.record(physics_stream)

        physics_event.wait(torch.cuda.current_stream(device=self.device))

        if not hasattr(self, "prev_energies_for_vel"):
            self.prev_energies_for_vel = self.energies.clone()
        energy_velocity_raw = self.energies - self.prev_energies_for_vel
        self.last_energy_replenish_mean = energy_velocity_raw.mean().item()
        self.prev_energies_for_vel = self.energies.clone()

        inactive_mask = self.hps <= 0.0
        self.positions = torch.where(
            inactive_mask.unsqueeze(1), torch.rand_like(self.positions) * 100.0, self.positions
        )
        self.energies = torch.where(inactive_mask, torch.tensor(100.0, device=self.device), self.energies)
        self.hps = torch.where(inactive_mask, torch.tensor(100.0, device=self.device), self.hps)
        self.velocities = torch.where(inactive_mask.unsqueeze(1), torch.zeros_like(self.velocities), self.velocities)

        spatial_vol = self.get_spatial_volume()
        PlannerValidator.assert_no_nan_inf(spatial_vol, "TensorizedEnvironment_SpatialVolume")
        return spatial_vol

    def get_spatial_volume(self) -> torch.Tensor:
        """
        O(1) local neighbor resolution via wp.HashGrid KNN.

        Returns: [Batch, 4, grid_size, grid_size, grid_size]
        """
        b = self.positions.size(0)
        device = self.device

        if not hasattr(self, "prev_energies"):
            self.prev_energies = self.energies.clone()
        energy_delta = (self.energies - self.prev_energies) / 10.0

        if not USE_3D_VOXELS:
            # Raycast bounds and anomalies to compute volumetric depth profiles
            state_vec = torch.stack([self.energies / 100.0, self.hps / 100.0, energy_delta], dim=-1)

            # Spherical Fibonacci Lattice for isotropic ray distribution
            num_rays = 16
            if not hasattr(self, "_cached_ray_dirs"):
                indices = torch.arange(0, num_rays, dtype=torch.float32, device=device) + 0.5
                phi = torch.acos(1 - 2 * indices / num_rays)
                theta = math.pi * (1 + 5**0.5) * indices
                self._cached_ray_dirs = torch.stack(
                    [torch.cos(theta) * torch.sin(phi), torch.sin(theta) * torch.sin(phi), torch.cos(phi)], dim=-1
                ).unsqueeze(0)

            ray_dirs = self._cached_ray_dirs.expand(b, -1, -1)
            origins = self.positions.unsqueeze(1).expand(-1, num_rays, -1)

            active_anomalies = self.anomalies[self.anomaly_states == 1]
            if active_anomalies.size(0) > 0:
                expanded_anomalies = active_anomalies.unsqueeze(0).unsqueeze(0).expand(b, num_rays, -1, -1)
                expanded_origins = origins.unsqueeze(2).expand(-1, -1, active_anomalies.size(0), -1)

                L = expanded_anomalies - expanded_origins
                tca = torch.sum(L * ray_dirs.unsqueeze(2), dim=-1)

                d2 = torch.sum(L * L, dim=-1) - tca * tca
                radius_sq = 2.0 * 2.0

                valid_hits = (d2 < radius_sq) & (tca > 0)
                thc = torch.sqrt(torch.clamp(radius_sq - d2, min=0.0))
                t0 = tca - thc

                t0 = torch.where(valid_hits, t0, torch.tensor(float("inf"), device=device))
                min_t0, _ = torch.min(t0, dim=-1)
            else:
                min_t0 = torch.full((b, num_rays), float("inf"), device=device)

            # Slab intersection evaluation for environment bounds
            inv_dirs = 1.0 / (ray_dirs + 1e-8)
            t1 = (0.0 - origins) * inv_dirs
            t2 = (torch.tensor(self.bounds, device=device).unsqueeze(0).unsqueeze(0) - origins) * inv_dirs

            tmin = torch.min(t1, t2)
            tmax_walls, _ = torch.max(tmin, dim=-1)

            final_depth_map = torch.minimum(min_t0, tmax_walls) / 100.0

            flat_obs = torch.cat([state_vec, final_depth_map], dim=-1)
            return flat_obs.to(torch.bfloat16)

        grid_size = 7
        volume = torch.zeros((b, 4, grid_size, grid_size, grid_size), device=device, dtype=torch.bfloat16)
        center = grid_size // 2

        volume[:, 0, center, center, center] = (self.energies / 100.0).to(torch.bfloat16)
        volume[:, 0, center, center, center + 1] = (self.hps / 100.0).to(torch.bfloat16)
        volume[:, 0, center, center, center - 1] = energy_delta.to(torch.bfloat16)

        # Hash grid indexing for O(1) local neighbor resolution
        cutoff_radius = 10.0
        hash_grid = wp.HashGrid(dim_x=int(self.bounds[0]), dim_y=int(self.bounds[1]), dim_z=int(self.bounds[2]))
        hash_grid.build(self.wp_positions, cutoff_radius)

        k_neighbors = min(5, b - 1)

        topk_indices = torch.zeros((b, k_neighbors), dtype=torch.long, device=device)
        valid_knn_mask = torch.zeros((b, k_neighbors), dtype=torch.bool, device=device)

        wp_topk_indices = wp.from_torch(topk_indices, dtype=wp.int32)
        wp_valid_knn = wp.from_torch(valid_knn_mask, dtype=wp.int8)

        @wp.kernel
        def _fused_hash_query_kernel(
            grid: wp.uint64,
            positions: wp.array[wp.vec3],
            topk_idx: wp.array2d[wp.int32],
            valid_mask: wp.array2d[wp.int8],
            radius: wp.float32,
            max_neighbors: wp.int32,
        ):
            tid = wp.tid()
            pos = positions[tid]

            query = wp.hash_grid_query(grid, pos, radius)
            neighbor_count = int(0)

            for index in query:  # type: ignore[attr-defined]
                if neighbor_count >= max_neighbors:
                    break
                if index != tid:
                    topk_idx[tid, neighbor_count] = index  # type: ignore[index]
                    valid_mask[tid, neighbor_count] = wp.int8(1)  # type: ignore[index]
                    neighbor_count += 1

        wp.launch(
            kernel=_fused_hash_query_kernel,
            dim=b,
            inputs=[hash_grid.id, self.wp_positions, wp_topk_indices, wp_valid_knn, cutoff_radius, k_neighbors],
        )

        wp.synchronize()
        torch.cuda.synchronize(device)

        if k_neighbors > 0:
            b_indices = torch.arange(b, device=device).unsqueeze(1).expand(-1, k_neighbors)
            neighbor_positions = self.positions[topk_indices]
            relative_pos = neighbor_positions - self.positions.unsqueeze(1)

            grid_coords = torch.round(relative_pos).long() + center

            valid_grid_mask = (
                valid_knn_mask
                & (grid_coords[..., 0] >= 0)
                & (grid_coords[..., 0] < grid_size)
                & (grid_coords[..., 1] >= 0)
                & (grid_coords[..., 1] < grid_size)
                & (grid_coords[..., 2] >= 0)
                & (grid_coords[..., 2] < grid_size)
            )

            valid_b = b_indices[valid_grid_mask]
            valid_coords_flat = grid_coords[valid_grid_mask]

            # Project sparse neighbors to occupancy channel
            volume[valid_b, 1, valid_coords_flat[:, 0], valid_coords_flat[:, 1], valid_coords_flat[:, 2]] = 1.0

            neighbor_velocities = self.velocities[topk_indices]
            relative_vel = neighbor_velocities - self.velocities.unsqueeze(1)
            velocity_mags = torch.norm(relative_vel, dim=-1)[valid_grid_mask]

            volume[valid_b, 2, valid_coords_flat[:, 0], valid_coords_flat[:, 1], valid_coords_flat[:, 2]] = (
                velocity_mags.to(torch.bfloat16)
            )

        anomaly_positions = self.anomalies.unsqueeze(0).expand(b, -1, -1)
        relative_anomalies = anomaly_positions - self.positions.unsqueeze(1)
        anomaly_coords = torch.round(relative_anomalies).long() + center

        anomaly_mask = (
            (anomaly_coords[..., 0] >= 0)
            & (anomaly_coords[..., 0] < grid_size)
            & (anomaly_coords[..., 1] >= 0)
            & (anomaly_coords[..., 1] < grid_size)
            & (anomaly_coords[..., 2] >= 0)
            & (anomaly_coords[..., 2] < grid_size)
            & (self.anomaly_states.unsqueeze(0) == 1)
        )

        b_anom_idx, anom_idx = torch.where(anomaly_mask)
        valid_anom_coords = anomaly_coords[b_anom_idx, anom_idx]

        volume[b_anom_idx, 3, valid_anom_coords[:, 0], valid_anom_coords[:, 1], valid_anom_coords[:, 2]] = 1.0

        return volume

    def get_state(self, debug_mode: bool = False) -> torch.Tensor:
        """
        Returns:
            Tensor: [Batch, Channels] (velocities, energies, hps, energy_delta, relative_anomaly_pos)
        """
        if not debug_mode:
            raise RuntimeError("get_state is reserved strictly for debug_mode.")

        with torch.no_grad():
            sensory_radius = 25.0
            fov_angle = math.pi / 2.0
            b = self.positions.size(0)

            agent_positions = self.positions.unsqueeze(1)
            anomaly_positions = self.anomalies.unsqueeze(0).expand(b, -1, -1)

            relative_vecs = anomaly_positions - agent_positions
            distances = torch.norm(relative_vecs, dim=-1)

            speed = torch.norm(self.velocities, dim=-1, keepdim=True)
            fallback_dir = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).expand(speed.size(0), -1)
            facing_dirs = torch.where(speed > 1e-5, self.velocities / (speed + 1e-5), fallback_dir)

            relative_dirs = relative_vecs / (distances.unsqueeze(-1) + 1e-5)
            cos_sim = torch.sum(facing_dirs.unsqueeze(1) * relative_dirs, dim=-1)

            cone_mask = (distances <= sensory_radius) & (cos_sim >= math.cos(fov_angle / 2.0))
            visible_mask = cone_mask & (self.anomaly_states.unsqueeze(0) == 1)

            closest_anomaly_dist, closest_idx = torch.min(
                torch.where(visible_mask, distances, torch.tensor(1e4, device=self.device)), dim=-1
            )

            closest_anomaly_pos = torch.gather(
                anomaly_positions, 1, closest_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3)
            ).squeeze(1)
            relative_anomaly_pos = torch.where(
                (closest_anomaly_dist < 1e4).unsqueeze(-1),
                (closest_anomaly_pos - self.positions) / sensory_radius,
                torch.zeros_like(self.positions),
            )

            if not hasattr(self, "prev_energies"):
                self.prev_energies = self.energies.clone()

            energy_delta = (self.energies - self.prev_energies).unsqueeze(1) / 10.0

            return torch.cat(
                [
                    self.velocities / 10.0,
                    self.energies.unsqueeze(1) / 100.0,
                    self.hps.unsqueeze(1) / 100.0,
                    energy_delta,
                    relative_anomaly_pos,
                ],
                dim=1,
            )


def run_evolution_simulation(generations, load_path=None):
    world = VectorizedWorld4D()
    world.total_generations = generations
    if load_path is not None:
        ent = load_entity(load_path)
        world.entities.append(ent)
    else:
        logging.info("Initializing population...")
        temp_positions = []
        for _ in range(INIT_POPULATION):
            pos = [np.random.randint(0, dim) for dim in WORLD_DIM]
            temp_positions.append(pos)

        # Sample initial population fitness from a standard normal distribution.
        combined_scores = [np.random.normal(loc=1.0, scale=0.5) for _ in range(INIT_POPULATION)]
        sorted_indices = np.argsort(combined_scores)[::-1]

        for i in range(min(MAX_POPULATION, len(sorted_indices))):
            best_pos = temp_positions[sorted_indices[i]]
            elite_agent = Agent(best_pos)
            world.entities.append(elite_agent)

    # Synchronize high-level entity instances with pre-allocated tensor buffers.
    if hasattr(world, "batched_agents"):
        world.batched_agents.sync_with_entities_list(world.entities)

    import contextlib
    import gc

    @contextlib.contextmanager
    def isolated_initialization_context():
        """
        Context manager for garbage collection and CUDA cache clearing post-initialization.
        """
        try:
            yield
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    with isolated_initialization_context():
        initialization_matrix = torch.zeros(1, device=MODEL_DEVICE)
        del initialization_matrix

    logging.info("Starting simulation for %d generations.", generations)
    logging.info("Simulation staged. Awaiting execution or console commands.")
    return world


def console_listener(simulation):
    while getattr(simulation, "_listener_active", False):
        cmd = input("Enter command (pause/resume/stop): ").strip().lower()
        if cmd == "pause":
            simulation.pause()
            logging.info("Simulation paused.")
        elif cmd == "resume":
            simulation.resume()
            logging.info("Simulation resumed.")
        elif cmd == "stop":
            simulation.stop()
            logging.info("Simulation halted.")
            if hasattr(simulation, "checkpoint_simulation"):
                simulation.checkpoint_simulation(force=True)
            try:
                payload = {
                    "generation": simulation.generation,
                    "population_count": len(simulation.entities) if hasattr(simulation, "entities") else 64,
                }
                if hasattr(simulation, "metrics") and hasattr(simulation.metrics, "log"):
                    simulation.metrics.log(payload, step=simulation.generation)
                else:
                    if hasattr(simulation, "trainer") and hasattr(simulation.trainer, "global_train_step"):
                        payload["global_train_step"] = simulation.trainer.global_train_step
                    elif (
                        hasattr(simulation, "entities")
                        and len(simulation.entities) > 0
                        and hasattr(simulation.entities[0].agent_core, "trainer")
                        and simulation.entities[0].agent_core.trainer is not None
                    ):
                        payload["global_train_step"] = simulation.entities[0].agent_core.trainer.global_train_step

                    if hasattr(simulation, "entities") and len(simulation.entities) > 0:
                        try:
                            ent = simulation.entities[0]
                            core = ent.agent_core
                            epistemic_confidence = 0.0
                            empirical_error = 0.0

                            if hasattr(ent, "hidden_state") and isinstance(ent.hidden_state, torch.Tensor):
                                epistemic_confidence = torch.norm(ent.hidden_state.float(), p=2).item()

                            if (
                                hasattr(core, "jepa_surprisal_ema_tensor")
                                and core.jepa_surprisal_ema_tensor is not None
                            ):
                                surprisal_val = core.jepa_surprisal_ema_tensor
                                if isinstance(surprisal_val, torch.Tensor):
                                    empirical_error = surprisal_val.float().mean().item()
                                elif isinstance(surprisal_val, (float, int)):
                                    empirical_error = float(surprisal_val)

                            payload["metacognitive_calibration_error"] = abs(epistemic_confidence - empirical_error)
                        except Exception:
                            import logging

                            logging.getLogger(__name__).exception("Failed to compute planner predicted gain")

                    if wandb.run is not None:
                        payload["generation"] = simulation.generation

                        current_step = None
                        if hasattr(simulation, "trainer") and hasattr(simulation.trainer, "global_train_step"):
                            current_step = simulation.trainer.global_train_step
                        elif (
                            hasattr(simulation, "entities")
                            and len(simulation.entities) > 0
                            and hasattr(simulation.entities[0].agent_core, "trainer")
                            and simulation.entities[0].agent_core.trainer is not None
                        ):
                            current_step = simulation.entities[0].agent_core.trainer.global_train_step

                        if current_step is not None:
                            payload["global_train_step"] = current_step

                        if hasattr(simulation, "generation"):
                            payload["generation"] = simulation.generation
                            try:
                                from vrl_framework.trainer.ppo_engine import metrics_aggregator

                                metrics_aggregator.log({"generation": simulation.generation})
                            except Exception:
                                pass

                        wandb.log(payload)
            except Exception as e:
                logging.error(f"[W&B LOG ERROR] {e}")

            if hasattr(simulation, "trainer") and simulation.trainer and hasattr(simulation.trainer, "agent_core"):
                try:
                    with torch.no_grad():
                        if hasattr(simulation.trainer.agent_core, "lora_registry"):
                            if (
                                hasattr(simulation, "batched_agents")
                                and getattr(simulation.batched_agents, "last_state_features_batch", None) is not None
                            ):
                                simulation.trainer.agent_core._offline_consolidation(
                                    simulation.batched_agents.last_state_features_batch.mean(dim=0, keepdim=True)
                                )
                            lora_path = os.path.join(AGENTS_DIR, f"lora_skill_gen_{simulation.generation}.pt")
                            lora_data = {
                                k.cpu(): v.cpu() for k, v in simulation.trainer.agent_core.lora_registry.items()
                            }
                            torch.save(lora_data, lora_path)
                            logging.info(
                                "[AUTO-SAVE] O-LoRA Skill Snapshot for "
                                f"generation {simulation.generation} saved to: {lora_path}"
                            )
                except Exception as lora_auto_e:
                    logging.error(
                        "Failed to automatically save LoRA skills at "
                        f"generation {simulation.generation}: {lora_auto_e}"
                    )

            simulation.visualize_population(simulation.generation, force=True)
            if simulation.entities:
                best_ent = max(simulation.entities, key=lambda o: o.fitness)
                simulation.visualize_agent_core_structure(best_ent, simulation.generation, force=True)
            simulation.visualize_intelligence_metrics(force=True)

            # Break listener thread
            break
        else:
            logging.warning("Unknown command. Available: pause, resume, stop.")
