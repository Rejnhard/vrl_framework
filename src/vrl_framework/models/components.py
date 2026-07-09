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

import copy
import logging
import math
import random
from typing import List, Optional, Tuple

import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from vrl_framework.core.contracts import (
    ActorCriticOutput,
    Float,
    LatentState,
    PlanningBudget,
    ValueEstimation,
    beartype,
    jaxtyped,
)
from vrl_framework.core.settings import CFG, DESIRED_ENERGY, MODEL_DEVICE, TRAIN_CFG
from vrl_framework.environment.replay_buffer import EpisodicReplayBuffer
from vrl_framework.math_ops.geometry import LorentzGeometry, RMSNorm

logger = logging.getLogger(__name__)


class SharedWorkspaceBottleneck(nn.Module):
    """Regularizes phase coherence across parallel latent representations via an information bottleneck."""

    def __init__(self, embed_dim: int, num_subsystems: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.amplitude_proj = nn.Linear(embed_dim, embed_dim)
        self.phase_proj = nn.Linear(embed_dim, embed_dim)
        self.coupling_strength = nn.Parameter(torch.tensor(0.8))
        self.broadcast_norm = nn.LayerNorm(embed_dim)

        self.register_buffer("latest_thermodynamic_cost", torch.zeros(1))

    def forward(
        self, subsystem_states: torch.Tensor, epistemic_uncertainty: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        amplitudes = F.softplus(self.amplitude_proj(subsystem_states))
        phases = self.phase_proj(subsystem_states) * math.pi

        with torch.no_grad():
            phase_coherence = torch.abs(torch.mean(torch.exp(1j * phases.float()), dim=1)).to(subsystem_states.dtype)
            entropy_change = -torch.mean(phase_coherence * torch.log(phase_coherence + 1e-5))
            self.latest_thermodynamic_cost = (entropy_change**2) * self.embed_dim

        if epistemic_uncertainty is None:
            epistemic_uncertainty = subsystem_states.new_ones(subsystem_states.size(0), subsystem_states.size(1), 1)

        # Apply top-1 masking to attention scores.
        attention_scores = amplitudes.mean(dim=-1, keepdim=True) * epistemic_uncertainty
        winner_idx = torch.argmax(attention_scores, dim=1, keepdim=True)

        core_mask = torch.zeros_like(attention_scores).scatter_(1, winner_idx, 1.0)
        periphery_mask = 1.0 - core_mask

        core_signal = amplitudes * core_mask
        periphery_signal = amplitudes * periphery_mask

        core_phase = torch.gather(phases, 1, winner_idx.expand(-1, -1, phases.size(-1)))

        # Regularize non-dominant signals via variance bounds.
        complex_core = core_signal * torch.cos(core_phase)
        complex_periphery = periphery_signal * torch.cos(phases + torch.randn_like(phases) * 0.5)

        broadcast_vector = torch.sum((complex_core * 0.9) + (complex_periphery * 0.1), dim=1)
        return self.broadcast_norm(broadcast_vector)


class StochasticPolicy(nn.Module):
    """Projects continuous representations into a categorical action distribution modulated by dynamic temperature."""

    def __init__(self, num_actions: int):
        super().__init__()
        self.num_actions = num_actions
        self.register_buffer("global_step", torch.tensor(0.0))
        self.register_buffer("temperature", torch.tensor(1.5))
        self.min_temperature = 0.05

    def update_temperature(self, step: torch.Tensor, surprisal: torch.Tensor, homeostatic_divergence: torch.Tensor):
        step_tensor = torch.as_tensor(step, dtype=self.global_step.dtype, device=self.global_step.device)
        current_step = torch.where(step_tensor == 0, self.global_step + 1.0, step_tensor)
        self.global_step.copy_(current_step)

        epistemic_err = (
            torch.as_tensor(surprisal, dtype=torch.float32, device=self.global_step.device)
            .detach()
            .nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
            .clamp(min=0.0, max=5.0)
        )
        somatic_err = (
            torch.as_tensor(homeostatic_divergence, dtype=torch.float32, device=self.global_step.device)
            .detach()
            .nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
            .clamp(min=0.0, max=5.0)
        )

        warmup = 10000.0
        evolution_amortizer = torch.where(
            self.global_step < warmup,
            torch.tensor(1.0, device=self.global_step.device),
            1.0 / (1.0 + (self.global_step - warmup) / 20000.0),
        )
        sys_stress = torch.tanh((0.7 * epistemic_err) + (0.3 * somatic_err)) * (2.9 * evolution_amortizer)

        dynamic_temp = torch.clamp(sys_stress + self.min_temperature, min=self.min_temperature, max=3.0)
        self.temperature.copy_(dynamic_temp)

    def forward(self, logits: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        temp_clamped = torch.clamp(self.temperature, min=self.min_temperature, max=10.0)

        scaled_logits = torch.clamp(logits / temp_clamped, min=-20.0, max=2.0)

        if self.training:
            noise = torch.empty_like(scaled_logits).exponential_().log()
            scaled_logits = scaled_logits - 0.05 * noise

        dist = torch.distributions.Categorical(logits=scaled_logits)

        if deterministic:
            action = torch.argmax(scaled_logits, dim=-1)
        else:
            action = dist.sample()

        return action, dist.log_prob(action)


class CPUOffloadedEMA:
    """Maintains an Exponential Moving Average of model parameters in host memory."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = (
                    param.detach().cpu().clone().pin_memory()
                    if torch.cuda.is_available()
                    else param.detach().cpu().clone()
                )

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        if not hasattr(self, "_dma_stream"):
            self._dma_stream = torch.cuda.Stream()

        torch.cuda.current_stream().wait_stream(self._dma_stream)
        with torch.cuda.stream(self._dma_stream):
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    ema_weight_gpu = self.shadow_params[name].to(param.device, non_blocking=True)
                    param_diff = torch.norm(ema_weight_gpu - param.detach())
                    adaptive_decay = torch.clamp(self.decay - (param_diff * 0.001), min=0.9)
                    ema_weight_gpu.lerp_(param.detach(), 1.0 - adaptive_decay)
                    self.shadow_params[name].copy_(ema_weight_gpu, non_blocking=True)

    @torch.no_grad()
    def load_shadow_into(self, model: torch.nn.Module) -> None:
        if not hasattr(self, "shadow_params"):
            raise RuntimeError("EMA shadow parameters missing. Cannot execute load_shadow_into.")

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow_params:
                    param.copy_(self.shadow_params[name].to(param.device))

    def materialize_shadow_copy(self, base_model: torch.nn.Module) -> torch.nn.Module:
        import copy

        shadow_model = copy.deepcopy(base_model)
        self.load_shadow_into(shadow_model)
        shadow_model.eval()
        return shadow_model

    @torch.no_grad()
    def evaluate_kl_drift(self, online_logits: torch.Tensor, model: nn.Module, inputs: torch.Tensor) -> float:
        """Computes Kullback-Leibler divergence between the online and EMA shadow models."""
        original_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        self.load_shadow_into(model)
        shadow_output = model(inputs)

        model.load_state_dict(original_state)
        shadow_logits = shadow_output.policy_logits if hasattr(shadow_output, "policy_logits") else shadow_output

        num_actions = online_logits.size(-1)
        shadow_logits_clean = shadow_logits[..., :num_actions]

        online_probs = F.softmax(online_logits.float(), dim=-1)
        shadow_log_probs = F.log_softmax(shadow_logits_clean.float(), dim=-1)

        kl_div = F.kl_div(shadow_log_probs, online_probs, reduction="batchmean")
        return kl_div.item()


class RunningMeanStd(nn.Module):
    """Exponential Moving Average (EMA) layer for computing streaming batch statistics."""

    def __init__(self, shape=(), momentum=0.01):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32, device=MODEL_DEVICE))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float32, device=MODEL_DEVICE))
        self.register_buffer("initialized", torch.tensor(0.0, device=MODEL_DEVICE))

    def update(self, x):
        if x.size(0) == 0:
            return

        with torch.no_grad():
            if not torch.isfinite(x).all():
                return
            x_filtered = torch.sgn(x) * torch.log1p(torch.abs(x))

            batch_mean = x_filtered.mean(dim=0)
            batch_var = (
                x_filtered.var(dim=0, unbiased=False) if x_filtered.size(0) > 1 else torch.zeros_like(batch_mean)
            )

            is_init = self.initialized > 0.5
            delta = batch_mean - self.mean
            new_mean = torch.where(is_init, self.mean + self.momentum * delta, batch_mean)
            new_var_step = torch.clamp(
                (1.0 - self.momentum) * (self.var + self.momentum * (delta**2)) + self.momentum * batch_var, min=1e-4
            )
            new_var = torch.where(is_init, new_var_step, batch_var)

            self.mean.copy_(new_mean)
            self.var.copy_(new_var)
            self.initialized.fill_(1.0)


class PopArtValueHead(nn.Module):
    """PopArt continuous value estimation head."""

    def __init__(self, in_features, out_features=1, momentum=0.01):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.rms = RunningMeanStd(shape=(out_features,), momentum=momentum)

    def forward(self, x):
        return self.linear(x)

    def update_stats(self, target):
        with torch.no_grad():
            if not torch.isfinite(target).all():
                return
            old_mean = self.rms.mean.clone()
            old_std = torch.sqrt(self.rms.var)
            self.rms.update(target)
            new_mean = self.rms.mean
            new_std = torch.sqrt(self.rms.var)

            self.linear.weight.mul_(old_std / new_std)
            self.linear.bias.mul_(old_std).add_(old_mean - new_mean).div_(new_std)

    def normalize(self, target):
        return (target - self.rms.mean) / (torch.sqrt(self.rms.var) + 1e-4)


class IntrinsicMotivationModule(nn.Module):
    """Implements RND-based intrinsic motivation with inverse dynamics filtering."""

    def __init__(self, input_dim=256, action_dim=16, hidden_dim=512, ema_decay=0.999):
        super().__init__()
        self.obs_rms = RunningMeanStd(shape=(input_dim,))
        self.reward_rms = RunningMeanStd(shape=())
        self.action_dim = action_dim

        # Action-conditioned predictor network.
        self.rnd_predictor = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.Mish(),
            nn.LayerNorm(hidden_dim),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Mish(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
        )

        # Inverse dynamics head suppressing environmental aleatoric noise.
        self.inverse_dynamics = nn.Sequential(
            layer_init(nn.Linear(input_dim * 2, hidden_dim)),
            nn.Mish(),
            nn.LayerNorm(hidden_dim),
            layer_init(nn.Linear(hidden_dim, action_dim)),
        )

        # Frozen spectral-normalized target network.
        from torch.nn.utils import spectral_norm

        self.rnd_target = nn.Sequential(
            spectral_norm(layer_init(nn.Linear(input_dim, hidden_dim))),
            nn.Mish(),
            nn.LayerNorm(hidden_dim),
            spectral_norm(layer_init(nn.Linear(hidden_dim, hidden_dim))),
            nn.Mish(),
            spectral_norm(layer_init(nn.Linear(hidden_dim, hidden_dim))),
        )

        for param in self.rnd_target.parameters():
            param.requires_grad = False

        self.register_buffer("global_step_counter", torch.zeros(1))

    def forward(self, state, next_state=None, action_one_hot=None, external_reward=None, abstract_knowledge=None):
        self.global_step_counter = self.global_step_counter + 1.0

        if self.training:
            self.obs_rms.update(state.detach())

        if next_state is not None and action_one_hot is not None:
            normalized_next_state = (next_state - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-5)
            normalized_next_state = torch.clamp(normalized_next_state, -5.0, 5.0)

            normalized_state = (state - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-5)
            normalized_state = torch.clamp(normalized_state, -5.0, 5.0)

            with torch.no_grad():
                target_features = torch.clamp(self.rnd_target(normalized_next_state), min=-20.0, max=20.0)

            predicted_features = torch.clamp(self.rnd_predictor(normalized_next_state), min=-20.0, max=20.0)

            forward_error = (
                F.mse_loss(predicted_features.float(), target_features.detach().float(), reduction="none")
                .mean(dim=-1)
                .to(predicted_features.dtype)
            )
            forward_error = torch.clamp(torch.nan_to_num(forward_error, nan=0.0, posinf=50.0), max=50.0)

            if not hasattr(self, "rnd_error_rms"):
                self.rnd_error_rms = RunningMeanStd(shape=())
            if self.training:
                self.rnd_error_rms.update(forward_error.detach())

            state_next_cat = torch.cat([normalized_state, normalized_next_state], dim=-1)
            pred_action_logits = self.inverse_dynamics(state_next_cat)
            pred_action_logits = torch.clamp(torch.nan_to_num(pred_action_logits, nan=0.0), min=-50.0, max=50.0)

            action_targets = torch.argmax(action_one_hot, dim=-1)
            inverse_loss = F.cross_entropy(pred_action_logits, action_targets, reduction="none")
            inverse_loss = torch.clamp(torch.nan_to_num(inverse_loss, nan=0.0), max=50.0)

            safe_std = torch.clamp(torch.sqrt(torch.nan_to_num(self.rnd_error_rms.var, nan=1.0)), min=1e-4)
            rnd_intrinsic_reward = torch.nan_to_num(forward_error.detach(), nan=0.0) / safe_std

            if getattr(self, "episodic_memory_buffer", None) is None or self.episodic_memory_buffer.size(
                1
            ) != normalized_next_state.size(0):
                self.episodic_memory_buffer = torch.zeros(500, *normalized_next_state.shape, device=state.device)
                self.episodic_memory_ptr = 0
                self.episodic_memory_count = 0

            ptr = self.episodic_memory_ptr
            self.episodic_memory_buffer[ptr] = normalized_next_state.detach()
            self.episodic_memory_ptr = (ptr + 1) % 500
            self.episodic_memory_count = min(self.episodic_memory_count + 1, 500)

            if self.episodic_memory_count > 10:
                valid_mem = self.episodic_memory_buffer[: self.episodic_memory_count]
                dists = torch.norm(valid_mem - normalized_next_state.detach(), dim=-1)
                episodic_novelty = torch.clamp(torch.min(dists, dim=0)[0], min=0.01)
            else:
                episodic_novelty = torch.tensor(1.0, device=state.device)

            gated_intrinsic_reward = rnd_intrinsic_reward * episodic_novelty

            empowerment_loss = forward_error.mean() + inverse_loss.mean()

            if self.training:
                self.reward_rms.update(gated_intrinsic_reward)

            safe_reward_std = torch.clamp(torch.sqrt(self.reward_rms.var), min=1e-2)
            z_scored_surprise = (gated_intrinsic_reward - self.reward_rms.mean) / safe_reward_std
            intrinsic_reward = F.softplus(z_scored_surprise) + 0.1

            intrinsic_reward = torch.clamp(intrinsic_reward, min=0.0, max=5.0)
            if external_reward is not None:
                final_synthesized_reward = external_reward + (intrinsic_reward * 0.1)
            else:
                final_synthesized_reward = intrinsic_reward

            return final_synthesized_reward, empowerment_loss


class TangentSpaceFSQ(nn.Module):
    """Applies Finite Scalar Quantization (FSQ) with level annealing."""

    anneal_step: torch.Tensor

    def __init__(self, levels: List[int] = [8, 5, 5, 5], dim: int = 256):
        super().__init__()
        self.target_levels = levels
        self.dim = dim

        self.register_buffer("current_levels", torch.full((len(levels),), 2.0, dtype=torch.float32))
        self.register_buffer("target_levels_tensor", torch.tensor(levels, dtype=torch.float32))
        self.register_buffer("anneal_step", torch.tensor(0.0, dtype=torch.float32))

        num_basis = len(levels)
        self.basis_proj = nn.Linear(dim - 1, num_basis)

        self.basis_correction = nn.Sequential(nn.Linear(num_basis, dim - 1), nn.Mish())
        self.basis_unproj = nn.Linear(dim - 1, dim - 1)
        self.eps = 1e-4

    @torch.amp.autocast(device_type="cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if self.training:
            self.anneal_step += 1.0
            progress = torch.clamp(self.anneal_step / 50000.0, 0.0, 1.0)
            self.current_levels = 2.0 + (self.target_levels_tensor - 2.0) * progress

        x_f32 = x.float()
        spatial_f32 = x_f32[..., 1:]

        norm_raw_f32 = torch.sqrt(torch.sum(spatial_f32**2, dim=-1, keepdim=True) + 1e-8)
        norm_f32 = torch.clamp(norm_raw_f32, min=1e-6)

        scale_in = torch.where(norm_f32 < 1e-3, 1.0 - (norm_f32**2) / 6.0, torch.arcsinh(norm_f32) / norm_f32)
        x_flat_f32 = spatial_f32 * scale_in.detach()

        z = self.basis_proj(x_flat_f32)

        active_levels = torch.round(self.current_levels).detach()
        half_l = (active_levels - 1) / 2
        offset = torch.where(active_levels % 2 == 0, 0.5, 0.0)
        shift = offset / half_l

        z_bound = torch.tanh(z + shift) * half_l - offset

        z_q = torch.round(z_bound)
        z_q_ste = z + (z_q - z).detach()

        corrected_basis = self.basis_correction(z_q_ste)
        out_flat_f32 = self.basis_unproj(corrected_basis)

        out_norm_raw_f32 = torch.sqrt(torch.sum(out_flat_f32**2, dim=-1, keepdim=True) + 1e-8)
        out_norm_f32 = torch.clamp(out_norm_raw_f32, min=1e-6, max=5.0)

        temporal_f32 = torch.cosh(out_norm_f32)

        scale_out = torch.where(
            out_norm_f32 < 1e-3, 1.0 + (out_norm_f32**2) / 6.0, torch.sinh(out_norm_f32) / out_norm_f32
        )
        out_spatial_f32 = out_flat_f32 * scale_out.detach()
        out_hyperbolic_f32 = torch.cat([temporal_f32, out_spatial_f32], dim=-1)

        out_hyperbolic = out_hyperbolic_f32.to(x.dtype)

        quantized = LorentzGeometry.project(out_hyperbolic)

        commitment_loss = F.mse_loss(z_q.detach(), z_bound)

        return quantized, commitment_loss, z_q


class DualStreamJEPACore(nn.Module):
    """Implements Joint Embedding Predictive Architecture (JEPA).

    Optimizes representations asymmetrically from online to EMA target.
    Requires L2-normalized input latent spaces to prevent dimensional collapse.
    """

    def __init__(self, latent_dim=256, ema_decay=0.996):
        super().__init__()
        self.latent_dim = latent_dim
        self.ema_decay = ema_decay

        self.fsq_encoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim)
        )
        self.quantizer = TangentSpaceFSQ(levels=[8, 5, 5, 5], dim=latent_dim)

        self.fp16_encoder = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Linear(512, latent_dim)
        )

        self.target_encoder = copy.deepcopy(self.fp16_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.cross_attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=4, batch_first=True)

        self.predictor = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(latent_dim, 512),
                    nn.LayerNorm(512),
                    nn.GELU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, latent_dim),
                    nn.LayerNorm(latent_dim),
                )
                for _ in range(3)
            ]
        )

        self.semantic_to_physics_projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.Mish(), nn.Linear(latent_dim, latent_dim)
        )
        self.temperature = nn.Parameter(torch.tensor(0.07))

        self.register_buffer("epistemic_anchor_physics", torch.zeros(1024, latent_dim))
        self.register_buffer("epistemic_anchor_semantic", torch.zeros(1024, latent_dim))
        self.register_buffer("anchor_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("anchor_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("epistemic_variance_ema", torch.tensor(1.0))

        self.register_buffer("global_target_queue", torch.zeros(1024, latent_dim))
        self.register_buffer("global_target_ptr", torch.tensor(0, dtype=torch.long))

    def update_target_network(self):
        """Updates the target encoder via Exponential Moving Average (EMA)."""
        with torch.no_grad():
            device = next(self.fp16_encoder.parameters()).device
            current_error = (
                self.proxy_surprisal.mean() if hasattr(self, "proxy_surprisal") else torch.tensor(0.0, device=device)
            )
            current_error = torch.as_tensor(current_error, dtype=torch.float32, device=device)

            if not hasattr(self, "prev_error_ema"):
                self.register_buffer("prev_error_ema", current_error.clone().detach())

            error_spike = current_error - self.prev_error_ema
            self.prev_error_ema = torch.lerp(self.prev_error_ema, current_error, 0.1)

            adaptive_decay = torch.where(
                error_spike > 10.0, torch.tensor(1.0, device=device), torch.tensor(self.ema_decay, device=device)
            )
            weight = 1.0 - adaptive_decay
            for online_param, target_param in zip(self.fp16_encoder.parameters(), self.target_encoder.parameters()):
                target_param.data = torch.lerp(target_param.data, online_param.data, weight)

    def forward(self, x):
        if x.dim() == 3:
            x = x.mean(dim=1)

        x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        x = torch.clamp(x, min=-20.0, max=20.0)

        fsq_proj = self.fsq_encoder(x)
        subconscious_state, vq_commitment_loss, _ = self.quantizer(fsq_proj)
        fp16_proj = self.fp16_encoder(x)

        subconscious_f32 = subconscious_state.float()
        fp16_proj_f32 = fp16_proj.float()

        sub_seq_f32 = LorentzGeometry.project(10.0 * torch.tanh(subconscious_f32 / 10.0)).unsqueeze(1)
        abs_seq_f32 = LorentzGeometry.project(10.0 * torch.tanh(fp16_proj_f32 / 10.0)).unsqueeze(1)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            attn_out, _ = self.cross_attention(query=abs_seq_f32 * 0.05, key=sub_seq_f32 * 0.05, value=sub_seq_f32)

        fused_state = LorentzGeometry.project(10.0 * torch.tanh((fp16_proj_f32 + attn_out.squeeze(1)) / 10.0)).to(
            fp16_proj.dtype
        )

        if self.training:
            preds = [head(fused_state) for head in self.predictor]
            online_pred = torch.mean(torch.stack(preds), dim=0)
            with torch.no_grad():
                target_proj = self.target_encoder(x)

            return fused_state, online_pred, target_proj, vq_commitment_loss

        with torch.no_grad():
            preds = [head(fused_state) for head in self.predictor]
            online_pred = torch.mean(torch.stack(preds), dim=0)
            target_proj = self.target_encoder(x)
        return fused_state, online_pred, target_proj, vq_commitment_loss


class LatentQuantizer(nn.Module):
    """Applies EMA-based vector quantization to latent representations."""

    def __init__(self, dict_size=256, latent_dim=256):
        super().__init__()
        self.dict_size = dict_size
        self.latent_dim = latent_dim
        self.decay = 0.99

        self.register_buffer("cluster_centers", torch.randn(dict_size, latent_dim))
        self.register_buffer("cluster_usage", torch.ones(dict_size))

    def update_correlations(self, latent_vectors: torch.Tensor, actions=None) -> None:
        """Updates VQ codebook via EMA.

        Args:
            latent_vectors: Target representations.
            actions: Unused.
        """
        with torch.no_grad():
            if not torch.isfinite(latent_vectors).all():
                return
            flat_latents = latent_vectors.view(-1, self.latent_dim)  # [N, D]

            latents_norm = F.normalize(flat_latents.float(), p=2, dim=-1).to(flat_latents.dtype)
            centers_norm = F.normalize(self.cluster_centers.float(), p=2, dim=-1).to(self.cluster_centers.dtype)

            similarities = torch.matmul(latents_norm, centers_norm.t())
            closest_cluster_idx = torch.argmax(similarities, dim=-1)

            one_hot_assignments = F.one_hot(closest_cluster_idx, num_classes=self.dict_size).float()
            cluster_counts = one_hot_assignments.sum(dim=0)

            self.cluster_usage.mul_(self.decay).add_(cluster_counts * (1 - self.decay))
            cluster_sums = torch.matmul(one_hot_assignments.t(), flat_latents)

            safe_counts = torch.clamp(cluster_counts.unsqueeze(-1), min=1.0)
            cluster_means = cluster_sums / safe_counts

            active_clusters = (cluster_counts > 0).unsqueeze(-1).expand_as(self.cluster_centers)
            new_centers = torch.where(active_clusters, cluster_means, self.cluster_centers)
            self.cluster_centers.mul_(self.decay).add_(new_centers * (1 - self.decay))

            usage_ratio = self.cluster_usage / torch.clamp(self.cluster_usage.sum(), min=1.0)
            if not hasattr(self, "inactive_steps_counter"):
                self.register_buffer(
                    "inactive_steps_counter", torch.zeros(self.dict_size, device=latent_vectors.device)
                )

            is_active = usage_ratio >= 0.005
            self.inactive_steps_counter[is_active] = 0
            self.inactive_steps_counter[~is_active] += 1

            dead_mask = (usage_ratio < 0.005) & (self.inactive_steps_counter > 500)
            if dead_mask.any() and latent_vectors.size(0) > 0:
                num_dead = dead_mask.sum().item()
                if num_dead > latent_vectors.size(0):
                    indices = torch.randint(0, latent_vectors.size(0), (num_dead,), device=latent_vectors.device)
                else:
                    indices = torch.randperm(latent_vectors.size(0), device=latent_vectors.device)[:num_dead]
                revival_vectors = flat_latents[indices]
                self.cluster_centers[dead_mask] = revival_vectors + torch.randn_like(revival_vectors) * 0.01
                self.cluster_usage[dead_mask] = self.cluster_usage.mean()
                self.inactive_steps_counter[dead_mask] = 0

    def get_cluster_id(self, latent_vector):
        """Computes closest codebook cluster assignment via cosine similarity.

        Args:
            latent_vector: Continuous latent tensor of shape [..., D].

        Returns:
            Tuple containing the best cluster index and its similarity score.
        """
        with torch.no_grad():
            latent_norm = F.normalize(latent_vector.squeeze().float(), p=2, dim=-1).to(latent_vector.dtype)
            centers_norm = F.normalize(self.cluster_centers.float(), p=2, dim=-1).to(self.cluster_centers.dtype)

            similarities = torch.matmul(latent_norm, centers_norm.t())
            best_score, best_idx = torch.max(similarities, dim=-1)

            return best_idx.item(), best_score.item()


class FrequencyDomainBinding:
    """
    Vector Symbolic Architecture (VSA) operations via frequency-domain mappings.
    Implements circular convolution binding.
    """

    @staticmethod
    def normalize_spherical(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """Projects vectors onto a hypersphere via L2 normalization."""
        return F.normalize(x, p=2, dim=-1, eps=eps)

    @staticmethod
    def bind(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Computes circular convolution via the Convolution Theorem (FFT -> Element-wise Mul -> IFFT)."""
        u_f32 = u.float()
        v_f32 = v.float()

        u_fft = torch.fft.rfft(u_f32, dim=-1)
        v_fft = torch.fft.rfft(v_f32, dim=-1)
        bound_fft = u_fft * v_fft
        bound_spatial = torch.fft.irfft(bound_fft, n=u.size(-1), dim=-1)

        return FrequencyDomainBinding.normalize_spherical(bound_spatial.to(u.dtype))

    @staticmethod
    def unbind(bound_vector: torch.Tensor, known_vector: torch.Tensor) -> torch.Tensor:
        """Computes circular correlation to extract bound components.

        Uses the complex conjugate of the known vector in the frequency domain.
        """
        bound_f32 = bound_vector.float()
        known_f32 = known_vector.float()

        bound_fft = torch.fft.rfft(bound_f32, dim=-1)
        known_fft = torch.fft.rfft(known_f32, dim=-1)
        unbound_fft = bound_fft * torch.conj(known_fft)
        unbound_spatial = torch.fft.irfft(unbound_fft, n=bound_vector.size(-1), dim=-1)

        return FrequencyDomainBinding.normalize_spherical(unbound_spatial.to(bound_vector.dtype))

    @staticmethod
    def bundle(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Aggregates multiple representations via vector addition and renormalizes."""
        return FrequencyDomainBinding.normalize_spherical(x + y)


class NeuralPredicate(nn.Module):
    """Projects subject-object pairs into relation embeddings."""

    def __init__(self, dim):
        super().__init__()
        self.projection = nn.Sequential(
            layer_init(nn.Linear(dim * 2, dim)), nn.Tanh()  # Enforce bounded [-1, 1] mapping for stable retrieval
        )

    def forward(self, subject, obj):
        combined = torch.cat([subject, obj], dim=-1)
        return self.projection(combined)


class DenseAssociativeMemory(nn.Module):
    """Continuous Hopfield Network layer using LogSumExp attention for dense memory retrieval."""

    def __init__(self, dim=256, num_predicates=4, max_facts=10000):
        super().__init__()
        self.dim = dim
        self.max_facts = max_facts
        self.beta = nn.Parameter(torch.tensor(math.sqrt(dim)))

        self.predicates = nn.ModuleList([NeuralPredicate(dim) for _ in range(num_predicates)])

        self.register_buffer("kb_subjects", torch.zeros(max_facts, dim))
        self.register_buffer("kb_objects", torch.zeros(max_facts, dim))
        self.register_buffer("kb_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("utility_scores", torch.zeros(max_facts))
        self.register_buffer("kb_age", torch.zeros(max_facts, dtype=torch.long))

        self.reasoning_query = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim)

    def evaluate_truth_gate(self, fact_tensor):
        """Evaluates associative retrieval validity using LogSumExp energy bounded by dynamic temperature."""
        with torch.no_grad():
            ptr = self.kb_ptr.item()
            if ptr > 0:
                self.kb_age[:ptr] += 1
                inactive_mask = (self.kb_age[:ptr] > 5000) & (self.utility_scores[:ptr] < 0.1)
                if inactive_mask.any():
                    self.kb_subjects[:ptr][inactive_mask] = 0.0
                    self.kb_objects[:ptr][inactive_mask] = 0.0
                    self.utility_scores[:ptr][inactive_mask] = 0.0

            if ptr == 0:
                return 1.0

            valid_memory = self.kb_subjects[:ptr]
            sim = torch.matmul(fact_tensor.float(), valid_memory.t().float())
            energy = torch.logsumexp(self.beta.float() * sim, dim=-1)

            if not hasattr(self, "recent_adds_ema"):
                self.register_buffer("recent_adds_ema", torch.tensor(1.0, device=fact_tensor.device))

            dynamic_temp = torch.clamp(1.0 - (self.recent_adds_ema * 0.1), min=0.5)

            dynamic_energy_baseline = energy.mean().detach() if energy.numel() > 0 else 0.0
            threshold = dynamic_energy_baseline + (
                torch.clamp(torch.tensor(math.log(ptr + 1.0), device=fact_tensor.device), max=5.0) * dynamic_temp * 0.1
            )
            is_valid = energy > threshold
            return is_valid.item() if fact_tensor.dim() == 1 else is_valid.float().mean().item()

    def evaluate_truth_gate_differentiable(self, fact_tensor):
        is_empty = self.kb_ptr == 0
        mask = torch.arange(self.max_facts, device=fact_tensor.device) < self.kb_ptr
        valid_memory = torch.where(mask.unsqueeze(-1), self.kb_subjects.detach(), torch.zeros_like(self.kb_subjects))

        sim = torch.matmul(fact_tensor.float(), valid_memory.t().float())
        sim = torch.where(mask, sim, torch.tensor(float("-inf"), device=sim.device))
        energy = torch.logsumexp(self.beta.float() * sim, dim=-1)

        if not hasattr(self, "recent_adds_ema"):
            self.register_buffer("recent_adds_ema", torch.tensor(1.0, device=fact_tensor.device))

        dynamic_temp = torch.clamp(1.0 - (self.recent_adds_ema * 0.1), min=0.5)
        dynamic_energy_baseline = energy.mean().detach() if energy.numel() > 0 else 0.0
        threshold = dynamic_energy_baseline + (
            torch.clamp(torch.log(self.kb_ptr.float() + 1.0), max=5.0) * dynamic_temp * 0.1
        )
        diff = (energy - threshold).to(fact_tensor.dtype)
        return torch.where(is_empty, torch.ones_like(diff), diff)

    def _clean(self, noisy_vector: torch.Tensor) -> torch.Tensor:
        ptr = self.kb_ptr.item()
        if ptr == 0:
            return noisy_vector

        valid_memory = self.kb_subjects[:ptr]
        sim = torch.matmul(noisy_vector.float(), valid_memory.t().float())
        attn = F.softmax(self.beta.float() * sim, dim=-1).to(noisy_vector.dtype)

        retrieved = torch.matmul(attn, valid_memory)

        if self.training:
            with torch.no_grad():
                top_idx = torch.argmax(sim, dim=-1)
                self.utility_scores.scatter_add_(
                    0, top_idx.view(-1), torch.ones_like(top_idx.view(-1), dtype=torch.float32)
                )

        return retrieved

    def add_fact(self, subject: torch.Tensor, pred_id: int, obj: torch.Tensor):
        truth_margin = self.evaluate_truth_gate_differentiable(subject)
        should_add = truth_margin.mean() >= 0.0

        current_ptr = self.kb_ptr
        is_full = current_ptr >= self.max_facts

        ptr = torch.where(is_full, torch.argmin(self.utility_scores), current_ptr).long()

        self.kb_subjects[ptr] = torch.where(should_add, subject.detach().view(-1), self.kb_subjects[ptr])
        self.kb_objects[ptr] = torch.where(should_add, obj.detach().view(-1), self.kb_objects[ptr])

        base_utility = torch.where(is_full, torch.tensor(0.0, device=subject.device), self.utility_scores[ptr])
        self.utility_scores[ptr] = torch.where(should_add, base_utility + 1.0, self.utility_scores[ptr])

        self.kb_age[ptr] = torch.where(
            should_add, torch.tensor(0, dtype=torch.long, device=subject.device), self.kb_age[ptr]
        )

        self.kb_ptr.copy_(torch.where(should_add & ~is_full, current_ptr + 1, current_ptr))

    def store_experience(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor):
        if state.dim() > 1:
            variances = state.var(dim=-1)
            best_idx = torch.argmax(variances)
            state = state[best_idx]
            action = action[best_idx]
            next_state = next_state[best_idx]

        bound_transition = FrequencyDomainBinding.bind(state, action)
        fact_vector = FrequencyDomainBinding.bundle(bound_transition, next_state)
        self.add_fact(state, pred_id=1, obj=fact_vector)

    def reason(self, state: torch.Tensor) -> torch.Tensor:
        if self.kb_ptr.item() == 0:
            return state

        self.utility_scores *= 0.99
        if state.dim() == 1:
            state = state.unsqueeze(0)

        accumulated_state = self.layer_norm(state)
        current_concept = self.reasoning_query(accumulated_state)

        num_hops = 3
        for _ in range(num_hops):
            retrieved_memory = self._clean(current_concept)
            accumulated_state = FrequencyDomainBinding.bundle(accumulated_state, retrieved_memory)
            current_concept = self.reasoning_query(accumulated_state)

        return state + accumulated_state


class ModelPredictivePlanner(nn.Module):
    """Implements a Latent Cross-Entropy Method (CEM) planner for action sequence optimization."""

    def __init__(self, num_actions=8, latent_dim=256, k_samples=64, elite_fraction=0.1, max_iters=3):
        super().__init__()
        self.k_samples = k_samples
        self.num_actions = num_actions
        self.elite_fraction = elite_fraction
        self.max_iters = max_iters
        self.num_elites = max(1, int(k_samples * elite_fraction))

        self.proposal_net = nn.Sequential(nn.Linear(latent_dim, 128), nn.Mish(), nn.Linear(128, num_actions))

    def forward(self, latent_context, dynamics_model, critic_model, fuzzy_kb, causal_reasoner, tau=None):
        batch_size = latent_context.size(0)
        base_logits = self.proposal_net(latent_context)
        safe_logits = torch.clamp(torch.nan_to_num(base_logits, nan=0.0), min=-20.0, max=20.0)

        if self.training:
            noise = torch.empty_like(safe_logits).exponential_().log()
            safe_logits = safe_logits - 0.1 * noise

        action_dist_probs = F.softmax(safe_logits, dim=-1)

        latent_exp = latent_context.repeat_interleave(self.k_samples, dim=0).detach()

        for _ in range(self.max_iters):
            sampled_actions_raw = torch.multinomial(action_dist_probs.repeat_interleave(self.k_samples, dim=0), 1)
            action_one_hot = F.one_hot(sampled_actions_raw.squeeze(-1), num_classes=self.num_actions).float()

            with torch.no_grad():
                pred_future_mu = dynamics_model(latent_exp, action_one_hot)
                pred_future_hyp = LorentzGeometry.project(torch.clamp(pred_future_mu, min=-15.0, max=15.0))

                logical_context = fuzzy_kb.reason(pred_future_hyp)
                causal_context = causal_reasoner(logical_context, action=action_one_hot)

                critic_context = torch.cat([pred_future_hyp, logical_context, causal_context], dim=-1)
                crit_out = critic_model(pred_future_hyp, critic_context)

                norm_value = crit_out.pessimistic_value.squeeze(-1)
                intrinsic_val = crit_out.intrinsic_value.squeeze(-1)

                epistemic_weight = 0.5
                efe_scores = norm_value + epistemic_weight * intrinsic_val

            efe_matrix = efe_scores.view(batch_size, self.k_samples)
            _, top_indices = torch.topk(efe_matrix, self.num_elites, dim=-1)

            batch_offsets = torch.arange(batch_size, device=latent_context.device).unsqueeze(-1) * self.k_samples
            global_elite_indices = (top_indices + batch_offsets).view(-1)

            elite_actions = action_one_hot[global_elite_indices].view(batch_size, self.num_elites, self.num_actions)

            new_probs = elite_actions.mean(dim=1)
            action_dist_probs = 0.2 * action_dist_probs + 0.8 * new_probs

        final_actions = torch.multinomial(action_dist_probs, 1).squeeze(-1)
        return F.one_hot(final_actions, num_classes=self.num_actions).float()


class TemperatureDecayScheduler:
    """Applies dynamic temperature and decay scheduling based on environment metrics."""

    def __init__(self, base_temperature=1.0, base_safety=0.5):
        self.base_temperature = base_temperature
        self.base_safety = base_safety
        self.penalty_integral = 0.0

    def step_scheduler(self, agent):
        if not hasattr(agent.agent_core, "exploration_layer"):
            return

        current_stress = 1.0 if agent.energy < (DESIRED_ENERGY * 0.4) else 0.0
        self.penalty_integral = 0.9 * self.penalty_integral + 0.1 * current_stress

        age_factor = min(1.0, agent.age / 1000.0)

        target_temp = self.base_temperature * (1.0 - (age_factor * 0.5)) * (1.0 - (self.penalty_integral * 0.8))
        agent.agent_core.exploration_layer.temperature.data.fill_(max(0.1, target_temp))

        if hasattr(agent.agent_core, "pid_controller"):
            agent.agent_core.pid_controller.safety_limit = self.base_safety * (1.0 + self.penalty_integral)


class ImplicitFunctionTheoremSolver(torch.autograd.Function):
    """Computes analytical gradients for DEQ models via IFT and Neumann series."""

    @staticmethod
    def forward(ctx, f, z_star, x):
        ctx.save_for_backward(z_star, x)
        ctx.f = f
        return z_star

    @staticmethod
    def backward(ctx, grad_output):
        z_star, x = ctx.saved_tensors
        f = ctx.f

        v_total = grad_output.clone()
        current_v = grad_output.clone()
        neumann_terms = 5

        with torch.enable_grad():
            z_star_req = z_star.detach().requires_grad_(True)
            x_req = x.detach().requires_grad_(x.requires_grad)
            f_z = f(z_star_req, x_req)

            prev_norm = current_v.norm(p=2, dim=-1, keepdim=True)
            active_mask = torch.ones_like(prev_norm, dtype=torch.bool)

            for step in range(neumann_terms):
                grad_tuple = torch.autograd.grad(
                    outputs=f_z,
                    inputs=z_star_req,
                    grad_outputs=current_v,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )

                step_v = grad_tuple[0].detach() if grad_tuple[0] is not None else torch.zeros_like(z_star_req)
                step_norm = step_v.norm(p=2, dim=-1, keepdim=True)

                divergence_mask = (step_norm > prev_norm) & active_mask
                safe_v = torch.where(divergence_mask, step_v * (prev_norm / (step_norm + 1e-8)), step_v)

                v_total = v_total + torch.where(active_mask, safe_v, torch.zeros_like(safe_v))

                active_mask = active_mask & (~divergence_mask)
                current_v = torch.where(active_mask, step_v, torch.zeros_like(step_v))
                prev_norm = torch.where(active_mask, step_norm, prev_norm)
                del grad_tuple

            torch.autograd.backward(f_z, v_total)

        return None, None, x_req.grad


class ImplicitDeepEquilibriumLayer(nn.Module):
    """Implements a Deep Equilibrium (DEQ) layer with constant-memory backpropagation via IFT."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, max_iter: int = 15, tol: float = 1e-3):
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol

        self.weight_z = nn.Parameter(torch.randn(output_dim, output_dim) / math.sqrt(output_dim))
        self.weight_x = nn.Parameter(torch.randn(output_dim, input_dim) / math.sqrt(input_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))

        self.survival_factor = nn.Parameter(torch.ones(1))
        self.synergy_factor = nn.Parameter(torch.tensor(1.0))

    def _f(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        w_z = self.weight_z.to(z.dtype)

        w_z_norm = torch.norm(w_z, p="fro")
        w_z_safe = w_z * torch.clamp(0.85 / (w_z_norm + 1e-6), max=1.0)

        w_x = self.weight_x.to(x.dtype)
        b = self.bias.to(x.dtype)
        return F.gelu(F.linear(z, w_z_safe) + F.linear(x, w_x, b))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        dim = self.weight_z.size(0)
        z = torch.zeros(batch_size, dim, device=x.device, dtype=x.dtype)

        convergence_steps = torch.full((batch_size,), float(self.max_iter), device=x.device)
        converged_mask = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        with torch.no_grad():
            momentum_z = torch.zeros_like(z)
            beta = 0.9  # Nesterov momentum coefficient

            for k in range(self.max_iter):
                z_nesterov = z + beta * momentum_z
                f_z = self._f(z_nesterov, x)

                step_diff = torch.norm(f_z - z, dim=-1)

                newly_converged = (step_diff < self.tol) & (~converged_mask)
                convergence_steps = torch.where(
                    newly_converged, torch.tensor(float(k), device=x.device), convergence_steps
                )
                converged_mask = converged_mask | newly_converged

                momentum_z = torch.where(converged_mask.unsqueeze(-1), momentum_z, beta * momentum_z + (f_z - z))
                z = torch.where(converged_mask.unsqueeze(-1), z, f_z)

        self.last_convergence_steps = convergence_steps

        z_star = ImplicitFunctionTheoremSolver.apply(self._f, z.detach(), x)

        out = z_star * torch.sigmoid(self.survival_factor)
        out = out * (1.0 + 0.1 * torch.tanh(self.synergy_factor * out.mean()))

        return out

    def learning_phase(self) -> None:
        """Adjusts scaling multipliers during active learning phase."""
        with torch.no_grad():
            self.survival_factor.mul_(0.99).add_(0.01)
            self.synergy_factor.mul_(0.99).add_(0.05)


class UniversalPatchingStem(nn.Module):
    """Maps 1D/2D/3D inputs to sequence projections."""

    def __init__(self, patch_dim=64, embed_dim=128):
        super().__init__()
        self.patch_dim = patch_dim
        self.embed_dim = embed_dim
        self.projection = nn.LazyLinear(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        if x.dim() == 4:
            # Reshape 2D spatial grid [B, C, H, W] into sequence patches.
            B, C, H, W = x.shape
            p = int(math.sqrt(self.patch_dim // C))
            if p == 0:
                p = 1

            pad_h = (p - H % p) % p
            pad_w = (p - W % p) % p
            x_padded = F.pad(x, (0, pad_w, 0, pad_h))

            unfolded = x_padded.unfold(2, p, p).unfold(3, p, p)
            x_seq = unfolded.permute(0, 2, 3, 1, 4, 5).contiguous().view(batch_size, -1, C * p * p)

        elif x.dim() == 5:
            # 3D Topology (e.g. Volumetric POMDP: B, C, D, H, W)
            B, C, D, H, W = x.shape
            p = int((self.patch_dim // C) ** (1 / 3.0))
            if p == 0:
                p = 1

            pad_d = (p - D % p) % p
            pad_h = (p - H % p) % p
            pad_w = (p - W % p) % p
            x_padded = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

            unfolded = x_padded.unfold(2, p, p).unfold(3, p, p).unfold(4, p, p)
            x_seq = unfolded.permute(0, 2, 3, 4, 1, 5, 6, 7).contiguous().view(batch_size, -1, C * p * p * p)

        else:
            # 1D/Arbitrary Topology Fallback
            x_flat = x.view(batch_size, -1)
            remainder = x_flat.size(1) % self.patch_dim
            if remainder != 0:
                x_flat = F.pad(x_flat, (0, self.patch_dim - remainder))
            x_seq = x_flat.view(batch_size, -1, self.patch_dim)

        projected_seq = self.projection(x_seq)
        return F.layer_norm(projected_seq, [projected_seq.size(-1)])


class UnifiedGatedLinearBackbone(nn.Module):
    """Sequential backbone abstracting over Linear State Space (SSM) and GRU architectures."""

    temporal_context: torch.Tensor

    def __init__(self, embed_dim=128, output_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        self.stem = UniversalPatchingStem(patch_dim=64, embed_dim=embed_dim)

        if CFG.USE_MAMBA:
            self.temporal_model = SelectiveStateSpaceModel(embed_dim)
        else:
            self.temporal_model = nn.GRU(input_size=embed_dim, hidden_size=embed_dim, batch_first=True)

        self.pcn_layer = ImplicitDeepEquilibriumLayer(
            input_dim=embed_dim, hidden_dim=512, output_dim=output_dim, max_iter=15
        )

        self.register_buffer("temperature", torch.tensor(1.0))
        self.register_buffer("temporal_context", torch.zeros(1, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes sequences and applies equilibrium predictive coding.

        Args:
            x: Input tensor of shape [B, C, H, W] (Spatial) or [B, L, D] (Sequential).
        """
        b = x.size(0)

        seq_embedded = self.stem(x)  # [B, N, D]

        if self.temporal_context.size(0) != b:
            self.temporal_context = torch.zeros(b, self.embed_dim, device=x.device, dtype=x.dtype)

        if CFG.USE_MAMBA:
            integrated_sequence = self.temporal_model(seq_embedded)
            global_feature = integrated_sequence[:, -1, :]
        else:
            integrated_sequence, _ = self.temporal_model(seq_embedded)
            global_feature = integrated_sequence[:, -1, :]

        self.temporal_context = (self.temporal_context.detach() * 0.9) + (global_feature * 0.1)
        surprisal_vector = self.pcn_layer(self.temporal_context)

        return FrequencyDomainBinding.normalize_spherical(surprisal_vector)


class MemoryModule(nn.Module):
    def __init__(self, runtime_context, mem_size):
        super().__init__()
        self.memory = EpisodicReplayBuffer(runtime_context=runtime_context, dim=mem_size)

    def update(self, new_info):
        if new_info.dim() > 1:
            new_info = new_info.mean(dim=0)
        self.memory.store(new_info)

    def get_memory(self):
        return self.memory.get_memory()

    def store_episode(self, episode):
        self.memory.store_episode(episode)

    def sample_replay(self, batch_size=8):
        """Delegates random batch sampling to the underlying episodic store."""
        return self.memory.sample_replay(batch_size)


class MotorModule(nn.Module):
    def __init__(self, input_dim, num_actions=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.Mish(), nn.LayerNorm(input_dim), nn.Linear(input_dim, num_actions)
        )

    def forward(self, x):
        return self.fc(x)


class CommunicationModule(nn.Module):
    """Implements differentiable communication via Gumbel-Softmax channel relaxation."""

    def __init__(self, input_dim, comm_dim, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.encoder = nn.Sequential(layer_init(nn.Linear(input_dim, comm_dim)), nn.Tanh())
        self.decoder = nn.Sequential(layer_init(nn.Linear(comm_dim, input_dim)), nn.Tanh())

        # Freeze first 3 indices to anchor the representation space.
        def _anchor_hook(grad):
            grad_clone = grad.clone()
            grad_clone[:3] = 0.0
            return grad_clone

        self.encoder[0].weight.register_hook(_anchor_hook)
        self.decoder[0].weight.register_hook(_anchor_hook)

    def encode(self, x):
        logits = self.encoder(x)
        message = F.gumbel_softmax(logits, tau=self.temperature, hard=True)

        if self.training:
            self.communication_tax = torch.norm(message, p=1, dim=-1).mean() * 0.05

        return message

    def decode(self, msg):
        return self.decoder(msg)


def custom_load_state_dict(model, state_dict):
    """Loads state_dict safely by filtering out mismatched tensor dimensions."""
    model_dict = model.state_dict()
    valid_dict = {}

    for key, param in state_dict.items():
        if key in model_dict:
            if model_dict[key].shape == param.shape:
                valid_dict[key] = param
            elif param.numel() == model_dict[key].numel():
                valid_dict[key] = param.view(model_dict[key].shape)
            else:
                logger.warning(
                    f"Shape mismatch rejected for {key}: expected {model_dict[key].shape}, got {param.shape}"
                )
                continue

    model.load_state_dict(valid_dict, strict=False)


class PIDLagrangianController:
    """Implements a PID Lagrangian controller for Constrained MDPs."""

    def __init__(self, kp=0.1, ki=0.01, kd=0.001, safety_limit=0.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.safety_limit = safety_limit
        self.error_integral = 0.0
        self.previous_error = 0.0
        self.lagrangian_multiplier = 0.0

    def calculate_multiplier(self, current_cost: float) -> float:
        """Computes multiplier step."""
        error = current_cost - self.safety_limit

        self.error_integral = 0.99 * self.error_integral + error

        error_derivative = error - self.previous_error

        penalty = (self.kp * error) + (self.ki * self.error_integral) + (self.kd * error_derivative)

        self.lagrangian_multiplier = max(0.0, penalty)
        self.previous_error = error
        return self.lagrangian_multiplier

    def regulate_agent(self, entity):
        """Applies a Lagrangian penalty to the agent's objective based on strict boundary constraints."""
        cost = max(0.0, 0.1 - entity.energy) + max(0.0, entity.energy - 5.0)

        if cost > 0:
            lambda_penalty = self.calculate_multiplier(cost)
            entity.fitness -= (lambda_penalty * cost) / (1.0 + lambda_penalty)

            if entity.energy < 0.1:
                entity.energy = 0.1
            if entity.energy > 5.0:
                entity.energy = 5.0
        return cost > 0


class GradientShortTermMemory(nn.Module):
    """Implements GRU-based short-term memory."""

    def __init__(self, input_dim=256, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.recurrent_core = nn.GRU(input_dim, hidden_dim, batch_first=True)

        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, stm_sequence):
        out_seq, _ = self.recurrent_core(stm_sequence)
        y = out_seq[:, -1, :]
        return self.proj(self.norm(y + stm_sequence[:, -1, :]))

    def consolidate_memory(self, stm_buffer):
        if not stm_buffer:
            device = next(self.parameters()).device
            return torch.zeros(self.hidden_dim, device=device)

        stm_seq = torch.stack([t.squeeze(0) for t in stm_buffer], dim=0).unsqueeze(0)
        consolidated = self.forward(stm_seq)
        return consolidated.squeeze(0)


class DualPathSparseAutoencoder(nn.Module):
    """Dual-Path Sparse Autoencoder decomposing features into Top-K sparse and continuous residual components."""

    def __init__(self, d_model=256, dict_size=4096, k=15, residual_dim=16):
        super().__init__()
        self.k = k
        self.dict_size = dict_size
        self.register_buffer("surprise_density_ema", torch.ones(self.dict_size))
        self.register_buffer("activation_ema", torch.ones(self.dict_size))
        self.register_buffer("pred_error_ema", torch.ones(self.dict_size))
        self.residual_dim = residual_dim

        self.encoder = nn.Sequential(nn.Linear(d_model, dict_size), RMSNorm(dict_size))
        self.decoder = nn.Linear(dict_size, d_model, bias=False)

        self.residual_encoder = nn.Sequential(nn.Linear(d_model, residual_dim), RMSNorm(residual_dim))
        self.residual_decoder = nn.Linear(residual_dim, d_model, bias=False)

        self.register_buffer("unit_usage_tracker", torch.zeros(dict_size, dtype=torch.long))
        self.register_buffer("max_historical_activation", torch.zeros(dict_size, dtype=torch.float32))

    def forward(self, x):
        """Computes Top-K sparse projection and continuous residual decomposition."""
        pre_acts = self.encoder(x)

        if self.training:
            pre_acts = pre_acts + (torch.randn_like(pre_acts) * 0.05)

        top_acts, top_indices = torch.topk(pre_acts, self.k, dim=-1)

        sparse_acts = torch.zeros_like(pre_acts)
        sparse_acts.scatter_(-1, top_indices, F.relu(top_acts))

        symbolic_recon = self.decoder(sparse_acts)

        residual_latent = self.residual_encoder(x)
        continuous_recon = self.residual_decoder(residual_latent)

        total_recon = symbolic_recon + continuous_recon

        x_safe = torch.clamp(torch.nan_to_num(x, nan=0.0), min=-20.0, max=20.0)
        recon_safe = torch.clamp(torch.nan_to_num(total_recon.detach(), nan=0.0), min=-20.0, max=20.0)

        x_hyp = LorentzGeometry.project(x_safe)
        recon_hyp = LorentzGeometry.project(recon_safe)

        batch_surprise = LorentzGeometry.distance(recon_hyp, x_hyp)

        if self.training:
            self.activation_ema *= 0.99
            self.activation_ema.index_add_(
                0, top_indices.view(-1), torch.ones_like(top_indices.view(-1), dtype=torch.float32)
            )

            if batch_surprise.dim() == 0:
                expanded_surprise = batch_surprise.unsqueeze(0).expand(top_indices.size(-1))
            else:
                expanded_surprise = batch_surprise.unsqueeze(1).expand(-1, top_indices.size(-1)).reshape(-1)

            self.surprise_density_ema *= 0.999
            self.surprise_density_ema.index_add_(
                0, top_indices.view(-1), expanded_surprise.to(self.surprise_density_ema.dtype)
            )

            if not hasattr(self, "error_reservoir"):
                self.error_reservoir = []

            if len(self.error_reservoir) < 1024:
                high_error_mask = batch_surprise > batch_surprise.mean()
                if high_error_mask.any():
                    self.error_reservoir.append(x[high_error_mask].detach())

        return sparse_acts, total_recon

    def execute_reservoir_resampling(self):
        """Processes offline dictionary resampling for dead latents."""
        if not hasattr(self, "error_reservoir") or len(self.error_reservoir) == 0:
            return

        valid_errors = torch.cat(self.error_reservoir, dim=0)
        self.apply_resampling_gradients(valid_errors)

        self.error_reservoir.clear()

    def apply_resampling_gradients(self, dataset_x):
        """Reinitializes dead dictionary atoms towards high-error activations."""
        with torch.no_grad():
            if not hasattr(self, "intrinsic_reward_ema"):
                return

            if not hasattr(self, "max_historical_activation"):
                self.register_buffer("max_historical_activation", torch.zeros(self.dict_size, device=dataset_x.device))

            self.max_historical_activation = torch.max(self.max_historical_activation, self.activation_ema)

            inactive_mask = (
                (self.activation_ema < 1e-3) & (self.pred_error_ema < 1e-4) & (self.max_historical_activation < 0.1)
            )
            num_inactive = inactive_mask.sum().item()

            if num_inactive > 0:
                _, recon = self.forward(dataset_x)
                errors = F.mse_loss(recon, dataset_x, reduction="none").mean(dim=-1)

                _, hard_indices = torch.topk(errors, min(num_inactive, errors.size(0)))
                hard_vectors = dataset_x[hard_indices]

                normalized_hard = F.normalize(hard_vectors, p=2, dim=-1)
                limit = hard_vectors.size(0)
                inactive_idx = torch.where(inactive_mask)[0][:limit]

                # Reinitialize inactive embeddings.
                with torch.no_grad():
                    self.encoder.weight[inactive_idx].copy_(normalized_hard)
                    self.decoder.weight[:, inactive_idx].copy_(normalized_hard.t())

                self.activation_ema[inactive_idx].fill_(1.0)
                self.pred_error_ema[inactive_idx].fill_(1.0)

                # Trigger hooks for optimizer state resets on reinitialized indices.
                active_hooks = getattr(self, "optim_state_hooks", [])
                for hook in active_hooks:
                    if callable(hook):
                        hook(inactive_idx)

                encoder_ref = getattr(self, "encoder", None)
                if encoder_ref is not None and getattr(encoder_ref.weight, "grad", None) is not None:
                    encoder_ref.weight.grad[inactive_idx].zero_()

                decoder_ref = getattr(self, "decoder", None)
                if decoder_ref is not None and getattr(decoder_ref.weight, "grad", None) is not None:
                    decoder_ref.weight.grad[:, inactive_idx].zero_()


class ThresholdAttentionOptimized(nn.Module):
    """Scaled Dot-Product Attention (SDPA) wrapper managing dynamic memory padding and alignment."""

    def __init__(self, dim, heads=4):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self, query: torch.Tensor, key: Optional[torch.Tensor] = None, value: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        original_shape_2d = False
        if query.dim() == 2:
            query = query.unsqueeze(1)
            original_shape_2d = True

        if key is None or value is None:
            key, value = query, query
        elif key.dim() == 2:
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)

        b, seq_q, _ = query.shape
        _, seq_kv, _ = key.shape

        q = self.q_proj(query).view(b, seq_q, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(b, seq_kv, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(b, seq_kv, self.heads, self.head_dim).transpose(1, 2)

        # Pad sequence lengths to the nearest power of 2 to enforce strict SDPA CUDA kernel memory alignment
        pad_q = (1 << (seq_q - 1).bit_length()) - seq_q if seq_q > 1 else 0
        pad_kv = (1 << (seq_kv - 1).bit_length()) - seq_kv if seq_kv > 1 else 0

        if pad_q > 0:
            q = F.pad(q, (0, 0, 0, pad_q))
        if pad_kv > 0:
            k = F.pad(k, (0, 0, 0, pad_kv))
            v = F.pad(v, (0, 0, 0, pad_kv))

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
            out_padded = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        out = out_padded[:, :, :seq_q, :].transpose(1, 2).contiguous().view(b, seq_q, self.dim)
        out = self.out_proj(out)

        if original_shape_2d:
            out = out.squeeze(1)
        return out


class DCRL_MAP_Elites_Archive:
    """Maintains a Quality Diversity MAP-Elites archive using a Latent SOM for behavioral routing."""

    def __init__(self, resolution=20, max_capacity=1024):
        self.resolution = resolution
        self.max_capacity = max_capacity
        self.archive = {}
        self.performances = {}
        self.som = LatentSOM_GPU(input_dim=256, map_size=resolution, dim=256).to(CFG.MODEL_DEVICE)

    def _compute_behavioral_descriptor(self, entity):
        with torch.no_grad():
            latent_state = entity.hidden_state.to(CFG.MODEL_DEVICE)
            if latent_state.dim() == 1:
                latent_state = latent_state.unsqueeze(0)
            distances = self.som(latent_state)
            winner_idx = torch.argmin(distances, dim=1).item()

        grid_x = int(self.som.grid[winner_idx][0].item())
        grid_y = int(self.som.grid[winner_idx][1].item())
        return (grid_x, grid_y)

    def evaluate_and_archive(self, entity):
        bd = self._compute_behavioral_descriptor(entity)
        fitness = entity.fitness

        if bd not in self.archive or fitness > self.performances[bd]:
            if len(self.archive) >= self.max_capacity and bd not in self.archive:
                weakest_bd = min(self.performances, key=self.performances.get)
                del self.archive[weakest_bd]
                del self.performances[weakest_bd]

            if entity.pop_ref is not None:
                idx = entity.idx
                cpu_state = {
                    "gamma": entity.pop_ref.population_gamma[idx].cpu().clone().half(),
                    "beta": entity.pop_ref.population_beta[idx].cpu().clone().half(),
                    "masks": entity.pop_ref.population_masks[idx].cpu().clone(),
                }
                self.archive[bd] = cpu_state
                self.performances[bd] = fitness

    def sample_elite_and_mutate(self, weak_ent, mutation_rate=0.02):
        if not self.archive or weak_ent.pop_ref is None:
            return False

        random_bd = random.choice(list(self.archive.keys()))
        elite_state = self.archive[random_bd]

        idx = weak_ent.idx
        pop_ref = weak_ent.pop_ref

        with torch.no_grad():
            gpu_gamma = elite_state["gamma"].to(
                pop_ref.population_gamma.device, dtype=torch.float16, non_blocking=True
            )
            gpu_beta = elite_state["beta"].to(pop_ref.population_beta.device, dtype=torch.float16, non_blocking=True)
            gpu_masks = elite_state["masks"].to(pop_ref.population_masks.device, non_blocking=True)

            pop_ref.population_gamma[idx].copy_(gpu_gamma + torch.randn_like(gpu_gamma) * mutation_rate)
            pop_ref.population_beta[idx].copy_(gpu_beta + torch.randn_like(gpu_beta) * mutation_rate)

            flip_mask = torch.rand_like(gpu_masks.float()) < (mutation_rate * 0.1)
            pop_ref.population_masks[idx].copy_(gpu_masks ^ flip_mask)

        return True


class LatentSOM_GPU(nn.Module):
    """Latent Self-Organizing Map (SOM)."""

    def __init__(self, input_dim, map_size=16, dim=128):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(map_size * map_size, dim, device=CFG.MODEL_DEVICE))
        grid_coords = (
            torch.stack(torch.meshgrid(torch.arange(map_size), torch.arange(map_size), indexing="ij"), dim=-1)
            .view(-1, 2)
            .float()
            .to(CFG.MODEL_DEVICE)
        )
        self.register_buffer("grid", grid_coords)
        self.sigma = nn.Parameter(torch.tensor(1.0, device=CFG.MODEL_DEVICE))
        self.register_buffer("usage_count", torch.zeros(map_size * map_size, device=CFG.MODEL_DEVICE))
        self.register_buffer("lsh_hashes", torch.randn(dim, 16, device=CFG.MODEL_DEVICE))

    def _prune_candidates(self, x):
        x_proj = (x @ self.lsh_hashes.to(x.dtype)) > 0
        proto_proj = (self.prototypes @ self.lsh_hashes.to(self.prototypes.dtype)) > 0

        matches = (x_proj.unsqueeze(1) == proto_proj.unsqueeze(0)).sum(dim=-1)
        return matches > (self.lsh_hashes.size(1) // 2)

    def forward(self, x):
        if x.dim() > 2:
            x = x.contiguous().view(x.size(0), -1)

        candidate_mask = self._prune_candidates(x)

        fallback_mask = (~candidate_mask.any(dim=-1)).unsqueeze(-1).expand_as(candidate_mask)
        candidate_mask = torch.where(fallback_mask, torch.ones_like(candidate_mask), candidate_mask)

        diff = x.unsqueeze(1) - self.prototypes.unsqueeze(0)  # [B, 1, D] - [1, K, D]
        distances = torch.norm(diff, p=2, dim=-1)

        # Mask out pruned candidates via static thresholding to avoid dynamic indexing
        max_dist = distances.max() + 1e4
        distances = torch.where(candidate_mask, distances, max_dist)

        winners = torch.argmin(distances, dim=1)
        delta = x - self.prototypes[winners]

        dist_sq = delta.pow(2).sum(dim=1, keepdim=True)
        sigma = self.sigma.clamp(0.5, 5.0)

        decay = torch.exp(-dist_sq / (2 * sigma**2))

        with torch.no_grad():
            lr = torch.clamp(0.1 / (1.0 + torch.log(1.0 + self.usage_count[winners].unsqueeze(-1))), min=1e-4)
            updates = lr * decay * delta

            target_indices = winners.unsqueeze(-1).expand(-1, x.size(-1))
            self.prototypes.scatter_add_(0, target_indices, updates)

            ones = torch.ones_like(winners, dtype=self.usage_count.dtype)
            self.usage_count.scatter_add_(0, winners, ones)

        return distances


class IdentityModule(nn.Module):
    """Neutral bypass module for disabled architectural components."""

    def forward(self, *args, **kwargs):
        if args and isinstance(args[0], torch.Tensor):
            return torch.zeros_like(args[0])
        return torch.tensor(0.0, device=MODEL_DEVICE)

    def process_audio(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def process_text(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class VectorQuantizer(nn.Module):
    """Projects dense spatial representations into a learned discrete embedding codebook."""

    def __init__(self, hidden_dim: int, codebook_size: int = 24):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size

        basis = torch.randn(codebook_size, hidden_dim)
        nn.init.orthogonal_(basis)
        self.codebook_embeddings = nn.Parameter(basis)

        self.projector = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.temperature = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        projected_x = self.projector(x)

        x_norm = F.normalize(projected_x, p=2, dim=-1)
        embeddings_norm = F.normalize(self.codebook_embeddings, p=2, dim=-1)

        similarities = torch.matmul(x_norm, embeddings_norm.t())

        safe_temp = torch.clamp(self.temperature, min=0.01, max=5.0)
        attention_weights = F.softmax((similarities / safe_temp).float(), dim=-1).to(similarities.dtype)

        quantized_output = torch.matmul(attention_weights, self.codebook_embeddings)

        return quantized_output, attention_weights


class QuantizedLinear(nn.Module):
    """8-bit quantized linear layer for reducing VRAM footprint.

    Uses bitsandbytes INT8 matrix multiplication, keeping weights in INT8
    and automatically extracting high-magnitude activation outliers to FP16.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Outlier threshold parameterization mapping to LLM.int8() GEMM stability conditions
        self.layer = bnb.nn.Linear8bitLt(in_features, out_features, bias=True, has_fp16_weights=False, threshold=6.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device_type = x.device.type
        if device_type == "cpu":
            return self.layer(x.float())

        with torch.autocast(device_type=device_type, dtype=torch.float16):
            return self.layer(x)


class TopKActivation(nn.Module):
    """Top-K sparsity activation. Retains highest activations and applies variance-preserving scalar."""

    def __init__(self, sparsity_k: int = 32, eps: float = 1e-5):
        super().__init__()
        self.k = sparsity_k
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_flat = x.contiguous().view(-1, shape[-1])  # [N, D]

        safe_k = min(self.k, x_flat.shape[-1])
        _, top_idx = torch.topk(x_flat, safe_k, dim=-1)  # [N, K]

        mask = torch.zeros_like(x_flat)
        mask.scatter_(-1, top_idx, 1.0)  # [N, D]

        sparse_x = x_flat * mask

        x_flat_f32 = x_flat.float()
        sparse_x_f32 = sparse_x.float()

        original_var = torch.var(x_flat_f32, dim=-1, keepdim=True, unbiased=False)
        sparse_var = torch.var(sparse_x_f32, dim=-1, keepdim=True, unbiased=False)

        variance_scale = torch.clamp(torch.sqrt(original_var / (sparse_var + self.eps)), max=10.0).to(x.dtype)
        sparse_x_scaled = sparse_x * variance_scale

        if len(shape) > 2:
            sparse_x_scaled = sparse_x_scaled.view(shape)

        return sparse_x_scaled


class HebbianLinear(nn.Module):
    """Linear layer combining learned structural parameters with an online, fast-weight plasticity trace."""

    def __init__(self, in_features, out_features, sparsity=0.8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity

        self.weight_base = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight_base, a=math.sqrt(5))

        self.alpha = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.constant_(self.alpha, 0.01)

        self.register_buffer("hebbian_trace", torch.zeros(out_features, in_features, dtype=torch.float32))
        self.register_buffer("weight_mask", torch.ones(out_features, in_features, dtype=torch.float32))

        num_zeros = int(self.weight_base.numel() * sparsity)
        indices = torch.randperm(self.weight_base.numel())[:num_zeros]
        self.weight_mask.view(-1)[indices] = 0.0

        self.register_buffer("last_x", torch.zeros(1, in_features, dtype=torch.float32))
        self.register_buffer("last_y", torch.zeros(1, out_features, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Detach inputs to isolate the fast-weight outer-product from policy gradients.
            self.last_x = x.detach().mean(dim=0, keepdim=True).float()

        effective_weight = (
            self.weight_base + self.alpha * self.hebbian_trace.to(self.weight_base.dtype)
        ) * self.weight_mask.to(self.weight_base.dtype)
        y = F.linear(x, effective_weight)  # [Batch, out_features]

        if self.training:
            self.last_y = y.detach().mean(dim=0, keepdim=True).float()  # [1, out_features]

        return y

    def apply_hebbian_update(self, surprise_signal: float, decay_rate: float = 0.9999):
        """Updates the local fast-weight trace using the outer product of inputs and outputs.

        Args:
            surprise_signal (float): Scalar scaling factor representing predictive mismatch or TD error.
            decay_rate (float): Decay factor for the historical fast-weight trace.
        """
        with torch.no_grad():
            input_activations = self.last_x.t()  # [D_in, 1]
            output_activations = self.last_y  # [1, D_out]

            hebb_update = (output_activations.t() @ input_activations.t()) * float(surprise_signal)
            self.hebbian_trace.mul_(decay_rate).add_(hebb_update * self.weight_mask)

    def update_topology(self, drop_fraction=0.1):
        """Prunes and reallocates sparse structural connections based on weight magnitude."""
        if self.weight_base.grad is None:
            return

        with torch.no_grad():
            self.weight_base.add_((self.alpha * self.hebbian_trace.to(self.weight_base.dtype)) * 0.1)
            self.hebbian_trace.fill_(0.0)

            active_weights = torch.abs(self.weight_base * self.weight_mask.to(self.weight_base.dtype))
            num_drop = int((self.weight_base.numel() * (1.0 - self.sparsity)) * drop_fraction)

            active_weights_flat = active_weights.view(-1).clone()
            active_weights_flat = torch.where(
                self.weight_mask.view(-1) == 0,
                torch.tensor(float("inf"), device=active_weights.device),
                active_weights_flat,
            )

            _, drop_indices = torch.topk(active_weights_flat, k=num_drop, largest=False)
            self.weight_mask.view(-1)[drop_indices] = 0.0
            self.weight_base.view(-1)[drop_indices] = 0.0

            inactive_weights = torch.abs(self.weight_base * (1.0 - self.weight_mask.to(self.weight_base.dtype)))
            _, grow_indices = torch.topk(inactive_weights.view(-1), k=num_drop, largest=True)
            self.weight_mask.view(-1)[grow_indices] = 1.0

            fan_in = self.weight_base.size(1)
            std = math.sqrt(2.0 / fan_in) * 0.01
            self.weight_base.view(-1)[grow_indices] = (
                torch.randn(len(grow_indices), device=self.weight_base.device) * std
            )

            if not hasattr(self, "warmup_mask"):
                self.register_buffer("warmup_mask", torch.ones_like(self.weight_base))
            self.warmup_mask.view(-1)[grow_indices] = 0.0

            def _warmup_hook(grad):
                return grad * self.warmup_mask.to(grad.dtype)

            if hasattr(self, "_hook_handle"):
                self._hook_handle.remove()
            self._hook_handle = self.weight_base.register_hook(_warmup_hook)


class StabilityGate(nn.Module):
    """Applies dynamic stability gating via hyperbolic tangent scaling for norm outliers."""

    def __init__(self, max_norm: float = 15.0):
        super().__init__()
        self.max_norm = max_norm
        self.register_buffer("running_norm", torch.ones(1) * 5.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.isnan(x).any() or torch.isinf(x).any():
            x_safe = torch.nan_to_num(x, nan=0.0, posinf=self.max_norm, neginf=-self.max_norm)
        else:
            x_safe = x

        current_norm = x_safe.norm(p=2, dim=-1, keepdim=True)

        if self.training:
            with torch.no_grad():
                self.running_norm.copy_(0.99 * self.running_norm + 0.01 * current_norm.mean())

        if self.training and x_safe.requires_grad:

            def _dynamic_clip(grad):
                grad_norm = grad.norm(p=2, dim=-1, keepdim=True)
                dynamic_limit = torch.clamp(self.running_norm * 2.0, min=1.0, max=self.max_norm)
                return torch.where(grad_norm > dynamic_limit, grad * (dynamic_limit / (grad_norm + 1e-8)), grad)

            x_safe.register_hook(_dynamic_clip)

        scale_factor = self.max_norm * torch.tanh(current_norm / self.max_norm) / (current_norm + 1e-8)
        return torch.where(current_norm > self.max_norm * 0.5, x_safe * scale_factor, x_safe)


class GeneralizationGate:
    """Monitors ensemble prediction variance to trigger asynchronous model validation."""

    def __init__(self, runtimecontext):
        self.runtimecontext = runtimecontext
        self.validationthreshold = 2.5
        self.interruptflag = False
        self.lastvariance = 0.0
        self.consecutivebreaches = 0
        self.requiredbreaches = 3
        self.validationbuffer = torch.linspace(-1.0, 1.0, steps=16 * 224, dtype=torch.float32).view(16, 224)

    def _computevariance(self, statedictcpu, validationbuffertensors):
        if validationbuffertensors is None:
            return None

        stackedpreds = []
        with torch.no_grad():
            for headidx in range(3):
                w1 = statedictcpu.get(f"ensemble_heads.{headidx}.0.weight")
                b1 = statedictcpu.get(f"ensemble_heads.{headidx}.0.bias")
                w2 = statedictcpu.get(f"ensemble_heads.{headidx}.2.weight")
                b2 = statedictcpu.get(f"ensemble_heads.{headidx}.2.bias")
                if w1 is None or b1 is None or w2 is None or b2 is None:
                    continue

                hidden = F.mish(F.linear(validationbuffertensors, w1.float(), b1.float()))
                pred = F.linear(hidden, w2.float(), b2.float())
                stackedpreds.append(pred)

        if len(stackedpreds) <= 1:
            return None

        predictions = torch.stack(stackedpreds, dim=0)
        variance = predictions.var(dim=0, unbiased=False).mean().item()
        if not math.isfinite(variance):
            return None
        return float(variance)

    def triggerasyncvalidation(self, predictorstatedict, optimizer):
        if predictorstatedict is None:
            return None
        if not isinstance(predictorstatedict, dict):
            return None

        cpustate = {}
        for k, v in predictorstatedict.items():
            if torch.is_tensor(v):
                cpustate[k] = v.detach().cpu().float().clone()
            else:
                cpustate[k] = v

        variance = self._computevariance(cpustate, self.validationbuffer)
        if variance is None:
            self.interruptflag = False
            self.consecutivebreaches = 0
            return None

        self.lastvariance = variance

        if variance > self.validationthreshold:
            self.consecutivebreaches += 1
        else:
            self.consecutivebreaches = 0

        self.interruptflag = self.consecutivebreaches >= self.requiredbreaches
        return variance

    def checkandresetmomentum(self, optimizer):
        if not self.interruptflag or optimizer is None:
            return False

        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state.get(p, None)
                if state is None:
                    continue
                if "exp_avg" in state and torch.is_tensor(state["exp_avg"]):
                    state["exp_avg"].zero_()
                if "exp_avg_sq" in state and torch.is_tensor(state["exp_avg_sq"]):
                    state["exp_avg_sq"].mul_(0.1)

        self.interruptflag = False
        self.consecutivebreaches = 0
        return True


class RepresentationGate(nn.Module):
    """Latent representation regularization applying Total Correlation and variance preservation constraints."""

    cov_ema: torch.Tensor

    def __init__(
        self,
        target_variance: float = 1.0,
        hinge_margin: float = 0.2,
        decorr_weight: float = 0.05,
        tc_penalty: float = 0.1,
    ):
        super().__init__()
        self.target_variance = target_variance
        self.hinge_margin = hinge_margin
        self.decorr_weight = decorr_weight
        self.tc_penalty = tc_penalty

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.size(0) > 1:
            x_f32 = x.float()

            std = torch.sqrt(x_f32.var(dim=0) + 1e-4)
            variance_loss = F.relu((self.target_variance - self.hinge_margin) - std).mean()

            x_centered = x_f32 - x_f32.mean(dim=0, keepdim=True)
            current_cov = (x_centered.T @ x_centered) / max(1, x_f32.size(0) - 1)

            if not hasattr(self, "cov_ema"):
                self.register_buffer("cov_ema", current_cov.detach().clone())

            self.cov_ema = 0.9 * self.cov_ema + 0.1 * current_cov.detach()
            tc_loss = torch.norm(self.cov_ema - torch.diag(torch.diag(self.cov_ema)), p="fro")

            kde_temperature = 0.5
            bin_centers = torch.linspace(-3.0, 3.0, 64, dtype=x_f32.dtype, device=x_f32.device)

            dist = x_f32.unsqueeze(-1) - bin_centers.unsqueeze(0).unsqueeze(0)

            soft_counts = torch.softmax((-torch.abs(dist) / kde_temperature).float(), dim=-1).to(dist.dtype)

            probs = soft_counts.mean(dim=0)
            probs_f32 = probs.float()
            safe_probs = torch.clamp(probs_f32, min=1e-7)
            entropy = -torch.sum(safe_probs * torch.log(safe_probs), dim=-1).mean()

            max_theoretical_entropy = math.log(bin_centers.size(0))
            entropy_target = max_theoretical_entropy * 0.95

            entropy_penalty = F.relu(entropy_target - entropy).to(probs.dtype)

            self.latent_health_loss = variance_loss + (self.tc_penalty * tc_loss) + (0.01 * entropy_penalty)
        else:
            self.latent_health_loss = torch.tensor(0.0, device=x.device)
        return x


class ChebyshevPolynomialLayer(nn.Module):
    """Expands features using orthogonal Chebyshev polynomials.

    Approximates non-linear continuous functions by expanding inputs into a polynomial basis
    prior to linear projection.
    """

    def __init__(self, in_features: int, out_features: int, degree: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.poly_weight = nn.Parameter(torch.Tensor(out_features, in_features * degree))
        self.layer_norm = nn.LayerNorm(in_features)

        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.poly_weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes Chebyshev polynomial expansion iteratively and applies learned linear weights."""
        x_norm = self.layer_norm(x)
        x_clamped = torch.tanh(x_norm)

        cheb_basis = [torch.ones_like(x_clamped), x_clamped]
        for _ in range(2, self.degree):
            next_basis = 2.0 * x_clamped * cheb_basis[-1] - cheb_basis[-2]
            cheb_basis.append(next_basis)

        poly_features = torch.cat(cheb_basis, dim=-1)

        base_out = F.linear(x_clamped, self.base_weight)
        poly_out = F.linear(poly_features, self.poly_weight)
        return base_out + poly_out


def layer_init(layer, std=math.sqrt(2), bias_const=0.0, use_mup=False):
    """PyTorch initialization. Supports Maximal Update Parametrization (mup) scaling."""
    if use_mup:
        fan_in = layer.weight.size(1)
        mup_std = std / math.sqrt(fan_in)
        torch.nn.init.normal_(layer.weight, mean=0.0, std=mup_std)
    else:
        torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCriticModule(nn.Module):
    """Actor-Critic network supporting generalized state-dependent exploration."""

    value_mean: torch.Tensor
    value_std: torch.Tensor

    def __init__(self, input_dim, num_actions, intent_dim=256):
        super().__init__()
        self.k_chunk = TRAIN_CFG.k_chunk
        self.num_actions = num_actions

        actor_input_dim = input_dim + intent_dim

        self.style_encoder = nn.Sequential(
            layer_init(nn.Linear(actor_input_dim, 128), std=math.sqrt(2), use_mup=False),
            nn.Mish(),
            layer_init(nn.Linear(128, 128 * 2), std=math.sqrt(2), use_mup=False),
        )

        self.cross_attention = nn.MultiheadAttention(embed_dim=128, num_heads=2, batch_first=True)

        self.actor_core = nn.Sequential(
            layer_init(nn.Linear(actor_input_dim, 256)),
            nn.Mish(),
            RMSNorm(256),
            layer_init(nn.Linear(256, 128)),
            nn.Mish(),
        )

        self.chunk_pos_embed = nn.Parameter(torch.randn(1, self.k_chunk, 128) * 0.02)
        self.actor_head_continuous = nn.Sequential(layer_init(nn.Linear(128, 3), std=0.01, use_mup=False), nn.Tanh())
        self.actor_head_discrete = layer_init(nn.Linear(128, num_actions - 3), std=0.01, use_mup=False)

        critic_dim = (input_dim * 3) + intent_dim
        self.num_value_bins = TRAIN_CFG.num_value_bins
        self.symlog_max = TRAIN_CFG.symlog_max

        linear_support = torch.linspace(-self.symlog_max, self.symlog_max, self.num_value_bins)
        symlog_support = torch.sign(linear_support) * torch.log1p(torch.abs(linear_support))
        self.register_buffer("value_support", symlog_support)

        self.critic_1 = nn.Sequential(
            layer_init(nn.Linear(critic_dim, 256)),
            nn.Mish(),
            RMSNorm(256),
            layer_init(nn.Linear(256, 128)),
            nn.Mish(),
            spectral_norm(layer_init(nn.Linear(128, self.num_value_bins))),
        )

        self.critic_2 = nn.Sequential(
            layer_init(nn.Linear(critic_dim, 256)),
            nn.Mish(),
            RMSNorm(256),
            layer_init(nn.Linear(256, 128)),
            nn.Mish(),
            spectral_norm(layer_init(nn.Linear(128, self.num_value_bins))),
        )
        self.cost_critic = nn.Sequential(
            layer_init(nn.Linear(critic_dim, 128)), nn.Mish(), spectral_norm(layer_init(nn.Linear(128, 1)))
        )
        self.intrinsic_critic = nn.Sequential(
            layer_init(nn.Linear(critic_dim, 128)), nn.Mish(), spectral_norm(layer_init(nn.Linear(128, 1)))
        )

        self.register_buffer("value_mean", torch.zeros(1))
        self.register_buffer("value_std", torch.ones(1))

    @jaxtyped(typechecker=beartype)
    def update_value_norm(self, target_value: ValueEstimation) -> None:
        """Updates running moments for Symlog value support scaling."""
        with torch.no_grad():
            target_value_f32 = target_value.float()
            if not torch.isfinite(target_value_f32).all():
                return

            if target_value_f32.size(0) > 1:
                old_mean = self.value_mean.clone()
                old_std = self.value_std.clone()

                self.value_mean = 0.99 * self.value_mean + 0.01 * target_value_f32.mean().to(self.value_mean.dtype)
                batch_variance = target_value_f32.var(unbiased=False)
                new_std = torch.clamp(torch.sqrt(batch_variance + 1e-5), min=0.1).to(self.value_std.dtype)
                self.value_std = 0.99 * self.value_std + 0.01 * new_std

                scale_factor = old_std / self.value_std
                shift_factor = (old_mean - self.value_mean) / self.value_std

                for critic in [self.critic_1, self.critic_2]:
                    final_module = critic[-1]
                    if hasattr(final_module, "weight_orig"):
                        final_module.weight_orig.data.mul_(scale_factor)
                        if getattr(final_module, "bias", None) is not None:
                            final_module.bias.data.mul_(scale_factor).add_(shift_factor)
                    else:
                        final_module.weight.data.mul_(scale_factor)
                        if getattr(final_module, "bias", None) is not None:
                            final_module.bias.data.mul_(scale_factor).add_(shift_factor)

    def _augment_actor_context_with_lookahead(
        self,
        actor_context: torch.Tensor,
        intent_context: torch.Tensor,
        dynamics_model: nn.Module,
        strategy: str = "uniform",
    ) -> torch.Tensor:
        """Augments the actor context with model-based lookahead representations."""
        batch_size = actor_context.size(0)
        device = actor_context.device

        with torch.no_grad():
            if strategy == "uniform":
                action_probs = torch.ones(batch_size, self.num_actions, device=device) / self.num_actions
            elif strategy == "greedy" or strategy == "sample":
                base_features = self.actor_core(torch.cat([actor_context, intent_context], dim=-1))
                if hasattr(self, "actor_head_discrete"):
                    discrete_logits = self.actor_head_discrete(base_features)
                    immediate_logits = discrete_logits[:, 0, :] if discrete_logits.dim() == 3 else discrete_logits
                else:
                    base_logits = self.actor_head(base_features)
                    immediate_logits = (
                        base_logits[:, 0, : self.num_actions]
                        if base_logits.dim() == 3
                        else base_logits[..., : self.num_actions]
                    )

                if strategy == "greedy":
                    action_probs = F.one_hot(torch.argmax(immediate_logits, dim=-1), self.num_actions).float()
                else:
                    action_probs = F.softmax(immediate_logits.float(), dim=-1).to(actor_context.dtype)
            else:
                action_probs = torch.zeros(batch_size, self.num_actions, device=device)

            imagined_next = dynamics_model(actor_context, action_probs)

            delta = F.layer_norm(imagined_next - actor_context, normalized_shape=[actor_context.size(-1)])
            gate = torch.sigmoid((actor_context * delta).sum(dim=-1, keepdim=True) * 0.1)

        return actor_context + (torch.sigmoid(gate) * 0.15 * delta).to(actor_context.dtype)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        actor_context: LatentState,
        critic_context: Optional[Float[torch.Tensor, "..."]] = None,
        intent_context: Optional[LatentState] = None,
        dynamics_model: Optional[nn.Module] = None,
        planning_budget: Optional[PlanningBudget] = None,
    ) -> ActorCriticOutput:
        """Computes policy estimation forward pass."""
        if intent_context is None:
            intent_shape = list(actor_context.shape)
            intent_shape[-1] = 256
            intent_context = torch.zeros(*intent_shape, device=actor_context.device)

        allow_lookahead = True if planning_budget is None else planning_budget.allow_actor_lookahead

        if dynamics_model is not None and allow_lookahead:
            actor_context = self._augment_actor_context_with_lookahead(
                actor_context, intent_context, dynamics_model, strategy=CFG.LOOKAHEAD_STRATEGY
            )

        full_actor_context = torch.cat([actor_context, intent_context], dim=-1)

        style_params = self.style_encoder(full_actor_context)
        mu, logvar = style_params.chunk(2, dim=-1)

        logvar = 5.0 - F.softplus(5.0 - logvar)
        logvar = -20.0 + F.softplus(logvar + 20.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.shape, device=std.device, dtype=torch.float32).to(std.dtype)

        ambient_z = mu + eps * std
        z_style = LorentzGeometry.project(ambient_z)
        core_features = self.actor_core(full_actor_context)

        if CFG.ENABLE_ACTION_CHUNKING:
            z_style_seq = z_style.unsqueeze(1).expand(-1, self.k_chunk, -1)
            core_seq = core_features.unsqueeze(1).expand(-1, self.k_chunk, -1)

            core_seq = core_seq + self.chunk_pos_embed

            original_shape = core_seq.shape
            if core_seq.dim() == 4:
                b, t, s, d = original_shape
                core_seq_3d = core_seq.view(b * t, s, d)
                z_style_seq_3d = z_style_seq.view(b * t, s, d)
                attn_out_3d, _ = self.cross_attention(query=core_seq_3d, key=z_style_seq_3d, value=z_style_seq_3d)
                attn_out = attn_out_3d.view(b, t, s, d)
            else:
                attn_out, _ = self.cross_attention(query=core_seq, key=z_style_seq, value=z_style_seq)

            fused_features = core_seq + attn_out

            continuous_actions = self.actor_head_continuous(fused_features)
            discrete_logits = self.actor_head_discrete(fused_features)
            policy_logits = torch.cat([continuous_actions, discrete_logits], dim=-1)
        else:
            fused_features = core_features + z_style

            dummy_input = (
                fused_features.unsqueeze(1)
                if fused_features.dim() == 2
                else fused_features.view(-1, 1, fused_features.shape[-1])
            )

            dummy_attn, _ = self.cross_attention(query=dummy_input, key=dummy_input, value=dummy_input)

            fused_features = fused_features + (self.chunk_pos_embed.sum() * 0.0) + (dummy_attn.sum() * 0.0)

            continuous_actions = self.actor_head_continuous(fused_features).unsqueeze(1)
            discrete_logits = self.actor_head_discrete(fused_features).unsqueeze(1)
            policy_logits = torch.cat([continuous_actions, discrete_logits], dim=-1)

        if critic_context is None:
            batch_size = actor_context.size(0)
            device = actor_context.device
            dummy_value = torch.zeros(batch_size, 1, device=device)
            dummy_latent = torch.zeros(batch_size, 128, device=device)

            needed_dim = next(self.critic_1.parameters()).shape[1]
            dummy_shape = list(actor_context.shape)
            dummy_shape[-1] = needed_dim
            full_critic_context = torch.zeros(*dummy_shape, device=device, dtype=actor_context.dtype)

            v1 = self.critic_1(full_critic_context)
            v2 = self.critic_2(full_critic_context)
            cv = self.cost_critic(full_critic_context)
            iv = self.intrinsic_critic(full_critic_context)

            dummy_value = dummy_value + (v1.sum() * 0.0) + (v2.sum() * 0.0) + (cv.sum() * 0.0) + (iv.sum() * 0.0)

            return ActorCriticOutput(
                policy_logits=policy_logits,
                pessimistic_value=dummy_value,
                cost_value=dummy_value,
                intrinsic_value=dummy_value,
                value_logits_1=dummy_value,
                value_logits_2=dummy_value,
                style_mu=dummy_latent,
                style_logvar=dummy_latent,
            )

        # Centralized Training with Decentralized Execution (CTDE) context
        full_critic_context = torch.cat([critic_context, intent_context], dim=-1)

        value_logits_1 = self.critic_1(full_critic_context)
        value_logits_2 = self.critic_2(full_critic_context)

        value_probs_1 = F.softmax(value_logits_1.float(), dim=-1).to(value_logits_1.dtype)
        value_probs_2 = F.softmax(value_logits_2.float(), dim=-1).to(value_logits_2.dtype)

        raw_value_1 = torch.sum(value_probs_1 * self.value_support, dim=-1, keepdim=True)
        raw_value_2 = torch.sum(value_probs_2 * self.value_support, dim=-1, keepdim=True)

        cost_value = self.cost_critic(full_critic_context)
        intrinsic_value = self.intrinsic_critic(full_critic_context)

        efe_1 = raw_value_1 + intrinsic_value - cost_value
        efe_2 = raw_value_2 + intrinsic_value - cost_value
        pessimistic_efe_value = torch.min(efe_1, efe_2)

        pessimistic_norm_value = (pessimistic_efe_value * (self.value_std + 1e-5)) + self.value_mean

        return ActorCriticOutput(
            policy_logits=policy_logits,
            pessimistic_value=pessimistic_norm_value,
            cost_value=cost_value,
            intrinsic_value=intrinsic_value,
            value_logits_1=value_logits_1,
            value_logits_2=value_logits_2,
            style_mu=mu,
            style_logvar=logvar,
        )


class TestTimeAdaptationModule(nn.Module):
    """Self-supervised Test-Time Adaptation module mapping forward dynamics consistency via auxiliary adapters."""

    def __init__(self, dynamics_model, target_encoder_ref, consistency_weight=5.0):
        super().__init__()
        self.dynamics_model = dynamics_model
        self.target_encoder = target_encoder_ref
        self.consistency_weight = consistency_weight

        self.adaptation_adapter = nn.Sequential(nn.Linear(256, 128), nn.Mish(), nn.Linear(128, 256))

        self.local_optimizer = torch.optim.AdamW(self.adaptation_adapter.parameters(), lr=1e-3)

    def adapt_step(self, states, actions, next_states, global_optimizer=None):

        if not hasattr(self, "anomaly_reservoir"):
            self.anomaly_reservoir = []

        self.anomaly_reservoir.append((states.detach().cpu(), actions.detach().cpu(), next_states.detach().cpu()))

        if len(self.anomaly_reservoir) >= 16:

            def _background_adaptation(batch_data):
                import torch
                from torch.nn import functional as F

                device = next(self.adaptation_adapter.parameters()).device

                states, actions, next_states = zip(*batch_data)
                batch_states = torch.cat(states, dim=0).to(device, non_blocking=True)
                batch_actions = torch.cat(actions, dim=0).to(device, non_blocking=True)
                batch_next_states = torch.cat(next_states, dim=0).to(device, non_blocking=True)

                self.local_optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    target_latent = self.target_encoder(batch_next_states)
                    source_latent = self.target_encoder(batch_states)

                    if hasattr(self.dynamics_model, "predict_next_latent"):
                        pred_latent = self.dynamics_model.predict_next_latent(source_latent, batch_actions)
                    else:
                        pred_latent = self.dynamics_model(source_latent, batch_actions)

                adapted_latent = self.adaptation_adapter(pred_latent)

                loss = F.mse_loss(adapted_latent, target_latent)
                weighted_loss = loss * self.consistency_weight

                weighted_loss.backward()

                valid_grads = True
                for p in self.adaptation_adapter.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        valid_grads = False
                        break

                if valid_grads:
                    torch.nn.utils.clip_grad_norm_(self.adaptation_adapter.parameters(), max_norm=1.0)
                    self.local_optimizer.step()
                else:
                    self.local_optimizer.zero_grad()

                return weighted_loss.item()

            adaptation_loss = _background_adaptation(list(self.anomaly_reservoir))
            self.anomaly_reservoir.clear()
            return adaptation_loss


class SelectiveStateSpaceModel(nn.Module):
    """Implements a Selective State Space Model (SSM) architecture.

    Models continuous-time dynamics discretized via parallel prefix scans.
    """

    def __init__(self, dim: int, state_dim: int = 16):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim

        self.proj_in = nn.Linear(dim, dim * 2)
        self.conv1d = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=4, groups=dim, padding=3)
        self.proj_x = nn.Linear(dim, state_dim + state_dim + dim)

        self.dt_proj = nn.Linear(dim, dim)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32).repeat(dim, 1)))
        self.D = nn.Parameter(torch.ones(dim))

        self.proj_out = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len, d = x.shape
        xz = self.proj_in(x)
        x_proj, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_proj.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        x_act = F.silu(x_conv)

        x_params = self.proj_x(x_act)
        delta, B, C = torch.split(x_params, [self.dim, self.state_dim, self.state_dim], dim=-1)

        delta = F.softplus(self.dt_proj(delta) * torch.tanh(x_act.mean(dim=-1, keepdim=True)))
        A = -torch.exp(self.A_log.float())

        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB_x = delta.unsqueeze(-1) * B.unsqueeze(2) * x_act.unsqueeze(-1)

        h = torch.zeros(b, d, self.state_dim, device=x.device, dtype=x.dtype)
        ys = []

        for i in range(seq_len):
            h = deltaA[:, i] * h + deltaB_x[:, i]
            y = torch.einsum("bdn,bn->bd", h, C[:, i])
            ys.append(y)

        y = torch.stack(ys, dim=1)
        y = y + (x_act * self.D)
        y = y * F.silu(z)

        out = self.proj_out(y)
        return self.layer_norm(x + out)


class MultimodalSensoryHub(nn.Module):
    """Multimodal sensory projection routing to a shared temporal backbone."""

    def __init__(self, latent_dim=256, sample_rate=44100):
        super().__init__()
        self.latent_dim = latent_dim
        self.unified_backbone = UnifiedGatedLinearBackbone(embed_dim=128, output_dim=latent_dim)
        self.register_buffer("last_text_loss", torch.zeros(1))

    def process_audio(self, waveform):
        if waveform.dim() == 3:
            x = waveform.transpose(1, 2).float()
        else:
            x = waveform.float()
        return self.unified_backbone(x)

    def process_text(self, byte_tensor, goal_context=None):
        out = self.unified_backbone(byte_tensor.float())
        if hasattr(self.unified_backbone, "pcn_layer") and hasattr(
            self.unified_backbone.pcn_layer, "last_convergence_steps"
        ):
            self.last_text_loss = self.unified_backbone.pcn_layer.last_convergence_steps.float().mean()
        return out

    def project_ood_semantics(self, text):
        return text


class TemporalCreditAssigner(nn.Module):
    """Computes Generalized Advantage Estimation (GAE) modulated by uncertainty gating."""

    def __init__(self, gamma: float = 0.995, lambda_gae: float = 0.95):
        super().__init__()
        self.gamma = gamma
        self.lambda_gae = lambda_gae

    def forward(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        done_flags: torch.Tensor,
        critic_divergence: torch.Tensor,
        truth_margin: torch.Tensor,
        controllability_score: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes GAE adjusted by epistemic reliability metrics."""
        deltas = rewards + self.gamma * next_values * (1.0 - done_flags) - values
        advantages = torch.zeros_like(rewards)
        last_gae_lam = torch.tensor(0.0, dtype=rewards.dtype, device=rewards.device)

        reliability_gate = torch.sigmoid(3.0 - critic_divergence)
        truth_gate = torch.clamp(torch.sigmoid(truth_margin * 10.0), min=0.1)

        if controllability_score is None:
            controllability_score = torch.ones_like(rewards)

        conservative_multiplier = reliability_gate * truth_gate * torch.clamp(controllability_score, min=0.01, max=1.0)

        for t in reversed(range(rewards.size(0))):
            effective_lambda = self.lambda_gae * conservative_multiplier[t]
            advantages[t] = deltas[t] + self.gamma * effective_lambda * (1.0 - done_flags[t]) * last_gae_lam
            last_gae_lam = advantages[t]

        return advantages


class InterventionalCausalEngine(nn.Module):
    """Evaluates action causality via antithetic counterfactual perturbations."""

    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim
        self.perturbation_scale = 0.1

    def evaluate_controllability(
        self, base_state: torch.Tensor, action: torch.Tensor, dynamics_model: nn.Module
    ) -> torch.Tensor:
        """Evaluates controllability by computing exact Antithetic Sampling differentials."""
        with torch.no_grad():
            base_state.size(0)

            pred_next_state = dynamics_model(base_state, action)
            base_state_f32 = base_state.float()
            safe_variance = torch.var(base_state_f32, dim=-1, keepdim=True, unbiased=False) + 1e-8
            epistemic_uncertainty = torch.clamp(torch.sqrt(safe_variance), min=0.01, max=1.0).to(base_state.dtype)
            dynamic_perturbation = self.perturbation_scale * epistemic_uncertainty

            uniform_noise = torch.rand_like(action, dtype=torch.float32).clamp(min=1e-5, max=1.0 - 1e-5)
            gumbel_noise = (-torch.log(-torch.log(uniform_noise)) * dynamic_perturbation.float()).to(action.dtype)

            action_plus = F.softmax((action + gumbel_noise) / 0.5, dim=-1)
            action_minus = F.softmax((action - gumbel_noise) / 0.5, dim=-1)

            future_plus = dynamics_model(base_state, action_plus)
            future_minus = dynamics_model(base_state, action_minus)

            div_plus = F.smooth_l1_loss(pred_next_state.float(), future_plus.float(), reduction="none").mean(dim=-1)
            div_minus = F.smooth_l1_loss(pred_next_state.float(), future_minus.float(), reduction="none").mean(dim=-1)

            mean_divergence = ((div_plus + div_minus) * 0.5).to(base_state.dtype)

            var_divergence = torch.clamp(
                ((div_plus - mean_divergence) ** 2 + (div_minus - mean_divergence) ** 2) * 0.5, min=1e-8, max=1e4
            )

            controllability_score = torch.clamp(mean_divergence, max=1.0) - (
                epistemic_uncertainty.squeeze(-1) * var_divergence
            )

        return torch.clamp(controllability_score, min=0.0, max=1.0)
