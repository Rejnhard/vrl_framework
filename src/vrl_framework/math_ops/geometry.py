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

import logging
import math
import sys
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast to fp32 during variance accumulation to prevent overflow in half-precision
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_f32 = x.float()
            x_f32 = x_f32.clamp(-1e4, 1e4)
            variance = x_f32.pow(2).mean(-1, keepdim=True)
            output = x_f32 * torch.rsqrt(variance + self.eps)
            return torch.nan_to_num((output * self.weight.float()).to(x.dtype), nan=0.0, posinf=1e4, neginf=-1e4)


def compute_moe_load_balancing_loss(
    routing_weights: torch.Tensor, num_experts: int, raw_routing_logits: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Computes auxiliary load balancing loss for sparsely gated MoE layers.
    Incorporates sharpness penalty to discourage uniform routing distribution.
    """
    expert_usage = routing_weights.mean(dim=0)
    loss = num_experts * torch.sum(expert_usage * expert_usage)

    sharpness_penalty = torch.mean(torch.sum(routing_weights * torch.log(routing_weights + 1e-8), dim=-1))
    loss = loss + (0.05 * sharpness_penalty)

    if raw_routing_logits is not None:
        z_loss = torch.logsumexp(raw_routing_logits, dim=-1).pow(2).mean() * 1e-3
        loss += z_loss
    return loss


def compute_manifold_curvature_lambda(step: int, warmup_steps: int = 10000) -> float:
    if step < warmup_steps:
        return 0.001 + (step / warmup_steps) * 0.999
    return 1.0


class LorentzGeometry:
    @staticmethod
    def minkowski_dot(x: torch.Tensor, y: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        upcast = x.dtype in [torch.float16, torch.bfloat16]
        x_calc = x.float() if upcast else x
        y_calc = y.float() if upcast else y
        xy = x_calc * y_calc
        spatial_dot = xy[..., 1:].sum(dim=-1, keepdim=keepdim)
        temporal_dot = xy[..., 0:1] if keepdim else xy[..., 0]
        return spatial_dot - temporal_dot

    @staticmethod
    def project(x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            upcast = x.dtype in [torch.float16, torch.bfloat16]
            x_calc = x.float() if upcast else x
            spatial = x_calc[..., 1:]

            spatial_sq_norm = spatial.pow(2).sum(dim=-1, keepdim=True)
            temporal = torch.sqrt(spatial_sq_norm + 1.0)

            max_fp16 = 65500.0
            if upcast:
                overflow_mask = temporal > max_fp16
                if overflow_mask.any():
                    scale = max_fp16 / temporal
                    spatial = torch.where(overflow_mask, spatial * scale, spatial)
                    temporal = torch.where(overflow_mask, max_fp16 * torch.ones_like(temporal), temporal)

            projected = torch.cat([temporal, spatial], dim=-1)
        return projected.to(x.dtype)

    @staticmethod
    def distance(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        m_dot = -LorentzGeometry.minkowski_dot(x, y)
        m_dot_safe = torch.clamp(m_dot, min=1.0 + eps)
        sq_val = torch.clamp(m_dot_safe * m_dot_safe - 1.0, min=eps)
        dist = torch.log(m_dot_safe + torch.sqrt(sq_val))
        return dist.to(x.dtype)

    @staticmethod
    def exp_map(x: torch.Tensor, v: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        upcast = x.dtype in [torch.float16, torch.bfloat16]
        x_calc = x.float() if upcast else x
        v_calc = v.float() if upcast else v

        v_norm_sq_raw = LorentzGeometry.minkowski_dot(v_calc, v_calc)
        v_norm_raw_expanded = torch.sqrt(torch.clamp(v_norm_sq_raw, min=eps)).unsqueeze(-1)
        v_calc = torch.where(v_norm_raw_expanded > 11.0, v_calc * (11.0 / v_norm_raw_expanded), v_calc)

        v_norm_sq = LorentzGeometry.minkowski_dot(v_calc, v_calc)
        v_norm_sq_safe = torch.clamp(v_norm_sq, min=0.0)
        v_norm = torch.sqrt(torch.clamp(v_norm_sq_safe, min=eps))

        v_norm_expanded = v_norm.unsqueeze(-1)
        small_norm_mask = v_norm_expanded < 1e-4

        v_norm_denominator = v_norm_expanded.clamp(min=1e-4)

        sinh_term_standard = (v_calc / (v_norm_denominator + eps)) * torch.sinh(v_norm_expanded)
        cosh_term_standard = torch.cosh(v_norm_expanded)

        sinh_term_taylor = v_calc + (v_calc * v_norm_sq_safe.unsqueeze(-1) / 6.0)
        cosh_term_taylor = 1.0 + (v_norm_sq_safe.unsqueeze(-1) / 2.0)

        sinh_term = torch.where(small_norm_mask, sinh_term_taylor, sinh_term_standard)
        cosh_term = torch.where(small_norm_mask, cosh_term_taylor, cosh_term_standard)

        mapped = x_calc * cosh_term + sinh_term
        mapped = torch.clamp(mapped, min=-1e4, max=1e4)
        return LorentzGeometry.project(mapped).to(x.dtype)


@torch.amp.autocast(device_type="cuda", enabled=False)
def sigreg_weak_loss(
    x: torch.Tensor, sketch_dim: int = 64, cov_ema_tensor: Optional[torch.Tensor] = None
) -> torch.Tensor:
    upcast = x.dtype in [torch.float16, torch.bfloat16]
    x_calc = x.float() if upcast else x
    N, C = x_calc.size()

    x_centered = x_calc - x_calc.mean(dim=0, keepdim=True)

    with torch.no_grad():
        generator = torch.Generator(device=x_calc.device)
        generator.manual_seed(42)
        base_matrix = torch.randn(C, sketch_dim, generator=generator, device=x_calc.device, dtype=x_calc.dtype)
        proj_matrix, _ = torch.linalg.qr(base_matrix)
        proj_matrix = proj_matrix / math.sqrt(C)

    x_projected = torch.matmul(x_centered, proj_matrix)

    x_norm = F.normalize(x_projected, p=2, dim=0, eps=1e-6) * math.sqrt(N)

    current_cov = (x_norm.T @ x_norm) / max(1, N - 1)
    if cov_ema_tensor is not None:
        cov_ema_tensor.lerp_(current_cov, 0.1)
        cov = cov_ema_tensor
    else:
        cov = current_cov

    cov_sq_norm = torch.sum(cov * cov)
    cov_trace = torch.trace(cov)
    cov_loss = torch.sqrt(torch.clamp(cov_sq_norm - 2.0 * cov_trace + sketch_dim, min=1e-6))

    std = torch.sqrt(x_centered.var(dim=0, unbiased=False) + 1e-4)

    target_std = 1.0 / math.sqrt(C) if C > 0 else 1.0
    var_loss = torch.mean(F.relu(target_std - std)) + torch.mean(F.relu(std - target_std)) * 0.5

    return (cov_loss + var_loss).to(x.dtype)


def run_math_diagnostics():
    logger = logging.getLogger("Diagnostics")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        try:
            import os

            from vrl_framework.core.settings import LOGS_DIR

            file_path = os.path.join(LOGS_DIR, "geometry_diagnostics.log")
            file_handler = logging.FileHandler(file_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except ImportError as e:
            logger.warning(f"Failed to load LOGS_DIR from core settings: {e}")

    logger.info("Initializing analytical gradient checks...")

    x_fp64 = torch.randn(4, 256, dtype=torch.float64, requires_grad=True)
    y_fp64 = torch.randn(4, 256, dtype=torch.float64, requires_grad=True)

    try:
        torch.autograd.gradcheck(LorentzGeometry.distance, (x_fp64, y_fp64), eps=1e-6, atol=1e-4)
        logger.info("LorentzGeometry.distance: Gradcheck PASSED")
    except Exception as e:
        logger.error(f"LorentzGeometry.distance: Gradcheck FAILED - {e}")

    try:
        torch.autograd.gradcheck(sigreg_weak_loss, (x_fp64,), eps=1e-6, atol=1e-4)
        logger.info("sigreg_weak_loss: Gradcheck PASSED")
    except Exception as e:
        logger.error(f"sigreg_weak_loss: Gradcheck FAILED - {e}")

    logger.info("Initializing numerical stability stress tests...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_fp16 = torch.randn(1024, 256, dtype=torch.float16, device=device)
    y_fp16 = torch.randn(1024, 256, dtype=torch.float16, device=device)

    dist_fp16 = LorentzGeometry.distance(x_fp16, y_fp16)
    if torch.isnan(dist_fp16).any() or torch.isinf(dist_fp16).any():
        logger.error("LorentzGeometry.distance: FP16 Stress Test FAILED")
    else:
        logger.info("LorentzGeometry.distance: FP16 Stress Test PASSED")

    logger.info("Executing End-to-End Hyperbolic Safety Contract validation...")
    try:
        jepa_out = LorentzGeometry.project(x_fp16)
        quantized = LorentzGeometry.project(jepa_out + torch.randn_like(jepa_out) * 0.1)
        planner_drift = LorentzGeometry.exp_map(quantized, torch.randn_like(quantized) * 0.05)
        critic_dist = LorentzGeometry.distance(jepa_out, planner_drift)

        if torch.isnan(critic_dist).any() or torch.isinf(critic_dist).any():
            raise ValueError("E2E Validation Failed: Geodesic bounds corrupted along the pipeline.")
        logger.info("E2E Hyperbolic Contract: PASSED")
    except Exception as e:
        logger.error(f"E2E Validation FAILED - {e}")


def _gaussian_entropy(cov_matrix: torch.Tensor, dim: int) -> torch.Tensor:
    symmetric_cov = (cov_matrix + cov_matrix.T) * 0.5
    trace_val = torch.clamp(symmetric_cov.trace(), min=1e-6)
    jitter = torch.eye(dim, device=symmetric_cov.device) * (1e-5 * (trace_val / dim))
    stable_cov = (symmetric_cov + jitter).float()
    sign, logdet = torch.slogdet(stable_cov)
    safe_logdet = torch.where(torch.isinf(logdet), torch.tensor(-100.0, device=logdet.device), logdet)
    return 0.5 * (dim * math.log(2 * math.pi * math.e) + safe_logdet)


@torch.no_grad()
def compute_traj_entropy(state_sequence: torch.Tensor) -> torch.Tensor:
    if state_sequence.dim() == 1:
        state_sequence = state_sequence.unsqueeze(0)

    if state_sequence.size(0) < 2 and state_sequence.dim() == 2:
        state_sequence = torch.cat([state_sequence, state_sequence + torch.randn_like(state_sequence) * 1e-4], dim=0)

    temporal_sequence = state_sequence.clone()
    if temporal_sequence.dim() == 2:
        temporal_sequence = temporal_sequence.unsqueeze(0)
    if state_sequence.dim() > 2:
        state_sequence = rearrange(state_sequence, "b t d -> (b t) d")

    total_dim = state_sequence.size(-1)
    mid = total_dim // 2

    sys_A = state_sequence[:, :mid]
    sys_B = state_sequence[:, mid:]

    eps = 1e-5
    cov_A = torch.cov(sys_A.T)
    cov_A.diagonal().add_(eps)

    cov_B = torch.cov(sys_B.T)
    cov_B.diagonal().add_(eps)

    cov_AB = torch.cov(state_sequence.T)
    cov_AB.diagonal().add_(eps)

    total_dim = state_sequence.size(-1)
    symmetric_cov_ab = (cov_AB + cov_AB.T) * 0.5
    H_AB = _gaussian_entropy(symmetric_cov_ab, total_dim)

    K_partitions = 20
    split_point = total_dim // 2

    rand_vals = torch.rand(K_partitions, total_dim, device=state_sequence.device)
    perms = torch.argsort(rand_vals, dim=1)

    idx_A = perms[:, :split_point]
    idx_B = perms[:, split_point:]

    sys_A_mc = torch.gather(
        state_sequence.unsqueeze(0).expand(K_partitions, -1, -1),
        2,
        idx_A.unsqueeze(1).expand(-1, state_sequence.size(0), -1),
    )
    sys_B_mc = torch.gather(
        state_sequence.unsqueeze(0).expand(K_partitions, -1, -1),
        2,
        idx_B.unsqueeze(1).expand(-1, state_sequence.size(0), -1),
    )

    sys_A_mc_centered = sys_A_mc - sys_A_mc.mean(dim=1, keepdim=True)
    sys_B_mc_centered = sys_B_mc - sys_B_mc.mean(dim=1, keepdim=True)

    b_cov_A = torch.bmm(sys_A_mc_centered.transpose(1, 2), sys_A_mc_centered) / (state_sequence.size(0) - 1)
    b_cov_A.diagonal(dim1=-2, dim2=-1).add_(eps)

    b_cov_B = torch.bmm(sys_B_mc_centered.transpose(1, 2), sys_B_mc_centered) / (state_sequence.size(0) - 1)
    b_cov_B.diagonal(dim1=-2, dim2=-1).add_(eps)

    sym_cov_A = (b_cov_A + b_cov_A.transpose(1, 2)) * 0.5
    sym_cov_B = (b_cov_B + b_cov_B.transpose(1, 2)) * 0.5

    _, logdet_A = torch.slogdet(sym_cov_A.float())
    _, logdet_B = torch.slogdet(sym_cov_B.float())

    H_A_mc = 0.5 * (split_point * math.log(2 * math.pi * math.e) + logdet_A)
    H_B_mc = 0.5 * ((total_dim - split_point) * math.log(2 * math.pi * math.e) + logdet_B)

    current_h_traj = torch.clamp(H_A_mc + H_B_mc - H_AB, min=0.0)
    min_h_traj = torch.min(current_h_traj)

    base_h_traj = min_h_traj / state_sequence.shape[0]

    with torch.no_grad():
        epsilon = 1e-5
        quantized_sequence = torch.round(torch.clamp(temporal_sequence.float(), min=-10.0, max=10.0) * 100.0) / 100.0
        power_spectrum = torch.abs(torch.fft.rfft(quantized_sequence, dim=1)) ** 2
        power_spectrum = torch.where(
            torch.isnan(power_spectrum) | torch.isinf(power_spectrum),
            torch.tensor(epsilon, device=quantized_sequence.device),
            power_spectrum,
        )
        power_spectrum_f32 = power_spectrum.float()
        geometric_mean = torch.exp(torch.mean(torch.log(power_spectrum_f32 + epsilon), dim=-1))
        arithmetic_mean = torch.mean(power_spectrum_f32, dim=-1) + epsilon
        ratio = (geometric_mean / arithmetic_mean).mean()

    complexity_penalty = 1.0 - torch.abs(ratio - 0.5) * 2.0
    complexity_multiplier = torch.clamp(complexity_penalty, min=0.1)

    return base_h_traj * complexity_multiplier


@torch.no_grad()
def compute_eff_dim(
    latent_batch: torch.Tensor,
    threshold: float = 1e-3,
    optimizer: Optional[torch.optim.Optimizer] = None,
    layer_to_reset: Optional[nn.Module] = None,
) -> torch.Tensor:
    if latent_batch.size(0) < 2:
        return torch.tensor(1.0, device=latent_batch.device)

    latent_centered = latent_batch - latent_batch.mean(dim=0, keepdim=True)
    cov_matrix = (latent_centered.T @ latent_centered) / (latent_batch.size(0) - 1)

    try:
        stable_matrix = cov_matrix.float()
        stable_matrix.add_(torch.rand_like(stable_matrix), alpha=1e-8)
        stable_matrix.diagonal().add_(1e-6)

        stable_matrix = (stable_matrix + stable_matrix.T) * 0.5

        if torch.isnan(stable_matrix).any() or torch.isinf(stable_matrix).any():
            import logging

            logging.error("NaN or Inf detected in covariance matrix during effective dimensionality computation.")
            if optimizer is not None:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= 0.5
            if layer_to_reset is not None and hasattr(layer_to_reset, "weight"):
                torch.nn.init.orthogonal_(layer_to_reset.weight)
                if hasattr(layer_to_reset, "bias") and layer_to_reset.bias is not None:
                    torch.nn.init.zeros_(layer_to_reset.bias)
            return torch.tensor(1.0, device=latent_batch.device)

        eigenvalues = torch.linalg.eigvalsh(stable_matrix)

        eigenvalues = torch.clamp(eigenvalues, min=1e-9)
        assert eigenvalues.dim() == 1, f"Expected 1D tensor array, encountered dimension size: {eigenvalues.dim()}"

        eig_sum = torch.clamp(eigenvalues.sum(), min=1e-9)
        norm_eigenvalues = (eigenvalues / eig_sum).float()

        safe_eigenvalues = torch.clamp(norm_eigenvalues, min=1e-7)
        entropy = -torch.sum(safe_eigenvalues * torch.log(safe_eigenvalues))

        return torch.exp(entropy)

    except torch.linalg.LinAlgError as linalg_err:
        import logging

        logging.error(f"SVD decomposition collapsed computing effective dimensionality: {linalg_err}")
        return torch.tensor(1.0, device=latent_batch.device)
    except Exception as e:
        import logging
        import traceback

        logging.error(f"Unexpected fault during dimensionality reduction phase: {e}")
        logging.error(traceback.format_exc())
        return torch.tensor(1.0, device=latent_batch.device)


@torch.no_grad()
def compute_predictive_surprisal(grid_tensor: torch.Tensor, dynamics_model=None) -> torch.Tensor:
    if dynamics_model is None:
        return torch.tensor(0.0, device=grid_tensor.device)

    if isinstance(dynamics_model, torch.Tensor):
        if grid_tensor.size() != dynamics_model.size():
            return torch.tensor(0.0, device=grid_tensor.device)
        pred_projected = LorentzGeometry.project(grid_tensor)
        target_projected = LorentzGeometry.project(dynamics_model)
        surprisal_error = LorentzGeometry.distance(pred_projected, target_projected)
        return torch.clamp(surprisal_error, min=0.01, max=100.0).mean()

    if grid_tensor.size(0) < 2:
        return torch.tensor(0.0, device=grid_tensor.device)

    if grid_tensor.dim() == 3:
        state_t = grid_tensor[:, :-1].reshape(-1, grid_tensor.size(-1))
        state_t1 = grid_tensor[:, 1:].reshape(-1, grid_tensor.size(-1))
    else:
        state_t = grid_tensor[:-1]
        state_t1 = grid_tensor[1:]

    if hasattr(dynamics_model, "action_dim"):
        dummy_action = torch.zeros(state_t.size(0), dynamics_model.action_dim, device=grid_tensor.device)
        dummy_action[:, 0] = 1.0
        predicted_t1 = dynamics_model(state_t, dummy_action)
    else:
        predicted_t1 = dynamics_model(state_t)

    pred_projected = LorentzGeometry.project(predicted_t1)
    target_projected = LorentzGeometry.project(state_t1)

    surprisal_error = LorentzGeometry.distance(pred_projected, target_projected)
    surprisal_error = torch.clamp(surprisal_error, min=0.01, max=100.0)

    return surprisal_error.mean()


@torch.no_grad()
def compute_expert_orthogonality(w1_weights: torch.Tensor) -> torch.Tensor:
    num_experts = w1_weights.size(0)
    if num_experts < 2:
        return torch.tensor(0.0, device=w1_weights.device)

    w1_flat = w1_weights.view(num_experts, -1)
    w_norm = F.normalize(w1_flat, p=2, dim=-1)
    sim_matrix = torch.matmul(w_norm, w_norm.T)

    off_diag_sum = sim_matrix.sum() - sim_matrix.trace()
    mean_sim = off_diag_sum / (num_experts * (num_experts - 1))
    return F.relu(mean_sim)


def compute_subspace_orthogonality(latent_tensor: torch.Tensor, dim_split: int = 3) -> torch.Tensor:
    chunk_size = latent_tensor.size(-1) // dim_split
    if chunk_size == 0 or dim_split < 2:
        return torch.tensor(0.0, device=latent_tensor.device)

    truncated_tensor = latent_tensor[..., : dim_split * chunk_size]
    reshaped_chunks = truncated_tensor.view(*latent_tensor.shape[:-1], dim_split, chunk_size)

    reshaped_chunks = reshaped_chunks + torch.randn_like(reshaped_chunks) * 1e-6
    normalized_chunks = F.normalize(reshaped_chunks, p=2, dim=-1, eps=1e-6)

    similarity_matrix = torch.matmul(normalized_chunks, normalized_chunks.transpose(-1, -2))

    triu_sims = similarity_matrix.triu(diagonal=1)
    soft_margin = 0.15

    batch_dims_product = similarity_matrix.numel() // (dim_split * dim_split)
    num_elements = (dim_split * (dim_split - 1)) // 2
    total_elements = batch_dims_product * num_elements

    loss = torch.sum(F.relu(torch.abs(triu_sims) - soft_margin)) / max(1, total_elements)
    return loss


def compute_gradient_variance(model_parameters: List[nn.Parameter]) -> torch.Tensor:
    variances = [
        torch.var(torch.norm(p.grad.detach(), dim=-1)) for p in model_parameters if p.grad is not None and p.dim() > 1
    ]
    if not variances:
        return torch.tensor(0.0, device="cuda" if torch.cuda.is_available() else "cpu")

    return torch.stack(variances).mean()


def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    rem_seconds = seconds % 60

    if hours > 0:
        return f"{hours:02d}h {minutes:02d}m {rem_seconds:05.2f}s"
    if minutes > 0:
        return f"{minutes:02d}m {rem_seconds:05.2f}s"
    return f"{rem_seconds:05.2f}s"


def get_input(prompt: str) -> str:
    try:
        sys.stdout.write(prompt)
        sys.stdout.flush()
        user_input = sys.stdin.readline()
        if not user_input:
            return ""
        return user_input.strip()
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        sys.stdout.flush()
        return ""
    except Exception as e:
        logging.getLogger("IO_Stream").error(f"Input Stream Fault: {e}")
        return ""


def format_abbreviated(n: Union[int, float]) -> str:
    try:
        n = float(n)
        if abs(n) < 1000:
            return f"{n:g}"
        magnitude = min(int(math.log10(abs(n)) // 3), 5)
        suffixes = ["", "K", "M", "B", "T", "Q"]
        scaled_value = n / (1000**magnitude)
        return f"{scaled_value:.2f}{suffixes[magnitude]}".rstrip("0").rstrip(".")
    except (ValueError, TypeError):
        return str(n)


_ACTIVATION_REGISTRY = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "selu": nn.SELU,
    "tanh": nn.Tanh,
}


def get_activation(name: str) -> nn.Module:
    if name.lower() == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)

    return _ACTIVATION_REGISTRY.get(name.lower(), nn.Mish)()


def get_layer_params(layer):
    norm_val = layer.weight.norm().item()
    return format_abbreviated(norm_val)
