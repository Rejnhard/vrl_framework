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

"""Tests for gradient determinism and backward pass reproducibility.

Validates parameter updates, stochastic sampling, and mixed-precision scaling.
"""

import copy
import random
import sys
from pathlib import Path

import bitsandbytes as bnb
import numpy as np
import pytest
import torch
import torch.nn.functional as F

current_file = Path(__file__).resolve()
repo_root = current_file.parents[2]
src_root = str(repo_root / "src")

if src_root not in sys.path:
    sys.path.insert(0, src_root)

if str(repo_root) not in sys.path:
    sys.path.insert(1, str(repo_root))

from vrl_framework.core.settings import MODEL_DEVICE  # noqa: E402
from vrl_framework.models.agents import ActorCriticModule  # noqa: E402


def enforce_strict_determinism(seed: int = 101) -> torch.Generator:
    """Enforces hardware and software determinism for the test context."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.use_deterministic_algorithms(True, warn_only=False)

        import os

        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return torch.Generator(device=MODEL_DEVICE).manual_seed(seed)


class TestGradientDeterminism:

    @pytest.fixture
    def mock_environment_batch(self, batch_size=32, seq_len=8, obs_dim=128):
        rng = torch.Generator(device=MODEL_DEVICE).manual_seed(42)
        states = torch.randn(batch_size, seq_len, obs_dim, device=MODEL_DEVICE, generator=rng)
        actions = torch.randint(0, 16, (batch_size, seq_len), device=MODEL_DEVICE, generator=rng)
        rewards = torch.randn(batch_size, seq_len, 1, device=MODEL_DEVICE, generator=rng)
        return states, actions, rewards

    def setup_identical_models(self, seed: int):
        enforce_strict_determinism(seed)
        model_alpha = ActorCriticModule(input_dim=128, num_actions=16).to(MODEL_DEVICE)
        model_beta = copy.deepcopy(model_alpha)

        return model_alpha, model_beta

    def test_forward_backward_bit_exactness(self, mock_environment_batch):
        """Asserts bit-exact equality of computational graphs across independent passes."""
        states, actions, rewards = mock_environment_batch
        seed = 1337

        model_alpha, model_beta = self.setup_identical_models(seed)

        enforce_strict_determinism(seed)
        out_alpha = model_alpha(states)
        pred_alpha = out_alpha.policy_logits.max(dim=-1)[0].squeeze(1)
        loss_alpha = F.mse_loss(pred_alpha, rewards.squeeze(-1))
        loss_alpha.backward()

        enforce_strict_determinism(seed)
        out_beta = model_beta(states)
        pred_beta = out_beta.policy_logits.max(dim=-1)[0].squeeze(1)
        loss_beta = F.mse_loss(pred_beta, rewards.squeeze(-1))
        loss_beta.backward()

        assert torch.equal(out_alpha.policy_logits, out_beta.policy_logits), "Forward pass mismatch."
        assert loss_alpha.item() == loss_beta.item(), f"Loss mismatch: {loss_alpha.item()} != {loss_beta.item()}"

        checked = 0
        for (name_alpha, param_alpha), (name_beta, param_beta) in zip(
            model_alpha.named_parameters(), model_beta.named_parameters()
        ):
            if not param_alpha.requires_grad:
                continue

            if param_alpha.grad is None and param_beta.grad is None:
                continue

            assert param_alpha.grad is not None, f"Missing grad in Alpha: {name_alpha}"
            assert param_beta.grad is not None, f"Missing grad in Beta: {name_beta}"

            checked += 1
            match = torch.equal(param_alpha.grad, param_beta.grad)

            if not match:
                max_diff = torch.max(torch.abs(param_alpha.grad - param_beta.grad)).item()
                pytest.fail(f"Gradient determinism violation in layer '{name_alpha}'. Max drift: {max_diff}")

        assert checked > 0, "No gradients were produced for the selected loss."

    def test_optimizer_state_and_quantization_determinism(self, mock_environment_batch):
        """Validates that quantized optimizer state updates deterministically."""
        states, actions, rewards = mock_environment_batch
        seed = 404

        model_alpha, model_beta = self.setup_identical_models(seed)

        opt_alpha = bnb.optim.AdamW8bit(model_alpha.parameters(), lr=1e-4)
        opt_beta = bnb.optim.AdamW8bit(model_beta.parameters(), lr=1e-4)

        for step in range(3):
            enforce_strict_determinism(seed + step)
            opt_alpha.zero_grad()
            loss_alpha = model_alpha(states).policy_logits.sum()
            loss_alpha.backward()
            opt_alpha.step()

            enforce_strict_determinism(seed + step)
            opt_beta.zero_grad()
            loss_beta = model_beta(states).policy_logits.sum()
            loss_beta.backward()
            opt_beta.step()

        for (name_alpha, param_alpha), (name_beta, param_beta) in zip(
            model_alpha.named_parameters(), model_beta.named_parameters()
        ):
            match = torch.equal(param_alpha, param_beta)
            if not match:
                max_diff = torch.max(torch.abs(param_alpha - param_beta)).item()
                pytest.fail(
                    f"Optimizer update diverged in layer '{name_alpha}' after 3 steps. Max difference: {max_diff}"
                )

    def test_autocast_gradient_scaling_determinism(self, mock_environment_batch):
        """Validates loss scaling behavior within FP16 AMP context."""
        states, actions, rewards = mock_environment_batch
        seed = 505

        model_alpha, model_beta = self.setup_identical_models(seed)

        scaler_alpha = torch.amp.GradScaler("cuda")
        scaler_beta = torch.amp.GradScaler("cuda")

        def run_amp_step(model, scaler, reset_seed):
            enforce_strict_determinism(reset_seed)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=MODEL_DEVICE, dtype=torch.float16):
                logits = model(states).policy_logits

            loss = logits.float().var()
            assert torch.isfinite(loss), "AMP loss became non-finite."

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            grads = {}
            for name, p in model.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(), f"Non-finite AMP gradient in '{name}'."
                    grads[name] = p.grad.clone()
            return grads

        grads_alpha = run_amp_step(model_alpha, scaler_alpha, reset_seed=seed)
        grads_beta = run_amp_step(model_beta, scaler_beta, reset_seed=seed)

        for name in grads_alpha.keys():
            assert torch.equal(
                grads_alpha[name], grads_beta[name]
            ), f"AMP Gradient Scaling caused non-determinism in '{name}'."
