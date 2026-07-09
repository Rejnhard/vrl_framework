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

"""Tests for system contracts and environment validation."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

current_file = Path(__file__).resolve()
repo_root = current_file.parents[3]
src_root = str(repo_root / "src")

if src_root not in sys.path:
    sys.path.insert(0, src_root)
if str(repo_root) not in sys.path:
    sys.path.insert(1, str(repo_root))

from vrl_framework.core.contracts import ActorCriticOutput, ComputeTaskContract, PlannerRegime  # noqa: E402
from vrl_framework.core.settings import validate_environment  # noqa: E402


class TestEnvironmentValidation:

    @patch("importlib.metadata.distribution")
    def test_validate_environment_success(self, mock_distribution) -> None:
        """Verifies validation passes silently when all dependencies are present."""
        mock_distribution.return_value = MagicMock()

        try:
            validate_environment()
        except Exception as e:
            pytest.fail(f"validate_environment raised an unexpected exception: {e}")

    @patch("importlib.metadata.distribution")
    def test_validate_environment_missing_dependency(self, mock_distribution, caplog) -> None:
        """Verifies missing dependencies are caught and logged appropriately."""
        import importlib.metadata
        import logging

        def side_effect(pkg_name):
            if pkg_name == "bitsandbytes":
                raise importlib.metadata.PackageNotFoundError(pkg_name)
            return MagicMock()

        mock_distribution.side_effect = side_effect

        import pytest

        with caplog.at_level(logging.WARNING):
            with pytest.raises(RuntimeError, match="Critical dependencies missing: bitsandbytes"):
                validate_environment()

        assert "Missing dependencies detected: bitsandbytes" in caplog.text


class TestHardwareConfiguration:

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA to validate CuDNN boundaries.")
    def test_cuda_matmul_precision_flags(self) -> None:
        """Verifies TF32 is enabled globally for matrix multiplications."""
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True

    def test_environmental_isolation_variables(self) -> None:
        """Verifies standard OS-level environment variables required for PyTorch stability."""
        alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        assert "max_split_size_mb" in alloc_conf
        assert os.environ.get("NCCL_P2P_DISABLE") == "1"


class TestSystemContracts:

    def test_compute_task_contract_instantiation(self) -> None:
        """Verifies schema attribute mapping for ComputeTaskContract."""
        mock_tensor = torch.zeros(4, 256)

        contract = ComputeTaskContract(
            task_name="latent_projection",
            request_id="test_req_001",
            input_shape=(4, 256),
            input_dtype=torch.float32,
            expected_out_shape=(4, 64),
            expected_out_dtype=torch.float32,
            shared_tensor=mock_tensor,
            kwargs={"temperature": 0.5},
        )

        assert contract.input_shape == mock_tensor.shape
        assert contract.task_name == "latent_projection"

    def test_actor_critic_output_shapes(self) -> None:
        """Verifies output tensor dimension invariants."""
        b_size, act_dim = 8, 17

        out = ActorCriticOutput(
            policy_logits=torch.randn(b_size, act_dim),
            pessimistic_value=torch.randn(b_size, 1),
            cost_value=torch.randn(b_size, 1),
            intrinsic_value=torch.randn(b_size, 1),
            value_logits_1=torch.randn(b_size, 256),
            value_logits_2=torch.randn(b_size, 256),
            style_mu=torch.randn(b_size, 64),
            style_logvar=torch.randn(b_size, 64),
        )

        assert out.policy_logits.size(0) == b_size
        assert out.style_mu.shape == out.style_logvar.shape

    def test_planner_regime_enumerations(self) -> None:
        regimes = [PlannerRegime.OBSERVE_ONLY, PlannerRegime.ADVICE_ONLY]

        assert len(set(regimes)) == len(regimes)
