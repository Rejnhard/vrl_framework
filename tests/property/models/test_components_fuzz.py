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

"""Property tests for representation bottlenecks and stability trackers."""

import sys
from pathlib import Path

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

current_file = Path(__file__).resolve()
repo_root = current_file.parents[3]
src_root = str(repo_root / "src")

if src_root not in sys.path:
    sys.path.insert(0, src_root)

if str(repo_root) not in sys.path:
    sys.path.insert(1, str(repo_root))

from tests.property.strategies import float_tensor_strategy  # noqa: E402
from vrl_framework.models.components import SharedWorkspaceBottleneck  # noqa: E402
from vrl_framework.models.planners import StabilityTracker  # noqa: E402


class TestComponentProperties:
    """Test SharedWorkspaceBottleneck shape invariance."""

    @settings(deadline=None)
    @given(
        tensor=float_tensor_strategy(st.tuples(st.integers(1, 16), st.just(4), st.integers(32, 32))),
        epistemic_noise=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False),
    )
    def test_shared_workspace_bottleneck(self, tensor: torch.Tensor, epistemic_noise: float):
        """Test shape preservation and numerical stability under epistemic noise."""
        # tensor: [batch_size, num_subsystems, embed_dim]
        embed_dim = tensor.shape[-1]
        batch_size = tensor.shape[0]
        bottleneck = SharedWorkspaceBottleneck(embed_dim=embed_dim, num_subsystems=4)

        noise_tensor = torch.tensor([epistemic_noise], dtype=torch.float32)

        # output: [batch_size, embed_dim]
        output = bottleneck(tensor, epistemic_uncertainty=noise_tensor)

        expected_shape = (batch_size, embed_dim)
        assert output.shape == expected_shape, f"Expected output shape {expected_shape}, got {output.shape}."
        assert not torch.isnan(output).any(), "Detected NaNs in output; possible numerical instability in projections."


class TestPlannerProperties:
    """Test StabilityTracker boundary conditions."""

    @given(surprisal=st.floats(allow_nan=False, allow_infinity=True))
    def test_stability_tracker_epistemic_gate(self, surprisal: float):
        """Validate epistemic gate response to extreme surprisal values."""
        tracker = StabilityTracker()
        try:
            gate_status = tracker.evaluate_epistemic_gate(surprisal)
            assert isinstance(gate_status, bool), "Gate evaluation must strictly return a boolean flag."
        except OverflowError:
            pytest.fail("StabilityTracker raised OverflowError on extreme floats.")
        except Exception as e:
            pytest.fail(f"Unhandled exception in epistemic gate: {e}")
