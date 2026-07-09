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

import sys
from pathlib import Path

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
from vrl_framework.math_ops.geometry import LorentzGeometry, RMSNorm, compute_moe_load_balancing_loss  # noqa: E402


class TestGeometryProperties:
    """Property tests for geometry module operations."""

    @settings(deadline=None)
    @given(tensor=float_tensor_strategy(st.tuples(st.integers(1, 32), st.integers(16, 128))))
    def test_rmsnorm_numerical_stability(self, tensor: torch.Tensor):
        """Test RMSNorm numerical stability during forward and backward passes."""
        # tensor: [..., dim]
        dim = tensor.shape[-1]
        layer = RMSNorm(dim=dim, eps=1e-5)

        output = layer(tensor)

        assert output.shape == tensor.shape, "Output shape mismatch in RMSNorm forward pass."
        assert not torch.isnan(output).any(), "NaNs detected in RMSNorm forward pass."
        assert not torch.isinf(output).any(), "Infs detected in RMSNorm forward pass."

        output.sum().backward()
        assert layer.weight.grad is not None, "Missing gradient for RMSNorm weight."
        assert not torch.isnan(layer.weight.grad).any(), "NaNs detected in RMSNorm weight gradients."

    @settings(deadline=None)
    @given(tensor=float_tensor_strategy(st.tuples(st.integers(1, 32), st.integers(3, 64))))
    def test_lorentz_projection_manifold_constraint(self, tensor: torch.Tensor):
        """Validate Lorentz manifold metric signature constraint."""
        # Avoid undefined origin projection.
        tensor = tensor.clamp(-10.0, 10.0) + 1e-4
        projected = LorentzGeometry.project(tensor)

        self_dot = LorentzGeometry.minkowski_dot(projected, projected)
        expected_dot = torch.full_like(self_dot, -1.0)

        assert not torch.isnan(projected).any(), "NaNs generated during Lorentz projection."
        assert torch.allclose(
            self_dot, expected_dot, atol=5e-2
        ), "Projected points broke the Lorentz manifold constraints."

    @settings(deadline=None)
    @given(tensor_x=float_tensor_strategy(st.tuples(st.integers(1, 16), st.integers(4, 32))))
    def test_lorentz_distance_stability(self, tensor_x: torch.Tensor):
        """Verify geodesic distance stability between nearby points."""
        tensor_x = tensor_x.clamp(-10.0, 10.0)
        tensor_y = (tensor_x + torch.randn_like(tensor_x)).clamp(-10.0, 10.0)

        x_proj = LorentzGeometry.project(tensor_x)
        y_proj = LorentzGeometry.project(tensor_y)

        self_dist = LorentzGeometry.distance(x_proj, x_proj)
        assert not torch.isnan(self_dist).any(), "Distance to self yielded NaNs."
        assert torch.allclose(
            self_dist, torch.zeros_like(self_dist), atol=5e-2
        ), "Geodesic distance to self must be 0."

        cross_dist = LorentzGeometry.distance(x_proj, y_proj)
        assert not torch.isnan(cross_dist).any(), "Geodesic distance calculation exploded into NaNs."
        assert (cross_dist >= 0).all(), "Geodesic distance cannot be negative."

    @settings(deadline=None)
    @given(logits=float_tensor_strategy(st.tuples(st.integers(1, 128), st.integers(4, 16))))
    def test_moe_load_balancing_stability(self, logits: torch.Tensor):
        """Test MoE load balancing loss numerical bounds."""
        # logits: [batch_size, num_experts]
        num_experts = logits.shape[-1]
        routing_weights = torch.softmax(logits, dim=-1)

        loss = compute_moe_load_balancing_loss(
            routing_weights=routing_weights, num_experts=num_experts, raw_routing_logits=logits
        )

        assert not torch.isnan(loss).any(), "MoE load balancing loss generated NaNs."
        assert not torch.isinf(loss).any(), "MoE load balancing loss generated Infinity."
        assert loss.item() >= 0, "Loss cannot be structurally negative."
