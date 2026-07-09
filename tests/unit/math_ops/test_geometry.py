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
Unit tests for mathematical and geometric operations.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close

current_file = Path(__file__).resolve()
repo_root = current_file.parents[3]
src_root = str(repo_root / "src")

if src_root not in sys.path:
    sys.path.insert(0, src_root)
if str(repo_root) not in sys.path:
    sys.path.insert(1, str(repo_root))

from vrl_framework.math_ops.geometry import (  # noqa: E402
    LorentzGeometry,
    RMSNorm,
    compute_eff_dim,
    compute_moe_load_balancing_loss,
    format_abbreviated,
    format_time,
    get_activation,
    sigreg_weak_loss,
)


class TestRMSNorm:

    @pytest.mark.parametrize("batch_size, dim", [(2, 256), (16, 1024), (1, 64)])
    def test_forward_shape_and_type(self, batch_size: int, dim: int) -> None:
        layer = RMSNorm(dim=dim)
        x = torch.randn(batch_size, dim)
        out = layer(x)

        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_mathematical_variance_scaling(self) -> None:
        """Verifies output tensor normalization constraints (RMS ≈ 1.0)."""
        dim = 256
        layer = RMSNorm(dim=dim)

        x = torch.randn(4, dim) * 10.0 + 5.0
        out = layer(x)

        rms = torch.sqrt(torch.mean(out**2, dim=-1))
        expected_rms = torch.ones_like(rms)

        assert_close(rms, expected_rms, rtol=1e-3, atol=1e-3)

    def test_gradient_flow_integrity(self) -> None:
        """Verifies autograd flow and absence of vanishing/exploding gradients."""
        dim = 128
        layer = RMSNorm(dim=dim)
        x = torch.randn(2, dim, requires_grad=True)

        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert layer.weight.grad is not None


class TestLorentzGeometry:

    def test_minkowski_dot_product(self) -> None:
        vec_a = torch.tensor([[2.0, 1.0, 0.0, 0.0]])
        vec_b = torch.tensor([[3.0, 0.0, 1.0, 0.0]])

        expected_dot = torch.tensor([-6.0])
        computed_dot = LorentzGeometry.minkowski_dot(vec_a, vec_b)

        assert_close(computed_dot, expected_dot)

    def test_hyperboloid_projection_invariant(self) -> None:
        """Verifies projection satisfies the upper hyperboloid constraint."""
        batch_size, dim = 8, 256
        euclidean_state = torch.randn(batch_size, dim)

        manifold_state = LorentzGeometry.project(euclidean_state)

        m_dot = LorentzGeometry.minkowski_dot(manifold_state, manifold_state)
        expected_m_dot = torch.full((batch_size,), -1.0, dtype=euclidean_state.dtype)

        assert_close(m_dot, expected_m_dot, rtol=1e-4, atol=1e-4)
        assert (manifold_state[:, 0] >= 1.0).all()

    def test_geodesic_distance_properties(self) -> None:
        """Verifies geometric axioms of the distance function (identity, symmetry, non-negativity)."""
        batch_size, dim = 4, 128
        x_euclidean = torch.randn(batch_size, dim)
        y_euclidean = torch.randn(batch_size, dim)

        x_manifold = LorentzGeometry.project(x_euclidean)
        y_manifold = LorentzGeometry.project(y_euclidean)

        dist_self = LorentzGeometry.distance(x_manifold, x_manifold)

        assert (dist_self < 0.05).all()

        dist_xy = LorentzGeometry.distance(x_manifold, y_manifold)
        dist_yx = LorentzGeometry.distance(y_manifold, x_manifold)
        assert_close(dist_xy, dist_yx)

        assert (dist_xy >= 0.0).all()

    def test_exponential_map_numerical_stability(self) -> None:
        """Verifies exponential map numerical stability and gradient flow for small vectors."""
        dim = 64
        base_point = LorentzGeometry.project(torch.randn(1, dim))

        tangent_v = torch.randn(1, dim) * 1e-6
        tangent_v.requires_grad_(True)

        mapped_point = LorentzGeometry.exp_map(base_point, tangent_v)

        assert_close(mapped_point, base_point, atol=1e-3, rtol=1e-3)

        loss = mapped_point.sum()
        loss.backward()

        assert not torch.isnan(tangent_v.grad).any()


class TestLatentRegularizers:

    def test_sigreg_weak_loss_stability(self) -> None:
        batch, channels, sketch = 32, 512, 64
        x = torch.randn(batch, channels, requires_grad=True)

        loss = sigreg_weak_loss(x, sketch_dim=sketch)

        assert loss.dim() == 0
        assert loss.item() > 0.0

        loss.backward()
        assert not torch.isnan(x.grad).any()

    def test_compute_eff_dim_svd_fallback(self) -> None:
        """Verifies computation stability during matrix rank collapse."""
        collapsed_batch = torch.ones(10, 128)

        eff_dim = compute_eff_dim(collapsed_batch)

        assert not torch.isnan(eff_dim)
        assert eff_dim.item() >= 1.0

    def test_compute_moe_load_balancing_loss(self) -> None:
        routing_weights = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        loss = compute_moe_load_balancing_loss(routing_weights, num_experts=2)
        assert loss.item() > 0.0


class TestUtilities:

    @pytest.mark.parametrize(
        "activation_name, expected_type",
        [("relu", nn.ReLU), ("mish", nn.Mish), ("leaky_relu", nn.LeakyReLU), ("UNKNOWN_TRIGGER", nn.Mish)],
    )
    def test_get_activation(self, activation_name: str, expected_type: type) -> None:
        act_fn = get_activation(activation_name)
        assert isinstance(act_fn, expected_type)

    def test_format_time(self) -> None:
        assert format_time(45.5) == "45.50s"
        assert format_time(125.0) == "02m 05.00s"
        assert format_time(3665.0) == "01h 01m 05.00s"

    def test_format_abbreviated(self) -> None:
        assert format_abbreviated(999) == "999"
        assert format_abbreviated(1500) == "1.50K"
        assert format_abbreviated(2_500_000) == "2.50M"
        assert format_abbreviated("not_a_number") == "not_a_number"
