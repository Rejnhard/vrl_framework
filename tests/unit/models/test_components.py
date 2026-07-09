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

"""Unit tests for core neural network components."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.testing import assert_close

current_file = Path(__file__).resolve()
repo_root = current_file.parents[3]
src_root = str(repo_root / "src")

if src_root not in sys.path:
    sys.path.insert(0, src_root)
if str(repo_root) not in sys.path:
    sys.path.insert(1, str(repo_root))

from vrl_framework.core.settings import MODEL_DEVICE  # noqa: E402
from vrl_framework.models.components import (  # noqa: E402
    FrequencyDomainBinding,
    HebbianLinear,
    ImplicitDeepEquilibriumLayer,
    RunningMeanStd,
    StochasticPolicy,
    TopKActivation,
)


class TestFrequencyDomainBinding:

    def test_spherical_normalization_invariant(self) -> None:
        """Tests if bound representations strictly project onto the L2 unit hypersphere."""
        dim = 256
        u = torch.randn(4, dim)
        v = torch.randn(4, dim)

        bound_state = FrequencyDomainBinding.bind(u, v)
        norms = torch.norm(bound_state, p=2, dim=-1)
        expected_norms = torch.ones_like(norms)

        assert_close(
            norms, expected_norms, rtol=1e-4, atol=1e-4, msg="VSA bound states breached the unit hypersphere boundary."
        )

    def test_binding_unbinding_identity(self) -> None:
        """Evaluates Holographic Reduced Representation (HRR) unbinding approximation."""
        # High dimensionality is required to mitigate crosstalk noise in HRR.
        dim = 512
        vec_a = F.normalize(torch.randn(2, dim, device=MODEL_DEVICE), p=2, dim=-1)
        vec_b = F.normalize(torch.randn(2, dim, device=MODEL_DEVICE), p=2, dim=-1)

        bound_ab = FrequencyDomainBinding.bind(vec_a, vec_b)
        recovered_a = FrequencyDomainBinding.unbind(bound_ab, vec_b)

        cos_sim = F.cosine_similarity(vec_a, recovered_a, dim=-1)

        assert (cos_sim > 0.50).all(), f"Expected empirical cosine similarity > 0.50 for HRR retrieval, got: {cos_sim}"


class TestRunningMeanStd:

    def test_convergence_to_true_moments(self) -> None:
        """Verifies EMA convergence to population mean and variance."""
        dim = 64
        tracker = RunningMeanStd(shape=(dim,), momentum=0.1).to(MODEL_DEVICE)

        init_batch = torch.randn(10, dim, device=MODEL_DEVICE)
        tracker.update(init_batch)

        assert tracker.initialized.item() == 1.0, "Tracker failed to register initialization."
        assert not torch.isnan(tracker.mean).any(), "NaN in EMA mean."
        assert (tracker.var > 0).all(), "Variance must remain strictly positive."


class TestTopKActivation:

    def test_sparsity_constraint(self) -> None:
        """Tests strict enforcement of K non-zero elements per vector."""
        k = 16
        dim = 128
        layer = TopKActivation(sparsity_k=k)
        x = torch.randn(4, dim)

        out = layer(x)
        non_zero_counts = (out != 0.0).sum(dim=-1)
        expected_counts = torch.full((4,), k, dtype=torch.long)

        assert (
            non_zero_counts == expected_counts
        ).all(), f"TopK violated sparsity constraint. Expected {k} non-zeros."

    def test_variance_preservation(self) -> None:
        """Tests if sparsification maintains input signal variance bounds."""
        dim = 256
        layer = TopKActivation(sparsity_k=32)
        x = torch.randn(16, dim)

        out = layer(x)

        var_in = torch.var(x, dim=-1).mean()
        var_out = torch.var(out, dim=-1).mean()

        ratio = var_out / var_in
        assert 0.5 < ratio < 2.0, f"Variance scaling failed. Ratio In/Out diverges significantly: {ratio:.2f}"


class TestHebbianLinear:

    def test_hebbian_trace_accumulation(self) -> None:
        """Validates trace accumulation against the structural sparsity mask."""
        in_dim, out_dim = 64, 32
        layer = HebbianLinear(in_dim, out_dim, sparsity=0.5).to(MODEL_DEVICE)
        layer.train()

        x = torch.randn(4, in_dim, device=MODEL_DEVICE)

        initial_trace = layer.hebbian_trace.clone()
        layer(x)

        surprise_signal = 1.5
        layer.apply_hebbian_update(surprise_signal=surprise_signal)

        assert not torch.allclose(
            initial_trace, layer.hebbian_trace
        ), "Hebbian trace failed to accumulate local updates."

        masked_trace_elements = layer.hebbian_trace[layer.weight_mask == 0.0]
        assert (masked_trace_elements == 0.0).all(), "Hebbian trace violated the structural sparsity mask."


class TestStochasticPolicy:

    def test_temperature_clamping_and_sampling(self) -> None:
        """Validates deterministic argmax routing and stochastic sampling dimensions."""
        num_actions = 8
        policy = StochasticPolicy(num_actions=num_actions)
        logits = torch.randn(2, num_actions)

        action_det, log_prob_det = policy(logits, deterministic=True)
        expected_det = torch.argmax(logits / policy.temperature, dim=-1)
        assert (action_det == expected_det).all(), "Deterministic routing failed to extract argmax."

        action_stoch, log_prob_stoch = policy(logits, deterministic=False)
        assert action_stoch.shape == (2,), "Stochastic sampling dimension mismatch."
        assert log_prob_stoch.shape == (2,), "Log probability dimension mismatch."


class TestImplicitDeepEquilibriumLayer:

    def test_equilibrium_forward_and_implicit_backward(self) -> None:
        """Tests DEQ forward convergence and implicit gradient flow via IFT."""
        in_dim, hid_dim, out_dim = 16, 64, 32
        layer = ImplicitDeepEquilibriumLayer(in_dim, hid_dim, out_dim, max_iter=20).to(MODEL_DEVICE)

        x = torch.randn(4, in_dim, requires_grad=True, device=MODEL_DEVICE)

        out = layer(x)
        assert out.shape == (4, out_dim), "DEQ output topology mismatch."
        assert not torch.isnan(out).any(), "DEQ forward iteration diverged to NaN."

        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Implicit theorem solver severed the autograd graph."
        assert not torch.isnan(x.grad).any(), "Implicit backward pass induced NaN gradients."
        assert layer.weight_z.grad is not None, "DEQ internal weights bypassed by autograd."
