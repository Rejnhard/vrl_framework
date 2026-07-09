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

import random
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_seeding() -> None:
    """Configures explicit PRNG seeds for reproducible test runtime."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


@pytest.fixture(autouse=True)
def isolate_telemetry() -> None:
    """Mocks tracking integrations to prevent CI network requests."""
    with patch("wandb.init") as mock_init, patch("wandb.log"), patch("wandb.finish"):

        mock_init.return_value = MagicMock()
        yield


@pytest.fixture(autouse=True)
def force_cpu_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forces CPU device allocation."""
    import os
    import sys

    sys.path.insert(0, os.path.abspath("src"))
    monkeypatch.setattr("vrl_framework.core.settings.MODEL_DEVICE", "cpu")
    monkeypatch.setattr("bitsandbytes.optim.Adam8bit", torch.optim.Adam)


@pytest.fixture
def mock_hyperbolic_state() -> torch.Tensor:
    """Generates a geometrically valid Lorentz manifold state tensor [B, D].

    Dimensions correspond to [time (t), spatial constraints (s)].
    Satisfies constraint: -t^2 + sum(s^2) = -1, t > 0.
    """
    batch_size = 4
    spatial_dim = 255

    spatial_coords = torch.randn(batch_size, spatial_dim)
    spatial_norm_sq = torch.sum(spatial_coords**2, dim=-1, keepdim=True)
    time_coords = torch.sqrt(1.0 + spatial_norm_sq)

    state_tensor = torch.cat([time_coords, spatial_coords], dim=-1)
    state_tensor.requires_grad_(True)

    return state_tensor


@pytest.fixture
def mock_sparse_world_grid() -> torch.Tensor:
    """Generates a 3D binary tensor mask [10, 10, 10] simulating a 10% density grid."""
    grid = (torch.rand(10, 10, 10) > 0.9).to(torch.uint8)
    return grid
