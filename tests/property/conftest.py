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

"""Hypothesis configurations and environment fixtures."""

import gc
import os
import random
import warnings
from datetime import timedelta
from typing import Generator

import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, Phase, Verbosity, settings


def _enforce_determinism(seed: int = 42) -> None:
    """Enforce global determinism."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Suppress false-positive health checks due to GPU sync/allocation latency.
ML_SUPPRESSED_CHECKS = (
    HealthCheck.data_too_large,
    HealthCheck.too_slow,
    HealthCheck.filter_too_much,
)

# Dev profile.
settings.register_profile(
    "dev",
    max_examples=100,
    verbosity=Verbosity.normal,
    phases=[Phase.generate, Phase.target, Phase.shrink],
    suppress_health_check=ML_SUPPRESSED_CHECKS,
    deadline=timedelta(milliseconds=800),
)

# CI profile.
settings.register_profile(
    "ci",
    max_examples=2500,
    verbosity=Verbosity.normal,
    derandomize=True,
    suppress_health_check=ML_SUPPRESSED_CHECKS,
    deadline=timedelta(seconds=2),
)

# Deep profile.
settings.register_profile(
    "deep",
    max_examples=50000,
    verbosity=Verbosity.quiet,
    phases=[Phase.explicit, Phase.generate, Phase.target, Phase.shrink],
    suppress_health_check=ML_SUPPRESSED_CHECKS,
    deadline=None,
)

settings.load_profile(os.getenv("FUZZ_PROFILE", "dev"))


def pytest_configure(config: pytest.Config) -> None:
    """Register pytest configurations and suppress warnings."""
    config.addinivalue_line("markers", "fuzz: marks tests using Hypothesis.")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")


@pytest.fixture(autouse=True)
def isolate_fuzzing_environment() -> Generator[None, None, None]:
    """Enforce determinism and collect CUDA memory between iterations."""
    _enforce_determinism()

    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
