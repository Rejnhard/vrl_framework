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

"""Hypothesis data generation strategies."""

from typing import Tuple

import numpy as np
import torch
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


def valid_tensor_shapes() -> st.SearchStrategy[Tuple[int, ...]]:
    """Generate 3D tensor shapes."""
    return st.tuples(
        # Prevent CI OOM errors.
        st.integers(min_value=1, max_value=64),
        st.integers(min_value=1, max_value=128),
        st.integers(min_value=8, max_value=256),
    )


def float_tensor_strategy(shape_strategy: st.SearchStrategy) -> st.SearchStrategy[torch.Tensor]:
    """Generate float32 PyTorch tensors bounded to [-1e10, 1e10]."""
    return shape_strategy.flatmap(
        lambda shape: arrays(
            dtype=np.float32,
            shape=shape,
            elements=st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False, width=32),
        ).map(lambda arr: torch.from_numpy(arr))
    )


def corrupt_binary_data() -> st.SearchStrategy[bytes]:
    """Generate arbitrary byte strings up to 64KB."""
    return st.binary(min_size=0, max_size=1024 * 64)
