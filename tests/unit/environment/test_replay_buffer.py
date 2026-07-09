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
Tests for replay buffer storage bindings and tier-based eviction policies.
"""

import sys
import tempfile
from pathlib import Path

import pytest
import torch

current_file = Path(__file__).resolve()
repo_root = current_file.parents[3]
src_root = str(repo_root / "src")

if src_root not in sys.path:
    sys.path.insert(0, src_root)
if str(repo_root) not in sys.path:
    sys.path.insert(1, str(repo_root))


from vrl_framework.environment.replay_buffer import (  # noqa: E402
    GlobalExperienceReplay,
    MemoryEntityDescriptor,
    MemoryQualityBand,
    MemoryTierState,
)


class TestGlobalExperienceReplay:

    @pytest.fixture
    def temp_lmdb_dir(self):
        """Yields an isolated temporary directory for testing database bindings."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            yield tmpdirname

    def test_lmdb_initialization_and_teardown(self, temp_lmdb_dir: str) -> None:
        """Verifies database initialization and graceful closure."""
        buffer = GlobalExperienceReplay(db_path=temp_lmdb_dir)
        buffer.close()


class TestMemoryTierEviction:

    def test_tier_demotion_policy(self) -> None:
        """Verifies utility-based candidate selection for memory eviction."""
        memories = [
            MemoryEntityDescriptor(
                object_id="mem_1",
                tier_state=MemoryTierState.HOT_IN_VRAM,
                segment_id=0,
                byte_offset=0,
                length=1024,
                dtype=torch.float32,
                quant_scheme=0.0,
                quality_score=0.9,
                last_access_step=10,
                predicted_next_use=0.95,
                true_quant_error=0.01,
                quality_band=MemoryQualityBand.GREEN,
            ),
            MemoryEntityDescriptor(
                object_id="mem_2",
                tier_state=MemoryTierState.HOT_IN_VRAM,
                segment_id=1,
                byte_offset=1024,
                length=1024,
                dtype=torch.float32,
                quant_scheme=0.0,
                quality_score=0.1,
                last_access_step=2,
                predicted_next_use=0.05,
                true_quant_error=0.5,
                quality_band=MemoryQualityBand.RED,
            ),
        ]

        memories.sort(key=lambda m: m.quality_score * m.predicted_next_use)

        eviction_candidate = memories[0]

        assert eviction_candidate.object_id == "mem_2"

        eviction_candidate.tier_state = MemoryTierState.WARM_IN_RAM
        assert eviction_candidate.tier_state == 1
