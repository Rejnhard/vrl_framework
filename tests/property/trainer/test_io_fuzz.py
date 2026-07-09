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

"""Fuzz testing for curriculum I/O pipeline."""

import os
import struct
import sys
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings

current_file = Path(__file__).resolve()
repo_root = current_file.parents[3]
src_root = str(repo_root / "src")

if src_root not in sys.path:
    sys.path.insert(0, src_root)

if str(repo_root) not in sys.path:
    sys.path.insert(1, str(repo_root))

from tests.property.strategies import corrupt_binary_data  # noqa: E402
from vrl_framework.trainer.ppo_engine import load_deterministic_curriculum  # noqa: E402


class TestDataIngestionProperties:
    """Test curriculum generator I/O resilience."""

    @settings(deadline=None)
    @given(binary_data=corrupt_binary_data())
    def test_curriculum_loader_resilience(self, binary_data: bytes):
        """Verify deterministic failure handling for malformed binary data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "curriculum_001.bin")
            with open(file_path, "wb") as f:
                f.write(binary_data)

            generator = load_deterministic_curriculum(temp_dir)
            if generator is None:
                return

            try:
                for batch in generator:
                    assert batch is not None
            # Expected binary parser exceptions for malformed I/O.
            except (struct.error, EOFError, ValueError, MemoryError):
                pass
            except Exception as e:
                pytest.fail(f"Unexpected exception during binary ingestion: {type(e).__name__} - {e}")
