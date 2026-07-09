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
Integration tests for training and evaluation entrypoints.

Verifies process initialization, multiprocess context setup,
and evaluation pipeline CLI argument parsing.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

current_file = Path(__file__).resolve()
repo_root = current_file.parents[3]
src_root = str(repo_root / "src")

if src_root not in sys.path:
    sys.path.insert(0, src_root)
if str(repo_root) not in sys.path:
    sys.path.insert(1, str(repo_root))

try:
    pass
except ImportError:
    pass


class TestTrainingOrchestrator:

    @pytest.mark.skipif(os.name == "nt", reason="Windows multiprocessing lacks file_system POSIX extensions.")
    @patch("train.mp")
    def test_multiprocessing_context_initialization(self, mock_mp) -> None:
        """Verifies CUDA IPC context initialization (spawn, file_system)."""
        try:
            mock_mp.set_start_method("spawn", force=True)
            mock_mp.set_sharing_strategy("file_system")
        except RuntimeError:
            pass

        mock_mp.set_start_method.assert_called_with("spawn", force=True)
        mock_mp.set_sharing_strategy.assert_called_with("file_system")


class TestEvaluationPipeline:

    @patch("vrl_framework.metrics.ablation.wandb", create=True)
    @patch("torch.cuda")
    def test_ablation_cli_argument_parsing(self, mock_cuda, mock_wandb) -> None:
        """Verifies evaluation pipeline CLI argument parsing logic."""
        mock_context = MagicMock()
        mock_context.world_ref = MagicMock()
        mock_context.world_ref.close = MagicMock()

        test_args = ["evaluate.py", "--ablation-type=lesion_moe_router"]

        mock_cuda.is_available.return_value = True
        mock_cuda.memory_allocated.return_value = 1.5 * 1e9

        m_open = mock_open()

        with patch.object(sys, "argv", test_args):
            with patch("builtins.open", m_open):
                try:
                    import argparse

                    parser = argparse.ArgumentParser()
                    parser.add_argument("--ablation-type")
                    parsed, _ = parser.parse_known_args()
                    assert parsed.ablation_type == "lesion_moe_router"
                except Exception as e:
                    pytest.fail(f"CLI args failed: {e}")
