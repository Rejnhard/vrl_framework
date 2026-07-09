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


import os
import subprocess
import sys

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="E2E test requires CUDA to compile Warp physics kernels.")
class TestFullSystemExecution:
    """End-to-end smoke tests for the primary training pipeline."""

    def test_train_entrypoint_micro_run(self) -> None:
        """Executes train.py as a subprocess with minimal hyperparameter constraints."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "../../"))
        train_script = os.path.join(project_root, "train.py")

        assert os.path.exists(train_script), f"E2E framework cannot locate entrypoint at {train_script}"

        # Override core/settings.py via ENV for micro-run simulation.
        env_overrides = os.environ.copy()

        env_overrides["VRL_WORLD_DIM"] = "4"
        env_overrides["VRL_INIT_POP"] = "2"
        env_overrides["VRL_MAX_POP"] = "4"

        env_overrides["VRL_BATCH_SIZE"] = "2"
        env_overrides["VRL_LATENT_DIM"] = "32"

        env_overrides["VRL_MAX_TRAIN_STEPS"] = "2"
        env_overrides["VRL_MAX_EPOCHS"] = "1"
        env_overrides["VRL_EPOCHS"] = "1"

        env_overrides["WANDB_MODE"] = "disabled"
        env_overrides["PYTHONPATH"] = os.path.abspath(os.path.join(project_root, "src"))

        try:
            result = subprocess.run([sys.executable, train_script], env=env_overrides, cwd=project_root, timeout=900)
        except subprocess.TimeoutExpired:
            pytest.fail("E2E System Test timed out. Potential MCTS infinite loop or IPC deadlock.")

        if result.returncode != 0:
            pytest.fail(f"E2E System Execution FAILED with Exit Code {result.returncode}.")
