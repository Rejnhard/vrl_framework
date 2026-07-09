#!/usr/bin/env bash

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


# Runs pytest suite with deterministic cuBLAS flags.

set -euo pipefail

declare -r GREEN='\033[0;32m'
declare -r YELLOW='\033[1;33m'
declare -r NC='\033[0m'

echo -e "${YELLOW}>> Starting VRL tests...${NC}"

cd "$(dirname "$0")/.."

# cuBLAS matmul locks for strict reproducibility
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED=42

# Define target testing directories and their respective scope
declare -a TEST_DOMAINS=(
    "tests/unit"             # Isolated logic (loss functions, model layers, utilities)
    "tests/integration"      # Dataloaders, model state loading, and communication primitives
    "tests/property"         # Property-based testing for tensor shapes and bounding
    "tests/sanity"           # Hardware availability checks (CUDA contexts, multi-GPU visibility)
    "tests/performance"      # Memory leak checks and dataloader throughput benchmarks
    "tests/reproducibility"  # Seed determinism validation across environments
    "tests/e2e"              # Full pipeline smoke tests (initialization to gradient update)
)

run_test_domain() {
    local domain=$1
    echo -e "\n${GREEN}### Target: ${domain} ###${NC}"
    
    # Execute pytest strictly enforcing markers and avoiding hidden warnings
    pytest "${domain}" -v -ra --strict-markers
}

# Run tests sequentially to prevent VRAM OOM.
for domain in "${TEST_DOMAINS[@]}"; do
    if [ -d "${domain}" ]; then
        run_test_domain "${domain}"
    else
        echo -e "${YELLOW}Skip: ${domain} is missing.${NC}"
    fi
done

echo -e "\n${GREEN}All checks done.${NC}"