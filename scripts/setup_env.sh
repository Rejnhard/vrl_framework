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

# env init & cuda limits

c_g='\033[0;32m'
c_y='\033[1;33m'
c_r='\033[0;31m'
c_x='\033[0m'

cd "${0%/*}/.." || exit 1
set -eo pipefail

TARGET_ENV="vrl_env"

echo -e "${c_y}>> vrl_env init${c_x}"

hash nvcc 2>/dev/null || echo -e "${c_y}NVCC missing. C++ builds will drop.${c_x}"

_build_conda() {
    echo -e "${c_g}Conda active. Loading yml.${c_x}"
    conda info --envs | grep -q "^${TARGET_ENV} " && {
        echo -e "${c_y}Env exists -> updating.${c_x}"
        conda env update -f environment.yml --prune
    } || conda env create -f environment.yml

    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate "${TARGET_ENV}"
}

_build_venv() {
    echo -e "${c_r}Conda absent. Fallback: venv + pip.${c_x}"
    python3 -m venv .venv
    [ -f ".venv/Scripts/activate" ] && . .venv/Scripts/activate || . .venv/bin/activate
    
    echo -e "${c_g}Fetching requirements...${c_x}"
    pip install -U pip >/dev/null
    pip install -r requirements.txt
}

type conda >/dev/null 2>&1 && _build_conda || _build_venv

echo -e "${c_g}Mounting editable package.${c_x}"
pip install -e ".[dev]"

echo -e "${c_g}Dumping .env state.${c_x}"

m_split="max_split_size_mb:128"
m_garb="garbage_collection_threshold:0.8"
m_div="roundup_power2_divisions:8"

cuda_mem_cfg="${m_split},${m_garb},${m_div}"

{
    echo "PYTORCH_CUDA_ALLOC_CONF=\"${cuda_mem_cfg}\""
    echo 'BITSANDBYTES_NOWELCOME=1'
    echo 'TF_CPP_MIN_LOG_LEVEL=3'
} > .env

export PYTORCH_CUDA_ALLOC_CONF="${cuda_mem_cfg}"
export BITSANDBYTES_NOWELCOME=1
export TF_CPP_MIN_LOG_LEVEL=3

echo -e "${c_g}Ready. Run -> conda activate ${TARGET_ENV}${c_x}"