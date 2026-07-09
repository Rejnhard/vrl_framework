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

# VRL Framework Multi-GPU training entrypoint via torchrun.

# NCCL crashes on certain topologies without this
export NCCL_P2P_DISABLE=1

set -euo pipefail

# Mitigate thread contention in dataloaders by distributing available cores
TOTAL_CORES=${NUMBER_OF_PROCESSORS:-$(nproc)}
NUM_GPUS=${NUM_GPUS:-1}
export OMP_NUM_THREADS=$(( TOTAL_CORES / NUM_GPUS ))
[ "$OMP_NUM_THREADS" -eq 0 ] && export OMP_NUM_THREADS=1

# Memory and scheduling locks
_ALLOC_BASE="max_split_size_mb:128,garbage_collection_threshold:0.8"
export PYTORCH_CUDA_ALLOC_CONF="${_ALLOC_BASE},roundup_power2_divisions:8"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export USE_LIBUV=0

cd "$(dirname "$0")/.."


MODE=${TRAIN_MODE:-"omni"}
EPOCHS=${MAX_EPOCHS:-10000}
MASTER_PORT=${MASTER_PORT:-29500}

echo " Launching VRL Framework Distributed Training"
echo "-> Mode: ${MODE} (Epochs: ${EPOCHS})"
echo "-> GPUs: ${NUM_GPUS}"
echo "-> MemConfig: ${PYTORCH_CUDA_ALLOC_CONF}"


python -c "
import os
import sys
import runpy
import torch.distributed as rdzv

os.environ.setdefault('USE_LIBUV', '0')

original_tcp_store = rdzv.TCPStore

class PatchedTCPStore(original_tcp_store):
    def __init__(self, *args, **kwargs):
        kwargs['use_libuv'] = False
        super().__init__(*args, **kwargs)

rdzv.TCPStore = PatchedTCPStore

sys.argv = [
    'torchrun',
    '--standalone',
    '--nnodes=1',
    '--nproc_per_node=${NUM_GPUS}',
    '--master_port=${MASTER_PORT}',
    'train.py',
    '--mode', '${MODE}',
    '--epochs', '${EPOCHS}',
    '--deterministic'
]

_target_module = 'torch.distributed.run'
_exec_scope = '__main__'
runpy.run_module(_target_module, run_name=_exec_scope)
"