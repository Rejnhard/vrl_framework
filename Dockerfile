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

# NVCC and C++ headers mandatory for warp-lang/bitsandbytes JIT
# Runtime base image lacks these dependencies
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda PATH=/opt/conda/bin:$PATH
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
ENV WANDB_DISABLE_GIT=True CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes

WORKDIR /workspace

# OS-level prerequisites for PyTorch C++ extensions build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    curl \
    wget \
    libffi-dev \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && /bin/bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda clean --all --yes \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# Environment file copied first for Docker layer caching
COPY environment.yml /workspace/
RUN conda env create -f /workspace/environment.yml \
    && conda clean -afy

# Bring in the application codebase
COPY . /workspace/

# Live code edits require mounting local workspace over this image
SHELL ["conda", "run", "-n", "vrl_env", "/bin/bash", "-c"]
RUN pip install --no-cache-dir .

# Default container entrypoint
ENTRYPOINT ["/opt/conda/envs/vrl_env/bin/python", "train.py"]