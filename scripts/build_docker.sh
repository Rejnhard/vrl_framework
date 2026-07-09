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

# VRL Framework Docker Build Script

# build_docker.sh [--no-cache]

c_g='\033[0;32m'
c_y='\033[1;33m'
c_r='\033[0;31m'
c_x='\033[0m'

cd "${0%/*}/.." || exit 1
set -eo pipefail

IMG="vrl_framework_env"
DFILE="Dockerfile"
B_ARGS=""

echo -e "${c_y}>> Triggering local docker build${c_x}"

[ "${1:-}" == "--no-cache" ] && {
    echo -e "${c_y}Cache disabled.${c_x}"
    B_ARGS="--no-cache"
}

[ ! -f "${DFILE}" ] && {
    echo -e "${c_r}Missing ${DFILE} at root.${c_x}"
    exit 1
}

# grep cuda tag, fallback to cu121
CU_TAG=$(grep -oP 'cu\d+' requirements.txt | head -n 1 || echo "cu121")
echo -e "${c_g}Targeting CUDA: ${CU_TAG}${c_x}"

docker build ${B_ARGS} \
    -t "${IMG}:latest" \
    -t "${IMG}:${CU_TAG}" \
    -f "${DFILE}" .

# post-build check
docker image inspect "${IMG}:latest" >/dev/null 2>&1 && {
    echo -e "${c_g}Image ready: ${IMG}:latest${c_x}"
} || {
    echo -e "${c_r}Build step dropped or image missing.${c_x}"
    exit 1
}