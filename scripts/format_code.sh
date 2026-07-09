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

c_g='\033[0;32m'
c_y='\033[1;33m'
c_r='\033[0;31m'
c_x='\033[0m'

cd "${0%/*}/.." || exit 1
set -eo pipefail

T_DIRS=("src" "tests" "scripts")

echo -e "${c_y}>> Triggering format pipeline${c_x}"

python -c "import black, isort, flake8, autoflake, mypy" 2>/dev/null || {
    echo -e "${c_r}Missing requirements. Run: pip install black isort flake8 autoflake mypy types-PyYAML${c_x}"
    exit 1
}

echo -e "${c_g}-> autoflake${c_x}"
python -m autoflake --in-place --remove-all-unused-imports --remove-unused-variables --recursive "${T_DIRS[@]}"

echo -e "${c_g}-> isort${c_x}"
python -m isort "${T_DIRS[@]}" --profile black --line-length 119

echo -e "${c_g}-> black${c_x}"
python -m black "${T_DIRS[@]}" --line-length 119

echo -e "${c_g}-> flake8${c_x}"
python -m flake8 "${T_DIRS[@]}" --max-line-length=119 --extend-ignore=E203,W503

echo -e "${c_g}-> mypy${c_x}"
python -m mypy src/ --ignore-missing-imports --no-implicit-optional --install-types --non-interactive

echo -e "${c_g}Pipeline passed successfully.${c_x}"