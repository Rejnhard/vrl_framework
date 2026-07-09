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

SHELL := /bin/bash

MODULE_NAME := vrl_framework
SRC_DIR := src/$(MODULE_NAME)
TESTS_DIR := tests
SCRIPTS_DIR := scripts

DOCKER_IMAGE_NAME := vrl_framework_env
DOCKER_TAG := latest

PYTHON := python3

.PHONY: help install clean
.PHONY: lint format test
.PHONY: train train-cluster
.PHONY: docker-build docker-run docker-all

help:
	@echo "VRL Framework - Available commands:"
	@echo "  make install      - Install dependencies."
	@echo "  make test         - Run pytest suite."
	@echo "  make lint         - Run static analysis."
	@echo "  make format       - Auto-format code."
	@echo "  make train        - Run local training loop."
	@echo "  make train-cluster- Submit training job to cluster."
	@echo "  make docker-build - Build Docker image."
	@echo "  make docker-run   - Launch Docker container interactively."
	@echo "  make docker-all   - Build and launch Docker container."
	@echo "  make clean        - Remove build artifacts and caches."

install:
	@echo "--> Installing dependencies..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

format:
	@echo "--> Formatting code..."
	@bash $(SCRIPTS_DIR)/format_code.sh

lint:
	@echo "--> Running linters..."
	@bash $(SCRIPTS_DIR)/format_code.sh --check-only

test:
	@echo "--> Running tests..."
	@bash $(SCRIPTS_DIR)/run_tests.sh

train:
	@echo "--> Starting local training..."
	$(PYTHON) train.py

train-cluster:
	@echo "--> Submitting training job..."
	@bash $(SCRIPTS_DIR)/run_cluster_training.sh

docker-build:
	@echo "--> Building $(DOCKER_IMAGE_NAME):$(DOCKER_TAG)..."
	@bash $(SCRIPTS_DIR)/build_docker.sh

docker-run:
	@echo "--> Starting container..."
	docker run -it --rm \
		--gpus all \
		-v $(PWD):/workspace \
		-w /workspace \
		$(DOCKER_IMAGE_NAME):$(DOCKER_TAG) /bin/bash

docker-all: docker-build docker-run

clean:
	@echo "--> Cleaning workspace..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache
	rm -rf .hypothesis/examples
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info