# VRL Framework

[![Warp Physics](https://img.shields.io/badge/Physics-NVIDIA%20Warp-76B900?style=flat-square)](https://github.com/NVIDIA/warp)
[![Backend-PyTorch](https://img.shields.io/badge/Backend-PyTorch%202.x-EE4C2C?style=flat-square)](https://pytorch.org/)
[![Development-Status](https://img.shields.io/badge/Status-Prototype%20/%20Research%20Scaffolding-orange?style=flat-square)]()

Core implementation of an experimental engine simulating open-ended evolution across multiple agents.

Core stack: JEPA + latent MCTS mapped to a Lorentz manifold. NVIDIA Warp handles the physics step. 

Strictly a research prototype - not production code. It's just a research baseline. I mainly focused on getting stable numbers on 8k epochs runs and pushing as much data through as possible.

---

## Empirical Validation & Training Dynamics

Lorentz manifold limits and MCTS budgets tuned over 700+ runs (>380h GPU runtime).

[![W&B Run (8000 Epochs)](https://img.shields.io/badge/W&B_Report-8000_Epochs-FFCC33?logo=WeightsAndBiases&logoColor=black)](https://api.wandb.ai/links/rejnhard-/7cnkxlw3)

[![W&B All Metrics (8000 Epochs)](https://img.shields.io/badge/W&B_All_Metrics-8000_Epochs-FFCC33?logo=WeightsAndBiases&logoColor=black)](https://api.wandb.ai/links/rejnhard-/hl24v0bh)

*Visual confirmation of stable dual-stream JEPA gradients. Zero representation collapse. Lorentz metric boundaries fully preserved.*

![Manifold topology render](docs/training_evolution.gif)
*8,000 epoch progression.*

**Hardware Footprint:**
8k epochs baseline hardware: 3060 Ti, i7-8700, 24GB.
VRAM consumption kept strictly within consumer limits via `bitsandbytes` 8-bit quantization and dynamic batching. Local execution ready, no cluster GPUs required.

---

## Documentation

See the `docs/` folder for details:
* [`architecture_and_theory.md`](docs/architecture_and_theory.md) - Lorentz manifold math, `TangentSpaceFSQ` and `SharedWorkspaceBottleneck`.
* [`environment_and_physics.md`](docs/environment_and_physics.md) - NVIDIA Warp backend, voxel generation and metabolism.
* [`execution_and_training_pipeline.md`](docs/execution_and_training_pipeline.md) - Configs (`CFG`, `PLAN_CFG`, `TRAIN_CFG`).
* [`metrics_and_evaluation.md`](docs/metrics_and_evaluation.md) - Evaluation metrics.

---

## Core Architecture

The architecture has 5 parts:

1. Dual-Stream JEPA Core - maps data to a hyperbolic manifold to avoid representation collapse.
2. Latent-Space MCTS Planner - plans trajectories in the hidden space. Compute budget scales with prediction error.
3. Warp Physics Substrate - runs physics on the GPU to avoid PCIe bottlenecks.
4. Asynchronous PPO Engine - distributed optimizer with semantic rollback for recovering from gradient spikes.
5. Procedural Anomaly Generator - modulates environment topology based on learning velocity.

---

## Data Pipeline & Memory Management

Processing large data streams in open-ended RL often leads to OOM errors or silent process crashes. VRL bypasses standard I/O overhead by managing data at the OS level.

### Deterministic Curriculum Streaming (`mmap`)
Standard data loading for 10 GB - 1.5 TB datasets is too slow. The PPO Engine (`ppo_engine.py`) bypasses this with `load_deterministic_curriculum`, which uses memory-mapped file descriptors.

Curriculum files are read as binary chunks (16 MB blocks by default) directly from the kernel address space. As a result, virtual memory conforms dynamically to the file structure, allowing the framework to process hundreds of gigabytes sequentially on RAM-constrained machines without allocation overhead.

> **Warning:** The `--curriculum` flag and LMDB external memory integration (e.g., streaming Wikipedia priors) are strictly experimental. While the I/O pipeline is stable and avoids hard crashes, optimal convergence behavior when mixing the Lorentz representation with external text priors requires further validation.

---

## Evaluation Metrics

Optimizing a single scalar reward function is not viable in an autotelic system. Instead, the framework tracks a vector of five indicators to measure stability and evolutionary progress:

| Metric | Source | Description |
|---|---|---|
| **Generational Fitness** | `world_dynamics.py` | Energy balance – shows if agents can survive and replicate. |
| **Surprisal Velocity** | `agents.py` / `planners.py` | The derivative of the world model's prediction error. Used to dynamically allocate the MCTS compute budget. |
| **CBF Penalty** | `planners.py` | Captures Control Barrier Function violations to assess how well agents adapt to hard physical boundaries. |
| **Manifold Metric Norm** | `planners.py` | A geometric sanity check ensuring latent state vectors maintain their metric signature on the Lorentz manifold without degrading to NaN/Inf. |
| **Active Population Density** | `world_dynamics.py` | Prevents both demographic collapse and uncontrolled geometric explosion of the population. |

These metrics, alongside spatial distributions and entity trajectories, are aggregated in real-time and streamed to Weights & Biases via the `visualize_intelligence_metrics` pipeline.

---

## Execution Pipeline

The main entry point for the asynchronous workflow between physics steps and network updates is `train.py`.

* `wp.init()` is isolated to `MainProcess` to prevent CUDA context collisions.
* Physics, `mmap` streaming, and PPO steps run concurrently via `concurrent.futures`.
* Optimizer state is quantized with `bitsandbytes` to reduce VRAM usage on single GPUs.

---

## Testing

The testing suite covers geometric invariants during Lorentz transformations and semantic rollback stability under gradient fault injection.

**Geometry & Math Diagnostics:**

    python train.py --run-diagnostics

**Ablation & Fault Injection (Lesion Mode):**

    python train.py --mode research --ablation-type lesion

---

## Quick Start

**New Training Run**
Start the simulation loop without loading checkpoints.
```bash
python train.py --mode default --epochs 400
```

**Resume Continuous Training**
Load a saved checkpoint and continue policy optimization.
1. Download weights from [Releases](https://github.com/Rejnhard/vrl_framework/releases/tag/v0.1.0-alpha).
2. Run execution:
```bash
python train.py --resume test_data/checkpoint_gen_8000.pt --lora-path test_data/lora_skill_gen_8000.pt --epochs 8000
```

**Run a Diagnostic Benchmark**
Execute a frozen-weight verification pass. The script runs standalone diagnostic metrics and saves network topology layouts directly to W&B without stepping the active environment.
```bash
python train.py --eval --resume test_data/checkpoint_gen_8000.pt --lora-path test_data/lora_skill_gen_8000.pt
```

**Temporal Reasoning Test:**
Single forward pass to verify MCTS and text conditioning.
*Note:* The sandbox environment operates without explicit language-grounding datasets. Using "hello world" provides a deterministic seed for the latent embedding layer. This isolates the raw mechanical output of the planner (like T+0 -> T+1 discrete intents), entirely bypassing semantic processing.
```bash
python train.py --resume test_data/checkpoint_gen_8000.pt --lora-path test_data/lora_skill_gen_8000.pt --query "Hello world"
```
**Expected Output:**
Console logs raw, discrete intent trajectories from the internal state machine (50-step horizon). It is normal for the ungrounded policy to eventually settle into an attractor state (repeating tokens) deep in the temporal rollout.

```text
[Query Output]
T+0: [8, 14] ->
T+1: [72, 254] ->
T+2: [195, 11] ->
...
T+37: [93, 129] ->
T+38: [93, 129] ->
T+39: [93, 129] ->
T+40: [93, 253] ->
...
```