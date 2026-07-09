# Metrics and Evaluation in Open-Ended Embodied Systems

Standard reinforcement learning typically optimizes a fixed reward signal. This framework does not optimize a single fixed reward. Instead of following one loss curve, I track stability, representation, and uncertainty metrics over time.

This document describes the main evaluation metrics, the equations behind them, and where they are logged.

## 1. The Core Problem: No Global Objective

Most RL setups define progress as the maximization of expected cumulative reward.  In my setup, a Procedural Anomaly Generator increases difficulty as the agents improve.The agents rely on intrinsic motivation—driven by Random Network Distillation (RND)—and operate under hard metabolic limits.

A useful agent is one that stays stable while handling harder environments. To verify if the agents are actually learning, the framework tracks five baseline metrics across the `world_dynamics.py`, `planners.py`, and `agents.py` modules.

## 2. Core Evaluation Metrics

### A. Generational Fitness and Energy Homeostasis
**Location:** `world_dynamics.py`

Every physical action, collision with the structure, and latent computation costs energy. I don't program fitness as an explicit objective. Instead, fitness emerges strictly from an agent's ability to maintain the `DESIRED_ENERGY` state.

Fitness is integrated as the energy delta over time. Exceeding the metabolic threshold triggers replication; dropping below the `DEACTIVATION_THRESHOLD` results in culling. Stable generational fitness against increasing environmental complexity indicates successful policy transfer.

### B. Surprisal Velocity ($\Delta \mathcal{S}$)
**Location:** `agents.py` (MCTS compute allocation)

Surprisal velocity ($\Delta \mathcal{S}$) tracks the step-wise derivative of the JEPA prediction error:

$$\Delta \mathcal{S} = \mathcal{S}_{\text{current}} - \mathcal{S}_{\text{previous}}$$

* **Mechanism:** The system tracks this metric via an exponential moving average (EMA) to dynamically allocate compute budget for the Latent-Space MCTS Planner. A spike in prediction error ($\Delta \mathcal{S} > 0$) increases the planning budget for MCTS.
* **Diagnostic Value:** Frequent, massive spikes mean the environment is too chaotic, and the procedural generator is beating the agent. A flat zero means the agent has hit a plateau or its representations have collapsed. A functional training run shows cycles of surprisal spikes followed by rapid stabilization.

### C. CBF Penalty (Constraint Violation)
**Location:** `planners.py`

I use Control Barrier Functions (CBF) to enforce physical safety constraints and prevent catastrophic collisions that would mathematically destabilize the NVIDIA Warp physics engine.

* **Mechanism:** The planner checks nominal actions against the barrier gradient constraint. The penalty is derived from the Lagrange multiplier associated with the barrier constraint:

$$L_f h(x) + L_g h(x)u + \alpha h(x) \ge 0$$

* **Diagnostic Value:** This measures the gap between the raw Actor output ($u_{\text{nom}}$) and the safe action bounds. If the CBF penalty decreases over time, the policy is producing safer actions with less correction from the safety layer.

### D. Manifold Metric Norm
**Location:** `planners.py`

Running networks on a hyperbolic manifold (the Lorentz model) introduces severe numerical instability. If vectors cross the light cone boundary, gradients explode and output NaNs.

The system evaluates the pseudo-Riemannian metric norm over the time/scale ($t$) and space ($s$) dimensions:

$$\|x\|_{\mathcal{L}} = \sqrt{\left| \sum s_{\text{comp}}^2 - t_{\text{comp}}^2 \right| + \epsilon}$$

* **Diagnostic Value:** This metric is a numerical stability check. Large swings or near-zero values indicate that the latent state may be leaving the stable operating range.

### E. Active Population Density
**Location:** `world_dynamics.py`

* **Mechanism:** I count the absolute number of active entities in the simulation and check them against the `MIN_POPULATION` and `MAX_POPULATION` hardcodes.
* **Diagnostic Value:** It shows if the current evolutionary setup is actually viable. If the population crashes toward the `DEACTIVATION_THRESHOLD`, the environment configuration is too hostile. If it constantly rides the upper population cap, my metabolic costs are too forgiving.

## 3. Telemetry and Visual Diagnostics

Multiprocessing telemetry is routed directly through Weights & Biases (W&B).

### `visualize_intelligence_metrics` Pipeline
This function fires at predefined intervals inside `world_dynamics.py` to dump the simulation state:

* **Entity Snapshot:** Grabs the 3D spatial coordinates and latent embeddings for the entire population.
* **Core Structure Extraction:** Finds the entity with the highest generational fitness and pulls its latent representations and FiLM modulation weights via `visualize_agent_core_structure`. This gives a snapshot of the latent features and modulation weights used by the current best agent.
* **W&B Payload:** Bundles the generation counts, `StabilityTracker` EMA stats, and metabolic distributions into a single dictionary, then fires off a non-blocking `wandb.log(payload)`.

### Evaluating Intrinsic Motivation
Because external rewards are non-existent, the agent explores based on the variance in the RND target network's prediction error. I verify this system by tracking the correlation between the RND's epistemic uncertainty and the actual voxel map regions the agent targets. This helps me check whether exploration is focused on regions where prediction error is informative rather than purely random.