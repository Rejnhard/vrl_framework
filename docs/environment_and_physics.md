# Environment Physics, Simulation Dynamics, and Metabolic Homeostasis

The VRL Framework runs the physics simulation on the GPU with NVIDIA Warp. This avoids repeated CPU-to-GPU transfers during environment updates.

This document describes the simulation environment, CUDA initialization rules, and the energy model implemented in `world_dynamics.py`.

---

## Simulation via NVIDIA Warp

The physical environment is implemented as a 3D sparse voxel grid to optimize ray-casting, collision resolution, and distance fields. Warp JIT-compiles the Python kernels directly to CUDA, minimizing step latency.
Environmental updates, collision checks, and constraints are written to PyTorch-compatible CUDA tensors (`wp.to_torch`). The model and PPO pipeline read those tensors directly on the device.

---

## 2. Concurrency and CUDA Context Fork Safety

Running an asynchronous training loop where simulation and network updates happen concurrently requires strict isolation of CUDA contexts. In `world_dynamics.py`, I handle this using a multiprocessing barrier:

```python
import multiprocessing

if multiprocessing.current_process().name == 'MainProcess':
    wp.init()
```

### The Forking Hazard

PyTorch and Warp bind CUDA context to the initializing process. Forking child processes post-initialization copies invalid state, leading to silent memory corruption or hard crashes. To prevent this, initialization is isolated to the `MainProcess`.

### Resolution

I call `wp.init()` only in the `MainProcess`. Telemetry and file I/O are handled separately so the main simulation loop keeps control of the CUDA work.

---

## 3. Metabolic Cost Model and Energy Homeostasis

The framework uses energy costs instead of an explicit external reward.

### Energy Decay Model

Agents spawn with an initial energy vector close to `DESIRED_ENERGY`. At each simulation step $t$, the internal energy $E$ updates based on an action-dependent non-linear decay function:

$$E_{t+1} = E_t - \left( \gamma_{\text{base}} + \sum_{i} \alpha_i \cdot |a_i|^2 + \psi_{\text{collision}} \cdot \|\mathbf{v}\|_{\mathcal{L}} \right)$$

In this formulation, $\gamma_{\text{base}}$ defines the static latent upkeep cost, $\alpha_i \cdot |a_i|^2$ is the quadratic penalty for Warp physics actions, and $\psi_{\text{collision}} \cdot \|\mathbf{v}\|_{\mathcal{L}}$ applies kinetic damage upon voxel collision, scaled by the Lorentz velocity norm.

### Homeostatic Thresholds

The environment monitors the population against hard energetic constraints defined in the settings:

* **DEACTIVATION_THRESHOLD:** If $E_t \le \text{Threshold}$, the agent is terminated. The PPO engine stops tracking its trajectory, prunes its state from the active replay buffer, and marks it for deactivation.
* **RECOVERY_MARGIN:** Deactivated agents can sometimes return if a local event restores enough energy and their state is still valid.

---

## 4. Evolutionary Mechanics: Replication and Selection

When an agent's policy minimizes metabolic expenditure and world-model surprisal effectively, it accumulates an energy surplus. This triggers replication intents (tracked via `replication_intents` in `world_dynamics.py`).

## Inheritance and Mutation

Agents replicate upon reaching the threshold condition: $E_t \ge \text{DESIRED\\_ENERGY} \cdot (1.0 + 2.0 \cdot \text{local\\_neighbor\\_density})$.

The new agent inherits the base weights directly. To explore the adversarial modulation space, the FiLM conditioning parameters ($\gamma$, $\beta$) are mutated via additive Gaussian noise:

$$\gamma_{\text{child}} = \gamma_{\text{parent}} + \mathcal{N}(0, \sigma_{\text{mutation}})$$

This introduces variation in behavior while preserving the shared base representation.

---

## 5. Procedural Adversary and Co-Evolution

To prevent the population from converging on trivial low-cost behavior, the framework changes the environment over time.

A convolutional network pipeline periodically modifies the 3D voxel terrain by spawning geometric barriers, dynamic hazards, and topological anomalies. I couple the complexity and frequency of these anomalies directly to the population's median fitness and the JEPA model's learning velocity. As the agents improve, the environment increases difficulty to keep the task challenging.