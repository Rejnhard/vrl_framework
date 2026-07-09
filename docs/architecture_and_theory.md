# Architectural Theory and Representation Geometry

The VRL Framework uses a different latent representation than a standard RL stack. Instead of assuming a flat Euclidean space, it uses a geometry that better fits hierarchical and expanding state structure.

This maps the core math directly to the PyTorch code inside `components.py` and `agents.py`.


---

## Hyperbolic Geometry (Lorentz Manifold)

State components are projected into hyperbolic space. This models expanding, tree-like state structures more accurately than Euclidean mapping.

We split latent vectors $x \in \mathbb{R}^{d+1}$ into scale $t$ and spatial coordinates $s$. The code in `math_ops/geometry.py` and `planners.py` computes their pseudo-Riemannian metric norm like this:

$$\|x\|_{\mathcal{L}} = \sqrt{\left| \sum_{i=1}^{d} s_i^2 - t^2 \right| + \epsilon}$$
The Latent-Space MCTS Planner checks this norm during each forward pass. If a vector moves too close to the light-cone boundary, training becomes unstable, so we clamp and regularize the representation before optimization.

---

## 2. Joint Embedding Predictive Architecture (JEPA)

Representation collapse occurs when different inputs are mapped to nearly the same vector. I avoid this by using a Dual-Stream JEPA Core.

Generative models like Autoencoders or Diffusion networks try to reconstruct raw pixel or voxel inputs. JEPA predicts the future state's latent representation instead of reconstructing the raw input. The architecture splits into two streams:
* **Context Stream:** Processes the current state observation.
* **Target Stream:** Generates the uncorrupted target representation for the future state.

I optimize the network so that the predicted future latent state matches the target latent state for a given action. This makes the model focus on predictable structure rather than reconstructing every variation in the input.

---

## 3. Shared Workspace Bottleneck

Different latent streams need a shared coordination step. The `SharedWorkspaceBottleneck` (`components.py`) aligns latent streams. It projects vectors into amplitude and phase components (`amplitude_proj`, `phase_proj`) and computes $\cos(\Delta \theta)$ to penalize phase divergence across subsystems. This encourages the streams to stay aligned before their outputs are passed to the Actor-Critic networks or the MCTS planner.

---

## 4. Epistemic Uncertainty and Adversarial Modulation

Systems without a global, external loss function depend heavily on intrinsic motivation. The risk here is naive exploration, which often results in the "noisy TV problem" where agents fixate on uncontrollable random noise.

### Dynamic Perturbation Injection (`components.py`)
The framework uses variance in JEPA predictions as an estimate of epistemic uncertainty and uses that signal to guide exploration.

Using this uncertainty, we compute a `dynamic_perturbation` scale and inject Gumbel noise into the continuous action space during internal simulation:

$$a_{\text{perturbed}} = \text{softmax}\left( \frac{a + \text{Gumbel}(0, \text{scale} \cdot \sigma_{\text{epistemic}})}{\tau} \right)$$

To enforce action stability, we compute the divergence $\Delta_{\text{div}}$ between state predictions generated with positive ($a_+$) and negative ($a_-$) perturbations.

---

## 5. Temporal Abstraction and Dynamic MCTS Depth

Computing deep MCTS trajectories in the latent space is expensive in terms of processing overhead. To manage this, the system scales the planning horizon dynamically.

### Surprisal-Gated Budgeting (`agents.py`)
I link the compute budget directly to `surprisal_velocity`—the derivative of the prediction error.

* In familiar environments where surprisal is low, the dynamic horizon stays at a minimal depth. The reactive base policy executes quickly.
* If the environment generates a novel architectural anomaly, surprisal spikes. The system computes a new `dynamic_horizon` that scales exponentially based on the learning progress reward. This forces the agent to pause and build a deep MCTS tree to resolve the specific topological anomaly.

In practice, this gives the system a fast path for familiar states and a deeper planning path for uncertain ones.