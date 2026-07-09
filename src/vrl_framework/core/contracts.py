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

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple, TypeVar

import torch
from jaxtyping import Bool, Float


@dataclass
class RuntimeContext:
    """Global context registry for runtime dependencies."""

    metrics: Any
    io_worker: Any
    compute_worker: Any
    lmdb_bank: Any
    sim_dir: str


@dataclass
class ComputeTaskContract:
    """Defines memory layout and routing metadata for zero-copy IPC compute requests."""

    task_name: str
    request_id: str
    input_shape: Tuple[int, ...]
    input_dtype: torch.dtype
    expected_out_shape: Tuple[int, ...]
    expected_out_dtype: torch.dtype
    shared_tensor: torch.Tensor
    kwargs: dict


@dataclass
class ActorCriticOutput:
    """Standardized tensor output for policy and multiple value head estimations."""

    policy_logits: torch.Tensor
    pessimistic_value: torch.Tensor
    cost_value: torch.Tensor
    intrinsic_value: torch.Tensor
    value_logits_1: torch.Tensor
    value_logits_2: torch.Tensor
    style_mu: torch.Tensor
    style_logvar: torch.Tensor


@dataclass
class PlannerOutput:
    truth_margin: float
    critic_divergence: float
    dynamics_error: float
    quantization_error: float


class PlannerRegime:
    OBSERVE_ONLY = 0
    ADVICE_ONLY = 1
    TEACHER_AUTHORITY = 2
    DISTILLATION_MODE = 3


@dataclass
class PlannerDecisionTrace:
    """Records MCTS trajectory, gating metrics, and confidence bounds."""

    baseline_logits: torch.Tensor
    planner_logits_preblend: torch.Tensor
    executed_planner_logits: torch.Tensor
    final_action_logits: torch.Tensor
    truth_margin: float
    critic_divergence: float
    control_cost: float
    selected_depth: int
    selected_samples: int
    divergence_from_base: float
    predicted_gain: float
    batch_adv: float
    conf_score: float
    ood_risk: float
    planner_confidence: float
    planner_blend_weight: float
    reject_ratio: float
    survivor_ratio: float
    same_batch_advantage: float
    teacher_admitted: bool
    teacher_confirmed: bool
    halting_budget_used: float
    planner_regime: int
    planner_temperature: float
    baselogits_temperature: float
    prefilter_time_ms: float
    rollout_time_ms: float


@dataclass
class PlanningBudget:
    health_score: float
    health_band: int
    max_depth: int
    num_samples: int
    distill_enabled: bool
    teacher_ttl: int
    allow_actor_lookahead: bool
    allow_teacher_write: bool
    allow_distillation: bool
    max_branch_survivors: int
    min_survivor_floor: int
    max_ood_risk: float
    max_critic_divergence: float
    max_planner_calls_per_env_step: int


@dataclass
class LatentMCTSOutput:
    """Encapsulates final policy distribution after MCTS search and optional decision tracing."""

    final_blended_logits: torch.Tensor
    decision_trace: Optional[PlannerDecisionTrace] = None


@dataclass
class TrainStepMetrics:
    policy_loss: float
    value_loss: float
    intrinsic_loss: float
    causal_loss: float
    planner_regret: float
    planning_gain: float
    conditional_ecq: float
    epistemic_dissonance: float
    latent_rank: float
    quant_recon_error: float
    retrieval_hit_rate: float
    health_score: float
    health_band: int
    decision_compute_ms: float
    sample_efficiency: float
    dynamics_mse: float
    teacher_admission_rate: float
    teacher_confirmation_rate: float
    planner_ood_rate: float
    planner_blend_weight: float
    halting_budget_mean: float
    reject_ratio: float
    survivor_ratio: float
    hyperbolic_contract_failures: int
    semantic_rollback_count: int
    same_batch_planner_advantage: float


class ExperimentPreset:
    REACTIVE_ONLY = "reactive_only"
    PLUS_MEMORY = "plus_memory"
    PLUS_PLANNER = "plus_planner"
    HIERARCHY_NO_PLANNER = "hierarchy_no_planner"
    FULL_NO_INTRINSIC = "full_no_intrinsic"
    FULL_SYSTEM = "full_system"


@dataclass
class MetricsDict:
    sample_efficiency: float
    planning_gain: float
    latent_rank: float
    retrieval_hit_rate: float
    robustness: float
    compute_efficiency: float
    dynamics_mse: float

    @property
    def synthetic_score(self) -> float:
        """Computes scalar fitness score. Dynamics MSE is inverted as bonus."""
        dynamics_fitness = 1.0 / (1.0 + self.dynamics_mse)
        return (
            self.sample_efficiency * 0.25
            + self.planning_gain * 0.25
            + self.latent_rank * 0.1
            + self.retrieval_hit_rate * 0.1
            + self.robustness * 0.1
            + self.compute_efficiency * 0.1
            + dynamics_fitness * 0.1
        )


@dataclass
class ConvergenceStats:
    holdout_dynamics_mse: float
    hyperbolic_failure_rate: float
    critic_divergence_ema: float
    confirmed_planner_gain: float
    teacher_confirmation_rate: float
    semantic_rollback_frequency: float
    ablation_survival_rate: float
    zero_shot_drift_margin: float = 0.0

    @property
    def is_converged(self) -> bool:
        zero_shot_margin_acceptable = self.zero_shot_drift_margin < 0.15

        return (
            self.holdout_dynamics_mse < 0.05
            and self.hyperbolic_failure_rate < 0.001
            and self.critic_divergence_ema < 1.0
            and self.confirmed_planner_gain > 0.01
            and self.teacher_confirmation_rate > 0.5
            and self.semantic_rollback_frequency < 0.01
            and self.ablation_survival_rate > 0.9
            and zero_shot_margin_acceptable
        )


@dataclass
class ReplaySegmentMeta:
    segment_id: int
    record_count: int
    quality_band: int
    true_quant_error: float
    scale: float
    record_dim: int
    td_dim: int
    record_stride_bytes: int
    td_stride_bytes: int
    payload_version: int
    retrieval_hits: int
    utility_score: float
    last_compacted_step: int
    prototype_popcount: int
    format_signature: bytes = b"VRLX"
    checksum: int = 0
    tombstone: bool = False


@dataclass
class RetrievedMemoryBatch:
    episodic_context: torch.Tensor
    alarm_context: torch.Tensor
    procedural_context: torch.Tensor
    episodic_source_segment_ids: torch.Tensor
    episodic_source_record_indices: torch.Tensor
    episodic_source_qualitybands: torch.Tensor
    context_confidence: torch.Tensor
    is_empty: bool = False


@dataclass
class RecurrentStateSnapshot:
    metagru_h: torch.Tensor
    pondergru_h: torch.Tensor
    gradientstm_h: torch.Tensor
    stmptr: torch.Tensor
    stmtensor_k: torch.Tensor


@dataclass
class PolicySequenceBatch:
    states: torch.Tensor
    actions: torch.Tensor
    old_logprobs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    next_states: torch.Tensor
    costs: torch.Tensor
    dones: torch.Tensor
    episode_ids: torch.Tensor
    recurrent_snapshots: RecurrentStateSnapshot
    valid_mask: torch.Tensor
    burnin: int
    learn_length: int
    buildstats: Optional[Any] = None


class PlannerValidator:
    """Validation routines for tensor invariants, sequence boundaries, and configuration bounds."""

    @staticmethod
    def assert_policy_only_integrity(critic_context: Optional[torch.Tensor], mode: str) -> None:
        if mode == "full" and critic_context is None:
            raise ValueError("critic_context cannot be None when mode is 'full'.")

    @staticmethod
    def assert_critic_context_origin(critic_context: Optional[torch.Tensor]) -> None:
        if critic_context is not None:
            if not torch.isfinite(critic_context).all():
                raise ValueError("critic_context tensor contains NaN or Inf values.")
            if torch.all(critic_context == 0):
                raise ValueError("critic_context tensor contains only zeros, indicating manifold collapse.")

    @staticmethod
    def assert_no_nan_inf(tensor: torch.Tensor, name: str) -> None:
        if os.environ.get("RUNTIME_PROFILE", "PROFILE_FAST") == "PROFILE_REPRO":
            if not torch.isfinite(tensor).all():
                raise ValueError(f"NaN/Inf values detected in tensor: {name}")

    @staticmethod
    def assert_sequence_boundaries(dones: torch.Tensor, episode_ids: Optional[torch.Tensor] = None) -> None:
        if dones[:, :-1].any():
            raise ValueError("Terminal states found within the interior of contiguous segments.")
        if episode_ids is not None:
            if not (episode_ids == episode_ids[:, 0:1]).all():
                raise ValueError("Episode ID mismatch within contiguous segment.")

    @staticmethod
    def assert_actorcritic_output_contract(output: ActorCriticOutput) -> None:
        if (
            output.policy_logits is None
            or output.pessimistic_value is None
            or output.value_logits_1 is None
            or output.value_logits_2 is None
            or output.cost_value is None
        ):
            raise AssertionError("Missing tensor in ActorCriticOutput.")

    @staticmethod
    def assert_planner_budget_contract(budget: PlanningBudget) -> None:
        assert budget.max_depth >= 0 and budget.num_samples >= 0
        assert budget.health_band in [0, 1, 2]
        assert budget.teacher_ttl >= 0
        assert isinstance(budget.allow_actor_lookahead, bool)
        assert budget.min_survivor_floor > 0
        assert budget.max_branch_survivors >= budget.min_survivor_floor

    @staticmethod
    def assert_lmdb_stride_contract(meta: ReplaySegmentMeta) -> None:
        assert meta.record_stride_bytes > 0

        # Enforce exact byte-level memory alignment for zero-copy IPC.
        assert meta.record_dim * 4 == meta.record_stride_bytes
        assert meta.td_dim * 2 == meta.td_stride_bytes

    @staticmethod
    def execute_step_validation_pipeline(
        states: torch.Tensor, memory_output: torch.Tensor, planner_budget: PlanningBudget, ac_output: ActorCriticOutput
    ) -> None:
        PlannerValidator.assert_no_nan_inf(states, "Pipeline_States")
        PlannerValidator.assert_no_nan_inf(memory_output, "Pipeline_Memory")
        PlannerValidator.assert_planner_budget_contract(planner_budget)
        PlannerValidator.assert_actorcritic_output_contract(ac_output)


class RuntimeShutdownRequest(Exception):
    pass


def _resolve_typechecker():
    if os.environ.get("RUNTIME_PROFILE", "PROFILE_FAST") == "PROFILE_FAST":

        def dummy_jaxtyped(typechecker=None):
            def decorator(func):
                return func

            return decorator

        def dummy_beartype(func):
            return func

        return dummy_jaxtyped, dummy_beartype
    else:
        from beartype import beartype as true_beartype
        from jaxtyping import jaxtyped as true_jaxtyped

        return true_jaxtyped, true_beartype


if TYPE_CHECKING:
    from beartype import beartype
    from jaxtyping import jaxtyped
else:
    jaxtyped, beartype = _resolve_typechecker()

# Suppress Pyright reportInvalidTypeForm for structural TypeVar definitions.
Batch = TypeVar("Batch")  # pyright: ignore[reportInvalidTypeForm, reportAssignmentType]
Seq = TypeVar("Seq")  # pyright: ignore[reportInvalidTypeForm, reportAssignmentType]
Dim = TypeVar("Dim")  # pyright: ignore[reportInvalidTypeForm, reportAssignmentType]
Heads = TypeVar("Heads")  # pyright: ignore[reportInvalidTypeForm, reportAssignmentType]
Actions = TypeVar("Actions")  # pyright: ignore[reportInvalidTypeForm, reportAssignmentType]
Experts = TypeVar("Experts")  # pyright: ignore[reportInvalidTypeForm, reportAssignmentType]
Slots = TypeVar("Slots")  # pyright: ignore[reportInvalidTypeForm, reportAssignmentType]

LatentState = Float[torch.Tensor, "Batch Dim"]
TemporalState = Float[torch.Tensor, "Batch Seq Dim"]
ActionLogits = Float[torch.Tensor, "Batch Actions"]
SequentialActionLogits = Float[torch.Tensor, "Batch Seq Actions"]
RewardSignal = Float[torch.Tensor, "Batch 1"]
ValueEstimation = Float[torch.Tensor, "Batch 1"]
CostEstimation = Float[torch.Tensor, "Batch 1"]
AttentionWeights = Float[torch.Tensor, "Batch Heads Seq Seq"]
MaskTensor = Bool[torch.Tensor, "Batch"]
ExpertRouting = Float[torch.Tensor, "Batch Experts"]
SlotState = Float[torch.Tensor, "Batch Slots Dim"]
