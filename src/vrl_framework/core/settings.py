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

"""Global configuration, hardware allocation policies, and environment setup.

Manages deterministic execution constraints, VRAM optimizations, and distributed
process initialization for the reinforcement learning framework.
"""

import datetime
import importlib.metadata
import logging
import multiprocessing as mp
import os
import platform
import random
import sys
from dataclasses import dataclass
from typing import Literal, cast

import matplotlib
import numpy as np
import torch
import torch._dynamo
import torch.backends.cudnn
import wandb
from torch.utils.checkpoint import checkpoint

matplotlib.use("Agg")

PROJECT_ROOT: str = ""
OUTPUTS_BASE: str = ""
SIM_DIR: str = ""
AGENTS_DIR: str = ""
METRICS_DIR: str = ""
LOGS_DIR: str = ""
BASE_DIR: str = ""


def setup_env_paths() -> None:
    if "VRL_RUN_ID" not in os.environ:
        if not any("multiprocessing.spawn" in arg for arg in sys.argv) and not os.environ.get("LOCAL_RANK"):
            os.environ["VRL_RUN_ID"] = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
        else:
            os.environ["VRL_RUN_ID"] = "run_distributed_sync_pending"

    global PROJECT_ROOT, OUTPUTS_BASE, SIM_DIR, AGENTS_DIR, METRICS_DIR, LOGS_DIR, BASE_DIR

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    OUTPUTS_BASE = os.path.join(PROJECT_ROOT, "outputs", "runs")

    SIM_DIR = os.path.join(OUTPUTS_BASE, os.environ["VRL_RUN_ID"])
    AGENTS_DIR = os.path.join(SIM_DIR, "agents")
    METRICS_DIR = os.path.join(SIM_DIR, "metrics")
    LOGS_DIR = os.path.join(SIM_DIR, "logs")

    os.makedirs(AGENTS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "max_split_size_mb:128,garbage_collection_threshold:0.8,roundup_power2_divisions:8"
    )
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["BITSANDBYTES_NOWELCOME"] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))


setup_env_paths()

logger = logging.getLogger("vrl_framework")


def validate_environment() -> None:
    """Verifies the presence of required runtime dependencies.

    Raises a RuntimeError if critical dependencies are missing to prevent
    OS-level deadlocks caused by dynamic package resolution during multiprocessing.
    """
    required_packages = [
        "torch",
        "warp-lang",
        "bitsandbytes",
        "lmdb",
        "einops",
        "jaxtyping",
        "beartype",
        "networkx",
        "matplotlib",
        "safetensors",
        "scipy",
        "decord",
        "wandb",
        "pyzmq",
    ]

    missing_packages = []
    for pkg in required_packages:
        try:
            importlib.metadata.distribution(pkg)
        except importlib.metadata.PackageNotFoundError:
            missing_packages.append(pkg)

    if missing_packages:
        missing_str = " ".join(missing_packages)
        logging.warning(f"Missing dependencies detected: {missing_str}. Attempting dynamic resolution...")

        # Unpinned package resolution via subprocess triggers system deadlocks on Windows during multiprocessing
        raise RuntimeError(
            f"Critical dependencies missing: {missing_str}. "
            "Dynamic installation via subprocess on Windows can lock the system. "
            f"Please exit and run manually: pip install {missing_str}"
        )


def set_seed(seed_value: int = 42) -> None:
    """Sets global random seeds for Python, NumPy, and PyTorch environments."""
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def pre_import_env():
    matplotlib.use("Agg")


SYMLOG_MAX = 20.0
K_CHUNK = 2
ENTROPY_LOSS = 0.01
WATCHDOG_EMA_DECAY = 0.99


class RuntimePolicy:
    """Configures thread affinity, CUDNN benchmarking, and OS-level process priorities."""

    @staticmethod
    def enforce_determinism(seed: int = 42) -> None:
        if torch.cuda.is_available():
            torch.cuda.init()
            _ = torch.tensor([0.0], device="cuda")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        set_seed(seed_value=seed)

    @staticmethod
    def apply_environment_policy(
        profile: str = "PROFILE_FAST", cluster_mode: str = "0", deterministic: bool = False, global_seed: int = 42
    ) -> str:
        import psutil

        os.environ["BITSANDBYTES_NOWELCOME"] = "1"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["QT_SCALE_FACTOR"] = "1"
        cpu_count = os.cpu_count()
        optimal_threads = max(1, cpu_count - 2) if cpu_count is not None else 6
        os.environ["OMP_NUM_THREADS"] = str(optimal_threads)
        os.environ["MKL_NUM_THREADS"] = str(optimal_threads)
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "1"
        os.environ["RUNTIME_PROFILE"] = profile

        if deterministic:
            RuntimePolicy.enforce_determinism(seed=global_seed)

        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError as e:
            logging.warning(f"Process start method already set. Unsafe IPC memory state imminent if not spawned: {e}")

        if cluster_mode == "1":
            import torch.distributed as dist

            if not dist.is_initialized():
                backend = (
                    "gloo" if platform.system() == "Windows" else ("nccl" if torch.cuda.is_available() else "gloo")
                )
                dist.init_process_group(backend=backend)
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                torch.cuda.set_device(local_rank)

        torch.set_num_threads(optimal_threads)
        torch.set_num_interop_threads(min(4, optimal_threads // 2))

        if platform.system() == "Windows":
            try:
                p = psutil.Process(os.getpid())
                p.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
            except (ImportError, psutil.AccessDenied, psutil.NoSuchProcess) as e:
                logging.warning(
                    f"Process priority elevation failed: {e}. "
                    "Run as Administrator to grant High Priority I/O to Warp threads."
                )

        if torch.cuda.is_available():
            if profile == "PROFILE_FAST":
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.cufft_plan_cache.max_size = 16
            elif profile == "PROFILE_REPRO":
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

        return SIM_DIR


@dataclass
class TrainingConfig:
    ppo_clip: float = 1.0
    dreamer_loss: float = 0.5
    value_loss: float = 0.5
    intrinsic_loss: float = 0.5
    entropy_loss: float = 0.01
    max_entropy_bonus: float = 0.05
    masker_loss: float = 1.0
    lpm_loss: float = 1.0
    moe_aux_loss: float = 0.1
    ponder_loss: float = 0.1
    distillation_loss: float = 0.5
    symlog_max: float = 20.0
    num_value_bins: int = 255
    k_chunk: int = 2
    plateau_patience: int = 5
    teacher_threshold: float = 0.85
    exploration_boost: float = 1.2
    health_band_green: float = 1.0
    health_band_red: float = 2.0


@dataclass
class PlannerConfig:
    max_depth: int = 3
    num_samples: int = 16
    lambda_temperature: float = 0.5
    noise_sigma: float = 1.0
    deliberation_steps: int = 50
    imagination_horizon: int = 8
    retry_policy_max_attempts: int = 3
    deep_planning_health_threshold: float = 1.5


@dataclass
class MemoryConfig:
    hot_capacity: int = 16384
    ring_buffer_capacity: int = 50000
    segment_size: int = 4096
    max_map_size_gb: int = 2
    expansion_size_gb: int = 2
    quantization_v_bound: float = 15.0
    true_quant_error_threshold: float = 2.0
    archive_diverse_ratio: float = 0.25
    retrieval_top_k_limit: int = 16
    prefetch_queue_limit: int = 64
    consolidation_chunk_size: int = 512


@dataclass
class MoEConfig:
    num_experts: int = 8
    capacity_factor: float = 1.25
    sparsity: float = 0.9
    bias_update_rate: float = 0.005
    drop_fraction: float = 0.1


TRAIN_CFG = TrainingConfig()
PLAN_CFG = PlannerConfig()
MEM_CFG = MemoryConfig()
MOE_CFG = MoEConfig()

ENABLE_MOE = True
ENABLE_EPISODIC_MEMORY = True
USE_3D_VOXELS = False
ENABLE_OFFLINE_ROLLOUT = False


@dataclass
class HardwareConfig:
    """Central registry for hyper-parameters, hardware allocations, and architectural toggles."""

    INIT_FROM_SCRATCH: bool = True
    STRICT_EX_NIHILO: bool = False
    USE_TORCH_COMPILE: bool = False
    ENABLE_ACTION_CHUNKING: bool = False
    ENABLE_MULTIMODAL: bool = False
    AGENT_MODE: str = "core"  # 'core' or 'memory_augmented'

    USE_KAN: bool = True
    USE_MAMBA: bool = True
    USE_MOE: bool = True

    INIT_POPULATION: int = 32
    MUTATION_RATE: float = 0.05
    LEARNING_RATE: float = 3e-4
    WORLD_DIM: tuple = (16, 16, 16, 17)
    MODEL_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    COMPUTE_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_COMPUTE_PER_AGENT: int = 1000
    MAX_POPULATION: int = 32
    MIN_POPULATION: int = 16
    BASELINE_ENERGY_COST: float = 0.01
    ENV_EVENT_RATE: float = 0.01
    DESIRED_ENERGY: float = 2.0
    LTM_INDEX_PATH: str = os.path.join(".", "ltm_index.bin")
    BATCH_SIZE: int = 2
    MIXED_PRECISION: bool = True
    AMP_DTYPE: torch.dtype = torch.float16
    AUTO_CURRICULUM: bool = True
    IDLE_EPOCH: int = 50000
    WARMUP_STEPS: int = 10000
    TEXT_NOISE_ALPHA: float = 0.1
    LOOKAHEAD_STRATEGY: str = "uniform"  # Options: uniform, greedy, sample

    GRADIENT_ACCUMULATION_STEPS: int = 1
    OPTIMIZER_BACKEND: str = "32bit"

    def __post_init__(self):
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            try:
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                vram_gb = torch.cuda.get_device_properties(local_rank).total_memory / (1024**3)
                if vram_gb <= 8.5:
                    self.BATCH_SIZE = 2
                    self.INIT_POPULATION = 16
                    self.MAX_POPULATION = 16
                    self.MIN_POPULATION = 8
                    self.GRADIENT_ACCUMULATION_STEPS = 32
                    self.OPTIMIZER_BACKEND = "32bit"
                elif vram_gb < 12.0:
                    self.BATCH_SIZE = 4
                    self.GRADIENT_ACCUMULATION_STEPS = 16
                    self.OPTIMIZER_BACKEND = "paged_8bit"
                elif vram_gb < 30.0:
                    self.BATCH_SIZE = 8
                    self.GRADIENT_ACCUMULATION_STEPS = 8
                    self.OPTIMIZER_BACKEND = "8bit"
                else:
                    self.BATCH_SIZE = 16
                    self.GRADIENT_ACCUMULATION_STEPS = 4
                    self.OPTIMIZER_BACKEND = "32bit"

                if os.environ.get("CLUSTER_MODE", "0") == "1":
                    self.GRADIENT_ACCUMULATION_STEPS = 4
                    self.OPTIMIZER_BACKEND = "8bit"

            except RuntimeError as e:
                logging.warning(f"VRAM property query failed. Defaulting to safe allocation limits. Error: {e}")

    LOSS_PPO_CLIP: float = 1.0
    LATENT_ROLLOUT_LOSS: float = 0.5
    LOSS_VALUE: float = 0.5
    LOSS_INTRINSIC: float = 0.5
    LOSS_ENTROPY: float = 0.01
    LOSS_MAX_ENTROPY_BONUS: float = 0.05
    LOSS_MASKER: float = 1.0
    LOSS_LPM: float = 1.0
    LOSS_MOE_AUX: float = 0.1
    LOSS_PONDER: float = 0.1
    LOSS_DISTILLATION: float = 0.5

    IMAGINATION_HORIZON: int = 4
    MAX_Z_SCORE: float = 2.0
    pred_error_ema_DECAY: float = 0.99
    WATCHDOG_EMA_DECAY: float = 0.99
    ANOMALY_MULTIPLIER: float = 4.0


CFG = HardwareConfig()
MODEL_DEVICE = CFG.MODEL_DEVICE
WORLD_DIM = CFG.WORLD_DIM
MUTATION_RATE = CFG.MUTATION_RATE
INIT_POPULATION = CFG.INIT_POPULATION
LEARNING_RATE = CFG.LEARNING_RATE
BATCH_SIZE = CFG.BATCH_SIZE
MIXED_PRECISION = CFG.MIXED_PRECISION
MAX_POPULATION = CFG.MAX_POPULATION
MIN_POPULATION = CFG.MIN_POPULATION
ENV_EVENT_RATE = CFG.ENV_EVENT_RATE
DESIRED_ENERGY = CFG.DESIRED_ENERGY
DEACTIVATION_THRESHOLD = 0.1
RECOVERY_MARGIN = 0.5
BASELINE_ENERGY_COST = CFG.BASELINE_ENERGY_COST
LTM_INDEX_PATH = CFG.LTM_INDEX_PATH
STM_BUFFER_SIZE = 50000
PENALTY_GAMMA = 0.05
EXPLORATION_NOISE = 0.05
GLOBAL_SCALING_FACTOR = torch.tensor(1.0, device="cpu")

pre_import_env()


def setup_backend() -> None:
    if "WANDB_API_KEY" in os.environ and mp.current_process().name == "MainProcess":
        wandb_mode = cast(
            Literal["online", "offline", "shared", "disabled", "dryrun", "run"], os.environ.get("WANDB_MODE", "online")
        )
        wandb.setup(wandb.Settings(mode=wandb_mode, x_disable_stats=False))
        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)

    torch.set_float32_matmul_precision("high")

    if MODEL_DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        os.environ["USE_GRADIENT_CHECKPOINTING"] = "1"

        if platform.system() == "Windows":
            torch._dynamo.config.disable = True
    elif MODEL_DEVICE == "cpu":
        try:
            torch.backends.quantized.engine = "qnnpack" if platform.system() != "Windows" else "fbgemm"
        except RuntimeError as e:
            if "already" not in str(e).lower():
                raise


setup_backend()

_USE_CHECKPOINTING = os.environ.get("USE_GRADIENT_CHECKPOINTING") == "1"


def execute_with_checkpointing(module: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
    if _USE_CHECKPOINTING and module.training:
        return checkpoint(module, *args, use_reentrant=False, **kwargs)
    return module(*args, **kwargs)
