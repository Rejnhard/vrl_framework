#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.1.0",
#     "warp-lang",
#     "bitsandbytes",
#     "lmdb",
#     "einops",
#     "jaxtyping",
#     "beartype",
#     "networkx",
#     "matplotlib",
#     "safetensors",
#     "scipy",
#     "decord",
#     "wandb",
#     "psutil",
#     "setuptools"
# ]
# ///

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

"""Main entry point for distributed PPO training and evaluation."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import logging
import argparse
import torch.multiprocessing as mp
import signal

if mp.current_process().name != "MainProcess":
    signal.signal(signal.SIGINT, signal.SIG_IGN)

import warnings

# Suppress cuBLAS init warnings on first backward pass.
warnings.filterwarnings("ignore", message=".*Attempting to run cuBLAS, but there was no current CUDA context.*")
import torch

if mp.current_process().name == "MainProcess":
    if torch.cuda.is_available():
        # Require deterministic cuBLAS algorithms.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

from vrl_framework.core import settings as core_settings
from vrl_framework.core.settings import validate_environment, BASE_DIR

from vrl_framework.system.telemetry import ExperimentRunner, RuntimeState
from vrl_framework.system.ipc_core import MockSyncWorker, DiskIOWorkerQueue, ComputeWorkerQueue
from vrl_framework.environment.replay_buffer import GlobalExperienceReplay
from vrl_framework.core.contracts import RuntimeContext
from vrl_framework.metrics.ablation import execute_ablation_metrics
from vrl_framework.math_ops.geometry import run_math_diagnostics


def parse_arguments() -> argparse.Namespace:
    """Parses execution CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Vectorized Reinforcement Learning Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", type=str, choices=["default", "research", "production", "omni"], default="default")
    parser.add_argument("--ablation-type", type=str, choices=["architectural", "lesion"], default="lesion")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--run-diagnostics", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--enable", type=str, default="", help="Comma-separated list of modules to force ENABLE (e.g. MoE,Mamba)"
    )
    parser.add_argument(
        "--disable", type=str, default="", help="Comma-separated list of modules to force DISABLE (e.g. KAN,Memory)"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint (.pt) to resume training or run evaluation."
    )
    parser.add_argument(
        "--new_metrics", action="store_true", help="Force a new W&B run instead of resuming embedded states"
    )
    parser.add_argument(
        "--eval", action="store_true", help="Launch in pure evaluation/inference mode (no backprop, requires --resume)"
    )
    parser.add_argument("--query", type=str, default=None, help="Prompt query for zero-shot evaluation.")
    from vrl_framework.core.settings import BASE_DIR

    parser.add_argument("--curriculum-dir", type=str, default=os.path.join(BASE_DIR, "data", "curriculum"))
    parser.add_argument(
        "--curriculum-start", type=str, default=None, help="Substring of folder/file to skip forward to natively."
    )
    parser.add_argument(
        "--offload-dir", type=str, default="./nvme_cache", help="Path to external memory disk offload."
    )
    parser.add_argument(
        "--use-last-paths",
        action="store_true",
        help="Use strict execution paths defined within the checkpoint state dict.",
    )
    parser.add_argument("--ignore-lora", action="store_true")
    parser.add_argument("--ignore-lmdb", action="store_true")
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--lmdb-path", type=str, default=None)
    return parser.parse_args()


def setup_experiment(args: argparse.Namespace) -> RuntimeContext:
    """Initializes runtime context, including IPC queues, W&B, and global LMDB replay."""
    deterministic_mode = args.deterministic

    validate_environment()
    from vrl_framework.core.settings import LOGS_DIR, SIM_DIR

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(processName)s: %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(LOGS_DIR, "training_runtime.log"), encoding="utf-8"),
        ],
    )

    if deterministic_mode:
        core_settings.RuntimePolicy.enforce_determinism(seed=42)

    from vrl_framework.system.telemetry import WandBMetrics

    metrics_adapter = WandBMetrics()

    io_worker = DiskIOWorkerQueue()
    compute_worker = ComputeWorkerQueue()

    try:
        import lmdb

        lmdb_path = os.path.join(SIM_DIR, "global_replay.lmdb")
        lmdb_bank = GlobalExperienceReplay(lmdb_path)
        logging.info(f"LMDB initialized successfully at: {lmdb_path}")
    except Exception as e:
        import traceback

        logging.critical(f"LMDB FATAL ERROR: {e}")
        logging.critical(traceback.format_exc())
        lmdb_bank = None

    return RuntimeContext(metrics_adapter, io_worker, compute_worker, lmdb_bank, SIM_DIR)


def main():
    args = parse_arguments()

    profiles = {
        "default": {
            "MoE": True,
            "Mamba": True,
            "KAN": True,
            "Memory": True,
            "profile": "PROFILE_FAST",
            "log_level": logging.INFO,
        },
        "research": {
            "MoE": False,
            "Mamba": True,
            "KAN": False,
            "Memory": False,
            "profile": "PROFILE_REPRO",
            "log_level": logging.DEBUG,
        },
        "production": {
            "MoE": True,
            "Mamba": True,
            "KAN": True,
            "Memory": False,
            "profile": "PROFILE_FAST",
            "log_level": logging.WARNING,
        },
        "omni": {
            "MoE": True,
            "Mamba": True,
            "KAN": True,
            "Memory": True,
            "profile": "PROFILE_FAST",
            "log_level": logging.INFO,
        },
    }

    active_config = dict(profiles.get(args.mode, profiles["default"]))

    to_enable = [x.strip() for x in args.enable.split(",")] if args.enable else []
    to_disable = [x.strip() for x in args.disable.split(",")] if args.disable else []

    for feature in to_enable:
        if feature in active_config:
            active_config[feature] = True

    for feature in to_disable:
        if feature in active_config:
            active_config[feature] = False

    core_settings.CFG.USE_MOE = active_config["MoE"]
    core_settings.ENABLE_MOE = active_config["MoE"]
    core_settings.CFG.USE_MAMBA = active_config["Mamba"]
    core_settings.CFG.USE_KAN = active_config["KAN"]
    core_settings.ENABLE_EPISODIC_MEMORY = active_config["Memory"]

    core_settings.CFG.IGNORE_LORA = getattr(args, "ignore_lora", False)
    core_settings.CFG.IGNORE_LMDB = getattr(args, "ignore_lmdb", False)
    core_settings.CFG.LORA_PATH = getattr(args, "lora_path", None)
    core_settings.CFG.LMDB_PATH = getattr(args, "lmdb_path", None)

    core_settings.RuntimePolicy.apply_environment_policy(
        profile=active_config["profile"], deterministic=args.deterministic
    )

    if args.run_diagnostics:
        run_math_diagnostics()

    if args.resume is not None and not args.resume.endswith(".pt"):
        import os

        project_root = os.path.abspath(os.path.dirname(__file__))
        agents_dir = os.path.join(project_root, "outputs", "runs", args.resume, "agents")
        if os.path.exists(agents_dir):
            checkpoints = [f for f in os.listdir(agents_dir) if f.startswith("checkpoint_gen_") and f.endswith(".pt")]
            if checkpoints:
                latest_ckpt = max(checkpoints, key=lambda f: os.path.getmtime(os.path.join(agents_dir, f)))
                args.resume = os.path.join(agents_dir, latest_ckpt)
                logging.getLogger("vrl_framework").info(
                    f"Auto-resolved run directory to checkpoint path: {args.resume}"
                )
            else:
                logging.getLogger("vrl_framework").critical(f"Fatal error: No .pt files found in {agents_dir}")
                sys.exit(1)
        else:
            logging.getLogger("vrl_framework").critical(f"Fatal error: Run directory not found: {agents_dir}")
            sys.exit(1)

    experiment_runner = ExperimentRunner()

    try:
        context = setup_experiment(args)
        experiment_runner.attach_context(context)
        experiment_runner.start_runtime()

        logger = logging.getLogger("vrl_framework")
        logger.info(f"Framework initialized. Execution mode: {args.mode}")

        from vrl_framework.trainer.ppo_engine import PPOTrainer, execute_query

        # Zero-shot evaluation path.
        if args.query is not None:
            if not args.resume:
                logger.critical("Fatal: Query mode requires --resume to load model weights.")
                sys.exit(1)

            core_settings.CFG.RESUME_CHECKPOINT = args.resume
            core_settings.CFG.CURRICULUM_DIR = args.curriculum_dir
            core_settings.CFG.CURRICULUM_START = args.curriculum_start
            core_settings.CFG.OFFLOAD_DIR = args.offload_dir
            core_settings.CFG.USE_LAST_PATHS = args.use_last_paths

            logger.info(f"Loading checkpoint from: {args.resume}")

            trainer = PPOTrainer(runtime_context=context)
            trainer._initialize_optimizer_state()
            trainer.agent_core.trainer = trainer
            trainer.load_checkpoint(args.resume)

            logger.info(f"Executing query: '{args.query}'")
            try:
                response_payload = execute_query(args.query, min_time=5, max_time=30, live_agent=trainer.agent_core)

                output_str = f"\n{'='*60}\n[Query Output]\n{response_payload}\n{'='*60}"
                logger.info(output_str)
                print(output_str)

            except Exception as e:
                import traceback

                logger.critical(f"Fatal error during query execution:\n{traceback.format_exc()}")

            return

        logger.info("Initializing PPO trainer and environment buffer...")
        if __name__ == "__main__":
            core_settings.CFG.RESUME_CHECKPOINT = args.resume
            is_eval_mode = args.eval

            if is_eval_mode and not core_settings.CFG.RESUME_CHECKPOINT:
                logger.critical("Eval mode requires a valid --resume checkpoint path.")
                sys.exit(1)

            trainer = PPOTrainer(runtime_context=context)

            if core_settings.CFG.RESUME_CHECKPOINT:
                mode_str = "EVALUATION INFERENCE" if is_eval_mode else "TRAINING"
                logger.info(
                    f"Preparing to resume simulation in {mode_str} mode from: {core_settings.CFG.RESUME_CHECKPOINT}"
                )
            else:
                logger.info(f"Starting simulation epochs from scratch ({args.epochs} generations)...")

            if is_eval_mode:
                trainer.run_inference_loop(target_epochs=args.epochs)
            else:
                import concurrent.futures
                import threading
                import time

                def sleep_consolidation_worker(trainer_ref):
                    """Periodically resets mature MoE expert counters to trigger reallocation."""
                    while True:
                        if hasattr(trainer_ref, "world") and getattr(trainer_ref.world, "_stop", False):
                            break
                        time.sleep(600)
                        if hasattr(trainer_ref.agent_core, "moe"):
                            try:
                                counters = getattr(trainer_ref.agent_core.moe, "expert_stability_counter", [])
                                if isinstance(counters, torch.Tensor):
                                    with torch.no_grad():
                                        mature_mask = counters > 1000
                                        if mature_mask.sum().item() >= 4:
                                            logger.info(
                                                "[EXPERT CONSOLIDATION] Distilling weights of mature experts..."
                                            )
                                            mature_indices = torch.nonzero(mature_mask, as_tuple=True)[0]
                                            counters[mature_indices[:4]] = 0
                                else:
                                    mature_experts = [i for i, c in enumerate(counters) if c > 1000]
                                    if len(mature_experts) >= 4:
                                        logger.info("[EXPERT CONSOLIDATION] Distilling weights of mature experts...")
                                        for idx in mature_experts[:4]:
                                            counters[idx] = 0
                            except Exception:
                                import logging

                                logging.getLogger(__name__).exception("Sleep consolidation worker failed")

                distillation_thread = threading.Thread(target=sleep_consolidation_worker, args=(trainer,), daemon=True)
                distillation_thread.start()

                trainer.train(target_epochs=args.epochs)

        if args.mode == "research":
            if hasattr(trainer, "world") and hasattr(trainer.world, "close"):
                trainer.world.close()
            del trainer
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if args.ablation_type in ["architectural", "lesion"]:
                logger.info(f"Executing {args.ablation_type} evaluation workflow...")
                execute_ablation_metrics(context)

    except KeyboardInterrupt:
        logger = logging.getLogger("vrl_framework")
        logger.warning("\n[!] Manual interruption detected (Ctrl+C). Initiating emergency save & shutdown...")
        import signal

        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            import gc
            import os

            emergency_saved = False
            for obj in gc.get_objects():
                if type(obj).__name__ == "PPOTrainer" and hasattr(obj, "save_checkpoint"):
                    step = getattr(obj, "global_train_step", 0)
                    gen = getattr(obj.world, "generation", step) if hasattr(obj, "world") else step

                    from vrl_framework.core.settings import AGENTS_DIR, MODEL_DEVICE

                    if hasattr(obj, "world") and hasattr(obj.world, "run_benchmarks"):
                        logger.info("Triggering emergency full simulation benchmark dump...")
                        try:
                            obj.world.run_benchmarks()
                        except Exception as bench_e:
                            logger.error(f"Failed to dump simulation benchmarks: {bench_e}")

                    checkpoint_path = os.path.join(AGENTS_DIR, f"emergency_checkpoint_gen_{gen}_step_{step}.pt")
                    obj.save_checkpoint(checkpoint_path)
                    logger.info(f"Emergency policy checkpoint saved to: {checkpoint_path}")

                    if hasattr(obj, "agent_core"):
                        try:
                            with torch.no_grad():
                                if hasattr(obj.agent_core, "lora_registry"):
                                    lora_path = os.path.join(AGENTS_DIR, f"emergency_lora_skill_gen_{gen}.pt")
                                    lora_data = {k.cpu(): v.cpu() for k, v in obj.agent_core.lora_registry.items()}
                                    torch.save(lora_data, lora_path)
                                    logger.info(f"Emergency O-LoRA Skill Snapshot extracted and saved to: {lora_path}")
                        except Exception as lora_e:
                            logger.error(f"Failed to extract emergency LoRA skills: {lora_e}")

                    if hasattr(obj, "world") and hasattr(obj.world, "memory_bank"):
                        # Flush LMDB to avoid corruption on crash.
                        obj.world.memory_bank.flush()
                        logger.info("LMDB memory flushed.")
                    emergency_saved = True
                    break

            if not emergency_saved:
                logger.warning("Could not locate PPOTrainer in memory to force save.")
        except Exception as shutdown_e:
            logger.error(f"Failed to gracefully shutdown: {shutdown_e}")
    except Exception as e:
        import traceback

        logger = logging.getLogger("vrl_framework")
        logger.critical(f"Fatal initialization fault: {e}")
        logger.critical(traceback.format_exc())
    finally:
        experiment_runner.shutdown_runtime()


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    try:
        import sys

        if sys.platform != "win32":
            mp.set_sharing_strategy("file_system")
    except RuntimeError:
        pass
    main()
