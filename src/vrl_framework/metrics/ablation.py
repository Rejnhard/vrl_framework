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


def execute_ablation_metrics(context):
    """Runs structural ablation tests (MoE/Mamba/KAN/Memory) and logs to W&B."""
    import csv
    import gc
    import logging
    import sys

    import torch
    import wandb

    import vrl_framework.core.settings
    from vrl_framework.environment import VectorizedPopulation

    logger = logging.getLogger("VRL_Engine")
    logger.info("Starting ablation evaluation matrix...")

    ablation_type = "lesion"
    target_epochs = 100
    for arg in sys.argv:
        if arg.startswith("--ablation-type="):
            ablation_type = arg.split("=")[1]
        elif arg == "--ablation-type" and sys.argv.index(arg) + 1 < len(sys.argv):
            ablation_type = sys.argv[sys.argv.index(arg) + 1]
        elif arg.startswith("--epochs="):
            target_epochs = int(arg.split("=")[1])
        elif arg == "--epochs" and sys.argv.index(arg) + 1 < len(sys.argv):
            target_epochs = int(sys.argv[sys.argv.index(arg) + 1])

    if hasattr(context, "world_ref"):
        if hasattr(context.world_ref, "close"):
            context.world_ref.close()
        del context.world_ref

    try:
        wandb.finish()
    except Exception:
        # Ignore network/teardown failures from wandb to prevent crashing the entire ablation loop.
        pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    presets = ["REACTIVE_ONLY", "PLUS_MEMORY", "FULL_SYSTEM"]
    results = []

    for preset in presets:
        logger.info(f"Evaluating preset: {preset} via {ablation_type.upper()}")

        # Tuple configuration maps to boolean flags: (USE_MOE, USE_MAMBA, USE_KAN, USE_MEMORY)
        preset_configs = {
            "REACTIVE_ONLY": (False, False, False, False),
            "PLUS_MEMORY": (False, True, False, True),
            "FULL_SYSTEM": (True, True, True, True),
        }

        use_moe, use_mamba, use_kan, use_memory = preset_configs[preset]

        vrl_framework.core.settings.CFG.USE_MOE = use_moe
        vrl_framework.core.settings.ENABLE_MOE = use_moe
        vrl_framework.core.settings.CFG.USE_MAMBA = use_mamba
        vrl_framework.core.settings.CFG.USE_KAN = use_kan
        vrl_framework.core.settings.ENABLE_EPISODIC_MEMORY = use_memory

        if ablation_type == "architectural":
            from vrl_framework.trainer.ppo_engine import PPOTrainer

            trainer = PPOTrainer(runtime_context=context)
            trainer.train(target_epochs=target_epochs)
            world = trainer.world if hasattr(trainer, "world") else getattr(trainer, "context").world_ref

            if hasattr(world, "active_mask") and hasattr(world, "fitness"):
                with torch.no_grad():
                    mask_f = world.active_mask.to(world.fitness.dtype)
                    metrics = torch.stack([(world.fitness * mask_f).sum(), mask_f.sum()]).cpu()
                    avg_fitness = (metrics[0] / metrics[1]).item() if metrics[1].item() > 0 else 0.0
            else:
                avg_fitness = 0.0

            if hasattr(world, "close"):
                world.close()
            del trainer
            del world
        else:
            logger.info(f"Allocating environment memory for preset: {preset}")

            world = VectorizedPopulation(
                initial_agents=64, world_dim=vrl_framework.core.settings.WORLD_DIM, max_agents=128
            )
            world.to("cuda" if torch.cuda.is_available() else "cpu")

            total_generations = 3
            random_actions = torch.empty((64,), dtype=torch.long, device=world.positions.device)
            for step_idx in range(total_generations):
                logger.info(f"[ABLATION: {preset}] Step: {step_idx + 1}/{total_generations}...")
                # Baseline: uniform sampling.
                random_actions.random_(0, 16)
                world.step(random_actions)
                world.action_cost_update_batch()

            if hasattr(world, "active_mask") and hasattr(world, "fitness"):
                with torch.no_grad():
                    mask_f = world.active_mask.to(world.fitness.dtype)
                    metrics = torch.stack([(world.fitness * mask_f).sum(), mask_f.sum()]).cpu()
                    avg_fitness = (metrics[0] / metrics[1]).item() if metrics[1].item() > 0 else 0.0
            else:
                avg_fitness = 0.0

            if hasattr(world, "close"):
                world.close()
            del world

        try:
            wandb.finish()
        except Exception:
            # Ignore network/teardown failures from wandb to prevent crashing the entire ablation loop.
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        results.append(
            {
                "preset": preset,
                "vram_usage_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
                "survivability": avg_fitness,
            }
        )

    import os

    from vrl_framework.core.settings import METRICS_DIR

    csv_path = os.path.join(METRICS_DIR, f"ablation_{ablation_type}_matrix.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["preset", "vram_usage_gb", "survivability"])
        writer.writeheader()
        writer.writerows(results)

    try:
        logger.info("Uploading ablation metrics to Weights & Biases...")
        wandb.init(project="VRL-Framework", name=f"ablation_{ablation_type}", job_type="evaluation")
        table = wandb.Table(columns=["Architecture", "VRAM Usage (GB)", "Fitness"])
        for row in results:
            table.add_data(row["preset"], round(row["vram_usage_gb"], 2), round(row["survivability"], 4))
        wandb.log({f"{ablation_type.capitalize()}_Ablation_Matrix": table})
        wandb.finish()
        logger.info("Ablation matrix successfully uploaded to WandB.")
    except Exception as e:
        logger.error(f"Failed to upload table to WandB: {e}")
