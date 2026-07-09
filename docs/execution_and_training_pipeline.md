# Execution Pipeline, Memory Streaming, and Fault Tolerance

Long training runs need predictable process management, safe CUDA initialization, and reliable checkpointing. This document describes how `train.py` and `ppo_engine.py` handle those concerns.`

---

## 1. Process Orchestration and Concurrency Model

The system uses multiple processes. The main entrypoint (`train.py`) is responsible for process startup and shutdown behavior.`

### Spawn Contexts and File System Sharing

PyTorch and Warp share the GPU context. To guarantee CUDA state safety across isolated workers, multiprocessing is forced to `spawn`:

    import torch.multiprocessing as mp
    mp.freeze_support()
    mp.set_start_method('spawn', force=True)
    mp.set_sharing_strategy('file_system')

* **`spawn` vs `fork`:** Bypassing the Linux `fork` default prevents child processes from inheriting an initialized, invalid CUDA state. Using `spawn` isolates each worker.`
* **`file_system` sharing:** I set the tensor sharing strategy to use the file system. This helps avoid file descriptor limits when many tensors are shared across processes.

---

## 2. Emergency Checkpointing and LMDB Integrity

Long-horizon runs on preemptible cloud instances, or runs that hit out-of-memory errors during procedural spikes, need a reliable recovery method.

I register a `signal.SIGINT` handler inside the `MainProcess` of `train.py`. If the script receives a termination signal, it skips the usual iteration checks and triggers a shutdown sequence:

1.  **Policy Preservation:** The orchestrator captures the best agent's current weights and writes them asynchronously to `champion_weights.pt`.
2.  **Replay Buffer Commit:** Calls `world.memory_bank.flush()` to sync LMDB memory pages directly to disk.

This guarantees replay integrity upon exit and enables safe resumption.

---

## 3. Fast Data Loading with mmap

The curriculum generator may need to read very large trajectory files. Standard disk I/O usually chokes the pipeline here, so I chosen a faster approach.

`load_deterministic_curriculum` uses zero-copy memory mapping instead of standard file reads.

    with open(file_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            total_size = mm.size()
            while current_offset < total_size:
                # Reads binary chunks directly from virtual memory

Memory mapping delegates paging to the kernel. This completely avoids OOM limits on massive trajectory datasets.

---

## 4. VRAM Quantization via `bitsandbytes`

The optimizer state for a dual-stream JEPA model combined with deep MCTS trees and continuous Actor-Critic networks easily exceeds standard consumer VRAM limits.

`ppo_engine.py` switches optimizer backends based on the available hardware.

* **Consumer GPUs (< 12GB VRAM):** The engine falls back to `bitsandbytes` 8-bit quantized optimizers (`paged_8bit`) to lower the memory footprint.
* **Larger Hardware:** It runs PyTorch's native `AdamW` with `fused=True` to maximize throughput without quantization overhead.

---

## 5. Runtime Configuration

Three configuration objects control runtime behavior:

* **`CFG` (Core Config):** Physical bounds (`MIN_POPULATION`, `MAX_POPULATION`) and metabolic culling logic (`DEACTIVATION_THRESHOLD`).
* **`PLAN_CFG` (MCTS Control):** MCTS constraints, uncertainty scaling, and halting triggers.`
* **`TRAIN_CFG` (Optimization Loop):** Contains PPO settings such as gradient clipping, entropy coefficients, and other optimization limits.`

You pass these configurations directly to the entrypoint via JSON or YAML arguments:

    python train.py --mode default --epochs 100 --deterministic