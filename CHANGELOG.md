# Changelog

I will track all major updates and changes for this repo here.

Format based on Keep a Changelog + strict SemVer.

## [0.1.0-alpha] - Initial public sync

First open-source release of VRL Framework.

This is an experimental research codebase, not a production-ready system. It contains the core components for multi-agent RL experiments.

### Added
- `bitsandbytes` 8-bit quantized optimizers with PyTorch `fused=True`.
- Lorentz Manifold projections (`math_ops/geometry.py`).
- NVIDIA Warp integration for GPU-native 3D voxel physics (`world_dynamics.py`).
- Dynamic Replay Buffers and memory tiering (VRAM -> RAM -> SSD).
- Thread management to prevent CUDA context corruption during MCTS rollouts.
- Tests for fuzzing, latency, and gradient determinism.