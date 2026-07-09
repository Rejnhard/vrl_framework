# Contributing to VRL Framework

First, thank you for your interest in the VRL Framework! 

I built this project as a solo research endeavor to explore emergent behaviors, non-linear latent topologies, and massively parallel embodied AI. It is currently in its prototype/scaffolding phase. As I open-source this foundation, I welcome contributions, ideas, and optimizations from the reinforcement learning, artificial life, and computational geometry communities.

Whether you want to patch a bug in the multiprocessing pipeline, optimize a CUDA kernel, or propose a new representation topology, your input is valuable for me.

## Development Setup

To maintain environmental consistency, I strongly recommend using `conda` for dependency management.

1. **Fork and Clone** the repository.
2. **Initialize the Environment**:

```bash
conda env create -f environment.yml
conda activate vrl_env
```

*Alternatively, using pip:*

```bash
pip install -r requirements.txt
```

## Code Style and Typing

Because my training runs hit 8k+ epochs, silent bugs are a nightmare. Try to stick to these rules:
- Add standard Python type hints for any new functions or methods you write. I utilize `jaxtyping` and `beartype` for runtime tensor shape validation.
- **Formatting**: Please format your code using `black` and sort imports with `isort`.
- **Docstrings**: Use the Google Docstring format. If you introduce new mathematical formulations, please document them using LaTeX syntax within the docstrings.

## Testing Architecture

Given the complexity of asynchronous physics rendering and quantized optimization, robust testing is critical. The test suite is divided into specific domains. Before submitting a pull request, ensure all relevant tests pass:

```bash
# Run the entire suite
pytest tests/

# Run specific domains based on your changes
pytest tests/unit/           # Fast, isolated algorithmic logic tests
pytest tests/integration/    # Memory buffer and IPC socket behavior
pytest tests/e2e/            # Full pipeline smoke tests and convergence
pytest tests/performance/    # Latency and memory leak benchmarks
pytest tests/property/       # Hypothesis-driven fuzzing for geometric invariants
pytest tests/sanity/         # CUDA context and hardware allocation checks
pytest tests/reproducibility/ # Determinism and trajectory hashing
```

*Note: Some performance and reproducibility tests require a dedicated CUDA GPU and will be automatically skipped on CPU-only machines.*

## Pull Request Process

1. Rebase onto the latest `main`.
2. Run tests to make sure nothing is broken.
3. Open a PR using my template.
4. I'll review it, mostly checking for VRAM usage, thread safety, and stable math.