# Scientific Computing Code Sample

A small, self-contained Python project demonstrating reliability and
reproducibility patterns for scientific computation.

The example intentionally uses a simple Monte Carlo calculation so that the
software behavior can be inspected independently of any domain-specific
research method. This repository contains no project-specific data, models,
algorithms, or computing infrastructure.

## What this repository demonstrates

- **Atomic output writing**  
  Results are written to temporary files, flushed to disk, validated where
  appropriate, and moved into place with `os.replace`.

- **Configuration identity**  
  Computational settings are serialized deterministically and identified with
  a SHA-256 hash, helping detect accidental reuse of artifacts from a different
  configuration.

- **Validated checkpointing**  
  Checkpoints record computational progress, the configuration hash, and the
  NumPy random-number-generator state.

- **Reproducible resume**  
  An interrupted calculation can restore both its accumulated state and RNG
  position before continuing.

- **End-to-end tests**  
  The test suite verifies that an interrupted-and-resumed calculation produces
  the same final result as an uninterrupted calculation under the same
  configuration.

## Repository structure

```text
scientific-computing-code-sample/
├── examples/
│   └── resumable_monte_carlo.py
├── src/
│   └── scicomp_sample/
│       ├── __init__.py
│       ├── atomic_io.py
│       ├── checkpoint.py
│       ├── config.py
│       └── runner.py
├── tests/
│   ├── test_atomic_io.py
│   ├── test_checkpoint.py
│   ├── test_config.py
│   ├── test_package.py
│   └── test_runner.py
└── pyproject.toml
```

## Quick start

Python 3.10 or newer is required.

Create a virtual environment:

```bash
python -m venv .venv
```

Install the package and development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the checks:

```bash
python -m ruff check .
python -m pytest -v
```

## Resumable example

The example estimates pi using uniformly sampled points in the unit square.

First, intentionally stop after three batches:

```bash
python examples/resumable_monte_carlo.py --samples 100000 --batch-size 10000 --seed 2026 --max-batches 3
```

This leaves a checkpoint but no final result.

Then run the same configuration again without `--max-batches`:

```bash
python examples/resumable_monte_carlo.py --samples 100000 --batch-size 10000 --seed 2026
```

The calculation restores its checkpoint, resumes from the saved RNG state, and
writes the final result.

## Design notes

### Fail loudly on incompatible state

A checkpoint includes the hash of the configuration that produced it.
Attempting to resume it under a different configuration raises an error rather
than silently mixing incompatible computational states.

### Keep intermediate artifacts from looking final

The completed result is written only after all requested samples have been
processed. An intentionally interrupted run retains a checkpoint instead.

### Avoid pickle-dependent numerical artifacts

NumPy result files reject object-dtype arrays and are loaded with
`allow_pickle=False`.

## Scope

This is a deliberately small code sample rather than a general workflow engine.

The checkpoint/resume guarantee demonstrated here applies to the state explicitly
captured by this program and to a compatible NumPy random-number generator. It
should not be interpreted as a guarantee of bit-for-bit reproducibility across
arbitrary software versions, hardware, or distributed systems.

Similarly, the configuration hash is used for identity and accidental-mismatch
detection; it is not intended as a security mechanism.