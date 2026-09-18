from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .atomic_io import atomic_write_json
from .config import config_hash

CHECKPOINT_SCHEMA_VERSION = 1


def save_checkpoint(
    path: str | Path,
    *,
    config: Mapping[str, Any],
    progress: Mapping[str, Any],
    rng: np.random.Generator,
) -> None:
    """Save progress and NumPy RNG state in an atomic checkpoint."""
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "config_sha256": config_hash(config),
        "progress": dict(progress),
        "rng_state": rng.bit_generator.state,
    }
    atomic_write_json(path, payload)


def load_checkpoint(
    path: str | Path,
    *,
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Load a checkpoint after validating its schema and configuration."""
    checkpoint_path = Path(path)

    with checkpoint_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise TypeError("Checkpoint must contain a JSON object.")

    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint schema version.")

    expected_hash = config_hash(config)
    stored_hash = payload.get("config_sha256")

    if stored_hash != expected_hash:
        raise ValueError(
            "Checkpoint configuration mismatch; refusing to resume."
        )

    progress = payload.get("progress")
    if not isinstance(progress, dict):
        raise TypeError("Checkpoint progress must be a JSON object.")

    rng_state = payload.get("rng_state")
    if not isinstance(rng_state, dict):
        raise TypeError("Checkpoint RNG state must be a JSON object.")

    saved_bit_generator = rng_state.get("bit_generator")
    current_bit_generator = rng.bit_generator.__class__.__name__

    if saved_bit_generator != current_bit_generator:
        raise ValueError(
            "Checkpoint RNG type does not match the supplied generator."
        )

    rng.bit_generator.state = rng_state

    return progress