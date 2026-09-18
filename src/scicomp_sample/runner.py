from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .atomic_io import atomic_write_json
from .checkpoint import load_checkpoint, save_checkpoint
from .config import config_hash


def _require_integer(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)

    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{key!r} must be an integer.")

    return value


def _validate_config(config: Mapping[str, Any]) -> tuple[int, int, int]:
    seed = _require_integer(config, "seed")
    num_samples = _require_integer(config, "num_samples")
    batch_size = _require_integer(config, "batch_size")

    if num_samples <= 0:
        raise ValueError("'num_samples' must be positive.")

    if batch_size <= 0:
        raise ValueError("'batch_size' must be positive.")

    return seed, num_samples, batch_size


def _validate_progress(
    progress: Mapping[str, Any],
    *,
    num_samples: int,
) -> tuple[int, int]:
    processed = progress.get("processed")
    inside_circle = progress.get("inside_circle")

    if not isinstance(processed, int) or isinstance(processed, bool):
        raise TypeError("Checkpoint field 'processed' must be an integer.")

    if not isinstance(inside_circle, int) or isinstance(inside_circle, bool):
        raise TypeError("Checkpoint field 'inside_circle' must be an integer.")

    if not 0 <= processed <= num_samples:
        raise ValueError("Checkpoint contains an invalid processed-sample count.")

    if not 0 <= inside_circle <= processed:
        raise ValueError("Checkpoint contains an invalid inside-circle count.")

    return processed, inside_circle


def run_resumable_pi_estimate(
    config: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
    output_path: str | Path,
    max_batches: int | None = None,
) -> dict[str, Any] | None:
    """Estimate pi with a resumable Monte Carlo calculation.

    Returning None means that execution stopped intentionally after
    ``max_batches`` batches and can be resumed from the checkpoint.
    """
    seed, num_samples, batch_size = _validate_config(config)

    if max_batches is not None:
        if not isinstance(max_batches, int) or isinstance(max_batches, bool):
            raise TypeError("'max_batches' must be an integer or None.")
        if max_batches <= 0:
            raise ValueError("'max_batches' must be positive.")

    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    rng = np.random.default_rng(seed)

    processed = 0
    inside_circle = 0

    if checkpoint_path.exists():
        progress = load_checkpoint(
            checkpoint_path,
            config=config,
            rng=rng,
        )
        processed, inside_circle = _validate_progress(
            progress,
            num_samples=num_samples,
        )

    batches_completed_this_run = 0

    while processed < num_samples:
        if (
            max_batches is not None
            and batches_completed_this_run >= max_batches
        ):
            return None

        current_batch_size = min(batch_size, num_samples - processed)

        points = rng.random((current_batch_size, 2))
        radius_squared = np.sum(points * points, axis=1)

        inside_circle += int(np.count_nonzero(radius_squared <= 1.0))
        processed += current_batch_size

        save_checkpoint(
            checkpoint_path,
            config=config,
            progress={
                "processed": processed,
                "inside_circle": inside_circle,
            },
            rng=rng,
        )

        batches_completed_this_run += 1

    result = {
        "config_sha256": config_hash(config),
        "estimate": 4.0 * inside_circle / num_samples,
        "inside_circle": inside_circle,
        "samples": num_samples,
    }

    atomic_write_json(output_path, result)
    checkpoint_path.unlink(missing_ok=True)

    return result