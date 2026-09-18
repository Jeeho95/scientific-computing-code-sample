from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_json


def canonical_json(config: Mapping[str, Any]) -> str:
    """Serialize a configuration deterministically."""
    return json.dumps(
        dict(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def config_hash(config: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 hash for a configuration."""
    payload = canonical_json(config).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_config_snapshot(
    path: str | Path,
    config: Mapping[str, Any],
) -> None:
    """Write a configuration together with its integrity hash."""
    payload = {
        "config": dict(config),
        "sha256": config_hash(config),
    }
    atomic_write_json(path, payload)


def load_config_snapshot(path: str | Path) -> dict[str, Any]:
    """Load a configuration snapshot and verify its stored hash."""
    snapshot_path = Path(path)

    with snapshot_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise TypeError("Configuration snapshot must contain a JSON object.")

    if "config" not in payload or "sha256" not in payload:
        raise ValueError("Configuration snapshot is missing required fields.")

    config = payload["config"]
    expected_hash = payload["sha256"]

    if not isinstance(config, dict):
        raise TypeError("The stored configuration must be a JSON object.")

    actual_hash = config_hash(config)

    if actual_hash != expected_hash:
        raise ValueError(
            "Configuration hash mismatch: the snapshot may have been modified."
        )

    return config