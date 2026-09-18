from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

PathLike = str | os.PathLike[str]


def _destination(path: PathLike) -> Path:
    """Return a destination path and create its parent directory if needed."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest


def atomic_write_bytes(path: PathLike, data: bytes) -> None:
    """Write bytes using a temporary file and atomically replace the destination."""
    dest = _destination(path)
    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=dest.parent,
            prefix=f".{dest.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temp_path, dest)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def atomic_write_text(
    path: PathLike,
    text: str,
    *,
    encoding: str = "utf-8",
) -> None:
    """Atomically write a text file."""
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: PathLike, obj: Any) -> None:
    """Atomically write deterministic, human-readable JSON."""
    text = json.dumps(
        obj,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    )
    atomic_write_text(path, text + "\n")


def atomic_save_npz(path: PathLike, **arrays: np.ndarray) -> None:
    """Atomically save NumPy arrays in a pickle-free compressed NPZ file."""
    if not arrays:
        raise ValueError("At least one array is required.")

    normalized = {name: np.asarray(value) for name, value in arrays.items()}

    object_arrays = [
        name for name, value in normalized.items() if value.dtype == object
    ]
    if object_arrays:
        raise TypeError(
            f"Object-dtype arrays are not allowed: {object_arrays}"
        )

    dest = _destination(path)
    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=dest.parent,
            prefix=f".{dest.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            np.savez_compressed(handle, **normalized)
            handle.flush()
            os.fsync(handle.fileno())

        with np.load(temp_path, allow_pickle=False) as saved:
            if set(saved.files) != set(normalized):
                raise RuntimeError("Saved NPZ fields do not match the requested fields.")

            for name, original in normalized.items():
                loaded = saved[name]
                if loaded.shape != original.shape:
                    raise RuntimeError(f"Shape mismatch while validating {name}.")
                if loaded.dtype != original.dtype:
                    raise RuntimeError(f"Dtype mismatch while validating {name}.")

        os.replace(temp_path, dest)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()