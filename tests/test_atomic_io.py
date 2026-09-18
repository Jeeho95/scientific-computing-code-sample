import json

import numpy as np
import pytest

from scicomp_sample.atomic_io import (
    atomic_save_npz,
    atomic_write_json,
    atomic_write_text,
)


def test_atomic_write_text_overwrites_existing_file(tmp_path):
    path = tmp_path / "result.txt"

    atomic_write_text(path, "first")
    assert path.read_text(encoding="utf-8") == "first"

    atomic_write_text(path, "second")
    assert path.read_text(encoding="utf-8") == "second"


def test_atomic_write_json_round_trip(tmp_path):
    path = tmp_path / "config.json"
    payload = {
        "seed": 1234,
        "samples": 1000,
        "method": "example",
    }

    atomic_write_json(path, payload)

    with path.open(encoding="utf-8") as handle:
        loaded = json.load(handle)

    assert loaded == payload


def test_atomic_save_npz_round_trip(tmp_path):
    path = tmp_path / "arrays.npz"

    x = np.arange(10, dtype=np.int64)
    y = np.linspace(0.0, 1.0, 5, dtype=np.float64)

    atomic_save_npz(path, x=x, y=y)

    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["x"], x)
        np.testing.assert_array_equal(saved["y"], y)


def test_atomic_save_npz_rejects_object_dtype(tmp_path):
    path = tmp_path / "unsafe.npz"
    unsafe = np.asarray([{"value": 1}], dtype=object)

    with pytest.raises(TypeError):
        atomic_save_npz(path, data=unsafe)