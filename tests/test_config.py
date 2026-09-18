import json

import pytest

from scicomp_sample.config import (
    config_hash,
    load_config_snapshot,
    write_config_snapshot,
)


def test_config_hash_is_independent_of_key_order():
    config_a = {
        "seed": 123,
        "simulation": {
            "samples": 1000,
            "scale": 0.5,
        },
    }

    config_b = {
        "simulation": {
            "scale": 0.5,
            "samples": 1000,
        },
        "seed": 123,
    }

    assert config_hash(config_a) == config_hash(config_b)


def test_config_hash_changes_when_configuration_changes():
    config_a = {
        "seed": 123,
        "samples": 1000,
    }

    config_b = {
        "seed": 123,
        "samples": 1001,
    }

    assert config_hash(config_a) != config_hash(config_b)


def test_config_snapshot_round_trip(tmp_path):
    path = tmp_path / "config.json"

    config = {
        "seed": 7,
        "samples": 5000,
        "batch_size": 100,
    }

    write_config_snapshot(path, config)

    loaded = load_config_snapshot(path)

    assert loaded == config


def test_config_snapshot_detects_modified_configuration(tmp_path):
    path = tmp_path / "config.json"

    config = {
        "seed": 7,
        "samples": 5000,
    }

    write_config_snapshot(path, config)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["config"]["samples"] = 5001
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_config_snapshot(path)


def test_config_hash_rejects_nan():
    config = {
        "value": float("nan"),
    }

    with pytest.raises(ValueError):
        config_hash(config)