import numpy as np
import pytest

from scicomp_sample.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip(tmp_path):
    path = tmp_path / "checkpoint.json"
    config = {
        "seed": 123,
        "samples": 1000,
    }
    progress = {
        "step": 250,
        "partial_sum": 17.5,
    }
    rng = np.random.default_rng(123)

    save_checkpoint(
        path,
        config=config,
        progress=progress,
        rng=rng,
    )

    restored_rng = np.random.default_rng(999)
    restored_progress = load_checkpoint(
        path,
        config=config,
        rng=restored_rng,
    )

    assert restored_progress == progress


def test_checkpoint_restores_exact_rng_position(tmp_path):
    path = tmp_path / "checkpoint.json"
    config = {
        "seed": 123,
        "samples": 1000,
    }

    rng = np.random.default_rng(123)

    _ = rng.normal(size=100)

    save_checkpoint(
        path,
        config=config,
        progress={"step": 100},
        rng=rng,
    )

    expected = rng.normal(size=50)

    restored_rng = np.random.default_rng(999)
    progress = load_checkpoint(
        path,
        config=config,
        rng=restored_rng,
    )
    actual = restored_rng.normal(size=50)

    assert progress["step"] == 100
    np.testing.assert_array_equal(actual, expected)


def test_checkpoint_rejects_different_configuration(tmp_path):
    path = tmp_path / "checkpoint.json"

    original_config = {
        "seed": 123,
        "samples": 1000,
    }
    changed_config = {
        "seed": 123,
        "samples": 2000,
    }

    rng = np.random.default_rng(123)

    save_checkpoint(
        path,
        config=original_config,
        progress={"step": 100},
        rng=rng,
    )

    restored_rng = np.random.default_rng(123)

    with pytest.raises(ValueError, match="configuration mismatch"):
        load_checkpoint(
            path,
            config=changed_config,
            rng=restored_rng,
        )