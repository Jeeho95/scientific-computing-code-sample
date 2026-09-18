import json

from scicomp_sample.runner import run_resumable_pi_estimate


def test_resumed_run_matches_uninterrupted_run(tmp_path):
    config = {
        "seed": 2026,
        "num_samples": 10_003,
        "batch_size": 333,
    }

    uninterrupted_dir = tmp_path / "uninterrupted"
    resumed_dir = tmp_path / "resumed"

    uninterrupted_result = run_resumable_pi_estimate(
        config,
        checkpoint_path=uninterrupted_dir / "checkpoint.json",
        output_path=uninterrupted_dir / "result.json",
    )

    partial_result = run_resumable_pi_estimate(
        config,
        checkpoint_path=resumed_dir / "checkpoint.json",
        output_path=resumed_dir / "result.json",
        max_batches=7,
    )

    assert partial_result is None
    assert (resumed_dir / "checkpoint.json").exists()
    assert not (resumed_dir / "result.json").exists()

    resumed_result = run_resumable_pi_estimate(
        config,
        checkpoint_path=resumed_dir / "checkpoint.json",
        output_path=resumed_dir / "result.json",
    )

    assert resumed_result == uninterrupted_result

    uninterrupted_json = json.loads(
        (uninterrupted_dir / "result.json").read_text(encoding="utf-8")
    )
    resumed_json = json.loads(
        (resumed_dir / "result.json").read_text(encoding="utf-8")
    )

    assert resumed_json == uninterrupted_json
    assert not (resumed_dir / "checkpoint.json").exists()


def test_partial_run_preserves_progress(tmp_path):
    config = {
        "seed": 7,
        "num_samples": 1000,
        "batch_size": 100,
    }

    checkpoint_path = tmp_path / "checkpoint.json"
    output_path = tmp_path / "result.json"

    result = run_resumable_pi_estimate(
        config,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        max_batches=3,
    )

    assert result is None
    assert checkpoint_path.exists()
    assert not output_path.exists()

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))

    assert checkpoint["progress"]["processed"] == 300
    assert 0 <= checkpoint["progress"]["inside_circle"] <= 300