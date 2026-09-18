from __future__ import annotations

import argparse
from pathlib import Path

from scicomp_sample.runner import run_resumable_pi_estimate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small resumable Monte Carlo calculation."
    )
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = {
        "seed": args.seed,
        "num_samples": args.samples,
        "batch_size": args.batch_size,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    result = run_resumable_pi_estimate(
        config,
        checkpoint_path=args.output_dir / "checkpoint.json",
        output_path=args.output_dir / "result.json",
        max_batches=args.max_batches,
    )

    if result is None:
        print(
            "Run paused with a checkpoint. "
            "Re-run with the same configuration to continue."
        )
        return

    print(f"Completed {result['samples']:,} samples.")
    print(f"Monte Carlo estimate of pi: {result['estimate']:.8f}")


if __name__ == "__main__":
    main()