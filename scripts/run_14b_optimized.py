"""Run an optimized 14B benchmark sweep.

This is a convenience wrapper around [`scripts/run_benchmarks.py`](scripts/run_benchmarks.py:1)
to match the optimization docs and provide a one-command entrypoint.

Notes
-----
- The real optimization logic lives in [`configs/benchmark_config.yaml`](configs/benchmark_config.yaml:1)
  and the main runner. This script just wires common flags.
"""

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run optimized 14B benchmarks")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/benchmark_config.yaml",
        help="Path to benchmark configuration file",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (overrides config)",
    )
    parser.add_argument(
        "--time_budget",
        type=float,
        default=None,
        help="Time budget in hours (overrides config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory",
    )

    args = parser.parse_args()

    cmd = [
        sys.executable,
        "scripts/run_benchmarks.py",
        "--config",
        args.config,
        "--model_sizes",
        "14B",
        "--quantize",
    ]

    if args.num_gpus is not None:
        cmd.extend(["--num_gpus", str(args.num_gpus)])
    if args.time_budget is not None:
        cmd.extend(["--time_budget", str(args.time_budget)])
    if args.output_dir is not None:
        cmd.extend(["--output_dir", args.output_dir])

    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
