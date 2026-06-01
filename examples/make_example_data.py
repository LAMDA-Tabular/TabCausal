#!/usr/bin/env python3
"""Create a small gp_hard benchmark example for local smoke tests."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="examples/example_data")
    parser.add_argument("--num-graphs", type=int, default=100, help="Number of f=5 graphs to create.")
    parser.add_argument("--variables", type=int, default=5, help="Number of variables per graph.")
    parser.add_argument("--observations", type=int, default=1000, help="Observational rows per graph.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-overwrite", action="store_true", help="Keep existing files instead of regenerating them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "generate_benchmark_suite.py"),
        "--output-root",
        args.output_root,
        "--families",
        "gp_hard",
        "--regimes",
        "obs",
        "--f-values",
        str(args.variables),
        "--graphs-per-f",
        str(args.num_graphs),
        "--observations",
        str(args.observations),
        "--seed",
        str(args.seed),
    ]
    if not args.no_overwrite:
        cmd.append("--overwrite")
    subprocess.run(cmd, check=True)
    print(f"example_dir={Path(args.output_root) / '[gp_hard]_obs'}")


if __name__ == "__main__":
    main()
