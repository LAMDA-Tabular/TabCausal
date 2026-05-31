#!/usr/bin/env python3
"""Evaluate TabCausal on a directory of benchmark graph files."""

from __future__ import annotations

import argparse

from tabcausal.evaluate import evaluate_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a TabCausal checkpoint.")
    parser.add_argument("--data-root", required=True, help="Directory containing benchmark graph files.")
    parser.add_argument("--output-dir", required=True, help="Directory for raw metrics, summary, and predictions.")
    parser.add_argument("--mode", choices=["auto", "obs", "mixed"], default="auto", help="How to assemble inputs.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for binary graph metrics.")
    parser.add_argument("--max-per-f", type=int, default=None, help="Optional cap per graph size f.")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.set_defaults(prefer_ema=True)
    parser.add_argument(
        "--prefer-ema",
        dest="prefer_ema",
        action="store_true",
        help="Use EMA weights when present (default).",
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA autocast.")
    parser.add_argument("--max-observations", type=int, default=None, help="Deterministically subsample rows before inference.")
    parser.add_argument("--observation-seed", type=int, default=0, help="Seed used when max-observations is active.")
    parser.add_argument("--save-embeddings", action="store_true", help="Save final-layer node embeddings for every graph.")
    parser.add_argument("--no-matrix-exports", action="store_true", help="Do not write per-graph adjacency/probability CSV files.")
    parser.add_argument("--no-sid", action="store_true", help="Skip SID computation.")
    parser.add_argument("--no-official-sid", action="store_true", help="Do not try the optional official R SID implementation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = evaluate_directory(
        checkpoint=args.checkpoint,
        data_root=args.data_root,
        output_dir=args.output_dir,
        mode=args.mode,
        threshold=args.threshold,
        max_per_f=args.max_per_f,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        prefer_ema=args.prefer_ema,
        use_amp=not args.no_amp,
        max_observations=args.max_observations,
        observation_seed=args.observation_seed,
        save_adjacencies=not args.no_matrix_exports,
        save_embeddings=args.save_embeddings,
        compute_sid=not args.no_sid,
        prefer_official_sid=not args.no_official_sid,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
