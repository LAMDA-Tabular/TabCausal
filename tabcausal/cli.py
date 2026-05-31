"""Command line interface for TabCausal inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .evaluate import evaluate_directory
from .inference import TabCausalPredictor
from .preprocessing import load_input_file


def _write_array(path: str | Path | None, array: np.ndarray, *, integer: bool = False) -> None:
    if path is None:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".csv":
        np.savetxt(output, array, delimiter=",", fmt="%d" if integer else "%.8g")
    elif suffix == ".json":
        output.write_text(json.dumps(array.tolist(), indent=2), encoding="utf-8")
    elif suffix == ".npy":
        np.save(output, array)
    else:
        np.savez_compressed(output, array=array)


def _parse_dtype(name: str) -> torch.dtype | None:
    if name == "float32":
        return None
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _predict_one(args: argparse.Namespace) -> None:
    predictor = TabCausalPredictor(
        args.checkpoint,
        device=args.device,
        dtype=_parse_dtype(args.dtype),
        prefer_ema=args.prefer_ema,
        use_amp=not args.no_amp,
        max_observations=args.max_observations,
        observation_seed=args.observation_seed,
    )
    example = load_input_file(args.input, mode=args.mode, intervention_path=args.intervention_input)
    logits = predictor.predict_logits(example.x, batch_size=1)[0]
    probs = 1.0 / (1.0 + np.exp(-logits))
    adjacency = (probs >= args.threshold).astype(np.int64)
    diag = np.arange(adjacency.shape[0])
    adjacency[diag, diag] = 0
    want_embeddings = args.include_embeddings or args.embedding_output is not None
    embeddings = predictor.predict_embeddings(example.x, batch_size=1)[0] if want_embeddings else None

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        np.savetxt(output, adjacency, delimiter=",", fmt="%d")
    elif output.suffix.lower() == ".json":
        output.write_text(json.dumps({"adjacency": adjacency.tolist()}, indent=2), encoding="utf-8")
    else:
        payload = {
            "logits": logits,
            "probabilities": probs,
            "adjacency": adjacency,
            "path": str(example.path),
        }
        if embeddings is not None:
            payload["embeddings"] = embeddings
        np.savez_compressed(output, **payload)
    _write_array(args.adjacency_output, adjacency, integer=True)
    _write_array(args.probability_output, probs)
    _write_array(args.logit_output, logits)
    if embeddings is not None:
        _write_array(args.embedding_output, embeddings)
    print(f"saved: {output}")


def _predict_dir(args: argparse.Namespace) -> None:
    dtype = args.dtype
    outputs = evaluate_directory(
        checkpoint=args.checkpoint,
        data_root=args.input_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        threshold=args.threshold,
        max_per_f=args.max_per_f,
        batch_size=args.batch_size,
        device=args.device,
        dtype=dtype,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TabCausal inference utilities.")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkpoint", required=True, help="Path to a TabCausal public checkpoint.")
    common.add_argument("--mode", choices=["auto", "obs", "mixed"], default="auto")
    common.add_argument("--threshold", type=float, default=0.5)
    common.add_argument("--device", default=None, help="Torch device, for example cuda:0 or cpu.")
    common.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    common.set_defaults(prefer_ema=True)
    common.add_argument(
        "--prefer-ema",
        dest="prefer_ema",
        action="store_true",
        help="Use EMA weights if present in the checkpoint (default).",
    )
    common.add_argument("--no-amp", action="store_true", help="Disable CUDA autocast.")
    common.add_argument("--max-observations", type=int, default=None, help="Deterministically subsample rows before inference.")
    common.add_argument("--observation-seed", type=int, default=0, help="Seed used when max-observations is active.")
    common.add_argument("--no-official-sid", action="store_true", help="Do not try the optional official R SID implementation.")

    one = sub.add_parser("predict", parents=[common], help="Predict one NPZ graph file.")
    one.add_argument("--input", required=True, help="Input .npz, .npy, .csv, .tsv, .parquet, or .pkl file.")
    one.add_argument("--intervention-input", default=None, help="Optional same-shaped intervention indicator table.")
    one.add_argument("--output", required=True, help="Output .npz, .csv, or .json path.")
    one.add_argument("--include-embeddings", action="store_true", help="Include final node embeddings in NPZ output.")
    one.add_argument("--adjacency-output", default=None, help="Optional adjacency output path (.csv/.json/.npy/.npz).")
    one.add_argument("--probability-output", default=None, help="Optional probability matrix output path.")
    one.add_argument("--logit-output", default=None, help="Optional logit matrix output path.")
    one.add_argument("--embedding-output", default=None, help="Optional node embedding output path.")
    one.set_defaults(func=_predict_one)

    many = sub.add_parser("predict-dir", parents=[common], help="Evaluate a directory of NPZ graph files.")
    many.add_argument("--input-dir", required=True, help="Directory containing .npz files.")
    many.add_argument("--output-dir", required=True, help="Directory for raw metrics and predictions.")
    many.add_argument("--batch-size", type=int, default=8)
    many.add_argument("--max-per-f", type=int, default=None)
    many.add_argument("--save-embeddings", action="store_true", help="Save final-layer node embeddings for every graph.")
    many.add_argument("--no-matrix-exports", action="store_true", help="Do not write per-graph adjacency/probability CSV files.")
    many.add_argument("--save-prob-csv", action="store_true", help=argparse.SUPPRESS)
    many.add_argument("--save-adj-csv", action="store_true", help=argparse.SUPPRESS)
    many.add_argument("--no-sid", action="store_true", help="Skip SID computation.")
    many.set_defaults(func=_predict_dir)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
