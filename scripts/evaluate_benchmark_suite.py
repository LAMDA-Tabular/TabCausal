#!/usr/bin/env python3
"""Evaluate TabCausal on the seven-family benchmark suite."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tabcausal.evaluate import evaluate_directory
from tqdm.auto import tqdm


DEFAULT_FAMILIES = (
    "gp_hard",
    "gp_simple",
    "linear_gauss",
    "linear_graph",
    "linear_nongauss",
    "mul_noise",
    "pfn",
)
DEFAULT_REGIMES = ("obs", "int")


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _read_summary(path: Path, *, dataset: str, regime: str) -> list[dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, object]] = []
        for row in reader:
            out: dict[str, object] = {"dataset": dataset, "regime": regime}
            out.update(row)
            rows.append(out)
        return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a TabCausal checkpoint.")
    parser.add_argument(
        "--suite-data-root",
        required=True,
        help="Directory containing dataset folders such as [gp_hard]_obs and [gp_hard]_int.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for per-dataset and aggregate results.")
    parser.add_argument(
        "--families",
        default=",".join(DEFAULT_FAMILIES),
        help="Comma-separated synthetic families to evaluate.",
    )
    parser.add_argument("--regimes", default=",".join(DEFAULT_REGIMES), help="Comma-separated regimes: obs,int.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-per-f", type=int, default=None, help="Optional cap per graph size.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.set_defaults(prefer_ema=True)
    parser.add_argument("--prefer-ema", dest="prefer_ema", action="store_true", help="Use EMA weights when present (default).")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-observations", type=int, default=None)
    parser.add_argument("--observation-seed", type=int, default=0)
    parser.add_argument(
        "--file-offset",
        type=int,
        default=0,
        help="Skip this many files within each f-value before applying --max-per-f.",
    )
    parser.add_argument("--skip-missing", action="store_true", help="Skip missing dataset folders instead of failing.")
    parser.add_argument("--save-embeddings", action="store_true", help="Save final-layer node embeddings for every graph.")
    parser.add_argument("--no-matrix-exports", action="store_true", help="Do not write per-graph adjacency/probability CSV files.")
    parser.add_argument("--no-sid", action="store_true", help="Skip SID computation.")
    parser.add_argument("--no-official-sid", action="store_true", help="Do not try the optional official R SID implementation.")
    parser.add_argument("--quiet", action="store_true", help="Disable progress messages.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_root = Path(args.suite_data_root)
    output_root = Path(args.output_dir)
    families = _parse_csv(args.families)
    regimes = _parse_csv(args.regimes)

    aggregate_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    tasks = [(family, regime) for family in families for regime in regimes]
    iterator = tqdm(tasks, desc="Benchmark suite", unit="dataset", disable=args.quiet)
    for family, regime in iterator:
        dataset = f"[{family}]_{regime}"
        iterator.set_postfix_str(dataset)
        data_root = suite_root / dataset
        result_dir = output_root / dataset
        if not data_root.exists():
            if args.skip_missing:
                manifest_rows.append({"dataset": dataset, "status": "missing", "data_root": str(data_root)})
                continue
            raise FileNotFoundError(f"Dataset folder not found: {data_root}")

        mode = "obs" if regime == "obs" else "mixed"
        outputs = evaluate_directory(
            checkpoint=args.checkpoint,
            data_root=data_root,
            output_dir=result_dir,
            mode=mode,
            threshold=args.threshold,
            max_per_f=args.max_per_f,
            batch_size=args.batch_size,
            device=args.device,
            dtype=args.dtype,
            prefer_ema=args.prefer_ema,
            use_amp=not args.no_amp,
            max_observations=args.max_observations,
            observation_seed=args.observation_seed,
            file_offset=args.file_offset,
            progress=not args.quiet,
            save_adjacencies=not args.no_matrix_exports,
            save_embeddings=args.save_embeddings,
            compute_sid=not args.no_sid,
            prefer_official_sid=not args.no_official_sid,
        )
        manifest_rows.append(
            {
                "dataset": dataset,
                "status": "ok",
                "data_root": str(data_root),
                "summary": str(outputs["summary"]),
                "raw_metrics": str(outputs["raw_metrics"]),
                "predictions": str(outputs["predictions"]),
            }
        )
        aggregate_rows.extend(_read_summary(outputs["summary"], dataset=family, regime=regime))

    _write_csv(output_root / "manifest.csv", manifest_rows)
    _write_csv(output_root / "benchmark_summary.csv", aggregate_rows)
    print(f"manifest: {output_root / 'manifest.csv'}")
    print(f"summary: {output_root / 'benchmark_summary.csv'}")


if __name__ == "__main__":
    main()
