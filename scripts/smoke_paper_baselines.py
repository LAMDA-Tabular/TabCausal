#!/usr/bin/env python3
"""Run a non-failing smoke test over the included baseline runners."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


OBS_METHODS = ("randomregress", "pc", "lingam", "notears", "notears_mlp", "dagma", "das")
MIXED_METHODS = ("avici", "sea", "sdcd", "dcdi", "gies", "igsp", "cdis", "nodags")


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Suite root containing folders like '[gp_hard]_obs'.")
    parser.add_argument("--output-root", required=True, help="Where baseline result folders and logs are written.")
    parser.add_argument("--datasets", default="gp_hard", help="Comma-separated dataset families to smoke test.")
    parser.add_argument(
        "--obs-methods",
        default=",".join(OBS_METHODS),
        help="Comma-separated observation-only methods to run on '[dataset]_obs'.",
    )
    parser.add_argument(
        "--mixed-methods",
        default=",".join(MIXED_METHODS),
        help="Comma-separated mixed/interventional methods to run on '[dataset]_int'.",
    )
    parser.add_argument("--max-per-f", type=int, default=1, help="Maximum graphs per f value for each method.")
    parser.add_argument("--save-preds", action="store_true", default=True, help="Save predictions.npz when supported.")
    parser.add_argument("--no-save-preds", dest="save_preds", action="store_false", help="Do not request predictions.npz.")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300.0,
        help="Shared wall-clock timeout in seconds for every method subprocess; <=0 disables timeout.",
    )
    parser.add_argument("--cuda-visible-devices", default=None, help="Forwarded to run_paper_baselines.")
    parser.add_argument("--external-algorithms-root", default=None, help="Forwarded to run_paper_baselines.")
    parser.add_argument("--avici-root", default=None, help="Forwarded to run_paper_baselines for official AVICI.")
    parser.add_argument("--avici-checkpoint", default=None, help="Forwarded to the AVICI runner.")
    parser.add_argument("--avici-cache-path", default=None, help="Forwarded to official AVICI for pretrained weights.")
    parser.add_argument("--sea-root", default=None, help="Forwarded to the SEA runner.")
    parser.add_argument("--sea-obs-checkpoint", default=None, help="Forwarded to the SEA runner for observational data.")
    parser.add_argument("--sea-int-checkpoint", default=None, help="Forwarded to the SEA runner for mixed/interventional data.")
    parser.add_argument(
        "--nodags-min-samples",
        type=int,
        default=1,
        help="NoDAGS min_samples used only for this tiny smoke test.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any method fails.")
    return parser.parse_args()


def _write_log(path: Path, text: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


def _run_case(
    *,
    args: argparse.Namespace,
    method: str,
    dataset: str,
    data_suffix: str,
    regime: str,
    output_root: Path,
) -> dict[str, object]:
    input_dir = Path(args.data_root) / f"[{dataset}]_{data_suffix}"
    exp_name = f"{method}_{dataset}_{regime}"
    summary_path = output_root / exp_name / "summary.csv"
    stdout_log = output_root / "_logs" / f"{exp_name}.stdout.txt"
    stderr_log = output_root / "_logs" / f"{exp_name}.stderr.txt"
    timeout = None if args.timeout_seconds <= 0 else args.timeout_seconds
    cmd = [
        sys.executable,
        "-m",
        "baselines.run_paper_baselines",
        "--method",
        method,
        "--input-dir",
        str(input_dir),
        "--output-root",
        str(output_root),
        "--exp-name",
        exp_name,
        "--regime",
        regime,
        "--max-per-f",
        str(args.max_per_f),
    ]
    if args.save_preds:
        cmd.append("--save-preds")
    if args.cuda_visible_devices is not None:
        cmd.extend(["--cuda-visible-devices", args.cuda_visible_devices])
    if args.external_algorithms_root:
        cmd.extend(["--external-algorithms-root", args.external_algorithms_root])
    if method == "avici" and args.avici_root:
        cmd.extend(["--avici-root", args.avici_root])
    if method == "avici" and args.avici_checkpoint:
        cmd.extend(["--avici-checkpoint", args.avici_checkpoint])
    if method == "avici" and args.avici_cache_path:
        cmd.extend(["--avici-cache-path", args.avici_cache_path])
    if method == "sea" and args.sea_root:
        cmd.extend(["--sea-root", args.sea_root])
    if method == "sea" and args.sea_obs_checkpoint:
        cmd.extend(["--sea-obs-checkpoint", args.sea_obs_checkpoint])
    if method == "sea" and args.sea_int_checkpoint:
        cmd.extend(["--sea-int-checkpoint", args.sea_int_checkpoint])
    if method == "dcdi" and timeout is not None:
        cmd.extend(["--dcdi-time-limit-seconds", str(max(1, int(timeout)))])
    if method == "nodags":
        cmd.extend(["--", "--min_samples", str(args.nodags_min_samples)])

    started = time.time()
    print(f"[baseline-smoke] {exp_name}", flush=True)
    if not input_dir.exists():
        message = f"missing input directory: {input_dir}"
        _write_log(stdout_log, "")
        _write_log(stderr_log, message + "\n")
        return {
            "method": method,
            "dataset": dataset,
            "regime": regime,
            "input_dir": str(input_dir),
            "exp_name": exp_name,
            "status": "missing_input",
            "returncode": "",
            "seconds": 0.0,
            "timeout_seconds": "" if timeout is None else timeout,
            "summary_path": str(summary_path),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "command": " ".join(cmd),
        }

    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        seconds = time.time() - started
        _write_log(stdout_log, completed.stdout)
        _write_log(stderr_log, completed.stderr)
        status = "ok" if completed.returncode == 0 and summary_path.exists() else "failed"
        returncode: int | str = completed.returncode
    except subprocess.TimeoutExpired as exc:
        seconds = time.time() - started
        _write_log(stdout_log, exc.stdout if isinstance(exc.stdout, str) else "")
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        _write_log(stderr_log, stderr + f"\nTimed out after {timeout} seconds.\n")
        status = "timeout"
        returncode = "timeout"

    return {
        "method": method,
        "dataset": dataset,
        "regime": regime,
        "input_dir": str(input_dir),
        "exp_name": exp_name,
        "status": status,
        "returncode": returncode,
        "seconds": f"{seconds:.3f}",
        "timeout_seconds": "" if timeout is None else timeout,
        "summary_path": str(summary_path),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "command": " ".join(cmd),
    }


def main() -> None:
    args = parse_args()
    datasets = _csv_list(args.datasets)
    obs_methods = _csv_list(args.obs_methods)
    mixed_methods = _csv_list(args.mixed_methods)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for dataset in datasets:
        for method in obs_methods:
            rows.append(
                _run_case(
                    args=args,
                    method=method,
                    dataset=dataset,
                    data_suffix="obs",
                    regime="obs",
                    output_root=output_root,
                )
            )
        for method in mixed_methods:
            rows.append(
                _run_case(
                    args=args,
                    method=method,
                    dataset=dataset,
                    data_suffix="int",
                    regime="mixed",
                    output_root=output_root,
                )
            )

    manifest = output_root / "baseline_smoke_manifest.csv"
    fieldnames = [
        "method",
        "dataset",
        "regime",
        "input_dir",
        "exp_name",
        "status",
        "returncode",
        "seconds",
        "timeout_seconds",
        "summary_path",
        "stdout_log",
        "stderr_log",
        "command",
    ]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(row["status"] == "ok" for row in rows)
    total = len(rows)
    print(f"manifest: {manifest}")
    print(f"passed: {ok}/{total}")
    if args.strict and ok != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
