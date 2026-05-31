#!/usr/bin/env python3
"""Run the baseline scripts included with the release.

This wrapper keeps the public command line compact while reusing the per-method
runners under ``baselines/paper_algorithms``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from baselines.registry import BASELINES, available_baseline_names


PAPER_DIR = Path(__file__).resolve().parent / "paper_algorithms"
DEFAULT_OBS_ONLY = {"pc", "lingam", "notears", "notears_mlp", "dagma", "das", "randomregress"}
DEFAULT_INT_ONLY = {"nodags"}


def _script_for(method: str) -> Path:
    script = PAPER_DIR / f"run_{method}.py"
    if not script.exists():
        raise FileNotFoundError(f"No baseline runner for method {method!r}: {script}")
    return script


def _list_methods() -> None:
    for name in available_baseline_names():
        info = BASELINES[name]
        runner = _script_for(name) if (PAPER_DIR / f"run_{name}.py").exists() else None
        runner_status = "runner=yes" if runner else "runner=no"
        print(f"{name:14s} {info.display_name:14s} {info.status:22s} {runner_status}")
        print(f"  regimes: {','.join(info.regimes)}")
        print(f"  dependency: {info.package_hint}")


def _prepend_path(env: dict[str, str], key: str, values: list[Path]) -> None:
    clean = [str(p) for p in values if p.exists()]
    if not clean:
        return
    existing = env.get(key, "")
    env[key] = os.pathsep.join(clean + ([existing] if existing else []))


def _remove_path_fragment(env: dict[str, str], key: str, fragment: str) -> None:
    existing = env.get(key, "")
    if not existing:
        return
    parts = [part for part in existing.split(os.pathsep) if fragment not in part]
    env[key] = os.pathsep.join(parts)


def _method_env(args: argparse.Namespace, method: str) -> dict[str, str]:
    env = os.environ.copy()
    if method == "avici":
        # The data engine includes an AVICI-derived simulator under
        # data_engine/. That package is not the official AVICI baseline package
        # and does not expose load_pretrained(), so avoid shadowing official AVICI.
        _remove_path_fragment(env, "PYTHONPATH", "data_engine/third_party/avici_data_engine")
    _prepend_path(
        env,
        "PYTHONPATH",
        [
            PAPER_DIR,
            PAPER_DIR / "dagma" / "src",
            PAPER_DIR / "sdcd",
            PAPER_DIR / "sea-reproduce" / "src",
            PAPER_DIR / "Varsortability" / "src",
            PAPER_DIR / "dcdi",
            PAPER_DIR / "dcdi" / "gies",
            PAPER_DIR / "dcdi" / "igsp",
            PAPER_DIR / "bicycle" / "src",
            PAPER_DIR.parent.parent,
        ],
    )
    env.setdefault("DCDI_ROOT", str(PAPER_DIR / "dcdi"))
    env.setdefault("GIES_ROOT", str(PAPER_DIR / "dcdi" / "gies"))
    env.setdefault("IGSP_ROOT", str(PAPER_DIR / "dcdi" / "igsp"))
    env.setdefault("BICYCLE_ROOT", str(PAPER_DIR / "bicycle" / "src"))
    if args.external_algorithms_root:
        root = Path(args.external_algorithms_root).expanduser().resolve()
        env.setdefault("DCDI_ROOT", str(root / "dcdi"))
        env.setdefault("GIES_ROOT", str(root / "dcdi" / "gies"))
        env.setdefault("IGSP_ROOT", str(root / "dcdi" / "igsp"))
        env.setdefault("BICYCLE_ROOT", str(root / "bicycle" / "src"))
        _prepend_path(
            env,
            "PYTHONPATH",
            [
                root,
                root / "dcdi",
                root / "dcdi" / "gies",
                root / "dcdi" / "igsp",
                root / "notears",
                root / "bicycle" / "src",
            ],
        )
    if method == "avici" and args.avici_root:
        root = Path(args.avici_root).expanduser().resolve()
        env["AVICI_ROOT"] = str(root)
        _prepend_path(env, "PYTHONPATH", [root])
    if method == "dcdi":
        # DCDI gets a 5-minute per-graph budget and exports the current learned
        # graph when the budget expires.
        if args.dcdi_time_limit_seconds is not None:
            env["DCDI_TIME_LIMIT_SECONDS"] = str(args.dcdi_time_limit_seconds)
        else:
            env.setdefault("DCDI_TIME_LIMIT_SECONDS", "300")
        env.setdefault("DCDI_EXPORT_ON_TIMEOUT", "1")
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    elif method in {"pc", "gies", "igsp", "lingam", "cdis", "notears", "dagma", "das", "randomregress"}:
        env.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    return env


def _build_command(args: argparse.Namespace, method: str) -> list[str]:
    cmd = [
        sys.executable,
        str(_script_for(method)),
        "--data_root",
        str(Path(args.input_dir).expanduser().resolve()),
        "--max_per_f",
        str(args.max_per_f),
        "--exp_name",
        args.exp_name or f"{method}_baseline",
        "--results_root",
        str(Path(args.output_root).expanduser().resolve()),
    ]
    if args.save_preds:
        cmd.append("--save_preds")
    if method == "avici":
        cmd.append("--use_official")
        if args.avici_checkpoint:
            cmd.extend(["--ckpt_path", str(Path(args.avici_checkpoint).expanduser().resolve())])
        if args.avici_cache_path:
            cmd.extend(["--avici_cache_path", str(Path(args.avici_cache_path).expanduser().resolve())])
    if method == "dagma":
        cmd.extend(["--dagma_root", str((PAPER_DIR / "dagma").resolve())])
    if method == "sdcd":
        cmd.extend(["--sdcd_root", str((PAPER_DIR / "sdcd").resolve())])
    if method == "sea":
        cmd.extend(["--sea_root", str(Path(args.sea_root).expanduser().resolve()) if args.sea_root else str((PAPER_DIR / "sea-reproduce").resolve())])
        if args.sea_obs_checkpoint:
            cmd.extend(["--sea_obs_checkpoint", str(Path(args.sea_obs_checkpoint).expanduser().resolve())])
        if args.sea_int_checkpoint:
            cmd.extend(["--sea_int_checkpoint", str(Path(args.sea_int_checkpoint).expanduser().resolve())])
    if args.extra_args:
        cmd.extend(args.extra_args)
    return cmd


def _check_method_regime(method: str, regime: str) -> None:
    if regime == "auto":
        return
    if regime == "obs" and method in DEFAULT_INT_ONLY:
        raise SystemExit(f"{method} is intervention-only in this benchmark setup; use --regime mixed/int.")
    if regime in {"mixed", "int"} and method in DEFAULT_OBS_ONLY:
        raise SystemExit(f"{method} is observational-only in this benchmark setup; use --regime obs.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List baseline runners and exit.")
    parser.add_argument("--method", choices=available_baseline_names(), help="Baseline method id.")
    parser.add_argument("--input-dir", help="Directory containing benchmark graph files.")
    parser.add_argument("--output-root", default="results/baselines", help="Directory where the method result folder is created.")
    parser.add_argument("--exp-name", default=None, help="Result folder name. Defaults to '<method>_baseline'.")
    parser.add_argument("--max-per-f", type=int, default=-1, help="Maximum graphs per f value; -1 means all.")
    parser.add_argument("--regime", choices=["auto", "obs", "mixed", "int"], default="auto", help="Safety check for method applicability.")
    parser.add_argument("--save-preds", action="store_true", help="Save predictions.npz when supported by the runner.")
    parser.add_argument("--cuda-visible-devices", default=None, help="Value for CUDA_VISIBLE_DEVICES, e.g. '0' or '-1'.")
    parser.add_argument("--external-algorithms-root", default=None, help="Optional override root for comparing against an external algorithms checkout.")
    parser.add_argument("--avici-root", default=None, help="Optional official AVICI package checkout/root to prepend for the AVICI baseline.")
    parser.add_argument("--avici-checkpoint", default=None, help="Optional AVICI checkpoint path for the AVICI runner.")
    parser.add_argument("--avici-cache-path", default=None, help="Optional cache root for official AVICI pretrained weights.")
    parser.add_argument("--sea-root", default=None, help="Optional SEA checkout/root for the SEA baseline.")
    parser.add_argument("--sea-obs-checkpoint", default=None, help="Optional SEA observational checkpoint path.")
    parser.add_argument("--sea-int-checkpoint", default=None, help="Optional SEA mixed/interventional checkpoint path.")
    parser.add_argument("--dcdi-time-limit-seconds", type=int, default=None, help="Optional DCDI per-graph time limit; defaults to 300 seconds.")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Additional arguments passed to the underlying runner after '--'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        _list_methods()
        return
    if not args.method or not args.input_dir:
        raise SystemExit("--method and --input-dir are required unless --list is used.")
    _check_method_regime(args.method, args.regime)
    if args.extra_args and args.extra_args[0] == "--":
        args.extra_args = args.extra_args[1:]
    if shutil.which(sys.executable) is None:
        raise RuntimeError(f"Cannot resolve current Python executable: {sys.executable}")
    cmd = _build_command(args, args.method)
    print("[baseline]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=_method_env(args, args.method))


if __name__ == "__main__":
    main()
