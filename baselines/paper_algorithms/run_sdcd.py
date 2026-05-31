#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_UTILS_CANDIDATES = [
    os.path.join(os.environ.get("CAUSAL_BENCHMARK_ROOT", ""), "utils"),
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "utils"),
]
for _candidate in _UTILS_CANDIDATES:
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from metrics import ResultTracker, dag_metrics, save_predictions_incremental

FIXED_THRESHOLD = 0.5
SDCD_STAGE_THRESHOLD = 0.1
SDCD_MASK_CUTOFF = 0.2
SDCD_FREEZE_GAMMA_CUTOFF = 0.01
SDCD_USE_SECOND_STAGE_REFINEMENT = True
from preprocess import standardize_x


def resolve_sdcd_root(cli_value: str | None) -> str | None:
    candidates = [
        cli_value,
        os.environ.get("SDCD_ROOT"),
        str(Path(__file__).resolve().parent / "sdcd"),
        str(Path.cwd() / "sdcd"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate)
        if (root / "sdcd" / "__init__.py").is_file():
            return str(root)
    return None


def _install_wandb_stub():
    wandb = types.ModuleType("wandb")
    wandb.run = None
    wandb.init = lambda *args, **kwargs: None
    wandb.log = lambda *args, **kwargs: None
    wandb.finish = lambda *args, **kwargs: None
    sys.modules["wandb"] = wandb


def import_sdcd_modules(sdcd_root: str | None):
    import_error = None

    if sdcd_root and sdcd_root not in sys.path:
        sys.path.insert(0, sdcd_root)

    try:
        import wandb  # noqa: F401
    except Exception:
        _install_wandb_stub()

    try:
        from sdcd.models import SDCD
        from sdcd.utils import create_intervention_dataset
        return SDCD, create_intervention_dataset
    except Exception as exc:  # pragma: no cover - import fallback
        import_error = exc

    raise ImportError(
        "Unable to import SDCD. Install `sdcd` or place the official repo under `algorithms/sdcd`.\n"
        f"sdcd_root={sdcd_root!r}\n"
        f"original_error={import_error}"
    )


def get_f_from_filename(path: str) -> int:
    try:
        match = re.search(r"_f(\d+)_", os.path.basename(path))
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return 0


def reset_experiment_dir(results_root: str, exp_name: str) -> str:
    exp_dir = os.path.join(results_root, exp_name)
    if os.path.isdir(exp_dir):
        shutil.rmtree(exp_dir)
    return exp_dir


def npz_to_sdcd_dataframe(path: str, *, normalize: bool = True) -> tuple[pd.DataFrame, np.ndarray]:
    data = np.load(path)
    x = data["x"]
    g = data["g"]
    mask = data["mask"].astype(bool) if "mask" in data else np.ones(g.shape[0], dtype=bool)
    if mask.ndim == 2:
        if mask.shape[1] != g.shape[0]:
            raise ValueError(f"Unexpected 2D mask shape {mask.shape} for graph shape {g.shape}")
        mask = mask[0] if np.all(mask == mask[0]) else np.any(mask, axis=0)
    if mask.ndim != 1 or mask.shape[0] != g.shape[0]:
        raise ValueError(f"Unexpected mask shape {mask.shape} for graph shape {g.shape}")

    values = x[..., 0] if x.ndim == 3 else x
    interventions = x[..., 1] if x.ndim == 3 else np.zeros_like(values)

    X = values[:, mask].astype(float)
    I = interventions[:, mask].astype(float)
    G_true = g[mask][:, mask].astype(int)

    if normalize:
        X = standardize_x(X)

    labels = []
    for row in I:
        idx = np.where(row > 0.5)[0]
        if len(idx) == 0:
            labels.append("obs")
        else:
            labels.append(",".join(str(int(i)) for i in idx))

    columns = [str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=columns)
    df["perturbation_label"] = labels
    return df, G_true


def threshold_with_sdcd(SDCD, W: np.ndarray, threshold: float | bool) -> np.ndarray:
    return SDCD.adjacency_dag_at_threshold(np.asarray(W, dtype=float), threshold=float(threshold)).astype(int)


def run_sdcd_on_file(path: str, args, SDCD, create_intervention_dataset):
    X_df, G_true = npz_to_sdcd_dataframe(path, normalize=args.normalize)
    dataset = create_intervention_dataset(X_df)

    device = None
    if (not args.cpu_only) and torch.cuda.is_available():
        device = torch.device("cuda")

    stage1_kwargs = {
        "learning_rate": args.stage1_lr,
        "batch_size": args.stage1_batch_size,
        "n_epochs": args.stage1_epochs,
        "alpha": args.stage1_alpha,
        "beta": args.stage1_beta,
        "gamma_increment": args.stage1_gamma_increment,
        "n_epochs_check": args.stage1_check_every,
        "mask_threshold": SDCD_MASK_CUTOFF,
    }
    stage2_kwargs = {
        "learning_rate": args.stage2_lr,
        "batch_size": args.stage2_batch_size,
        "n_epochs": args.stage2_epochs,
        "alpha": args.stage2_alpha,
        "beta": args.stage2_beta,
        "gamma_increment": args.stage2_gamma_increment,
        "gamma_schedule": args.stage2_gamma_schedule,
        "freeze_gamma_at_dag": args.freeze_gamma_at_dag,
        "freeze_gamma_threshold": SDCD_FREEZE_GAMMA_CUTOFF,
        "threshold": SDCD_STAGE_THRESHOLD,
        "n_epochs_check": args.stage2_check_every,
        "dag_penalty_flavor": args.dag_penalty_flavor,
    }
    model_kwargs = {
        "num_layers": args.num_layers,
        "dim_hidden": args.dim_hidden,
        "power_iteration_n_steps": args.power_iteration_n_steps,
    }

    model = SDCD(
        model_variance_flavor=args.model_variance_flavor,
        standard_scale=False,
        use_gumbel=args.use_gumbel,
    )
    model.train(
        dataset,
        val_dataset=None,
        val_fraction=args.val_fraction,
        log_wandb=False,
        finetune=SDCD_USE_SECOND_STAGE_REFINEMENT,
        B_true=G_true,
        stage1_kwargs=stage1_kwargs,
        stage2_kwargs=stage2_kwargs,
        model_kwargs=model_kwargs,
        verbose=args.verbose,
        device=device,
        skip_stage1=args.skip_stage1,
        warm_start=args.warm_start,
        skip_masking=args.skip_masking,
    )

    W = np.asarray(model.get_adjacency_matrix(threshold=False), dtype=float)
    np.fill_diagonal(W, 0.0)
    B_builtin = np.asarray(model.get_adjacency_matrix(threshold=True), dtype=int)
    np.fill_diagonal(B_builtin, 0)
    return G_true, W, B_builtin


def main():
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser("Run SDCD baseline")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--max_per_f", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--results_root", type=str, required=True)
    parser.add_argument("--sdcd_root", type=str, default=None)

    parser.add_argument("--model_variance_flavor", choices=["unit", "nn", "parameter"], default="nn")
    parser.add_argument("--use_gumbel", action="store_true", default=False)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--cpu_only", action="store_true", default=False)
    parser.add_argument("--skip_stage1", action="store_true", default=False)
    parser.add_argument("--warm_start", action="store_true", default=False)
    parser.add_argument("--skip_masking", action="store_true", default=False)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dim_hidden", type=int, default=10)
    parser.add_argument("--power_iteration_n_steps", type=int, default=15)

    parser.add_argument("--stage1_lr", type=float, default=2e-3)
    parser.add_argument("--stage1_batch_size", type=int, default=256)
    parser.add_argument("--stage1_epochs", type=int, default=2000)
    parser.add_argument("--stage1_alpha", type=float, default=1e-2)
    parser.add_argument("--stage1_beta", type=float, default=2e-4)
    parser.add_argument("--stage1_gamma_increment", type=float, default=0.0)
    parser.add_argument("--stage1_check_every", type=int, default=100)

    parser.add_argument("--stage2_lr", type=float, default=1e-3)
    parser.add_argument("--stage2_batch_size", type=int, default=256)
    parser.add_argument("--stage2_epochs", type=int, default=2000)
    parser.add_argument("--stage2_alpha", type=float, default=5e-4)
    parser.add_argument("--stage2_beta", type=float, default=5e-3)
    parser.add_argument("--stage2_gamma_increment", type=float, default=0.005)
    parser.add_argument("--stage2_gamma_schedule", type=str, default="linear")
    parser.add_argument("--freeze_gamma_at_dag", action="store_true", default=True)
    parser.add_argument("--stage2_check_every", type=int, default=100)
    parser.add_argument("--dag_penalty_flavor", type=str, default="power_iteration")
    parser.add_argument("--save_preds", action="store_true", default=False)
    parser.add_argument("--reset_exp_dir", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sdcd_root = resolve_sdcd_root(args.sdcd_root)
    print(f"[SDCD] sdcd_root={sdcd_root}", flush=True)
    SDCD, create_intervention_dataset = import_sdcd_modules(sdcd_root)
    if args.reset_exp_dir:
        reset_experiment_dir(args.results_root, args.exp_name)

    files = sorted(
        [
            os.path.join(args.data_root, f)
            for f in os.listdir(args.data_root)
            if f.endswith(".npz")
        ]
    )
    files_by_f = defaultdict(list)
    for path in files:
        files_by_f[get_f_from_filename(path)].append(path)

    tracker = ResultTracker(
        results_root=args.results_root,
        exp_name=args.exp_name,
        args=args,
    )
    predictions = {}

    print(f"[Info] Using fixed threshold={FIXED_THRESHOLD}", flush=True)
    print(
        f"[Info] stage1_epochs={args.stage1_epochs} stage2_epochs={args.stage2_epochs} "
        f"normalize={args.normalize}",
        flush=True,
    )
    print(f"[Info] Found {len(files)} files across {len(files_by_f)} f-values", flush=True)

    total_processed = 0
    total_failed = 0

    for f in sorted(files_by_f):
        paths = files_by_f[f][: args.max_per_f] if args.max_per_f > 0 else files_by_f[f]
        print(f"\n[SDCD] Processing f={f} ({len(paths)} files)...", flush=True)

        successes_this_f = 0
        for path in paths:
            try:
                G_true, W, _ = run_sdcd_on_file(path, args, SDCD, create_intervention_dataset)
            except Exception as exc:
                print(f"[FAIL] Inference failed on {os.path.basename(path)}: {exc}", flush=True)
                total_failed += 1
                continue

            B = threshold_with_sdcd(SDCD, W, FIXED_THRESHOLD)
            metrics = dag_metrics(G_true, np.asarray(B, dtype=int))
            fname = os.path.basename(path)
            tracker.log(metrics, f=f, filename=fname)
            if args.save_preds:
                predictions[fname] = W.astype(np.float16)
                save_path = os.path.join(args.results_root, args.exp_name, "predictions.npz")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_predictions_incremental(save_path, {fname: predictions[fname]}, verbose=False)
            total_processed += 1
            successes_this_f += 1

        if successes_this_f == 0:
            print(f"[WARN] No successful inference at f={f}", flush=True)

    tracker.finalize()

    if args.save_preds and predictions:
        save_path = os.path.join(args.results_root, args.exp_name, "predictions.npz")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_predictions_incremental(save_path, predictions, verbose=True)
        print(f"\n[Info] Predictions saved to {save_path}", flush=True)

    print(f"\n[Summary] {total_processed} succeeded, {total_failed} failed", flush=True)


if __name__ == "__main__":
    main()
