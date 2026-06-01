#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
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
DAGMA_SPARSIFICATION_CUTOFF = 0.3


def resolve_dagma_root(cli_value: str | None) -> str | None:
    candidates = [
        cli_value,
        os.environ.get("DAGMA_ROOT"),
        str(Path(__file__).resolve().parent / "dagma"),
        str(Path.cwd() / "dagma"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate)
        if (root / "src" / "dagma").is_dir():
            return str(root)
    return None


def import_dagma_modules(dagma_root: str | None):
    import_error = None

    if dagma_root:
        src_root = str(Path(dagma_root) / "src")
        if src_root not in sys.path:
            sys.path.insert(0, src_root)

    try:
        from dagma.linear import DagmaLinear
        from dagma.nonlinear import DagmaMLP, DagmaNonlinear
        return DagmaLinear, DagmaMLP, DagmaNonlinear
    except Exception as exc:  # pragma: no cover - import fallback
        import_error = exc

    raise ImportError(
        "Unable to import DAGMA. "
        "Install `dagma` or place the official repo under `algorithms/dagma`.\n"
        f"dagma_root={dagma_root!r}\n"
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


def infer_benchmark_family(exp_name: str, data_root: str) -> str | None:
    for token in [
        "gp_hard",
        "gp_simple",
        "linear_gauss",
        "linear_nongauss",
        "linear_graph",
        "mul_noise",
        "pfn",
    ]:
        if token in exp_name or token in data_root:
            return token
    return None


def infer_variant(family: str | None, cli_variant: str) -> str:
    if cli_variant != "auto":
        return cli_variant
    if family in {"gp_hard", "gp_simple", "mul_noise", "pfn"}:
        return "nonlinear"
    return "linear"


def reset_experiment_dir(results_root: str, exp_name: str) -> str:
    exp_dir = os.path.join(results_root, exp_name)
    if os.path.isdir(exp_dir):
        shutil.rmtree(exp_dir)
    return exp_dir


def apply_threshold(W: np.ndarray) -> np.ndarray:
    """Binarize a weighted adjacency matrix with the fixed release threshold."""
    W_abs = np.abs(W)
    B = (W_abs >= FIXED_THRESHOLD).astype(float)
    np.fill_diagonal(B, 0.0)
    return B


def load_npz_dataset(path: str, *, standardize: bool = False):
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
    X = values[:, mask].astype(np.float64)
    if standardize:
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        X = np.clip((X - mean) / std, -10.0, 10.0)
    G_true = g[mask][:, mask].astype(int)
    return X, G_true


def run_dagma_on_file(path: str, args, DagmaLinear, DagmaMLP, DagmaNonlinear):
    X, G_true = load_npz_dataset(path, standardize=args.standardize)
    d = X.shape[1]

    try:
        if args.variant == "linear":
            model = DagmaLinear(loss_type="l2", verbose=args.verbose, dtype=np.float64)
            W = model.fit(
                X.copy(),
                lambda1=args.lambda1,
                w_threshold=0.0,
                T=args.T,
                mu_init=args.mu_init,
                mu_factor=args.mu_factor,
                s=args.s,
                warm_iter=args.warm_iter,
                max_iter=args.max_iter,
                lr=args.lr,
                checkpoint=args.checkpoint,
            )
        else:
            eq_model = DagmaMLP(dims=[d, args.hidden, 1], bias=True, dtype=torch.double)
            model = DagmaNonlinear(eq_model, verbose=args.verbose, dtype=torch.double)
            W = model.fit(
                X.copy(),
                lambda1=args.lambda1,
                lambda2=args.lambda2,
                T=args.T,
                mu_init=args.mu_init,
                mu_factor=args.mu_factor,
                s=args.s if len(args.s) > 1 else args.s[0],
                warm_iter=args.warm_iter,
                max_iter=args.max_iter,
                lr=args.lr,
                w_threshold=0.0,
                checkpoint=args.checkpoint,
            )
        W = np.asarray(W, dtype=float)
        np.fill_diagonal(W, 0.0)
        # Match the official DAGMA examples by sparsifying the learned weights
        # before the shared release metric step.
        W = (np.abs(W) >= DAGMA_SPARSIFICATION_CUTOFF).astype(float)
        np.fill_diagonal(W, 0.0)
    except Exception as exc:
        print(f"[WARN] DAGMA failed on {os.path.basename(path)}: {exc}", flush=True)
        W = np.zeros_like(G_true, dtype=float)

    return G_true, W


def _dagma_worker(payload):
    path, args_dict = payload
    args = argparse.Namespace(**args_dict)
    torch.set_num_threads(1)
    dagma_root = resolve_dagma_root(args.dagma_root)
    DagmaLinear, DagmaMLP, DagmaNonlinear = import_dagma_modules(dagma_root)
    return path, *run_dagma_on_file(path, args, DagmaLinear, DagmaMLP, DagmaNonlinear)


def main():
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser("Run DAGMA baseline")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--max_per_f", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--results_root", type=str, required=True)
    parser.add_argument("--dagma_root", type=str, default=None)

    parser.add_argument("--variant", choices=["auto", "linear", "nonlinear"], default="auto")
    parser.add_argument("--lambda1", type=float, default=None)
    parser.add_argument("--lambda2", type=float, default=0.005)
    parser.add_argument("--hidden", type=int, default=10)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--mu_init", type=float, default=None)
    parser.add_argument("--mu_factor", type=float, default=0.1)
    parser.add_argument("--s", nargs="+", type=float, default=None)
    parser.add_argument("--warm_iter", type=int, default=None)
    parser.add_argument("--max_iter", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint", type=int, default=1000)
    parser.add_argument("--standardize", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--num_workers", type=int, default=5)

    parser.add_argument("--save_preds", action="store_true", default=False)
    parser.add_argument("--reset_exp_dir", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dagma_root = resolve_dagma_root(args.dagma_root)
    print(f"[DAGMA] dagma_root={dagma_root}", flush=True)
    DagmaLinear, DagmaMLP, DagmaNonlinear = import_dagma_modules(dagma_root)
    if args.reset_exp_dir:
        reset_experiment_dir(args.results_root, args.exp_name)

    family = infer_benchmark_family(args.exp_name, args.data_root)
    args.variant = infer_variant(family, args.variant)
    if args.variant == "linear":
        if args.lambda1 is None:
            args.lambda1 = 0.03
        if args.T is None:
            args.T = 5
        if args.mu_init is None:
            args.mu_init = 1.0
        if args.s is None:
            args.s = [1.0, 0.9, 0.8, 0.7, 0.6]
        if args.warm_iter is None:
            args.warm_iter = int(3e4)
        if args.max_iter is None:
            args.max_iter = int(6e4)
        if args.lr is None:
            args.lr = 3e-4
    else:
        if args.lambda1 is None:
            args.lambda1 = 0.02
        if args.T is None:
            args.T = 4
        if args.mu_init is None:
            args.mu_init = 0.1
        if args.s is None:
            args.s = [1.0] * args.T
        if args.warm_iter is None:
            args.warm_iter = int(5e4)
        if args.max_iter is None:
            args.max_iter = int(8e4)
        if args.lr is None:
            args.lr = 2e-4

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
        f"[Info] family={family} variant={args.variant} "
        f"lambda1={args.lambda1} lambda2={args.lambda2} T={args.T} "
        f"mu_init={args.mu_init} s={args.s} warm_iter={args.warm_iter} "
        f"max_iter={args.max_iter} lr={args.lr} "
        f"dagma_sparsification={DAGMA_SPARSIFICATION_CUTOFF} "
        f"standardize={args.standardize}",
        flush=True,
    )
    print(f"[Info] Found {len(files)} files across {len(files_by_f)} f-values", flush=True)

    total_processed = 0
    total_failed = 0

    for f in sorted(files_by_f):
        paths = files_by_f[f][: args.max_per_f] if args.max_per_f > 0 else files_by_f[f]
        print(f"\n[DAGMA] Processing f={f} ({len(paths)} files)...", flush=True)

        all_data = []
        worker_count = max(1, min(args.num_workers, len(paths)))
        args_dict = vars(args).copy()
        if worker_count == 1:
            for path in paths:
                try:
                    G_true, W = run_dagma_on_file(path, args, DagmaLinear, DagmaMLP, DagmaNonlinear)
                    all_data.append((path, G_true, W))
                except Exception as exc:
                    print(f"[FAIL] Inference failed on {os.path.basename(path)}: {exc}", flush=True)
                    total_failed += 1
        else:
            print(f"[Info] Running {len(paths)} DAGMA jobs in parallel with {worker_count} workers", flush=True)
            with ProcessPoolExecutor(max_workers=worker_count) as ex:
                futures = [ex.submit(_dagma_worker, (path, args_dict)) for path in paths]
                for fut in as_completed(futures):
                    try:
                        path, G_true, W = fut.result()
                        all_data.append((path, G_true, W))
                    except Exception as exc:
                        print(f"[FAIL] Parallel DAGMA worker failed: {exc}", flush=True)
                        total_failed += 1
            all_data.sort(key=lambda item: item[0])

        if not all_data:
            print(f"[WARN] No successful inference at f={f}", flush=True)
            continue

        for path, G_true, W in all_data:
            B = apply_threshold(W)
            metrics = dag_metrics(G_true, B.astype(int))
            fname = os.path.basename(path)
            tracker.log(metrics, f=f, filename=fname)
            if args.save_preds:
                predictions[fname] = W.astype(np.float16)
            total_processed += 1

    tracker.finalize()

    if args.save_preds and predictions:
        save_path = os.path.join(args.results_root, args.exp_name, "predictions.npz")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_predictions_incremental(save_path, predictions, verbose=True)
        print(f"\n[Info] Predictions saved to {save_path}", flush=True)

    print(f"\n[Summary] {total_processed} succeeded, {total_failed} failed", flush=True)


if __name__ == "__main__":
    main()
