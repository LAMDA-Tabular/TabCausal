#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_UTILS_CANDIDATES = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "utils"),
]
for _candidate in _UTILS_CANDIDATES:
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from metrics import ResultTracker, dag_metrics, save_predictions_incremental

FIXED_THRESHOLD = 0.5


def get_f_from_filename(path: str) -> int:
    try:
        match = re.search(r"_f(\d+)_", os.path.basename(path))
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return 0


def apply_threshold(W: np.ndarray) -> np.ndarray:
    """Binarize a weighted adjacency matrix with the fixed release threshold."""
    W_abs = np.abs(W)
    B = (W_abs >= FIXED_THRESHOLD).astype(float)
    np.fill_diagonal(B, 0.0)
    return B


def _random_sort_regress_from_varsortability(X: np.ndarray, seed: int) -> np.ndarray | None:
    """
    Use the official Varsortability/sortnregress recipe, but replace the
    variance-based ordering by a random ordering to match random_sort_regress.
    """
    candidates = [
        Path(__file__).resolve().parent / "Varsortability" / "src",
    ]
    last_error = None
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        try:
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            from sklearn.linear_model import LassoLarsIC, LinearRegression

            rng = np.random.default_rng(seed)
            X = np.asarray(X, dtype=float)
            d = X.shape[1]
            order = rng.permutation(d)
            lr = LinearRegression()
            ll = LassoLarsIC(criterion="bic")
            W = np.zeros((d, d), dtype=float)

            for pos in range(1, d):
                covariates = order[:pos]
                target = int(order[pos])
                lr.fit(X[:, covariates], X[:, target].ravel())
                weight = np.abs(np.asarray(lr.coef_, dtype=float))
                ll.fit(X[:, covariates] * weight, X[:, target].ravel())
                W[covariates, target] = np.asarray(ll.coef_, dtype=float) * weight

            np.fill_diagonal(W, 0.0)
            return W
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        print(
            f"[Info] Varsortability random-sort implementation unavailable, trying package fallback: {last_error}",
            flush=True,
        )
    return None


def _random_sort_regress_fallback(X: np.ndarray, seed: int) -> np.ndarray:
    """
    Minimal fallback only if neither Varsortability nor CausalDisco is available.
    """
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    order = rng.permutation(d)
    W = np.zeros((d, d), dtype=float)

    for pos in range(1, d):
        target = int(order[pos])
        parents = order[:pos]
        if len(parents) == 0:
            continue
        A = X[:, parents]
        y = X[:, target]
        try:
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        except Exception:
            coef = np.zeros(len(parents), dtype=float)
        for parent, value in zip(parents, coef):
            W[int(parent), target] = float(value)
    np.fill_diagonal(W, 0.0)
    return W


def _random_sort_regress_official(X: np.ndarray, seed: int) -> np.ndarray | None:
    """
    Try official CausalDisco implementation first.
    Accept a few possible import paths because package layouts vary.
    """
    candidates = [
        ("causaldisco.baselines", "random_sort_regress"),
        ("causaldisco.baselines.synthetic_baselines", "random_sort_regress"),
        ("CausalDisco.baselines", "random_sort_regress"),
    ]
    last_error = None
    for module_name, func_name in candidates:
        try:
            module = __import__(module_name, fromlist=[func_name])
            func = getattr(module, func_name)
            out = func(X, seed=seed)
            W = np.asarray(out, dtype=float)
            if W.ndim != 2 or W.shape[0] != W.shape[1]:
                continue
            np.fill_diagonal(W, 0.0)
            return W
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        print(f"[Info] Official random_sort_regress unavailable, using fallback: {last_error}", flush=True)
    return None


def run_randomregress_on_file(path: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    x, g = data["x"], data["g"]
    mask = data["mask"].astype(bool) if "mask" in data else np.ones(g.shape[0], dtype=bool)
    if x.ndim == 2:
        x = np.stack([x, np.zeros_like(x)], axis=-1)

    X = x[..., 0][:, mask].astype(float)
    G_true = g[mask][:, mask].astype(int)

    W = _random_sort_regress_from_varsortability(X, seed=seed)
    if W is None:
        W = _random_sort_regress_official(X, seed=seed)
    if W is None:
        W = _random_sort_regress_fallback(X, seed=seed)

    return G_true, W


def main() -> None:
    parser = argparse.ArgumentParser("Run RandomRegress baseline")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--max_per_f", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--results_root", type=str, required=True)
    parser.add_argument("--save_preds", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.data_root):
        print(f"[ERROR] Data root not found: {args.data_root}", flush=True)
        sys.exit(1)

    files = sorted(
        os.path.join(args.data_root, name)
        for name in os.listdir(args.data_root)
        if name.endswith(".npz")
    )
    if not files:
        print("[WARN] No .npz files found", flush=True)
        sys.exit(0)

    files_by_f = defaultdict(list)
    for path in files:
        files_by_f[get_f_from_filename(path)].append(path)

    tracker = ResultTracker(results_root=args.results_root, exp_name=args.exp_name, args=args)
    predictions = {}
    total_processed = 0
    total_failed = 0

    print(f"[Info] Using fixed threshold={FIXED_THRESHOLD}", flush=True)
    print(f"[Info] Found {len(files)} files across {len(files_by_f)} f-values", flush=True)

    for f_val in sorted(files_by_f):
        paths = files_by_f[f_val][:args.max_per_f] if args.max_per_f > 0 else files_by_f[f_val]
        print(f"\n[RandomRegress] Processing f={f_val} ({len(paths)} files)...", flush=True)

        all_data = []
        for idx, path in enumerate(paths):
            try:
                G_true, W = run_randomregress_on_file(path, seed=args.seed + idx)
                all_data.append((path, G_true, W))
            except Exception as exc:
                print(f"[FAIL] RandomRegress failed on {os.path.basename(path)}: {exc}", flush=True)
                total_failed += 1

        if not all_data:
            print(f"[WARN] No successful inference at f={f_val}", flush=True)
            continue

        for path, G_true, W in all_data:
            B = apply_threshold(W)
            metrics = dag_metrics(G_true, B.astype(int))
            fname = os.path.basename(path)
            tracker.log(metrics, f=f_val, filename=fname)
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
