#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
import sys
from pathlib import Path
import re
import warnings

# =====================================================
#  
# =====================================================
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from metrics import dag_metrics, ResultTracker, save_predictions_incremental
from dodiscover import make_context
from dodiscover.toporder import DAS

# =====================================================
# Utils
# =====================================================

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in ("true", "1", "yes", "y", "t"):
        return True
    if v in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


def get_f_from_filename(p):
    """  f  """
    try:
        match = re.search(r"_f(\d+)_", os.path.basename(p))
        if match:
            return int(match.group(1))
        match = re.search(r"f(\d+)", os.path.basename(p))
        if match:
            return int(match.group(1))
        return 0
    except Exception:
        return 0


def nx_digraph_to_adj(G, columns):
    """
      networkx.DiGraph   A
    A[i, j] = 1   i -> j
    """
    d = len(columns)
    A = np.zeros((d, d), dtype=int)
    col_to_idx = {col: i for i, col in enumerate(columns)}

    for u, v in G.edges():
        if u in col_to_idx and v in col_to_idx:
            A[col_to_idx[u], col_to_idx[v]] = 1

    return A

# =====================================================
# DAS core
# =====================================================

def run_das_on_file(
    path,
    eta_G=0.001,
    eta_H=0.001,
    alpha=0.05,
    prune=True,
    das_cutoff=None,
    n_splines=10,
    splines_degree=3,
    min_parents=5,
    max_parents=20,
):
    """
      DAS，  (G_true, A_pred)

     ：
    - DAS   observational order-based  ，  x[..., 0]
    - x[..., 1]（intervention flag） 
    """
    data = np.load(path)
    x, g = data["x"], data["g"]
    mask = data["mask"].astype(bool) if "mask" in data else np.ones(g.shape[0], dtype=bool)
    if x.ndim == 2:
        x = np.stack([x, np.zeros_like(x)], axis=-1)

    X = x[..., 0][:, mask].astype(float)
    G_true = g[mask][:, mask].astype(int)

    #   0,1,...,d-1，  graph_  
    df_X = pd.DataFrame(X)

    context = make_context().variables(data=df_X).build()

    model = DAS(
        eta_G=eta_G,
        eta_H=eta_H,
        alpha=alpha,
        prune=prune,
        das_cutoff=das_cutoff,
        n_splines=n_splines,
        splines_degree=splines_degree,
        min_parents=min_parents,
        max_parents=max_parents,
    )

    try:
        # dodiscover   spline   warning，benchmark  
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            model.learn_graph(df_X, context)

        G_dag = model.graph_   # pruning   DAG
        A_pred = nx_digraph_to_adj(G_dag, list(df_X.columns))

    except Exception as e:
        print(f"[WARN] DAS failed on {os.path.basename(path)}: {e}", flush=True)
        A_pred = np.zeros_like(G_true)

    return G_true, A_pred

# =====================================================
# Main
# =====================================================

def main():
    parser = argparse.ArgumentParser("Run DAS baseline")

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--max_per_f", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--results_root", type=str, required=True)

    parser.add_argument("--eta_G", type=float, default=0.001)
    parser.add_argument("--eta_H", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--prune", type=str2bool, default=True)

    # DAS specific fixed params
    parser.add_argument(
        "--das_cutoff",
        type=float,
        default=-1.0,
        help="If <0, interpreted as None"
    )
    parser.add_argument("--n_splines", type=int, default=10)
    parser.add_argument("--splines_degree", type=int, default=3)
    parser.add_argument("--min_parents", type=int, default=5)
    parser.add_argument("--max_parents", type=int, default=20)

    parser.add_argument("--save_preds", action="store_true")

    args = parser.parse_args()
    np.random.seed(args.seed)

    if not os.path.exists(args.data_root):
        print(f"[ERROR] Data root not found: {args.data_root}")
        sys.exit(1)

    try:
        files = sorted([
            os.path.join(args.data_root, f)
            for f in os.listdir(args.data_root)
            if f.endswith(".npz")
        ])
    except Exception as e:
        print(f"[ERROR] Cannot read data_root: {e}")
        sys.exit(1)

    if not files:
        print(f"[WARN] No .npz files found")
        sys.exit(0)

    files_by_f = defaultdict(list)
    for p in files:
        try:
            f_val = get_f_from_filename(p)
            files_by_f[f_val].append(p)
        except Exception as e:
            print(f"[WARN] Cannot parse f-value from {p}: {e}", flush=True)
            continue

    tracker = ResultTracker(
        results_root=args.results_root,
        exp_name=args.exp_name,
        args=args
    )

    predictions = {}
    total_processed = 0
    total_failed = 0

    das_cutoff = None if args.das_cutoff < 0 else args.das_cutoff

    print(
        f"[Info] Using eta_G={args.eta_G}, eta_H={args.eta_H}, "
        f"alpha={args.alpha}, prune={args.prune}",
        flush=True
    )
    print(f"[Info] Found {len(files)} files across {len(files_by_f)} f-values", flush=True)

    for f in sorted(files_by_f):
        paths = files_by_f[f][:args.max_per_f] if args.max_per_f > 0 else files_by_f[f]
        print(f"\n[DAS] Processing f={f} ({len(paths)} files)...", flush=True)

        for path in paths:
            try:
                G_true, A_pred = run_das_on_file(
                    path=path,
                    eta_G=args.eta_G,
                    eta_H=args.eta_H,
                    alpha=args.alpha,
                    prune=args.prune,
                    das_cutoff=das_cutoff,
                    n_splines=args.n_splines,
                    splines_degree=args.splines_degree,
                    min_parents=args.min_parents,
                    max_parents=args.max_parents,
                )
                metrics = dag_metrics(G_true, A_pred)
                fname = os.path.basename(path)
                tracker.log(metrics, f=f, filename=fname)
                if args.save_preds:
                    predictions[fname] = A_pred.astype(np.int8)
                total_processed += 1
            except Exception as e:
                print(f"[FAIL] {os.path.basename(path)}: {e}", flush=True)
                total_failed += 1

    tracker.finalize()

    if args.save_preds and predictions:
        save_path = os.path.join(args.results_root, args.exp_name, "predictions.npz")
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_predictions_incremental(save_path, predictions, verbose=True)
            print(f"\n[Info] Predictions saved to {save_path}", flush=True)
        except Exception as e:
            print(f"[Error] Failed to save predictions: {e}", flush=True)

    print(f"\n[Summary] {total_processed} succeeded, {total_failed} failed", flush=True)


if __name__ == "__main__":
    main()
