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

# =====================================================
#  
# =====================================================
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
if os.environ.get("GIES_ROOT"):
    sys.path.insert(0, os.environ["GIES_ROOT"])

from metrics import dag_metrics, ResultTracker, save_predictions_incremental
from preprocess import standardize_x
from gies import GIES

# =====================================================
# CPDAG → DAG  （ ）
# =====================================================

def cpdag_to_dag_strategy(G_raw, strategy="liberal"):
    """
     CPDAG DAG， ：
    
    - conservative:  ， 
    - liberal:  i<j 
    - max_edges:  
    """
    n = G_raw.shape[0]
    A = np.zeros((n, n), dtype=int)
    processed = set()

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            pair = tuple(sorted((i, j)))
            if pair in processed:
                continue
            processed.add(pair)

            # Case 1: i -> j ( )
            if G_raw[i, j] and not G_raw[j, i]:
                A[i, j] = 1
            
            # Case 2: j -> i ( )
            elif G_raw[j, i] and not G_raw[i, j]:
                A[j, i] = 1
            
            # Case 3: i - j ( )
            elif G_raw[i, j] and G_raw[j, i]:
                if strategy == "conservative":
                    #  
                    pass
                elif strategy == "liberal":
                    #  
                    if i < j:
                        A[i, j] = 1
                    else:
                        A[j, i] = 1
                elif strategy == "max_edges":
                    #  liberal （ ）
                    if i < j:
                        A[i, j] = 1
                    else:
                        A[j, i] = 1

    return A

# =====================================================
# GIES core
# =====================================================

def run_gies_on_file(path, lambda_gies, orientation_strategy="liberal"):
    """ GIES， DAG"""
    data = np.load(path)
    x, g = data["x"], data["g"]
    mask = data["mask"].astype(bool) if "mask" in data else np.ones(g.shape[0], dtype=bool)
    if x.ndim == 2:
        x = np.stack([x, np.zeros_like(x)], axis=-1)

    X = x[..., 0][:, mask].astype(float)
    interv = x[..., 1][:, mask].astype(float)
    G_true = g[mask][:, mask].astype(int)
    X = standardize_x(X)

    df_X = pd.DataFrame(X)
    n_samples = X.shape[0]

    # BIC penalty
    if lambda_gies < 0:
        lambda_gies = np.log(n_samples) / 2.0

    # Prepare intervention targets
    targets = []
    max_len = 0
    for row in interv:
        idx = np.where(row > 0.5)[0]
        targets.append(idx)
        max_len = max(max_len, len(idx))

    if max_len == 0:
        targets_pd = None
    else:
        T = np.full((n_samples, max_len), np.nan)
        for i, idx in enumerate(targets):
            if len(idx):
                T[i, :len(idx)] = idx
        targets_pd = pd.DataFrame(T)

    # Run GIES
    gies = GIES(score="int", verbose=False)
    try:
        G_cpdag = np.array(
            gies._run_gies(df_X, targets=targets_pd, lambda_gies=lambda_gies)
        ).astype(int)
    except Exception as e:
        print(f"[WARN] GIES failed: {e}")
        G_cpdag = np.zeros_like(G_true)

    # Convert CPDAG to DAG
    A_pred = cpdag_to_dag_strategy(G_cpdag, strategy=orientation_strategy)

    return G_true, A_pred

def get_f_from_filename(p):
    """ f """
    try:
        match = re.search(r"_f(\d+)_", os.path.basename(p))
        if match:
            return int(match.group(1))
        parts = os.path.basename(p).split("_")
        if len(parts) > 1 and parts[1].startswith("f"):
            return int(parts[1][1:])
        return 0
    except:
        return 0

# =====================================================
# Main
# =====================================================

def main():
    parser = argparse.ArgumentParser("Run GIES baseline")

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--max_per_f", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--results_root", type=str, required=True)

    parser.add_argument(
        "--lambda_gies",
        type=float,
        default=-1.0,  # -1 means use BIC
        help="GIES penalty coefficient; -1 means BIC"
    )
    parser.add_argument(
        "--orientation_strategy",
        type=str,
        choices=["conservative", "liberal", "max_edges"],
        default="liberal",
        help="CPDAG to DAG orientation strategy"
    )

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

    # Group by f-value
    files_by_f = defaultdict(list)
    for p in files:
        try:
            f_val = get_f_from_filename(p)
            files_by_f[f_val].append(p)
        except Exception as e:
            print(f"[WARN] Cannot parse f-value from {p}: {e}")
            continue

    tracker = ResultTracker(
        results_root=args.results_root,
        exp_name=args.exp_name,
        args=args
    )

    predictions = {}
    
    print(f"[Info] Using lambda={args.lambda_gies}, orientation={args.orientation_strategy}", flush=True)
    print(f"[Info] Found {len(files)} files across {len(files_by_f)} f-values", flush=True)

    total_processed = 0
    total_failed = 0

    for f in sorted(files_by_f):
        paths = files_by_f[f][:args.max_per_f] if args.max_per_f > 0 else files_by_f[f]
        print(f"\n[GIES] Processing f={f} ({len(paths)} files)...", flush=True)

        for path in paths:
            try:
                G_true, A_pred = run_gies_on_file(path, args.lambda_gies, args.orientation_strategy)
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
            #   np.savez_compressed   save_predictions_incremental
            save_predictions_incremental(save_path, predictions, verbose=True)
            print(f"\n[Info] Predictions saved to {save_path}", flush=True)
        except Exception as e:
            print(f"[Error] Failed to save predictions: {e}", flush=True)
    
    print(f"\n[Summary] {total_processed} succeeded, {total_failed} failed", flush=True)

if __name__ == "__main__":
    main()
