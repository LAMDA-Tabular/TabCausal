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
# IGSP  
# =====================================================
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
if os.environ.get("IGSP_ROOT"):
    sys.path.insert(0, os.environ["IGSP_ROOT"])
from igsp import run_igsp, run_ut_igsp

sys.path.insert(0, str(_HERE))
from metrics import dag_metrics, ResultTracker, save_predictions_incremental
from preprocess import standardize_x

# =====================================================
# Utils
# =====================================================

def format_avici_to_igsp(X: np.ndarray, interv: np.ndarray):
    """
      IGSP  :
      - data_pd: DataFrame(X)
      - targets_pd: DataFrame(intervention mask 0/1)
      - regimes:   mask   regime id，observational   0
    """
    n_samples, n_vars = X.shape
    interv_int = interv.astype(int)

    unique_masks = np.unique(interv_int, axis=0)
    obs_mask = tuple([0] * n_vars)

    mask_to_id = {}
    cur = 0

    # Ensure observational regime is 0
    if any(tuple(r) == obs_mask for r in unique_masks):
        mask_to_id[obs_mask] = 0
        cur = 1

    for r in unique_masks:
        t = tuple(r)
        if t not in mask_to_id:
            mask_to_id[t] = cur
            cur += 1

    regimes = np.array([mask_to_id[tuple(r)] for r in interv_int], dtype=int)

    return (
        pd.DataFrame(X),
        pd.DataFrame(interv_int),
        regimes
    )

def run_igsp_on_file(path: str, model: str, alpha: float, alpha_inv: float, ci_test: str, seed: int):
    """ IGSP， DAG"""
    data = np.load(path)
    x, g = data["x"], data["g"]
    mask = data["mask"].astype(bool) if "mask" in data else np.ones(g.shape[0], dtype=bool)
    if x.ndim == 2:
        x = np.stack([x, np.zeros_like(x)], axis=-1)

    X = x[..., 0][:, mask].astype(float)
    interv = x[..., 1][:, mask].astype(float)
    G_true = g[mask][:, mask].astype(int)
    X = standardize_x(X)

    df_X, df_targets, regimes = format_avici_to_igsp(X, interv)

    # Set random seed for reproducibility
    np.random.seed(seed)

    try:
        if model == "IGSP":
            A_est, _, _ = run_igsp(
                df_X, targets=df_targets, regimes=regimes,
                alpha=alpha, alpha_inv=alpha_inv, ci_test=ci_test
            )
        else:  # UTIGSP
            A_est, _, _, _ = run_ut_igsp(
                df_X, targets=df_targets, regimes=regimes,
                alpha=alpha, alpha_inv=alpha_inv, ci_test=ci_test
            )
        
        A_est = np.array(A_est).astype(int)
    except Exception as e:
        print(f"[WARN] IGSP failed: {e}")
        A_est = np.zeros_like(G_true)

    return G_true, A_est

def get_f_from_filename(p: str) -> int:
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
    parser = argparse.ArgumentParser("Run IGSP/UT-IGSP baseline")

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--max_per_f", type=int, default=-1)
    parser.add_argument("--model", type=str, default="IGSP", choices=["IGSP", "UTIGSP"])
    parser.add_argument("--ci_test", type=str, default="gaussian")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--results_root", type=str, required=True)

    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--alpha_inv", type=float, default=1e-3)

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
    
    print(f"[Info] Model: {args.model}", flush=True)
    print(f"[Info] Using alpha={args.alpha:.4f}, alpha_inv={args.alpha_inv:.4f}", flush=True)
    print(f"[Info] Found {len(files)} files across {len(files_by_f)} f-values", flush=True)

    total_processed = 0
    total_failed = 0

    for f in sorted(files_by_f):
        paths = files_by_f[f][:args.max_per_f] if args.max_per_f > 0 else files_by_f[f]
        print(f"\n[IGSP] Processing f={f} ({len(paths)} files)...", flush=True)

        for path in paths:
            try:
                G_true, A_pred = run_igsp_on_file(
                    path, args.model, args.alpha, args.alpha_inv, args.ci_test, args.seed
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
            #   np.savez_compressed   save_predictions_incremental
            save_predictions_incremental(save_path, predictions, verbose=True)
            print(f"\n[Info] Predictions saved to {save_path}", flush=True)
        except Exception as e:
            print(f"[Error] Failed to save predictions: {e}", flush=True)
    
    print(f"\n[Summary] {total_processed} succeeded, {total_failed} failed", flush=True)

if __name__ == "__main__":
    main()
