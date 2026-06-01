#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
from collections import defaultdict
import sys
from pathlib import Path
import re

# utils
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from metrics import dag_metrics, prob_metrics, ResultTracker, save_predictions_incremental
from preprocess import standardize_x

FIXED_THRESHOLD = 0.5

# NODAGS
# The Bicycle/NoDAGS package lives under the benchmark repo.
if os.environ.get("BICYCLE_ROOT"):
    sys.path.insert(0, os.environ["BICYCLE_ROOT"])
from bicycle.nodags_files.nodags import resflow_train_test_wrapper

# =====================================================
# Thresholding
# =====================================================

def apply_threshold(W: np.ndarray) -> np.ndarray:
    """Binarize a weighted adjacency matrix with the fixed release threshold."""
    W_abs = np.abs(W)
    B = (W_abs >= FIXED_THRESHOLD).astype(float)
    np.fill_diagonal(B, 0)
    return B

# =====================================================
# Data Preparation
# =====================================================

def process_avici_data_for_nodags(X, interv, min_samples):
    """
     AVICI NODAGS 
    
    Returns:
        D: list of numpy arrays (each is a group of observations)
        T: list of intervention targets (None for observational, int for single-target)
    """
    groups = defaultdict(list)
    
    for i in range(len(X)):
        idx = np.where(interv[i] == 1)[0]
        
        if len(idx) == 0:
            # Observational
            tid = None
        elif len(idx) == 1:
            # Single-target intervention
            tid = int(idx[0])
        else:
            # Multi-target intervention (skip for NODAGS)
            continue
        
        groups[tid].append(X[i])

    D, T = [], []
    
    # Sort: observational first, then by target index
    for tid in sorted(groups.keys(), key=lambda x: (x is not None, x)):
        if len(groups[tid]) >= min_samples:
            arr = np.stack(groups[tid]).astype(np.float32)
            D.append(arr)
            T.append(tid)

    if len(D) < 2:
        return None, None
    
    return D, T

# =====================================================
# NODAGS Runner
# =====================================================

def run_nodags_on_file(path, args):
    """ NODAGS， """
    data = np.load(path)
    x, g = data["x"], data["g"]
    mask = data["mask"].astype(bool) if "mask" in data else np.ones(g.shape[0], dtype=bool)
    if x.ndim == 2:
        x = np.stack([x, np.zeros_like(x)], axis=-1)

    X = x[..., 0][:, mask]
    interv = x[..., 1][:, mask]
    G_true = g[mask][:, mask].astype(int)
    X = standardize_x(X).astype(np.float32)

    # Prepare data
    D, T = process_avici_data_for_nodags(X, interv, args.min_samples)
    if D is None:
        raise RuntimeError("Insufficient interventional data for NODAGS")

    # Initialize model
    model = resflow_train_test_wrapper(
        n_nodes=G_true.shape[0],
        lambda_c=args.lambda_c,
        n_hidden=args.n_hidden,
        lr=args.lr,
        epochs=args.epochs,
        fun_type=args.fun_type,
        batch_size=args.batch_size,
        v=False,
        l1_reg=True,
        act_fun="none",
        optim="adam",
        inline=True,
        lin_logdet=False,
        full_input=False
    )

    # Train
    model.train(D, T, batch_size=args.batch_size)
    
    # Get adjacency matrix
    W = model.get_adjacency()
    np.fill_diagonal(W, 0)

    return G_true, W

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
    parser = argparse.ArgumentParser("Run NODAGS baseline")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--max_per_f", type=int, default=-1)
    
    # NODAGS parameters
    parser.add_argument("--lambda_c", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n_hidden", type=int, default=0)
    parser.add_argument("--fun_type", type=str, default="lin-mlp")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--min_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--results_root", type=str, required=True)

    parser.add_argument("--save_preds", action="store_true")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
    
    print(f"[Info] Using fixed threshold={FIXED_THRESHOLD}", flush=True)
    print(f"[Info] Found {len(files)} files across {len(files_by_f)} f-values", flush=True)

    total_processed = 0
    total_failed = 0

    for f in sorted(files_by_f):
        paths = files_by_f[f][:args.max_per_f] if args.max_per_f > 0 else files_by_f[f]
        print(f"\n[NODAGS] Processing f={f} ({len(paths)} files)...", flush=True)

        for path in paths:
            try:
                G_true, W = run_nodags_on_file(path, args)
                B = apply_threshold(W)
                metrics = dag_metrics(G_true, B.astype(int))
                fname = os.path.basename(path)
                tracker.log(metrics, f=f, filename=fname)
                if args.save_preds:
                    predictions[fname] = W.astype(np.float16)
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
