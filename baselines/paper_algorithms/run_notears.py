#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
from collections import defaultdict
import sys
from pathlib import Path
import re

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# NOTEARS linear
from notears.linear import notears_linear

# utils
from metrics import dag_metrics, prob_metrics, ResultTracker, save_predictions_incremental

FIXED_THRESHOLD = 0.5

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
# Runner
# =====================================================

def run_notears_on_file(path, lambda1):
    """ NOTEARS-Linear， """
    data = np.load(path)
    x, g = data["x"], data["g"]
    mask = data["mask"].astype(bool) if "mask" in data else np.ones(g.shape[0], dtype=bool)
    if x.ndim == 2:
        x = np.stack([x, np.zeros_like(x)], axis=-1)

    X = x[..., 0][:, mask].astype(float)
    G_true = g[mask][:, mask].astype(int)

    try:
        W = notears_linear(
            X,
            lambda1=lambda1,
            loss_type="l2",
            w_threshold=0.0,  # Keep continuous weights
            h_tol=1e-8,
            max_iter=100
        )
    except Exception as e:
        print(f"[WARN] NOTEARS failed: {e}")
        W = np.zeros_like(G_true, dtype=float)

    return G_true, W

def get_f_from_filename(p):
    """ f """
    try:
        match = re.search(r"_f(\d+)_", os.path.basename(p))
        if match:
            return int(match.group(1))
        match = re.search(r'f(\d+)', os.path.basename(p))
        if match:
            return int(match.group(1))
        return 0
    except:
        return 0

# =====================================================
# Main
# =====================================================

def main():
    parser = argparse.ArgumentParser("Run NOTEARS-Linear baseline")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--max_per_f", type=int, default=-1)
    parser.add_argument("--lambda1", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--results_root", type=str, required=True)

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
    
    print(f"[Info] Using fixed threshold={FIXED_THRESHOLD}", flush=True)
    print(f"[Info] Found {len(files)} files across {len(files_by_f)} f-values", flush=True)

    total_processed = 0
    total_failed = 0

    for f in sorted(files_by_f):
        paths = files_by_f[f][:args.max_per_f] if args.max_per_f > 0 else files_by_f[f]
        print(f"\n[NOTEARS] Processing f={f} ({len(paths)} files)...", flush=True)

        # ========== Inference Phase ==========
        all_data = []
        
        for path in paths:
            try:
                G_true, W = run_notears_on_file(path, args.lambda1)
                all_data.append((path, G_true, W))
            except Exception as e:
                print(f"[FAIL] Inference failed on {os.path.basename(path)}: {e}", flush=True)
                total_failed += 1

        if not all_data:
            print(f"[WARN] No successful inference at f={f}")
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
