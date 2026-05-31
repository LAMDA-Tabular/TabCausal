#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
from collections import defaultdict
import sys
from pathlib import Path
import re

# causal-learn
from causallearn.search.FCMBased.lingam import DirectLiNGAM

# Benchmark utils
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import dag_metrics, prob_metrics, ResultTracker, save_predictions_incremental

FIXED_THRESHOLD = 0.5

# =====================================================
# Thresholding
# =====================================================

def apply_threshold(W_pred: np.ndarray) -> np.ndarray:
    """Binarize a weighted adjacency matrix with the fixed release threshold."""
    W_abs = np.abs(W_pred)
    B = (W_abs >= FIXED_THRESHOLD).astype(float)
    np.fill_diagonal(B, 0)
    return B

# =====================================================
# LiNGAM Runner
# =====================================================

def run_lingam_on_file(path, seed):
    """ DirectLiNGAM， """
    data = np.load(path)
    x, g = data["x"], data["g"]
    mask = data["mask"].astype(bool) if "mask" in data else np.ones(g.shape[0], dtype=bool)
    if x.ndim == 2:
        x = np.stack([x, np.zeros_like(x)], axis=-1)

    X = x[..., 0][:, mask].astype(float)
    G_true = g[mask][:, mask].astype(int)

    try:
        model = DirectLiNGAM(random_state=seed)
        model.fit(X)
        W = model.adjacency_matrix_.T  # Transpose to match convention
        np.fill_diagonal(W, 0)
    except Exception as e:
        print(f"[WARN] LiNGAM failed: {e}")
        W = np.zeros_like(G_true, dtype=float)

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
    parser = argparse.ArgumentParser("Run DirectLiNGAM baseline")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--max_per_f", type=int, default=-1)
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
        print(f"\n[LiNGAM] Processing f={f} ({len(paths)} files)...", flush=True)

        # ========== Inference Phase: Get all weight matrices ==========
        all_data = []  # List of (path, G_true, W)
        
        for path in paths:
            try:
                G_true, W = run_lingam_on_file(path, args.seed)
                all_data.append((path, G_true, W))
            except Exception as e:
                print(f"[FAIL] Inference failed on {os.path.basename(path)}: {e}", flush=True)
                import traceback
                traceback.print_exc()
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
