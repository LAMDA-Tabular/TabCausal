#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import numpy as np
from collections import defaultdict
import sys
from pathlib import Path
import re

#  
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from cdis import cdis_from_data
from metrics import dag_metrics, ResultTracker, save_predictions_incremental
from preprocess import standardize_x

# =====================================================
# PAG  
# =====================================================

def reconstruct_pag_matrix(edge_dict, n):
    """  CDIS   PAG   (0=No, 1=Circle, 2=Arrow, 3=Tail)"""
    pag = np.zeros((n, n), dtype=int)
    
    # Directed edges: ->
    for k in ['->', '-->']:
        for i, j in edge_dict.get(k, []):
            if i < n and j < n:
                pag[i, j] = 2  # Arrow
                pag[j, i] = 3  # Tail
    
    # Bidirected: <->
    for k in ['<->']:
        for i, j in edge_dict.get(k, []):
            if i < n and j < n:
                pag[i, j] = 2
                pag[j, i] = 2
    
    # Undirected: --
    for k in ['--', '---']:
        for i, j in edge_dict.get(k, []):
            if i < n and j < n:
                pag[i, j] = 3
                pag[j, i] = 3
    
    # Partially directed: o->
    for k in ['o->', 'o-->']:
        for i, j in edge_dict.get(k, []):
            if i < n and j < n:
                pag[i, j] = 2
                pag[j, i] = 1  # Circle
    
    # Both circles: o-o
    for k in ['o-o']:
        for i, j in edge_dict.get(k, []):
            if i < n and j < n:
                pag[i, j] = 1
                pag[j, i] = 1
    
    return pag

# =====================================================
# PAG   DAG  
# =====================================================

def convert_pag_to_dag_strategy(PAG_mat, strategy="conservative"):
    """
      PAG   DAG， ：
    
    - conservative:  
    - liberal:  （i<j i→j）
    - bidirectional:  （ ）
    - max_edges:  （ ）
    """
    n = PAG_mat.shape[0]
    A_pred = np.zeros((n, n), dtype=int)
    processed_pairs = set()

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            pair = tuple(sorted((i, j)))
            if pair in processed_pairs:
                continue
            processed_pairs.add(pair)

            val_ij = PAG_mat[i, j]
            val_ji = PAG_mat[j, i]

            # Case 1:   i->j
            if val_ij == 2 and val_ji == 3:
                A_pred[i, j] = 1
            
            # Case 2:   j->i
            elif val_ij == 3 and val_ji == 2:
                A_pred[j, i] = 1
            
            # Case 3:  
            elif val_ij != 0 or val_ji != 0:
                if strategy == "conservative":
                    #  
                    pass
                
                elif strategy == "liberal":
                    #  ：i < j   i→j
                    if i < j:
                        A_pred[i, j] = 1
                    else:
                        A_pred[j, i] = 1
                
                elif strategy == "max_edges":
                    #  （ DAG ， ）
                    #  
                    if i < j:
                        A_pred[i, j] = 1
                    else:
                        A_pred[j, i] = 1

    return A_pred

# =====================================================
#  
# =====================================================

def build_cdis_data_list(X, interv, min_samples=10):
    n_vars = X.shape[1]
    interv_int = interv.astype(int)
    unique_rows, inverse = np.unique(interv_int, axis=0, return_inverse=True)

    data_list = []
    obs_mask = tuple([0] * n_vars)
    unique_tuples = [tuple(r) for r in unique_rows]

    try:
        obs_idx = unique_tuples.index(obs_mask)
        data_list.append(X[inverse == obs_idx])
    except ValueError:
        raise ValueError("CDIS requires observational data (all-zero mask).")

    for i, mask in enumerate(unique_tuples):
        if mask == obs_mask:
            continue
        group_data = X[inverse == i]
        if len(group_data) >= min_samples:
            data_list.append(group_data)

    return data_list

def run_cdis_on_file(path, min_samples):
    """
     CDIS， PAG 
    
     ：CDIS alpha ， 
    """
    data = np.load(path)
    x, g = data["x"], data["g"]
    mask = data["mask"].astype(bool) if "mask" in data else np.ones(g.shape[0], dtype=bool)
    if x.ndim == 2:
        x = np.stack([x, np.zeros_like(x)], axis=-1)

    X = x[..., 0][:, mask].astype(float)
    interv = x[..., 1][:, mask].astype(float)
    G_true = g[mask][:, mask].astype(int)
    n = G_true.shape[0]
    X = standardize_x(X)

    data_list = build_cdis_data_list(X, interv, min_samples)

    try:
        # CDIS alpha ， 
        edge_raw = cdis_from_data(data_list)
        pag = reconstruct_pag_matrix(edge_raw.get("final_PAG", {}), n)
    except Exception as e:
        print(f"[WARN] CDIS failed: {e}")
        pag = np.zeros((n, n), dtype=int)

    return G_true, pag

# =====================================================
# Main
# =====================================================

def main():
    parser = argparse.ArgumentParser("Run CDIS baseline")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--max_per_f", type=int, default=-1)
    parser.add_argument("--min_samples", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--results_root", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--orientation_strategy",
        type=str,
        choices=["conservative", "liberal", "max_edges"],
        default="liberal",
        help="PAG to DAG orientation strategy"
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
    for path in files:
        try:
            match = re.search(r"_f(\d+)_", os.path.basename(path))
            f_val = int(match.group(1)) if match else 0
            files_by_f[f_val].append(path)
        except Exception as e:
            print(f"[WARN] Cannot parse f-value from {path}: {e}")
            continue

    tracker = ResultTracker(
        results_root=args.results_root,
        exp_name=args.exp_name,
        args=args
    )

    predictions = {}
    
    print(f"[Info] Using orientation strategy: {args.orientation_strategy}", flush=True)
    print(f"[Info] Found {len(files)} files across {len(files_by_f)} f-values", flush=True)

    total_processed = 0
    total_failed = 0

    for f in sorted(files_by_f):
        paths = files_by_f[f][:args.max_per_f] if args.max_per_f > 0 else files_by_f[f]
        print(f"\n[CDIS] Processing f={f} ({len(paths)} files)...", flush=True)

        # ========== Inference Phase: Get all PAGs ==========
        all_data = []  # List of (path, G_true, PAG)
        
        for path in paths:
            try:
                G_true, PAG = run_cdis_on_file(path, args.min_samples)
                all_data.append((path, G_true, PAG))
            except Exception as e:
                print(f"[FAIL] {os.path.basename(path)}: {e}", flush=True)
                total_failed += 1

        if not all_data:
            print(f"[WARN] No successful inference at f={f}")
            continue

        for path, G_true, PAG in all_data:
            A_pred = convert_pag_to_dag_strategy(PAG, strategy=args.orientation_strategy)
            metrics = dag_metrics(G_true, A_pred)
            fname = os.path.basename(path)
            tracker.log(metrics, f=f, filename=fname)
            if args.save_preds:
                predictions[fname] = A_pred.astype(np.int8)
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
