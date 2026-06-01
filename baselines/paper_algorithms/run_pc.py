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
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz, chisq

# utils
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import dag_metrics, ResultTracker, save_predictions_incremental
from preprocess import standardize_x

# =====================================================
# CPDAG -> DAG  （ ）
# =====================================================

def cpdag_to_dag_strategy(cg, strategy="liberal"):
    """
    Convert CPDAG to DAG using deterministic strategies.
    
    Strategies:
    - conservative: Keep only compelled edges
    - liberal: Orient undirected edges by node index (i<j → i→j)
    - max_edges: Same as liberal (can be enhanced with other heuristics)
    """
    G = cg.G.graph
    d = G.shape[0]
    A = np.zeros((d, d), dtype=int)
    processed = set()

    for i in range(d):
        for j in range(d):
            if i == j:
                continue

            pair = tuple(sorted((i, j)))
            if pair in processed:
                continue
            processed.add(pair)

            # Compelled direction: i → j
            # (tail at i: -1, arrow at j: 1)
            if G[j, i] == 1 and G[i, j] == -1:
                A[i, j] = 1
            
            # Compelled direction: j → i
            elif G[i, j] == 1 and G[j, i] == -1:
                A[j, i] = 1
            
            # Undirected: i - j
            elif G[i, j] == -1 and G[j, i] == -1:
                if strategy == "conservative":
                    # Don't add undirected edges
                    pass
                elif strategy == "liberal":
                    # Orient by index
                    if i < j:
                        A[i, j] = 1
                    else:
                        A[j, i] = 1
                elif strategy == "max_edges":
                    # Same as liberal
                    if i < j:
                        A[i, j] = 1
                    else:
                        A[j, i] = 1

    return A

# =====================================================
# Runner
# =====================================================

def run_pc_on_file(path, alpha, indep_test_name, orientation_strategy="liberal"):
    """ PC ， DAG"""
    print(f"[PC CHILD] Loading file: {path}", flush=True)
    data = np.load(path)
    x, g = data["x"], data["g"]
    mask = data["mask"].astype(bool) if "mask" in data else np.ones(g.shape[0], dtype=bool)
    if x.ndim == 2:
        x = np.stack([x, np.zeros_like(x)], axis=-1)
    print(f"[PC CHILD] x.shape={x.shape}, g.shape={g.shape}, mask.shape={mask.shape}, mask.sum={mask.sum()}", flush=True)

    X = x[..., 0][:, mask].astype(float)
    G_true = g[mask][:, mask].astype(int)
    X = standardize_x(X)
    print(f"[PC CHILD] X.shape={X.shape}, G_true.shape={G_true.shape}", flush=True)

    # Resolve CI test
    if indep_test_name == "fisherz":
        indep_test = fisherz
    elif indep_test_name == "chisq":
        indep_test = chisq
    else:
        raise ValueError(f"Unsupported indep_test: {indep_test_name}")

    try:
        cg = pc(
            X,
            alpha=alpha,
            indep_test=indep_test,
            stable=True,
            show_progress=False
        )
        A_pred = cpdag_to_dag_strategy(cg, strategy=orientation_strategy)
        print(f"[PC CHILD] PC run completed for {os.path.basename(path)}; A_pred.shape={A_pred.shape}", flush=True)
    except Exception as e:
        print(f"[WARN] PC failed: {e}", flush=True)
        A_pred = np.zeros_like(G_true)

    return G_true, A_pred

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
    parser = argparse.ArgumentParser("Run PC baseline")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--max_per_f", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--results_root", type=str, required=True)

    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--indep_test", type=str, default="fisherz",
                        choices=["fisherz", "chisq"])
    parser.add_argument(
        "--orientation_strategy",
        type=str,
        choices=["conservative", "liberal", "max_edges"],
        default="liberal",
        help="CPDAG to DAG orientation strategy"
    )

    parser.add_argument("--save_preds", action="store_true")

    args = parser.parse_args()

    print("=" * 80, flush=True)
    print("[PC CHILD] Script started", flush=True)
    print(f"[PC CHILD] sys.executable = {sys.executable}", flush=True)
    print(f"[PC CHILD] cwd = {os.getcwd()}", flush=True)
    print(f"[PC CHILD] data_root = {args.data_root}", flush=True)
    print(f"[PC CHILD] results_root = {args.results_root}", flush=True)
    print(f"[PC CHILD] exp_name = {args.exp_name}", flush=True)
    print("[PC CHILD] evaluation = direct", flush=True)
    print(f"[PC CHILD] save_preds = {args.save_preds}", flush=True)
    print("=" * 80, flush=True)

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

    print(f"[PC CHILD] Found {len(files)} npz files", flush=True)
    for i, p in enumerate(files[:5]):
        print(f"[PC CHILD] sample file[{i}] = {p}", flush=True)

    # Group by f-value
    files_by_f = defaultdict(list)
    for p in files:
        try:
            f_val = get_f_from_filename(p)
            files_by_f[f_val].append(p)
        except Exception as e:
            print(f"[WARN] Cannot parse f-value from {p}: {e}")
            continue

    print(f"[PC CHILD] Initializing ResultTracker...", flush=True)
    tracker = ResultTracker(
        results_root=args.results_root,
        exp_name=args.exp_name,
        args=args
    )
    print(f"[PC CHILD] ResultTracker initialized", flush=True)
    print(f"[PC CHILD] expected result dir = {os.path.join(args.results_root, args.exp_name)}", flush=True)

    predictions = {}
    
    print(f"[Info] Using alpha={args.alpha}, test={args.indep_test}, orient={args.orientation_strategy}", flush=True)
    print(f"[Info] Found {len(files)} files across {len(files_by_f)} f-values", flush=True)

    total_processed = 0
    total_failed = 0

    for f in sorted(files_by_f):
        paths = files_by_f[f][:args.max_per_f] if args.max_per_f > 0 else files_by_f[f]
        print(f"\n[PC] Processing f={f} ({len(paths)} files)...", flush=True)

        for path in paths:
            try:
                G_true, A_pred = run_pc_on_file(path, args.alpha, args.indep_test, args.orientation_strategy)
                metrics = dag_metrics(G_true, A_pred)
                fname = os.path.basename(path)
                print(f"[PC CHILD] Logging result for {fname}, metrics keys = {list(metrics.keys())}", flush=True)
                tracker.log(metrics, f=f, filename=fname)
                print(f"[PC CHILD] Logged result for {fname}", flush=True)
                if args.save_preds:
                    predictions[fname] = A_pred.astype(np.int8)
                total_processed += 1
            except Exception as e:
                print(f"[FAIL] {os.path.basename(path)}: {e}", flush=True)
                total_failed += 1

    print(f"[PC CHILD] Calling tracker.finalize()", flush=True)
    try:
        tracker.finalize()
        print(f"[PC CHILD] tracker.finalize() done", flush=True)
    except Exception as e:
        print(f"[PC CHILD] tracker.finalize() failed: {e}", flush=True)
        raise

    result_dir = os.path.join(args.results_root, args.exp_name)
    print(f"[PC CHILD] result_dir exists = {os.path.exists(result_dir)}", flush=True)
    if os.path.exists(result_dir):
        try:
            print(f"[PC CHILD] result_dir files = {sorted(os.listdir(result_dir))}", flush=True)
        except Exception as e:
            print(f"[PC CHILD] failed to list result_dir: {e}", flush=True)

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
    try:
        main()
    except Exception as e:
        import traceback
        print(f"[PC CHILD FATAL] {e}", flush=True)
        traceback.print_exc()
        raise
