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

# ================================
#  
# ================================
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def _prepend_avici_candidates() -> None:
    """Prefer the official AVICI package over the data-generator module."""
    candidates = [
        os.environ.get("AVICI_ROOT"),
        _HERE / "avici_official",
    ]
    for candidate in reversed([c for c in candidates if c]):
        root = Path(candidate).expanduser()
        if (root / "avici" / "__init__.py").exists():
            sys.path.insert(0, str(root.resolve()))


_prepend_avici_candidates()

import avici
from metrics import prob_metrics, ResultTracker, save_predictions_incremental

FIXED_THRESHOLD = 0.5

# ================================
# Thresholding
# ================================
def apply_threshold(P_pred: np.ndarray) -> np.ndarray:
    """Binarize an edge-probability matrix with the fixed release threshold."""
    P_bin = (P_pred >= FIXED_THRESHOLD).astype(float)
    np.fill_diagonal(P_bin, 0)
    return P_bin

# ================================
# Inference
# ================================
def run_avici_on_file(path, model):
    """
    Run AVICI inference, returns RAW probability matrix.
    """
    data = np.load(path)
    x, g = data["x"], data["g"]
    mask = data["mask"].astype(bool) if "mask" in data else np.ones(g.shape[0], dtype=bool)
    if x.ndim == 2:
        x = np.stack([x, np.zeros_like(x)], axis=-1)

    X = x[..., 0][:, mask].astype(float)
    interv = x[..., 1][:, mask].astype(float)
    G_true = g[mask][:, mask].astype(int)

    # AVICI Forward
    P_pred = model(x=X, interv=interv)
    np.fill_diagonal(P_pred, 0)

    return G_true, P_pred

# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser("Run AVICI baseline")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--max_per_f", type=int, default=-1)
    parser.add_argument("--use_official", action="store_true")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--results_root", type=str, required=True)

    # For compatibility with benchmark script
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Ignored for AVICI (uses official pretrained model)")
    parser.add_argument(
        "--avici_cache_path",
        type=str,
        default=None,
        help="Optional cache root for official AVICI pretrained weights.",
    )

    # Save predictions
    parser.add_argument("--save_preds", action="store_true", help="Save predictions to .npz")

    args = parser.parse_args()

    # Validate args
    if not args.use_official:
        raise ValueError("run_avici.py only supports --use_official flag.")

    # ================================
    # 1. Load Model
    # ================================
    print("[AVICI] Loading official pretrained model (scm-v0)...", flush=True)
    try:
        if not hasattr(avici, "load_pretrained"):
            raise AttributeError(
                f"Imported avici from {getattr(avici, '__file__', '<namespace>')}, "
                "but it has no load_pretrained(). Set AVICI_ROOT or --avici-root "
                "to the official AVICI package root."
            )
        load_kwargs = {"download": "scm-v0"}
        cache_path = args.avici_cache_path or os.environ.get("AVICI_CACHE_PATH")
        if cache_path:
            load_kwargs["cache_path"] = cache_path
        model = avici.load_pretrained(**load_kwargs)
    except Exception as e:
        print(f"[ERROR] Failed to load AVICI model: {e}", flush=True)
        sys.exit(1)

    # ================================
    # 2. Collect Data
    # ================================
    if not os.path.exists(args.data_root):
        print(f"[ERROR] Data root does not exist: {args.data_root}", flush=True)
        sys.exit(1)

    try:
        all_files = sorted([
            os.path.join(args.data_root, f)
            for f in os.listdir(args.data_root)
            if f.endswith(".npz")
        ])
    except Exception as e:
        print(f"[ERROR] Cannot read data_root: {e}", flush=True)
        sys.exit(1)

    if not all_files:
        print(f"[WARN] No .npz files found in {args.data_root}", flush=True)
        sys.exit(0)

    files_by_f = defaultdict(list)
    for path in all_files:
        try:
            match = re.search(r"_f(\d+)_", os.path.basename(path))
            f_val = int(match.group(1)) if match else 0
            files_by_f[f_val].append(path)
        except Exception as e:
            print(f"[WARN] Cannot parse f-value from {path}: {e}", flush=True)
            continue

    tracker = ResultTracker(
        results_root=args.results_root,
        exp_name=args.exp_name,
        args=args
    )

    preds_to_save = {}

    print(f"[Info] Using fixed threshold={FIXED_THRESHOLD}", flush=True)
    print(f"[Info] Found {len(all_files)} files across {len(files_by_f)} f-values", flush=True)

    # ================================
    # 3. Main Execution Loop
    # ================================
    total_processed = 0
    total_failed = 0

    for f in sorted(files_by_f):
        paths = files_by_f[f][:args.max_per_f] if args.max_per_f > 0 else files_by_f[f]
        print(f"\n[Eval] Processing f={f} ({len(paths)} files)...", flush=True)

        # ========== Inference Phase: Get all predictions ==========
        all_data = []  # List of (path, G_true, P_pred)
        
        for path in paths:
            try:
                G_true, P_pred = run_avici_on_file(path, model)
                all_data.append((path, G_true, P_pred))
            except Exception as e:
                print(f"[FAIL] Inference failed on {os.path.basename(path)}: {e}", flush=True)
                total_failed += 1

        if not all_data:
            print(f"[WARN] No successful inference at f={f}", flush=True)
            continue

        for path, G_true, P_pred in all_data:
            P_eval = apply_threshold(P_pred)
            metrics = prob_metrics(G_true, P_eval)
            fname = os.path.basename(path)
            tracker.log(metrics, f=f, filename=fname)
            if args.save_preds:
                preds_to_save[fname] = P_pred.astype(np.float16)
            total_processed += 1

    tracker.finalize()

    # ================================
    # 4. Save Predictions
    # ================================
    if args.save_preds and preds_to_save:
        save_path = os.path.join(
            args.results_root,
            args.exp_name,
            "predictions.npz"
        )
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # ==========   ==========
            save_predictions_incremental(save_path, preds_to_save, verbose=True)
            preds_to_save.clear()
            # ===========================================
            
        except Exception as e:
            print(f"[Error] Failed to save predictions: {e}", flush=True)


    print(f"\n[Summary] {total_processed} succeeded, {total_failed} failed", flush=True)

if __name__ == "__main__":
    main()
