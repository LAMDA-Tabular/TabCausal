#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DCDI official adapter.

The defaults here are chosen to be close to the official perfect-known
intervention example: DCDI-DSF, long training, GPU when available, and no
silent partial-result export after training errors.
"""

import os
import sys
import argparse
import numpy as np
import torch
import csv
from collections import defaultdict
import re
import tempfile
import shutil
import signal
from pathlib import Path

# =====================================================
# 1. Metric imports with lightweight fallbacks
# =====================================================
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
try:
    from metrics import dag_metrics, ResultTracker, save_predictions_incremental
    from preprocess import standardize_x
except ImportError:
    print("[Warn] Could not import shared metrics; using lightweight fallbacks.")
    def dag_metrics(true, pred): 
        # Minimal SHD fallback.
        diff = np.abs(true - pred)
        return {'shd': np.sum(diff), 'fdr': 0, 'tpr': 0}
    
    class ResultTracker:
        def __init__(self, *args, **kwargs): self.exp_dir = kwargs.get('results_root', '.')
        def log(self, *args, **kwargs): pass
        def finalize(self): pass
    
    def save_predictions_incremental(*args, **kwargs): pass

    def standardize_x(X, *, clip=10.0, eps=1e-8):
        X = np.asarray(X, dtype=float)
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        std = np.where(std < eps, 1.0, std)
        X = (X - mean) / std
        return np.clip(X, -float(clip), float(clip))

# =====================================================
# 2. Environment helpers
# =====================================================
def _env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return int(default)
    try:
        return int(value)
    except Exception:
        return int(default)


def _env_float(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


class DCDITimeLimitReached(TimeoutError):
    """Raised inside the DCDI training loop when the per-graph time budget expires."""


def convert_npz_to_official_format(npz_path, output_dir):
    data = np.load(npz_path)
    x = data["x"]           
    g = data["g"]           
    mask = data["mask"].astype(bool) if "mask" in data else np.ones(data["g"].shape[0], dtype=bool)
    
    X_vals = x[..., 0][:, mask].astype(np.float32)
    if _env_bool("DCDI_PRESTANDARDIZE", False):
        X_vals = standardize_x(X_vals).astype(np.float32)
    I_flags = x[..., 1][:, mask]     
    G_true = g[mask][:, mask]        
    
    n_samples, n_vars = X_vals.shape
    
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "DAG.npy"), G_true)
    # Write common observational/interventional file names for DCDI variants.
    np.save(os.path.join(output_dir, "data.npy"), X_vals)
    np.save(os.path.join(output_dir, "data_interv.npy"), X_vals)
    
    unique_interventions, regime_ids = np.unique(I_flags, axis=0, return_inverse=True)
    np.savetxt(os.path.join(output_dir, "regime.csv"), regime_ids, delimiter=",", fmt="%d")
    
    with open(os.path.join(output_dir, "intervention.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        for sample_idx in range(n_samples):
            sample_intervention = I_flags[sample_idx]
            intervened_vars = np.where(sample_intervention > 0)[0]
            writer.writerow(intervened_vars.tolist())

    has_interventions = bool(np.any(I_flags > 0))
    return n_vars, n_samples, has_interventions

# =====================================================
# 3. Core DCDI runner with compatibility patches
# =====================================================
def run_dcdi_official(data_dir, exp_dir, d, n_total_samples, seed=42, has_interventions=False):
    """
    Run official DCDI and return a weighted adjacency matrix (float, shape [d,d]).
    Priority:
      1) read exp_dir/train/DAG.npy if exists
      2) fallback to captured model.get_w_adj()
      3) fallback to exp_dir/DAG.npy or exp_dir/train/DAG (rare)
      4) last resort: zeros
    """
    # Optional external DCDI root.
    if os.environ.get("DCDI_ROOT"):
        sys.path.insert(0, os.environ["DCDI_ROOT"])

    import dcdi.train
    import dcdi.main

    # -------------------------------------------------
    # PATCH 0: compatibility with newer PyTorch and GPU defaults. Some DCDI
    # code paths assign torch tensors into numpy arrays when constructing
    # intervention masks; allow numpy to explicitly copy CUDA tensors to CPU.
    # -------------------------------------------------
    original_tensor_array = getattr(torch.Tensor, "__array__", None)

    def _dcdi_tensor_array(self, dtype=None):
        arr = self.detach().cpu().numpy()
        if dtype is not None:
            return arr.astype(dtype, copy=False)
        return arr

    torch.Tensor.__array__ = _dcdi_tensor_array

    # -------------------------------------------------
    # PATCH 1: disable plotting to avoid Matplotlib compatibility failures.
    # -------------------------------------------------
    dummy_func = lambda *args, **kwargs: None
    dcdi.train.plot_weighted_adjacency = dummy_func
    dcdi.train.plot_adjacency = dummy_func
    dcdi.train.plot_learning_curves = dummy_func
    dcdi.train.plot_interv_w = dummy_func
    dcdi.train.plot_learned_density = dummy_func
    dcdi.train.plot_learning_curves_retrain = dummy_func
    print("  [Info] Disabled DCDI plotting hooks to avoid Matplotlib issues.")

    # -------------------------------------------------
    # PATCH 2: capture the model object and patch both import sites. DCDI's
    # main.py imports train via `from .train import train`, so both references
    # must be patched.
    # -------------------------------------------------
    captured = {"model": None, "ret": None}
    allow_partial_on_error = _env_bool("DCDI_ALLOW_PARTIAL_ON_ERROR", False)
    # Match the paper benchmark harness: a 5-minute per-graph budget. If the
    # budget expires, the current in-memory DCDI graph is exported below.
    time_limit_seconds = _env_int("DCDI_TIME_LIMIT_SECONDS", 300)
    export_on_timeout = _env_bool("DCDI_EXPORT_ON_TIMEOUT", True)

    original_train_func = dcdi.train.train
    original_main_train = getattr(dcdi.main, "train", None)

    def intercepted_train(model, *args, **kwargs):
        captured["model"] = model
        if not allow_partial_on_error:
            ret = original_train_func(model, *args, **kwargs)
            captured["ret"] = ret
            return ret if ret is not None else model

        ret = None
        try:
            ret = original_train_func(model, *args, **kwargs)
        except Exception:
            import traceback
            print("  [Warn] Training failed; DCDI_ALLOW_PARTIAL_ON_ERROR=1, exporting current model if possible:")
            traceback.print_exc()
        captured["ret"] = ret
        return ret if ret is not None else model

    # Apply patches.
    dcdi.train.train = intercepted_train
    if original_main_train is not None:
        dcdi.main.train = intercepted_train

    # -------------------------------------------------
    # Paths.
    # -------------------------------------------------
    os.makedirs(exp_dir, exist_ok=True)

    class Args:
        pass

    args = Args()

    # Base data/model configuration.
    args.exp_path = exp_dir
    args.data_path = data_dir
    args.i_dataset = ""
    args.num_vars = d
    args.random_seed = seed

    # Training mode.
    args.train = True
    args.retrain = False
    args.dag_for_retrain = None
    args.test_on_new_regimes = False

    # Model architecture: official perfect-known examples use DCDI-DSF.
    args.model = os.environ.get("DCDI_MODEL", "DCDI-DSF")
    args.num_layers = 2
    args.hid_dim = 16
    args.nonlin = "leaky-relu"
    args.flow_num_layers = 2
    args.flow_hid_dim = 16

    # Intervention settings.
    args.intervention = bool(has_interventions)
    args.intervention_type = "perfect"
    args.intervention_knowledge = "known"
    args.dcd = False
    args.coeff_interv_sparsity = 1e-8
    args.regimes_to_ignore = None

    # Optimizer settings.
    args.optimizer = os.environ.get("DCDI_OPTIMIZER", "rmsprop")
    args.lr = _env_float("DCDI_LR", 1e-3)
    args.lr_reinit = None
    args.lr_schedule = None
    args.reg_coeff = _env_float("DCDI_REG_COEFF", 0.5)

    # Train/test split.
    args.train_samples = max(1, int(0.8 * n_total_samples))
    args.test_samples = max(0, n_total_samples - args.train_samples)
    args.num_folds = 1
    args.fold = 0
    requested_batch_size = _env_int("DCDI_TRAIN_BATCH_SIZE", 64)
    args.train_batch_size = max(1, min(requested_batch_size, args.train_samples))
    args.normalize_data = _env_bool("DCDI_NORMALIZE_DATA", True)

    # Augmented Lagrangian settings.
    args.omega_gamma = _env_float("DCDI_OMEGA_GAMMA", 1e-4)
    args.omega_mu = _env_float("DCDI_OMEGA_MU", 0.9)
    args.mu_init = _env_float("DCDI_MU_INIT", 1e-8)
    args.mu_mult_factor = _env_float("DCDI_MU_MULT_FACTOR", 2.0)
    args.gamma_init = 0.0
    args.h_threshold = _env_float("DCDI_H_THRESHOLD", 1e-8)

    # Iteration control.
    args.num_train_iter = _env_int("DCDI_NUM_TRAIN_ITER", 1000000)
    args.stop_crit_win = _env_int("DCDI_STOP_CRIT_WIN", 100)
    args.patience = _env_int("DCDI_PATIENCE", 10)
    args.train_patience = _env_int("DCDI_TRAIN_PATIENCE", 5)
    args.train_patience_post = _env_int("DCDI_TRAIN_PATIENCE_POST", 5)

    # Miscellaneous runtime settings.
    args.plot_freq = 999999999
    args.no_w_adjs_log = True
    args.plot_density = False
    args.gpu = _env_bool("DCDI_GPU", torch.cuda.is_available())
    args.float = _env_bool("DCDI_FLOAT", True)

    # -------------------------------------------------
    # Execute DCDI.
    # -------------------------------------------------
    print(
        f"  Starting training ({args.model})... "
        f"[train_samples={args.train_samples}, test_samples={args.test_samples}, "
        f"batch_size={args.train_batch_size}, iter={args.num_train_iter}, "
        f"gpu={args.gpu}, reg_coeff={args.reg_coeff}, normalize={args.normalize_data}, "
        f"prestandardize={_env_bool('DCDI_PRESTANDARDIZE', False)}, "
        f"time_limit={time_limit_seconds}s, export_on_timeout={export_on_timeout}]"
    )
    old_alarm_handler = None
    if time_limit_seconds > 0:
        def _handle_time_limit(signum, frame):
            raise DCDITimeLimitReached(f"DCDI per-graph time limit reached: {time_limit_seconds}s")

        old_alarm_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _handle_time_limit)
        signal.alarm(int(time_limit_seconds))

    try:
        dcdi.main.main(args)
    except DCDITimeLimitReached:
        print(f"  [Warn] DCDI reached {time_limit_seconds}s time limit.")
        if not export_on_timeout:
            raise
        print("  [Warn] Exporting current DCDI model as the final prediction for this graph.")
    except Exception:
        import traceback
        print("  [Error] DCDI main process failed:")
        traceback.print_exc()
        if not allow_partial_on_error:
            raise
        print("  [Warn] DCDI_ALLOW_PARTIAL_ON_ERROR=1, trying to export the current model/files.")
    finally:
        # Restore patched objects.
        dcdi.train.train = original_train_func
        if original_main_train is not None:
            dcdi.main.train = original_main_train
        if original_tensor_array is not None:
            torch.Tensor.__array__ = original_tensor_array
        if time_limit_seconds > 0:
            signal.alarm(0)
            if old_alarm_handler is not None:
                signal.signal(signal.SIGALRM, old_alarm_handler)

    # -------------------------------------------------
    # Extract results: prefer files, then fall back to the captured model.
    # -------------------------------------------------
    # The official train.py writes exp_path/train/DAG.npy.
    train_dir = os.path.join(exp_dir, "train")

    candidates = [
        os.path.join(train_dir, "DAG.npy"),
        os.path.join(train_dir, "DAG"),       # np.save("DAG", ...) may create this path.
        os.path.join(exp_dir, "DAG.npy"),
        os.path.join(exp_dir, "train", "DAG.npy"),
    ]

    for p in candidates:
        if os.path.exists(p):
            try:
                A = np.load(p)
                print(f"  [Info] Loaded result file: {p}")
                # Return weighted scores; the release metric step binarizes them.
                return A.astype(np.float32)
            except Exception:
                pass

    # Fall back to the in-memory model.
    model = captured.get("model", None)
    if model is not None:
        try:
            model.eval()
            with torch.no_grad():
                w_adj = model.get_w_adj().detach().cpu().numpy().astype(np.float32)
            print("  [Info] Exported w_adj from the in-memory model.")
            return w_adj
        except Exception as e:
            print(f"  [Warn] In-memory export failed: {e}")

    raise RuntimeError("Could not obtain DCDI result: no DAG file and no exportable w_adj.")


# =====================================================
# CLI entry point
# =====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--max_per_f", type=int, default=-1)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--results_root", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_preds", action="store_true")
    
    args = parser.parse_args()
    
    tracker = ResultTracker(results_root=args.results_root, exp_name=args.exp_name, args=args)

    if args.save_preds:
        stale_pred_path = os.path.join(tracker.exp_dir, "predictions.npz")
        if os.path.exists(stale_pred_path):
            try:
                os.remove(stale_pred_path)
                print(f"[Clean] Removed stale predictions file: {stale_pred_path}")
            except Exception as e:
                print(f"[Warn] Failed to remove stale predictions file {stale_pred_path}: {e}")
    
    # Collect input files.
    all_files = sorted([os.path.join(args.data_root, f) for f in os.listdir(args.data_root) if f.endswith(".npz")])
    
    files_by_f = defaultdict(list)
    for p in all_files:
        try:
            f_val = int(re.search(r"_f(\d+)_", os.path.basename(p)).group(1))
        except:
            f_val = 0
        files_by_f[f_val].append(p)
    
    temp_root = tempfile.mkdtemp(prefix="dcdi_run_")
    print(f"[Info] Temporary directory: {temp_root}")
    
    try:
        for f in sorted(files_by_f.keys()):
            paths = files_by_f[f]
            if args.max_per_f > 0: paths = paths[:args.max_per_f]
            
            print(f"\nProcessing f={f}, {len(paths)} files...")
            all_preds = []
            
            for idx, npz_path in enumerate(paths):
                try:
                    sample_name = os.path.basename(npz_path).replace(".npz", "")
                    work_dir = os.path.join(temp_root, sample_name)
                    data_dir = os.path.join(work_dir, "data")
                    exp_dir = os.path.join(work_dir, "exp")
                    
                    # 1. Convert data.
                    d, n_samples, has_interventions = convert_npz_to_official_format(npz_path, data_dir)
                    
                    # 2. Run DCDI.
                    A_weighted = run_dcdi_official(
                        data_dir,
                        exp_dir,
                        d,
                        n_samples,
                        args.seed,
                        has_interventions=has_interventions,
                    )
                    
                    # 3. Evaluate. DCDI may return probabilities or weights.
                    A_binary = (A_weighted > 0.5).astype(int)
                    np.fill_diagonal(A_binary, 0)
                    
                    data_orig = np.load(npz_path)
                    mask = data_orig["mask"].astype(bool)
                    G_true = data_orig["g"][mask][:, mask].astype(int)
                    
                    metrics = dag_metrics(G_true, A_binary)
                    
                    # Print a compact progress line.
                    shd_val = metrics.get('shd', 'N/A')
                    print(f"  Sample {idx+1}/{len(paths)}: SHD={shd_val}")
                    
                    tracker.log(metrics, f=f, filename=os.path.basename(npz_path))
                    
                    if args.save_preds:
                        all_preds.append((npz_path, A_weighted))
                        
                except Exception as e:
                    print(f"  Failed {npz_path}: {e}")
                    # Keep the traceback for failed per-graph runs.
                    import traceback
                    traceback.print_exc()

            if args.save_preds and all_preds:
                save_path = os.path.join(tracker.exp_dir, "predictions.npz")
                pred_dict = {os.path.basename(p): a for p, a in all_preds}
                save_predictions_incremental(save_path, pred_dict)

    finally:
        if os.path.exists(temp_root):
            shutil.rmtree(temp_root, ignore_errors=True)
        tracker.finalize()

if __name__ == "__main__":
    main()
