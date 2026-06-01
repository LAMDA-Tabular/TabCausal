import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)


# =====================================================
# Helpers
# =====================================================

def binarize_prob(P, thresh=0.5):
    return (P >= thresh).astype(int)


def to_skeleton(A):
    # Convert first so downstream code can safely call .astype.
    A = np.asarray(A)
    return np.asarray((A + A.T) > 0).astype(int)


def confusion(true, pred):
    """return tp, fp, fn, tn using sklearn"""
    true_f = true.flatten()
    pred_f = pred.flatten()

    tp = int(np.sum((pred_f == 1) & (true_f == 1)))
    fp = int(np.sum((pred_f == 1) & (true_f == 0)))
    fn = int(np.sum((pred_f == 0) & (true_f == 1)))
    tn = int(np.sum((pred_f == 0) & (true_f == 0)))

    return tp, fp, fn, tn


# =====================================================
# DAG METRICS: main + skeleton
# =====================================================

import pandas as pd

def get_reachability_matrix(adj):
    """Compute the transitive-closure reachability matrix."""
    n = adj.shape[0]
    reach = np.eye(n, dtype=bool)
    adj_bool = adj.astype(bool)
    for _ in range(n):
        new_reach = reach | (reach @ adj_bool)
        if np.array_equal(new_reach, reach):
            break
        reach = new_reach
    return reach


def _parents(adj, node):
    return np.where(adj[:, node] != 0)[0].tolist()


def _children(adj, node):
    return np.where(adj[node, :] != 0)[0].tolist()


def _remove_outgoing_edges(adj, node):
    adj2 = np.array(adj, copy=True)
    adj2[node, :] = 0
    return adj2


def _ancestors_of_set(adj, nodes):
    anc = set()
    stack = list(nodes)
    while stack:
        v = stack.pop()
        for p in _parents(adj, v):
            if p not in anc:
                anc.add(p)
                stack.append(p)
    return anc


def _is_d_separated(adj, x, y, conditioned):
    """
    Bayes-ball style d-separation check on a DAG.
    Returns True iff x and y are d-separated given conditioned.
    """
    conditioned = set(conditioned)
    ancestors_of_conditioned = _ancestors_of_set(adj, conditioned)
    queue = [(x, "up"), (x, "down")]
    visited = set()

    while queue:
        node, direction = queue.pop()
        if (node, direction) in visited:
            continue
        visited.add((node, direction))

        if node == y and node not in conditioned:
            return False

        if direction == "up":
            if node not in conditioned:
                for p in _parents(adj, node):
                    queue.append((p, "up"))
                for c in _children(adj, node):
                    queue.append((c, "down"))
        else:  # direction == "down"
            if node not in conditioned:
                for c in _children(adj, node):
                    queue.append((c, "down"))
            if node in conditioned or node in ancestors_of_conditioned:
                for p in _parents(adj, node):
                    queue.append((p, "up"))

    return True


def _nodes_on_directed_paths(adj, src, dst):
    """Return nodes W that lie on at least one directed path src -> ... -> dst, excluding src."""
    reach = get_reachability_matrix(adj)
    nodes = set()
    for w in range(adj.shape[0]):
        if w == src:
            continue
        if reach[src, w] and reach[w, dst]:
            nodes.add(w)
    return nodes


def _descendants_of_nodes(adj, nodes):
    reach = get_reachability_matrix(adj)
    desc = set()
    for w in nodes:
        desc.update(np.where(reach[w])[0].tolist())
    return desc


def _parent_adjustment_valid_for_pair(true_adj, pred_adj, i, j):
    """
    Approximate the official SID parent-adjustment logic:
    use Pa_pred(i) as adjustment set and check whether it is valid in the true DAG.
    """
    z = set(_parents(pred_adj, i))
    if i in z or j in z:
        return False

    # Adjustment sets should not contain descendants of nodes on a proper causal path i -> ... -> j.
    path_nodes = _nodes_on_directed_paths(true_adj, i, j)
    forbidden = _descendants_of_nodes(true_adj, path_nodes)
    if len(z & forbidden) > 0:
        return False

    mutilated = _remove_outgoing_edges(true_adj, i)
    return _is_d_separated(mutilated, i, j, z)


def compute_sid(target, prediction):
    """
    Approximate the official SID definition.

    For each ordered variable pair (i, j), use Pa_pred(i) as the adjustment
    set when the predicted graph implies an effect. If the predicted graph
    implies no effect, count an error only when the true graph has a directed
    path i -> ... -> j. Inputs should be DAG adjacency matrices.
    """
    true_adj = np.asarray(target).astype(int)
    pred_adj = np.asarray(prediction).astype(int)

    n = true_adj.shape[0]
    true_reach = get_reachability_matrix(true_adj)
    pred_reach = get_reachability_matrix(pred_adj)

    sid = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            pred_has_effect = bool(pred_reach[i, j])
            true_has_effect = bool(true_reach[i, j])

            if not pred_has_effect:
                if true_has_effect:
                    sid += 1
                continue

            if not _parent_adjustment_valid_for_pair(true_adj, pred_adj, i, j):
                sid += 1

    return int(sid)

def dag_metrics(A_true, A_pred):
    """Compute the five core DAG metrics."""
    A_true = np.asarray(A_true)
    A_pred = np.asarray(A_pred)
    
    # Basic classification metrics.
    precision = precision_score(A_true.flatten(), A_pred.flatten(), zero_division=0)
    recall = recall_score(A_true.flatten(), A_pred.flatten(), zero_division=0)
    f1 = f1_score(A_true.flatten(), A_pred.flatten(), zero_division=0)
    
    # SHD.
    diff = np.abs(A_true - A_pred)
    shd = np.sum(diff) - 0.5 * np.sum(diff * diff.T)
    
    # SID assumes DAG inputs.
    sid = compute_sid(A_true, A_pred)
    
    return {
        "SHD": int(shd),
        "SID": int(sid),
        "F1": f1,
        "Precision": precision,
        "Recall": recall
    }

# =====================================================
# PROBABILISTIC DAG METRICS
# =====================================================

def prob_metrics(A_true, P_pred, thresh=0.5):

    A_bin = binarize_prob(P_pred, thresh)

    base = dag_metrics(A_true, A_bin)

    y_true = A_true.flatten()
    y_score = P_pred.flatten()

    if len(np.unique(y_true)) < 2:
        base["AUROC"] = float("nan")
        base["AP"] = float("nan")
    else:
        base["AUROC"] = roc_auc_score(y_true, y_score)
        base["AP"] = average_precision_score(y_true, y_score)

    return base


# =====================================================
# PAG METRICS FOR CDIS-STYLE OUTPUT
# =====================================================

def pag_metrics(A_true, PAG_mat):
    """
    Minimal robust PAG metric computation.
    """
    # Convert inputs to numpy arrays for consistent downstream operations.
    A_true = np.asarray(A_true)
    PAG_mat = np.asarray(PAG_mat)

    # --- 1. Skeleton metrics ---
    A_true_s = to_skeleton(A_true)
    # Evaluate the logical condition first, then cast to an integer array.
    A_pred_s_raw = np.asarray(PAG_mat != 0).astype(int)
    A_pred_s = np.asarray((A_pred_s_raw + A_pred_s_raw.T) > 0).astype(int)

    sk_precision = precision_score(A_true_s.flatten(), A_pred_s.flatten(), zero_division=0)
    sk_recall = recall_score(A_true_s.flatten(), A_pred_s.flatten(), zero_division=0)
    sk_f1 = f1_score(A_true_s.flatten(), A_pred_s.flatten(), zero_division=0)
    sk_shd = int(np.sum(A_true_s != A_pred_s))

    # --- 2. Certain directed-edge metrics (i -> j) ---
    directed_true = set(zip(*np.where(np.asarray(A_true == 1))))
    directed_pred = set()
    n = PAG_mat.shape[0]
    for i in range(n):
        for j in range(n):
            # PAG code: tail at i (3) and arrow at j (2) means i -> j.
            if PAG_mat[i, j] == 2 and PAG_mat[j, i] == 3:
                directed_pred.add((i, j))

    tp = len(directed_true & directed_pred)
    fp = len(directed_pred - directed_true)
    fn = len(directed_true - directed_pred)

    prec_dir = tp / (tp + fp + 1e-8)
    rec_dir = tp / (tp + fn + 1e-8)
    f1_dir = 2 * prec_dir * rec_dir / (prec_dir + rec_dir + 1e-8)

    # --- 3. Edgemark accuracy ---
    true_pag_marks = np.zeros_like(A_true, dtype=int)
    for i, j in directed_true:
        true_pag_marks[i, j] = 2 
        true_pag_marks[j, i] = 3
        
    em_acc = accuracy_score(true_pag_marks.flatten(), PAG_mat.flatten())

    return {
        "SK_SHD": sk_shd,
        "SK_Precision": sk_precision,
        "SK_Recall": sk_recall,
        "SK_F1": sk_f1,
        "DIR_Precision": prec_dir,
        "DIR_Recall": rec_dir,
        "DIR_F1": f1_dir,
        "Edgemark_Acc": em_acc
    }

import pandas as pd
import os
import csv
import json
import time

class ResultTracker:
    def __init__(self, results_root, exp_name, args=None):
        """
        results_root: root directory for results, e.g. "./results"
        exp_name: experiment name, e.g. "avici_er_gauss"
        args: argparse namespace or config object to save
        """
        self.exp_dir = os.path.join(results_root, exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # 1. Raw per-graph metric file.
        self.csv_path = os.path.join(self.exp_dir, "raw_metrics.csv")
        
        file_exists = os.path.exists(self.csv_path)

        self.csv_file = open(self.csv_path, mode='a', newline='', encoding='utf-8')
        self.writer = None  # Initialized when the first row is written.
        self._need_header = not file_exists
        
        # 2. Summary file.
        self.summary_path = os.path.join(self.exp_dir, "summary.csv")

        # 3. Experiment config.
        if args:
            config_path = os.path.join(self.exp_dir, "config.json")
            with open(config_path, 'w') as f:
                # Convert args to a serializable dictionary.
                args_dict = vars(args) if hasattr(args, '__dict__') else args
                json.dump(args_dict, f, indent=4)
        
        print(f"[Tracker] Results will be saved to: {self.exp_dir}")

    def log(self, metrics, **kwargs):
        """
        metrics: metric dictionary, e.g. {'SHD': 1, 'F1': 0.5}
        kwargs: extra identifiers, e.g. f=10, filename='data_01.npz'
        """
        # Merge identifiers and metrics.
        row = {}
        row.update(kwargs)
        row.update(metrics)
        
        # Initialize the CSV header on the first row.
        if self.writer is None:
            fieldnames = list(row.keys())
            self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
            if self._need_header:
                self.writer.writeheader()
        
        # Flush every row so partial long-running results are retained.
        self.writer.writerow(row)
        self.csv_file.flush()

    def finalize(self):
        """
        Close the raw file and write aggregate summaries.
        """
        if self.csv_file:
            self.csv_file.close()
        
        print("[Tracker] Computing summary...")
        try:
            # Read the raw metrics just written above.
            df = pd.read_csv(self.csv_path)
            
            # Group by graph size and aggregate numeric columns.
            if 'f' in df.columns:
                summary = df.groupby('f').mean(numeric_only=True)
                # Also compute an overall mean row.
                overall = df.mean(numeric_only=True).to_frame().T
                overall.index = ['Overall']
                
                # Save the summary CSV.
                summary.to_csv(self.summary_path)
                # Append the overall row after the grouped summary.
                with open(self.summary_path, 'a') as f:
                    f.write("\n")
                overall.to_csv(self.summary_path, mode='a')
                
                return summary
            else:
                # If no graph-size column exists, compute only the overall mean.
                summary = df.mean(numeric_only=True)
                summary.to_csv(self.summary_path)
                return summary.to_frame().T
                
        except Exception as e:
            print(f"[Tracker] Failed to compute summary with pandas: {e}")
            return None
        

def save_predictions_incremental(predictions_file, new_predictions, verbose=True):
    """Incrementally merge and save prediction matrices."""
    import os
    import numpy as np
    import re
    from collections import defaultdict
    
    # 1. Load existing predictions.
    merged_preds = {}
    if os.path.exists(predictions_file):
        try:
            old_preds = np.load(predictions_file)
            merged_preds = {k: old_preds[k] for k in old_preds.files}
            if verbose:
                print(f"[Save] Loaded {len(merged_preds)} existing predictions", flush=True)
        except:
            pass
    
    # 2. Normalize file names.
    for filename, pred_matrix in new_predictions.items():
        clean_name = filename
        if re.match(r'^\d{4}_', filename):
            clean_name = filename.split('_', 1)[1]
        if not clean_name.endswith('.npz'):
            clean_name += '.npz'
        merged_preds[clean_name] = pred_matrix
    
    # 3. Report counts by graph size.
    if verbose:
        f_counts = defaultdict(int)
        for k in merged_preds.keys():
            match = re.search(r'_f(\d+)_', k)
            if match:
                f_counts[int(match.group(1))] += 1
        if f_counts:
            print(f"[Save] by f: {dict(sorted(f_counts.items()))}, total={len(merged_preds)}", flush=True)
    
    # 4. Save directly.
    np.savez_compressed(predictions_file, **merged_preds)
    if verbose:
        print(f"[Save] Saved to {predictions_file}", flush=True)
