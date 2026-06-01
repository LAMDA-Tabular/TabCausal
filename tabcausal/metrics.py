"""Metric helpers used by the public benchmark scripts."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np


_OFFICIAL_SID_UNAVAILABLE = False


def _offdiag_mask(d: int) -> np.ndarray:
    return ~np.eye(d, dtype=bool)


def structural_hamming_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute SHD with edge reversals counted as one edit."""

    diff = np.abs(np.asarray(y_true, dtype=np.int64) - np.asarray(y_pred, dtype=np.int64))
    mistakes = diff + diff.T
    mistakes = np.where(mistakes > 1, 1, mistakes)
    return float(np.triu(mistakes, k=1).sum())


def _reachability_matrix(adj: np.ndarray) -> np.ndarray:
    n = adj.shape[0]
    reach = np.eye(n, dtype=bool)
    adj_bool = adj.astype(bool)
    for _ in range(n):
        new_reach = reach | (reach @ adj_bool)
        if np.array_equal(new_reach, reach):
            break
        reach = new_reach
    return reach


def _parents(adj: np.ndarray, node: int) -> list[int]:
    return np.where(adj[:, node] != 0)[0].tolist()


def _children(adj: np.ndarray, node: int) -> list[int]:
    return np.where(adj[node, :] != 0)[0].tolist()


def _ancestors_of_set(adj: np.ndarray, nodes: set[int]) -> set[int]:
    ancestors: set[int] = set()
    stack = list(nodes)
    while stack:
        node = stack.pop()
        for parent in _parents(adj, node):
            if parent not in ancestors:
                ancestors.add(parent)
                stack.append(parent)
    return ancestors


def _is_d_separated(adj: np.ndarray, x: int, y: int, conditioned: set[int]) -> bool:
    """Bayes-ball d-separation check on a DAG."""

    ancestors_of_conditioned = _ancestors_of_set(adj, conditioned)
    queue = [(x, "up"), (x, "down")]
    visited: set[tuple[int, str]] = set()

    while queue:
        node, direction = queue.pop()
        if (node, direction) in visited:
            continue
        visited.add((node, direction))

        if node == y and node not in conditioned:
            return False

        if direction == "up":
            if node not in conditioned:
                for parent in _parents(adj, node):
                    queue.append((parent, "up"))
                for child in _children(adj, node):
                    queue.append((child, "down"))
        else:
            if node not in conditioned:
                for child in _children(adj, node):
                    queue.append((child, "down"))
            if node in conditioned or node in ancestors_of_conditioned:
                for parent in _parents(adj, node):
                    queue.append((parent, "up"))

    return True


def _nodes_on_directed_paths(adj: np.ndarray, src: int, dst: int) -> set[int]:
    reach = _reachability_matrix(adj)
    nodes: set[int] = set()
    for node in range(adj.shape[0]):
        if node != src and reach[src, node] and reach[node, dst]:
            nodes.add(node)
    return nodes


def _descendants_of_nodes(adj: np.ndarray, nodes: set[int]) -> set[int]:
    reach = _reachability_matrix(adj)
    descendants: set[int] = set()
    for node in nodes:
        descendants.update(np.where(reach[node])[0].tolist())
    return descendants


def _remove_outgoing_edges(adj: np.ndarray, node: int) -> np.ndarray:
    out = np.array(adj, copy=True)
    out[node, :] = 0
    return out


def _parent_adjustment_valid(true_adj: np.ndarray, pred_adj: np.ndarray, src: int, dst: int) -> bool:
    adjustment = set(_parents(pred_adj, src))
    if src in adjustment or dst in adjustment:
        return False
    path_nodes = _nodes_on_directed_paths(true_adj, src, dst)
    forbidden = _descendants_of_nodes(true_adj, path_nodes)
    if adjustment & forbidden:
        return False
    return _is_d_separated(_remove_outgoing_edges(true_adj, src), src, dst, adjustment)


def approximate_structural_intervention_distance(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """Approximate SID when the official R package is not installed.

    This follows the parent-adjustment characterization used by common SID
    implementations, but the official R package remains preferred for reported
    numbers.
    """

    true_adj = np.asarray(y_true, dtype=np.int64)
    pred_adj = np.asarray(y_pred, dtype=np.int64)
    true_reach = _reachability_matrix(true_adj)
    pred_reach = _reachability_matrix(pred_adj)

    sid = 0
    n = true_adj.shape[0]
    for src in range(n):
        for dst in range(n):
            if src == dst:
                continue
            pred_effect = bool(pred_reach[src, dst])
            true_effect = bool(true_reach[src, dst])
            if not pred_effect:
                sid += int(true_effect)
            elif not _parent_adjustment_valid(true_adj, pred_adj, src, dst):
                sid += 1
    return int(sid)


def official_structural_intervention_distance(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """Compute SID with the official R ``SID`` package.

    Requires an R installation with ``SID`` available, for example:
    ``R -q -e 'install.packages("SID")'``.
    """

    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("Rscript executable not found; cannot compute official SID.")

    script = """
    suppressPackageStartupMessages(library(SID))
    args <- commandArgs(trailingOnly=TRUE)
    trueGraph <- as.matrix(read.csv(args[1], header=FALSE, check.names=FALSE))
    estGraph <- as.matrix(read.csv(args[2], header=FALSE, check.names=FALSE))
    mode(trueGraph) <- "integer"
    mode(estGraph) <- "integer"
    value <- structIntervDist(trueGraph, estGraph, output=FALSE, spars=FALSE)$sid
    cat(as.integer(value))
    """
    with tempfile.TemporaryDirectory(prefix="tabcausal_sid_") as tmpdir:
        true_path = f"{tmpdir}/true.csv"
        pred_path = f"{tmpdir}/pred.csv"
        np.savetxt(true_path, np.asarray(y_true, dtype=np.int32), fmt="%d", delimiter=",")
        np.savetxt(pred_path, np.asarray(y_pred, dtype=np.int32), fmt="%d", delimiter=",")
        result = subprocess.run(
            [rscript, "--vanilla", "-e", script, true_path, pred_path],
            check=True,
            capture_output=True,
            text=True,
        )
    return int(result.stdout.strip().splitlines()[-1])


def structural_intervention_distance(y_true: np.ndarray, y_pred: np.ndarray, *, prefer_official: bool = True) -> tuple[int, str]:
    """Return ``(SID, source)`` using official SID when available."""

    global _OFFICIAL_SID_UNAVAILABLE
    if prefer_official and not _OFFICIAL_SID_UNAVAILABLE:
        try:
            return official_structural_intervention_distance(y_true, y_pred), "official_r_sid"
        except Exception:
            _OFFICIAL_SID_UNAVAILABLE = True
    return approximate_structural_intervention_distance(y_true, y_pred), "python_approx"


def binary_graph_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
    sid: bool = True,
    prefer_official_sid: bool = True,
) -> dict[str, float | str]:
    """Compute common directed-graph metrics on off-diagonal entries."""

    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    if y_true.shape != y_score.shape:
        raise ValueError(f"Shape mismatch: true={y_true.shape}, score={y_score.shape}")

    mask = _offdiag_mask(y_true.shape[0])
    true_flat = y_true[mask] > 0
    score_flat = y_score[mask]
    pred_flat = score_flat >= threshold
    y_pred = (y_score >= threshold).astype(np.int64)
    np.fill_diagonal(y_pred, 0)

    tp = float(np.logical_and(pred_flat, true_flat).sum())
    fp = float(np.logical_and(pred_flat, ~true_flat).sum())
    fn = float(np.logical_and(~pred_flat, true_flat).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    shd = structural_hamming_distance(y_true, y_pred)
    sid_value: int | float = float("nan")
    sid_source = "not_computed"
    if sid:
        sid_value, sid_source = structural_intervention_distance(
            y_true,
            y_pred,
            prefer_official=prefer_official_sid,
        )

    auroc = float("nan")
    ap = float("nan")
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        if len(np.unique(true_flat)) == 2:
            auroc = float(roc_auc_score(true_flat, score_flat))
        ap = float(average_precision_score(true_flat, score_flat))
    except Exception:
        pass

    return {
        "SHD": shd,
        "SID": float(sid_value),
        "SID_source": sid_source,
        "F1": float(f1),
        "Precision": float(precision),
        "Recall": float(recall),
        "AUROC": auroc,
        "AP": ap,
    }


def summarize_rows(rows: list[dict[str, Any]], group_key: str = "f") -> list[dict[str, Any]]:
    """Return mean and standard deviation by ``group_key``."""

    if not rows:
        return []
    keys = [key for key, value in rows[0].items() if isinstance(value, (int, float, np.floating))]
    groups = sorted({row[group_key] for row in rows})
    out: list[dict[str, Any]] = []
    for group in groups:
        subset = [row for row in rows if row[group_key] == group]
        summary: dict[str, Any] = {group_key: group, "n": len(subset)}
        for key in keys:
            if key == group_key:
                continue
            vals = np.asarray([row[key] for row in subset], dtype=np.float64)
            summary[f"{key}_mean"] = float(np.nanmean(vals))
            summary[f"{key}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        out.append(summary)
    return out
