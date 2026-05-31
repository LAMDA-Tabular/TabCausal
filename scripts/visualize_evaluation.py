#!/usr/bin/env python3
"""Create lightweight figures from TabCausal evaluation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCORE_METRICS = ["F1_mean", "Precision_mean", "Recall_mean", "AUROC_mean", "AP_mean"]
ERROR_METRICS = ["SHD_mean", "SID_mean"]
METRIC_LABELS = {
    "F1_mean": "F1",
    "Precision_mean": "Precision",
    "Recall_mean": "Recall",
    "AUROC_mean": "AUROC",
    "AP_mean": "AP",
    "SHD_mean": "SHD",
    "SID_mean": "SID",
}


def _metric_columns(frame: pd.DataFrame) -> list[str]:
    preferred = ["F1_mean", "Precision_mean", "Recall_mean", "AUROC_mean", "AP_mean", "SHD_mean", "SID_mean"]
    return [col for col in preferred if col in frame.columns]


def _clean_axis(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def _annotate_matrix(ax, values: np.ndarray, *, fmt: str = ".2f") -> None:
    if values.size > 100:
        return
    finite = values[np.isfinite(values)]
    midpoint = float(np.nanmean(finite)) if finite.size else 0.0
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if not np.isfinite(value):
                text = "nan"
                color = "#172026"
            else:
                text = format(float(value), fmt)
                color = "white" if value > midpoint else "#172026"
            ax.text(col, row, text, ha="center", va="center", fontsize=8, color=color)


def _draw_heatmap(
    fig,
    ax,
    matrix: np.ndarray,
    *,
    title: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    fmt: str = ".2f",
) -> None:
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color="#172026")
    ax.set_xlabel("Target node", color="#46545c")
    ax.set_ylabel("Source node", color="#46545c")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_xticklabels(range(matrix.shape[1]), color="#46545c")
    ax.set_yticklabels(range(matrix.shape[0]), color="#46545c")
    ax.set_aspect("equal")
    ax.set_facecolor("#f8f4ea")
    _clean_axis(ax)
    _annotate_matrix(ax, matrix, fmt=fmt)
    cbar = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.03)
    cbar.outline.set_visible(False)


def _plot_single_f_summary(df: pd.DataFrame, metrics: list[str], output: Path) -> None:
    import matplotlib.pyplot as plt

    group_cols = [col for col in ["dataset", "regime"] if col in df.columns]
    if group_cols:
        labels = df[group_cols].astype(str).agg(" / ".join, axis=1).tolist()
    else:
        labels = ["summary"] * len(df)

    score_cols = [col for col in SCORE_METRICS if col in metrics]
    error_cols = [col for col in ERROR_METRICS if col in metrics]
    ncols = int(bool(score_cols)) + int(bool(error_cols))
    width_ratios = []
    if score_cols:
        width_ratios.append(max(len(score_cols), 1))
    if error_cols:
        width_ratios.append(max(len(error_cols), 1))
    fig_width = max(8.5, 1.15 * sum(width_ratios) + 3.8)
    fig_height = max(4.8, 0.48 * len(df) + 1.8)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        constrained_layout=True,
        gridspec_kw={"width_ratios": width_ratios} if width_ratios else None,
    )
    fig.patch.set_facecolor("#fbf7ef")

    axis_iter = iter(axes.ravel())
    if score_cols:
        ax = next(axis_iter)
        values = df[score_cols].to_numpy(dtype=float)
        im = ax.imshow(values, cmap="YlGnBu", vmin=0.0, vmax=1.0)
        ax.set_title("Scores (higher is better)", loc="left", fontsize=12, fontweight="bold", color="#172026")
        ax.set_xticks(range(len(score_cols)))
        ax.set_xticklabels([METRIC_LABELS[col] for col in score_cols], rotation=30, ha="right", color="#46545c")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, color="#46545c")
        _clean_axis(ax)
        _annotate_matrix(ax, values, fmt=".2f")
        cbar = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.02)
        cbar.outline.set_visible(False)

    if error_cols:
        ax = next(axis_iter)
        values = df[error_cols].to_numpy(dtype=float)
        vmax = max(float(np.nanmax(values)), 1.0) if values.size else 1.0
        im = ax.imshow(values, cmap="OrRd", vmin=0.0, vmax=vmax)
        ax.set_title("Errors", loc="left", fontsize=12, fontweight="bold", color="#172026")
        ax.set_xticks(range(len(error_cols)))
        ax.set_xticklabels([METRIC_LABELS[col] for col in error_cols], rotation=30, ha="right", color="#46545c")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, color="#46545c")
        _clean_axis(ax)
        _annotate_matrix(ax, values, fmt=".1f")
        cbar = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.02)
        cbar.outline.set_visible(False)

    f_value = df["f"].iloc[0] if "f" in df.columns and len(df) else "single"
    fig.suptitle(f"Smoke summary at f={f_value}", fontsize=15, fontweight="bold", color="#172026")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_metric_grid(summary: Path, output: Path) -> None:
    import matplotlib.pyplot as plt

    df = pd.read_csv(summary)
    if "f" in df.columns:
        df = df[df["f"].astype(str).str.lower() != "avg"].copy()
        df["f"] = pd.to_numeric(df["f"], errors="coerce")
    metrics = _metric_columns(df)
    if not metrics:
        raise ValueError(f"No metric columns found in {summary}")

    group_cols = [col for col in ["dataset", "regime"] if col in df.columns]
    f_values = df["f"].dropna().unique() if "f" in df.columns else np.asarray([])
    if len(f_values) <= 1:
        _plot_single_f_summary(df, metrics, output)
        return

    if group_cols:
        df["series"] = df[group_cols].astype(str).agg("/".join, axis=1)
    else:
        df["series"] = "summary"

    ncols = min(3, len(metrics))
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.3 * nrows), squeeze=False, constrained_layout=True)
    fig.patch.set_facecolor("#fbf7ef")
    for ax, metric in zip(axes.ravel(), metrics):
        for name, sub in df.groupby("series"):
            sub = sub.sort_values("f") if "f" in sub.columns else sub
            x = sub["f"] if "f" in sub.columns else np.arange(len(sub))
            ax.plot(x, sub[metric], marker="o", markersize=5, linewidth=1.8, label=name)
        ax.set_title(METRIC_LABELS.get(metric, metric.replace("_mean", "")), loc="left", fontweight="bold", color="#172026")
        ax.set_xlabel("Number of variables")
        ax.grid(True, axis="y", color="#d8d1c4", alpha=0.7)
        ax.set_facecolor("#fbf7ef")
        _clean_axis(ax)
        if metric.startswith("SHD") or metric.startswith("SID"):
            ax.set_ylabel("Lower is better")
        else:
            ax.set_ylabel("Higher is better")
    for ax in axes.ravel()[len(metrics):]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncols=min(4, len(labels)), frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _load_prediction_matrix(prediction: Path, key: str, index: int) -> np.ndarray:
    with np.load(prediction, allow_pickle=True) as data:
        if key not in data:
            raise KeyError(f"{prediction} does not contain {key!r}")
        arr = data[key]
        if arr.dtype == object:
            return np.asarray(arr[index], dtype=float)
        if arr.ndim == 3:
            return np.asarray(arr[index], dtype=float)
        return np.asarray(arr, dtype=float)


def _plot_heatmap(matrix: np.ndarray, output: Path, title: str, *, cmap: str, vmin: float | None = None, vmax: float | None = None) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("#fbf7ef")
    _draw_heatmap(fig, ax, matrix, title=title, cmap=cmap, vmin=vmin, vmax=vmax, fmt=".2f")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_prediction_overview(prob: np.ndarray, adj: np.ndarray, output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("#fbf7ef")
    _draw_heatmap(fig, axes[0], prob, title="Edge probabilities", cmap="YlGnBu", vmin=0.0, vmax=1.0, fmt=".2f")
    _draw_heatmap(fig, axes[1], adj, title="Thresholded adjacency", cmap="Greens", vmin=0.0, vmax=1.0, fmt=".0f")
    fig.suptitle("TabCausal prediction overview", fontsize=15, fontweight="bold", color="#172026")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, facecolor=fig.get_facecolor())
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", "--summary-csv", dest="summary", default=None, help="summary.csv or benchmark_summary.csv to plot.")
    parser.add_argument("--prediction", "--prediction-npz", dest="prediction", default=None, help="predictions.npz or single prediction file.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prediction-index", "--index", dest="prediction_index", type=int, default=0, help="Graph index when plotting predictions.npz.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.summary is not None:
        _plot_metric_grid(Path(args.summary), output_dir / "metric_summary.png")
    if args.prediction is not None:
        pred = Path(args.prediction)
        prob = _load_prediction_matrix(pred, "probabilities", args.prediction_index)
        _plot_heatmap(prob, output_dir / "probability_heatmap.png", "Edge probabilities", cmap="YlGnBu", vmin=0.0, vmax=1.0)
        try:
            adj = _load_prediction_matrix(pred, "adjacencies", args.prediction_index)
        except KeyError:
            adj = _load_prediction_matrix(pred, "adjacency", args.prediction_index)
        _plot_heatmap(adj, output_dir / "adjacency_heatmap.png", "Thresholded adjacency", cmap="Greens", vmin=0.0, vmax=1.0)
        _plot_prediction_overview(prob, adj, output_dir / "prediction_overview.png")
    print(f"wrote figures to: {output_dir}")


if __name__ == "__main__":
    main()
