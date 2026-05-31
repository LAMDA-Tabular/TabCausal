#!/usr/bin/env python3
"""Visualize TabCausal prediction outputs.

The script reads a single-prediction ``.npz`` file produced by
``python -m tabcausal.cli predict`` and writes heatmaps for probabilities and
binary adjacency.  If embeddings are present, it also writes a two-dimensional
PCA scatter plot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


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
            text = "nan" if not np.isfinite(value) else format(float(value), fmt)
            color = "white" if np.isfinite(value) and value > midpoint else "#172026"
            ax.text(col, row, text, ha="center", va="center", fontsize=8, color=color)


def _plot_heatmap(array: np.ndarray, path: Path, title: str, *, vmin: float | None = None, vmax: float | None = None) -> None:
    import matplotlib.pyplot as plt

    is_binary = np.array_equal(array, array.astype(bool))
    cmap = "Greens" if is_binary else "YlGnBu"
    fig, ax = plt.subplots(figsize=(5.6, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("#fbf7ef")
    im = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color="#172026")
    ax.set_xlabel("Target node", color="#46545c")
    ax.set_ylabel("Source node", color="#46545c")
    ax.set_xticks(range(array.shape[1]))
    ax.set_yticks(range(array.shape[0]))
    ax.set_xticklabels(range(array.shape[1]), color="#46545c")
    ax.set_yticklabels(range(array.shape[0]), color="#46545c")
    _clean_axis(ax)
    _annotate_matrix(ax, array, fmt=".0f" if is_binary else ".2f")
    cbar = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.03)
    cbar.outline.set_visible(False)
    fig.savefig(path, dpi=240, facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_embeddings(embeddings: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    coords = PCA(n_components=2).fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(5.6, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("#fbf7ef")
    ax.set_facecolor("#fbf7ef")
    ax.scatter(coords[:, 0], coords[:, 1], s=78, color="#0f766e", edgecolor="white", linewidth=1.1)
    for idx, (x, y) in enumerate(coords):
        ax.text(x, y, str(idx), fontsize=8, ha="center", va="center", color="white")
    ax.set_title("Node embedding PCA", loc="left", fontsize=12, fontweight="bold", color="#172026")
    ax.set_xlabel("PC1", color="#46545c")
    ax.set_ylabel("PC2", color="#46545c")
    ax.grid(True, color="#d8d1c4", alpha=0.7)
    _clean_axis(ax)
    fig.savefig(path, dpi=240, facecolor=fig.get_facecolor())
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True, help="Prediction .npz from tabcausal predict.")
    parser.add_argument("--output-dir", required=True, help="Directory for PNG figures.")
    parser.add_argument("--prefix", default="prediction", help="Output filename prefix.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction = Path(args.prediction)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = _load_npz(prediction)

    if "probabilities" in arrays:
        _plot_heatmap(arrays["probabilities"], output_dir / f"{args.prefix}_probabilities.png", "Edge Probabilities", vmin=0.0, vmax=1.0)
    if "adjacency" in arrays:
        _plot_heatmap(arrays["adjacency"], output_dir / f"{args.prefix}_adjacency.png", "Predicted Adjacency", vmin=0.0, vmax=1.0)
    if "embeddings" in arrays:
        _plot_embeddings(arrays["embeddings"], output_dir / f"{args.prefix}_embedding_pca.png")

    print(f"wrote figures to: {output_dir}")


if __name__ == "__main__":
    main()
