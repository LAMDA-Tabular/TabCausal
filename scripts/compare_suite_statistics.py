#!/usr/bin/env python3
"""Compare two TabCausal NPZ suites at the data-distribution level."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


F_RE = re.compile(r"(?:^|[_-])f(?P<f>\d+)(?:[_-]|\.|$)")


def _infer_f(path: Path, data: np.lib.npyio.NpzFile) -> int:
    if "f" in data:
        return int(data["f"])
    match = F_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot infer f from {path}")
    return int(match.group("f"))


def _dataset_key(folder: Path) -> tuple[str, str]:
    name = folder.name
    if not (name.startswith("[") and "]_" in name):
        return name, "unknown"
    family, regime = name.split("]_", 1)
    return family.strip("[]"), regime


def summarize_suite(root: Path, *, max_per_f: int | None = None) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, int], list[Path]] = defaultdict(list)
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        family, regime = _dataset_key(folder)
        for path in sorted(folder.glob("*.npz")):
            with np.load(path, allow_pickle=False) as data:
                f = _infer_f(path, data)
            groups[(family, regime, f)].append(path)

    rows: list[dict[str, object]] = []
    for (family, regime, f), paths in sorted(groups.items()):
        if max_per_f is not None and max_per_f > 0:
            paths = paths[:max_per_f]
        edge_counts = []
        densities = []
        dims = []
        row_counts = []
        mask_sums = []
        value_means = []
        value_stds = []
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                x = np.asarray(data["x"])
                g = np.asarray(data["g"])
                feature_mask = np.asarray(data["mask"]).astype(bool) if "mask" in data else np.ones(g.shape[0], dtype=bool)
                if feature_mask.ndim > 1:
                    feature_mask = feature_mask.reshape(-1)
                g_eff = g[np.ix_(feature_mask, feature_mask)]
                d_eff = int(g_eff.shape[0])
                possible_edges = max(d_eff * (d_eff - 1), 1)
                edge_counts.append(float(g_eff.sum()))
                densities.append(float(g_eff.sum()) / float(possible_edges))
                dims.append(float(d_eff))
                row_counts.append(float(x.shape[0]))
                if x.ndim == 3 and x.shape[-1] >= 2:
                    x_eff = x[:, feature_mask, :]
                    values = x_eff[..., 0]
                    mask_sums.append(float(x_eff[..., 1].sum()))
                else:
                    values = x[:, feature_mask] if x.ndim == 2 else x
                    mask_sums.append(0.0)
                value_means.append(float(np.mean(values)))
                value_stds.append(float(np.std(values)))
        rows.append(
            {
                "family": family,
                "regime": regime,
                "f": f,
                "n_graphs": len(paths),
                "dim_mean": np.mean(dims),
                "edges_mean": np.mean(edge_counts),
                "edges_std": np.std(edge_counts, ddof=1) if len(edge_counts) > 1 else 0.0,
                "density_mean": np.mean(densities),
                "density_std": np.std(densities, ddof=1) if len(densities) > 1 else 0.0,
                "rows_mean": np.mean(row_counts),
                "mask_sum_mean": np.mean(mask_sums),
                "value_mean_mean": np.mean(value_means),
                "value_std_mean": np.mean(value_stds),
            }
        )
    return rows


def _index(rows: list[dict[str, object]]) -> dict[tuple[str, str, int], dict[str, object]]:
    return {(str(r["family"]), str(r["regime"]), int(r["f"])): r for r in rows}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", default=None, help="Single suite to summarize without comparing against another suite.")
    parser.add_argument("--reference-suite", default=None)
    parser.add_argument("--candidate-suite", default=None)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--max-per-f", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_csv)
    if args.suite_root is not None:
        rows = summarize_suite(Path(args.suite_root), max_per_f=args.max_per_f)
        write_csv(out, rows)
        print(f"wrote: {out}")
        return

    if args.reference_suite is None or args.candidate_suite is None:
        raise SystemExit("Either pass --suite-root, or pass both --reference-suite and --candidate-suite.")

    reference = summarize_suite(Path(args.reference_suite), max_per_f=args.max_per_f)
    candidate = summarize_suite(Path(args.candidate_suite), max_per_f=args.max_per_f)
    ref_idx = _index(reference)
    cand_idx = _index(candidate)

    rows: list[dict[str, object]] = []
    for key in sorted(set(ref_idx) | set(cand_idx)):
        r = ref_idx.get(key)
        c = cand_idx.get(key)
        family, regime, f = key
        row: dict[str, object] = {"family": family, "regime": regime, "f": f}
        if r is None or c is None:
            row["status"] = "missing_reference" if r is None else "missing_candidate"
        else:
            row["status"] = "ok"
            for metric in (
                "n_graphs",
                "dim_mean",
                "edges_mean",
                "edges_std",
                "density_mean",
                "density_std",
                "rows_mean",
                "mask_sum_mean",
                "value_mean_mean",
                "value_std_mean",
            ):
                rv = float(r[metric])
                cv = float(c[metric])
                row[f"ref_{metric}"] = rv
                row[f"cand_{metric}"] = cv
                row[f"diff_{metric}"] = cv - rv
            ref_density = float(r["density_mean"])
            cand_density = float(c["density_mean"])
            row["ratio_density"] = cand_density / ref_density if ref_density > 0 else ""
            ref_edges = float(r["edges_mean"])
            cand_edges = float(c["edges_mean"])
            row["ratio_edges"] = cand_edges / ref_edges if ref_edges > 0 else ""
        rows.append(row)

    write_csv(out, rows)
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
