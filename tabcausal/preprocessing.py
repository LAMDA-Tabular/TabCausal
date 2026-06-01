"""Input conversion utilities for TabCausal.

TabCausal expects tensors shaped as ``[batch, observations, variables, channels]``.
The first channel stores variable values. The optional second channel stores
intervention indicators, where 1 means the variable was directly intervened on
for that observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch


Mode = Literal["auto", "obs", "mixed"]


@dataclass(frozen=True)
class GraphExample:
    """One graph instance loaded from a benchmark ``.npz`` file."""

    path: Path
    f_value: int
    x: np.ndarray
    adjacency: np.ndarray | None


def _as_feature_mask(mask: np.ndarray | None, d: int) -> np.ndarray:
    if mask is None:
        return np.ones(d, dtype=bool)
    arr = np.asarray(mask)
    if arr.ndim == 0:
        return np.ones(d, dtype=bool)
    arr = arr.astype(bool)
    if arr.shape[0] != d:
        raise ValueError(f"Feature mask length {arr.shape[0]} does not match d={d}.")
    return arr


def _as_two_channel(x: np.ndarray) -> np.ndarray:
    """Convert value-only arrays to the two-channel public input format."""

    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        flags = np.zeros_like(x, dtype=np.float32)
        return np.stack([x, flags], axis=-1)
    if x.ndim == 3 and x.shape[-1] == 2:
        return x
    if x.ndim == 3:
        flags = np.zeros_like(x, dtype=np.float32)
        return np.stack([x, flags], axis=-1)
    raise ValueError(f"Unsupported data shape {x.shape}; expected [n, d] or [n, d, 2].")


def _load_array_file(path: str | Path) -> np.ndarray:
    """Load a value table from common NumPy/Pandas file formats."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            arr = _find_first(data, ("x", "X", "data", "values", "table"))
            if arr is None:
                raise ValueError(f"{path} does not contain x/X/data/values/table.")
            return arr
    if suffix in {".csv", ".tsv", ".txt"}:
        import pandas as pd

        sep = "\t" if suffix == ".tsv" else None
        frame = pd.read_csv(path, sep=sep, engine="python")
        numeric = frame.select_dtypes(include="number")
        if numeric.empty:
            raise ValueError(f"{path} does not contain numeric columns.")
        return numeric.to_numpy(dtype=np.float32)
    if suffix in {".parquet", ".pq"}:
        import pandas as pd

        frame = pd.read_parquet(path)
        numeric = frame.select_dtypes(include="number")
        if numeric.empty:
            raise ValueError(f"{path} does not contain numeric columns.")
        return numeric.to_numpy(dtype=np.float32)
    if suffix in {".pkl", ".pickle"}:
        import pandas as pd

        obj = pd.read_pickle(path)
        if hasattr(obj, "select_dtypes"):
            numeric = obj.select_dtypes(include="number")
            if numeric.empty:
                raise ValueError(f"{path} does not contain numeric columns.")
            return numeric.to_numpy(dtype=np.float32)
        return np.asarray(obj, dtype=np.float32)
    raise ValueError(
        f"Unsupported input format {suffix!r}. Use .npz, .npy, .csv, .tsv, .parquet, or .pkl."
    )


def _find_first(data: np.lib.npyio.NpzFile, keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        if key in data:
            return data[key]
    return None


def load_graph_npz(path: str | Path, *, mode: Mode = "auto") -> GraphExample:
    """Load a benchmark graph from a flexible ``.npz`` schema."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        adjacency = _find_first(data, ("g", "G", "graph", "dag", "adjacency", "A", "target"))
        x_direct = _find_first(data, ("x", "X", "data"))
        x_obs = _find_first(data, ("x_obs", "X_obs", "obs"))
        x_int = _find_first(data, ("x_int", "X_int", "int"))
        feature_mask_raw = _find_first(data, ("mask", "feature_mask", "node_mask"))

        if x_direct is not None:
            x = _as_two_channel(x_direct)
        elif mode == "obs":
            if x_obs is None:
                raise ValueError(f"{path} does not contain observational samples.")
            x = _as_two_channel(x_obs)
        elif x_obs is not None and x_int is not None:
            x = np.concatenate([_as_two_channel(x_obs), _as_two_channel(x_int)], axis=0)
        elif x_obs is not None:
            x = _as_two_channel(x_obs)
        else:
            raise ValueError(f"{path} does not contain a recognized data array.")

    if x.ndim != 3 or x.shape[-1] != 2:
        raise ValueError(f"{path} produced invalid TabCausal input shape {x.shape}.")
    feature_mask = _as_feature_mask(feature_mask_raw, x.shape[1])
    x = x[:, feature_mask, :]
    f_value = int(x.shape[1])
    if adjacency is None:
        adj = None
    else:
        adj_full = np.asarray(adjacency, dtype=np.int64)
        adj = adj_full[np.ix_(feature_mask, feature_mask)]
    return GraphExample(path=path, f_value=f_value, x=x.astype(np.float32, copy=False), adjacency=adj)


def load_input_file(
    path: str | Path,
    *,
    mode: Mode = "auto",
    intervention_path: str | Path | None = None,
) -> GraphExample:
    """Load one inference input from benchmark NPZ or table-like files.

    For ``.npz`` files this preserves the benchmark schema and optional ground
    truth graph.  For table-like files, the loaded numeric table is interpreted
    as observational data.  Pass ``intervention_path`` to provide a same-shaped
    binary intervention indicator table.
    """

    path = Path(path)
    if path.suffix.lower() == ".npz" and intervention_path is None:
        return load_graph_npz(path, mode=mode)

    values = _load_array_file(path)
    x = _as_two_channel(values)
    if intervention_path is not None:
        flags = np.asarray(_load_array_file(intervention_path), dtype=np.float32)
        if flags.shape != x[..., 0].shape:
            raise ValueError(f"Intervention flags {flags.shape} do not match values {x[..., 0].shape}.")
        x = np.stack([x[..., 0], flags], axis=-1)
    return GraphExample(path=path, f_value=int(x.shape[1]), x=x.astype(np.float32, copy=False), adjacency=None)


def normalize_batch(x: torch.Tensor, *, clip: float = 10.0) -> torch.Tensor:
    """Standardize each variable within each graph and preserve intervention flags."""

    values = x[..., 0]
    flags = x[..., 1]
    mean = values.mean(dim=1, keepdim=True)
    std = values.std(dim=1, keepdim=True, unbiased=False)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    values = ((values - mean) / std).clamp(-clip, clip)
    return torch.stack([values, flags], dim=-1)


def stack_examples(examples: list[GraphExample]) -> torch.Tensor:
    """Stack examples with identical shapes into a batch tensor."""

    shapes = {example.x.shape for example in examples}
    if len(shapes) != 1:
        raise ValueError(f"Cannot batch different shapes: {sorted(shapes)}")
    return torch.from_numpy(np.stack([example.x for example in examples], axis=0))
