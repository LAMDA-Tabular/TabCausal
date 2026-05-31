"""Reusable benchmark evaluation utilities."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .inference import TabCausalPredictor
from .metrics import binary_graph_metrics, summarize_rows
from .preprocessing import GraphExample, load_graph_npz, stack_examples


F_RE = re.compile(r"(?:^|[_-])f(?P<f>\d+)(?:[_-]|\\.|$)")


def infer_f_from_path(path: Path, fallback: int | None = None) -> int:
    """Infer graph size ``f`` from a benchmark filename."""

    match = F_RE.search(path.name)
    if match:
        return int(match.group("f"))
    if fallback is not None:
        return int(fallback)
    raise ValueError(f"Could not infer f-value from {path}")


def find_npz_files(data_root: str | Path) -> list[Path]:
    """Find benchmark graph files in deterministic order."""

    root = Path(data_root)
    files = sorted(path for path in root.rglob("*.npz") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No .npz files found under {root}")
    return files


def select_files_by_f(
    files: list[Path],
    max_per_f: int | None = None,
    *,
    file_offset: int = 0,
) -> list[Path]:
    """Optionally keep a slice of files for each f-value."""

    if max_per_f is None or max_per_f <= 0:
        return files[file_offset:] if file_offset > 0 else files
    if file_offset < 0:
        raise ValueError("file_offset must be non-negative.")
    seen: dict[int, int] = {}
    selected: list[Path] = []
    for path in files:
        f_value = infer_f_from_path(path, fallback=None)
        count = seen.get(f_value, 0)
        if file_offset <= count < file_offset + max_per_f:
            selected.append(path)
        seen[f_value] = count + 1
    return selected


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_directory(
    checkpoint: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    *,
    mode: str = "auto",
    threshold: float = 0.5,
    max_per_f: int | None = None,
    batch_size: int = 8,
    device: str | None = None,
    dtype: str = "float32",
    prefer_ema: bool = True,
    use_amp: bool = True,
    max_observations: int | None = None,
    observation_seed: int = 0,
    file_offset: int = 0,
    progress: bool = True,
    save_adjacencies: bool = True,
    save_embeddings: bool = False,
    compute_sid: bool = True,
    prefer_official_sid: bool = True,
) -> dict[str, Path]:
    """Run TabCausal on a directory of benchmark ``.npz`` files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_dtype = None
    if dtype == "float16":
        import torch

        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        import torch

        torch_dtype = torch.bfloat16

    predictor = TabCausalPredictor(
        checkpoint,
        device=device,
        dtype=torch_dtype,
        normalize=True,
        prefer_ema=prefer_ema,
        use_amp=use_amp,
        max_observations=max_observations,
        observation_seed=observation_seed,
    )
    files = select_files_by_f(find_npz_files(data_root), max_per_f=max_per_f, file_offset=file_offset)
    if progress:
        print(f"[evaluate] data_root={data_root} files={len(files)} batch_size={batch_size}", flush=True)
    examples = [load_graph_npz(path, mode=mode) for path in files]

    rows: list[dict[str, object]] = []
    pred_probs: list[np.ndarray] = []
    pred_logits: list[np.ndarray] = []
    pred_adjacencies: list[np.ndarray] = []
    pred_embeddings: list[np.ndarray] = []
    paths: list[str] = []
    adjacency_dir = output_dir / "adjacency_csv"
    probability_dir = output_dir / "probability_csv"
    embedding_dir = output_dir / "embedding_npy"
    if save_adjacencies:
        adjacency_dir.mkdir(parents=True, exist_ok=True)
        probability_dir.mkdir(parents=True, exist_ok=True)
    if save_embeddings:
        embedding_dir.mkdir(parents=True, exist_ok=True)

    starts = range(0, len(examples), batch_size)
    iterator = tqdm(starts, desc=f"Eval {Path(data_root).name}", unit="batch", disable=not progress)
    for start in iterator:
        batch = examples[start : start + batch_size]
        # Keep batching simple and safe: split further when shapes differ.
        shape_groups: dict[tuple[int, ...], list[GraphExample]] = {}
        for example in batch:
            shape_groups.setdefault(example.x.shape, []).append(example)
        for group in shape_groups.values():
            x_batch = stack_examples(group)
            logits = predictor.predict_logits(x_batch, batch_size=len(group))
            embeddings = predictor.predict_embeddings(x_batch, batch_size=len(group)) if save_embeddings else None
            probs = 1.0 / (1.0 + np.exp(-logits))
            for local_idx, (example, logit, prob) in enumerate(zip(group, logits, probs)):
                adjacency = (prob >= threshold).astype(np.int64)
                np.fill_diagonal(adjacency, 0)
                paths.append(str(example.path))
                pred_logits.append(logit)
                pred_probs.append(prob)
                pred_adjacencies.append(adjacency)
                row: dict[str, object] = {
                    "path": str(example.path),
                    "f": infer_f_from_path(example.path, fallback=example.f_value),
                }
                if example.adjacency is not None:
                    row.update(
                        binary_graph_metrics(
                            example.adjacency,
                            prob,
                            threshold=threshold,
                            sid=compute_sid,
                            prefer_official_sid=prefer_official_sid,
                        )
                    )
                rows.append(row)
                if save_adjacencies:
                    stem = example.path.stem
                    np.savetxt(adjacency_dir / f"{stem}_adjacency.csv", adjacency, delimiter=",", fmt="%d")
                    np.savetxt(probability_dir / f"{stem}_probabilities.csv", prob, delimiter=",", fmt="%.8g")
                if embeddings is not None:
                    embedding = embeddings[local_idx]
                    pred_embeddings.append(embedding)
                    np.save(embedding_dir / f"{example.path.stem}_embeddings.npy", embedding)

    _write_csv(output_dir / "raw_metrics.csv", rows)
    _write_csv(output_dir / "summary.csv", summarize_rows(rows))
    np.savez_compressed(
        output_dir / "predictions.npz",
        paths=np.asarray(paths),
        logits=np.asarray(pred_logits, dtype=object),
        probabilities=np.asarray(pred_probs, dtype=object),
        adjacencies=np.asarray(pred_adjacencies, dtype=object),
        embeddings=np.asarray(pred_embeddings, dtype=object) if pred_embeddings else np.asarray([], dtype=object),
    )
    outputs = {
        "raw_metrics": output_dir / "raw_metrics.csv",
        "summary": output_dir / "summary.csv",
        "predictions": output_dir / "predictions.npz",
    }
    if save_adjacencies:
        outputs["adjacency_csv"] = adjacency_dir
        outputs["probability_csv"] = probability_dir
    if save_embeddings:
        outputs["embedding_npy"] = embedding_dir
    return outputs
