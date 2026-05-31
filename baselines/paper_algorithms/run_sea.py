#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

_UTILS_CANDIDATES = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parent / "utils"),
    str(Path(__file__).resolve().parents[1] / "utils"),
]
for _candidate in _UTILS_CANDIDATES:
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from metrics import ResultTracker, dag_metrics, prob_metrics, save_predictions_incremental

FIXED_THRESHOLD = 0.5


def get_f_from_filename(path: str) -> int:
    try:
        match = re.search(r"_f(\d+)_", os.path.basename(path))
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return 0


def apply_threshold(P_pred: np.ndarray) -> np.ndarray:
    """Binarize an edge-probability matrix with the fixed release threshold."""
    P_bin = (P_pred >= FIXED_THRESHOLD).astype(float)
    np.fill_diagonal(P_bin, 0.0)
    return P_bin


def compute_metrics_from_scores(
    G_true: np.ndarray,
    P_pred: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """Compute graph metrics with the fixed release threshold."""
    metrics = prob_metrics(G_true, P_pred, thresh=FIXED_THRESHOLD)
    return metrics, apply_threshold(P_pred)


def _find_first_existing(candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def reset_experiment_dir(results_root: str, exp_name: str) -> str:
    exp_dir = os.path.join(results_root, exp_name)
    if os.path.isdir(exp_dir):
        shutil.rmtree(exp_dir)
    return exp_dir


def resolve_sea_root(cli_value: str | None) -> str:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    cwd = Path.cwd()
    candidates = [
        cli_value,
        os.environ.get("SEA_ROOT"),
        str(script_dir / "sea-reproduce"),
        str(cwd / "sea-reproduce"),
        str(cwd.parent / "sea-reproduce"),
        str(project_root / "sea-reproduce"),
        str(project_root.parent / "sea-reproduce"),
    ]
    filtered = [c for c in candidates if c]
    root = _find_first_existing(filtered)
    if root is None:
        searched = "\n  - " + "\n  - ".join(filtered)
        raise FileNotFoundError(
            "SEA repo not found. Set --sea_root or SEA_ROOT to a valid sea-reproduce checkout. "
            f"Searched:{searched}"
        )
    return os.path.abspath(root)


def resolve_checkpoint(sea_root: str, regime: str, cli_value: str | None) -> str:
    if cli_value:
        return cli_value
    rel = (
        "checkpoints/fci_synthetic/model_best_epoch=373_auprc=0.842.ckpt"
        if regime == "obs"
        else "checkpoints/gies_synthetic/model_best_epoch=535_auprc=0.849.ckpt"
    )
    ckpt = os.path.join(sea_root, rel)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"SEA checkpoint not found: {ckpt}")
    return ckpt


def resolve_config_file(sea_root: str, regime: str) -> str:
    rel = "config/aggregator_tf_fci.yaml" if regime == "obs" else "config/aggregator_tf_gies.yaml"
    config_path = os.path.join(sea_root, rel)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"SEA config file not found: {config_path}")
    return config_path


def build_runtime_config(
    base_config_file: str,
    output_config_file: str,
    manifest_csv: str,
    results_file: str,
    num_workers: int,
    fci_batches_inference: int,
) -> str:
    with open(base_config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    data_cfg = config.setdefault("data", {})
    data_cfg["data_file"] = manifest_csv
    data_cfg["results_file"] = results_file
    data_cfg["batch_size"] = 1
    data_cfg["num_workers"] = num_workers
    data_cfg["fci_batches_inference"] = fci_batches_inference

    with open(output_config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return output_config_file


def infer_regime_from_file(npz_path: str) -> str:
    data = np.load(npz_path)
    x = data["x"]
    mask = data["mask"].astype(bool) if "mask" in data else np.ones(data["g"].shape[0], dtype=bool)
    interv = x[..., 1][:, mask]
    return "int" if np.any(interv > 0) else "obs"


def _flatten_to_square(scores_1d: np.ndarray) -> np.ndarray:
    scores_1d = np.asarray(scores_1d, dtype=float)
    if scores_1d.ndim == 2:
        np.fill_diagonal(scores_1d, 0.0)
        return scores_1d

    length = int(scores_1d.size)
    n_float = (1.0 + math.sqrt(1.0 + 4.0 * length)) / 2.0
    n = int(round(n_float))
    if n * (n - 1) != length:
        raise ValueError(f"Cannot infer graph size from flattened SEA scores of length {length}")

    E = n * (n - 1) // 2
    # SEA's symmetrize enumerates upper-tri pairs (r,c) with r<c in row-major
    # order.  For each pair e=(r,c), Aggregator.compute_metrics_per_graph
    # returns:
    #   pred[:E][e]  = P(class1) = P(r→c)
    #   pred[E:][e]  = P(class2) = P(c→r)
    # Therefore score_matrix[i,j] should store P(i→j) directly.
    score_matrix = np.zeros((n, n), dtype=float)
    e = 0
    for r in range(n):
        for c in range(r + 1, n):
            score_matrix[r, c] = scores_1d[e]        # P(r→c)
            score_matrix[c, r] = scores_1d[E + e]    # P(c→r)
            e += 1
    return score_matrix


def convert_npz_to_sea_dataset(npz_path: str, output_dir: str, regime: str) -> tuple[np.ndarray, str]:
    data = np.load(npz_path)
    x = data["x"]
    g = data["g"]
    mask = data["mask"].astype(bool) if "mask" in data else np.ones(data["g"].shape[0], dtype=bool)

    X_vals = x[..., 0][:, mask].astype(np.float32)
    I_flags = x[..., 1][:, mask]
    G_true = g[mask][:, mask].astype(int)

    os.makedirs(output_dir, exist_ok=True)
    graph_path = os.path.join(output_dir, "DAG.npy")
    data_path = os.path.join(output_dir, "data.npy" if regime == "obs" else "data_interv.npy")
    interv_path = os.path.join(output_dir, "intervention.csv")

    np.save(graph_path, G_true)
    np.save(data_path, X_vals)

    with open(interv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in I_flags:
            writer.writerow(np.where(row > 0)[0].tolist())

    return G_true, data_path


def build_manifest_for_group(paths: list[str], work_dir: str, regime: str) -> tuple[str, dict[str, np.ndarray]]:
    csv_path = os.path.join(work_dir, "manifest.csv")
    truths: dict[str, np.ndarray] = {}
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fp_data", "fp_graph", "fp_regime", "split"])
        for idx, npz_path in enumerate(paths):
            stem = os.path.splitext(os.path.basename(npz_path))[0]
            item_key = f"{idx:04d}_{stem}"
            item_dir = os.path.join(work_dir, item_key)
            G_true, data_path = convert_npz_to_sea_dataset(npz_path, item_dir, regime)
            truths[item_key] = G_true
            writer.writerow(
                [
                    data_path,
                    os.path.join(item_dir, "DAG.npy"),
                    os.path.join(item_dir, "intervention.csv"),
                    "test",
                ]
            )
    return csv_path, truths


def run_sea_inference_group(
    python_executable: str,
    sea_root: str,
    checkpoint_path: str,
    base_config_file: str,
    manifest_csv: str,
    regime: str,
    save_dir: str,
    num_workers: int,
    fci_batches_inference: int,
) -> dict:
    algorithm = "fci" if regime == "obs" else "gies"
    num_edge_types = 8 if regime == "obs" else 5
    runtime_config_file = build_runtime_config(
        base_config_file=base_config_file,
        output_config_file=os.path.join(save_dir, "runtime_config.yaml"),
        manifest_csv=manifest_csv,
        results_file="results.pkl",
        num_workers=num_workers,
        fci_batches_inference=fci_batches_inference,
    )
    bootstrap_path = os.path.join(save_dir, "bootstrap_inference.py")
    bootstrap_code = f"""import os\nimport runpy\nimport sys\nimport torch\nimport numpy as np\n\nSEA_ROOT = {sea_root!r}\nif SEA_ROOT not in sys.path:\n    sys.path.insert(0, SEA_ROOT)\nSRC_ROOT = SEA_ROOT + '/src'\nif SRC_ROOT not in sys.path:\n    sys.path.insert(0, SRC_ROOT)\n\n# NumPy compatibility for older GIES implementation.\nif not hasattr(np, 'bool'):\n    np.bool = np.bool_\n\n# SEA samplers assume training-style dataset sizes. Some semantic benchmark\n# scenarios are smaller, so when a sampler asks for more examples than exist,\n# we only then fall back to sampling with replacement.\n_orig_choice = np.random.choice\n\ndef _safe_choice(a, size=None, replace=False, p=None):\n    try:\n        population = int(a) if np.isscalar(a) else len(a)\n    except Exception:\n        population = None\n    if population is not None and not replace and size is not None:\n        requested = int(np.prod(size)) if isinstance(size, tuple) else int(size)\n        if requested > population:\n            replace = True\n    return _orig_choice(a, size=size, replace=replace, p=p)\n\nnp.random.choice = _safe_choice\n\n# PyTorch 2.6 compatibility for older SEA checkpoints.\n_orig_load = torch.load\n\ndef _patched_load(*args, **kwargs):\n    kwargs['weights_only'] = False\n    return _orig_load(*args, **kwargs)\n\ntorch.load = _patched_load\n\n# SEA assumes only a restricted subset of FCI endpoint combinations, but\n# newer causallearn versions / broader benchmarks may produce more PAG edge\n# patterns. Unknown pairs are mapped to the most generic non-directional PAG\n# class already seen during pretraining so inference can proceed.\nfrom data import utils as sea_utils\n\ndef _patched_convert_result_to_lg(g, edge_map):\n    edge_attr = []\n    # Scope the FCI compatibility patch to FCI only. In GIES, (1, 1)\n    # is the official no-edge code; mapping it to an FCI fallback makes\n    # interventional SEA massively over-dense.\n    is_fci = (2, 2) in edge_map and (4, 4) in edge_map and (0, 0) not in edge_map\n    fci_no_edge_type = edge_map.get((2, 2), 1)\n    fci_fallback = edge_map.get((4, 4), max(edge_map.values()))\n    for i in range(len(g)):\n        for j in range(len(g)):\n            if i == j:\n                continue\n            ij = g[i, j]\n            ji = g[j, i]\n            pair = (ij, ji)\n            if is_fci and pair == (1, 1):\n                edge_attr.append(fci_no_edge_type)\n            elif is_fci:\n                edge_attr.append(edge_map.get(pair, fci_fallback))\n            else:\n                edge_attr.append(edge_map[pair])\n    return edge_attr\n\nsea_utils.convert_result_to_lg = _patched_convert_result_to_lg\nimport data.dataset as sea_dataset_mod\nsea_dataset_mod.convert_result_to_lg = _patched_convert_result_to_lg\n\nimport utils as sea_top_utils\n_orig_save_pickle = sea_top_utils.save_pickle\n_SEA_SAVE_PATH = None\n\ndef _patched_save_pickle(fp, data):\n    global _SEA_SAVE_PATH\n    if not os.path.isabs(fp) and _SEA_SAVE_PATH:\n        fp = os.path.join(_SEA_SAVE_PATH, fp)\n    os.makedirs(os.path.dirname(fp) or '.', exist_ok=True)\n    print('SEA_SAVE_PICKLE', fp, flush=True)\n    return _orig_save_pickle(fp, data)\n\nsea_top_utils.save_pickle = _patched_save_pickle\n\n# SEA's interventional sampler assumes enough examples per regime for\n# sampling without replacement. Our mixed benchmarks can be sparser, so we\n# fall back to replacement only when a regime is too small.\nfrom data import samplers as sea_samplers\n\ndef _safe_interventional_sample_batches(self, num_batches, batch_size, num_vars_batch):\n    batches = []\n    points_per_env = batch_size // (num_vars_batch + 1)\n    for _ in range(num_batches):\n        nodes = self.sample_nodes(num_vars_batch)\n        reg_idx = []\n        for v in nodes:\n            reg_idx.extend(self.dataset.node_to_regime[v])\n        if len(reg_idx) < num_vars_batch:\n            reg_idx = sorted(set(reg_idx))\n            for _pad in range(num_vars_batch - len(reg_idx)):\n                reg_idx.append(0)\n        else:\n            reg_idx = np.random.choice(sorted(set(reg_idx)), num_vars_batch, replace=False)\n        batch = []\n        for reg in reg_idx:\n            pool = self.dataset.regimes[reg]\n            replace = len(pool) < points_per_env\n            idxs = np.random.choice(pool, points_per_env, replace=replace)[:, np.newaxis]\n            batch.append(self.dataset.data[idxs, nodes])\n        obs_pool = self.dataset.regimes[0]\n        obs_replace = len(obs_pool) < points_per_env\n        idxs = np.random.choice(obs_pool, points_per_env, replace=obs_replace)[:, np.newaxis]\n        batch.append(self.dataset.data[idxs, nodes])\n        batch = np.stack(batch, axis=0)\n\n        node_renumber = {{node:i for i, node in enumerate(nodes)}}\n        regimes = [self.dataset.idx_to_regime[reg] for reg in reg_idx]\n        regimes = [[node_renumber.get(x) for x in reg] for reg in regimes]\n        regimes = [[x for x in reg if x is not None] for reg in regimes]\n        regimes.append([])\n\n        batches.append((batch, nodes, regimes))\n        self.callback([(batch, nodes, regimes)])\n\n    idxs = np.random.choice(len(self.dataset), (batch_size,), replace=len(self.dataset) < batch_size)\n    feats = sea_samplers.compute_features(self.dataset.data[idxs].T)\n    return batches, feats\n\nsea_samplers.InterventionalSampler.sample_batches = _safe_interventional_sample_batches\n\nimport args as sea_args_mod\n_orig_process_args = sea_args_mod.process_args\n_orig_parse_args = sea_args_mod.parse_args\n\ndef _normalize_output_paths(args):\n    global _SEA_SAVE_PATH\n    _SEA_SAVE_PATH = args.save_path\n    if not os.path.isabs(args.results_file):\n        args.results_file = os.path.join(args.save_path, args.results_file)\n    if not os.path.isabs(args.args_file):\n        args.args_file = os.path.join(args.save_path, args.args_file)\n    return args\n\ndef _patched_process_args(args):\n    _orig_process_args(args)\n    _normalize_output_paths(args)\n    print('SEA_ARGS_RESULTS_FILE', args.results_file, flush=True)\n    print('SEA_ARGS_ARGS_FILE', args.args_file, flush=True)\n    print('SEA_ARGS_SAVE_PATH', args.save_path, flush=True)\n\nsea_args_mod.process_args = _patched_process_args\n\ndef _patched_parse_args():\n    args = _orig_parse_args()\n    _normalize_output_paths(args)\n    print('SEA_ARGS_RESULTS_FILE', args.results_file, flush=True)\n    print('SEA_ARGS_ARGS_FILE', args.args_file, flush=True)\n    print('SEA_ARGS_SAVE_PATH', args.save_path, flush=True)\n    return args\n\nsea_args_mod.parse_args = _patched_parse_args\n\nrunpy.run_path(SEA_ROOT + '/src/inference.py', run_name='__main__')\n"""
    with open(bootstrap_path, "w", encoding="utf-8") as f:
        f.write(bootstrap_code)

    gpu_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_arg = "0" if gpu_env not in ("", "-1") else "-1"

    cmd = [
        python_executable,
        bootstrap_path,
        "--config_file", runtime_config_file,
        "--data_file", manifest_csv,
        "--save_path", save_dir,
        "--results_file", "results.pkl",
        "--run_name", f"sea_{algorithm}",
        "--gpu", gpu_arg,
        "--checkpoint_path", checkpoint_path,
        "--algorithm", algorithm,
        "--model", "aggregator",
        "--num_vars", "1000",
        "--num_edge_types", str(num_edge_types),
        "--embed_dim", "64",
        "--transformer_num_layers", "4",
        "--n_heads", "8",
        "--ffn_embed_dim", "8",
        "--batch_size", "1",
        "--fci_vars", "5",
        "--fci_batch_size", "500",
        "--fci_batches_inference", str(fci_batches_inference),
        "--num_workers", str(num_workers),
    ]
    proc = subprocess.run(
        cmd,
        cwd=sea_root,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "SEA inference failed.\n"
            f"STDOUT:\n{proc.stdout[-4000:]}\n\nSTDERR:\n{proc.stderr[-4000:]}"
        )

    results_path = os.path.join(save_dir, "results.pkl")
    if not os.path.exists(results_path):
        temp_root = Path(save_dir).parent
        candidate_pkls = sorted(str(p) for p in temp_root.rglob("*.pkl"))
        if len(candidate_pkls) == 1:
            results_path = candidate_pkls[0]
        elif len(candidate_pkls) > 1:
            preferred = [p for p in candidate_pkls if os.path.basename(p) == "results.pkl"]
            results_path = preferred[0] if preferred else candidate_pkls[0]
        else:
            tree_entries = []
            for p in sorted(temp_root.rglob("*")):
                try:
                    rel = p.relative_to(temp_root)
                except Exception:
                    rel = p
                suffix = "/" if p.is_dir() else ""
                tree_entries.append(f"{rel}{suffix}")
                if len(tree_entries) >= 200:
                    tree_entries.append("... (truncated)")
                    break
            raise FileNotFoundError(
                "SEA results file not found. "
                f"Expected: {results_path}\n"
                f"save_dir={save_dir}\n"
                f"runtime_config_file={runtime_config_file}\n"
                f"candidate_pkls={candidate_pkls}\n"
                "temp_tree=\n" + "\n".join(tree_entries) + "\n"
                f"STDOUT:\n{proc.stdout[-4000:]}\n\nSTDERR:\n{proc.stderr[-4000:]}"
            )
    with open(results_path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    parser = argparse.ArgumentParser("Run SEA baseline")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--max_per_f", type=int, default=-1)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--results_root", type=str, required=True)
    parser.add_argument("--save_preds", action="store_true")
    parser.add_argument("--sea_root", type=str, default=None)
    parser.add_argument("--sea_python", type=str, default=os.environ.get("SEA_PYTHON", sys.executable))
    parser.add_argument("--sea_obs_checkpoint", type=str, default=None)
    parser.add_argument("--sea_int_checkpoint", type=str, default=None)
    parser.add_argument("--sea_num_workers", type=int, default=0)
    parser.add_argument("--sea_batches_inference", type=int, default=500)
    parser.add_argument("--reset_exp_dir", action="store_true")
    args = parser.parse_args()
    if args.reset_exp_dir:
        reset_experiment_dir(args.results_root, args.exp_name)

    sea_root = resolve_sea_root(args.sea_root)

    if not os.path.exists(args.data_root):
        raise FileNotFoundError(f"Data root not found: {args.data_root}")

    files = sorted(
        os.path.join(args.data_root, f)
        for f in os.listdir(args.data_root)
        if f.endswith(".npz")
    )
    if not files:
        print(f"[WARN] No .npz files found in {args.data_root}", flush=True)
        return

    regime = infer_regime_from_file(files[0])
    checkpoint_path = resolve_checkpoint(
        sea_root,
        regime=regime,
        cli_value=args.sea_obs_checkpoint if regime == "obs" else args.sea_int_checkpoint,
    )
    config_file = resolve_config_file(sea_root, regime=regime)

    files_by_f = defaultdict(list)
    for path in files:
        files_by_f[get_f_from_filename(path)].append(path)

    tracker = ResultTracker(results_root=args.results_root, exp_name=args.exp_name, args=args)
    preds_to_save: dict[str, np.ndarray] = {}

    print(f"[SEA] sea_root={sea_root}", flush=True)
    print(f"[SEA] sea_python={args.sea_python}", flush=True)
    print(f"[SEA] regime={regime}, checkpoint={checkpoint_path}", flush=True)
    print(f"[SEA] config_file={config_file}", flush=True)
    print(f"[SEA] release_threshold={FIXED_THRESHOLD}", flush=True)

    for f_val in sorted(files_by_f.keys()):
        paths = files_by_f[f_val][: args.max_per_f] if args.max_per_f > 0 else files_by_f[f_val]
        if not paths:
            continue

        eval_paths = paths

        print(f"[SEA] Processing f={f_val} with {len(eval_paths)} files", flush=True)

        with tempfile.TemporaryDirectory(prefix=f"sea_f{f_val}_") as tmp_dir:
            manifest_csv, truths = build_manifest_for_group(eval_paths, tmp_dir, regime)
            save_dir = os.path.join(tmp_dir, "sea_outputs")
            os.makedirs(save_dir, exist_ok=True)
            results = run_sea_inference_group(
                python_executable=args.sea_python,
                sea_root=sea_root,
                checkpoint_path=checkpoint_path,
                base_config_file=config_file,
                manifest_csv=manifest_csv,
                regime=regime,
                save_dir=save_dir,
                num_workers=args.sea_num_workers,
                fci_batches_inference=args.sea_batches_inference,
            )

            pred_by_name: dict[str, np.ndarray] = {}
            for key, payload in results.items():
                if key.startswith("data_"):
                    key = key[5:]
                if key not in truths:
                    continue
                # For our use, each dataset should have one result entry.
                pred = np.asarray(payload["pred"][0], dtype=float)
                # Reconstruct per-edge scores: score_matrix[i,j] = P(i→j).
                pred_by_name[key] = _flatten_to_square(pred)

            for local_idx, path in enumerate(eval_paths):
                key = f"{local_idx:04d}_{os.path.splitext(os.path.basename(path))[0]}"
                G_true = truths[key]
                P_pred = pred_by_name[key]
                metrics, P_eval = compute_metrics_from_scores(G_true, P_pred)
                tracker.log(metrics, f=f_val, filename=os.path.basename(path))
                if args.save_preds:
                    preds_to_save[os.path.basename(path)] = P_eval

    summary = tracker.finalize()
    if args.save_preds and preds_to_save:
        save_predictions_incremental(
            os.path.join(tracker.exp_dir, "predictions.npz"),
            preds_to_save,
        )

    if summary is not None:
        print(summary, flush=True)


if __name__ == "__main__":
    main()
