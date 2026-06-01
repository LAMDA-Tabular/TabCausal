#!/usr/bin/env python3
"""Generate the seven-family TabCausal synthetic benchmark suite."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import math
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm.auto import tqdm

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for data-only use
    torch = None


CONFIGURED_FAMILIES = ("gp_hard", "gp_simple", "linear_gauss", "linear_graph", "linear_nongauss", "mul_noise")
PFN_FAMILY = "pfn"
DEFAULT_FAMILIES = (*CONFIGURED_FAMILIES, PFN_FAMILY)
DEFAULT_REGIMES = ("obs", "int")


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_family_config(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected FAMILY=CONFIG for --family-config, got: {item}")
        family, path = item.split("=", 1)
        family = family.strip()
        path = path.strip()
        if not family or not path:
            raise ValueError(f"Invalid --family-config entry: {item}")
        out[family] = path
    return out


def _seed_global_generators(seed: int) -> None:
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def _edges_per_var_for_f(f_value: int) -> list[float]:
    table = {
        5: [0.8, 1.0, 1.2],
        10: [1.2, 1.5, 1.8],
        20: [1.6, 2.0, 2.4],
        50: [2.0, 2.5, 3.0],
        100: [2.4, 3.0, 3.6],
        300: [3.0, 3.6, 4.2],
    }
    if f_value in table:
        return table[f_value]
    base = 2.0 + 0.012 * max(f_value - 20, 0)
    return [round(base * 0.8, 1), round(base, 1), round(base * 1.2, 1)]


def _edges_per_var_for_graph(f_value: int, graph_class: str | None) -> list[float] | list[int]:
    values = _edges_per_var_for_f(f_value)
    if graph_class in {"ScaleFree", "ScaleFreeTranspose"}:
        return [max(1, int(math.floor(v + 0.5))) for v in values]
    return values


def _resolve_config(family: str, config_root: Path, config_map: dict[str, str]) -> Path:
    if family in config_map:
        path = Path(config_map[family])
        return path if path.is_absolute() else Path.cwd() / path
    candidates = [config_root / f"{family}.yaml", config_root / f"{family}.yml"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No config found for family '{family}' under {config_root}.")


def _materialize_config_for_f(
    src_path: Path,
    f_value: int,
    *,
    output_dir: Path,
    adjust_density: bool = True,
) -> tuple[Path, str]:
    with src_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if adjust_density and "data" in config:
        for data_block in config["data"]:
            for graph_spec in data_block.get("graph", []):
                if "edges_per_var" in graph_spec:
                    graph_spec["edges_per_var"] = _edges_per_var_for_graph(
                        f_value,
                        graph_spec.get("__class__"),
                    )

    output_dir.mkdir(parents=True, exist_ok=True)
    domain = f"temp_{src_path.stem}_{f_value}_{uuid.uuid4().hex}"
    out_path = output_dir / f"{domain}.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f)
    return out_path, domain


def _load_configured_simulator(engine_root: Path, module_name: str, function_name: str):
    engine_root = engine_root.resolve()
    package_dir = engine_root / module_name
    init_file = package_dir / "__init__.py"
    if not init_file.exists():
        raise FileNotFoundError(f"Cannot find included simulator package: {init_file}")

    for name in list(sys.modules):
        if name == module_name or name.startswith(module_name + "."):
            del sys.modules[name]

    sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != engine_root]
    sys.path.insert(0, str(engine_root))

    spec = importlib.util.spec_from_file_location(
        module_name,
        init_file,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load included simulator package from {init_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    module_file = Path(getattr(module, "__file__", "")).resolve()
    if engine_root not in module_file.parents:
        raise ImportError(f"Expected {module_name!r} from {engine_root}, but imported {module_file}.")
    return getattr(module, function_name)


def _to_array(value: Any | None) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value)


def _mask_from_interventions(interventions: np.ndarray | None, n: int, d: int) -> np.ndarray:
    if interventions is None:
        return np.zeros((n, d), dtype=np.float32)
    arr = np.asarray(interventions)
    if arr.shape == (n, d):
        return (arr > 0).astype(np.float32)
    if arr.shape == (n,):
        mask = np.zeros((n, d), dtype=np.float32)
        valid = (arr >= 0) & (arr < d)
        mask[np.arange(n)[valid], arr[valid].astype(int)] = 1.0
        return mask
    if arr.ndim == 3 and arr.shape[:2] == (n, d):
        return (arr[..., -1] > 0).astype(np.float32)
    raise ValueError(f"Unsupported intervention mask shape: {arr.shape}")


def _pack_values_and_mask(values: np.ndarray, interventions: np.ndarray | None) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[-1] >= 2:
        packed = values[..., :2].astype(np.float32)
        packed[..., 1] = (packed[..., 1] > 0).astype(np.float32)
        return packed
    if values.ndim != 2:
        raise ValueError(f"Expected data shape [n,d] or [n,d,2], got {values.shape}.")
    n, d = values.shape
    mask = _mask_from_interventions(interventions, n=n, d=d)
    return np.stack([values.astype(np.float32), mask], axis=-1)


def _call_configured_simulator(
    simulate,
    *,
    d: int,
    n_obs: int,
    n_int: int,
    seed: int,
    yaml_path: Path | None = None,
    domain: str | None = None,
):
    if (yaml_path is None) == (domain is None):
        raise ValueError("Specify exactly one of yaml_path or domain.")
    location_kwargs = {"domain": domain} if domain is not None else {"path": str(yaml_path)}
    try:
        return simulate(d=d, n=n_obs, n_interv=n_int, seed=seed, **location_kwargs)
    except TypeError as exc:
        message = str(exc)
        if "unexpected keyword" not in message and "positional" not in message:
            raise
        return simulate(
            d=d,
            n_observations=n_obs,
            n_interventions=n_int,
            seed=seed,
            **location_kwargs,
        )


def _generate_configured_graph(
    *,
    output_root: Path,
    family: str,
    regime: str,
    d: int,
    idx: int,
    yaml_path: Path,
    simulate,
    n_obs: int,
    n_int: int,
    seed: int,
    overwrite: bool,
    no_density_adjust: bool,
) -> bool:
    dataset_dir = output_root / f"[{family}]_{regime}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    out_path = dataset_dir / f"data_f{d}_{idx:03d}.npz"
    if out_path.exists() and not overwrite:
        return False

    _seed_global_generators(seed)
    simulator_seed = int(np.random.default_rng(seed).integers(0, 1_000_000_000))
    temp_yaml: Path | None = None
    temp_config_dir: tempfile.TemporaryDirectory[str] | None = None
    if no_density_adjust:
        temp_yaml = yaml_path
    else:
        temp_config_dir = tempfile.TemporaryDirectory(prefix="tabcausal_family_config_")
        temp_yaml, _ = _materialize_config_for_f(
            yaml_path,
            d,
            output_dir=Path(temp_config_dir.name),
        )

    try:
        result = _call_configured_simulator(
            simulate,
            d=d,
            n_obs=n_obs,
            n_int=n_int,
            seed=simulator_seed,
            yaml_path=temp_yaml,
        )
    finally:
        if temp_yaml is not None and temp_yaml != yaml_path and temp_yaml.exists():
            temp_yaml.unlink()
        if temp_config_dir is not None:
            temp_config_dir.cleanup()

    if not isinstance(result, tuple) or len(result) < 2:
        raise ValueError("Simulator must return at least (graph, data).")
    graph = np.asarray(result[0]).astype(np.int64)
    values = _to_array(result[1])
    interventions = _to_array(result[2]) if len(result) >= 3 else None
    if values is None:
        raise ValueError("Simulator returned no data array.")
    x = _pack_values_and_mask(values, interventions)
    feature_mask = np.ones(graph.shape[0], dtype=np.float32)
    np.savez_compressed(out_path, x=x, g=graph, mask=feature_mask, n_obs=n_obs, n_int=n_int, f=d)
    return True


class OODPFNDatasetGenerator:
    def __init__(self, base_cls, *, batch_size: int, min_f: int, max_f: int, seed: int):
        self._gen = base_cls(
            config_mode="ood_eval",
            batch_size=batch_size,
            min_f=min_f,
            max_f=max_f,
            seed=seed,
        )

    def _sample_old_pfn_mlp_hparams(self, f: int):
        rng = self._gen.rng
        num_layers = int(rng.integers(3, min(9, max(4, f // 2 + 2))))
        hidden_floor = max(2 * f + 4, 32)
        hidden_ceiling = max(hidden_floor + 1, 10 * f + 64)
        num_causes = int(rng.integers(2, max(3, f)))
        return {
            "num_layers": num_layers,
            "prior_mlp_hidden_dim": int(rng.integers(hidden_floor, hidden_ceiling)),
            "prior_mlp_dropout_prob": float(rng.uniform(0.10, 0.75)),
            "noise_std": float(np.exp(rng.uniform(np.log(0.02), np.log(0.8)))),
            "init_std": float(np.exp(rng.uniform(np.log(0.05), np.log(12.0)))),
            "num_causes": max(2, min(num_causes, f - 1 if f > 2 else f)),
            "is_causal": True,
            "pre_sample_weights": bool(rng.integers(0, 2)),
            "y_is_effect": bool(rng.integers(0, 2)),
            "sampling": str(rng.choice(["normal", "mixed"])),
            "prior_mlp_activations": str(rng.choice(["relu", "tanh", "identity"])),
            "block_wise_dropout": bool(rng.integers(0, 2)),
            "sort_features": bool(rng.integers(0, 2)),
            "in_clique": bool(rng.integers(0, 2)),
            "pre_sample_causes": True,
            "prior_mlp_scale_weights_sqrt": True,
            "random_feature_rotation": True,
        }

    def generate_single_test(self, *args, **kwargs):
        self._gen._sample_old_pfn_mlp_hparams = self._sample_old_pfn_mlp_hparams
        return self._gen.generate_single_test(*args, **kwargs)


def _load_pfn_generator(pfn_engine_root: Path):
    pfn_engine_root = pfn_engine_root.resolve()
    sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != pfn_engine_root]
    sys.path.insert(0, str(pfn_engine_root))
    for name in list(sys.modules):
        if name == "pfn_dataset":
            del sys.modules[name]
    module = importlib.import_module("pfn_dataset")
    return getattr(module, "PFNDatasetGenerator")


def _generate_pfn_graph(
    *,
    output_root: Path,
    regime: str,
    d: int,
    idx: int,
    pfn_generator_cls,
    n_obs: int,
    n_int: int,
    seed: int,
    overwrite: bool,
    pfn_standard: bool,
) -> bool:
    dataset_dir = output_root / f"[{PFN_FAMILY}]_{regime}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    out_path = dataset_dir / f"data_f{d}_{idx:03d}.npz"
    if out_path.exists() and not overwrite:
        return False

    _seed_global_generators(seed)
    if pfn_standard:
        gen = pfn_generator_cls(config_mode="default", batch_size=1, min_f=d, max_f=d, seed=seed)
    else:
        gen = OODPFNDatasetGenerator(pfn_generator_cls, batch_size=1, min_f=d, max_f=d, seed=seed)
    data = gen.generate_single_test(f=d, n_obs=n_obs, n_int=n_int, pad_to_dim=None)
    np.savez_compressed(
        out_path,
        x=data["x"],
        g=data["g"],
        mask=data["mask"],
        f=data["f"],
        n_obs=n_obs,
        n_int=n_int,
    )
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    parser.add_argument("--regimes", default=",".join(DEFAULT_REGIMES))
    parser.add_argument("--f-values", default="5,10,20")
    parser.add_argument("--graphs-per-f", type=int, default=10)
    parser.add_argument("--observations", type=int, default=1000)
    parser.add_argument("--interventions", type=int, default=200)
    parser.add_argument(
        "--mixed-observations",
        type=int,
        default=None,
        help=(
            "Observational rows used in mixed-interventional datasets. "
            "Default: observations - interventions, matching the paper benchmark "
            "where 1000 total rows are split into 800 observational + 200 interventional."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Disable progress messages.")
    parser.add_argument(
        "--engine-root",
        default="data_engine/third_party/avici_data_engine",
        help="Simulator root for the benchmark data generator.",
    )
    parser.add_argument("--engine-module", default="avici")
    parser.add_argument("--simulate-function", default="simulate_data")
    parser.add_argument("--config-root", default="data_engine/configs/bench")
    parser.add_argument(
        "--family-config",
        action="append",
        default=[],
        help="Explicit FAMILY=CONFIG mapping. May be repeated and overrides --config-root.",
    )
    parser.add_argument("--no-density-adjust", action="store_true", help="Do not create f-specific family configs.")
    parser.add_argument("--pfn-engine-root", default="data_engine/pfn_engine")
    parser.add_argument(
        "--pfn-standard",
        action="store_true",
        help="Use the base PFN generator instead of the benchmark OOD evaluation variant.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root

    families = _parse_csv(args.families)
    unknown = sorted(set(families) - set(DEFAULT_FAMILIES))
    if unknown:
        raise ValueError(f"Unknown families: {', '.join(unknown)}")

    regimes = _parse_csv(args.regimes)
    f_values = _parse_csv_ints(args.f_values)
    mixed_observations = args.mixed_observations
    if mixed_observations is None:
        mixed_observations = args.observations - args.interventions if "int" in regimes else args.observations
    if "int" in regimes and mixed_observations < 0:
        raise ValueError(
            "--mixed-observations is negative. Use --mixed-observations explicitly "
            "or choose --observations >= --interventions."
        )

    engine_root = Path(args.engine_root)
    if not engine_root.is_absolute():
        engine_root = repo_root / engine_root
    config_root = Path(args.config_root)
    if not config_root.is_absolute():
        config_root = repo_root / config_root
    pfn_engine_root = Path(args.pfn_engine_root)
    if not pfn_engine_root.is_absolute():
        pfn_engine_root = repo_root / pfn_engine_root

    config_map = _parse_family_config(args.family_config)
    if "linear_graph" not in config_map:
        config_map["linear_graph"] = str(config_root / "linear_struct.yaml")

    simulate = None
    if any(family in CONFIGURED_FAMILIES for family in families):
        simulate = _load_configured_simulator(engine_root, args.engine_module, args.simulate_function)

    pfn_generator_cls = None
    if PFN_FAMILY in families:
        pfn_generator_cls = _load_pfn_generator(pfn_engine_root)

    rng = np.random.default_rng(args.seed)
    tasks = [
        (family, regime, d, idx)
        for family in families
        for regime in regimes
        for d in f_values
        for idx in range(args.graphs_per_f)
    ]

    generated = 0
    skipped = 0
    iterator = tqdm(tasks, desc="Synthetic suite", unit="graph", disable=args.quiet)
    for family, regime, d, idx in iterator:
        iterator.set_postfix_str(f"{family}/{regime}/f={d}")
        n_obs = args.observations if regime == "obs" else mixed_observations
        n_int = 0 if regime == "obs" else args.interventions
        seed = int(rng.integers(0, 2**31 - 1))

        if family == PFN_FAMILY:
            assert pfn_generator_cls is not None
            did_generate = _generate_pfn_graph(
                output_root=output_root,
                regime=regime,
                d=d,
                idx=idx,
                pfn_generator_cls=pfn_generator_cls,
                n_obs=n_obs,
                n_int=n_int,
                seed=seed,
                overwrite=args.overwrite,
                pfn_standard=args.pfn_standard,
            )
        else:
            assert simulate is not None
            yaml_path = _resolve_config(family, config_root, config_map)
            did_generate = _generate_configured_graph(
                output_root=output_root,
                family=family,
                regime=regime,
                d=d,
                idx=idx,
                yaml_path=yaml_path,
                simulate=simulate,
                n_obs=n_obs,
                n_int=n_int,
                seed=seed,
                overwrite=args.overwrite,
                no_density_adjust=args.no_density_adjust,
            )

        if did_generate:
            generated += 1
        else:
            skipped += 1

    print(f"generated_graphs={generated}")
    print(f"skipped_existing={skipped}")
    print(f"output_root={output_root}")


if __name__ == "__main__":
    main()
