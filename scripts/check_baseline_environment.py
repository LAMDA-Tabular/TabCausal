#!/usr/bin/env python3
"""Print resolved import paths for the paper baseline dependencies.

This is a read-only diagnostic helper. It does not install packages or modify
the active Python environment.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
import types
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "baselines" / "paper_algorithms"


def _prepend(path: Path) -> None:
    if path.exists():
        sys.path.insert(0, str(path))


def _module_path(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return f"MISSING ({type(exc).__name__}: {exc})"
    return str(getattr(module, "__file__", "<namespace>"))


def _install_wandb_stub() -> None:
    if "wandb" in sys.modules:
        return
    wandb = types.ModuleType("wandb")
    wandb.run = None
    wandb.init = lambda *args, **kwargs: None
    wandb.log = lambda *args, **kwargs: None
    wandb.finish = lambda *args, **kwargs: None
    sys.modules["wandb"] = wandb


def _avici_status() -> str:
    candidate_paths = [
        Path(os.environ["AVICI_ROOT"]).expanduser() if "AVICI_ROOT" in os.environ else None,
        PAPER / "avici_official",
    ]
    paths = [path for path in candidate_paths if path and (path / "avici" / "__init__.py").exists()]
    with _isolated_import(paths, ("avici",)):
        try:
            module = importlib.import_module("avici")
        except Exception as exc:
            return f"MISSING ({type(exc).__name__}: {exc})"
        path = str(getattr(module, "__file__", "<namespace>"))
        has_loader = hasattr(module, "load_pretrained")
        status = "official=yes" if has_loader else "official=no"
        return f"{path} ({status}, has_load_pretrained={has_loader})"


@contextmanager
def _isolated_import(paths: list[Path], clear_prefixes: tuple[str, ...] = ()):
    old_path = list(sys.path)
    old_modules = {
        name: module
        for name, module in sys.modules.items()
        if name in clear_prefixes or any(name.startswith(prefix + ".") for prefix in clear_prefixes)
    }
    for name in list(old_modules):
        del sys.modules[name]
    sys.path = [str(path) for path in paths if path.exists()] + old_path
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name in clear_prefixes or any(name.startswith(prefix + ".") for prefix in clear_prefixes):
                del sys.modules[name]
        sys.modules.update(old_modules)
        sys.path = old_path


def _module_path_with_paths(name: str, paths: list[Path], clear_prefixes: tuple[str, ...]) -> str:
    with _isolated_import(paths, clear_prefixes):
        return _module_path(name)


def main() -> None:
    paths = [
        PAPER,
        PAPER / "dagma" / "src",
        PAPER / "sdcd",
        PAPER / "sea-reproduce" / "src",
        PAPER / "Varsortability" / "src",
        PAPER / "dcdi",
        PAPER / "dcdi" / "gies",
        PAPER / "dcdi" / "igsp",
        PAPER / "bicycle" / "src",
        ROOT,
    ]
    for path in reversed(paths):
        _prepend(path)
    _install_wandb_stub()

    checks = [
        ("TabCausal", "tabcausal", None),
        ("NOTEARS", "notears.linear", None),
        ("NOTEARS-MLP", "notears.nonlinear", None),
        ("DAGMA", "dagma.nonlinear", None),
        ("SDCD", "sdcd", None),
        ("DCDI", "dcdi.models.learnables", None),
        ("DCDI-Flow", "dcdi.models.flows", None),
        ("GIES/GES", "gies", ([PAPER / "dcdi" / "gies"], ("gies",))),
        ("IGSP", "igsp", ([PAPER / "dcdi" / "igsp"], ("igsp",))),
        ("NoDAGS", "bicycle.nodags_files.nodags", None),
        ("DAS", "dodiscover.toporder", None),
        ("RandomRegress helper", "varsortability", None),
        ("SEA", "inference", ([PAPER / "sea-reproduce" / "src"], ("inference", "data", "model", "utils", "args"))),
        ("PC dependency", "causallearn", None),
        ("LiNGAM dependency", "causallearn.search.FCMBased.lingam", None),
        ("causaldag dependency", "causaldag", None),
        ("CDT dependency", "cdt", ([PAPER / "sea-reproduce" / "src"], ("cdt",))),
        ("rpy2 dependency", "rpy2", None),
    ]

    print(f"release_root: {ROOT}")
    print(f"paper_algorithms: {PAPER}")
    print(f"R executable: {shutil.which('R')}")
    print(f"Rscript executable: {shutil.which('Rscript')}")
    for label, module_name, special in checks:
        if special is None:
            path = _module_path(module_name)
        else:
            special_paths, clear_prefixes = special
            path = _module_path_with_paths(module_name, special_paths, clear_prefixes)
        print(f"{label:20s} {module_name:32s} {path}")
    print(f"{'AVICI official':20s} {'avici':32s} {_avici_status()}")


if __name__ == "__main__":
    main()
