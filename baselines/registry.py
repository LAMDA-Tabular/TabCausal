"""Baseline registry and implementation notes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineInfo:
    name: str
    display_name: str
    regimes: tuple[str, ...]
    status: str
    package_hint: str
    note: str


BASELINES: dict[str, BaselineInfo] = {
    "randomregress": BaselineInfo(
        "randomregress",
        "RandomRegress",
        ("obs",),
        "portable",
        "scikit-learn",
        "Regression/ranking control baseline implemented in this release.",
    ),
    "pc": BaselineInfo(
        "pc",
        "PC",
        ("obs",),
        "optional_dependency",
        "causal-learn",
        "Uses causal-learn PC; CPDAG undirected edges are oriented deterministically by node index.",
    ),
    "lingam": BaselineInfo(
        "lingam",
        "LiNGAM",
        ("obs",),
        "optional_dependency",
        "causal-learn",
        "Uses causal-learn's DirectLiNGAM and binarizes the learned adjacency by absolute coefficient.",
    ),
    "gies": BaselineInfo(
        "gies",
        "GIES/GES",
        ("obs", "int"),
        "included_runner",
        "Included GIES wrapper plus cdt/R/rpy2/pcalg dependencies",
        "Uses interventional targets when available; with no targets this is the observational GES setting.",
    ),
    "cdis": BaselineInfo(
        "cdis",
        "CDIS",
        ("obs", "int"),
        "included_runner",
        "Included CDIS implementation plus optional causal-learn-style dependencies",
        "Requires observational rows plus intervention groups; PAG uncertain marks are oriented deterministically by node index.",
    ),
    "igsp": BaselineInfo("igsp", "IGSP", ("obs", "int"), "included_runner", "Included IGSP wrapper plus causaldag", "Interventional permutation baseline."),
    "notears": BaselineInfo("notears", "NOTEARS", ("obs",), "included_runner", "Included official NOTEARS linear implementation", "Continuous optimization baseline."),
    "notears_mlp": BaselineInfo("notears_mlp", "NOTEARS-MLP", ("obs",), "included_runner", "Included official NOTEARS nonlinear implementation", "Nonlinear continuous optimization baseline."),
    "dagma": BaselineInfo("dagma", "DAGMA", ("obs",), "included_runner", "Included DAGMA implementation", "Continuous optimization baseline."),
    "sdcd": BaselineInfo("sdcd", "SDCD", ("obs", "int"), "included_runner", "Included SDCD implementation", "Differentiable causal discovery baseline."),
    "dcdi": BaselineInfo("dcdi", "DCDI", ("obs", "int"), "included_runner", "Included DCDI implementation", "Iterative neural optimization baseline."),
    "avici": BaselineInfo("avici", "AVICI", ("obs", "int"), "included_runner", "Included official AVICI source plus released checkpoint", "Amortized baseline with its released checkpoint."),
    "sea": BaselineInfo("sea", "SEA", ("obs", "int"), "included_runner", "Included SEA implementation and checkpoints", "Sample-estimate-aggregate pretrained baseline."),
    "das": BaselineInfo("das", "DAS", ("obs",), "included_runner", "Included dodiscover DAS implementation", "Ordering/control baseline."),
    "nodags": BaselineInfo("nodags", "NoDAGS", ("int",), "included_runner", "Included Bicycle/NoDAGS implementation", "Interventional neural baseline."),
}


def available_baseline_names() -> list[str]:
    return sorted(BASELINES)
