"""Read-only collection of independent Bellman-lookahead policy evidence."""
from __future__ import annotations

from numbers import Integral
from pathlib import Path


def bellman_policy_study(context, *, seed=42, compute_figures=True):
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed != 42:
        raise ValueError("The declared Bellman research seed is42")
    if not isinstance(compute_figures, bool):
        raise TypeError("compute_figures must be boolean")
    context.progress("Verifying independent state-dependent Bellman lookahead policies")
    folder = Path(context.output_dir).resolve()/"rough_bellman_policy_v1/one_step_v1"
    from ._pricing_rough_bellman_collection import collect_bellman_campaign
    return collect_bellman_campaign(folder, compute_figures=compute_figures)
