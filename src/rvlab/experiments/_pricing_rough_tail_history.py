"""Verified stopping-gap and physical-time evidence before large-tail attribution.

Replay preserves original paths, fitted-rule identities, paired statistics, settings and costs.
It never reruns the spent campaign under the source of a later experiment.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pandas as pd


ARCHIVE = (Path(__file__).resolve().parents[3] /
    "notebooks/legacy/pricing/rough_tail_attribution_2026-10-19")
GAP_TIME_PROFILE = "bd65fbcb240dd3d88001b55f355be067101252b9e76b323f16a5eae663c2cbca"


def load_historical_gap_time_refinement(context=None, *, archive=None, compute_figures=True):
    """Load all verified stopping/delta tables and original PNGs without computation."""
    from ._pricing_rough_precision import _read_verified_archive

    if not isinstance(compute_figures, bool):
        raise TypeError("compute_figures must be boolean")
    archive = ARCHIVE if archive is None else Path(archive)
    inventory, digest = _read_verified_archive(archive)
    recorded = json.loads((archive / "results/manifest.json").read_text())
    stage = recorded.get("rough_gap_time_refinement", {})
    if (recorded.get("notebook") != "01_pricing_and_exercise" or
            recorded.get("evidence_version") != "pricing_research_v1" or
            stage.get("backend") != "rough_heston_gap_time_refinement" or
            stage.get("profile_fingerprint") != GAP_TIME_PROFILE or
            stage.get("primary_complete") is not True or
            not isinstance(stage.get("input_files"), list) or
            len(stage["input_files"]) != 252 or
            len({row.get("path") for row in stage["input_files"]}) != 252):
        raise ValueError("Preserved evidence is not the declared complete stopping-gap/time-refinement campaign")
    prefix, tables = "rough_gap_time_refinement_", {}
    for name, count in recorded["tables"].items():
        if not name.startswith(prefix):
            continue
        try:
            table = pd.read_csv(archive / "results" / (name + ".csv"),
                float_precision="round_trip")
        except pd.errors.EmptyDataError:
            table = pd.DataFrame()
        if len(table) != count:
            raise ValueError(f"Changed historical gap/time table count: {name}")
        tables[name.removeprefix(prefix)] = table
    figures = ({name.removeprefix(prefix): archive / "results" / (name + ".png")
        for name in recorded["figures"] if name.startswith(prefix)}
        if compute_figures else {})
    manifest = deepcopy(stage)
    manifest.update(historical_replay=True, current_engine_execution=False,
        historical_archive=str(archive.resolve()),
        historical_archive_manifest_sha256=digest,
        historical_archive_files=len(inventory["files"]),
        historical_execution_created_at=recorded["created_at"],
        american_pricing_ready=False, stock_delta_ready=False)
    if context is not None:
        context.progress("Loading verified gap/time refinement; no prior path, control, fitted rule or sensitivity is recomputed")
    return {"tables": tables, "figures": figures, "manifest": manifest}
