"""Verified selected-policy evidence before further stopping-gap and time refinement.

Replay preserves original paths, fitted-rule identities, paired statistics, settings and costs.
It never reruns the spent campaign under the source of a later experiment.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pandas as pd


ARCHIVE = (Path(__file__).resolve().parents[3] /
    "notebooks/legacy/pricing/rough_gap_time_refinement_2026-10-19")
POLICY_VALIDATION_PROFILE = "c8a947fdfc2e322e6272df88a9ba8a17062c0bd047278c238a975d308cf3dcc0"


def load_historical_policy_validation(context=None, *, archive=None, compute_figures=True):
    """Load all verified stopping/delta tables and original PNGs without computation."""
    from ._pricing_rough_precision import _read_verified_archive

    if not isinstance(compute_figures, bool):
        raise TypeError("compute_figures must be boolean")
    archive = ARCHIVE if archive is None else Path(archive)
    inventory, digest = _read_verified_archive(archive)
    recorded = json.loads((archive / "results/manifest.json").read_text())
    stage = recorded.get("rough_policy_validation", {})
    if (recorded.get("notebook") != "01_pricing_and_exercise" or
            recorded.get("evidence_version") != "pricing_research_v1" or
            stage.get("backend") != "rough_heston_policy_validation" or
            stage.get("profile_fingerprint") != POLICY_VALIDATION_PROFILE or
            stage.get("primary_complete") is not True or
            not isinstance(stage.get("input_files"), list) or
            len(stage["input_files"]) != 119 or
            len({row.get("path") for row in stage["input_files"]}) != 119):
        raise ValueError("Preserved evidence is not the declared complete selected-policy validation campaign")
    prefix, tables = "rough_policy_validation_", {}
    for name, count in recorded["tables"].items():
        if not name.startswith(prefix):
            continue
        try:
            table = pd.read_csv(archive / "results" / (name + ".csv"),
                float_precision="round_trip")
        except pd.errors.EmptyDataError:
            table = pd.DataFrame()
        if len(table) != count:
            raise ValueError(f"Changed historical selected-policy table count: {name}")
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
        context.progress("Loading verified selected-policy validation; no prior path, martingale, fitted rule or sensitivity is recomputed")
    return {"tables": tables, "figures": figures, "manifest": manifest}
