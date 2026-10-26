"""Verified prior stopping and stock-delta evidence for selected-policy validation.

Replay preserves original paths, fitted-rule identities, paired statistics, settings and costs.
It never reruns the spent campaign under the source of a later experiment.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pandas as pd


ARCHIVE = (Path(__file__).resolve().parents[3] /
    "notebooks/legacy/pricing/rough_policy_validation_2026-10-19")
STOPPING_DELTA_PROFILE = "552fed82eff3b4a9cee82a535a366bc86f79fb15f66d418d4a92869d85b7158d"


def load_historical_stopping_delta(context=None, *, archive=None, compute_figures=True):
    """Load all verified stopping/delta tables and original PNGs without computation."""
    from ._pricing_rough_precision import _read_verified_archive

    if not isinstance(compute_figures, bool):
        raise TypeError("compute_figures must be boolean")
    archive = ARCHIVE if archive is None else Path(archive)
    inventory, digest = _read_verified_archive(archive)
    recorded = json.loads((archive / "results/manifest.json").read_text())
    stage = recorded.get("rough_stopping_delta", {})
    if (recorded.get("notebook") != "01_pricing_and_exercise" or
            recorded.get("evidence_version") != "pricing_research_v1" or
            stage.get("backend") != "rough_heston_stopping_delta" or
            stage.get("profile_fingerprint") != STOPPING_DELTA_PROFILE or
            stage.get("primary_complete") is not True):
        raise ValueError("Preserved evidence is not the declared complete stopping/delta campaign")
    prefix, tables = "rough_stopping_delta_", {}
    for name, count in recorded["tables"].items():
        if not name.startswith(prefix):
            continue
        try:
            table = pd.read_csv(archive / "results" / (name + ".csv"),
                float_precision="round_trip")
        except pd.errors.EmptyDataError:
            table = pd.DataFrame()
        if len(table) != count:
            raise ValueError(f"Changed historical stopping/delta table count: {name}")
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
        context.progress("Loading verified stopping/delta campaign; no prior paths, fitted rule or sensitivity is recomputed")
    return {"tables": tables, "figures": figures, "manifest": manifest}
