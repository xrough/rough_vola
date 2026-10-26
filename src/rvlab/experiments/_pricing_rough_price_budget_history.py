"""Verified frozen-rule delta and stopping-price evidence before budget follow-up.

Replay preserves original paths, fitted-rule identities, paired statistics, settings and costs.
It never reruns the spent campaign under the source of a later experiment.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pandas as pd


ARCHIVE = (Path(__file__).resolve().parents[3] /
    "notebooks/legacy/pricing/rough_price_budget_2026-10-19")
BELLMAN_DELTA_PROFILE = "e6188b320bfc29ce96f7edf5c78427cf9cd65e65ff6a518b5f4131c816e6cf5e"


def load_historical_bellman_delta(context=None, *, archive=None, compute_figures=True):
    """Load all verified Bellman-delta tables and original PNGs without computation."""
    from ._pricing_rough_precision import _read_verified_archive

    if not isinstance(compute_figures, bool):
        raise TypeError("compute_figures must be boolean")
    archive = ARCHIVE if archive is None else Path(archive)
    inventory, digest = _read_verified_archive(archive)
    recorded = json.loads((archive / "results/manifest.json").read_text())
    stage = recorded.get("rough_bellman_delta", {})
    if (recorded.get("notebook") != "01_pricing_and_exercise" or
            recorded.get("evidence_version") != "pricing_research_v1" or
            stage.get("backend") != "rough_heston_bellman_delta" or
            stage.get("profile_fingerprint") != BELLMAN_DELTA_PROFILE or
            stage.get("primary_complete") is not True or
            not isinstance(stage.get("input_files"), list) or
            len(stage["input_files"]) != 252 or
            len({row.get("path") for row in stage["input_files"]}) != 252):
        raise ValueError("Preserved evidence is not the declared complete Bellman-delta campaign")
    prefix, tables = "rough_bellman_delta_", {}
    table_names = [name for name in recorded["tables"] if name.startswith(prefix)]
    figure_names = [name for name in recorded["figures"] if name.startswith(prefix)]
    if len(table_names) != 23 or len(figure_names) != 3 or len(set(figure_names)) != 3:
        raise ValueError("Preserved Bellman-delta table or figure set is incomplete")
    preserved = {row["preserved"] for row in inventory["files"]}
    if any("results/"+name+".csv" not in preserved for name in table_names) or any(
            "results/"+name+".png" not in preserved for name in figure_names):
        raise ValueError("A declared historical Bellman-delta output is outside its inventory")
    for name, count in recorded["tables"].items():
        if not name.startswith(prefix):
            continue
        try:
            table = pd.read_csv(archive / "results" / (name + ".csv"),
                float_precision="round_trip")
        except pd.errors.EmptyDataError:
            table = pd.DataFrame()
        if len(table) != count:
            raise ValueError(f"Changed historical Bellman-delta table count: {name}")
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
        american_pricing_ready=False, stock_delta_ready=False, optimal_delta_ready=False)
    if context is not None:
        context.progress("Loading verified Bellman-rule sensitivities and price controls; no prior path, selected rule, control or uncertainty is recomputed")
    return {"tables": tables, "figures": figures, "manifest": manifest}
