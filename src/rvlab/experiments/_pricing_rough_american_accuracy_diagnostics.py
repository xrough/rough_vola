"""Postprocess stopping refinements without changing cached price estimates."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def add_accuracy_refinements(result):
    """Compare all declared central probes with the independent base trial.

    Different trials use disjoint valuation streams, so their conditional
    standard errors combine in quadrature. Within-trial retraining shares
    paths; absent stored covariance, the sum of marginal standard errors is
    the Cauchy upper bound rather than an assumed-independent error estimate.
    These normal diagnostics exclude kernel and training-distribution error.
    """
    from ._pricing_rough_american import (
        PRICE_ATOL, PRICE_RTOL, DELTA_ATOL, DELTA_RTOL, CONFIDENCE_MULTIPLIER,
    )
    prices = result["tables"]["prices"]
    if not prices.empty and "training_method" in prices:
        fourier = prices.training_method.isin(["fourier_premium", "fourier_bellman"])
        prices["effective_surrogate_interpolation"] = np.where(fourier,
            "Fourier value-premium interpolation (surrogate_interpolation config is N/A)",
            "Whole-value " + prices.surrogate_interpolation.astype(str))
    result["manifest"]["interpolation_configuration_scope"] = (
        "surrogate_interpolation config applies to whole-value backends only; "
        "Fourier methods interpolate their own fitted value-premium functions.")
    rows, retrained = [], []
    if not prices.empty:
        base_rows = prices.loc[prices.trial.eq("rough base")]
        if len(base_rows) == 1:
            base = base_rows.iloc[0]
            for row in prices.loc[prices.contract_id.eq("rough_put_60_100") &
                    ~prices.trial.eq("rough base")].to_dict("records"):
                price_change, delta_change = abs(row["price"]-base.price), abs(row["delta"]-base.delta)
                price_ci = CONFIDENCE_MULTIPLIER*np.hypot(row["price_standard_error"], base.price_standard_error)
                delta_ci = CONFIDENCE_MULTIPLIER*np.hypot(row["delta_standard_error"], base.delta_standard_error)
                price_allowance = PRICE_ATOL+PRICE_RTOL*abs(base.price)
                delta_allowance = DELTA_ATOL+DELTA_RTOL*abs(base.delta)
                rows.append({"trial": row["trial"], "price_change": price_change,
                    "price_change_ci_half_width": price_ci, "price_allowance": price_allowance,
                    "price_change_gate_passed": price_change+price_ci <= price_allowance,
                    "delta_change": delta_change, "delta_change_ci_half_width": delta_ci,
                    "delta_allowance": delta_allowance,
                    "delta_change_gate_passed": delta_change+delta_ci <= delta_allowance,
                    "both_discrete_price_gates_passed": bool(base.discrete_price_gate_passed and row["discrete_price_gate_passed"]),
                    "paired_between_trials": False, "american_pricing_ready": False, "stock_delta_ready": False})
        for row in prices.to_dict("records"):
            if np.isfinite(row.get("retrained_deltas", np.nan)):
                change = abs(row["retrained_deltas"]-row["delta"])
                ci = CONFIDENCE_MULTIPLIER*(row["retrained_delta_standard_errors"]+row["delta_standard_error"])
                allowance = DELTA_ATOL+DELTA_RTOL*abs(row["delta"])
                retrained.append({"trial": row["trial"], "frozen_delta": row["delta"],
                    "retrained_delta": row["retrained_deltas"], "absolute_change": change,
                    "change_ci_half_width_upper_bound": ci, "delta_allowance": allowance,
                    "retraining_gate_passed": change+ci <= allowance,
                    "covariance_treatment": "Cauchy upper bound: sum of marginal standard errors",
                    "stock_delta_ready": False})
    result["tables"]["refinement"] = pd.DataFrame(rows)
    result["tables"]["retraining"] = pd.DataFrame(retrained)
    result["manifest"]["refinement_diagnostics_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["manifest"]["refinement_diagnostics_scope"] = (
        "Display-only postprocessing of frozen prices; independent-stream conditional changes, "
        "with a conservative unknown-covariance retraining bound. No numerical estimate or tolerance changes.")
    return result


def add_accuracy_extras(result, *, archives=None):
    """Attach later exploratory evidence without changing the price profile.

    The official four-archive design remains frozen. New exploratory results
    receive their own verified inventory and display-only provenance, rather
    than invalidating or relabelling any running pricing calculation.
    """
    if archives is None and result["manifest"].get("study_version") != "rough_american_accuracy_v1":
        return result
    from ._pricing_rough_american_accuracy import load_accuracy_probes
    root = Path(__file__).resolve().parents[3]
    if archives is None:
        archives = tuple(root/"notebooks/_output/pricing/research_v1/01_pricing_and_exercise"/name for name in (
            "rough_nodal_residual_probe_2026-10-18", "rough_premium_bridge_probe_2026-10-18"))
    probes, provenance = load_accuracy_probes(archives)
    result["tables"]["additional_probes"] = probes
    result["manifest"]["experimental_extras"] = {
        "scope": "Later byte-verified exploratory evidence; display only, outside the official numerical profile",
        "archives": provenance,
        "diagnostic_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return result


def add_accuracy_execution(result, *, output_dir=None):
    """Verify each executed runner and distinguish cold wall time from replay."""
    if result["manifest"].get("study_version") != "rough_american_accuracy_v1":
        return result
    import json
    from datetime import datetime
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[3]/"notebooks/_output/pricing/research_v1/01_pricing_and_exercise"
    folder = Path(output_dir)/"rough_american_accuracy_v1/campaign_v1"
    profile = result["manifest"]["profile_fingerprint"]
    rows = []
    for path in sorted(folder.rglob("manifest.json")):
        record = json.loads(path.read_text())
        if record.get("profile_fingerprint") != profile:
            continue
        runner = path.parent/"executed_runner.py"
        if not runner.is_file():
            # Earlier startup audits predate the explicit runner snapshot and
            # keep their own record, but do not establish campaign timing.
            continue
        digest = hashlib.sha256(runner.read_bytes()).hexdigest()
        if digest != record.get("runner_sha256"):
            raise ValueError(f"Changed executed American accuracy runner: {runner}")
        rows.append({"execution": str(path.parent.relative_to(folder)),
            "mode": record["mode"], "started_at": record["started_at"],
            "completed_at": record.get("completed_at"),
            "elapsed_seconds": record["elapsed_seconds"],
            "checkpoint_replay": record["checkpoint_replay"],
            "selected_trials": len(record["selected_trial_indices"]),
            "runner_sha256": digest, "manifest": str(path)})
    table = pd.DataFrame(rows)
    result["tables"]["execution"] = table
    cold = table.loc[table["mode"].eq("worker") & ~table.checkpoint_replay] if not table.empty else table
    expected = {f"worker_{letter}" for letter in "abcdefg"}
    complete = not cold.empty and expected.issubset(set(cold.execution))
    span = None
    if complete:
        span = (max(datetime.fromisoformat(value) for value in cold.completed_at)
            -min(datetime.fromisoformat(value) for value in cold.started_at)).total_seconds()
    result["manifest"]["execution_timing"] = {
        "all_seven_cold_workers_recorded": complete,
        "cold_parallel_wall_seconds_excluding_reference_audit": span,
        "unique_checkpoint_wall_seconds": result["manifest"]["source_elapsed_seconds"],
        "cold_serial_wall_seconds": None,
        "interpretation": "Sum of per-checkpoint wall times under concurrent prefills is neither CPU time, measured serial runtime, nor campaign wall time. Fresh-kernel cache replay is separate."}
    return result
