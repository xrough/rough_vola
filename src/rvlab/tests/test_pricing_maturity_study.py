"""Matched inputs, numerical gates and disjoint expensive-study workers."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rvlab.experiments import _pricing_maturity_study as study
from rvlab.experiments import _pricing_american_identification as joint
from rvlab.pricing import FDConfig


def test_long_contracts_preserve_short_prefix_and_holdouts():
    frame = study.maturity_contracts()
    pd.testing.assert_frame_equal(frame.iloc[:18], joint.joint_contracts())
    assert len(frame) == 36 and frame.contract_id.is_unique
    assert frame.split.eq("train").sum() == 24
    assert frame.split.eq("heldout").sum() == 12
    assert tuple(np.unique(np.round(frame.maturity*365))) == (14, 30, 60, 90, 180, 365)
    for _, part in frame.groupby("maturity"):
        assert set(part.loc[part.split.eq("heldout"), "strike"]) == {97., 103.}


@pytest.fixture
def cheap_study(monkeypatch, tmp_path):
    calls = {"generation": 0, "geometry": 0, "calibration": []}
    accepted = {"value": True}

    def generation(truth, contracts, market, config):
        calls["generation"] += 1
        return {"prices": [1.]*len(contracts), "deltas": [.2]*len(contracts),
                "grid_sha256": "reference-grid", "elapsed_seconds": 0.}

    def accuracy(*args, **kwargs):
        return {"accepted": accepted["value"], "accuracy": [{"accepted": accepted["value"]}],
                "refinement": [], "contracts": [], "by_maturity": []}

    def geometry(*args):
        calls["geometry"] += 1
        return {"geometries": {}, "linearized_noise": [], "by_maturity": []}

    def calibrate(context, design, quotes, truth, initial, bounds, market, fit,
                  generation, optimizer, seed, budget, start, stage):
        calls["calibration"].append((design, start, stage))
        row = joint._empty_attempt("heston_curve", "noisy", start, stage, initial, bounds, truth, fit, budget)
        row.update(params=joint.parameter_values(truth), status="converged", optimizer_success=True,
                   price_stable=True, delta_stable=True, spread_stable=True, ready=True,
                   training_objective={1: 3., 2: 1., 3: 2.}[start],
                   test_clean_rmse={1: .1, 2: .5, 3: .01}[start])
        return {"summary": {**row, "design": design}, "parameters": [], "predictions": [], "refinement": []}

    monkeypatch.setattr(study, "_generate", generation)
    monkeypatch.setattr(study, "maturity_truth_accuracy", accuracy)
    monkeypatch.setattr(study, "_local_geometry", geometry)
    monkeypatch.setattr(study, "_calibrate", calibrate)
    context = SimpleNamespace(output_dir=tmp_path, progress=lambda message: None)
    return context, calls, accepted


def test_audit_failure_blocks_every_fit(cheap_study):
    context, calls, accepted = cheap_study
    accepted["value"] = False
    result = study.maturity_extension(context, compute_figures=False)
    assert not calls["calibration"] and not calls["geometry"]
    assert result["tables"]["summary"].empty
    assert "blocked" in result["tables"]["failures"].iloc[0].error.lower()


def test_worker_filters_reuse_same_math_and_never_select_partial_starts(cheap_study):
    context, calls, _ = cheap_study
    one = study.maturity_extension(context, designs=("augmented",), start_numbers=(1,), compute_figures=False)
    two = study.maturity_extension(context, designs=("augmented",), start_numbers=(2,), compute_figures=False)
    all_starts = study.maturity_extension(context, designs=("augmented",), compute_figures=False)
    assert one["manifest"]["run_fingerprint"] == two["manifest"]["run_fingerprint"] == all_starts["manifest"]["run_fingerprint"]
    assert one["tables"]["selection"].empty and two["tables"]["selection"].empty
    assert calls["generation"] == calls["geometry"] == 1
    assert calls["calibration"] == [("augmented", 1, "exploratory"), ("augmented", 2, "exploratory"), ("augmented", 3, "exploratory")]
    selection = all_starts["tables"]["selection"].iloc[0]
    assert selection.selected_start == 2 and not selection.heldout_used_for_selection


def test_paired_noise_and_weights_remain_exact_in_executed_inputs(cheap_study):
    context, _, _ = cheap_study
    result = study.maturity_extension(context, run_calibration=False, compute_figures=False)
    quotes = result["tables"]["observations"]
    old = joint.synthetic_quotes(joint.joint_contracts(), np.ones(18),
        np.random.default_rng(42).normal(0., .005, 18), np.full(18, .2))["noisy"]
    pd.testing.assert_frame_equal(quotes.iloc[:18], old)
    assert result["manifest"]["acceptance"] == {"price_atol": .001, "price_rtol": .0001,
        "delta_atol": .002, "delta_rtol": .005, "spread_refinement_fraction": .1}


def test_different_grid_settings_preserve_previous_checkpoint_bytes(cheap_study):
    context, _, _ = cheap_study
    first = study.maturity_extension(context, fit_config=FDConfig(192, 48, 180, spot_concentration=.05),
                                    run_calibration=False, compute_figures=False)
    folder = first["manifest"]["checkpoint_directory"]
    from pathlib import Path
    before = {p.name: p.read_bytes() for p in Path(folder).glob("*.json")}
    second = study.maturity_extension(context, fit_config=FDConfig(192, 64, 180, spot_concentration=.05),
                                     run_calibration=False, compute_figures=False)
    assert folder != second["manifest"]["checkpoint_directory"]
    assert before == {p.name: p.read_bytes() for p in Path(folder).glob("*.json")}


@pytest.mark.parametrize("kwargs", [{"designs": ("short", "short")}, {"start_numbers": (True,)},
                                    {"start_numbers": (1, 1)}, {"start_numbers": ([1],)},
                                    {"designs": (["short"],)}, {"compute_figures": "yes"}])
def test_invalid_workers_rejected_before_pricing(cheap_study, kwargs):
    context, calls, _ = cheap_study
    with pytest.raises(ValueError):
        study.maturity_extension(context, **kwargs)
    assert not calls["generation"]


def test_corrupted_old_short_checkpoint_is_not_reused(tmp_path):
    directory = tmp_path/joint.STUDY_VERSION/"837b8286fe4df229"
    directory.mkdir(parents=True)
    (directory/"heston_curve_noisy_1_exploratory.json").write_text(json.dumps(
        {"specification": {}, "fingerprint": "corrupted"}))
    context = SimpleNamespace(output_dir=tmp_path)
    with pytest.raises(ValueError, match="fingerprint"):
        study._old_short_attempt(context, pd.DataFrame(), None, None, {}, None, None, None, None, 42, 80, 1)
