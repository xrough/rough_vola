"""Policy gains must retain a fixed upper, independent selection and failures."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rvlab.experiments import _pricing_rough_policy_improvement as study


def _samples():
    x = np.array([-.03, .02, .01, -.01, .025, -.015])
    raw = np.column_stack((2+x, 2.01+x, 1.99+x))
    lower = np.column_stack((2+x/10, 2.01+x/12, 1.99+x/8))
    upper = np.full(6, 2.03)
    stops = np.column_stack((np.ones(6), [1, 1, 2, 2, 1, 1], np.full(6, 2)))
    return raw, lower, upper, stops


def test_paired_policy_gains_keep_common_upper_covariance_and_failed_gates():
    raw, lower, upper, stops = _samples()
    result = study.policy_comparison_tables(raw, lower, upper, stops,
        policy_names=["original", "improved", "worse"], selected_name="improved")
    matrix, changes = result["comparison"], result["paired_policy_changes"]
    assert matrix.dual_price.nunique() == 1
    assert not matrix.discrete_price_gate_passed.any()
    assert changes.same_upper_samples.all() and changes.dual_change.eq(0).all()
    assert changes.resolved_lower_gain.tolist() == [True, False]
    assert not changes.absolute_rough_policy_error_identified.any()
    gain = lower[:, 1]-lower[:, 0]
    first = changes.iloc[0]
    assert first.controlled_lower_change == pytest.approx(gain.mean())
    assert first.controlled_change_standard_error == pytest.approx(gain.std(ddof=1)/np.sqrt(6))
    ratio = gain.mean()/(upper-lower[:, 0]).mean()
    expected = (gain-ratio*(upper-lower[:, 0])).std(ddof=1)/np.sqrt(6)/(upper-lower[:, 0]).mean()
    assert first.gap_fraction_standard_error == pytest.approx(expected)
    assert matrix.exercise_change_fraction.tolist() == [0., 2/6, 1.]
    assert not matrix.american_pricing_ready.any() and not matrix.stock_delta_ready.any()


@pytest.mark.parametrize("bad", ["shape", "nonfinite", "domination", "selected"])
def test_policy_comparisons_reject_invalid_evidence(bad):
    raw, lower, upper, stops = _samples()
    selected = "improved"
    if bad == "shape": stops = stops[:, :1]
    elif bad == "nonfinite": lower[0, 0] = np.nan
    elif bad == "domination": upper[:] = 0
    else: selected = "missing"
    with pytest.raises(ValueError):
        study.policy_comparison_tables(raw, lower, upper, stops,
            policy_names=["original", "improved", "worse"], selected_name=selected)


def test_formal_design_has_five_distinct_streams_and_explicit_separate_knots():
    design = study._validated_policy_design(study.policy_improvement_trial_specs(), 42)
    streams = [v for row in design for v in (row["config"].training_seed, row["dual_seed"],
        row["improvement_seed"], row["selection_seed"], row["config"].valuation_seed)]
    assert len(set(streams)) == 30
    assert all(row["config"].valuation_seed >= 350000042 for row in design)
    assert all(len(row["policy_candidates"]) == 6 for row in design)
    assert all(len(row["policy_candidates"]["dense tensor quadratic"]["stock_knots"]) == 20 for row in design)
    assert all("stock_knots" not in row["policy_candidates"]["tensor quadratic"] for row in design)


@pytest.mark.parametrize("seed", [True, -1, .5, "42"])
def test_seed_is_validated_before_namespace_arithmetic(seed):
    with pytest.raises(ValueError, match="seed"):
        study._validated_policy_design(study.policy_improvement_trial_specs(), seed)


@pytest.mark.parametrize("options,error", [
    ({"run_rough_policy_improvement": 1}, ValueError),
    ({"rough_policy_improvement_kwargs": []}, TypeError),
    ({"rough_policy_improvement_kwargs": {"seed": 7}}, ValueError),
    ({"run_rough_policy_improvement": True, "rough_dual_tightening_kwargs": {"trial_specs": []}}, ValueError),
    ({"run_rough_policy_improvement": True, "rough_american_accuracy_kwargs": {"trial_specs": []}}, ValueError),
])
def test_workflow_rejects_ambiguous_policy_or_historical_options(options, error):
    from rvlab.experiments._pricing_research import pricing_research
    with pytest.raises(error): pricing_research(SimpleNamespace(), **options)


def _archive(tmp_path):
    root = tmp_path/"archive"; (root/"results").mkdir(parents=True)
    (root/"results/rough_dual_tightening_summary.csv").write_text("paired_gap,ready\n0.00241583912345678,False\n")
    (root/"results/rough_dual_tightening_failures.csv").write_text("\n")
    record = {"evidence_version": "pricing_research_v1", "notebook": "01_pricing_and_exercise", "created_at": "original date",
        "tables": {"rough_dual_tightening_summary": 1, "rough_dual_tightening_failures": 0}, "figures": [],
        "rough_dual_tightening": {"backend": "rough_heston_dual_tightening", "profile_fingerprint": study.HISTORICAL_TIGHTENING_PROFILE,
            "source_elapsed_seconds": 555.123, "sources": {"core.py": "oldhash"}}}
    (root/"results/manifest.json").write_text(json.dumps(record))
    files = [{"preserved": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(root.rglob('*')) if p.is_file()]
    (root/"preservation_manifest.json").write_text(json.dumps({"files": files}))
    return root


def test_replay_keeps_historical_identity_and_original_cost_without_repricing(tmp_path):
    root = _archive(tmp_path)
    result = study.load_historical_tightening(archive=root)
    manifest = result["manifest"]
    assert manifest["sources"] == {"core.py": "oldhash"}
    assert manifest["source_elapsed_seconds"] == 555.123
    assert manifest["historical_replay"] and not manifest["current_engine_execution"]
    assert result["tables"]["summary"].paired_gap.item() == 0.00241583912345678
    assert result["tables"]["failures"].empty
    (root/"results/rough_dual_tightening_summary.csv").write_text("changed")
    with pytest.raises(ValueError, match="Changed preserved"):
        study.load_historical_tightening(archive=root)


def test_figures_retain_residual_failed_acceptance():
    import matplotlib.pyplot as plt
    summary = pd.DataFrame([{"case": "central rough", "status": "completed", "original_gap": .003,
        "selected_gap": .002, "paired_lower_gain": .001, "gain_interval_low": .0007, "gain_interval_high": .0013,
        "removed_baseline_gap_fraction": 1/3, "gap_fraction_interval_low": .2, "gap_fraction_interval_high": .4,
        "selected_price_error_budget": .0025, "selected_price_allowance": .0013}])
    figures = study._policy_figures({"summary": summary})
    assert set(figures) == {"policy_gap_and_acceptance", "paired_policy_gains"}
    for figure in figures.values(): figure.canvas.draw(); plt.close(figure)
    assert study._policy_figures({}) == {}
