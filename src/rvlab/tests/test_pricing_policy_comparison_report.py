"""Report assembly preserves old identity and never substitutes missing science."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from rvlab.experiments import _pricing_policy_comparison_report as report


def baseline(tmp_path, monkeypatch, **updates):
    stem = "01_pricing_and_exercise"
    archive = tmp_path / "archive"
    folder = archive / "results/first_stage_v1" / stem
    folder.mkdir(parents=True)
    manifest = {"notebook": stem, "evidence_version": "pricing_first_stage_v1",
        "data_kind": "synthetic", "source_sha256": {"historical.py": "original"},
        "tables": {"rough_result": 1}, "figures": []}
    manifest.update(updates)
    (folder / "rough_result.csv").write_text("price\n1.25\n")
    (folder / "manifest.json").write_text(json.dumps(manifest))
    rows = [{"preserved": str(p.relative_to(archive)), "bytes": p.stat().st_size,
        "sha256": report.run.sha(p)} for p in sorted(folder.iterdir())]
    inventory = archive / "preservation_manifest.json"
    inventory.write_text(json.dumps({"files": rows}))
    monkeypatch.setattr(report, "BASELINE", archive)
    monkeypatch.setattr(report, "BASELINE_SHA", report.run.sha(inventory))
    return stem, folder, manifest


def test_baseline_loads_verified_prices_and_retains_original_execution_identity(tmp_path, monkeypatch):
    stem, folder, original = baseline(tmp_path, monkeypatch)
    result = report.load_baseline(stem, prefixes=("rough_",))
    assert result["tables"]["rough_result"].price.tolist() == [1.25]
    assert result["manifest"] == original
    assert result["manifest"]["source_sha256"] == {"historical.py": "original"}
    assert len(result["input_files"]) == 3
    assert json.loads((folder / "manifest.json").read_text()) == original


@pytest.mark.parametrize("changed", ["inventory", "csv", "missing", "rows", "identity"])
def test_changed_baseline_evidence_cannot_be_presented_as_new_data(tmp_path, monkeypatch, changed):
    kwargs = {"tables": {"rough_result": 2}} if changed == "rows" else {}
    if changed == "identity":
        kwargs["evidence_version"] = "pricing_research_v1"
    stem, folder, _ = baseline(tmp_path, monkeypatch, **kwargs)
    if changed == "inventory":
        (report.BASELINE / "preservation_manifest.json").write_text("{}")
    if changed == "csv":
        (folder / "rough_result.csv").write_text("price\n9.25\n")
    if changed == "missing":
        (folder / "rough_result.csv").unlink()
    with pytest.raises(ValueError):
        report.load_baseline(stem)


@pytest.mark.parametrize("updates", [{"data_kind": "real_spy"}, {"spy_inputs": {"source": "market"}}])
def test_synthetic_baseline_cannot_import_market_observations(tmp_path, monkeypatch, updates):
    stem, _, _ = baseline(tmp_path, monkeypatch, **updates)
    with pytest.raises(ValueError, match="Synthetic"):
        report.load_baseline(stem)


def test_missing_new_campaign_raises_without_relabelled_historical_fallback(monkeypatch):
    from rvlab.experiments import _pricing_research
    calls = []
    monkeypatch.setattr(_pricing_research, "pricing_research",
        lambda context, **kwargs: {"tables": {}, "figures": {}, "manifest": {}})
    monkeypatch.setattr(report, "load_baseline", lambda *args, **kwargs:
        calls.append("verified historical evidence") or {"tables": {}, "figures": {}, "manifest": {}})
    def missing(folder):
        raise FileNotFoundError("No sealed policy-comparison campaign")
    monkeypatch.setattr(report.run, "load_closed", missing)
    with pytest.raises(FileNotFoundError, match="No sealed policy-comparison"):
        report.policy_comparison_pricing(SimpleNamespace(progress=lambda message: None))
    assert calls == ["verified historical evidence"]


def test_report_stamp_changes_only_identity_and_does_not_promote_accuracy():
    result = {"manifest": {"american_pricing_ready": False, "stock_delta_ready": False,
        "tolerances_changed": False, "historical_numerical_manifest": {"version": "old"}}}
    previous = deepcopy(result)
    report._stamp(result, "matched_policy")
    assert result["manifest"]["artifact_family"] == "first_stage_v2"
    assert result["manifest"]["evidence_version"] == "pricing_first_stage_v2"
    for key in previous["manifest"]:
        assert result["manifest"][key] == previous["manifest"][key]


def test_all_seven_comparison_figures_render_both_arms_and_keep_training_scope():
    import matplotlib.pyplot as plt
    policies = report.run.n.POLICY_NAMES
    values = pd.DataFrame([{"case_id": case, "arm": arm, "policy": policy,
        "primary": True, "price": 2.+i*.001, "price_se": .0002,
        "upper": 2.01, "upper_se": .0003, "price_error_budget": .012, "price_allowance": .0012}
        for case in report.CASE_IDS for arm in report.run.ARMS for i, policy in enumerate(policies)])
    gains = pd.DataFrame([{"case_id": case, "arm": arm, "candidate_policy": policy,
        "primary": True, "estimate": .001, "half_width": .0002}
        for case in report.CASE_IDS for arm in report.run.ARMS for policy in policies[1:]])
    costs = pd.DataFrame([{"case_id": case, "arm": arm, "policy": policy,
        "replica": replica, "fit_seconds": 1.+replica,
        "online_decision_seconds": 2.+i, "shared_trace_control_seconds": 3.}
        for case in report.CASE_IDS for arm in report.run.ARMS
        for i, policy in enumerate(policies) for replica in (0, 1)])
    exercise = pd.DataFrame([{"case_id": f"put_90d_k{strike}", "arm": report.run.ARMS[0],
        "policy": policy, "record_type": "exercise_date_distribution", "exercise_time": time,
        "probability": probability}
        for strike in (90, 100, 110) for policy in policies for time, probability in ((0., .1), (.1, .3), (90/365, .6))])
    refinement = pd.DataFrame([{"refinement": kind, "control": report.run.PRIMARY,
        "policy": policy, "change": .0002, "change_se": .0001, "allowance": .0012}
        for kind in ("lookahead_order", "physical_steps", "kernel_factors") for policy in policies])
    tables = {"values": values, "paired_policy_gains": gains, "costs": costs,
        "exercise": exercise, "refinement": refinement}
    tables["controls"] = pd.DataFrame([{"case_id": case, "arm": arm,
        "policy": policy, "control": control, "price_se": .001,
        "raw_price_se": 0. if case == report.CASE_IDS[0] else .002,
        "variance_ratio_to_raw": float("nan") if case == report.CASE_IDS[0] else .25}
        for case in report.CASE_IDS for arm in report.run.ARMS for policy in policies
        for control in ("raw", "european", "european_premium")])
    before = deepcopy(tables)
    figures = report.comparison_figures(tables)
    assert set(figures) == {"policy_comparison_" + name for name in
        ("design_flow", "policy_values", "policy_gains", "cost_accuracy", "exercise", "refinement", "control_precision")}
    assert len(figures["policy_comparison_policy_values"].axes) == 6
    assert all(len(axis.patches) == 1 for axis in figures["policy_comparison_policy_values"].axes)
    gains_figure = figures["policy_comparison_policy_gains"]
    assert len(gains_figure.axes) == 4
    assert any("Bellman − Fourier" in text.get_text() for axis in gains_figure.axes for text in axis.get_legend().get_texts())
    assert len(figures["policy_comparison_control_precision"].axes) == 6
    assert any("N/A" in text.get_text() for axis in figures["policy_comparison_control_precision"].axes for text in axis.texts)
    for key in tables:
        pd.testing.assert_frame_equal(tables[key], before[key])
    for figure in figures.values():
        figure.canvas.draw()
        plt.close(figure)


def cost_rows():
    return pd.DataFrame([{"case_id": "put_90d_k100", "arm": "matched_allowance",
        "policy": "cashflow_lsm", "replica": replica, "fit_seconds": 1.+replica,
        "online_decision_seconds": 3.+replica, "shared_trace_control_seconds": 5.}
        for replica in (0, 1)])


def test_reported_two_fit_estimate_cost_sums_both_rules_and_shared_work_once():
    costs = cost_rows()
    before = costs.copy(deep=True)
    result = report.reported_estimate_costs(costs)
    assert len(result) == 1
    row = result.iloc[0]
    assert row.training_replicas == 2
    assert row.combined_fit_seconds == 3.
    assert row.combined_online_seconds == 7.
    assert row.shared_trace_control_seconds_once == 5.
    assert row.reported_estimate_seconds == 15.
    assert row.per_replica_allowance_unchanged
    assert not row.artifact_io_and_redundant_campaign_fits_included
    pd.testing.assert_frame_equal(costs, before)


@pytest.mark.parametrize("replicas", [(0,), (0, 0), (0, 2), (0, 1, 2)])
def test_reported_estimate_requires_exactly_the_two_declared_training_replicas(replicas):
    costs = pd.concat([cost_rows().iloc[[0]]] * len(replicas), ignore_index=True)
    costs["replica"] = list(replicas)
    with pytest.raises(ValueError, match="both distinct training replicas"):
        report.reported_estimate_costs(costs)


@pytest.mark.parametrize("different", [6., float("nan"), float("inf")])
def test_reported_estimate_cannot_average_or_duplicate_inconsistent_shared_cost(different):
    costs = cost_rows()
    costs.loc[1, "shared_trace_control_seconds"] = different
    with pytest.raises(ValueError, match="share exactly one"):
        report.reported_estimate_costs(costs)


def test_reported_estimate_keeps_budget_arms_separate():
    first = cost_rows()
    second = first.assign(arm="matched_paths", fit_seconds=10., online_decision_seconds=20.,
        shared_trace_control_seconds=30.)
    result = report.reported_estimate_costs(pd.concat([first, second], ignore_index=True)).set_index("arm")
    assert result.loc["matched_allowance", "reported_estimate_seconds"] == 15.
    assert result.loc["matched_paths", "reported_estimate_seconds"] == 90.


def test_inception_diagnostic_preserves_subintrinsic_policy_value_without_selecting_or_flooring():
    cases = [{"case_id": "itm", "spot": 100., "strike": 110.},
        {"case_id": "other", "spot": 100., "strike": 102.}]
    values = pd.DataFrame([{"case_id": case, "arm": "matched_paths", "policy": policy,
        "primary": True, "price": price, "price_interval_high": high}
        for case, policy, price, high in (("itm", "cashflow_lsm", 9.9, 9.95),
            ("itm", "fourier_premium_lsm", 10., 10.), ("other", "cashflow_lsm", 1.8, 1.9))])
    previous = values.copy(deep=True)
    result = report.inception_value_diagnostics(values, cases)
    assert result.known_immediate_payoff.tolist() == [10., 10., 2.]
    assert result.unmodified_policy_value.tolist() == [9.9, 10., 1.8]
    assert result.policy_value_minus_immediate.tolist() == pytest.approx([-.1, 0., -.2])
    assert result.conditional_interval_below_immediate.tolist() == [True, False, True]
    assert not result.new_policy_selected_or_value_floored.any()
    assert not result.cause_is_identified_as_implementation_bug.any()
    pd.testing.assert_frame_equal(values, previous)


def test_inception_diagnostic_distinguishes_point_shortfall_from_conditional_interval():
    cases = [{"case_id": "itm", "spot": 100., "strike": 110.}]
    values = pd.DataFrame([{"case_id": "itm", "arm": "matched_allowance", "policy": "cashflow_lsm",
        "primary": primary, "price": 9.99, "price_interval_high": high}
        for primary, high in ((True, 10.05), (False, 9.999))])
    result = report.inception_value_diagnostics(values, cases)
    assert len(result) == 1
    assert result.policy_value_minus_immediate.iloc[0] < 0
    assert not result.conditional_interval_below_immediate.iloc[0]
