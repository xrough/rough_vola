"""Small workflow fixtures exercise the runner without acceptance simulation."""
from copy import deepcopy
import json

import numpy as np
import pytest

from rvlab.experiments import _pricing_policy_comparison_run as study


def timing():
    return {"paths": 32, "training_seconds": {"cashflow_lsm": 1.,
        "fourier_premium_lsm": 2., "bellman_lookahead_total": 3.},
        "policy_evaluation_seconds": {"cashflow_lsm": .1,
            "fourier_premium_lsm": .2, "bellman_lookahead": .3},
        "trace_generation_seconds": .5}


def test_clock_allocation_counts_online_work_and_uses_timings_only():
    record = timing()
    choices, allowance, rows = study.select_training_paths(record)
    assert choices == {"cashflow_lsm": 8192, "fourier_premium_lsm": 4096,
        "bellman_lookahead": 2048}
    assert allowance == pytest.approx(368.)
    poisoned = {**record, "price": 1e100, "fit_error": -1e100,
        "payoffs": [float("nan")], "selected_policy": "arbitrary"}
    assert study.select_training_paths(poisoned) == (choices, allowance, rows)
    for row in rows:
        assert row["projection_is_accuracy_or_runtime_guarantee"] is False
        assert row["projected_total_seconds"] <= allowance
        assert row["projected_unused_seconds"] == pytest.approx(allowance-row["projected_total_seconds"])


def test_impossible_minimum_budget_is_reported_without_hidden_retry():
    record = timing()
    # This explicit tiny allowance fixture makes fit cost dominate even N512.
    choices, _, rows = study.select_training_paths(record, train_paths=1, value_paths=1)
    assert all(value == 512 for value in choices.values())
    assert all(row["minimum_projected_ineligible"] for row in rows)


@pytest.mark.parametrize("phase", ["value", "refine"])
def test_no_final_path_generation_before_training_closes(tmp_path, monkeypatch, phase):
    def reject(_): raise ValueError("training has not closed")
    monkeypatch.setattr(study, "verify_training", reject)
    def forbidden(*args, **kwargs): raise AssertionError("paths must not be drawn")
    monkeypatch.setattr(study.n, "generate_shared_trace", forbidden)
    with pytest.raises(ValueError, match="training has not closed"):
        getattr(study, phase)(tmp_path, workers=1)
    assert not (tmp_path/"valuation").exists()
    assert not (tmp_path/"refinement").exists()


@pytest.mark.parametrize("phase", ["train", "value", "refine"])
def test_declared_worker_limit_applies_to_all_public_phases(tmp_path, monkeypatch, phase):
    monkeypatch.setattr(study, "get_design", lambda _: {})
    monkeypatch.setattr(study, "verify_training", lambda _: ({}, {}))
    with pytest.raises(ValueError, match="two workers"):
        getattr(study, phase)(tmp_path, workers=3)


def endpoint_fixture():
    payoff = np.arange(4*5, dtype=float).reshape(4, 5, 1)
    euro = .1*np.broadcast_to(np.arange(5)[None, :, None], payoff.shape)
    premium = 2*euro
    stops = np.zeros((4, 2, 3, 1), dtype=int)
    for r in range(2):
        for p in range(3): stops[:, r, p, 0] = (r+p)%5
    arrays = {"stop_indices": stops}
    for name, m in (("raw", np.zeros_like(euro)), ("european", euro),
            ("european_premium", euro+premium)):
        adjusted = payoff-m
        lower = np.empty(stops.shape)
        for r in range(2):
            for p in range(3): lower[:, r, p, 0] = adjusted[np.arange(4), stops[:, r, p, 0], 0]
        arrays[name+"_lower"], arrays[name+"_upper"] = lower, adjusted.max(axis=1)
    return {"discounted_payoffs": payoff, "european_martingale": euro,
        "premium_martingale": premium}, arrays


def test_cold_endpoint_audit_rederives_all_own_stops_and_common_upper():
    trace, arrays = endpoint_fixture()
    assert study.endpoint_audit(trace, arrays) == 21


@pytest.mark.parametrize("defect", ["wrong_lower", "wrong_upper", "wrong_stops", "axes", "fractional_stop"])
def test_cold_endpoint_audit_detects_saved_sample_errors(defect):
    trace, arrays = endpoint_fixture()
    if defect == "wrong_lower": arrays["raw_lower"][0, 0, 0, 0] += 1.
    elif defect == "wrong_upper": arrays["european_upper"][0, 0] += 1.
    elif defect == "wrong_stops": arrays["stop_indices"][0, 0, 0, 0] = 3
    elif defect == "axes": arrays["stop_indices"] = arrays["stop_indices"][:, :1]
    else: arrays["stop_indices"] = arrays["stop_indices"].astype(float)
    with pytest.raises(ValueError): study.endpoint_audit(trace, arrays)


def test_bounded_own_policy_audit_uses_predeclared_path_indices(monkeypatch):
    trace = {"stocks": np.broadcast_to(np.arange(10)[:, None, None], (10, 3, 1)).copy(),
        "factors": np.zeros((10, 3, 2)), "decision_european": np.zeros((10, 3, 1)),
        "actual_time_grid": np.linspace(0, 1, 5)}
    seen = []
    def own(policy, selected):
        seen.append(selected["stocks"][:, 0, 0].tolist())
        np.testing.assert_array_equal(selected["actual_time_grid"], trace["actual_time_grid"])
        return np.full((4, 1), policy), None, None, {}
    monkeypatch.setattr(study.n, "stopping_rule", own)
    bundles = [(dict(zip(study.n.POLICY_NAMES, (0, 1, 2))),
        np.broadcast_to(np.arange(3)[None, :, None], (10, 3, 1)).copy()) for _ in range(2)]
    assert study.decision_audit(trace, bundles) == 6
    assert seen == [[0, 3, 6, 9]]*6


def failed_design():
    cases = [{"case_id": f"fixture_{i}", "strike": 100., "maturity": 30/365,
        "training_seeds": [100+i*10, 101+i*10], "valuation_seeds": [1000+i*10, 1001+i*10],
        "clock_training_paths": dict.fromkeys(study.n.POLICY_NAMES, 512)} for i in range(6)]
    return {"cases": cases, "kernels": {"4": {"nodes": [1.], "weights": [1.],
        "hurst": .15, "horizon": 1.}}, "refinement": {"training_seeds": [200, 201],
            "valuation_seeds": {"physical_steps": [2000, 2001]}}}


def test_training_exception_is_written_as_failed_attempt(tmp_path, monkeypatch):
    design = failed_design()
    study.write_json(tmp_path/"design_before_training.json", design)
    monkeypatch.setattr(study, "get_design", lambda _: design)
    def numerical_failure(*args, **kwargs): raise FloatingPointError()
    monkeypatch.setattr(study.n, "train_policy_set", numerical_failure)
    result = study._training_job(tmp_path, 0, study.ARMS[0], 0)
    assert result["status"] == "failed" and result["error_type"] == "FloatingPointError"
    assert result["message"] == "" and "FloatingPointError" in result["traceback"]
    saved = study.read_json(tmp_path/"training/fixture_0_matched_paths_r0/manifest.json")
    assert saved == result and saved["training_seed"] == 100
    assert saved["training_paths"] == dict.fromkeys(study.n.POLICY_NAMES, 2048)


def test_missing_fitted_rule_blocks_value_without_substitution(tmp_path, monkeypatch):
    design = failed_design()
    study.write_json(tmp_path/"design_before_training.json", design)
    study.write_json(tmp_path/"training_complete.json", {"fixture": True})
    monkeypatch.setattr(study, "verify_training", lambda _: (design, {}))
    def missing(*args): raise ValueError("training unavailable; no replacement")
    monkeypatch.setattr(study, "load_bundle", missing)
    def forbidden(*args, **kwargs): raise AssertionError("no substitute paths")
    monkeypatch.setattr(study.n, "generate_shared_trace", forbidden)
    result = study._valuation_job(tmp_path, 0, 0)
    assert result["status"] == "failed" and "no replacement" in result["message"]
    target = tmp_path/"valuation/fixture_0_s0"
    assert (target/"manifest.json").exists() and not (target/"samples.npz").exists()


def test_closed_inventory_rejects_changed_and_extra_files(tmp_path):
    for kind in ("change", "extra"):
        target = tmp_path/kind; target.mkdir()
        (target/"retained.txt").write_text("original")
        study.inventory(target)
        assert study.verify_inventory(target)["entry_count"] == 1
        if kind == "change": (target/"retained.txt").write_text("altered")
        else: (target/"extra.txt").write_text("unsealed")
        with pytest.raises(ValueError): study.verify_inventory(target)
