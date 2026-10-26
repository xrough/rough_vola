"""A completed research report cannot hide missing streams or changed inputs."""
from dataclasses import asdict, replace
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from rvlab.experiments import _pricing_rough_benchmark as b
from rvlab.experiments import _pricing_rough_benchmark_collection as c
from rvlab.pricing.rough_stopping import _Continuation
from rvlab.pricing.rough_stopping_fourier import FourierStoppingPolicy, fourier_policy_fingerprint
from rvlab.pricing import DiscountCurve, DividendSchedule, SpotMarket


@pytest.fixture
def design():
    definitions = b.benchmark_trials()
    training = {r["policy_id"]: 1146005042 + 3000000 * i
                for i, r in enumerate(definitions) if r["policy_id"] == r["trial_id"]}
    trials = [{**r, "index": i, "training_seed": training[r["policy_id"]],
               "training_paths": 2048, "valuation_paths": 2048,
               "valuation_seeds": [1147005042 + 3000000 * i, 1148005042 + 3000000 * i]}
              for i, r in enumerate(definitions)]
    return {"version": b.VERSION, "parameters": b.PARAMETERS, "trials": trials,
        "created_at": "2026-10-20T08:00:00+00:00", "trial_count": 10, "stream_count": 20,
        "policy_training_count": 9, "total_valuation_paths": 40960,
        "relative_spots": list(b.RELATIVE_SPOTS), "bump_relatives": [.005, .0025],
        "estimators": list(b.ESTIMATORS), "old_source_count": 58, "maximum_workers": 2,
        "regression_ridge": 1e-5, "premium_cap": .7, "batch_size": 512,
        "stock_integral": "exact_drift", "joint_refinement_included": True,
        "no_fresh_selection": True, "no_fitting_on_valuation_paths": True,
        "no_empirical_valuation_recentering": True, "all_paths_retained": True,
        "all_failures_retained": True, "acceptance_requires_all_streams_reported": True,
        "historical_or_pilot_paths_used_for_acceptance": False,
        "independent_training_valuation": True, "focused_GH16_shared193_reused": False,
        "all_policy_training_attempts_complete_before_any_valuation": True,
        "optimal_american_delta_ready": False, "new_hedging_readiness": False, **b.TOLERANCES}


def test_exact_design_keeps_every_case_and_independent_stream(design):
    c.validate_design(design)
    specs = c.expected_streams(design)
    assert len(specs) == 20 and len({r["seed"] for r in specs}) == 20
    assert sum(r["paths"] for r in specs) == 40960
    assert specs[12]["training_seed"] == specs[8]["training_seed"]
    assert specs[12]["seed"] != specs[8]["seed"]


@pytest.mark.parametrize("key,value", [("price_atol", .002), ("delta_rtol", .01),
    ("confidence_multiplier", 1.), ("all_failures_retained", False),
    ("no_fresh_selection", False), ("focused_GH16_shared193_reused", True),
    ("trial_count", 6), ("total_valuation_paths", 81920)])
def test_changed_tolerances_scope_or_budget_are_rejected(design, key, value):
    design[key] = value
    with pytest.raises(ValueError):
        c.validate_design(design)


@pytest.mark.parametrize("change", ["omit", "reorder", "seed", "paths", "policy"])
def test_case_or_stream_selection_cannot_change_the_protocol(design, change):
    if change == "omit": design["trials"].pop()
    elif change == "reorder": design["trials"][:2] = design["trials"][1::-1]
    elif change == "seed": design["trials"][0]["valuation_seeds"][0] += 1
    elif change == "paths": design["trials"][0]["valuation_paths"] = 4096
    else: design["trials"][6]["policy_id"] = "put_90d_k100_physical"
    with pytest.raises((ValueError, KeyError)):
        c.validate_design(design)


def seal(folder):
    rows = [{"path": p.name, "bytes": p.stat().st_size, "sha256": c.sha(p)}
            for p in sorted(folder.iterdir()) if p.is_file()]
    (folder / "preservation_inventory.json").write_text(json.dumps({"files": rows,
        "entry_count": len(rows), "entries_sha256": hashlib.sha256(b.canonical(rows).encode()).hexdigest()}))


def test_inventory_preserves_failed_records_and_rejects_extra_outcomes(tmp_path):
    (tmp_path / "failed.json").write_text('{"status":"failed","message":""}')
    seal(tmp_path)
    assert len(c.closed_inventory(tmp_path)) == 2
    (tmp_path / "unreported_success.json").write_text("{}")
    with pytest.raises(ValueError, match="omits"):
        c.closed_inventory(tmp_path)


def test_inventory_detects_changed_even_same_size_payload(tmp_path):
    p = tmp_path / "samples.npz"
    p.write_bytes(b"first")
    seal(tmp_path)
    p.write_bytes(b"other")
    with pytest.raises(ValueError, match="artifact changed"):
        c.closed_inventory(tmp_path)


def test_chronology_rejects_timezone_free_records():
    with pytest.raises(ValueError, match="timezones"):
        c.timestamp("2026-10-20T08:00:00")


def policy_fixture(design):
    trial = design["trials"][4]
    config = b.benchmark_config(training_seed=trial["training_seed"], valuation_seed=trial["valuation_seeds"][0])
    kernel = b.ExponentialKernel((1., 2., 3., 4.), (.1, .2, .3, .4), .15, 90 / 365)
    model = _Continuation(np.zeros(4), np.ones(4), np.zeros(28), config.stock_knots, 0, 1.)
    times = np.linspace(0., 90 / 365, 65)
    policy = FourierStoppingPolicy(b.RoughHestonParams(**b.PARAMETERS),
        SpotMarket(100., DiscountCurve.flat(.04, 1.), DividendSchedule((), 1.)),
        kernel, 100., 90 / 365, False, "american", times, times,
        (model,) * 65, (model,) * 65, config, config.premium_control, False, {}, .7)
    design["kernels"] = {"4": asdict(kernel)}
    trained = {"policy_fingerprint": fourier_policy_fingerprint(policy)}
    return policy, trial, trained


def test_loaded_policy_must_match_dynamics_not_just_its_own_seal(design):
    policy, trial, trained = policy_fixture(design)
    c.validate_policy(policy, trial, design, trained)
    wrong = replace(policy, params=replace(policy.params, rho=-.5))
    with pytest.raises(ValueError, match="declared model"):
        c.validate_policy(wrong, trial, design, {"policy_fingerprint": fourier_policy_fingerprint(wrong)})
    wrong = replace(policy, config=replace(policy.config, ridge=1e-4))
    with pytest.raises(ValueError, match="declared model"):
        c.validate_policy(wrong, trial, design, {"policy_fingerprint": fourier_policy_fingerprint(wrong)})


def test_current_runner_compatibility_is_version_and_ast_guarded(tmp_path):
    old, new = tmp_path / "old.py", tmp_path / "new.py"
    old.write_text("def train(): return 1\n")
    new.write_text("def train(): return 2\n")
    with pytest.raises(ValueError, match="Unknown runner version"):
        c.runner_guard_compatibility(old, new)


def test_price_bracket_contrast_uses_independent_lower_and_upper_errors():
    prices = pd.DataFrame([
        {"trial_id": "coarse", "control": "european_premium", "price": 3.4,
         "upper": 3.41, "price_se": .001, "upper_se": .002},
        {"trial_id": "fine", "control": "european_premium", "price": 3.18,
         "upper": 3.2, "price_se": .003, "upper_se": .004}])
    result = c.price_bracket_separation(prices,
        [{"coarse_trial": "coarse", "fine_trial": "fine", "same_full_policy": False}])
    row = result.iloc[0]
    assert row.estimate == pytest.approx(.2)
    assert row.standard_error == pytest.approx(np.sqrt(.001 ** 2 + .004 ** 2))
    assert row.interval_low == pytest.approx(.2 - 1.96 * np.sqrt(.001 ** 2 + .004 ** 2))
    assert row.positive_separation and not result.iloc[1].positive_separation
    assert not result.used_for_selection.any() and not result.pure_kernel_bias_identified.any()


def test_failed_trials_cannot_supply_partial_price_brackets():
    result = c.price_bracket_separation(pd.DataFrame(),
        [{"coarse_trial": "coarse", "fine_trial": "failed", "same_full_policy": False}])
    assert len(result) == 1 and not result.complete.any()
    assert "estimate" not in result
