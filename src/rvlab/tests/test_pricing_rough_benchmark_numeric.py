"""Numerical/contract checks for the NEW generic coverage benchmark."""
from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from rvlab.experiments import _pricing_rough_benchmark as benchmark
from rvlab.pricing import RoughHestonParams
from rvlab.pricing.rough_estimators import RoughEuropeanControlConfig
from rvlab.pricing.rough_lift import ExponentialKernel
from rvlab.pricing.rough_stopping_fourier import train_fourier_stopping_policy
from rvlab.pricing.vanilla_types import DiscountCurve, DividendSchedule, SpotMarket


def test_six_cases_and_predeclared_refinements():
    cases = benchmark.benchmark_cases()
    assert len(cases) == 6
    assert {(r["days"], r["strike"]) for r in cases} == {
        (days, strike) for days in (30, 90) for strike in (90., 100., 110.)}
    trials = benchmark.benchmark_trials()
    assert len(trials) == 10
    physical = next(r for r in trials if r["stage"] == "physical")
    assert physical["policy_id"] == "put_90d_k100"
    assert physical["exercise_steps"] == 64 and physical["steps"] == 256
    assert len(benchmark.benchmark_trials(include_joint=False)) == 9
    with pytest.raises(TypeError):
        benchmark.benchmark_trials(include_joint=1)


def test_tolerances_and_declared_independent_configs():
    config = benchmark.benchmark_config(training_seed=123, valuation_seed=456)
    assert config.training_paths == config.valuation_paths == 2048
    assert config.ridge == 1e-5 and config.premium_cap == .7
    assert config.dual_config is None
    assert benchmark.TOLERANCES == {"price_atol": .001, "price_rtol": 1e-4,
        "delta_atol": .002, "delta_rtol": .005, "confidence_multiplier": 1.96,
        "spread_fraction": .1, "tolerances_changed": False}
    with pytest.raises(ValueError, match="independent"):
        benchmark.benchmark_config(training_seed=123, valuation_seed=123)


def test_endpoints_use_own_stops_and_common_eligible_dates():
    payoff = np.array([[[1., 2.], [3., 4.], [2., 1.]],
        [[0., 1.], [1., 0.], [4., 5.]]])
    euro = np.array([[[0., 0.], [.1, .2], [.3, .4]],
        [[0., 0.], [-.1, -.2], [-.3, -.4]]])
    premium = 2*euro
    stops = np.array([[1, 2], [2, 1]])
    result = benchmark.estimator_endpoints(payoff, euro, premium, stops,
        np.array([True, True, True]))
    for name, m in (("raw", 0.), ("european", euro), ("european_premium", euro+premium)):
        adjusted = payoff-m
        np.testing.assert_array_equal(result[name+"_lower"],
            adjusted[np.arange(2)[:, None], stops, np.arange(2)[None]])
        np.testing.assert_array_equal(result[name+"_upper"], adjusted.max(axis=1))
        assert np.all(result[name+"_upper"] >= result[name+"_lower"])


@pytest.mark.parametrize("defect", ["shape", "negative_stop", "float_stop", "not_eligible", "nan"])
def test_endpoint_input_rejections(defect):
    payoff = np.ones((2, 3, 5))
    euro = np.zeros_like(payoff)
    premium = np.zeros_like(payoff)
    stops = np.ones((2, 5), dtype=int)
    eligible = np.ones(3, bool)
    if defect == "shape": premium = premium[..., :1]
    if defect == "negative_stop": stops[0, 0] = -1
    if defect == "float_stop": stops = stops.astype(float)
    if defect == "not_eligible": eligible[1] = False
    if defect == "nan": euro[0, 1, 2] = np.nan
    with pytest.raises(ValueError):
        benchmark.estimator_endpoints(payoff, euro, premium, stops, eligible)


@pytest.fixture(scope="module")
def constant_policy():
    market = SpotMarket(100., DiscountCurve.flat(.04, 1.), DividendSchedule((), 1.))
    params = RoughHestonParams(.04, .04, 1., 0., -.6, .15)
    kernel = ExponentialKernel((1.,), (1.,), .15, 1.)
    config = replace(benchmark.benchmark_config(training_seed=123321,
        valuation_seed=123322, training_paths=32, valuation_paths=16,
        steps=8, exercise_steps=4),
        premium_control=RoughEuropeanControlConfig(8, 40., 32))
    return train_fourier_stopping_policy(params, market, kernel, strike=100.,
        maturity=30/365, is_call=False, config=config,
        control_config=config.premium_control, premium_cap=.7)


def test_base_public_replay_and_original_operator_on_fine_grid(constant_policy):
    base, diagnostic = benchmark.value_benchmark_policy(constant_policy, constant_policy.config)
    assert diagnostic["base_public_replay_checked"]
    np.testing.assert_array_equal(base["stop_indices"], base["public_stop_indices"])
    fine, diagnostic = benchmark.value_benchmark_policy(constant_policy,
        replace(constant_policy.config, steps=16, valuation_seed=123323))
    assert not diagnostic["same_physical_and_operator_grid"]
    assert len(fine["actual_time_grid"]) == 17
    np.testing.assert_array_equal(fine["operator_time_grid"], base["operator_time_grid"])
    assert len(fine["operator_time_grid"]) == 9
    assert np.isfinite(fine["european_premium_lower"]).all()


def test_stopping_decision_is_nonanticipating(constant_policy):
    arrays, _ = benchmark.value_benchmark_policy(constant_policy, constant_policy.config)
    trace = {"stocks": arrays["stocks"].copy(), "factors": arrays["factors"].copy(),
        "times": arrays["times"]}
    euro = arrays["decision_european"].copy()
    old_stops, old_c, old_d = benchmark.replay_stopping_rule(constant_policy, trace, euro)
    trace["stocks"][:, 3:] *= 1.2
    trace["factors"][:, 3:] += .01
    euro[:, 3:] += 20.
    _, new_c, new_d = benchmark.replay_stopping_rule(constant_policy, trace, euro)
    np.testing.assert_array_equal(old_c[:, :3], new_c[:, :3])
    np.testing.assert_array_equal(old_d[:, :3], new_d[:, :3])
    before = old_stops < 3
    assert np.array_equal(old_stops[before], arrays["stop_indices"][before])


def test_explicit_policy_roundtrip_and_tamper(constant_policy, tmp_path):
    folder = tmp_path/"policy"
    benchmark.save_policy(constant_policy, folder)
    loaded = benchmark.load_policy(folder)
    original, _ = benchmark.value_benchmark_policy(constant_policy, constant_policy.config)
    restored, _ = benchmark.value_benchmark_policy(loaded, loaded.config)
    for name in original:
        np.testing.assert_array_equal(original[name], restored[name])
    with (folder/"coefficients.npz").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="seal changed"):
        benchmark.load_policy(folder)


def _runner():
    path = Path(__file__).resolve().parents[3]/"examples/rough_benchmark.py"
    spec = importlib.util.spec_from_file_location("benchmark_runner_numeric_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_predeclared_stream_firewall_and_fixed_policy_reuse():
    rows = _runner().stream_specs(benchmark.benchmark_trials())
    train = {r["training_seed"] for r in rows}
    values = [seed for row in rows for seed in row["valuation_seeds"]]
    assert len(rows) == 10 and len(train) == 9 and len(values) == len(set(values)) == 20
    assert not train.intersection(values)
    assert rows[6]["training_seed"] == rows[4]["training_seed"]
    assert rows[6]["valuation_seeds"] != rows[4]["valuation_seeds"]
    assert all(row["training_paths"] == row["valuation_paths"] == 2048 for row in rows)


def test_closed_implementation_inventory_rejects_changed_and_extra_files(tmp_path):
    runner = _runner()
    (tmp_path/"result.txt").write_text("retained")
    runner.inventory(tmp_path)
    runner.verify_preserved(tmp_path)
    (tmp_path/"extra.txt").write_text("late unsealed output")
    with pytest.raises(ValueError, match="not closed"):
        runner.verify_preserved(tmp_path)
    (tmp_path/"extra.txt").unlink()
    (tmp_path/"result.txt").write_text("changed")
    with pytest.raises(ValueError, match="file changed"):
        runner.verify_preserved(tmp_path)
