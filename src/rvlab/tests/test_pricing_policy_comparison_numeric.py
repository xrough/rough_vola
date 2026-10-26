"""Independent-path and no-lookahead invariants for matched policy experiments."""
from dataclasses import replace
import json

import numpy as np
import pytest

from rvlab.experiments import _pricing_policy_comparison_numeric as numeric
from rvlab.experiments._pricing_rough_benchmark import benchmark_config
from rvlab.pricing import BlackScholesParams, RoughHestonParams
from rvlab.pricing.black_scholes import black_scholes_price
from rvlab.pricing.rough_estimators import RoughEuropeanControlConfig
from rvlab.pricing.rough_lift import ExponentialKernel
from rvlab.pricing.rough_simulation import RoughSimulationConfig, simulate_rough_heston
from rvlab.pricing.rough_stopping_fourier import _stream
from rvlab.pricing.vanilla_types import DiscountCurve, DividendSchedule, SpotMarket


@pytest.fixture(scope="module")
def fixture():
    params = RoughHestonParams(.04, .04, 1., .3, -.6, .15)
    market = SpotMarket(100., DiscountCurve.flat(.04, 1.), DividendSchedule((), 1.))
    kernel = ExponentialKernel((1., 10.), (.5, 1.), .15, 1.)
    config = replace(benchmark_config(training_seed=12880101, valuation_seed=12880102,
        training_paths=64, valuation_paths=64, steps=8, exercise_steps=4),
        premium_control=RoughEuropeanControlConfig(8, 40., 64))
    policies, training = numeric.train_policy_set(params, market, kernel,
        strike=100., maturity=30/365, config=config, lookahead_order=8)
    trace, timing = numeric.generate_shared_trace(policies, config)
    return policies, trace, config, training, timing


def test_training_records_explicit_all_state_variant_and_common_random_stream(fixture):
    policies, trace, config, training, _ = fixture
    assert tuple(policies) == numeric.POLICY_NAMES
    assert training["cashflow_regression_states"] == "all training states, not ITM-only"
    assert training["same_training_random_stream"] and training["same_training_factor_basis"]
    assert training["bellman_reuses_fourier_fit"]
    assert training["selection_or_final_paths_used"] is False
    assert all(value >= 0 for value in training["training_seconds"].values())
    assert trace["seed"] != config.training_seed


def test_existing_trainers_really_see_identical_training_trajectories(fixture):
    policies, _, config, _, _ = fixture
    base = policies["fourier_premium_lsm"].base_policy
    snapshots = _stream(base.params, base.market, base.kernel, base.strike,
        base.maturity, base.is_call, base.exercise_times, config, base.control_config,
        training=True)
    sim = simulate_rough_heston(base.params, base.market, base.kernel,
        observation_times=base.exercise_times, config=RoughSimulationConfig(
            config.training_paths, config.steps, config.training_seed,
            config.batch_size, config.memory_budget_mb, config.stock_integral))
    for index, time in enumerate(base.exercise_times):
        # The two established engines use algebraically equal log/exp expressions;
        # stock roundoff is not a different random trajectory.
        np.testing.assert_allclose(snapshots[float(time)][0], sim.spots[:, index], rtol=0, atol=2e-12)
        np.testing.assert_array_equal(snapshots[float(time)][1], sim.factors[:, index])


def test_every_policy_has_own_stops_and_exact_common_upper(fixture):
    policies, trace, _, _, _ = fixture
    result, diagnostics = numeric.evaluate_policy_set(trace, policies)
    assert result["stop_indices"].shape == (64, 3, 1)
    assert diagnostics["same_paths_same_upper_for_all_policies"]
    assert diagnostics["policy_selection_performed"] is False
    payoff = np.asarray(trace["discounted_payoffs"])
    path = np.arange(64)[:, None]
    for control in numeric.CONTROL_NAMES:
        m = np.zeros_like(payoff)
        if control != "raw":
            m += trace["european_martingale"]
        if control == "european_premium":
            m += trace["premium_martingale"]
        adjusted = payoff-m
        np.testing.assert_array_equal(result[control+"_upper"], adjusted.max(axis=1))
        for j in range(3):
            own = adjusted[path, result["stop_indices"][:, j], 0]
            np.testing.assert_array_equal(result[control+"_lower"][:, j], own)
            assert np.all(result[control+"_upper"] >= own-2e-11)
    # Gap reductions under a shared upper are exactly controlled policy gains.
    gains = result["european_premium_lower"][:, 1]-result["european_premium_lower"][:, 0]
    gap0 = result["european_premium_upper"]-result["european_premium_lower"][:, 0]
    gap1 = result["european_premium_upper"]-result["european_premium_lower"][:, 1]
    np.testing.assert_allclose(gap0-gap1, gains, rtol=0, atol=4e-15)


def test_policy_decisions_never_use_realized_future(fixture):
    policies, trace, _, _, _ = fixture
    changed = {**trace, "stocks": trace["stocks"].copy(),
        "factors": trace["factors"].copy(), "european_proxy": trace["european_proxy"].copy()}
    changed["stocks"][:, 3:] *= 1.2
    changed["factors"][:, 3:] += .1
    changed["european_proxy"][:, 3:] += 10.
    for policy in policies.values():
        old = numeric.stopping_rule(policy, trace)
        new = numeric.stopping_rule(policy, changed)
        np.testing.assert_array_equal(old[1][:, :3], new[1][:, :3])
        np.testing.assert_allclose(old[2][:, :3], new[2][:, :3], rtol=0, atol=0, equal_nan=True)
        stopped = old[0] < 3
        np.testing.assert_array_equal(old[0][stopped], new[0][stopped])


def test_lookahead_terminal_expectation_matches_independent_gaussian_formula(fixture):
    from rvlab.experiments._pricing_rough_exercise_block import exercise_block_law
    from rvlab.pricing.rough_estimators import _conditional_payoffs
    policies, trace, _, _, _ = fixture
    base = policies["fourier_premium_lsm"].base_policy
    evaluator = numeric.StateBellmanLookahead(base, order=8)
    j = len(base.exercise_times)-2
    s, u = trace["stocks"][:4, j, 0], trace["factors"][:4, j]
    a, b = np.searchsorted(evaluator.grid, base.exercise_times[j:j+2])
    law = exercise_block_law(base, u, times=evaluator.grid[a:b+1])
    payoff = _conditional_payoffs(np.log(s)[:, None, None]+law["log_stock_shifts"],
        law["orthogonal"], base.strike, base.is_call)
    expected = (evaluator.discounts[j+1]/evaluator.discounts[j]
        *np.sum(payoff.mean(axis=2)*law["probabilities"], axis=1))
    np.testing.assert_allclose(evaluator.conditional_values(j, s, u), expected, rtol=0, atol=2e-14)


def test_constant_variance_zero_rate_european_semigroup(fixture):
    policies, _, _, _, _ = fixture
    original = policies["fourier_premium_lsm"].base_policy
    zero = tuple(original.continuations[0] for _ in original.continuations)
    base = replace(original, params=replace(original.params, v0=.04, theta=.04, vol_of_vol=0.),
        market=replace(original.market, discount_curve=DiscountCurve.flat(0., 1.)),
        continuations=zero, value_premiums=zero)
    evaluator = numeric.StateBellmanLookahead(base, order=8)
    stocks = np.array([90., 99., 101.])
    factors = np.zeros((3, len(base.kernel.nodes)))
    for index in (0, 1, len(base.exercise_times)-2):
        actual = evaluator.conditional_values(index, stocks, factors)
        expected = black_scholes_price(BlackScholesParams(.2), forward=stocks,
            discount=1., strike=100., maturity=base.maturity-base.exercise_times[index], is_call=False)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-10)


def test_inception_rule_is_state_only_and_correct_for_deep_itm_deterministic_limit(fixture):
    policies, trace, _, _, _ = fixture
    original = policies["fourier_premium_lsm"].base_policy
    zero = tuple(original.continuations[0] for _ in original.continuations)
    base = replace(original, params=replace(original.params, v0=0., theta=0., vol_of_vol=0.),
        strike=110., continuations=zero, value_premiums=zero)
    rule = numeric.build_lookahead_policy(base, order=8)
    assert rule.exercise_at_inception is True
    assert rule.inception_continuation < 10.
    changed = {**trace, "stocks": trace["stocks"].copy()}
    changed["stocks"][:, 1:] = 1.
    old_stops = numeric.stopping_rule(rule, trace)[0]
    new_stops = numeric.stopping_rule(rule, changed)[0]
    np.testing.assert_array_equal(old_stops, np.zeros_like(old_stops))
    np.testing.assert_array_equal(old_stops, new_stops)


@pytest.mark.parametrize("defect", ["training_seed", "forbidden_seed", "calendar", "fingerprint",
    "initial_stock", "initial_factor", "martingale_start", "payoff"])
def test_trace_provenance_and_independence_rejections(fixture, defect):
    policies, trace, config, _, _ = fixture
    bad = dict(trace)
    forbidden = ()
    if defect == "training_seed": bad["seed"] = config.training_seed
    elif defect == "forbidden_seed": forbidden = (bad["seed"],)
    elif defect == "calendar": bad["actual_time_grid"] = bad["actual_time_grid"]+.001
    elif defect == "fingerprint": bad["policy_fingerprint"] = "wrong"
    elif defect == "initial_stock":
        bad["stocks"] = bad["stocks"].copy(); bad["stocks"][1, 0, 0] += 1.
    elif defect == "initial_factor":
        bad["factors"] = bad["factors"].copy(); bad["factors"][1, 0, 0] += 1.
    elif defect == "martingale_start":
        bad["european_martingale"] = bad["european_martingale"].copy(); bad["european_martingale"][0, 0, 0] = 1.
    else:
        bad["discounted_payoffs"] = bad["discounted_payoffs"].copy(); bad["discounted_payoffs"][0, -1, 0] += 1.
    with pytest.raises(ValueError):
        numeric.evaluate_policy_set(bad, policies, forbidden_seeds=forbidden)


def test_common_anchor_is_fixed_across_different_training_replicas(fixture):
    policies, trace, config, _, _ = fixture
    base = policies["fourier_premium_lsm"].base_policy
    changed = {name: replace(rule, base_policy=replace(rule.base_policy,
        config=replace(rule.base_policy.config, training_seed=12880103)))
        for name, rule in policies.items()}
    # A different learned rule may use the ORIGINAL frozen control anchor.
    result, _ = numeric.evaluate_policy_set(trace, changed, control_policy=base)
    expected, _ = numeric.evaluate_policy_set(trace, policies)
    for control in numeric.CONTROL_NAMES:
        np.testing.assert_array_equal(result[control+"_upper"], expected[control+"_upper"])
    with pytest.raises(ValueError, match="reference policy"):
        numeric.evaluate_policy_set(trace, changed)


def test_declared_clock_arm_allows_different_N_but_never_claims_same_trajectories(fixture):
    policies, trace, config, _, _ = fixture
    anchor = policies["fourier_premium_lsm"].base_policy
    clock, timing = numeric.train_policy_set(anchor.params, anchor.market, anchor.kernel,
        strike=anchor.strike, maturity=anchor.maturity, config=config,
        cashflow_config=replace(config, training_paths=32),
        fourier_config=replace(config, training_paths=64),
        bellman_config=replace(config, training_paths=128), lookahead_order=8)
    assert timing["same_training_seed"] is True
    assert timing["same_training_random_stream"] is False
    assert timing["bellman_reuses_fourier_fit"] is False
    assert timing["training_paths"] == {"cashflow_lsm": 32,
        "fourier_premium_lsm": 64, "bellman_lookahead": 128}
    value, _ = numeric.evaluate_policy_set(trace, clock, control_policy=anchor)
    old, _ = numeric.evaluate_policy_set(trace, policies)
    for control in numeric.CONTROL_NAMES:
        np.testing.assert_array_equal(value[control+"_upper"], old[control+"_upper"])


def test_physical_refinement_freezes_original_decision_operator_and_uses_actual_control_law(fixture):
    from rvlab.experiments._pricing_rough_benchmark import decision_european
    from rvlab.pricing.rough_stopping_fourier import value_fourier_stopping_policy
    policies, _, config, _, _ = fixture
    base = policies["fourier_premium_lsm"].base_policy
    fine_config = replace(config, valuation_seed=12880106)
    # A nonnested grid also forces distinct Riccati target nodes; on nested
    # grids both functions can legitimately coincide to roundoff.
    fine, settings = numeric.generate_shared_trace(policies, fine_config, physical_steps=24)
    assert settings["physical_steps"] == 24 and settings["decision_operator_steps"] == 8
    assert len(fine["actual_time_grid"]) == 25
    assert len(fine["decision_operator_grid"]) == 9
    np.testing.assert_array_equal(fine["decision_operator_grid"], numeric.operator_grid(base))
    np.testing.assert_array_equal(fine["decision_european"], decision_european(base, fine))
    assert np.max(np.abs(fine["decision_european"]-fine["european_proxy"])) > 1e-10
    public = value_fourier_stopping_policy(base, config=replace(fine_config, steps=24),
        spots=np.array([100.]), trace=True)["trace"]
    for key in ("stocks", "factors", "actual_time_grid", "european_proxy",
            "european_martingale", "premium_martingale"):
        np.testing.assert_array_equal(fine[key], public[key])
    result, diagnostics = numeric.evaluate_policy_set(fine, policies)
    assert diagnostics["frozen_entire_decision_function"]
    assert diagnostics["kernel_time_calendar_unchanged"] is False
    assert result["raw_lower"].shape == (64, 3, 1)
    invalid = {**fine, "comparison_physical_steps": 8}
    with pytest.raises(ValueError, match="matching physical"):
        numeric.evaluate_policy_set(invalid, policies)


def test_supported_explicit_serializer_roundtrips_all_decisions_and_bool(fixture, tmp_path):
    policies, trace, _, _, _ = fixture
    original = {name: replace(rule, exercise_at_inception=np.bool_(rule.exercise_at_inception))
        for name, rule in policies.items()}
    before = numeric.policy_set_fingerprint(original)
    folder = tmp_path/"policies"
    record = numeric.save_policy_set(original, folder)
    restored = numeric.load_policy_set(folder)
    assert record["policy_set_fingerprint"] == before == numeric.policy_set_fingerprint(restored)
    for name in numeric.POLICY_NAMES:
        assert type(restored[name].exercise_at_inception) is bool
        for left, right in zip(numeric.stopping_rule(original[name], trace)[:3],
                numeric.stopping_rule(restored[name], trace)[:3]):
            np.testing.assert_array_equal(left, right)
    with np.load(folder/"coefficients.npz", allow_pickle=False) as arrays:
        assert all(arrays[key].dtype.kind != "O" for key in arrays.files)


def test_itm_numpy_inception_flag_normalized_before_trace_fingerprint(fixture, tmp_path):
    _, _, config, _, _ = fixture
    anchor = fixture[0]["fourier_premium_lsm"].base_policy
    policies, _ = numeric.train_policy_set(anchor.params, anchor.market, anchor.kernel,
        strike=110., maturity=anchor.maturity, config=config, lookahead_order=8)
    for rule in policies.values():
        assert type(rule.base_policy.exercise_at_inception) is bool
    trace, _ = numeric.generate_shared_trace(policies, config)
    expected, _ = numeric.evaluate_policy_set(trace, policies)
    numeric.save_policy_set(policies, tmp_path/"itm")
    loaded = numeric.load_policy_set(tmp_path/"itm")
    actual, _ = numeric.evaluate_policy_set(trace, loaded)
    for key in expected:
        np.testing.assert_array_equal(expected[key], actual[key])


@pytest.mark.parametrize("defect", ["inception", "coefficient", "config"])
def test_logical_fingerprint_detects_all_decision_relevant_changes(fixture, defect):
    policies = dict(fixture[0])
    old = numeric.policy_set_fingerprint(policies)
    rule = policies["cashflow_lsm"]
    if defect == "inception": rule = replace(rule, exercise_at_inception=not rule.exercise_at_inception)
    elif defect == "coefficient":
        base = rule.base_policy
        model = base.continuations[1]
        beta = model.beta.copy(); beta[0] = np.nextafter(beta[0], np.inf)
        values = list(base.continuations); values[1] = replace(model, beta=beta)
        rule = replace(rule, base_policy=replace(base, continuations=tuple(values)))
    else:
        rule = replace(rule, base_policy=replace(rule.base_policy,
            config=replace(rule.base_policy.config, training_seed=12880105)))
    policies["cashflow_lsm"] = rule
    assert old != numeric.policy_set_fingerprint(policies)


def test_disk_seal_rejects_coefficient_and_metadata_tampering(fixture, tmp_path):
    for kind in ("coefficient", "inception"):
        folder = tmp_path/kind
        numeric.save_policy_set(fixture[0], folder)
        if kind == "coefficient":
            with (folder/"coefficients.npz").open("ab") as handle: handle.write(b"changed")
        else:
            path = folder/"policies.json"
            record = json.loads(path.read_text())
            record["policies"]["cashflow_lsm"]["exercise_at_inception"] = True
            path.write_text(json.dumps(record))
        with pytest.raises(ValueError, match="seal|differ"):
            numeric.load_policy_set(folder)


@pytest.mark.parametrize("change", ["ridge", "basis", "calendar", "bounded"])
def test_matched_policy_economics_invariants(fixture, change):
    policies = dict(fixture[0])
    rule = policies["cashflow_lsm"]
    options = {"ridge": {"ridge": .01}, "basis": {"stock_knots": (.8, 1., 1.2)},
        "calendar": {"steps": 16}, "bounded": {"bounded_surrogate": False}}[change]
    policies["cashflow_lsm"] = replace(rule,
        base_policy=replace(rule.base_policy, config=replace(rule.base_policy.config, **options)))
    with pytest.raises(ValueError, match="physical law"):
        numeric.validate_policy_set(policies)


def test_numerical_failure_is_identifiable_and_not_replaced_with_zero(fixture):
    evaluator = numeric.StateBellmanLookahead(fixture[0]["fourier_premium_lsm"].base_policy)
    class InvalidProxy:
        def values(self, time, factors, means, clocks, active):
            return np.full(means.shape+(len(active),), np.nan)
    evaluator.affine = InvalidProxy()
    with pytest.raises(FloatingPointError, match="nonfinite"):
        evaluator.conditional_values(0, np.array([100.]), np.zeros((1, 2)))
