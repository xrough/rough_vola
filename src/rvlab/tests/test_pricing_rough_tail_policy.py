"""Margin challengers cannot inspect future paths or change the common upper."""
from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np
import pytest

from rvlab.experiments._pricing_rough_tail_policy import (
    BASELINE_NAME, MarginContinuation, build_margin_policies,
    evaluate_policy_candidates, margin_policy_specs, select_margin_policy,
    spent_gap_timing)
from rvlab.pricing import (DiscountCurve, DividendSchedule, RoughHestonParams,
    SpotMarket, approximate_fractional_kernel)
from rvlab.pricing.rough_estimators import RoughEuropeanControlConfig
from rvlab.pricing.rough_policy_diagnostics import ControlledExercisePolicy
from rvlab.pricing.rough_stopping import RoughAmericanConfig
from rvlab.pricing.rough_stopping_fourier import FourierStoppingPolicy, fourier_policy_fingerprint


@dataclass(frozen=True)
class ConstantPremium:
    amount: float

    def value(self, stock, factors, strike):
        return np.full_like(stock, self.amount, dtype=float)


@pytest.fixture
def policy_trace():
    params = RoughHestonParams(.04, .04, 1.5, .3, -.6, .15)
    market = SpotMarket(100., DiscountCurve.flat(.04, 1.), DividendSchedule((), 1.))
    kernel = approximate_fractional_kernel(.15, 1., 4, min_time_ratio=.003)
    times = np.array([0., .1, .2]); models = (ConstantPremium(0.), ConstantPremium(.006), ConstantPremium(0.))
    config = RoughAmericanConfig(training_paths=8, valuation_paths=4, training_seed=10,
        valuation_seed=11, steps=4, exercise_steps=2, batch_size=2)
    proxy = FourierStoppingPolicy(params, market, kernel, 100., .2, False, "american", times, times,
        models, models, config, RoughEuropeanControlConfig(), False, {}, .7)
    policy = ControlledExercisePolicy(proxy, times, models, False, (10, 12), 0., .7, {})
    stocks = np.array([[[100.], [98.], [97.]], [[100.], [98.], [99.]]])
    discounts = market.discount_curve.discount(times)
    payoff = np.maximum(100. - stocks, 0.) * discounts[None, :, None]
    euro = payoff.copy(); euro[:, 0] = 2.86; euro[:, 1] = discounts[1] * 1.995
    zeros = np.zeros_like(stocks)
    trace = {"times": times, "eligible": np.ones(3, bool), "discounted_payoffs": payoff,
        "european_martingale": zeros.copy(), "premium_martingale": zeros.copy(),
        "stocks": stocks, "factors": np.zeros((2, 3, len(kernel.nodes))), "european_proxy": euro,
        "stop_indices": np.full((2, 1), 2, dtype=int), "seed": 30,
        "policy_fingerprint": fourier_policy_fingerprint(proxy)}
    return policy, trace


def test_margin_clips_base_before_adjustment_then_reclips_and_vanishes_at_expiry():
    stock = np.ones(3); factors = np.zeros((3, 4))
    clipped_first = MarginContinuation(ConstantPremium(1.), 0., .9, .5, .7)
    np.testing.assert_allclose(clipped_first.value(stock, factors, 100.), .63)
    shifted = MarginContinuation(ConstantPremium(.006), .02, 1., .5, .7)
    np.testing.assert_allclose(shifted.value(stock, factors, 100.), .016)
    terminal = replace(shifted, remaining_fraction=0.)
    np.testing.assert_array_equal(terminal.value(stock, factors, 100.), np.zeros(3))
    for base in (-1., 1.):
        wrapped = MarginContinuation(ConstantPremium(base), -.02, 1.1, .5, .7)
        assert np.all((wrapped.value(stock, factors, 100.) >= 0) & (wrapped.value(stock, factors, 100.) <= .7))


def test_seven_exact_predeclared_rules_leave_fallback_unchanged(policy_trace):
    policy, _ = policy_trace
    policies = build_margin_policies(policy)
    assert len(policies) == len(margin_policy_specs()) == 7
    assert policies[BASELINE_NAME] is policy
    for candidate in policies.values():
        assert candidate.training_seeds == policy.training_seeds
        assert candidate.premium_floor == 0. and candidate.premium_cap == .7
        assert candidate.base_policy is policy.base_policy
    itm = replace(policy, base_policy=replace(policy.base_policy,
        market=replace(policy.base_policy.market, spot=99.)))
    with pytest.raises(ValueError, match="inception"):
        build_margin_policies(itm)


def test_own_decisions_are_nonanticipating_and_common_upper_is_identical(policy_trace):
    policy, trace = policy_trace
    candidates = build_margin_policies(policy)
    m = np.zeros((2, 3))
    evaluated = evaluate_policy_candidates(trace, candidates, m)
    baseline = evaluated[BASELINE_NAME]
    earlier = evaluated["premium offset -0.005"]
    np.testing.assert_array_equal(baseline["stops"], [2, 2])
    # Identical information at t=.1 must give the same action despite opposite futures.
    np.testing.assert_array_equal(earlier["stops"], [1, 1])
    for result in evaluated.values():
        np.testing.assert_array_equal(result["upper"], baseline["upper"])
        assert np.all(result["upper"] >= result["lower"] - 2e-11)
    assert not np.array_equal(earlier["raw"], baseline["raw"])


def test_all_training_and_selection_seed_firewalls(policy_trace):
    policy, trace = policy_trace
    candidates = build_margin_policies(policy); m = np.zeros((2, 3))
    for seed in (10, 12, 44):
        changed = {**trace, "seed": seed}
        with pytest.raises(ValueError, match="independent"):
            evaluate_policy_candidates(changed, candidates, m, forbidden_seeds=(44,))
        evaluate_policy_candidates(changed, candidates, m, require_independent=False, forbidden_seeds=(44,))
    evaluated = evaluate_policy_candidates(trace, candidates, m)
    _, sealed, _ = select_margin_policy(evaluated, candidates, selection_seed=30)
    with pytest.raises(ValueError, match="independent"):
        evaluate_policy_candidates(trace, sealed, m)


def test_selection_uses_paired_lower_lcb_with_fallback_and_no_upper_change(policy_trace):
    policy, trace = policy_trace
    candidates = build_margin_policies(policy)
    evaluated = evaluate_policy_candidates(trace, candidates, np.zeros((2, 3)))
    same = {name: deepcopy(evaluated[BASELINE_NAME]) for name in candidates}
    selected, _, rows = select_margin_policy(same, candidates, selection_seed=30)
    assert selected == BASELINE_NAME and not rows.final_paths_used_for_selection.any()
    first = next(name for name in candidates if name != BASELINE_NAME)
    same[first]["lower"] += .001
    selected, _, _ = select_margin_policy(same, candidates, selection_seed=30)
    assert selected == first
    same[first]["upper"][0] += .001
    with pytest.raises(ValueError, match="identical upper"):
        select_margin_policy(same, candidates, selection_seed=30)


def test_spent_peak_times_are_descriptive_not_future_information_for_a_rule():
    payoff = np.array([[0., 1., 2.], [0., 3., 1.]])
    data = spent_gap_timing(payoff, np.zeros_like(payoff), np.array([1, 2]),
        times=np.array([0., .1, .2]), seed=42)
    assert data.peak_after_stop.tolist() == [True, False]
    assert data.peak_before_stop.tolist() == [False, True]
    np.testing.assert_array_equal(data.gap, [1., 2.])
    assert not data.acceptance_input.any() and not data.used_for_selection.any()


@pytest.mark.parametrize("kind", ["shape", "nonfinite", "initial_nonzero"])
def test_invalid_common_control_is_rejected(policy_trace, kind):
    policy, trace = policy_trace; m = np.zeros((2, 3))
    if kind == "shape": m = m[:, 1:]
    if kind == "nonfinite": m[0, 1] = np.nan
    if kind == "initial_nonzero": m[0, 0] = .01
    with pytest.raises(ValueError, match="common martingale"):
        evaluate_policy_candidates(trace, build_margin_policies(policy), m)
