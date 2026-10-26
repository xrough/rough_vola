"""A deterministic same-calendar reference changes decisions, never the dual."""
from dataclasses import replace

import numpy as np
import pytest

from rvlab.pricing import RoughHestonParams, approximate_fractional_kernel
from rvlab.pricing.rough_policy_diagnostics import compare_trace_policies
from rvlab.pricing.rough_policy_reference import build_black_scholes_reference_policy
from rvlab.pricing.rough_stopping import RoughAmericanConfig
from rvlab.pricing.rough_stopping_fourier import (
    train_fourier_stopping_policy, value_fourier_stopping_policy,
)
from rvlab.pricing.rough_stopping_reference import black_scholes_stopping_reference
from rvlab.pricing.vanilla_types import (
    CashDividend, DiscountCurve, DividendSchedule, SpotMarket, UnsupportedModelError,
)


@pytest.fixture(scope="module")
def base():
    params = RoughHestonParams(.04, .04, 1.5, 0., -.6, .15)
    kernel = approximate_fractional_kernel(.15, 1., 2, min_time_ratio=.003)
    market = SpotMarket(100., DiscountCurve.flat(.04, 1.), DividendSchedule((), 1.))
    config = RoughAmericanConfig(training_paths=64, valuation_paths=64,
        steps=4, exercise_steps=4, batch_size=32, training_seed=260000042,
        valuation_seed=261000042, stock_knots=(.95, 1., 1.05))
    return train_fourier_stopping_policy(params, market, kernel,
        strike=100., maturity=60/365, config=config, premium_cap=.7)


def test_reference_decisions_share_the_trace_and_frozen_upper(base):
    policy = build_black_scholes_reference_policy(base, spot_nodes=2049)
    trace = value_fourier_stopping_policy(base, trace=True)["trace"]
    result = compare_trace_policies(trace, policy)
    np.testing.assert_array_equal(result["baseline"].dual_samples,
        result["candidate"].dual_samples)
    assert policy.training_seeds == (base.config.training_seed,)
    assert policy.diagnostics["training_paths"] == 0
    assert policy.diagnostics["actions_override_approximate_european_proxy"]
    assert not policy.diagnostics["valuation_used_in_training"]
    assert not policy.diagnostics["rough_model_substitution"]


def test_reference_actions_and_price_match_same_calendar_dp(base):
    policy = build_black_scholes_reference_policy(base, spot_nodes=4097)
    reference = black_scholes_stopping_reference(base.market, volatility=.2,
        strike=100., exercise_times=base.exercise_times, spot_nodes=4097,
        check_refinement=False)
    assert policy.diagnostics["reference_price"] == pytest.approx(reference.price, abs=1e-12)
    action = policy.continuations[-2].value(np.array([80., 100., 120.]),
        np.zeros((3, 2)), 100.)
    assert action[0] < 0 and action[1] > 0 and action[2] > 0
    with pytest.raises(ValueError):
        policy.continuations[-2].stocks[0] = 0.


def test_reference_does_not_substitute_constant_vol_for_rough_dynamics(base):
    with pytest.raises(ValueError, match="constant variance"):
        build_black_scholes_reference_policy(replace(base,
            params=replace(base.params, vol_of_vol=.3)))
    with pytest.raises(ValueError, match="constant variance"):
        build_black_scholes_reference_policy(replace(base,
            params=replace(base.params, theta=.05)))


@pytest.mark.parametrize("volatility", [.21, True, np.nan, "a", .2j, [.2]])
def test_reference_rejects_a_different_volatility(base, volatility):
    with pytest.raises(ValueError, match="volatility"):
        build_black_scholes_reference_policy(base, volatility=volatility)


def test_reference_rejects_different_calendar_and_cash_dividends(base):
    with pytest.raises(ValueError, match="calendar"):
        build_black_scholes_reference_policy(base, exercise_times=np.array([0., .1]))
    cash = SpotMarket(100., base.market.discount_curve,
        DividendSchedule((CashDividend(.08, 1.),), 1.))
    with pytest.raises(UnsupportedModelError, match="cash"):
        build_black_scholes_reference_policy(replace(base, market=cash))
