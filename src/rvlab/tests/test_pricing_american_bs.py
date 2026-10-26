"""Independent stopping references and stock sensitivities for the 1D engine."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import ndtr

from rvlab.pricing import (
    BlackScholesParams, CashDividend, DiscountCurve, DividendSchedule, FDConfig,
    HestonCurveParams, HestonParams, MeanReversionCurve, NumericalConvergenceError,
    RoughHestonParams, SpotMarket, UnsupportedModelError, evaluate_vanilla,
)
from rvlab.pricing._vanilla_common import time_grid


def _market(spot=100., rate=.05, events=(), coverage=2.):
    return SpotMarket(spot, DiscountCurve.flat(rate, coverage),
                      DividendSchedule(tuple(events), coverage))


def _evaluate(params=None, *, market=None, strike=100., maturity=.5,
              is_call=False, style="american", config=None, **kwargs):
    return evaluate_vanilla(BlackScholesParams(.2) if params is None else params,
        market=_market() if market is None else market, strike=strike,
        maturity=maturity, is_call=is_call, exercise_style=style,
        config=FDConfig(spot_steps=600, time_steps=600) if config is None else config,
        **kwargs)


def _black(spot, strike, maturity, rate, volatility, call):
    sign = 1 if call else -1
    sd = volatility*math.sqrt(maturity)
    d1 = (math.log(spot/strike)+(rate+.5*volatility**2)*maturity)/sd
    d2 = d1-sd
    price = sign*(spot*ndtr(sign*d1)-strike*math.exp(-rate*maturity)*ndtr(sign*d2))
    return price, sign*ndtr(sign*d1)


def _tree(spot, strike, maturity, rate, volatility, call, steps=1200,
          cash=0., event_step=None, american=True):
    """Independent CRR stopping tree; one cash jump retains separate subtrees.

    Cash-event subtrees never recombine across the event, avoiding the usual
    adjusted-spot shortcut. The reference uses no finite-difference operators.
    """
    dt = maturity/steps
    up = math.exp(volatility*math.sqrt(dt))
    probability = (math.exp(rate*dt)-1/up)/(up-1/up)
    discount = math.exp(-rate*dt)
    sign = 1 if call else -1
    payoff = lambda s: np.maximum(sign*(s-strike), 0.)
    if event_step is None:
        node_spots = spot*up**(2*np.arange(steps+1)-steps)
        values = payoff(node_spots)
        for step in range(steps-1, -1, -1):
            values = discount*((1-probability)*values[:-1]+probability*values[1:])
            if american:
                values = np.maximum(values, payoff(spot*up**(2*np.arange(step+1)-step)))
        return float(values[0])
    before = spot*up**(2*np.arange(event_step+1)-event_step)
    after = np.maximum(before-cash, 0.)
    remaining = steps-event_step
    values = payoff(after[:, None]*up**(2*np.arange(remaining+1)-remaining))
    for step in range(remaining-1, -1, -1):
        values = discount*((1-probability)*values[:, :-1]+probability*values[:, 1:])
        if american:
            values = np.maximum(values, payoff(after[:, None]*up**(2*np.arange(step+1)-step)))
    values = values[:, 0]
    if american:
        values = np.maximum(values, payoff(before))
    for step in range(event_step-1, -1, -1):
        values = discount*((1-probability)*values[:-1]+probability*values[1:])
        if american:
            values = np.maximum(values, payoff(spot*up**(2*np.arange(step+1)-step)))
    return float(values[0])


@pytest.mark.parametrize("call", [True, False])
@pytest.mark.parametrize("rate", [.05, -.04])
def test_european_bs_price_and_stock_delta_match_analytic(call, rate):
    result = _evaluate(market=_market(rate=rate), strike=105., is_call=call,
                       style="european", config=FDConfig(spot_steps=800, time_steps=800))
    expected_price, expected_delta = _black(100., 105., .5, rate, .2, call)
    assert result.prices == pytest.approx(expected_price, abs=.002)
    assert result.deltas == pytest.approx(expected_delta, abs=.002)


@pytest.mark.parametrize("call,rate,strike", [(False, .05, 105.), (True, -.07, 90.)])
def test_american_prices_match_independent_tree_with_positive_and_negative_rates(call, rate, strike):
    result = _evaluate(market=_market(rate=rate), strike=strike, is_call=call)
    reference = .5*sum(_tree(100., strike, .5, rate, .2, call, steps=n) for n in (1200, 1201))
    assert result.prices == pytest.approx(reference, abs=.006)
    european = _evaluate(market=_market(rate=rate), strike=strike, is_call=call, style="european")
    assert result.prices >= european.prices-1e-8
    assert result.prices > european.prices+.02


def test_no_dividend_american_call_matches_european_for_nonnegative_rates():
    american = _evaluate(is_call=True)
    european = _evaluate(is_call=True, style="european")
    assert american.prices == pytest.approx(european.prices, abs=1e-7)
    assert american.deltas == pytest.approx(european.deltas, abs=1e-7)


@pytest.mark.parametrize("call", [False, True])
def test_discrete_dividend_prices_match_nonrecombining_cash_tree(call):
    market = _market(events=(CashDividend(.25, 3.),))
    result = _evaluate(market=market, is_call=call,
                       config=FDConfig(spot_steps=1000, time_steps=800))
    reference = .5*sum(_tree(100., 100., .5, .05, .2, call,
        steps=2*n, cash=3., event_step=n) for n in (400, 401))
    assert result.prices == pytest.approx(reference, abs=.012)
    european = _evaluate(market=market, is_call=call, style="european",
                         config=FDConfig(spot_steps=1000, time_steps=800))
    assert result.prices >= european.prices-1e-8
    event = result.diagnostics["dividend_event_sensitivities"][0]
    assert event["time"] == .25
    assert event["post_spot"] == 97.
    assert np.isfinite(event["pre_delta"]).all()
    assert np.isfinite(event["post_delta"]).all()


def test_cash_jump_is_limited_by_stock_and_exercise_precedes_the_jump():
    market = _market(events=(CashDividend(.25, 150.),))
    american = _evaluate(market=market, is_call=True, strike=90.)
    european = _evaluate(market=market, is_call=True, strike=90., style="european")
    reference = .5*sum(_tree(100., 90., .5, .05, .2, True,
        steps=2*n, cash=150., event_step=n) for n in (300, 301))
    assert american.prices == pytest.approx(reference, abs=.015)
    assert american.prices > european.prices+5.
    assert american.diagnostics["dividend_event_sensitivities"][0]["post_spot"] == 0.
    assert american.prices <= 100.


def test_stock_delta_holds_cash_and_curve_fixed_during_spot_bumps():
    market = _market(events=(CashDividend(.25, 3.),))
    config = FDConfig(spot_steps=800, time_steps=600)
    base = _evaluate(market=market, config=config)
    bump = .1
    values = [_evaluate(market=_market(spot=100.+sign*bump, events=market.dividends.events),
        config=config, grid=base.grid).prices for sign in (-1, 1)]
    assert base.deltas == pytest.approx((values[1]-values[0])/(2*bump), abs=.002)


def test_expiration_payoff_and_one_sided_atm_deltas():
    result = _evaluate(strike=[90., 100., 110.], maturity=0., is_call=[True, True, False])
    np.testing.assert_array_equal(result.prices, [10., 0., 10.])
    np.testing.assert_allclose(result.deltas, [1., .5, -1.], atol=1e-8)
    assert result.delta_left[1] == 0
    assert result.delta_right[1] == pytest.approx(1)


def test_broadcast_batch_preserves_contract_positions_and_matches_scalar_same_grid():
    config = FDConfig(spot_steps=100, time_steps=80)
    strikes = np.array([90., 100., 110.])
    maturities = np.array([[.2], [.5]])
    calls = np.array([True, False, True])
    result = _evaluate(strike=strikes, maturity=maturities, is_call=calls, config=config)
    assert result.prices.shape == result.deltas.shape == (2, 3)
    for i, t in enumerate(maturities[:, 0]):
        for j, k in enumerate(strikes):
            scalar = _evaluate(strike=float(k), maturity=float(t), is_call=bool(calls[j]),
                               config=config, grid=result.grid)
            assert result.prices[i, j] == pytest.approx(scalar.prices, abs=1e-10)
            assert result.deltas[i, j] == pytest.approx(scalar.deltas, abs=1e-10)


def test_constant_deterministic_heston_is_exactly_the_bs_engine():
    heston = HestonParams(.04, .04, 2., 0., -.7)
    for style in ("european", "american"):
        bs = _evaluate(style=style)
        actual = _evaluate(heston, style=style)
        assert actual.prices == pytest.approx(bs.prices, abs=1e-12)
        assert actual.deltas == pytest.approx(bs.deltas, abs=1e-12)
        assert actual.diagnostics["solvers"][0]["backend"] == "bs_cn_policy_iteration"


def test_deterministic_time_varying_heston_european_recovers_integrated_variance():
    v0, theta, kappa, maturity = .16, .01, 5., .6
    params = HestonParams(v0, theta, kappa, 0., -.5)
    integrated = theta*maturity+(v0-theta)*(-math.expm1(-kappa*maturity))/kappa
    expected, _ = _black(100., 105., maturity, .05, math.sqrt(integrated/maturity), False)
    result = _evaluate(params, strike=105., maturity=maturity, style="european",
                       config=FDConfig(spot_steps=1000, time_steps=1000))
    assert result.prices == pytest.approx(expected, abs=.002)


def test_deterministic_american_variance_path_cannot_be_replaced_by_average_volatility():
    params = HestonParams(.36, .01, 8., 0., -.5)
    maturity = .7
    integrated = .01*maturity+(.36-.01)*(-math.expm1(-8*maturity))/8
    market = _market(rate=.1)
    config = FDConfig(spot_steps=800, time_steps=1000)
    actual = _evaluate(params, market=market, strike=120., maturity=maturity, config=config)
    average = _evaluate(BlackScholesParams(math.sqrt(integrated/maturity)),
                        market=market, strike=120., maturity=maturity, config=config)
    assert abs(actual.prices-average.prices) > .1
    refined = _evaluate(params, market=market, strike=120., maturity=maturity,
        config=FDConfig(spot_steps=1200, time_steps=1600))
    assert actual.prices == pytest.approx(refined.prices, abs=.008)


def test_calendar_time_knots_and_dividends_enter_the_exact_event_clock():
    params = HestonCurveParams(.08, 4., 0., -.5,
        MeanReversionCurve((.137, .343), (.01, .2, .02)))
    market = SpotMarket(100., DiscountCurve((0., .173, 1.), (1., .995, .96)),
        DividendSchedule((CashDividend(.221, 2.),), 1.))
    clock = time_grid(params, market, .6, FDConfig(spot_steps=100, time_steps=30))
    assert all(knot in clock for knot in (.137, .343, .173, .221))
    # A flat schedule is the same calendar-time deterministic diffusion.
    flat = HestonCurveParams(.04, 2., 0., -.7, MeanReversionCurve((.137,), (.04, .04)))
    actual = _evaluate(flat, market=market)
    constant = _evaluate(HestonParams(.04, .04, 2., 0., -.7), market=market)
    assert actual.prices == pytest.approx(constant.prices, abs=2e-5)


def test_nonflat_deterministic_heston_curve_matches_piecewise_integrated_variance():
    params = HestonCurveParams(.08, 4., 0., -.5,
        MeanReversionCurve((.137, .343), (.01, .2, .02)))
    maturity = .6
    integrated, variance, start = 0., params.v0, 0.
    for end, level in zip((.137, .343, maturity), (.01, .2, .02)):
        dt = end-start
        decay = math.exp(-params.kappa*dt)
        integrated += level*dt+(variance-level)*(1-decay)/params.kappa
        variance = level+(variance-level)*decay
        start = end
    expected, _ = _black(100., 105., maturity, .05, math.sqrt(integrated/maturity), False)
    result = _evaluate(params, strike=105., maturity=maturity, style="european",
                       config=FDConfig(spot_steps=1000, time_steps=1000))
    assert result.prices == pytest.approx(expected, abs=.002)


@pytest.mark.parametrize("rate", [.05, -.05])
def test_american_vanilla_bounds_across_moneyness_and_option_type(rate):
    strikes = np.array([80., 100., 120., 80., 100., 120.])
    calls = np.array([True, True, True, False, False, False])
    result = _evaluate(market=_market(rate=rate), strike=strikes, is_call=calls)
    intrinsic = np.maximum(np.where(calls, 1., -1.)*(100.-strikes), 0.)
    # For negative rates a put can be worth more than its nominal strike.
    upper = np.where(calls, 100., strikes*max(1., math.exp(-rate*.5)))
    assert np.all(result.prices >= intrinsic-1e-8)
    assert np.all(result.prices <= upper+1e-8)


def test_deterministic_zero_volatility_uses_optimal_stopping_and_event_kinks():
    market = _market(rate=0., events=(CashDividend(.25, 20.),))
    call = _evaluate(BlackScholesParams(0.), market=market, strike=90., is_call=True)
    euro = _evaluate(BlackScholesParams(0.), market=market, strike=90., is_call=True,
                     style="european")
    assert call.prices == 10. and euro.prices == 0.
    assert call.deltas == pytest.approx(1., abs=1e-8)


@pytest.mark.parametrize("kwargs", [
    {"strike": 0.}, {"strike": True}, {"strike": complex(100, 0)},
    {"maturity": -.1}, {"maturity": float("nan")}, {"is_call": 1},
    {"strike": [90., 100.], "maturity": [.1, .2, .3]},
    {"style": "bermudan"}, {"price_atol": -1.},
])
def test_invalid_vanilla_requests_raise_instead_of_zero_prices(kwargs):
    with pytest.raises((TypeError, ValueError)):
        _evaluate(**kwargs)


def test_missing_dividend_schedule_and_insufficient_market_coverage_are_explicit():
    with pytest.raises(TypeError, match="explicit DividendSchedule"):
        SpotMarket(100., DiscountCurve.flat(.05, 1.), None)
    with pytest.raises(ValueError, match="coverage"):
        _evaluate(market=_market(coverage=.1), maturity=.5)
    with pytest.raises(ValueError, match="dividend expectation coverage"):
        _evaluate(market=SpotMarket(100., DiscountCurve.flat(.05, 1.), DividendSchedule((), .1)))


def test_discount_curve_accepts_negative_rates_without_extrapolation():
    curve = DiscountCurve((0., .2, 1.), (1., 1.01, 1.03))
    assert curve.discount(.1) == pytest.approx(math.sqrt(1.01))
    assert curve.rate(.1) < 0
    with pytest.raises(ValueError, match="coverage"):
        curve.discount(1.01)
    with pytest.raises(ValueError, match="positive"):
        DiscountCurve((0., 1.), (1., 0.))


def test_american_rough_heston_requires_its_explicit_kernel():
    with pytest.raises(ValueError, match="explicit ExponentialKernel"):
        _evaluate(RoughHestonParams(.04, .04, 1., .2, -.6, .15))


def test_policy_iteration_exhaustion_is_identifiable():
    with pytest.raises(NumericalConvergenceError, match="policy iteration exhausted"):
        _evaluate(config=FDConfig(spot_steps=100, time_steps=20, max_lcp_iterations=1))
