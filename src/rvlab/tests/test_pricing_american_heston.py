"""Independent references for the Heston exercise-constrained PDE backend."""
from dataclasses import replace

import numpy as np
import pytest

from rvlab.pricing._vanilla_heston import solve_heston
from rvlab.pricing._vanilla_common import time_grid
from rvlab.pricing.vanilla import evaluate_vanilla
from rvlab.pricing.heston import heston_price
from rvlab.pricing.types import BlackScholesParams, HestonParams, PricingConfig, PricingError
from rvlab.pricing.vanilla_types import (
    CashDividend, DiscountCurve, DividendSchedule, FDConfig, SpotMarket,
)
from rvlab.pricing.variance_curve import (
    HestonCurveParams, MeanReversionCurve, heston_curve_price,
)


def market(spot=100., rate=.03, horizon=1., events=()):
    return SpotMarket(spot, DiscountCurve.flat(rate, horizon),
                      DividendSchedule(tuple(events), horizon))


def test_european_heston_batch_matches_independent_fourier_prices_and_deltas():
    params = HestonParams(.04, .04, 1.5, .3, -.6)
    m = market()
    strikes = np.array([90., 100., 110., 90., 100., 110.])
    calls = np.array([True, True, True, False, False, False])
    maturity = .5
    numerical = solve_heston(params, m, strikes, maturity, calls,
                             FDConfig(240, 80, 200), exercise_style="european")
    config = PricingConfig(256, 400, 1024)

    def reference(spot):
        return heston_price(params, forward=spot*np.exp(.03*maturity),
                            discount=np.exp(-.03*maturity), strike=strikes,
                            maturity=maturity, is_call=calls, config=config)

    np.testing.assert_allclose(numerical["prices"], reference(100.), atol=.002)
    reference_delta = (reference(100.01) - reference(99.99))/.02
    np.testing.assert_allclose(numerical["deltas"], reference_delta, atol=.001)
    np.testing.assert_allclose(numerical["prices"][:3] - numerical["prices"][3:],
                               100.-strikes[:3]*np.exp(-.03*maturity), atol=1e-6)
    # A maturity batch reuses each factorization across all six contracts.
    assert numerical["diagnostics"]["factorizations"] <= 4
    assert numerical["diagnostics"]["contract_count"] == 6


@pytest.mark.parametrize("v0,reference", [(.0625, .5202), (.25, .7961)])
def test_published_american_heston_put_benchmarks(v0, reference):
    # Haentjens & in 't Hout, arXiv:1309.0110, Tables 2--3. The references
    # below are the independent ZFV method; published methods differ by up to
    # ~.0005 at these finite grids. We compare within .001, not rounded digits.
    params = HestonParams(v0, .16, 5., .9, .1)
    result = solve_heston(params, market(10., .1, .25), [10.], .25, [False],
                          FDConfig(240, 120, 240))
    assert result["prices"][0] == pytest.approx(reference, abs=.001)


def test_published_feller_violating_case_and_spatial_convergence():
    # Same paper, equation (5.3)/Table 4. Its MCS-IT approximation is 3.2126
    # and the independently computed 60-exercise-date COS value is 3.2079.
    # They are finite-grid/stopping approximations, not exact four-digit truth.
    params = HestonParams(.0348, .0348, 1.15, .39, -.64)
    assert 2*params.kappa*params.theta < params.vol_of_vol**2
    m = market(100., .04, .25)
    coarse = solve_heston(params, m, [100.], .25, [False], FDConfig(120, 60, 240))
    fine = solve_heston(params, m, [100.], .25, [False], FDConfig(240, 120, 240))
    assert fine["prices"][0] == pytest.approx(3.2126, abs=.006)
    assert fine["prices"][0] >= 3.2079 - .002
    assert abs(fine["prices"][0] - 3.2126) < abs(coarse["prices"][0] - 3.2126)


def test_time_dependent_mean_reversion_runs_in_calendar_time():
    curve = MeanReversionCurve((.2, .5), (.08, .01, .06))
    params = HestonCurveParams(.04, 2., .4, -.7, curve)
    m, strikes, maturity = market(), np.array([90., 100., 110.]), .8
    result = solve_heston(params, m, strikes, maturity, [False]*3,
                          FDConfig(240, 80, 240), exercise_style="european")
    reference = heston_curve_price(params, forward=100*np.exp(.03*maturity),
                                    discount=np.exp(-.03*maturity), strike=strikes,
                                    maturity=maturity, is_call=False,
                                    config=PricingConfig(256, 400, 1024))
    np.testing.assert_allclose(result["prices"], reference, atol=.002)
    # A backward-reversed theta schedule gives a materially different answer.
    reversed_params = replace(params, curve=MeanReversionCurve((.3, .6), (.06, .01, .08)))
    reversed_prices = heston_curve_price(reversed_params, forward=100*np.exp(.03*maturity),
                                         discount=np.exp(-.03*maturity), strike=strikes,
                                         maturity=maturity, is_call=False,
                                         config=PricingConfig(256, 400, 1024))
    assert max(abs(result["prices"]-reversed_prices)) > .05


def test_flat_curve_recovers_constant_heston_without_remeshing():
    constant = HestonParams(.04, .045, 1.5, .3, -.6)
    curved = HestonCurveParams(.04, 1.5, .3, -.6, MeanReversionCurve((.1,), (.045, .045)))
    m, config = market(), FDConfig(100, 40, 100)
    a = solve_heston(constant, m, [100.], .25, [False], config)
    b = solve_heston(curved, m, [100.], .25, [False], config, grid=a["grid"])
    np.testing.assert_allclose(a["prices"], b["prices"], atol=1e-7)
    np.testing.assert_allclose(a["deltas"], b["deltas"], atol=1e-7)


def test_cash_dividend_has_pre_event_exercise_and_restarts_damping():
    params = HestonParams(.04, .04, 2., .1, -.6)
    config = FDConfig(120, 40, 120)
    m = market(events=(CashDividend(.1, 3.),))
    american = solve_heston(params, m, [90.], .25, [True], config)
    european = solve_heston(params, m, [90.], .25, [True], config, exercise_style="european")
    assert american["prices"][0] > european["prices"][0] + 1.
    records = [r for r in american["exercise_boundaries"] if r.get("event_side") == "before"]
    assert records and records[0]["spot_boundary"] is not None
    assert american["diagnostics"]["fully_implicit_damping_steps"] == 4
    assert american["diagnostics"]["dividend_events"][0]["time"] == .1


def test_exercise_disabled_call_and_negative_rate_put_boundaries():
    params = HestonParams(.04, .04, 2., .2, -.6)
    config = FDConfig(120, 40, 120)
    positive = market(rate=.03)
    american = solve_heston(params, positive, [100.], .25, [True], config)
    european = solve_heston(params, positive, [100.], .25, [True], config,
                            exercise_style="european")
    np.testing.assert_allclose(american["prices"], european["prices"], atol=1e-7)
    negative = market(spot=5., rate=-.05)
    american = solve_heston(params, negative, [100.], .5, [False], config)
    european = solve_heston(params, negative, [100.], .5, [False], config,
                            exercise_style="european")
    assert american["prices"][0] > 95.
    assert american["prices"][0] >= european["prices"][0] - 1e-6


def test_dividend_at_expiry_uses_pre_event_exercise_and_post_event_payoff():
    params = HestonParams(.04, .04, 2., .1, -.6)
    config = FDConfig(100, 40, 100)
    m = market(events=(CashDividend(.25, 3.),))
    american = solve_heston(params, m, [90.], .25, [True], config)
    european = solve_heston(params, m, [90.], .25, [True], config,
                            exercise_style="european")
    no_dividend = solve_heston(params, market(), [90.], .25, [True], config)
    assert american["prices"][0] > european["prices"][0] + 1.
    np.testing.assert_allclose(american["prices"], no_dividend["prices"], atol=.001)
    assert {r["event_side"] for r in american["exercise_boundaries"]
            if "event_side" in r} == {"before", "after"}


def test_cash_event_is_applied_once_next_to_roundoff_time_node():
    params = HestonParams(.04, .04, 2., .1, -.6)
    m = market(events=(CashDividend(.05, 3.),))
    # np.linspace(0,.1,301)[150] is .05000000000000001. Matching events
    # approximately used to map this cash payment at both neighboring times.
    config = FDConfig(60, 20, 300)
    clock = time_grid(params, m, .1, config)
    assert np.count_nonzero(clock == .05) == 1
    assert np.min(np.diff(clock)) > 1e-10
    result = solve_heston(params, m, [100.], .1, [True], config)
    assert len(result["diagnostics"]["dividend_event_sensitivities"]) == 1
    assert result["diagnostics"]["fully_implicit_damping_steps"] == 4


def test_variance_domain_failure_is_identifiable():
    params = HestonParams(6., .04, 2., .2, -.6)
    with pytest.raises(PricingError, match="contain v0"):
        solve_heston(params, market(), [100.], .25, [False], FDConfig(40, 20, 40))


def test_future_only_mean_reversion_does_not_spoil_absorbing_zero_variance():
    params = HestonCurveParams(0., 2., .3, -.6,
                               MeanReversionCurve((1.,), (0., .04)))
    m = market(horizon=2.)
    config = FDConfig(40, 20, 20)
    for style in ("american", "european"):
        result = evaluate_vanilla(params, market=m, strike=[100., 100.],
            maturity=[.25, .5], is_call=[True, False],
            exercise_style=style, config=config)
        np.testing.assert_allclose(result.prices, [100.-100.*np.exp(-.03*.25), 0.], atol=1e-12)
        # The American ATM put can exercise immediately: its zero value has
        # a kink, whose left/right sensitivities are -1/0 respectively.
        np.testing.assert_allclose(result.deltas,
                                   [1., -.5 if style == "american" else 0.], atol=1e-8)
        if style == "american":
            assert result.delta_left[1] == pytest.approx(-1.)
            assert result.delta_right[1] == 0.
        assert {row["backend"] for row in result.diagnostics["solvers"]} == {"deterministic_optimal_stopping"}


def test_american_heston_delta_grid_and_fixed_input_bump_convergence():
    params = HestonParams(.04, .04, 2., .2, -.6)
    m = market(events=(CashDividend(.1, 2.),))
    contracts = dict(strike=[90., 100., 110.], maturity=.25,
                     is_call=[True, False, False], exercise_style="american")
    resolutions = (FDConfig(120, 40, 120), FDConfig(240, 80, 240))
    results = []
    for config in resolutions:
        result = evaluate_vanilla(params, market=m, config=config, **contracts)
        results.append(result)
        bump_errors = []
        for bump in (.1, .05):
            # Only observed spot changes. In particular, no recalibration,
            # dividend adjustment, or candidate-dependent remeshing occurs.
            up = evaluate_vanilla(params, market=replace(m, spot=m.spot+bump),
                                  config=config, grid=result.grid, **contracts)
            down = evaluate_vanilla(params, market=replace(m, spot=m.spot-bump),
                                    config=config, grid=result.grid, **contracts)
            assert up.diagnostics["grid_sha256"] == down.diagnostics["grid_sha256"] == result.diagnostics["grid_sha256"]
            bumped = (up.prices-down.prices)/(2*bump)
            error = float(np.max(np.abs(bumped-result.deltas)))
            bump_errors.append(error)
            np.testing.assert_allclose(bumped, result.deltas, atol=5e-6)
        assert bump_errors[1] <= .3*bump_errors[0]+1e-9
    np.testing.assert_allclose(results[0].deltas, results[1].deltas, atol=.001)


def test_small_nonzero_vol_of_vol_cash_dividend_limit_matches_independent_bs():
    # A genuinely stochastic 2D solve approaches the independently validated
    # BS policy-iteration engine; this must not delegate at eta>0. The BS
    # engine is also checked against a nonrecombining cash-dividend stopping
    # tree in test_pricing_american_bs.py.
    params = HestonParams(.04, .04, 2., .0001, -.6)
    m = market(events=(CashDividend(.1, 2.),))
    contracts = dict(strike=[90., 100., 110.], maturity=.25,
                     is_call=[True, False, False], exercise_style="american")
    reference = evaluate_vanilla(BlackScholesParams(.2), market=m,
                                 config=FDConfig(800, 80, 800), **contracts)
    errors = []
    for config in (FDConfig(120, 40, 120), FDConfig(240, 80, 240)):
        result = evaluate_vanilla(params, market=m, config=config, **contracts)
        assert result.diagnostics["solvers"][0]["backend"] == "heston_mcs_ikonen_toivanen"
        errors.append(float(np.max(np.abs(result.prices-reference.prices))))
        np.testing.assert_allclose(result.deltas, reference.deltas, atol=.001)
    assert errors[1] < .5*errors[0]
    assert errors[1] < .001
