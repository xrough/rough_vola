"""Guarded American-call equivalence and shared deterministic calibration."""
from dataclasses import replace
import math

import numpy as np
import pandas as pd
import pytest

from rvlab.pricing import (
    BlackScholesParams, CashDividend, DiscountCurve, DividendSchedule, FDConfig,
    HestonParams, PricingConfig, PricingError, RoughHestonParams, SpotMarket,
    UnsupportedModelError, calibrate_american_call_surface, calibrate_surface,
    evaluate_vanilla, price, price_american_calls,
)
from rvlab.pricing import american_calls


@pytest.fixture
def market():
    return SpotMarket(100., DiscountCurve.flat(.04, 1.), DividendSchedule((), 1.))


def surface(params, market, config=None):
    k = np.tile([95., 100., 105.], 3)
    t = np.repeat([14/365, 30/365, 60/365], 3)
    q = pd.DataFrame({"strike": k, "maturity": t, "is_call": True,
                      "exercise_style": "american"})
    q["market_price"] = price_american_calls(params, market=market, strike=k, maturity=t, config=config)
    return q


def test_bs_equivalence_broadcasting_expiration_and_price_bounds(market):
    p = BlackScholesParams(.2)
    k, t = np.array([[90.], [100.], [110.]]), np.array([0., .1, .5])
    actual = price_american_calls(p, market=market, strike=k, maturity=t)
    d = market.discount_curve.discount(t)
    expected = price(p, forward=100/d, discount=d, strike=k, maturity=t)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (3, 3)
    np.testing.assert_array_equal(actual[:, 0], [10., 0., 0.])
    assert np.all(actual >= np.maximum(100-k, 0))
    assert np.all(actual <= 100)
    assert isinstance(price_american_calls(p, market=market, strike=100, maturity=.1), float)


def test_heston_matches_american_finite_difference_without_dividends(market):
    p = HestonParams(.04, .045, 1.5, .3, -.6)
    strikes = np.array([95., 100., 105.])
    reduced = price_american_calls(p, market=market, strike=strikes, maturity=.25,
                                  config=PricingConfig(192, 160, 512))
    reference = evaluate_vanilla(p, market=market, strike=strikes, maturity=.25,
                                exercise_style="american", is_call=True,
                                config=FDConfig(spot_steps=400, variance_steps=120,
                                                time_steps=600, spot_concentration=.05))
    np.testing.assert_array_less(np.abs(reduced-reference.prices), .001+1e-4*np.abs(reduced))


@pytest.mark.parametrize("maturity", [.03, .25])
def test_heston_and_rough_limits(market, maturity):
    p = HestonParams(.04, .04, 1., 0., -.6)
    bs = price_american_calls(BlackScholesParams(.2), market=market, strike=[95.,100.,105.], maturity=maturity)
    for candidate in [p, RoughHestonParams(.04,.04,1.,0.,-.6,.15),
                      RoughHestonParams(.04,.04,1.,0.,-.6,.5)]:
        actual = price_american_calls(candidate, market=market, strike=[95.,100.,105.], maturity=maturity)
        np.testing.assert_allclose(actual, bs, atol=1e-10)
    active = HestonParams(.04,.045,1.5,.3,-.6)
    rough = RoughHestonParams(.04,.045,1.5,.3,-.6,.5)
    np.testing.assert_array_equal(
        price_american_calls(active, market=market, strike=[95.,100.,105.], maturity=maturity),
        price_american_calls(rough, market=market, strike=[95.,100.,105.], maturity=maturity))


@pytest.mark.parametrize("flag", [False, [True,False]])
def test_puts_are_not_silently_sent_to_european_engine(market, flag):
    with pytest.raises(ValueError, match="calls only"):
        price_american_calls(BlackScholesParams(.2), market=market, strike=100, maturity=.1, is_call=flag)


@pytest.mark.parametrize("conditional", [{}, {"memory": 0}])
def test_conditional_state_rejected(market, conditional):
    with pytest.raises(UnsupportedModelError, match="conditional"):
        price_american_calls(BlackScholesParams(.2), market=market, strike=100,
                             maturity=.1, conditional_state=conditional)


def test_curve_models_and_subclasses_are_not_accidentally_accepted(market):
    from rvlab.pricing import HestonCurveParams, MeanReversionCurve
    curve = HestonCurveParams(.04, 1.5, .3, -.6, MeanReversionCurve((),(.04,)))
    with pytest.raises(UnsupportedModelError, match="constant"):
        price_american_calls(curve, market=market, strike=100, maturity=.1)
    class OtherBS(BlackScholesParams):
        pass
    with pytest.raises(UnsupportedModelError):
        price_american_calls(OtherBS(.2), market=market, strike=100, maturity=.1)


def test_dividends_missing_information_and_incomplete_coverage_rejected(market):
    with pytest.raises(TypeError, match="SpotMarket"):
        price_american_calls(BlackScholesParams(.2), market=None, strike=100, maturity=.1)
    for schedule, message in [(DividendSchedule((),.05), "coverage"),
                              (DividendSchedule((CashDividend(.05,1.),),1.), "empty")]:
        with pytest.raises(ValueError, match=message):
            price_american_calls(BlackScholesParams(.2), market=replace(market, dividends=schedule),
                                 strike=100, maturity=.1)
    with pytest.raises(ValueError, match="coverage"):
        price_american_calls(BlackScholesParams(.2), market=market, strike=100, maturity=2.)


def test_every_forward_interval_checked_not_only_positive_average_rate(market):
    curve = DiscountCurve((0., .1, .2, .3), (1., .99, .995, .98))
    m = replace(market, discount_curve=curve)
    assert curve.discount(.3) < 1.
    with pytest.raises(ValueError, match="every interval"):
        price_american_calls(BlackScholesParams(.2), market=m, strike=100., maturity=.3)
    # The future negative-rate segment is irrelevant to the earlier horizon.
    assert price_american_calls(BlackScholesParams(.2), market=m, strike=100, maturity=.1) > 0
    zero = replace(market, discount_curve=DiscountCurve.flat(0., 1.))
    assert price_american_calls(BlackScholesParams(.2), market=zero, strike=100, maturity=.2) > 0


def test_bs_calibration_preserves_order_styles_and_exposes_jacobian(market):
    q = surface(BlackScholesParams(.22), market).iloc[[7, 0, 6, 2, 5]].copy()
    q.index = ["duplicate", "z", "duplicate", "a", "c"]
    original = q.copy(deep=True)
    result = calibrate_american_call_surface(q, BlackScholesParams(.3), market=market)
    assert result.ready
    assert result.params.volatility == pytest.approx(.22, abs=1e-9)
    assert result.metadata["exercise_style"] == "american"
    assert result.metadata["conditioning"].startswith("new_implied_surface")
    assert not result.metadata["hedging_ready"]
    assert result.fitted_quotes.index.equals(q.index)
    assert (result.fitted_quotes.exercise_style == "american").all()
    pd.testing.assert_frame_equal(q, original)
    assert result.free_parameters == ("volatility",)
    assert result.final_jacobian.shape == (5,1)
    assert not result.final_jacobian.flags.writeable
    assert len(result.jacobian_singular_values) == 1
    assert result.jacobian_singular_values[0] > 0
    d = market.discount_curve.discount(q.maturity.to_numpy())
    np.testing.assert_allclose(result.fitted_quotes.forward, market.spot/d)
    np.testing.assert_allclose(result.fitted_quotes.residual, result.fitted_quotes.price_residual)
    with pytest.raises(ValueError, match="non-European"):
        calibrate_surface(q, BlackScholesParams(.3), forward=100, discount=1.)


@pytest.mark.parametrize("change", ["missing_style", "european", "put", "numeric_flag", "negative_maturity"])
def test_malformed_calibration_input_is_not_relabeled(market, change):
    q = surface(BlackScholesParams(.2), market)
    if change == "missing_style": q = q.drop(columns="exercise_style")
    elif change == "european": q.loc[1, "exercise_style"] = "european"
    elif change == "put": q.loc[1, "is_call"] = False
    elif change == "numeric_flag": q["is_call"] = 1
    else: q.loc[1, "maturity"] = -.1
    with pytest.raises(ValueError):
        calibrate_american_call_surface(q, BlackScholesParams(.3), market=market)


def test_spread_weighting_uses_midpoint_and_floor(market):
    q = pd.DataFrame({"strike": [100.,100.], "maturity": [.25,.25], "is_call": True,
                      "exercise_style": "american", "bid": [2.,2.], "ask": [2.,4.]})
    result = calibrate_american_call_surface(q, BlackScholesParams(.2), market=market,
                                            price_scale_floor=.01)
    target = (2/.01**2 + 3)/(1/.01**2+1)
    np.testing.assert_allclose(result.fitted_quotes.model_price, target, atol=1e-8)
    np.testing.assert_allclose(result.fitted_quotes.price_scale, [.01,1.])
    np.testing.assert_allclose(result.fitted_quotes.residual,
                               result.fitted_quotes.price_residual/[.01,1.])


def test_every_probe_must_meet_spread_budget_even_when_joint_change_is_zero(market, monkeypatch):
    config = PricingConfig(48,40,128)
    q = pd.DataFrame({"strike":[100.],"maturity":[.25],"is_call":True,
                      "exercise_style":"american", "bid":[4.995], "ask":[5.005]})
    def fake_price(params, **kwargs):
        settings = kwargs["config"]
        shift = .0008 if settings.quadrature_nodes == 96 and settings.fourier_cutoff == 40 else 0.
        return np.full(np.shape(kwargs["strike"]), 5.+shift)
    monkeypatch.setattr(american_calls, "price", fake_price)
    fit = calibrate_american_call_surface(q, BlackScholesParams(.2), market=market, config=config,
                                         bounds={"volatility":(.2,.2)})
    checks = fit.diagnostics.checks.set_index("check")
    assert checks.loc["joint", "numerical_stable"]
    assert checks.price_stable.all()
    assert not checks.loc["quadrature", "spread_stable"]
    assert not fit.numerical_stable and not fit.ready
    assert fit.optimizer_success


def test_refinement_failure_retains_fit_without_zero_substitution(market, monkeypatch):
    original = american_calls.price
    config = PricingConfig(48,40,128)
    q = surface(BlackScholesParams(.2), market)
    def fail_far(params, **kwargs):
        if kwargs["config"].fourier_cutoff == 80:
            raise PricingError("deliberate tail failure")
        return original(params, **kwargs)
    monkeypatch.setattr(american_calls, "price", fail_far)
    fit = calibrate_american_call_surface(q, BlackScholesParams(.2), market=market, config=config,
                                         bounds={"volatility":(.2,.2)})
    assert fit.optimizer_success and not fit.ready
    assert fit.fitted_quotes.model_price.notna().all()
    assert fit.fitted_quotes.refined_model_price.isna().all()
    assert fit.diagnostics.checks.error.dropna().str.contains("deliberate").all()


def test_exhaustion_is_separate_from_numerical_stability_and_observer_counts_calls(market):
    calls, observed = [], []
    q = surface(BlackScholesParams(.2), market)
    fit = calibrate_american_call_surface(q, BlackScholesParams(.4), market=market, max_nfev=1,
        evaluation_callback=lambda p,c,s: calls.append((p,c,s)),
        evaluation_observer=lambda p,c,s,v: observed.append((p,c,s,v.copy())))
    assert not fit.optimizer_success and fit.optimizer_status == 0 and not fit.ready
    assert fit.numerical_stable
    assert fit.pricing_evaluations == len(calls) == len(observed)
    assert fit.pricing_evaluations > fit.nfev
    assert {x[2] for x in calls} == {"calibration pricing", "final calibration pricing", "American-call refinement pricing"}


def test_deadline_callback_propagates_and_keeps_prior_candidate_available(market):
    q = surface(BlackScholesParams(.2), market)
    checkpoint = []
    class BudgetEnded(RuntimeError): pass
    def before(p,c,s):
        if checkpoint:
            raise BudgetEnded("deadline")
    with pytest.raises(BudgetEnded, match="deadline"):
        calibrate_american_call_surface(q, BlackScholesParams(.3), market=market,
            evaluation_callback=before,
            evaluation_observer=lambda p,c,s,v: checkpoint.append((p,v.copy())))
    assert len(checkpoint) == 1
    assert np.isfinite(checkpoint[0][1]).all()


def test_synthetic_rough_workflow_uses_finer_generation_and_physical_price_recovery(market):
    p = RoughHestonParams(.04,.045,1.5,.2,-.6,.15)
    # Numerical checks remain stringent. Fixing nuisance parameters isolates
    # source/API correctness; the full campaign measures joint identification.
    q = surface(p, market, PricingConfig(128,100,512))
    bounds = {name:(value,value) for name,value in vars(p).items() if name != "v0"}
    result = calibrate_american_call_surface(q, replace(p,v0=.05), market=market,
        bounds=bounds, config=PricingConfig(96,80,256), max_nfev=30)
    assert result.optimizer_success
    assert result.price_rmse < .001
    assert result.refined_price_rmse < .001
    assert result.fixed_parameters["hurst"] == .15
    assert result.diagnostics.checks["check"].tolist() == ["quadrature","cutoff_mid","cutoff_far","cutoff_cumulative","riccati","joint"]


@pytest.mark.parametrize("kwargs", [{"spread_refinement_fraction":0}, {"spread_refinement_fraction":True},
                                     {"max_nfev":0}, {"price_scale_floor":0},
                                     {"stability_atol":-1}, {"evaluation_callback":2}])
def test_invalid_calibration_settings(market, kwargs):
    with pytest.raises((ValueError,TypeError)):
        calibrate_american_call_surface(surface(BlackScholesParams(.2),market),
                                        BlackScholesParams(.3),market=market,**kwargs)


def test_positive_correlation_is_not_confused_with_rough_bergomi(market):
    p = RoughHestonParams(.04,.045,1.5,.2,.3,.15)
    value = price_american_calls(p,market=market,strike=100.,maturity=.1,
                                 config=PricingConfig(96,80,256))
    assert math.isfinite(value) and 0 < value < market.spot


def test_final_jacobian_is_physical_with_transformed_optimizer_coordinates(market):
    from scipy.stats import norm
    from rvlab.pricing import OptimizerConfig
    q = surface(BlackScholesParams(.27), market)
    q["bid"] = q.market_price-.01
    q["ask"] = q.market_price+.01
    result = calibrate_american_call_surface(q,BlackScholesParams(.4),market=market,
                                             optimizer=OptimizerConfig(parameterization="asinh"))
    t = q.maturity.to_numpy()
    f = market.spot/market.discount_curve.discount(t)
    d1 = (np.log(f/q.strike.to_numpy())+.5*.27**2*t)/(.27*np.sqrt(t))
    vega = market.spot*norm.pdf(d1)*np.sqrt(t)
    np.testing.assert_allclose(result.final_jacobian[:,0],vega/.01,rtol=2e-6)
    assert result.ready
