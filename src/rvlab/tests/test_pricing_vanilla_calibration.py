"""Exercise-aware fitting: recovered prices, frozen grids and honest failures."""
import importlib

import numpy as np
import pandas as pd
import pytest

from rvlab.pricing import (
    BlackScholesParams, CashDividend, DiscountCurve, DividendSchedule, FDConfig, HestonParams,
    HestonCurveParams, MeanReversionCurve, OptimizerConfig, PricingError,
    RoughHestonCurveParams, SpotMarket, UnsupportedModelError,
    calibrate_surface, calibrate_vanilla_surface, evaluate_vanilla,
)
from rvlab.pricing.parameters import parameter_values


MARKET = SpotMarket(100., DiscountCurve.flat(.04, 1.), DividendSchedule((), 1.))
COARSE = FDConfig(spot_steps=90, variance_steps=16, time_steps=60)


def _quotes(params=None, config=None):
    params = BlackScholesParams(.22) if params is None else params
    q = pd.DataFrame({"strike": [96., 103., 108., 94.],
                      "maturity": [.15, .3, .3, .15],
                      "is_call": [False, True, False, True]},
                     index=["first", "second", "third", "fourth"])
    q["market_price"] = evaluate_vanilla(params, market=MARKET,
        strike=q.strike, maturity=q.maturity, is_call=q.is_call,
        exercise_style="american", config=config or COARSE.refined()).prices
    q["exercise_style"] = "american"
    return q


def _fit(q, params=None, **kwargs):
    return calibrate_vanilla_surface(q,
        BlackScholesParams(.28) if params is None else params,
        market=MARKET, config=COARSE, **kwargs)


@pytest.mark.parametrize("coordinates", ["physical", "asinh", "log1p"])
def test_finer_synthetic_bs_surface_recovers_prices_and_freezes_mesh(coordinates):
    q = _quotes()
    result = _fit(q, max_nfev=15,
        optimizer=OptimizerConfig(parameterization=coordinates))
    assert result.optimizer_success
    assert result.price_rmse < .006
    assert abs(result.params.volatility-.22) < .001
    assert result.fitted_quotes.index.tolist() == q.index.tolist()
    np.testing.assert_array_equal(result.fitted_quotes.strike, q.strike)
    np.testing.assert_array_equal(result.fitted_quotes.price_scale, 1.)
    assert result.diagnostics["fixed_grid"]
    assert len(result.diagnostics["grid_fingerprints"]) == 1
    assert result.pricing_evaluations > result.nfev
    assert result.elapsed_seconds > 0
    assert result.diagnostics["backend"] == "american_vanilla_fd"
    assert result.optimizer_settings["physical_bounds"] == {"volatility": (0., 5.)}
    assert set(result.diagnostics["checks"].axis) == {"spot", "time", "domain", "joint"}
    assert result.ready == (result.optimizer_success and result.price_stable)
    assert result.hedge_delta_ready == (result.ready and result.delta_stable)


@pytest.mark.parametrize("curve", [False, True])
def test_heston_and_curve_controlled_recovery_use_finer_observations(curve):
    truth = (HestonCurveParams(.04, 1.5, .25, -.6,
                MeanReversionCurve((.2,), (.045, .06))) if curve
             else HestonParams(.04, .045, 1.5, .25, -.6))
    start = (HestonCurveParams(.06, truth.kappa, truth.vol_of_vol, truth.rho, truth.curve)
             if curve else HestonParams(.06, truth.theta, truth.kappa, truth.vol_of_vol, truth.rho))
    config = FDConfig(spot_steps=60, variance_steps=20, time_steps=40)
    q = _quotes(truth, config.refined())
    bounds = {name: (value, value) for name, value in parameter_values(start).items()}
    bounds["v0"] = (.01, .1)
    result = calibrate_vanilla_surface(q, start, market=MARKET,
        config=config, bounds=bounds, max_nfev=15)
    assert result.optimizer_success
    assert result.price_rmse < .035  # coarse finite-difference fit, not parameter identification
    assert result.refined_price_rmse < .04
    assert result.diagnostics["fixed_grid"]
    assert set(result.fixed_parameters) == set(bounds)-{"v0"}


def test_half_spread_normalization_floor_midpoint_and_contract_order():
    q = _quotes(config=COARSE).iloc[[2, 0, 3, 1]].copy()
    target = q.market_price.to_numpy()
    half = np.array([.001, .025, .1, .005])
    q["bid"], q["ask"] = target-half, target+half
    q["market_price"] = 999.  # targets come exclusively from the quotes
    result = _fit(q, BlackScholesParams(.2), bounds={"volatility": (.2, .2)})
    out = result.fitted_quotes
    np.testing.assert_allclose(out.target_price, target)
    np.testing.assert_allclose(out.price_scale, np.maximum(half, .005))
    np.testing.assert_allclose(out.residual, out.price_residual/out.price_scale)
    assert out.index.tolist() == q.index.tolist()
    assert result.nfev == 0 and result.optimizer_success
    assert result.fixed_parameters == {"volatility": .2}


def test_spread_refinement_is_a_separate_guard_even_with_loose_absolute_tolerances():
    q = _quotes(config=COARSE)
    q["bid"], q["ask"] = q.market_price-.005, q.market_price+.005
    result = _fit(q, BlackScholesParams(.22), bounds={"volatility": (.22, .22)},
        stability_atol=1., delta_atol=1., spread_refinement_fraction=1e-5)
    assert result.optimizer_success
    assert result.diagnostics["checks"].price_stable.all()
    assert not result.diagnostics["spread_stable"]
    assert not result.price_stable and not result.ready


def test_optimizer_exhaustion_and_active_bounds_remain_visible():
    exhausted = _fit(_quotes(), max_nfev=1)
    assert not exhausted.optimizer_success and exhausted.optimizer_status == 0
    assert exhausted.nfev == 1 and not exhausted.ready
    assert len(exhausted.fitted_quotes) == 4
    constrained = _fit(_quotes(), BlackScholesParams(.13),
        bounds={"volatility": (.1, .15)}, max_nfev=15)
    assert constrained.optimizer_success
    assert constrained.active_bounds == {"volatility": "upper"}
    assert constrained.price_rmse > .1


def test_pricing_failure_is_identifiable_with_calibration_context(monkeypatch):
    module = importlib.import_module("rvlab.pricing.vanilla_calibration")
    def fail(*args, **kwargs):
        raise PricingError("deliberate finite-difference failure")
    monkeypatch.setattr(module, "evaluate_vanilla", fail)
    with pytest.raises(PricingError, match="vanilla calibration failed.*deliberate"):
        _fit(_quotes())


def test_refinement_failure_preserves_fit_and_does_not_mark_ready(monkeypatch):
    module = importlib.import_module("rvlab.pricing.vanilla")
    q = _quotes(config=COARSE)
    original = module._run
    def fail_refined(params, market, k, t, calls, config, grid, style):
        if config.time_steps > COARSE.time_steps:
            raise PricingError("deliberate refinement failure")
        return original(params, market, k, t, calls, config, grid, style)
    monkeypatch.setattr(module, "_run", fail_refined)
    result = _fit(q, BlackScholesParams(.22), bounds={"volatility": (.22, .22)})
    assert result.optimizer_success and not result.ready
    assert not result.price_stable and not result.delta_stable
    assert np.isfinite(result.fitted_quotes.model_price).all()
    assert result.fitted_quotes.refined_model_price.isna().all()
    failed = result.diagnostics["checks"].query("status == 'failed'")
    assert set(failed.axis) == {"time", "joint"}
    assert failed.error.str.contains("deliberate refinement failure").all()


@pytest.mark.parametrize("column,value,pattern", [
    ("strike", -1., "strike.*first"), ("maturity", 0., "maturity.*first"),
    ("is_call", "put", "is_call.*first"),
    ("market_price", -1., "bounds.*first"),
    ("exercise_style", "european", "exercise style.*first"),
])
def test_malformed_rows_identify_the_contract(column, value, pattern):
    q = _quotes()
    q[column] = q[column].astype(object)
    q.loc["first", column] = value
    with pytest.raises(ValueError, match=pattern):
        _fit(q)


def test_american_labels_are_rejected_by_european_calibrator():
    with pytest.raises(ValueError, match="American|american"):
        calibrate_surface(_quotes(), BlackScholesParams(.2), forward=100., discount=.99)


def test_rough_curve_requests_are_rejected_before_any_optimization():
    params = RoughHestonCurveParams(.04, 1., .2, -.5,
        MeanReversionCurve((), (.04,)), .15)
    with pytest.raises(UnsupportedModelError, match="rough Heston"):
        _fit(_quotes(), params)


def test_deterministic_event_kink_and_mixed_maturity_diagnostics_are_identifiable():
    market = SpotMarket(100., DiscountCurve.flat(0., 1.),
        DividendSchedule((CashDividend(.2, 10.),), 1.))
    result = evaluate_vanilla(BlackScholesParams(0.), market=market,
        strike=[90., 90.], maturity=[.3, .1], is_call=True,
        exercise_style="american", config=COARSE)
    events = result.diagnostics["dividend_event_sensitivities"]
    assert len(events) == 1
    event = events[0]
    assert event["maturity"] == .3 and event["contract_indices"] == [0]
    np.testing.assert_allclose(event["pre_delta"], [1.], atol=1e-8)
    np.testing.assert_allclose(event["post_delta_left"], [0.], atol=1e-8)
    np.testing.assert_allclose(event["post_delta_right"], [1.], atol=1e-8)
    ordinary = evaluate_vanilla(BlackScholesParams(.2), market=market,
        strike=[95., 95.], maturity=[.3, .1], is_call=False,
        exercise_style="american", config=COARSE)
    boundaries = pd.DataFrame(ordinary.exercise_boundaries)
    assert boundaries.groupby("contract_index").maturity.first().to_dict() == {0: .3, 1: .1}


@pytest.mark.parametrize("kwargs", [
    {"price_scale_floor": 0.}, {"spread_refinement_fraction": 0.},
    {"delta_atol": float("nan")}, {"max_nfev": True},
    {"config": "coarse"}, {"optimizer": {}},
])
def test_invalid_calibration_controls(kwargs):
    args = dict(market=MARKET, config=COARSE)
    args.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        calibrate_vanilla_surface(_quotes(), BlackScholesParams(.2), **args)
