"""Synthetic surface fits, quote alignment, and calibration failure semantics."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from rvlab.pricing import (
    BlackScholesParams, HestonParams, PricingConfig, PricingError,
    RoughHestonParams, calibrate_surface, price,
)
from rvlab.pricing import calibration
from rvlab.pricing import diagnostics


def _surface(params, *, config=None, maturities=(0.05, 0.15, 0.4),
             strikes=(90.0, 95.0, 100.0, 105.0, 110.0)):
    strike = np.tile(strikes, len(maturities))
    maturity = np.repeat(maturities, len(strikes))
    is_call = strike >= 100.0
    quotes = pd.DataFrame({"strike": strike, "maturity": maturity, "is_call": is_call})
    quotes["market_price"] = price(
        params, forward=100.0, discount=0.99, strike=strike,
        maturity=maturity, is_call=is_call, config=config,
    )
    return quotes


def test_black_scholes_multimaturity_fit_preserves_rows_and_positional_market_inputs():
    params = BlackScholesParams(0.22)
    quotes = _surface(params).iloc[[7, 1, 12, 6, 13]].copy()
    quotes.index = ["same", "other", "same", "z", "other"]
    forwards = pd.Series([100.0, 101.0, 99.0, 102.0, 100.5], index=[9, 8, 7, 6, 5])
    discounts = np.array([0.99, 0.98, 0.995, 1.001, 0.97])
    quotes["market_price"] = price(
        params, forward=forwards.to_numpy(), discount=discounts,
        strike=quotes.strike.to_numpy(), maturity=quotes.maturity.to_numpy(),
        is_call=quotes.is_call.to_numpy(),
    )
    original = quotes.copy(deep=True)
    result = calibrate_surface(quotes, BlackScholesParams(0.35),
                               forward=forwards, discount=discounts)

    assert result.optimizer_success and result.numerical_stable and result.ready
    assert result.params.volatility == pytest.approx(0.22, abs=1e-10)
    assert result.price_rmse < 1e-9
    assert result.max_price_change == 0.0
    assert result.fitted_quotes.index.equals(quotes.index)
    np.testing.assert_array_equal(result.fitted_quotes.forward, forwards.to_numpy())
    np.testing.assert_array_equal(result.fitted_quotes.discount, discounts)
    np.testing.assert_allclose(result.fitted_quotes.price_scale, 1.0)
    pd.testing.assert_frame_equal(quotes, original)


def test_lower_boundary_initial_volatility_can_leave_the_boundary():
    result = calibrate_surface(_surface(BlackScholesParams(0.2)), BlackScholesParams(0.0),
                               forward=100.0, discount=0.99)
    assert result.ready
    assert result.params.volatility == pytest.approx(0.2, abs=1e-8)
    assert result.price_rmse < 1e-7


def test_spread_weighting_and_zero_spread_floor_determine_the_fitted_price():
    # Identical contracts with inconsistent prices have a closed-form weighted
    # least-squares target, independently of the volatility parametrization.
    quotes = pd.DataFrame({
        "strike": [100.0, 100.0], "maturity": [0.25, 0.25],
        "is_call": [True, True], "bid": [2.0, 2.0], "ask": [2.0, 4.0],
        "market_price": [99.0, 99.0],  # midpoint explicitly takes precedence
    })
    result = calibrate_surface(quotes, BlackScholesParams(0.2),
                               forward=100.0, discount=1.0, price_scale_floor=0.01)
    expected_price = (2.0 / 0.01**2 + 3.0) / (1.0 / 0.01**2 + 1.0)
    assert result.ready
    np.testing.assert_allclose(result.fitted_quotes.model_price, expected_price, atol=1e-8)
    np.testing.assert_allclose(result.fitted_quotes.target_price, [2.0, 3.0])
    np.testing.assert_allclose(result.fitted_quotes.price_scale, [0.01, 1.0])
    np.testing.assert_allclose(result.fitted_quotes.residual,
                               result.fitted_quotes.price_residual / [0.01, 1.0])
    assert result.objective == pytest.approx(np.mean(result.fitted_quotes.residual**2))


def test_bid_ask_quotes_do_not_require_market_price():
    quotes = _surface(BlackScholesParams(0.2))
    quotes["bid"] = np.maximum(quotes.market_price - 0.01, 0.0)
    quotes["ask"] = 2.0 * quotes.market_price - quotes.bid
    quotes = quotes.drop(columns="market_price")
    result = calibrate_surface(quotes, BlackScholesParams(0.3), forward=100.0, discount=0.99)
    assert result.ready
    assert result.params.volatility == pytest.approx(0.2, abs=1e-9)


def test_heston_multimaturity_surface_repricing_recovery():
    params = HestonParams(v0=0.04, theta=0.06, kappa=1.7, vol_of_vol=0.35, rho=-0.6)
    initial = HestonParams(v0=0.05, theta=0.05, kappa=1.3, vol_of_vol=0.3, rho=-0.5)
    config = PricingConfig(quadrature_nodes=96, fourier_cutoff=100, riccati_steps=128)
    quotes = _surface(params, config=config)
    result = calibrate_surface(quotes, initial, forward=100.0, discount=0.99,
                               config=config, max_nfev=80)
    assert result.ready
    assert result.price_rmse < 1e-6
    assert result.max_price_change < 0.001
    assert result.refined_price_rmse < 0.001
    assert result.nfev > 1
    assert set(result.active_bounds).issubset(vars(params))


def test_rough_heston_controlled_fit_with_fixed_parameters():
    params = RoughHestonParams(v0=0.04, theta=0.04, kappa=1.0,
                               vol_of_vol=0.2, rho=-0.6, hurst=0.15)
    initial = replace(params, v0=0.05)
    config = PricingConfig(quadrature_nodes=48, fourier_cutoff=60, riccati_steps=128)
    quotes = _surface(params, config=config, maturities=(0.1, 0.2), strikes=(97., 100., 103.))
    fixed = {name: (value, value) for name, value in vars(params).items() if name != "v0"}
    result = calibrate_surface(quotes, initial, forward=100.0, discount=0.99,
                               config=config, bounds=fixed, max_nfev=20)
    assert result.ready
    assert isinstance(result.params, RoughHestonParams)
    assert result.params.v0 == pytest.approx(0.04, abs=1e-8)
    assert result.price_rmse < 1e-7
    assert dict(result.fixed_parameters) == {name: getattr(params, name) for name in fixed}
    assert result.max_price_change < 0.005
    assert result.refined_config.riccati_steps == 2 * config.riccati_steps


def test_all_parameters_fixed_evaluates_fit_without_optimization():
    params = BlackScholesParams(0.2)
    result = calibrate_surface(_surface(params), params, forward=100.0, discount=0.99,
                               bounds={"volatility": (0.2, 0.2)})
    assert result.ready and result.nfev == 0
    assert result.objective == 0.0
    assert dict(result.fixed_parameters) == {"volatility": 0.2}


def test_optimizer_exhaustion_is_explicit_even_when_prices_are_stable():
    result = calibrate_surface(_surface(BlackScholesParams(0.2)), BlackScholesParams(0.4),
                               forward=100.0, discount=0.99, max_nfev=1)
    assert not result.optimizer_success
    assert result.optimizer_status == 0
    assert result.nfev == 1
    assert result.numerical_stable
    assert not result.ready
    assert result.price_rmse > 0.1


def test_active_search_bound_is_reported_without_claiming_good_fit():
    result = calibrate_surface(_surface(BlackScholesParams(0.4)), BlackScholesParams(0.2),
                               forward=100.0, discount=0.99,
                               bounds={"volatility": (0.05, 0.3)})
    assert result.ready
    assert dict(result.active_bounds) == {"volatility": "upper"}
    assert result.params.volatility == pytest.approx(0.3, abs=1e-9)
    assert result.price_rmse > 0.1


def test_refinement_failure_to_meet_tolerance_keeps_both_objectives(monkeypatch):
    params = BlackScholesParams(0.2)
    quotes = _surface(params)
    config = PricingConfig(quadrature_nodes=32)
    original_price = price

    def shifted_refined_price(params, **kwargs):
        settings = kwargs.pop("config")
        values = original_price(params, **kwargs)
        return values + (0.1 if settings.quadrature_nodes > 32 else 0.0)

    monkeypatch.setattr(calibration, "price", shifted_refined_price)
    monkeypatch.setattr(diagnostics, "price", shifted_refined_price)
    result = calibrate_surface(quotes, params, forward=100.0, discount=0.99,
                               bounds={"volatility": (0.2, 0.2)}, config=config,
                               stability_atol=0.001, stability_rtol=0.0)
    assert result.optimizer_success
    assert not result.numerical_stable and not result.ready
    assert result.objective == 0.0
    assert result.refined_objective == pytest.approx(0.01)
    assert result.max_price_change == pytest.approx(0.1)
    np.testing.assert_allclose(result.fitted_quotes.refinement_difference, 0.1)


def test_optimization_pricing_failures_raise_with_stage_context(monkeypatch):
    params = BlackScholesParams(0.2)
    quotes = _surface(params)
    def failing_price(params, **kwargs):
        raise PricingError("deliberate numerical failure")

    monkeypatch.setattr(calibration, "price", failing_price)
    with pytest.raises(PricingError, match="calibration.*deliberate numerical failure"):
        calibrate_surface(quotes, params, forward=100.0, discount=0.99,
                          config=PricingConfig(quadrature_nodes=32))


def test_failed_postfit_probes_preserve_parameters_and_block_readiness(monkeypatch):
    params = BlackScholesParams(.2)
    quotes = _surface(params)

    def failing_probe(params, **kwargs):
        if kwargs["config"].quadrature_nodes > 32:
            raise PricingError("deliberate numerical failure")
        return price(params, **kwargs)

    monkeypatch.setattr(diagnostics, "price", failing_probe)
    result = calibrate_surface(quotes, params, forward=100., discount=.99,
        bounds={"volatility": (.2, .2)}, config=PricingConfig(32, 80., 128))
    assert result.params == params and result.optimizer_success
    assert not result.numerical_stable and not result.ready
    assert result.fitted_quotes.model_price.notna().all()
    assert result.fitted_quotes.refined_model_price.isna().all()
    assert np.isnan(result.refined_objective)
    assert result.diagnostics.checks.status.eq("numerical_failure").all()
    assert result.diagnostics.checks.error.str.contains("deliberate").all()
    assert result.pricing_evaluations == 5


def test_pricing_evaluation_counts_include_finite_difference_jacobians():
    result = calibrate_surface(_surface(BlackScholesParams(.2)), BlackScholesParams(.3),
        forward=100., discount=.99)
    # nfev omits SciPy's finite-difference Jacobian calls and all post-fit checks.
    assert result.pricing_evaluations == result.nfev + result.njev + 1 + result.diagnostics.pricing_evaluations
    assert np.isfinite(result.optimality)


def test_nonfinite_model_prices_are_never_treated_as_residual_penalties(monkeypatch):
    quotes = _surface(BlackScholesParams(0.2))
    monkeypatch.setattr(calibration, "price", lambda *args, **kwargs: np.full(len(quotes), np.nan))
    with pytest.raises(PricingError, match="calibration pricing produced invalid prices"):
        calibrate_surface(quotes, BlackScholesParams(0.2), forward=100.0, discount=0.99)


@pytest.mark.parametrize("column,value,message", [
    ("strike", np.nan, "strike must be finite"),
    ("strike", True, "strike must contain real numeric"),
    ("market_price", 2 + 0j, "market_price must contain real numeric"),
    ("maturity", 0.0, "maturity must be positive"),
    ("is_call", "C", "is_call must contain booleans"),
    ("market_price", 1000.0, "no-arbitrage bounds"),
])
def test_invalid_quote_identifies_position_without_dropping_it(column, value, message):
    quotes = _surface(BlackScholesParams(0.2))
    quotes[column] = quotes[column].astype(object)
    quotes.iloc[2, quotes.columns.get_loc(column)] = value
    quotes.index = ["duplicate"] * len(quotes)
    with pytest.raises(ValueError, match=message) as error:
        calibrate_surface(quotes, BlackScholesParams(0.2), forward=100.0, discount=0.99)
    assert "row positions [2]" in str(error.value)
    assert "duplicate" in str(error.value)


def test_nullable_missing_quote_is_reported_with_its_row():
    quotes = _surface(BlackScholesParams(0.2))
    quotes["strike"] = quotes.strike.astype("Float64")
    quotes.loc[2, "strike"] = pd.NA
    with pytest.raises(ValueError, match=r"strike must be finite.*row positions \[2\]"):
        calibrate_surface(quotes, BlackScholesParams(0.2), forward=100.0, discount=0.99)


@pytest.mark.parametrize("bounds,message", [
    ({"unknown": (0.0, 1.0)}, "unknown parameter"),
    ({"volatility": (0.3, 0.4)}, "outside bounds"),
    ({"volatility": (0.4, 0.3)}, "lower <= upper"),
    ({"volatility": (-0.1, 1.0)}, "model parameter domains"),
    ({"volatility": (0.0, np.inf)}, "must be finite"),
    ({"volatility": (0.0, 1.0, 2.0)}, "must be a .*pair"),
])
def test_invalid_bounds_are_rejected_before_optimization(bounds, message):
    with pytest.raises(ValueError, match=message):
        calibrate_surface(_surface(BlackScholesParams(0.2)), BlackScholesParams(0.2),
                          forward=100.0, discount=0.99, bounds=bounds)


def test_market_inputs_must_broadcast_to_quote_rows():
    with pytest.raises(ValueError, match="broadcast to 15 quote rows"):
        calibrate_surface(_surface(BlackScholesParams(0.2)), BlackScholesParams(0.2),
                          forward=[100.0, 101.0], discount=0.99)


@pytest.mark.parametrize("bad_forward", [True, 100 + 0j, np.nan, "bad input"])
def test_invalid_market_values_have_row_level_errors(bad_forward):
    with pytest.raises(ValueError, match="forward.*row positions"):
        calibrate_surface(_surface(BlackScholesParams(0.2)), BlackScholesParams(0.2),
                          forward=bad_forward, discount=0.99)


@pytest.mark.parametrize("kwargs", [
    {"price_scale_floor": "0.005"}, {"stability_atol": 0.001j},
    {"stability_rtol": np.nan}, {"price_scale_floor": 0.0},
])
def test_invalid_numeric_controls_raise_readable_validation_errors(kwargs):
    with pytest.raises(ValueError, match="must be finite"):
        calibrate_surface(_surface(BlackScholesParams(0.2)), BlackScholesParams(0.2),
                          forward=100.0, discount=0.99, **kwargs)


def test_partial_or_crossed_quotes_fail_before_optimization():
    quotes = _surface(BlackScholesParams(0.2))
    quotes["bid"] = quotes.market_price
    with pytest.raises(ValueError, match="both bid and ask"):
        calibrate_surface(quotes, BlackScholesParams(0.2), forward=100.0, discount=0.99)
    quotes["ask"] = quotes.bid - 0.01
    with pytest.raises(ValueError, match="0 <= bid <= ask"):
        calibrate_surface(quotes, BlackScholesParams(0.2), forward=100.0, discount=0.99)
