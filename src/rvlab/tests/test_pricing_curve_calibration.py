"""Curve calibration uses the existing contract-aligned least-squares workflow."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from rvlab.pricing import (
    HestonCurveParams, RoughHestonCurveParams, MeanReversionCurve, PricingConfig,
    calibrate_surface, parameter_values, price, pricing_diagnostics, replace_parameters,
)
from rvlab.experiments._pricing_fits import fit_attempt


def _model(rough=False):
    kwargs = dict(v0=.035, kappa=2., vol_of_vol=.15, rho=-.6,
                  curve=MeanReversionCurve((.04, .08), (.025, .065, .035)))
    return RoughHestonCurveParams(**kwargs, hurst=.2) if rough else HestonCurveParams(**kwargs)


def _quotes(params, config):
    t = np.repeat([.04, .08, .12], 4)
    strike = np.tile([94., 98., 102., 106.], 3)
    q = pd.DataFrame(dict(strike=strike, maturity=t, is_call=strike > 100.,
                          forward=100., discount=np.exp(-.03*t)))
    q["market_price"] = price(params, forward=q.forward, discount=q.discount,
                              strike=q.strike, maturity=q.maturity,
                              is_call=q.is_call, config=config)
    q.index = [2, 2, 5, 1]*3
    return q


@pytest.mark.parametrize("rough", [False, True])
def test_curve_calibration_recovers_multi_maturity_prices(rough):
    truth = _model(rough)
    config = PricingConfig(96, 160, 128)
    q = _quotes(truth, PricingConfig(160, 320, 256))
    initial = replace(truth, curve=MeanReversionCurve(truth.curve.knot_times, (.04,)*3))
    bounds = {name: (value, value) for name, value in parameter_values(initial).items()}
    bounds.update({f"theta_{i}": (.001, .15) for i in range(3)})
    result = calibrate_surface(q, initial, forward=q.forward, discount=q.discount,
                               bounds=bounds, config=config, max_nfev=30)
    assert result.optimizer_success
    assert result.price_rmse < .0005
    assert result.fitted_quotes.index.tolist() == q.index.tolist()
    assert result.params.curve.knot_times == truth.curve.knot_times
    assert result.fixed_parameters["kappa"] == 2.
    assert "curve" not in result.fixed_parameters
    assert set(result.active_bounds) <= {"theta_0", "theta_1", "theta_2"}


def test_scalar_parameter_replacement_preserves_knots_and_original():
    original = _model(True)
    changed = replace_parameters(original, theta_1=.09, v0=.02, hurst=.3)
    assert changed.curve.knot_times == original.curve.knot_times
    assert changed.curve.levels == (.025, .09, .035)
    assert original.curve.levels == (.025, .065, .035)
    assert changed.hurst == .3
    with pytest.raises(ValueError, match="unknown model parameters"):
        replace_parameters(original, knot_times=(.1, .2))
    with pytest.raises(ValueError):
        replace_parameters(original, theta_0=-.01)


def test_curve_calibration_fixed_coordinates_and_wrong_bounds():
    params = _model()
    config = PricingConfig(64, 100, 32)
    q = _quotes(params, config)
    fixed = {name: (value, value) for name, value in parameter_values(params).items()}
    result = calibrate_surface(q, params, forward=q.forward, discount=q.discount,
                               bounds=fixed, config=config)
    assert result.nfev == 0
    assert result.objective == 0.
    assert result.params == params
    for bounds, text in [({"theta": (.01, .1)}, "unknown parameter bounds"),
                         ({"theta_0": (-.01, .1)}, "model parameter domains"),
                         ({"theta_0": (.1, .2)}, "outside bounds")]:
        with pytest.raises(ValueError, match=text):
            calibrate_surface(q, params, forward=q.forward, discount=q.discount, bounds=bounds)


def test_curve_diagnostics_includes_fractional_time_resolution():
    params = _model(True)
    result = pricing_diagnostics(params, forward=100., discount=.99, strike=[98.,102.],
                                 maturity=.12, config=PricingConfig(32,60,32))
    assert "riccati" in result.checks.check.values
    delegated = pricing_diagnostics(replace(params, hurst=.5), forward=100., discount=.99,
                                    strike=100., maturity=.12, config=PricingConfig(32,60,32))
    assert "riccati" not in delegated.checks.check.values


def test_curve_fit_attempt_retains_exhaustion_and_flat_coordinate_names():
    truth = _model()
    config = PricingConfig(64,100,32)
    q = _quotes(truth, config)
    initial = replace(truth, curve=MeanReversionCurve((.04,.08), (.04,)*3))
    bounds = {name: (value,value) for name,value in parameter_values(initial).items()}
    bounds.update({f"theta_{i}": (.001,.15) for i in range(3)})
    attempt = fit_attempt("curve exhausted", q, initial, bounds=bounds,
                          config=config, max_nfev=1)
    assert attempt.result is not None
    assert not attempt.summary["optimizer_success"]
    assert not attempt.summary["ready"]
    assert attempt.summary["free_parameter_count"] == 3
    assert attempt.summary["free_parameters"] == ["theta_0", "theta_1", "theta_2"]
    assert attempt.summary["params"]["curve"]["knot_times"] == (.04,.08)
