"""Reparameterization preserves the physical calibration problem and diagnostics."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rvlab.pricing import (
    BlackScholesParams, OptimizerConfig, PricingConfig, calibrate_surface,
    calibrate_surface_refined, price,
)
from rvlab.pricing.optimization import _Coordinates
from rvlab.experiments._pricing_fits import fit_attempt


@pytest.mark.parametrize("kind", ["physical", "asinh", "log1p"])
def test_coordinate_map_preserves_endpoints_zero_and_signed_correlation(kind):
    names = ("v0", "kappa", "rho", "theta_0", "theta_1", "hurst")
    lo = dict(zip(names, [0., 0., -.95, 0., 0., .05]))
    hi = dict(zip(names, [.25, 80., 0., 2., 2., .5]))
    mapping = _Coordinates(names, lo, hi, OptimizerConfig(parameterization=kind))
    for physical in (mapping.lower, mapping.upper,
                     np.array([.02, .21, -.65, .7, 2.5e-35, .31])):
        np.testing.assert_allclose(mapping.decode(mapping.encode(physical)), physical,
                                   rtol=3e-15, atol=1e-16)
    assert mapping.encode(np.array([0.,0.,-.65,0.,0.,.15]))[2] == -.65
    assert np.all(mapping.derivative(mapping.encode(mapping.lower)) > 0)


@pytest.mark.parametrize("kind", ["physical", "asinh", "log1p"])
def test_physical_optimality_uses_chain_rule_and_bound_direction(kind):
    names = ("v0", "kappa", "theta_0")
    lo, hi = dict(zip(names, [0.,0.,0.])), dict(zip(names, [1.,20.,2.]))
    mapping = _Coordinates(names, lo, hi, OptimizerConfig(parameterization=kind))
    physical = np.array([.2, .4, 0.])
    z = mapping.encode(physical)
    # Residual derivative is the identity in physical coordinates.
    jacobian = np.diag(mapping.derivative(z))
    gradient = np.array([-2., 3., 4.])
    expected = max(2.*(1.-.2), 3.*.4, 4.*0.)
    assert mapping.physical_optimality(z, jacobian, gradient) == pytest.approx(expected)


@pytest.mark.parametrize("kwargs", [
    {"parameterization": "immigration"}, {"jacobian": "cs"}, {"jacobian": None},
    {"ftol": 0.}, {"xtol": -1.}, {"gtol": True}, {"xtol": np.nan},
    {"ftol": None, "xtol": None, "gtol": None},
])
def test_invalid_optimizer_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        OptimizerConfig(**kwargs)


def test_optimizer_settings_immutable():
    config = OptimizerConfig(parameterization="asinh")
    with pytest.raises(FrozenInstanceError):
        config.xtol = 1e-3


def _surface(sigma):
    quotes = pd.DataFrame({"strike": [95.,100.,105.,95.,100.,105.],
        "maturity": [.04]*3+[.08]*3, "is_call": [False,True,True]*2,
        "forward": 100., "discount": .998})
    quotes["market_price"] = price(BlackScholesParams(sigma), forward=quotes.forward,
        discount=quotes.discount, strike=quotes.strike, maturity=quotes.maturity,
        is_call=quotes.is_call)
    return quotes


@pytest.mark.parametrize("kind", ["physical", "asinh", "log1p"])
@pytest.mark.parametrize("jacobian", ["2-point", "3-point"])
def test_transformed_calibration_recovers_prices_and_records_controls(kind, jacobian):
    q = _surface(.21)
    controls = OptimizerConfig(parameterization=kind, jacobian=jacobian)
    result = calibrate_surface(q, BlackScholesParams(.31), forward=q.forward,
        discount=q.discount, bounds={"volatility": (0., .5)}, optimizer=controls)
    assert result.ready
    assert result.price_rmse < 1e-8
    assert result.params.volatility == pytest.approx(.21, abs=1e-8)
    assert result.optimizer_settings["parameterization"] == kind
    assert result.optimizer_settings["jacobian"] == jacobian
    assert result.physical_optimality < 1e-5
    if kind == "physical":
        assert result.physical_optimality == pytest.approx(result.optimality, abs=1e-12)


@pytest.mark.parametrize("kind", ["asinh", "log1p"])
def test_transformed_search_keeps_the_original_physical_box(monkeypatch, kind):
    from rvlab.pricing import calibration
    q = _surface(.4)
    evaluate = calibration.price
    seen = []

    def observe(params, **kwargs):
        seen.append(params.volatility)
        assert .1 <= params.volatility <= .3
        return evaluate(params, **kwargs)

    monkeypatch.setattr(calibration, "price", observe)
    result = calibration.calibrate_surface(q, BlackScholesParams(.11), forward=q.forward,
        discount=q.discount, bounds={"volatility": (.1,.3)},
        optimizer=OptimizerConfig(parameterization=kind))
    assert seen
    assert result.active_bounds == {"volatility": "upper"}
    assert result.params.volatility == pytest.approx(.3)
    assert result.physical_optimality < 1e-5


def test_settings_propagate_through_fit_and_refinement():
    q = _surface(.21)
    controls = OptimizerConfig(parameterization="asinh", jacobian="3-point")
    attempt = fit_attempt("settings", q, BlackScholesParams(.3), optimizer=controls)
    assert attempt.summary["ready"]
    assert attempt.summary["optimizer_settings"]["parameterization"] == "asinh"
    assert attempt.summary["physical_optimality"] is not None
    study = calibrate_surface_refined(q, BlackScholesParams(.3), forward=q.forward,
        discount=q.discount, configs=[PricingConfig()], optimizer=controls)
    assert study.ready
    assert study.result.optimizer_settings["jacobian"] == "3-point"


def test_all_fixed_transformed_parameters_are_not_nudged():
    q = _surface(0.)
    result = calibrate_surface(q, BlackScholesParams(0.), forward=q.forward,
        discount=q.discount, bounds={"volatility": (0.,0.)},
        optimizer=OptimizerConfig(parameterization="asinh"))
    assert result.nfev == 0
    assert result.params.volatility == 0.
    assert result.physical_optimality == 0.


def test_bad_optimizer_type_rejected():
    q = _surface(.2)
    with pytest.raises(TypeError, match="OptimizerConfig"):
        calibrate_surface(q, BlackScholesParams(.3), forward=q.forward,
                          discount=q.discount, optimizer={"ftol": 1e-3})


@pytest.mark.parametrize("kind", ["asinh", "log1p"])
def test_transformed_jacobian_keeps_physical_step_floor_at_zero(monkeypatch, kind):
    from rvlab.pricing import calibration
    q = _surface(.2)
    original_price = calibration.price
    seen = []

    def observe(params, **kwargs):
        seen.append(params.volatility)
        return original_price(params, **kwargs)

    def inspect(fun, x0, *, jac, **kwargs):
        assert callable(jac)
        residual = fun(x0)
        derivative = jac(x0)
        # At-the-money BS vega at zero is D*F*sqrt(T)/sqrt(2*pi).
        atm = q.strike.to_numpy() == 100.
        expected = .998*100*np.sqrt(q.loc[atm, "maturity"])/(np.sqrt(2*np.pi))*.01
        np.testing.assert_allclose(derivative[atm,0], expected, rtol=3e-6)
        return SimpleNamespace(x=x0, fun=residual, jac=derivative, success=False,
            status=0, message="inspected", nfev=1, njev=1,
            optimality=1., active_mask=np.zeros(1))

    monkeypatch.setattr(calibration, "price", observe)
    monkeypatch.setattr(calibration, "least_squares", inspect)
    result = calibration.calibrate_surface(q, BlackScholesParams(1e-35),
        forward=q.forward, discount=q.discount, bounds={"volatility": (0.,.5)},
        optimizer=OptimizerConfig(parameterization=kind))
    assert any(1e-8 < value < 2e-8 for value in seen)
    assert result.optimizer_settings["finite_difference_space"] == "physical"
