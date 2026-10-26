"""Limits, contracts and numerical safeguards for scheduled-variance models."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from scipy.special import erfcx

from rvlab.pricing import (
    BlackScholesParams, HestonParams, PricingConfig, RoughHestonParams,
    black_scholes_price, heston_cf, heston_price, rough_heston_cf,
    rough_heston_price,
)
from rvlab.pricing import variance_curve as module
from rvlab.pricing.variance_curve import (
    HestonCurveParams, MeanReversionCurve, RoughHestonCurveParams,
    forward_variance, heston_curve_cf, heston_curve_price,
    integrated_forward_variance, rough_heston_curve_cf, rough_heston_curve_price,
)


CURVE = MeanReversionCurve((0.04, 0.071), (0.045, 0.08, 0.025))
HESTON = HestonCurveParams(0.035, 2.1, 0.32, -0.65, CURVE)
ROUGH = RoughHestonCurveParams(0.035, 2.1, 0.16, -0.65, CURVE, 0.15)
CONFIG = PricingConfig(128, 120, 512)


@pytest.mark.parametrize("knots,levels", [
    ((0.0,), (.04, .05)), ((-.1,), (.04, .05)),
    ((.1, .1), (.04, .05, .06)), ((.2, .1), (.04, .05, .06)),
    ((np.inf,), (.04, .05)), ((True,), (.04, .05)),
    ((.1,), (.04,)), ((), ()), ((), (-.04,)), ((), (np.nan,)),
    ((), (True,)),
])
def test_curve_rejects_invalid_schedules(knots, levels):
    with pytest.raises(ValueError):
        MeanReversionCurve(knots, levels)


def test_curve_is_immutable_canonical_and_right_continuous():
    curve = MeanReversionCurve([.1, .2], [.03, .05, .07])
    assert curve.knot_times == (.1, .2)
    assert curve.levels == (.03, .05, .07)
    assert curve([0, .0999, .1, .1999, .2, 10]) == pytest.approx(
        [.03, .03, .05, .05, .07, .07])
    assert isinstance(curve(0.1), float)
    with pytest.raises(FrozenInstanceError):
        curve.levels = (.04,)


@pytest.mark.parametrize("field,value", [
    ("v0", -.01), ("kappa", np.nan), ("vol_of_vol", -1),
    ("rho", 1.01), ("rho", True), ("curve", (.04,)),
])
def test_parameter_validation(field, value):
    with pytest.raises((ValueError, TypeError)):
        replace(HESTON, **{field: value})


@pytest.mark.parametrize("hurst", [0, -.01, .51, np.nan, True])
def test_rough_parameter_validation(hurst):
    with pytest.raises(ValueError):
        replace(ROUGH, hurst=hurst)


@pytest.mark.parametrize("rough", [False, True])
def test_constant_schedule_delegates_legacy_exactly(rough):
    curve = MeanReversionCurve((.03, .08), (.045, .045, .045))
    base = RoughHestonParams(.035, .045, 2.1, .16, -.65, .15) if rough else HestonParams(.035, .045, 2.1, .32, -.65)
    params = replace(ROUGH if rough else HESTON, curve=curve)
    u = np.array([0, 1, 7, 20, 100])-.5j
    cf = rough_heston_curve_cf(params, u, .1, config=CONFIG) if rough else heston_curve_cf(params, u, .1)
    expected_cf = rough_heston_cf(base, u, .1, config=CONFIG) if rough else heston_cf(base, u, .1)
    np.testing.assert_array_equal(cf, expected_cf)
    kwargs = dict(forward=100, discount=.99, strike=[90, 100, 110], maturity=[.03, .06, .1])
    actual = (rough_heston_curve_price if rough else heston_curve_price)(params, **kwargs, config=CONFIG)
    expected = (rough_heston_price if rough else heston_price)(base, **kwargs, config=CONFIG)
    np.testing.assert_array_equal(actual, expected)


def test_schedule_changes_after_maturity_cannot_change_prices():
    params = replace(HESTON, curve=MeanReversionCurve((.1,), (.045, .2)))
    u = np.array([0, 1, 10, 90])-.5j
    expected = heston_cf(HestonParams(.035, .045, 2.1, .32, -.65), u, .08)
    np.testing.assert_allclose(heston_curve_cf(params, u, .08), expected, atol=1e-14)
    assert forward_variance(params, .08) == pytest.approx(.045+(.035-.045)*np.exp(-2.1*.08))


def test_hurst_half_delegates_scheduled_heston_exactly():
    params = replace(ROUGH, hurst=.5, vol_of_vol=HESTON.vol_of_vol)
    u = np.array([0, 1, 7, 70])-.5j
    np.testing.assert_array_equal(rough_heston_curve_cf(params, u, .1), heston_curve_cf(HESTON, u, .1))
    kwargs = dict(forward=100, discount=.99, strike=[90, 100, 110], maturity=.1)
    np.testing.assert_array_equal(rough_heston_curve_price(params, **kwargs, config=CONFIG),
                                  heston_curve_price(HESTON, **kwargs, config=CONFIG))
    np.testing.assert_array_equal(forward_variance(params, [0, .03, .1]), forward_variance(HESTON, [0, .03, .1]))


@pytest.mark.parametrize("params,pricer", [(HESTON, heston_curve_price), (ROUGH, rough_heston_curve_price)])
def test_deterministic_schedule_is_exact_lognormal(params, pricer):
    params = replace(params, vol_of_vol=0)
    maturity = .09
    sigma = np.sqrt(integrated_forward_variance(params, maturity)/maturity)
    kwargs = dict(forward=100, discount=.997, strike=[90, 97, 102, 110], maturity=maturity,
                  is_call=[False, False, True, True])
    np.testing.assert_allclose(pricer(params, **kwargs),
                              black_scholes_price(BlackScholesParams(sigma), **kwargs), atol=1e-12)


@pytest.mark.parametrize("params,pricer", [(HESTON, heston_curve_price), (ROUGH, rough_heston_curve_price)])
def test_price_broadcasting_bounds_parity_and_expiration(params, pricer):
    f = np.array([[100], [101]])
    k = np.array([90, 100, 110])
    t = np.array([[0], [.09]])
    calls = pricer(params, forward=f, discount=.99, strike=k, maturity=t, config=CONFIG)
    puts = pricer(params, forward=f, discount=.99, strike=k, maturity=t, is_call=False, config=CONFIG)
    assert calls.shape == (2, 3)
    np.testing.assert_allclose(calls-puts, .99*(f-k), atol=1e-12)
    assert np.all(calls >= .99*np.maximum(f-k, 0)-1e-10)
    assert np.all(calls <= .99*f+1e-10)
    np.testing.assert_array_equal(calls[0], .99*np.maximum(100-k, 0))


@pytest.mark.parametrize("params", [HESTON, ROUGH])
def test_zero_kappa_makes_schedule_irrelevant(params):
    params = replace(params, kappa=0)
    np.testing.assert_array_equal(forward_variance(params, [0, .01, 2]), [params.v0]*3)
    np.testing.assert_allclose(integrated_forward_variance(params, [0, .01, 2]), params.v0*np.array([0, .01, 2]))
    if isinstance(params, RoughHestonCurveParams):
        actual = rough_heston_curve_cf(params, [1, 5], .1, config=CONFIG)
        expected = rough_heston_cf(RoughHestonParams(params.v0, .01, 0, params.vol_of_vol, params.rho, params.hurst), [1, 5], .1, config=CONFIG)
    else:
        actual = heston_curve_cf(params, [1, 5], .1)
        expected = heston_cf(HestonParams(params.v0, .01, 0, params.vol_of_vol, params.rho), [1, 5], .1)
    np.testing.assert_allclose(actual, expected, atol=2e-14)


@pytest.mark.parametrize("params", [HESTON, ROUGH])
def test_later_immigration_is_not_mistaken_for_absorption(params):
    params = replace(params, v0=0, curve=MeanReversionCurve((.04,), (0, .08)))
    assert forward_variance(params, .03) == 0
    assert integrated_forward_variance(params, .03) == 0
    assert forward_variance(params, .08) > 0
    assert integrated_forward_variance(params, .08) > 0
    cf = rough_heston_curve_cf(params, 1, .08, config=CONFIG) if isinstance(params, RoughHestonCurveParams) else heston_curve_cf(params, 1, .08)
    assert abs(cf) < 1


def test_piecewise_linear_integral_splits_partial_cells_exactly():
    times = np.linspace(0, 1, 8)
    values = np.column_stack([2+3*times, 1j*(1-times)])
    points = np.array([0, .037, .2, .4999, .9, 1])
    actual = module._linear_primitive(values, 1, points)
    expected = np.column_stack([2*points+1.5*points**2, 1j*(points-.5*points**2)])
    np.testing.assert_allclose(actual, expected, atol=2e-15)


def test_rough_terminal_basis_cache_excludes_v0_and_curve_levels(monkeypatch):
    module._rough_basis.cache_clear()
    original = module.solve_fractional_riccati
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "solve_fractional_riccati", counted)
    first = rough_heston_curve_cf(ROUGH, [1, 5], .1, config=CONFIG)
    shifted = replace(ROUGH, v0=.04, curve=replace(CURVE, levels=(.05, .09, .03)))
    second = rough_heston_curve_cf(shifted, [1, 5], .1, config=CONFIG)
    assert len(calls) == 1
    assert np.max(abs(first-second)) > 1e-4
    rough_heston_curve_cf(replace(ROUGH, rho=-.5), [1, 5], .1, config=CONFIG)
    assert len(calls) == 2


@pytest.mark.parametrize("x", [1e-14, .001, .75, 1, 3, 20, 100])
def test_positive_relaxation_mixture_matches_exact_alpha_half(x):
    relaxation, j, q = module._relaxation_moments(.5, x)
    assert relaxation == pytest.approx(erfcx(x), rel=2e-10, abs=2e-13)
    assert j+q == pytest.approx(1, abs=1e-12)
    assert 0 <= j <= 1
    assert 0 <= q <= 1


@pytest.mark.parametrize("params", [HESTON, ROUGH])
def test_forward_means_nonnegative_and_continuous_at_knots(params):
    times = np.r_[np.linspace(0, .3, 21), .04-1e-11, .04+1e-11]
    assert np.all(forward_variance(params, times) >= 0)
    assert np.all(integrated_forward_variance(params, times) >= 0)
    assert forward_variance(params, .04-1e-11) == pytest.approx(forward_variance(params, .04+1e-11), abs=2e-8)


@pytest.mark.parametrize("bad", [-1, np.nan, np.inf])
def test_invalid_maturity_rejected(bad):
    with pytest.raises(ValueError):
        forward_variance(ROUGH, bad)
    with pytest.raises(ValueError):
        rough_heston_curve_cf(ROUGH, 1, bad)


def test_rough_curve_time_refinement_with_non_grid_aligned_knots():
    u = np.array([1, 5, 20])-.5j
    coarse = rough_heston_curve_cf(ROUGH, u, .09, config=replace(CONFIG, riccati_steps=128))
    fine = rough_heston_curve_cf(ROUGH, u, .09, config=replace(CONFIG, riccati_steps=512))
    finer = rough_heston_curve_cf(ROUGH, u, .09, config=replace(CONFIG, riccati_steps=1024))
    assert np.max(abs(finer-fine)) < np.max(abs(fine-coarse))
    assert np.max(abs(finer-fine)) < 2e-5


@pytest.mark.parametrize("params,pricer", [(HESTON, heston_curve_price), (ROUGH, rough_heston_curve_price)])
def test_zero_variance_with_nonzero_diffusion_is_absorbing(params, pricer):
    params = replace(params, v0=0, curve=MeanReversionCurve((.02,), (0, 0)))
    np.testing.assert_array_equal(pricer(params, forward=100, discount=.99,
                                        strike=[90, 100, 110], maturity=.1),
                                  [.99*10, 0, 0])


def test_negative_immigration_weights_raise_instead_of_being_zeroed(monkeypatch):
    monkeypatch.setattr(module, "_relaxation_moments", lambda alpha, x: (x, .5, .5))
    with pytest.raises(module.PricingError, match="negative immigration weights"):
        forward_variance(HESTON, .1)


def test_heston_curve_extreme_coefficients_raise_identifiable_failure():
    with pytest.raises(module.PricingError, match="coefficients exceed"):
        heston_curve_cf(replace(HESTON, vol_of_vol=1e200), 1, .1)


def test_constant_rough_deterministic_limit_uses_exact_curve_integral():
    # Legacy deterministic rough pricing numerically integrates Riccati. The
    # new API instead uses the exact expected integral when eta is zero.
    params = replace(ROUGH, vol_of_vol=0, curve=MeanReversionCurve((), (.08,)))
    maturity = .1
    variance = integrated_forward_variance(params, maturity)
    assert rough_heston_curve_cf(params, 2-.5j, maturity, config=PricingConfig(16, 20, 2)) == pytest.approx(
        np.exp(-.5*((2-.5j)**2+1j*(2-.5j))*variance), abs=1e-14)
