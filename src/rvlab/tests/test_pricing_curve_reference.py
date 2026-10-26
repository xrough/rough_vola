"""Independent references for the time-dependent variance immigration model.

The ODE below integrates calendar-time immigration backwards through the
Riccati time-to-expiry variable.  It deliberately does not reuse the analytic
Heston exponent or the curve model's numerical integration helpers.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp

from rvlab.pricing.heston import heston_cf
from rvlab.pricing.rough_heston import solve_fractional_riccati
from rvlab.pricing.types import HestonParams, PricingConfig, RoughHestonParams
from rvlab.pricing.variance_curve import (
    HestonCurveParams,
    MeanReversionCurve,
    RoughHestonCurveParams,
    forward_variance,
    heston_curve_cf,
    integrated_forward_variance,
    rough_heston_curve_cf,
)


def _curve_params(*, rough=False, **overrides):
    values = dict(
        v0=0.019, kappa=4.2, vol_of_vol=0.7, rho=-0.65,
        curve=MeanReversionCurve((0.017, 0.044), (0.03, 0.09, 0.012)),
    )
    if rough:
        values["hurst"] = 0.15
    values.update(overrides)
    cls = RoughHestonCurveParams if rough else HestonCurveParams
    return cls(**values)


def _theta(curve, time):
    return curve.levels[np.searchsorted(curve.knot_times, time, side="right")]


def _ode_cf(params, u, maturity, *, reverse_calendar=True):
    """DOP853 reference, splitting exactly at every immigration discontinuity."""
    if maturity == 0:
        return 1.0 + 0.0j
    knots = np.asarray(params.curve.knot_times)
    knots = knots[(knots > 0) & (knots < maturity)]
    boundaries = np.sort(np.r_[0.0, maturity - knots if reverse_calendar else knots,
                               maturity])
    state = np.zeros(2, dtype=complex)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        riccati_time = 0.5 * (left + right)
        calendar_time = maturity - riccati_time if reverse_calendar else riccati_time
        immigration = params.kappa * _theta(params.curve, calendar_time)

        def rhs(_, y):
            h = y[0]
            forcing = (-0.5 * (u*u + 1j*u)
                       + (1j*u*params.rho*params.vol_of_vol - params.kappa)*h
                       + 0.5*params.vol_of_vol**2*h*h)
            return np.array([forcing, immigration*h])

        solution = solve_ivp(rhs, (left, right), state, method="DOP853",
                             rtol=2e-12, atol=2e-14)
        assert solution.success
        state = solution.y[:, -1]
    return np.exp(params.v0*state[0] + state[1])


@pytest.mark.parametrize("maturity", [0.01, 0.035, 0.083, 0.25])
def test_piecewise_heston_cf_against_independent_calendar_ode(maturity):
    params = _curve_params()
    frequencies = np.array([0.0, -1j, 0.25, 3.7, 18.0-0.5j])
    expected = np.array([_ode_cf(params, u, maturity) for u in frequencies])
    np.testing.assert_allclose(heston_curve_cf(params, frequencies, maturity),
                               expected, atol=3e-11, rtol=3e-11)


def test_piecewise_heston_reference_detects_reversed_calendar_forcing():
    params, u, maturity = _curve_params(), 18.0-0.5j, 0.083
    correct = _ode_cf(params, u, maturity)
    naive = _ode_cf(params, u, maturity, reverse_calendar=False)
    assert abs(correct - naive) > 1e-4
    assert abs(heston_curve_cf(params, u, maturity) - correct) < 3e-11


def _heston_mean(params, time):
    """Piecewise exact ODE propagation, independent of Mittag-Leffler formulas."""
    value, left = params.v0, 0.0
    boundaries = [*params.curve.knot_times, math.inf]
    for right, level in zip(boundaries, params.curve.levels):
        end = min(time, right)
        value = level + (value-level)*math.exp(-params.kappa*(end-left))
        if time <= right:
            return value
        left = right
    raise AssertionError("unreachable")


def test_heston_cf_also_matches_direct_forward_variance_representation():
    params, u, maturity = _curve_params(), 9.0-0.5j, 0.083
    c = -0.5*(u*u + 1j*u)

    def rhs(_, y):
        return [c + (1j*u*params.rho*params.vol_of_vol-params.kappa)*y[0]
                + 0.5*params.vol_of_vol**2*y[0]**2]

    solution = solve_ivp(rhs, (0, maturity), [0j], method="DOP853",
                         rtol=2e-12, atol=2e-14, dense_output=True)
    assert solution.success

    def integrand(s):
        h = solution.sol(maturity-s)[0]
        # The -kappa*h term stays in Riccati, but cancels in this integrand.
        return (_heston_mean(params, s)
                * (c + 1j*u*params.rho*params.vol_of_vol*h
                   + 0.5*params.vol_of_vol**2*h*h))

    points = [t for t in params.curve.knot_times if t < maturity]
    exponent = quad(lambda s: integrand(s).real, 0, maturity, points=points,
                    epsabs=1e-12, epsrel=1e-12)[0]
    exponent += 1j*quad(lambda s: integrand(s).imag, 0, maturity, points=points,
                       epsabs=1e-12, epsrel=1e-12)[0]
    np.testing.assert_allclose(heston_curve_cf(params, u, maturity),
                               np.exp(exponent), rtol=3e-11, atol=3e-11)


def _mittag_leffler_series(alpha, beta, z):
    """Independent convergent defining series; tests keep |z| below one."""
    terms = []
    for j in range(150):
        term = z**j / math.gamma(alpha*j+beta)
        terms.append(term)
        if j > 8 and abs(term) < 1e-18:
            return math.fsum(terms)
    raise AssertionError("reference series failed to converge")


def _fractional_mean_reference(params, time, *, integrated=False):
    alpha = params.hurst + 0.5
    if time == 0:
        return 0.0 if integrated else params.v0
    beta = 2.0 if integrated else 1.0
    power = 1.0 if integrated else 0.0
    value = params.v0*time**power*_mittag_leffler_series(
        alpha, beta, -params.kappa*time**alpha,
    )
    differences = np.diff(np.r_[0.0, params.curve.levels])
    for knot, change in zip((0.0, *params.curve.knot_times), differences):
        lag = time-knot
        if lag > 0:
            # These forms avoid subtracting E_alpha from one at small kappa.
            value += (change*params.kappa*lag**(alpha+power)
                      * _mittag_leffler_series(
                          alpha, alpha+beta, -params.kappa*lag**alpha,
                      ))
    return value


@pytest.mark.parametrize("hurst", [0.10, 0.35, 0.49, 0.499999])
@pytest.mark.parametrize("kappa", [1e-10, 0.7])
def test_rough_forward_means_against_independent_mittag_leffler_series(hurst, kappa):
    params = _curve_params(rough=True, hurst=hurst, kappa=kappa)
    maturities = np.array([0.0, 0.017, 0.031, 0.044, 0.083, 0.6])
    means = [_fractional_mean_reference(params, t) for t in maturities]
    integrated = [_fractional_mean_reference(params, t, integrated=True)
                  for t in maturities]
    np.testing.assert_allclose(forward_variance(params, maturities), means,
                               rtol=2e-10, atol=2e-12)
    np.testing.assert_allclose(integrated_forward_variance(params, maturities),
                               integrated, rtol=2e-10, atol=2e-12)


@pytest.mark.parametrize("hurst", [0.1, 0.35, 0.49])
def test_rough_deterministic_cf_uses_same_integrated_expected_variance(hurst):
    params = _curve_params(rough=True, hurst=hurst, kappa=0.7, vol_of_vol=0.0)
    maturity = 0.083
    frequencies = np.array([0.0, -1j, 0.3, 7.0-0.5j])
    integrated = _fractional_mean_reference(params, maturity, integrated=True)
    expected = np.exp(-0.5*(frequencies**2+1j*frequencies)*integrated)
    np.testing.assert_allclose(rough_heston_curve_cf(params, frequencies, maturity),
                               expected, rtol=2e-10, atol=2e-12)


@pytest.mark.parametrize("hurst", [0.1, 0.35])
def test_rough_cf_first_log_moment_agrees_with_expected_integrated_variance(hurst):
    params = _curve_params(rough=True, hurst=hurst, kappa=0.7)
    maturity, frequency = 0.083, 2e-5
    expected = _fractional_mean_reference(params, maturity, integrated=True)
    characteristic = rough_heston_curve_cf(
        params, frequency, maturity, config=PricingConfig(64, 100.0, 2048),
    )
    log_moment = np.log(characteristic).imag/frequency
    # E[log(S_T/F_T)] = -1/2 E[integral V_s ds]. The tolerance includes
    # independent fractional time discretisation, not just floating-point error.
    assert log_moment == pytest.approx(-0.5*expected, rel=2e-5, abs=2e-9)


def _original_pollard_moments(alpha, x):
    """Original spectral density on log-frequency, independently integrated.

    This does not use the production inverse-CDF transformation. Splits around
    zero resolve the density's narrow peak as alpha approaches one.
    """
    angle = math.pi*alpha
    width = math.sin(angle)
    edges = sorted({-50.0, -10.0, -1.0, -0.01, -10*width, -width,
                    0.0, width, 10*width, 0.01, 1.0, 10.0, 50.0})
    log_x = math.log(x)

    def integrand(log_frequency, moment):
        density = (math.sin(angle)
                   / (4*math.pi*alpha
                      * (math.sinh(log_frequency/2)**2 + math.cos(angle/2)**2)))
        log_argument = (log_x+log_frequency)/alpha
        if log_argument > 7:
            response = 0.0 if moment == 0 else math.exp(-log_argument)
        else:
            argument = math.exp(log_argument)
            response = (math.exp(-argument) if moment == 0
                        else -math.expm1(-argument)/argument)
        return density*response

    return tuple(math.fsum(quad(integrand, left, right, args=(moment,),
                                epsabs=2e-13, epsrel=2e-11, limit=300)[0]
                           for left, right in zip(edges[:-1], edges[1:]))
                 for moment in (0, 1))


@pytest.mark.parametrize("hurst,kappa,maturity", [
    (0.05, 20.0, 2.0), (0.35, 2.0, 0.6),
    (0.499999, 2.0, 0.6), (0.499999, 20.0, 0.6),
])
def test_rough_means_in_spectral_region_against_original_pollard_density(
        hurst, kappa, maturity):
    params = _curve_params(rough=True, hurst=hurst, kappa=kappa)
    alpha = hurst+0.5

    def moments(time):
        if time == 0:
            return 1.0, 1.0
        return _original_pollard_moments(alpha, kappa*time**alpha)

    r, j = moments(maturity)
    expected_mean, expected_integral = params.v0*r, params.v0*maturity*j
    for knot, change in zip((0.0, *params.curve.knot_times),
                            np.diff(np.r_[0.0, params.curve.levels])):
        lag = maturity-knot
        rr, jj = moments(lag)
        expected_mean += change*(1-rr)
        expected_integral += change*lag*(1-jj)
    np.testing.assert_allclose(forward_variance(params, maturity), expected_mean,
                               rtol=5e-9, atol=2e-11)
    np.testing.assert_allclose(integrated_forward_variance(params, maturity),
                               expected_integral, rtol=5e-9, atol=2e-11)


def test_rough_cf_matches_independent_direct_forward_variance_representation():
    params = _curve_params(rough=True, kappa=0.7)
    u, maturity, steps = 4.0-0.5j, 0.083, 2048
    legacy = RoughHestonParams(params.v0, 0.0, params.kappa,
                               params.vol_of_vol, params.rho, params.hurst)
    times, values = solve_fractional_riccati(legacy, np.array([u]), maturity,
                                           n_steps=steps)
    # Independent quadrature splits all interpolation cells and curve knots.
    boundaries = np.unique(np.r_[times, params.curve.knot_times])
    boundaries = boundaries[(boundaries >= 0) & (boundaries <= maturity)]
    nodes, weights = np.polynomial.legendre.leggauss(8)
    c = -0.5*(u*u+1j*u)
    terms = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        calendar_times = (left+right)/2 + (right-left)/2*nodes
        riccati_times = maturity-calendar_times
        h = (np.interp(riccati_times, times, values[:, 0].real)
             + 1j*np.interp(riccati_times, times, values[:, 0].imag))
        xi = np.array([_fractional_mean_reference(params, t) for t in calendar_times])
        integrand = xi*(c + 1j*u*params.rho*params.vol_of_vol*h
                        + 0.5*params.vol_of_vol**2*h*h)
        terms.append((right-left)/2*np.dot(weights, integrand))
    expected = np.exp(sum(terms))
    actual = rough_heston_curve_cf(params, u, maturity,
                                   config=PricingConfig(64, 100.0, steps))
    # Two representations have distinct fractional integration errors. Their
    # agreement is tested at a resolution finer than the notebook defaults.
    np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=3e-9)


def test_heston_tiny_time_and_vol_of_vol_retain_the_deterministic_limit():
    params = _curve_params(v0=0.04, kappa=1.0, vol_of_vol=1e-8, rho=-0.6,
                          curve=MeanReversionCurve((0.1,), (0.04, 0.03)))
    u, maturity = 1.0, 1e-10
    expected = np.exp(-0.5*(u*u+1j*u)*params.v0*maturity)
    np.testing.assert_allclose(heston_curve_cf(params, u, maturity), expected,
                               rtol=0, atol=3e-14)
    legacy = HestonParams(params.v0, 0.04, params.kappa, params.vol_of_vol, params.rho)
    np.testing.assert_allclose(heston_cf(legacy, u, maturity), expected,
                               rtol=0, atol=3e-14)
