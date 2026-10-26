"""Independent controls for the finite-lift rough-Heston dynamics foundation."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp
from scipy.special import gamma

from rvlab.pricing import (
    BlackScholesParams, HestonParams, HestonCurveParams, MeanReversionCurve,
    PricingConfig, PricingError, RoughHestonParams, RoughHestonCurveParams,
    black_scholes_price, heston_cf, heston_curve_cf, heston_price,
    forward_variance, integrated_forward_variance, rough_heston_price,
)
from rvlab.pricing.rough_lift import (
    ExponentialKernel, approximate_fractional_kernel, lifted_forward_variance,
    lifted_heston_cf, lifted_heston_price, lifted_integrated_variance,
    _solve_lifted_riccati,
)


PARAMS = RoughHestonParams(.04, .065, 1.5, .3, -.6, .15)


def test_kernel_is_immutable_positive_and_fingerprinted():
    kernel = ExponentialKernel([0., 2.], [1., 3.])
    assert kernel.nodes == (0., 2.)
    assert kernel.weights == (1., 3.)
    assert kernel(0) == 4.
    assert len(kernel.fingerprint) == 64
    assert kernel.fingerprint == ExponentialKernel((0., 2.), (1., 3.)).fingerprint
    assert kernel.fingerprint != ExponentialKernel((0., 2.), (1., 3.), horizon=1).fingerprint
    with pytest.raises(FrozenInstanceError):
        kernel.nodes = (1.,)


@pytest.mark.parametrize("nodes,weights", [([], []), ([1], []), ([2, 1], [1, 1]),
                                           ([1, 1], [1, 1]), ([-1], [1]),
                                           ([np.nan], [1]), ([1], [0]),
                                           ([1], [np.inf]), ([True], [1])])
def test_invalid_kernel_coefficients(nodes, weights):
    with pytest.raises(ValueError):
        ExponentialKernel(nodes, weights)


def test_exact_constant_forcing_and_zero_node_primitive():
    kernel = ExponentialKernel((0., .2, 30.), (.5, 1.2, 2.))
    times = np.array([[0., 1e-10], [.2, 1.]])
    exact = .5*times+1.2*(-np.expm1(-.2*times))/.2+2*(-np.expm1(-30*times))/30
    assert_allclose(kernel.integrated(times), exact, rtol=2e-15, atol=1e-16)
    # Each component solves y_i'=-x_i*y_i+c, y_i(0)=0. A nonzero c
    # must create an immediate state despite the zero initial condition.
    forcing = 3.2
    expected = forcing*kernel.integrated(.2)
    solution = solve_ivp(lambda t, y: -np.array(kernel.nodes)*y+forcing,
                         (0., .2), np.zeros(3), rtol=1e-11, atol=1e-13)
    assert_allclose(np.array(kernel.weights) @ solution.y[:, -1], expected, rtol=2e-11)
    assert expected > 0


def test_fractional_kernel_fits_refine_in_integrated_cells_including_zero():
    edges = np.r_[0., np.geomspace(1e-7, 1., 2048)]
    exact = np.diff(edges**.65/gamma(1.65))
    errors = []
    for n in (2, 4, 8):
        kernel = approximate_fractional_kernel(.15, 1., n)
        masses = np.diff(kernel.integrated(edges))
        assert np.all(masses > 0)
        errors.append(np.sum(np.abs(masses-exact)))
        # Report actual singular first-cell *integral* error; no singular K(0).
        assert np.isfinite(masses[0]-exact[0])
        assert kernel(0.) > kernel(1.) > 0
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < .003


def test_kernel_scaling_and_exact_heston_kernel():
    k1 = approximate_fractional_kernel(.15, 1., 4)
    k2 = approximate_fractional_kernel(.15, .25, 4)
    assert_allclose(k2.nodes, np.array(k1.nodes)/.25, rtol=2e-14)
    assert_allclose(k2.weights, np.array(k1.weights)*.25**(-.35), rtol=2e-14)
    ordinary = approximate_fractional_kernel(.5, 2., 8)
    assert ordinary.nodes == (0.,)
    assert ordinary.weights == (1.,)
    assert_allclose(ordinary([0., 1.]), [1., 1.])
    assert_allclose(ordinary.integrated([0., 1.]), [0., 1.])


@pytest.mark.parametrize("kwargs", [dict(hurst=0), dict(hurst=.6), dict(horizon=0),
                                    dict(factors=True), dict(factors=0),
                                    dict(fit_points=4), dict(min_time_ratio=1),
                                    dict(max_nfev=0)])
def test_bad_approximation_controls(kwargs):
    settings = dict(hurst=.15, horizon=1., factors=4)
    settings.update(kwargs)
    with pytest.raises(ValueError):
        approximate_fractional_kernel(**settings)


def test_kernel_fitting_exhaustion_remains_identifiable():
    with pytest.raises(PricingError, match="kernel fit failed"):
        approximate_fractional_kernel(.15, 1., 8, max_nfev=1)


def _mapped_curve(params, node, weight):
    kappa = node+weight*params.kappa
    levels = params.curve.levels if hasattr(params, "curve") else (params.theta,)
    knots = params.curve.knot_times if hasattr(params, "curve") else ()
    theta = tuple((node*params.v0+weight*params.kappa*x)/kappa for x in levels)
    return HestonCurveParams(params.v0, kappa, weight*params.vol_of_vol, params.rho,
                             MeanReversionCurve(knots, theta))


def test_single_exponential_and_hurst_half_have_exact_heston_mapping():
    kernel = ExponentialKernel((.4,), (1.7,))
    u = np.array([0., -.5j, 1.2-.5j, 20.-.5j, -1j])
    assert_allclose(lifted_heston_cf(PARAMS, u, .7, kernel),
                    heston_curve_cf(_mapped_curve(PARAMS, .4, 1.7), u, .7), atol=1e-14)
    p = RoughHestonParams(.04, .065, 1.5, .3, -.6, .5)
    constant = approximate_fractional_kernel(.5, 1., 8)
    assert_allclose(lifted_heston_cf(p, u, .7, constant), heston_cf(p, u, .7), atol=1e-14)


@pytest.mark.parametrize("node,weight", [(0., 1.), (.4, 1.7)])
def test_forced_riccati_backend_matches_heston_instead_of_only_delegation(node, weight):
    curve = MeanReversionCurve((.2, .45), (.035, .08, .045))
    p = RoughHestonCurveParams(.04, 1.5, .3, -.6, curve, .15)
    u = np.array([.5-.5j, 1.4-.5j, 20.-.5j])
    kernel = ExponentialKernel((node,), (weight,))
    expected = heston_curve_cf(_mapped_curve(p, node, weight), u, .7)
    coarse = np.exp(_solve_lifted_riccati(p, u, .7, kernel, 256))
    fine = np.exp(_solve_lifted_riccati(p, u, .7, kernel, 1024))
    assert np.max(abs(fine-expected)) < np.max(abs(coarse-expected))/8
    assert_allclose(fine, expected, atol=3e-8)


def test_multifactor_riccati_agrees_with_independent_adaptive_ode():
    p = RoughHestonCurveParams(.04, 1.5, .3, -.6,
                              MeanReversionCurve((.2, .45), (.035, .08, .045)), .15)
    kernel = ExponentialKernel((.3, 20.), (1.2, 2.))
    maturity = .7
    expected = []
    for u in [.7-.5j, 4.-.5j, 10.-.5j]:
        def rhs(time, y):
            psi = np.array(kernel.weights) @ y[:2]
            f = -.5*(u*u+1j*u)+(1j*u*p.rho*p.vol_of_vol-p.kappa)*psi+.5*p.vol_of_vol**2*psi**2
            return np.r_[-np.array(kernel.nodes)*y[:2]+f,
                         p.v0*f+p.kappa*p.curve(maturity-time)*psi]
        y = np.zeros(3, complex)
        for start, end in zip([0., .25, .5], [.25, .5, .7]):
            result = solve_ivp(rhs, (start, end), y, method="DOP853", rtol=3e-12, atol=1e-13)
            assert result.success
            y = result.y[:, -1]
        expected.append(np.exp(y[-1]))
    actual = lifted_heston_cf(p, [.7-.5j, 4.-.5j, 10.-.5j], maturity, kernel,
                              config=PricingConfig(128, 80, 2048))
    assert_allclose(actual, expected, atol=2e-8)


def test_exact_mean_and_integral_follow_calendar_theta_knots():
    p = RoughHestonCurveParams(.04, 1.5, 0., -.6,
                              MeanReversionCurve((.2, .45), (.035, .08, .045)), .15)
    kernel = ExponentialKernel((.4,), (1.7,))
    mapped = _mapped_curve(p, .4, 1.7)
    times = np.array([0., .1, .2, .4, .7])
    assert_allclose(lifted_forward_variance(p, times, kernel), forward_variance(mapped, times), atol=2e-15)
    assert_allclose(lifted_integrated_variance(p, times, kernel), integrated_forward_variance(mapped, times), atol=2e-15)
    # Different levels with the same expiry-average theta do not imply the
    # same expected variance: the drift schedule is evaluated in calendar time.
    reversed_p = RoughHestonCurveParams(.04, 1.5, 0., -.6,
                                       MeanReversionCurve((.2, .45), (.045, .08, .035)), .15)
    assert abs(lifted_forward_variance(p, .7, kernel)-lifted_forward_variance(reversed_p, .7, kernel)) > .001


def test_multifactor_means_match_independent_deterministic_ode():
    kernel = ExponentialKernel((.3, 20.), (1.2, 2.))
    p = RoughHestonCurveParams(.04, 1.5, 0., -.6,
                              MeanReversionCurve((.2,), (.035, .08)), .15)
    def rhs(t, y):
        variance = p.v0+np.array(kernel.weights) @ y[:2]
        return np.r_[-np.array(kernel.nodes)*y[:2]+p.kappa*(p.curve(t)-variance), variance]
    y = np.zeros(3)
    for start, end in [(0., .2), (.2, .7)]:
        y = solve_ivp(rhs, (start, end), y, rtol=2e-11, atol=1e-13).y[:, -1]
    assert_allclose(lifted_forward_variance(p, .7, kernel), p.v0+np.array(kernel.weights) @ y[:2], atol=3e-11)
    assert_allclose(lifted_integrated_variance(p, .7, kernel), y[-1], atol=3e-12)


def test_deterministic_constant_variance_matches_black_scholes_any_kernel():
    p = RoughHestonParams(.04, .04, 1.5, 0., -.6, .15)
    kernel = approximate_fractional_kernel(.15, 1., 8)
    prices = lifted_heston_price(p, forward=100., discount=.97, strike=[90., 100., 110.],
                                 maturity=.7, is_call=[True, False, True], kernel=kernel)
    reference = black_scholes_price(BlackScholesParams(.2), forward=100., discount=.97,
                                    strike=[90., 100., 110.], maturity=.7,
                                    is_call=[True, False, True])
    assert_allclose(prices, reference, atol=2e-11)


def test_nonconstant_deterministic_variance_converges_to_fractional_path():
    p = RoughHestonCurveParams(.04, 1.5, 0., -.6,
                              MeanReversionCurve((.2, .4), (.025, .08, .04)), .15)
    times = np.array([0., .05, .2, .4, .6, 1.])
    means = np.asarray(forward_variance(p, times))
    integrals = np.asarray(integrated_forward_variance(p, times))
    mean_errors, integral_errors = [], []
    for factors in (2, 4, 8):
        kernel = approximate_fractional_kernel(.15, 1., factors)
        mean_errors.append(np.max(abs(lifted_forward_variance(p, times, kernel)-means)))
        integral_errors.append(np.max(abs(lifted_integrated_variance(p, times, kernel)-integrals)))
    assert mean_errors[2] < mean_errors[1] < mean_errors[0]
    assert integral_errors[2] < integral_errors[1] < integral_errors[0]
    assert mean_errors[-1] < 2e-5
    assert integral_errors[-1] < 2e-6


def test_lifted_prices_converge_to_fractional_reference_without_relaxed_tolerance():
    p = RoughHestonParams(.04, .04, 1.5, .3, -.6, .15)
    config = PricingConfig(192, 120., 2048)
    kwargs = dict(forward=100., discount=1., strike=[90., 100., 110.], maturity=1., config=config)
    fractional = rough_heston_price(p, **kwargs)
    allowance = .001+1e-4*abs(fractional)
    ratios = []
    for factors in (2, 4, 8):
        kernel = approximate_fractional_kernel(.15, 1., factors)
        prices = lifted_heston_price(p, **kwargs, kernel=kernel)
        ratios.append(np.max(abs(prices-fractional)/allowance))
    assert ratios[0] > 1.2  # The inadequate two-factor trial remains a failure.
    assert ratios[1] < 1.
    assert ratios[2] < .05


def test_price_broadcast_expiration_parity_and_bounds():
    kernel = ExponentialKernel((.3, 20.), (1.2, 2.))
    strikes = np.array([[90., 100., 110.]])
    maturity = np.array([[0.], [.7]])
    kwargs = dict(forward=100., discount=.97, strike=strikes, maturity=maturity,
                  kernel=kernel, config=PricingConfig(192, 120, 1024))
    calls = lifted_heston_price(PARAMS, **kwargs)
    puts = lifted_heston_price(PARAMS, **kwargs, is_call=False)
    assert calls.shape == puts.shape == (2, 3)
    assert_allclose(calls-puts, .97*np.broadcast_to(100-strikes, (2, 3)), atol=2e-13)
    assert_allclose(calls[0], .97*np.maximum(100-strikes[0], 0))
    assert np.all(calls >= .97*np.maximum(100-strikes, 0)-1e-13)
    assert np.all(calls <= 97.)
    assert np.isscalar(lifted_heston_cf(PARAMS, .5, .7, kernel))
    assert_allclose(lifted_heston_cf(PARAMS, [[0.], [-1j]], [0., .7], kernel), 1.)


def test_price_matches_one_factor_reference():
    p = RoughHestonParams(.04, .065, 1.5, .3, -.6, .5)
    kernel = approximate_fractional_kernel(.5, 1., 4)
    kwargs = dict(forward=100., discount=.97, strike=[90., 100., 110.], maturity=.7,
                  config=PricingConfig(192, 120, 512))
    assert_allclose(lifted_heston_price(p, **kwargs, kernel=kernel), heston_price(p, **kwargs), atol=1e-13)


@pytest.mark.parametrize("value", [-1., np.nan, True, "1"])
def test_invalid_times(value):
    kernel = ExponentialKernel((.3,), (1.,))
    with pytest.raises(ValueError):
        kernel(value)
    with pytest.raises(ValueError):
        lifted_forward_variance(PARAMS, value, kernel)


def test_metadata_domain_and_bad_frequency_are_rejected():
    kernel = approximate_fractional_kernel(.15, .5, 2)
    with pytest.raises(ValueError, match="horizon"):
        lifted_heston_cf(PARAMS, 1., .6, kernel)
    with pytest.raises(ValueError, match="hurst metadata"):
        lifted_heston_cf(PARAMS, 1., .2, approximate_fractional_kernel(.2, .5, 2))
    with pytest.raises(ValueError, match="frequencies"):
        lifted_heston_cf(PARAMS, np.nan, .2, kernel)
    with pytest.raises(TypeError, match="config"):
        lifted_heston_cf(PARAMS, 1., .2, kernel, config={})
