"""Independent limits and stiff-frequency checks for the implicit corrector."""

import numpy as np
import pytest
from scipy.special import gamma

from rvlab.pricing import (
    PricingConfig, PricingError, RoughHestonParams, heston_cf,
    rough_heston_cf, solve_fractional_riccati,
)
from rvlab.pricing.rough_heston import _implicit_riccati_step, _terminal_integral


@pytest.mark.parametrize("eta", [0.0, 1e-8, 0.6])
def test_implicit_corrector_solves_the_quadratic_without_cancellation(eta):
    history = np.array([-4.0 + 3j, -4.0 - 3j, -0.03 + 0.2j])
    forcing = np.array([-50.0, -50.0, -0.3])
    linear = np.array([-1.0 + 25j, -1.0 - 25j, -0.1])
    scale = 0.004
    quadratic = 0.5 * eta**2
    solution = _implicit_riccati_step(history, forcing, linear, quadratic, scale)
    residual = solution - history - scale * (
        forcing + linear * solution + quadratic * solution**2)
    np.testing.assert_allclose(residual, 0, atol=2e-15, rtol=0)
    if eta <= 1e-8:
        np.testing.assert_allclose(
            solution, (history + scale * forcing) / (1 - scale * linear),
            atol=2e-15, rtol=0,
        )


def test_implicit_corrector_preserves_physical_branch_with_stiff_positive_rho():
    # On u=-i/2 the solution is real and nonpositive. At large positive rho,
    # q=1-a*b can be negative; choosing the root closest to q would give the
    # other, positive solution and hence an impossible half moment above one.
    values = []
    for eta in np.geomspace(1e-6, 100.0, 100):
        value = _implicit_riccati_step(np.array([0j]), np.array([-0.125]),
            np.array([0.5 * 0.95 * eta - 1.0]), 0.5 * eta**2, 0.5)[0]
        assert value.real < 0
        assert value.imag == 0
        values.append(value)
    assert np.all(np.isfinite(values))
    zero = _implicit_riccati_step(np.array([0j]), np.array([0j]),
        np.array([100.0 + 0j]), 0.5, 1.0)
    np.testing.assert_array_equal(zero, 0)


def _fractional_power_series_cf(params, frequencies, maturity, terms=60):
    """Local convergent stochastic Riccati series, independent of time stepping.

    For h(t)=sum(a_n*t**(alpha*n)), match coefficients in D**alpha h=F(h).
    Integrate each power analytically for the two terms in the log CF.
    The small frequencies and short tenor below lie well inside convergence.
    """
    alpha = params.hurst + 0.5
    forcing = -0.5 * (frequencies**2 + 1j * frequencies)
    linear = 1j * frequencies * params.rho * params.vol_of_vol - params.kappa
    coefficients = np.zeros((terms + 1, len(frequencies)), dtype=complex)
    coefficients[1] = forcing / gamma(alpha + 1)
    for n in range(1, terms):
        convolution = np.sum(
            coefficients[1:n] * coefficients[n - 1:0:-1], axis=0)
        coefficients[n + 1] = (
            linear * coefficients[n] + 0.5 * params.vol_of_vol**2 * convolution
        ) * gamma(alpha * n + 1) / gamma(alpha * (n + 1) + 1)
    powers = alpha * np.arange(1, terms + 1)
    ordinary = np.sum(coefficients[1:] * (
        maturity**(powers + 1) / (powers + 1))[:, None], axis=0)
    order = 1 - alpha
    fractional = np.sum(coefficients[1:] * (
        gamma(powers + 1) / gamma(powers + order + 1)
        * maturity**(powers + order))[:, None], axis=0)
    return np.exp(params.kappa * params.theta * ordinary + params.v0 * fractional)


@pytest.mark.parametrize("hurst", [0.05, 0.15, 0.4])
def test_stochastic_fractional_cf_converges_to_independent_power_series(hurst):
    params = RoughHestonParams(0.04, 0.045, 1.0, 0.3, -0.6, hurst)
    frequencies = np.array([0.3, 1.0, 5.0]) - 0.5j
    expected = _fractional_power_series_cf(params, frequencies, 0.08)
    # Verify that truncating the independent series is immaterial here.
    np.testing.assert_allclose(expected,
        _fractional_power_series_cf(params, frequencies, 0.08, terms=40),
        atol=1e-14, rtol=0)
    errors = [np.max(np.abs(rough_heston_cf(params, frequencies, 0.08,
        config=PricingConfig(riccati_steps=n)) - expected)) for n in (64, 128, 256)]
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 8.1e-7


def test_implicit_fractional_solver_at_alpha_one_matches_analytic_heston():
    # Call the solver directly: rough_heston_cf delegates H=.5 analytically,
    # which would bypass the numerical method that this test checks.
    params = RoughHestonParams(0.02, 0.05, 1.0, 0.6, -0.95, 0.5)
    frequencies = np.array([1.0, 80.0, 400.0, 1600.0]) - 0.5j
    expected = heston_cf(params, frequencies, 0.08)
    errors = []
    for n in (128, 256, 512):
        _, solution = solve_fractional_riccati(params, frequencies, 0.08, n_steps=n)
        actual = np.exp(params.kappa * params.theta
            * _terminal_integral(solution, 0.08, 1.0) + params.v0 * solution[-1])
        errors.append(np.max(np.abs(actual - expected)))
    assert errors[2] < 7e-7
    assert errors[1] < 0.26 * errors[0]
    assert errors[2] < 0.26 * errors[1]


def test_stiff_high_frequency_cf_remains_finite_and_time_refines():
    params = RoughHestonParams(0.0025, 0.25, 0.1, 0.6, -0.95, 0.05)
    frequencies = np.array([400.0, 640.0, 1000.0, 1600.0]) - 0.5j
    values = [rough_heston_cf(params, frequencies, 0.08,
        config=PricingConfig(riccati_steps=n)) for n in (256, 512, 1024)]
    assert np.all(np.isfinite(values))
    assert np.all(np.abs(values) <= 1)
    # The former explicit predictor diverged for u=400 at these coarse grids.
    # Being finite does not claim this grid is accurate: require convergence.
    coarse_change = np.max(np.abs(values[1] - values[0]))
    fine_change = np.max(np.abs(values[2] - values[1]))
    assert fine_change < 0.5 * coarse_change
    assert fine_change < 4e-4


def test_spy_cf_agrees_with_high_resolution_explicit_adams_reference():
    params = RoughHestonParams(
        0.02041023725793761, 0.045, 1.0, 0.3, -0.6845172910552568, 0.15)
    frequencies = np.array([10.0, 80.0, 160.0, 240.0]) - 0.5j
    # Independent PECE predictor/corrector run with 8192 steps. Its difference
    # from 4096 steps was at most 2.10e-7, below this comparison's tolerance.
    # Retaining the values avoids an O(8192²) reference run in the test suite.
    expected = np.array([
        0.9176860009179342 + 0.017743867143021362j,
        -0.004482116691848827 + 0.1391616842440029j,
        -0.008124891353484378 - 0.00878632353817613j,
        0.0009731295649846387 + 0.00007043779699673674j,
    ])
    actual = rough_heston_cf(params, frequencies, 0.08,
        config=PricingConfig(riccati_steps=1024))
    np.testing.assert_allclose(actual, expected, atol=1.1e-6, rtol=0)


def test_stiffness_is_distinct_from_nonfinite_numerical_failure():
    params = RoughHestonParams(0.04, 0.04, 1.0, 1e6, -0.7, 0.1)
    _, solution = solve_fractional_riccati(params, 100 - 0.5j, 10.0, n_steps=16)
    assert np.all(np.isfinite(solution))
    with pytest.raises(PricingError, match="initial forcing.*nonfinite"):
        solve_fractional_riccati(params, 2e154 - 0.5j, 10.0, n_steps=16)
    overflow = RoughHestonParams(0.04, 0.04, 1.0, 1e200, -0.7, 0.1)
    with pytest.raises(PricingError, match="coefficients exceed floating-point"):
        solve_fractional_riccati(overflow, 1.0 - 0.5j, 0.1, n_steps=16)
