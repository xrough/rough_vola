"""Short-dated SPY-like regressions that separate Fourier error sources.

The fixtures retain the scales and fitted parameter regimes of the August 7
demonstration, but require no market files. Small integration error on a finite
interval does not establish that the omitted characteristic-function tail is
small. These cases caught that distinction in the original demonstration.
"""

from __future__ import annotations

from math import log, pi, sqrt

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp

from rvlab.pricing import HestonParams, PricingConfig, heston_cf, heston_price


SLOW_TAIL = HestonParams(
    v0=0.019455920324654978, theta=0.2499999999999704,
    kappa=0.20979706584973362, vol_of_vol=0.6, rho=-0.95,
)
FITTED_TAIL = HestonParams(
    v0=0.019978211577066824, theta=0.24999999999984407,
    kappa=0.16949918021056282, vol_of_vol=0.6, rho=-0.8331920907620385,
)


def _contracts():
    """Three tenors, six OTM contracts each, with explicit forward inputs."""
    forward = np.repeat([632.5141036265508, 633.0802169001261,
                         633.5182839740953], 6)
    discount = np.repeat([0.9981552168359522, 0.9972960794336722,
                          0.9964376815126508], 6)
    maturity = np.repeat([0.04112442922374429, 0.0603025114155251,
                          0.0794805936073059], 6)
    strike = np.array([614., 620., 626., 639., 645., 651.,
                       614., 621., 627., 639., 646., 651.,
                       615., 621., 627., 640., 646., 651.])
    return dict(forward=forward, discount=discount, maturity=maturity,
                strike=strike, is_call=strike > forward)


def _infinite_integral_price(params, forward, discount, strike, maturity, is_call):
    """Adaptive scalar Lewis integral without the production BS control variate."""
    log_moneyness = log(forward / strike)

    def integrand(u):
        value = np.exp(1j * log_moneyness * u) * heston_cf(
            params, u - 0.5j, maturity,
        )
        return float(np.real(value) / (u * u + 0.25))

    integral, error = quad(integrand, 0., np.inf, epsabs=1e-12,
                           epsrel=1e-12, limit=500)
    assert error < 2e-11
    call = discount * (forward - sqrt(forward * strike) * integral / pi)
    return call if is_call else call - discount * (forward - strike)


@pytest.mark.parametrize("params", [SLOW_TAIL, FITTED_TAIL])
def test_short_dated_heston_matches_independent_infinite_scalar_integration(params):
    contracts = _contracts()
    # Both option sides at every tenor. Only the analytic CF is shared; the
    # scalar adaptive infinite integral has neither the finite production grid
    # nor its Black--Scholes control variate.
    selected = np.array([0, 5, 6, 11, 12, 17])
    selected_contracts = {key: values[selected] for key, values in contracts.items()}
    actual = heston_price(params, config=PricingConfig(192, 1200., 256),
                          **selected_contracts)
    expected = [_infinite_integral_price(params, **{
        key: values[i] for key, values in selected_contracts.items()
    }) for i in range(len(selected))]
    np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=0.)


@pytest.mark.parametrize(
    "params,low_error,medium_error,high_error",
    [(SLOW_TAIL, .02, .0006, .00004),
     (FITTED_TAIL, .008, .00002, .0000002)],
)
def test_heston_tail_requires_longer_range_across_all_three_expiries(
    params, low_error, medium_error, high_error,
):
    contracts = _contracts()
    reference = heston_price(params, config=PricingConfig(192, 1200., 256),
                             **contracts)
    prices = [heston_price(params, config=PricingConfig(192, cutoff, 256),
                           **contracts) for cutoff in (160., 400., 600.)]
    errors = [np.max(np.abs(values - reference)) for values in prices]
    assert errors[0] > low_error
    assert errors[1] < medium_error
    assert errors[2] < high_error
    assert errors[2] < errors[1] < errors[0]


@pytest.mark.parametrize("params", [SLOW_TAIL, FITTED_TAIL])
def test_more_fourier_nodes_do_not_repair_a_truncated_heston_tail(params):
    contracts = _contracts()
    prices = [heston_price(params, config=PricingConfig(nodes, 160., 256),
                           **contracts) for nodes in (64, 128, 256)]
    np.testing.assert_allclose(prices[0], prices[2], atol=2e-10, rtol=0.)
    np.testing.assert_allclose(prices[1], prices[2], atol=2e-10, rtol=0.)
    # The longer range is also resolved independently of the low-cutoff sweep.
    long_range = [heston_price(params, config=PricingConfig(nodes, 600., 256),
                               **contracts) for nodes in (96, 192)]
    np.testing.assert_allclose(*long_range, atol=2e-10, rtol=0.)


@pytest.mark.parametrize("params", [SLOW_TAIL, FITTED_TAIL])
def test_heston_closed_form_cf_matches_independent_riccati_ode(params):
    frequencies = np.array([1. - .5j, 80. - .5j, 400. - .5j])
    maturity = .04112442922374429

    def rhs(_time, state):
        coefficient = state[:len(frequencies)]
        q = frequencies * frequencies + 1j * frequencies
        derivative = (-.5 * q
                      + (1j * params.rho * params.vol_of_vol * frequencies
                         - params.kappa) * coefficient
                      + .5 * params.vol_of_vol**2 * coefficient**2)
        return np.concatenate((derivative, params.kappa * params.theta * coefficient))

    solution = solve_ivp(rhs, (0., maturity), np.zeros(6, dtype=complex),
                         method="DOP853", rtol=1e-11, atol=1e-12)
    assert solution.success
    terminal = solution.y[:, -1]
    expected = np.exp(params.v0 * terminal[:3] + terminal[3:])
    np.testing.assert_allclose(heston_cf(params, frequencies, maturity), expected,
                               atol=2e-12, rtol=2e-10)
