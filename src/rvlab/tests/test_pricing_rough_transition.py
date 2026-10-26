"""Independent generator/integral controls for the finite-lift stock split."""
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp

from rvlab.pricing import DiscountCurve, DividendSchedule, RoughHestonParams, SpotMarket
from rvlab.pricing.rough_lift import ExponentialKernel
from rvlab.pricing.rough_simulation import (
    RoughSimulationConfig, RoughSimulationError, _drift_flow_with_integral,
    _nonnegative_clock, _weak_transition_branches, simulate_rough_heston,
)


PARAMS = RoughHestonParams(.04, .045, 1.5, .3, -.6, .15)
NODES = np.array([.1, 4.])
WEIGHTS = np.array([.5, 1.5])


def _branches(h, *, scheme="exact_drift", old=None, nodes=NODES, weights=WEIGHTS):
    old = np.zeros((1, len(nodes))) if old is None else old
    matrix = -np.diag(nodes)-PARAMS.kappa*np.ones((len(nodes), 1))*weights
    forcing = np.full(len(nodes), PARAMS.kappa*(PARAMS.theta-PARAMS.v0))
    integral = _drift_flow_with_integral(matrix, forcing, weights, PARAMS.v0, h/2)
    branches = _weak_transition_branches(old, PARAMS, nodes=nodes, weights=weights,
        flow=integral[:len(nodes), :len(nodes)], shift=integral[:len(nodes), -1],
        integrated_flow=integral, dt=h, theta=PARAMS.theta, growth=.04*h,
        stock_integral=scheme)
    return branches, matrix, forcing


@pytest.mark.parametrize("invalid", [None, True, 3, "exact", [], np.array(["exact_drift"])])
def test_stock_integral_configuration_is_explicit(invalid):
    with pytest.raises(ValueError, match="stock_integral"):
        RoughSimulationConfig(stock_integral=invalid)


def test_stock_integral_construction_leaves_factor_transition_law_unchanged():
    exact, _, _ = _branches(.013)
    historical, _, _ = _branches(.013, scheme="trapezoidal")
    for name in ("new_factors", "probabilities", "before", "mid", "after", "orthogonal"):
        np.testing.assert_array_equal(exact[name], historical[name])
    assert not np.array_equal(exact["correlated"], historical["correlated"])
    np.testing.assert_allclose(exact["probabilities"].sum(1), 1., rtol=0., atol=3e-16)
    # Exact branch centering uses these post-roundoff probabilities, not an
    # idealized transition or the continuous-model stock martingale identity.
    assert np.min(exact["probabilities"]) >= 0.


def test_historical_mode_reproduces_archived_fixed_seed_paths():
    # Small byte-identical historical source fixture; the private report archive
    # is unnecessary for this fixed-seed numerical compatibility regression.
    archive = Path(__file__).with_name("fixtures")/"historical_rough_simulation_2026_10_18.py"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == "09b71db02df881ddbcca57cfad135d9249f0b49ffc787a3670637e13a28763fb"
    name = "rvlab.pricing._historical_rough_simulation_validation"
    spec = importlib.util.spec_from_file_location(name, archive)
    historical = importlib.util.module_from_spec(spec)
    sys.modules[name] = historical
    try:
        spec.loader.exec_module(historical)
        market = SpotMarket(100., DiscountCurve.flat(-.02, 1.), DividendSchedule((), 1.))
        kernel = ExponentialKernel(tuple(NODES), tuple(WEIGHTS), hurst=.15, horizon=1.)
        budgets = dict(paths=64, steps=19, seed=17, batch_size=7)
        old = historical.simulate_rough_heston(PARAMS, market, kernel,
            observation_times=[.03, .1], config=historical.RoughSimulationConfig(**budgets))
        current = simulate_rough_heston(PARAMS, market, kernel, observation_times=[.03, .1],
            config=RoughSimulationConfig(**budgets, stock_integral="trapezoidal"))
        for field in ("spots", "variances", "factors", "discounted_gains",
                      "conditional_log_spots", "conditional_variances"):
            np.testing.assert_allclose(getattr(current, field), getattr(old, field),
                rtol=3e-14, atol=1e-12)
        exact = simulate_rough_heston(PARAMS, market, kernel, observation_times=[.03, .1],
            config=RoughSimulationConfig(**budgets, stock_integral="exact_drift"))
        np.testing.assert_array_equal(exact.factors, current.factors)
        np.testing.assert_array_equal(exact.variances, current.variances)
        np.testing.assert_array_equal(exact.conditional_variances, current.conditional_variances)
        assert not np.array_equal(exact.spots, current.spots)
    finally:
        sys.modules.pop(name, None)


def test_stiff_drift_integrals_agree_with_independent_ode_and_quadrature():
    h = .04
    nodes, weights = np.array([.46, 7.81, 55.4, 541.7]), np.array([1.13, 1.17, 2.33, 8.17])
    branches, matrix, forcing = _branches(h, nodes=nodes, weights=weights)
    old = np.zeros(4)
    first_half = solve_ivp(lambda t, u: matrix@u+forcing, (0, h/2), old,
        method="DOP853", rtol=1e-12, atol=1e-15, dense_output=True)
    first = quad(lambda t: PARAMS.v0+weights@first_half.sol(t), 0, h/2,
        epsabs=1e-13, epsrel=1e-12)[0]
    mid = PARAMS.v0+weights@first_half.y[:, -1]
    # Recover each diffusion jump from the final factor state using an
    # independent backwards affine ODE, then integrate its forward drift.
    for branch, terminal in enumerate(branches["new_factors"][0]):
        recovered = solve_ivp(lambda t, u: -(matrix@u+forcing), (0, h/2), terminal,
            method="DOP853", rtol=1e-12, atol=1e-15).y[:, -1]
        second_half = solve_ivp(lambda t, u: matrix@u+forcing, (0, h/2), recovered,
            method="DOP853", rtol=1e-12, atol=1e-15, dense_output=True)
        second = quad(lambda t: PARAMS.v0+weights@second_half.sol(t), 0, h/2,
            epsabs=1e-13, epsrel=1e-12)[0]
        np.testing.assert_allclose(branches["correlated_clock"][0, branch], first+second,
            rtol=3e-8, atol=2e-11)
        diffusion_jump = recovered[0]-first_half.y[0, -1]
        expected = .04*h+PARAMS.rho/PARAMS.vol_of_vol*diffusion_jump-.5*PARAMS.rho**2*(first+second)
        np.testing.assert_allclose(branches["correlated"][0, branch], expected, rtol=2e-8, atol=2e-10)
    assert mid > 0.


def _continuous_joint_transform(h, p, q):
    """Independent affine generator ODE for E[exp(p*log(S/S0)+q.U)]."""
    matrix = -np.diag(NODES)-PARAMS.kappa*np.ones((2, 1))*WEIGHTS
    forcing = np.full(2, PARAMS.kappa*(PARAMS.theta-PARAMS.v0))
    def rhs(time, state):
        coefficient = state[1:]
        total = coefficient.sum()
        quadratic = (.5*PARAMS.vol_of_vol**2*total**2
            +p*PARAMS.rho*PARAMS.vol_of_vol*total+.5*(p*p-p))
        return np.r_[coefficient@forcing+PARAMS.v0*quadratic+p*.04,
                     matrix.T@coefficient+WEIGHTS*quadratic]
    result = solve_ivp(rhs, (0, h), np.r_[0j, q.astype(complex)],
        method="DOP853", rtol=1e-12, atol=1e-15)
    assert result.success
    return np.exp(result.y[0, -1])


@pytest.mark.parametrize("p,q", [(1.3j, np.array([.1, -.2])), (.7, np.array([-.3, .2]))])
def test_shared_stock_branch_generator_against_independent_affine_ode(p, q):
    errors = {scheme: [] for scheme in ("exact_drift", "trapezoidal")}
    for h in (.02, .01, .005):
        reference = _continuous_joint_transform(h, p, q)
        for scheme in errors:
            b, _, _ = _branches(h, scheme=scheme)
            # The independent Gaussian stock driver is integrated exactly.
            joint = np.exp(p*b["correlated"][:, :, None]
                +.5*(p*p-p)*b["orthogonal"]
                +(b["new_factors"]@q)[:, :, None])
            estimate = (joint*b["probabilities"][:, :, None]/2).sum()
            errors[scheme].append(abs(estimate-reference))
    # Local O(h^3) is a one-step consistency control on these smooth transforms,
    # not a general global order or vanilla-price accuracy certificate.
    for values in errors.values():
        ratios = np.asarray(values[:-1])/values[1:]
        assert np.all((ratios > 7.) & (ratios < 10.))
    assert errors["exact_drift"][-1] < errors["trapezoidal"][-1]


def test_nonnegative_drift_integrals_reject_material_failures():
    with pytest.raises(RoughSimulationError, match="integrated variance"):
        _nonnegative_clock(np.array([.001, -1e-7]))
    with pytest.raises(RoughSimulationError, match="integrated variance"):
        _nonnegative_clock(np.array([np.nan]))
    np.testing.assert_array_equal(_nonnegative_clock(np.array([-1e-16, .01])), [0., .01])
