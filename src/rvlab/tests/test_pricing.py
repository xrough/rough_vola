"""Independent identities and convergence checks for European model pricing."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from math import erf, gamma, log, sqrt
import importlib
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from scipy.integrate import quad

from rvlab.pricing import (
    BlackScholesParams,
    HestonParams,
    PricingConfig,
    PricingError,
    RoughHestonParams,
    black_scholes_price,
    heston_cf,
    heston_price,
    price,
    rough_heston_cf,
    rough_heston_price,
    solve_fractional_riccati,
)


def _normal_cdf(x):
    return (1.0 + erf(x / sqrt(2.0))) / 2.0


def _black_reference(forward, strike, total_variance, discount, is_call=True):
    """Scalar Black formula, independent of the package's vectorised routine."""
    if total_variance == 0.0:
        return discount * max((forward - strike) * (1 if is_call else -1), 0.0)
    root_variance = sqrt(total_variance)
    d1 = log(forward / strike) / root_variance + root_variance / 2.0
    d2 = d1 - root_variance
    if is_call:
        return discount * (forward * _normal_cdf(d1) - strike * _normal_cdf(d2))
    return discount * (strike * _normal_cdf(-d2) - forward * _normal_cdf(-d1))


HESTON = HestonParams(v0=0.04, theta=0.055, kappa=1.6, vol_of_vol=0.3, rho=-0.65)
ROUGH = RoughHestonParams(
    v0=0.04, theta=0.055, kappa=1.3, vol_of_vol=0.25, rho=-0.65, hurst=0.25,
)


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("strike", [70.0, 100.0, 135.0])
def test_black_scholes_matches_independent_scalar_formula(is_call, strike):
    args = dict(forward=103.0, discount=0.97, strike=strike, maturity=0.7, is_call=is_call)
    actual = black_scholes_price(BlackScholesParams(0.27), **args)
    assert isinstance(actual, float)
    assert actual == pytest.approx(
        _black_reference(103.0, strike, 0.27**2 * 0.7, 0.97, is_call), abs=2e-13,
    )


@pytest.mark.parametrize("params", [BlackScholesParams(0.2), HESTON, ROUGH])
def test_all_models_honour_broadcasting_expiration_and_put_call_parity(params):
    # Different forwards and maturities in rows, different strikes and sides in columns.
    forward = np.array([[101.0], [103.0]])
    maturity = np.array([[0.0], [0.3]])
    strike = np.array([90.0, 100.0, 110.0])
    discount = np.array([[1.0], [0.98]])
    args = dict(forward=forward, discount=discount, strike=strike, maturity=maturity)
    calls = price(params, **args)
    puts = price(params, **args, is_call=False)
    mixed = price(params, **args, is_call=np.array([True, False, True]))
    assert isinstance(calls, np.ndarray)
    assert calls.shape == puts.shape == mixed.shape == (2, 3)
    np.testing.assert_allclose(calls - puts, discount * (forward - strike), atol=1e-10)
    np.testing.assert_allclose(calls[0], np.maximum(forward[0] - strike, 0), atol=0)
    np.testing.assert_allclose(puts[0], np.maximum(strike - forward[0], 0), atol=0)
    np.testing.assert_allclose(mixed, np.where([True, False, True], calls, puts), atol=1e-12)
    assert np.all(calls >= discount * np.maximum(forward - strike, 0) - 1e-8)
    assert np.all(calls <= discount * forward + 1e-8)
    assert np.all(puts >= discount * np.maximum(strike - forward, 0) - 1e-8)
    assert np.all(puts <= discount * strike + 1e-8)


def test_black_scholes_zero_volatility_is_discounted_intrinsic_value():
    actual = price(
        BlackScholesParams(0.0), forward=100.0, discount=0.93,
        strike=[80.0, 100.0, 120.0], maturity=2.0,
    )
    np.testing.assert_array_equal(actual, [18.6, 0.0, 0.0])


@pytest.mark.parametrize("params", [BlackScholesParams(0.2), HESTON, ROUGH])
def test_scalar_model_price_returns_python_float(params):
    result = price(params, forward=100.0, discount=0.99, strike=100.0, maturity=0.2)
    assert isinstance(result, float)
    assert 0 < result < 99


@pytest.mark.parametrize(
    "override",
    [dict(forward=0), dict(forward=np.inf), dict(strike=-1), dict(strike=np.nan),
     dict(discount=0), dict(discount=np.nan), dict(maturity=-0.01), dict(maturity=np.inf)],
)
@pytest.mark.parametrize("params", [BlackScholesParams(0.2), HESTON, ROUGH])
def test_market_inputs_are_validated_even_at_expiry(params, override):
    args = dict(forward=100.0, discount=0.99, strike=100.0, maturity=0.0)
    args.update(override)
    with pytest.raises(ValueError):
        price(params, **args)


@pytest.mark.parametrize("params", [BlackScholesParams(0.2), HESTON, ROUGH])
@pytest.mark.parametrize("override", [dict(forward=100 + 1j), dict(strike=[90, 100 + 2j])])
def test_complex_market_inputs_are_rejected_instead_of_discarding_imaginary_parts(params, override):
    args = dict(forward=100.0, discount=0.99, strike=100.0, maturity=0.2)
    args.update(override)
    with pytest.raises(ValueError):
        price(params, **args)


@pytest.mark.parametrize(
    "factory,kwargs",
    [(BlackScholesParams, dict(volatility=-0.1)),
     (BlackScholesParams, dict(volatility=np.inf)),
     (HestonParams, dict(v0=-0.01)), (HestonParams, dict(theta=np.nan)),
     (HestonParams, dict(kappa=-1)), (HestonParams, dict(vol_of_vol=-0.2)),
     (HestonParams, dict(rho=1.01)), (RoughHestonParams, dict(hurst=0)),
     (RoughHestonParams, dict(hurst=0.51)), (RoughHestonParams, dict(hurst=np.nan))],
)
def test_invalid_parameters_are_rejected(factory, kwargs):
    if factory is BlackScholesParams:
        args = kwargs
    else:
        args = dict(v0=0.04, theta=0.04, kappa=1, vol_of_vol=0.3, rho=-0.5)
        if factory is RoughHestonParams:
            args["hurst"] = 0.1
        args.update(kwargs)
    with pytest.raises(ValueError):
        factory(**args)


def test_parameters_and_numerical_settings_are_immutable():
    for obj, field, value in [
        (BlackScholesParams(0.2), "volatility", 0.3),
        (HESTON, "rho", 0.0), (ROUGH, "hurst", 0.3),
        (PricingConfig(), "quadrature_nodes", 64),
    ]:
        with pytest.raises(FrozenInstanceError):
            setattr(obj, field, value)


def test_refinement_increases_all_relevant_numerical_resolutions():
    config = PricingConfig(quadrature_nodes=64, fourier_cutoff=50, riccati_steps=100)
    refined = config.refined()
    assert refined.quadrature_nodes == 128
    assert refined.riccati_steps == 200
    assert refined.fourier_cutoff == 62.5
    assert config.quadrature_nodes == 64


@pytest.mark.parametrize(
    "kwargs",
    [dict(quadrature_nodes=0), dict(quadrature_nodes=2.5), dict(riccati_steps=-1),
     dict(riccati_steps=True), dict(fourier_cutoff=0), dict(fourier_cutoff=np.inf)],
)
def test_invalid_numerical_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        PricingConfig(**kwargs)


@pytest.mark.parametrize("kappa", [0.0, 1.8])
def test_heston_deterministic_variance_matches_integrated_variance(kappa):
    params = HestonParams(v0=0.015, theta=0.09, kappa=kappa, vol_of_vol=0.0, rho=-0.8)
    maturities = np.array([0.04, 0.4, 2.0])
    strikes = np.array([90.0, 100.0, 115.0])
    actual = heston_price(
        params, forward=101.0, discount=0.96, strike=strikes, maturity=maturities,
    )
    variance = (
        params.v0 * maturities if kappa == 0 else
        params.theta * maturities + (params.v0 - params.theta)
        * (-np.expm1(-kappa * maturities)) / kappa
    )
    expected = [
        _black_reference(101.0, strike, total_var, 0.96)
        for strike, total_var in zip(strikes, variance)
    ]
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_heston_characteristic_function_is_continuous_at_zero_vol_of_vol():
    # Small eta exposes cancellation in the closed-form log-ratio, even when
    # exactly zero eta has a separate deterministic implementation.
    args = dict(v0=0.015, theta=0.09, kappa=1.8, rho=-0.8)
    u = np.array([0.5 - 0.5j, 1.0, 3.0])
    deterministic = heston_cf(HestonParams(**args, vol_of_vol=0.0), u, 0.7)
    actual = heston_cf(HestonParams(**args, vol_of_vol=1e-8), u, 0.7)
    np.testing.assert_allclose(actual, deterministic, atol=1e-8, rtol=0)


@pytest.mark.parametrize("hurst", [0.05, 0.25, 0.5])
def test_fractional_constant_forcing_includes_the_initial_rhs(hurst):
    params = RoughHestonParams(
        v0=0.04, theta=0.08, kappa=0, vol_of_vol=0, rho=-0.5, hurst=hurst,
    )
    frequencies = np.array([0.3, 1.0 - 0.5j, 2.0])
    times, solution = solve_fractional_riccati(params, frequencies, 0.4, n_steps=40)
    alpha = hurst + 0.5
    forcing = -0.5 * (frequencies**2 + 1j * frequencies)
    expected = times[:, None] ** alpha * forcing / gamma(alpha + 1)
    assert times.shape == (41,)
    assert solution.shape == expected.shape == (41, 3)
    assert times[0] == 0
    assert times[-1] == pytest.approx(0.4)
    np.testing.assert_allclose(solution, expected, rtol=1e-12, atol=1e-13)


def test_fractional_solver_scalar_frequency_shape_and_initial_value():
    times, solution = solve_fractional_riccati(ROUGH, 1.0 - 0.5j, 0.1, n_steps=32)
    assert times.shape == solution.shape == (33,)
    assert solution[0] == 0j
    assert np.all(np.isfinite(solution))


@pytest.mark.parametrize("hurst", [0.1, 0.3])
def test_rough_heston_constant_variance_reduces_to_black_scholes(hurst):
    params = RoughHestonParams(
        v0=0.0625, theta=0.0625, kappa=1.7, vol_of_vol=0, rho=-0.6, hurst=hurst,
    )
    args = dict(forward=100.0, discount=0.97, strike=[85.0, 100.0, 120.0], maturity=0.6)
    np.testing.assert_allclose(
        rough_heston_price(params, **args),
        black_scholes_price(BlackScholesParams(0.25), **args),
        atol=1e-9,
    )


def test_rough_deterministic_mean_reversion_against_mittag_leffler_series():
    # With eta=0, V(t)=theta+(v0-theta)*E_alpha(-kappa*t**alpha).
    # Integrating its absolutely convergent series gives a separate reference
    # for both the Riccati solution and the CF's fractional-integral assembly.
    params = RoughHestonParams(
        v0=0.02, theta=0.07, kappa=1.3, vol_of_vol=0, rho=-0.4, hurst=0.2,
    )
    maturity = 0.4
    alpha = params.hurst + 0.5
    argument = -params.kappa * maturity**alpha
    integrated_series = sum(argument**n / gamma(alpha * n + 2) for n in range(60))
    total_variance = (params.theta * maturity
                      + (params.v0 - params.theta) * maturity * integrated_series)
    u = np.array([0.5 - 0.5j, 1.0, 3.0])
    expected = np.exp(-0.5 * (u * u + 1j * u) * total_variance)
    actual = rough_heston_cf(
        params, u, maturity, config=PricingConfig(riccati_steps=256),
    )
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)


def test_rough_heston_half_hurst_is_the_heston_model():
    rough = RoughHestonParams(
        v0=HESTON.v0, theta=HESTON.theta, kappa=HESTON.kappa,
        vol_of_vol=HESTON.vol_of_vol, rho=HESTON.rho, hurst=0.5,
    )
    u = np.array([0.0, 0.7 - 0.5j, 3.0])
    np.testing.assert_allclose(rough_heston_cf(rough, u, 0.6), heston_cf(HESTON, u, 0.6), atol=1e-13)
    args = dict(forward=102.0, discount=0.98, strike=[85.0, 100.0, 120.0], maturity=0.6)
    np.testing.assert_allclose(rough_heston_price(rough, **args), heston_price(HESTON, **args), atol=1e-12)


@pytest.mark.parametrize("params,cf", [(HESTON, heston_cf), (ROUGH, rough_heston_cf)])
def test_characteristic_function_normalization_and_reality_symmetry(params, cf):
    np.testing.assert_allclose(cf(params, np.array([0.0, -1j]), 0.3), 1, atol=1e-12)
    u = np.array([0.1, 0.5, 2.0])
    values = cf(params, u, 0.3)
    assert np.all(np.abs(values) <= 1 + 1e-12)
    np.testing.assert_allclose(cf(params, -u, 0.3), np.conj(values), atol=1e-12)
    np.testing.assert_array_equal(cf(params, u, 0.0), np.ones_like(u))


def test_batch_fourier_prices_match_independent_adaptive_scalar_integration():
    forward, discount, maturity = 103.0, 0.97, 0.7
    strikes = np.array([85.0, 103.0, 125.0])
    config = PricingConfig(quadrature_nodes=256, fourier_cutoff=180, riccati_steps=128)
    actual = heston_price(
        HESTON, forward=forward, discount=discount, strike=strikes, maturity=maturity,
        config=config,
    )
    expected = []
    for strike in strikes:
        def integrand(u):
            value = np.exp(1j * u * log(forward / strike)) * heston_cf(
                HESTON, u - 0.5j, maturity,
            )
            return float(np.real(value) / (u * u + 0.25))

        integral, error = quad(integrand, 0, 250, epsabs=1e-9, epsrel=1e-10, limit=200)
        assert error < 1e-7
        expected.append(discount * (forward - sqrt(forward * strike) * integral / np.pi))
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=1e-7)


def test_nontrivial_rough_prices_converge_with_fractional_resolution():
    args = dict(forward=100.0, discount=0.99, strike=[90.0, 100.0, 110.0], maturity=0.2)
    prices = [
        rough_heston_price(
            ROUGH, **args,
            config=PricingConfig(quadrature_nodes=128, fourier_cutoff=60, riccati_steps=n),
        )
        for n in (64, 128, 256)
    ]
    coarse_change = np.max(np.abs(prices[1] - prices[0]))
    fine_change = np.max(np.abs(prices[2] - prices[1]))
    assert fine_change < coarse_change
    assert fine_change < 0.01  # under one cent per share, with an intentionally modest grid
    assert np.all(np.isfinite(prices))


def test_divergent_fractional_solution_raises_identifiable_numerical_error():
    # Stiffness alone is handled by the implicit corrector. This finite input
    # instead makes the quadratic forcing overflow its floating-point range.
    with pytest.raises(PricingError, match="initial forcing.*nonfinite"):
        solve_fractional_riccati(ROUGH, 2e154 - 0.5j, 10.0, n_steps=16)


@pytest.mark.parametrize("model", [HestonParams, RoughHestonParams])
@pytest.mark.parametrize("kappa,theta", [(0.0, 0.08), (1.3, 0.0)])
def test_zero_initial_variance_with_no_variance_drift_is_absorbing(model, kappa, theta):
    args = dict(v0=0.0, theta=theta, kappa=kappa, vol_of_vol=5.0, rho=-0.7)
    if model is RoughHestonParams:
        args["hurst"] = 0.1
    params = model(**args)
    cf = rough_heston_cf if model is RoughHestonParams else heston_cf
    # Nonzero vol-of-vol has no effect when the initial variance and its
    # immigration term are both zero, including on a coarse numerical grid.
    u = np.array([0.0, -1j, 2.0, 100 - 0.5j])
    np.testing.assert_array_equal(cf(params, u, 3.0), np.ones(4, dtype=complex))
    forward = np.array([[100.0], [103.0]])
    strike = np.array([85.0, 100.0, 120.0])
    calls = np.array([True, False, True])
    market = dict(
        forward=forward, discount=0.97, strike=strike,
        maturity=np.array([[0.0], [3.0]]), is_call=calls,
    )
    actual = price(params, **market, config=PricingConfig(riccati_steps=8))
    expected = 0.97 * np.maximum(np.where(calls, forward - strike, strike - forward), 0)
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError):
        price(params, forward=100, discount=1, strike=-1, maturity=0.5)


@pytest.mark.parametrize(
    "params,module_name,cf_name",
    [(HESTON, "heston", "heston_cf"), (ROUGH, "rough_heston", "rough_heston_cf")],
)
def test_fourier_characteristic_function_is_batched_once_per_positive_maturity(
    monkeypatch, params, module_name, cf_name,
):
    module = importlib.import_module(f"rvlab.pricing.{module_name}")
    implementation = getattr(module, cf_name)
    seen = []

    def record_call(params, u, maturity, **kwargs):
        seen.append((float(maturity), np.asarray(u).shape))
        return implementation(params, u, maturity, **kwargs)

    monkeypatch.setattr(module, cf_name, record_call)
    config = PricingConfig(quadrature_nodes=64, fourier_cutoff=40, riccati_steps=64)
    values = price(
        params, forward=100, discount=0.99, strike=[90, 100, 110, 95, 105, 100],
        maturity=[0.3, 0.1, 0.3, 0.1, 0.3, 0], config=config,
    )
    assert values.shape == (6,)
    assert sorted(seen) == [(0.1, (64,)), (0.3, (64,))]


def test_fresh_import_and_model_prices_do_not_require_external_research_libraries():
    script = textwrap.dedent("""\
        import importlib.abc
        import sys

        blocked = {"roughvol", "databento", "torch"}

        class BlockExternalLibraries(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in blocked:
                    raise ImportError(f"External dependency blocked: {fullname}")
                return None

        sys.meta_path.insert(0, BlockExternalLibraries())
        from rvlab.pricing import (
            BlackScholesParams, HestonParams, RoughHestonParams, price,
        )

        models = [
            BlackScholesParams(0.2),
            HestonParams(0.04, 0.05, 1.3, 0.25, -0.6),
            RoughHestonParams(0.04, 0.05, 1.3, 0.25, -0.6, 0.25),
        ]
        for model in models:
            value = price(model, forward=100, discount=0.99, strike=100, maturity=0.2)
            assert 0 < value < 99, (model, value)
        assert not blocked.intersection(name.split(".")[0] for name in sys.modules)
        print("self-contained pricing OK")
        """)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    result = subprocess.run(
        [sys.executable, "-c", script], env=environment, capture_output=True,
        text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "self-contained pricing OK"
