"""Rough Heston pricing from a batched implicit fractional-Adams solver.

K(t)=t**(H-.5)/Gamma(H+.5). Model coefficients are inside this convolution;
H=.5 recovers the ordinary Heston model with identical parameter values.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
from scipy.special import gamma

from .fourier import price_from_cf
from .heston import heston_cf, heston_price
from .types import PricingConfig, PricingError, RoughHestonParams


def _rhs(params: RoughHestonParams, u, h):
    return (-0.5 * (u * u + 1j * u)
            + (1j * u * params.rho * params.vol_of_vol - params.kappa) * h
            + 0.5 * params.vol_of_vol**2 * h * h)


def _implicit_riccati_step(history, forcing, linear, quadratic, scale):
    """Solve the Adams corrector's quadratic on its finite, small-step branch.

    The equation is ``h = history + scale*(forcing + linear*h + quadratic*h²)``.
    The discriminant root has nonnegative real part, continuing the physical
    root from small time steps. Rationalising it avoids cancellation as
    vol-of-vol tends to zero; the direct expression is used when its numerator
    is larger. This is an implicit correction, rather than evaluating the
    nonlinear RHS at an explicit predictor.
    """
    q = 1.0 - scale * linear
    r = history + scale * forcing
    if quadratic == 0.0:
        return r / q
    root = np.sqrt(q * q - 4.0 * scale * quadratic * r)
    denominator = q + root
    rationalise = np.abs(denominator) >= np.abs(q - root)
    result = np.empty_like(r, dtype=complex)
    np.divide(2.0 * r, denominator, out=result, where=rationalise)
    np.divide(q - root, 2.0 * scale * quadratic, out=result, where=~rationalise)
    # The exact normalising frequencies have a zero solution even if a very
    # coarse, stiff step gives the quadratic a second nonzero root.
    return np.where(r == 0.0, 0.0, result)


def solve_fractional_riccati(
    params: RoughHestonParams, u, maturity, *, n_steps: int,
):
    """Return times and fractional Riccati solutions with the time axis first.

    All Fourier frequencies advance together. The initial RHS is F(u,0),
    which is nonzero for ordinary frequencies despite the zero initial state.
    A piecewise-linear product-integration corrector is solved implicitly,
    avoiding the explicit predictor's high-frequency stability restriction.
    Time refinement is still required to assess its discretisation error.
    Complexity is O(n_steps**2 * u.size) with O(n_steps * u.size) storage.
    """
    if not isinstance(params, RoughHestonParams):
        raise TypeError("params must be RoughHestonParams")
    if isinstance(n_steps, bool) or not isinstance(n_steps, Integral) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    t = np.asarray(maturity, dtype=float)
    if t.ndim != 0 or not np.isfinite(t) or t < 0:
        raise ValueError("solver maturity must be a finite nonnegative scalar")
    u = np.asarray(u, dtype=complex)
    if not np.all(np.isfinite(u)):
        raise ValueError("CF frequencies must be finite")
    shape = u.shape
    frequencies = u.reshape(-1)
    if t == 0:
        return np.array([0.0]), np.zeros((1,) + shape, dtype=complex)
    alpha = params.hurst + 0.5
    dt = float(t) / n_steps
    times = np.linspace(0.0, float(t), n_steps + 1)
    values = np.zeros((n_steps + 1, frequencies.size), dtype=complex)
    rhs_values = np.zeros_like(values)
    with np.errstate(over="ignore", invalid="ignore"):
        try:
            rhs_values[0] = _rhs(params, frequencies, values[0])
        except OverflowError as exc:
            raise PricingError(
                "Fractional Riccati coefficients exceed floating-point range"
            ) from exc
    if not np.all(np.isfinite(rhs_values[0])):
        raise PricingError("Fractional Riccati initial forcing produced nonfinite values")
    corrector_scale = dt**alpha / gamma(alpha + 2.0)
    lag = np.arange(n_steps, dtype=float)
    history_weights = corrector_scale * (
        (lag + 2.0)**(alpha + 1.0) + lag**(alpha + 1.0)
        - 2.0 * (lag + 1.0)**(alpha + 1.0)
    )
    initial_weights = corrector_scale * (
        lag**(alpha + 1.0) - (lag - alpha) * (lag + 1.0)**alpha
    )
    forcing = rhs_values[0]
    linear = 1j * frequencies * params.rho * params.vol_of_vol - params.kappa
    quadratic = 0.5 * params.vol_of_vol**2
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for step in range(n_steps):
            history = initial_weights[step] * rhs_values[0]
            if step:
                history = history + np.sum(
                    history_weights[step - 1::-1, None] * rhs_values[1:step + 1],
                    axis=0,
                )
            values[step + 1] = _implicit_riccati_step(
                history, forcing, linear, quadratic, corrector_scale,
            )
            rhs_values[step + 1] = _rhs(params, frequencies, values[step + 1])
            if (not np.all(np.isfinite(values[step + 1]))
                    or not np.all(np.isfinite(rhs_values[step + 1]))):
                raise PricingError(
                    f"Fractional Riccati solver diverged at step {step + 1}; "
                    "increase riccati_steps or revise the parameter range"
                )
    return times, values.reshape((n_steps + 1,) + shape)


def _terminal_integral(values, maturity: float, order: float):
    """Product-integration of the piecewise-linear Riccati solution."""
    if order == 0.0:
        return values[-1]
    n = values.shape[0] - 1
    lag = np.arange(n - 1, -1, -1, dtype=float)
    weights = np.ones(n + 1, dtype=float)
    weights[:-1] = ((lag + 2.0)**(order + 1.0) + lag**(order + 1.0)
                    - 2.0 * (lag + 1.0)**(order + 1.0))
    weights[0] = ((n - 1)**(order + 1.0)
                  - (n - 1 - order) * n**order)
    return ((maturity / n)**order / gamma(order + 2.0)
            * np.sum(weights[:, None] * values, axis=0))


def rough_heston_cf(
    params: RoughHestonParams, u, maturity, *, config: PricingConfig | None = None,
):
    """Normalized log-forward-return CF, broadcasting frequency and maturity.

    This is a time-zero model with constant initial variance and no historical
    variance-factor state. It is not conditional rough-model valuation.
    """
    if not isinstance(params, RoughHestonParams):
        raise TypeError("params must be RoughHestonParams")
    if config is not None and not isinstance(config, PricingConfig):
        raise TypeError("config must be PricingConfig or None")
    if params.hurst == 0.5:
        return heston_cf(params, u, maturity)
    config = PricingConfig() if config is None else config
    u, t = np.broadcast_arrays(np.asarray(u, dtype=complex),
                               np.asarray(maturity, dtype=float))
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(t) & (t >= 0)):
        raise ValueError("CF frequencies must be finite and maturities finite and nonnegative")
    shape = u.shape
    u, t = u.reshape(-1), t.reshape(-1)
    result = np.ones(u.shape, dtype=complex)
    absorbing = params.v0 == 0 and (params.kappa == 0 or params.theta == 0)
    active = (t > 0) & (u != 0) & (u != -1j) & ~np.bool_(absorbing)
    if (params.vol_of_vol == 0
            and (params.kappa == 0 or params.theta == params.v0)):
        with np.errstate(over="ignore", invalid="ignore"):
            result[active] = np.exp(-0.5 * (u[active]**2 + 1j * u[active])
                                    * params.v0 * t[active])
    else:
        for tenor in np.unique(t[active]):
            selected = active & (t == tenor)
            _, values = solve_fractional_riccati(
                params, u[selected], float(tenor), n_steps=config.riccati_steps,
            )
            ordinary_integral = _terminal_integral(values, float(tenor), 1.0)
            fractional_integral = _terminal_integral(values, float(tenor), 0.5 - params.hurst)
            with np.errstate(over="ignore", invalid="ignore"):
                result[selected] = np.exp(params.kappa * params.theta * ordinary_integral
                                          + params.v0 * fractional_integral)
    if not np.all(np.isfinite(result)):
        raise PricingError("Rough Heston characteristic function produced nonfinite values")
    result = result.reshape(shape)
    return complex(result) if result.ndim == 0 else result


def rough_heston_price(
    params: RoughHestonParams, *, forward, discount, strike, maturity,
    is_call=True, config: PricingConfig | None = None,
):
    """Batched European prices with explicit fractional and Fourier controls."""
    if not isinstance(params, RoughHestonParams):
        raise TypeError("params must be RoughHestonParams")
    if config is not None and not isinstance(config, PricingConfig):
        raise TypeError("config must be PricingConfig or None")
    if params.hurst == 0.5:
        return heston_price(params, forward=forward, discount=discount,
                            strike=strike, maturity=maturity,
                            is_call=is_call, config=config)
    config = PricingConfig() if config is None else config
    return price_from_cf(
        lambda u, t: rough_heston_cf(params, u, t, config=config),
        reference_variance=params.v0, forward=forward, discount=discount,
        strike=strike, maturity=maturity, is_call=is_call, config=config,
    )


__all__ = ["solve_fractional_riccati", "rough_heston_cf", "rough_heston_price"]
