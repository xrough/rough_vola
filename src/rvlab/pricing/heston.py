"""Normalized Heston characteristic function and European vanilla prices."""

from __future__ import annotations

import numpy as np

from .black_scholes import black_scholes_price
from .fourier import price_from_cf
from .types import BlackScholesParams, HestonParams, PricingConfig, PricingError


def _complex_log1p(value):
    """Retain tiny real parts that NumPy's complex log1p can round away."""
    value = np.asarray(value, dtype=complex)
    result = np.log1p(value)
    small = np.abs(value) < 1e-4
    z = value[small]
    result[small] = z * (1.0 + z * (-0.5 + z * (1.0 / 3.0
                                                   + z * (-0.25 + z / 5.0))))
    return result


def _integrated_variance(params: HestonParams, maturity):
    maturity = np.asarray(maturity, dtype=float)
    if params.kappa == 0.0:
        return params.v0 * maturity
    decay_integral = -np.expm1(-params.kappa * maturity) / params.kappa
    return params.theta * maturity + (params.v0 - params.theta) * decay_integral


def _heston_riccati_basis(params, u, times):
    """Analytic D(t) and its time integral, including the short-time limit.

    The affine equation is D' = -.5*(u*u+iu) +
    (iu*rho*eta-kappa)*D + .5*eta**2*D**2. Its Taylor coefficients avoid
    cancellations in the characteristic exponent when both time and eta
    are small. The same basis supports time-dependent mean-reversion inputs.
    """
    z, t = np.broadcast_arrays(np.asarray(u, complex), np.asarray(times, float))
    try:
        eta2 = params.vol_of_vol**2
    except OverflowError as exc:
        raise PricingError("Heston Riccati coefficients exceed floating-point range") from exc
    q = z*z + 1j*z
    b = params.kappa - 1j * params.rho * params.vol_of_vol * z
    d = np.sqrt(b*b + eta2*q)
    d = np.where((d.real < 0) | ((d.real == 0) & (d.imag < 0)), -d, d)
    coefficient, primitive = np.zeros(z.shape, complex), np.zeros(z.shape, complex)
    active = (t > 0) & (q != 0)
    short = active & (np.maximum(np.abs(b*t), np.abs(d*t)) < 1e-4)
    small = active & ~short & (np.abs(d*t) < 1e-8)
    regular = active & ~short & ~small
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if np.any(short):
            tt, linear, forcing = t[short], -b[short], -0.5*q[short]
            second = linear*forcing
            third = linear*linear*forcing + eta2*forcing*forcing
            fourth = linear**3*forcing + 4*eta2*linear*forcing*forcing
            coefficient[short] = tt*(forcing + tt*(second/2 + tt*(third/6 + tt*fourth/24)))
            primitive[short] = tt*tt*(forcing/2 + tt*(second/6 + tt*(third/24 + tt*fourth/120)))
        if np.any(regular):
            bb, dd, tt, qq = (a[regular] for a in (b, d, t, q))
            minus, plus = bb-dd, bb+dd
            rational = np.abs(minus) < np.abs(plus)
            minus[rational] = -eta2 * qq[rational] / plus[rational]
            g = minus / plus
            one_minus_exp = -np.expm1(-dd*tt)
            coefficient[regular] = (minus / eta2 * one_minus_exp
                                    / (1 - g*np.exp(-dd*tt)))
            log_ratio = _complex_log1p(g * one_minus_exp / (1-g))
            primitive[regular] = (minus*tt - 2*log_ratio) / eta2
        if np.any(small):
            bb, tt, qq = (a[small] for a in (b, t, q))
            coefficient[small] = -0.5*qq*tt / (1 + 0.5*bb*tt)
            primitive[small] = (bb*tt - 2*_complex_log1p(0.5*bb*tt)) / eta2
    if not np.all(np.isfinite(coefficient)) or not np.all(np.isfinite(primitive)):
        raise PricingError("Heston Riccati basis produced nonfinite values")
    return coefficient, primitive


def heston_cf(params: HestonParams, u, maturity):
    """E[exp(i*u*log(S_T/F_T))], broadcasting frequency and year maturity.

    Uses the decaying-exponential Heston representation. Removable values at
    u=0 and u=-i are explicit, as is the deterministic-variance limit.
    """
    if not isinstance(params, HestonParams):
        raise TypeError("params must be HestonParams")
    u, t = np.broadcast_arrays(np.asarray(u, dtype=complex),
                               np.asarray(maturity, dtype=float))
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(t) & (t >= 0)):
        raise ValueError("CF frequencies must be finite and maturities finite and nonnegative")
    shape = u.shape
    u, t = u.reshape(-1), t.reshape(-1)
    output = np.ones(u.shape, dtype=complex)
    absorbing = params.v0 == 0 and (params.kappa == 0 or params.theta == 0)
    active = (t > 0) & (u != 0) & (u != -1j) & ~np.bool_(absorbing)
    if np.any(active):
        z, tenor = u[active], t[active]
        q = z * z + 1j * z
        if params.vol_of_vol == 0.0:
            with np.errstate(over="ignore", invalid="ignore"):
                output[active] = np.exp(-0.5 * q * _integrated_variance(params, tenor))
        else:
            coefficient, primitive = _heston_riccati_basis(params, z, tenor)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                output[active] = np.exp(params.kappa * params.theta * primitive
                                        + coefficient * params.v0)
    if not np.all(np.isfinite(output)):
        raise PricingError("Heston characteristic function produced nonfinite values")
    output = output.reshape(shape)
    return complex(output) if output.ndim == 0 else output


def heston_price(
    params: HestonParams, *, forward, discount, strike, maturity,
    is_call=True, config: PricingConfig | None = None,
):
    """Discounted prices per underlying unit, broadcasting contract inputs."""
    if not isinstance(params, HestonParams):
        raise TypeError("params must be HestonParams")
    if config is not None and not isinstance(config, PricingConfig):
        raise TypeError("config must be PricingConfig or None")
    if params.v0 == 0 and (params.kappa == 0 or params.theta == 0):
        return black_scholes_price(
            BlackScholesParams(volatility=0.0), forward=forward, discount=discount,
            strike=strike, maturity=maturity, is_call=is_call,
        )
    if params.vol_of_vol == 0.0:
        # Analytic deterministic mean reversion gives a lognormal terminal law.
        from ._validation import broadcast_inputs, scalar_or_array
        f, d, k, t, calls = broadcast_inputs(
            forward=forward, discount=discount, strike=strike,
            maturity=maturity, is_call=is_call,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            integrated = _integrated_variance(params, t)
            variance = np.divide(integrated, t, out=np.full_like(t, params.v0), where=t > 0)
        if not np.all(np.isfinite(variance)):
            raise PricingError("Deterministic Heston integrated variance is nonfinite")
        result = np.empty(t.shape, dtype=float)
        for level in np.unique(variance):
            selected = variance == level
            result[selected] = black_scholes_price(
                BlackScholesParams(volatility=float(np.sqrt(max(level, 0.0)))),
                forward=f[selected], discount=d[selected], strike=k[selected],
                maturity=t[selected], is_call=calls[selected],
            )
        return scalar_or_array(result)
    return price_from_cf(
        lambda u, t: heston_cf(params, u, t), reference_variance=params.v0,
        forward=forward, discount=discount, strike=strike,
        maturity=maturity, is_call=is_call, config=config,
    )


__all__ = ["heston_cf", "heston_price"]
