"""Batched Lewis inversion for characteristic functions of log(S_T / F_T).

The half-moment contour needs one characteristic-function grid per maturity.
A Black--Scholes control variate removes the known leading term and prices
calls and puts directly, avoiding put-call subtraction in the wings. Fixed
quadrature settings are a numerical budget; refinement is checked by callers.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import numpy as np

from ._validation import broadcast_inputs, scalar_or_array
from .black_scholes import black_scholes_price
from .types import BlackScholesParams, PricingConfig, PricingError


@lru_cache(maxsize=32)
def _quadrature(nodes: int, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    locations, weights = np.polynomial.legendre.leggauss(nodes)
    locations = 0.5 * cutoff * (locations + 1.0)
    weights = 0.5 * cutoff * weights
    locations.setflags(write=False)
    weights.setflags(write=False)
    return locations, weights


def price_from_cf(
    characteristic_function: Callable[[np.ndarray, float], np.ndarray],
    *,
    reference_variance: float,
    forward,
    discount,
    strike,
    maturity,
    is_call=True,
    config: PricingConfig | None = None,
):
    """Invert a normalized log-return CF, sharing its grid across strikes.

    ``characteristic_function(u, T)`` must return E[exp(i*u*log(S_T/F_T))].
    Rates and dividends have already been absorbed into forward and discount.
    """
    if config is not None and not isinstance(config, PricingConfig):
        raise TypeError("config must be PricingConfig or None")
    config = PricingConfig() if config is None else config
    f, d, k, t, calls = broadcast_inputs(
        forward=forward, discount=discount, strike=strike,
        maturity=maturity, is_call=is_call,
    )
    shape = f.shape
    f, d, k, t, calls = (a.reshape(-1) for a in (f, d, k, t, calls))
    result = np.asarray(black_scholes_price(
        BlackScholesParams(volatility=float(np.sqrt(reference_variance))),
        forward=f, discount=d, strike=k, maturity=t, is_call=calls,
    ), dtype=float).copy()
    nodes, weights = _quadrature(config.quadrature_nodes, config.fourier_cutoff)
    contour = nodes - 0.5j

    for tenor in np.unique(t[t > 0.0]):
        selected = t == tenor
        cf = np.asarray(characteristic_function(contour, float(tenor)), dtype=complex)
        if cf.shape != contour.shape or not np.all(np.isfinite(cf)):
            raise PricingError("Characteristic function returned invalid Fourier values")
        reference_cf = np.exp(-0.5 * reference_variance * tenor * (nodes**2 + 0.25))
        integrand = weights * (cf - reference_cf) / (nodes**2 + 0.25)
        phase = np.exp(1j * np.outer(np.log(f[selected] / k[selected]), nodes))
        integral = np.real(phase @ integrand)
        result[selected] -= d[selected] * np.sqrt(f[selected] * k[selected]) / np.pi * integral

    if not np.all(np.isfinite(result)):
        raise PricingError("Fourier inversion produced nonfinite option prices")
    lower = d * np.maximum(np.where(calls, f - k, k - f), 0.0)
    upper = d * np.where(calls, f, k)
    tolerance = 1e-10 * np.maximum(1.0, d * np.maximum(f, k))
    if np.any((result < lower - tolerance) | (result > upper + tolerance)):
        raise PricingError(
            "Fourier prices violate European option bounds; refine the numerical settings"
        )
    result = np.minimum(np.maximum(result, lower), upper)
    return scalar_or_array(result.reshape(shape))


__all__ = ["price_from_cf"]
