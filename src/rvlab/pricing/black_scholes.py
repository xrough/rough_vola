"""European Black prices with explicit forwards and discount factors."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from ._validation import broadcast_inputs, scalar_or_array
from .types import BlackScholesParams, PricingError


def black_scholes_price(params: BlackScholesParams, *, forward, discount, strike,
                        maturity, is_call=True):
    """Price calls or puts per underlying unit, including deterministic limits.

    F is the maturity-matched forward and D its discount factor. The caller
    supplies their economic conventions; neither interest nor dividend rates
    are guessed here. All quote inputs broadcast as NumPy arrays.
    """
    if not isinstance(params, BlackScholesParams):
        raise TypeError("params must be BlackScholesParams")
    f, d, k, t, call = broadcast_inputs(
        forward=forward, discount=discount, strike=strike,
        maturity=maturity, is_call=is_call)
    stddev = params.volatility * np.sqrt(t)
    # Price the OTM side directly to avoid subtracting two large ITM values.
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = np.divide(np.log(f) - np.log(k), stddev,
                       out=np.zeros_like(stddev), where=stddev > 0) + 0.5 * stddev
        d2 = d1 - stddev
        otm_call = d * (f * ndtr(d1) - k * ndtr(d2))
        otm_put = d * (k * ndtr(-d2) - f * ndtr(-d1))
    parity = d * (f - k)
    calls = np.where(f <= k, otm_call, otm_put + parity)
    puts = np.where(f >= k, otm_put, otm_call - parity)
    intrinsic = d * np.where(call, np.maximum(f - k, 0), np.maximum(k - f, 0))
    value = np.where(stddev > 0, np.where(call, calls, puts), intrinsic)
    if not np.all(np.isfinite(value)):
        raise PricingError("nonfinite Black-Scholes price; check input magnitudes")
    return scalar_or_array(np.maximum(value, 0.0))


__all__ = ["black_scholes_price"]
