"""Legacy vectorized European Black-Scholes helpers in forward form.

Every formula takes F, a supplied forward price, rather than observed spot.
Discounting uses the explicitly supplied scalar rate. The helpers retain their
historical interface; model-specific pricing with explicit discount factors and
validated inputs lives in ``rvlab.pricing``. They supply no American exercise
model and do not infer spot from American option parity.

All functions broadcast over NumPy arrays. ``delta`` is a forward derivative,
not a stock delta.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from ..config import RATE

SQRT_2PI = np.sqrt(2.0 * np.pi)


def d1d2(sigma, F, K, T):
    """The two Black-Scholes arguments. Returns (d1, d2)."""
    sigma = np.asarray(sigma, dtype=float)
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    v = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * v**2) / v
    return d1, d1 - v


def price(sigma, F, K, T, is_call=True, rate: float = RATE):
    """Undiscounted-forward Black-Scholes price, discounted by exp(-rate*T)."""
    d_1, d_2 = d1d2(sigma, F, K, T)
    disc = np.exp(-rate * np.asarray(T, dtype=float))
    call = disc * (F * ndtr(d_1) - K * ndtr(d_2))
    put = disc * (K * ndtr(-d_2) - F * ndtr(-d_1))
    return np.where(is_call, call, put)


def delta(sigma, F, K, T, is_call=True, rate: float = RATE):
    """dPrice/dF. Note this is the *forward* delta, not the spot delta."""
    d_1, _ = d1d2(sigma, F, K, T)
    disc = np.exp(-rate * np.asarray(T, dtype=float))
    return disc * np.where(is_call, ndtr(d_1), ndtr(d_1) - 1.0)


def vega(sigma, F, K, T, rate: float = RATE):
    """dPrice/dsigma, per 1.00 of vol (divide by 100 for a 'per vol point' vega)."""
    d_1, _ = d1d2(sigma, F, K, T)
    T = np.asarray(T, dtype=float)
    return np.exp(-rate * T) * F * np.exp(-0.5 * d_1**2) / SQRT_2PI * np.sqrt(T)


def gamma(sigma, F, K, T, rate: float = RATE):
    """d2Price/dF2."""
    d_1, _ = d1d2(sigma, F, K, T)
    T = np.asarray(T, dtype=float)
    return (np.exp(-rate * T) * np.exp(-0.5 * d_1**2)
            / (SQRT_2PI * F * np.asarray(sigma, dtype=float) * np.sqrt(T)))


def implied_vol(prices, F, K, T, is_call=True, rate: float = RATE,
                lo: float = 1e-4, hi: float = 4.0, n_iter: int = 60):
    """Implied volatility by vectorised bisection. NaN outside no-arbitrage bounds.

    Bisection rather than Newton on purpose: it cannot diverge, it needs no
    derivative, and 60 halvings of [1e-4, 4.0] reach machine epsilon. On an
    option chain the robustness matters more than the iteration count — a
    Newton solver fails exactly on the deep wings you care about.
    """
    prices = np.atleast_1d(np.asarray(prices, dtype=float))
    K = np.broadcast_to(np.asarray(K, dtype=float), prices.shape)
    is_call = np.broadcast_to(np.asarray(is_call), prices.shape)
    F_arr = np.broadcast_to(np.asarray(F, dtype=float), prices.shape)
    T_arr = np.broadcast_to(np.asarray(T, dtype=float), prices.shape)

    disc = np.exp(-rate * T_arr)
    fwd_disc, K_disc = F_arr * disc, K * disc
    lower = np.where(is_call, np.maximum(0.0, fwd_disc - K_disc),
                     np.maximum(0.0, K_disc - fwd_disc))
    upper = np.where(is_call, fwd_disc, K_disc)
    feasible = (prices > lower + 1e-12) & (prices < upper - 1e-12)

    lo_a = np.full(prices.shape, lo)
    hi_a = np.full(prices.shape, hi)
    for _ in range(n_iter):
        mid = 0.5 * (lo_a + hi_a)
        too_high = price(mid, F_arr, K, T_arr, is_call, rate) > prices
        hi_a = np.where(too_high, mid, hi_a)
        lo_a = np.where(too_high, lo_a, mid)

    out = np.where(feasible, 0.5 * (lo_a + hi_a), np.nan)
    return out if out.size > 1 else float(out[0])


def total_variance(sigma, T):
    """sigma^2 * T — the quantity that is additive in maturity, unlike sigma."""
    return np.asarray(sigma, dtype=float) ** 2 * np.asarray(T, dtype=float)


__all__ = ["d1d2", "price", "delta", "vega", "gamma", "implied_vol", "total_variance"]
