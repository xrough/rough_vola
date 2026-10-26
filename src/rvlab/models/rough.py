"""Legacy illustrative rough-smile and scaling interfaces.

These formulas preserve the historical ``rvlab.models`` API. They are simple
smile/scaling approximations and are separate from the validated model-specific
prices in ``rvlab.pricing``. No universal maturity power law or empirical Hurst
identification is established by this module.

The smile uses ``sigma(k)=sigma_ATM+psi*k+chi*k**2/2`` with log moneyness
``k=log(K/S)``. Its legacy sensitivity differentiates that stipulated smile map;
it is not a minimum-variance hedge or an optimal rough-state sensitivity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import blackscholes as bs


@dataclass(frozen=True)
class RoughVolParams:
    """Parameters for the retained illustrative rough-smile approximation."""

    H: float = 0.10        # Hurst exponent; H < 0.5 is "rough"
    eta: float = 1.50      # vol-of-vol
    rho: float = -0.70     # spot-vol correlation
    xi0: float = 0.0625    # flat forward variance at t=0 (= 0.25^2)

    @property
    def sigma_atm(self) -> float:
        return math.sqrt(self.xi0)


# ── 1. the scaling laws ────────────────────────────────────────────────────────
def skew_exponent(hurst: float) -> float:
    """The maturity exponent of the skew: H - 1/2. Negative whenever H < 1/2."""
    return hurst - 0.5


def curvature_exponent(hurst: float) -> float:
    """The maturity exponent of the butterfly: 2H - 1."""
    return 2.0 * hurst - 1.0


def structural_alpha(rr25, T, atm_iv, hurst: float):
    """Invert RR25 = alpha * T^(H-0.5) * atm_iv for alpha.

    A maturity-invariant alpha is a diagnostic implication of the stipulated
    scaling relation, rather than sufficient evidence for an identified H.
    """
    T = np.asarray(T, dtype=float)
    denom = T ** skew_exponent(hurst) * np.asarray(atm_iv, dtype=float)
    return np.where(np.abs(denom) > 1e-12, np.asarray(rr25, dtype=float) / denom, np.nan)


def structural_gamma(bf25, T, atm_total_var, hurst: float):
    """Invert BF25 = gamma * T^(2H-1) * atm_total_var for gamma."""
    T = np.asarray(T, dtype=float)
    denom = T ** curvature_exponent(hurst) * np.asarray(atm_total_var, dtype=float)
    return np.where(np.abs(denom) > 1e-12, np.asarray(bf25, dtype=float) / denom, np.nan)


def implied_hurst_from_slope(slope: float) -> float:
    """H implied by an observed log|RR25| vs log(T) regression slope.

    Because slope = H - 1/2, this is just slope + 1/2. Kept as a named function
    to make the assumed slope-to-H mapping explicit. Its use requires an
    independently justified scaling approximation.
    """
    return slope + 0.5


# ── 2. the closed-form smile ───────────────────────────────────────────────────
def psi(T, params: RoughVolParams = RoughVolParams()):
    """Leading skew coefficient of the illustrative smile approximation."""
    T = np.asarray(T, dtype=float)
    return (params.rho * params.eta * math.gamma(params.H + 0.5)
            / (2.0 * math.sqrt(math.pi)) * T ** (params.H - 0.5))


def chi(T, params: RoughVolParams = RoughVolParams()):
    """The leading curvature coefficient chi(T) = psi(T)^2 / sigma_ATM."""
    return psi(T, params) ** 2 / params.sigma_atm


def bergomi_guyon_smile(K, S, T, params: RoughVolParams = RoughVolParams()):
    """Second-order implied vol at strike K. Broadcasts over K and T.

    Floored at 0.001 because the quadratic approximation can become negative
    in the wings. The floor is a modelling convenience, not an accuracy check.
    """
    k = np.log(np.asarray(K, dtype=float) / np.asarray(S, dtype=float))
    sig = params.sigma_atm + psi(T, params) * k + 0.5 * chi(T, params) * k**2
    return np.maximum(sig, 0.001)


def rough_delta(K, S, T, params: RoughVolParams = RoughVolParams(),
                is_call: bool = True, rate: float | None = None):
    """Legacy smile-map derivative: BS forward delta plus a skew correction.

        Delta_rough = Delta_BS(sigma_K) + Vega(sigma_K) * dsigma_K/dS
        dsigma_K/dS = -(psi + chi*k) / S

    The formula evaluates the legacy forward helper at the supplied S and holds
    the smile parameters fixed. It is separate from the spot deltas returned
    by rvlab.pricing; it establishes no rough-state or optimal-hedge sensitivity.
    """
    from ..config import RATE
    rate = RATE if rate is None else rate

    S_a = np.asarray(S, dtype=float)
    k = np.log(np.asarray(K, dtype=float) / S_a)
    sigma_k = bergomi_guyon_smile(K, S, T, params)

    bs_d = bs.delta(sigma_k, S_a, K, T, is_call, rate)
    bs_v = bs.vega(sigma_k, S_a, K, T, rate)
    dsigma_dS = -(psi(T, params) + chi(T, params) * k) / S_a
    return bs_d + bs_v * dsigma_dS


def smile_grid(params: RoughVolParams = RoughVolParams(),
               moneyness=None, maturities=None, spot: float = 1.0):
    """Tidy (T, k, implied_vol) frame — the input to `charts.term_structure_plot`."""
    import pandas as pd

    moneyness = np.linspace(-0.15, 0.15, 61) if moneyness is None else np.asarray(moneyness)
    maturities = np.array([7, 14, 30, 60, 120]) / 365.0 if maturities is None \
        else np.asarray(maturities)

    kk, TT = np.meshgrid(moneyness, maturities)
    iv = bergomi_guyon_smile(spot * np.exp(kk), spot, TT, params)
    return pd.DataFrame({"T": TT.ravel(), "k": kk.ravel(), "implied_vol": iv.ravel()})


__all__ = [
    "RoughVolParams", "skew_exponent", "curvature_exponent", "structural_alpha",
    "structural_gamma", "implied_hurst_from_slope", "psi", "chi",
    "bergomi_guyon_smile", "rough_delta", "smile_grid",
]
