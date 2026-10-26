"""
rvlab.evaluate.backtest
=======================
Does the rough correction survive contact with a hedge?

A forecasting improvement is not a trading result. This module builds delta
policies on simulated rough-volatility paths and runs them through a common
accounting ledger, so Black-Scholes delta and rough delta are compared on
*identical* paths, at identical rebalance times, under identical costs. Any one
of those held unequal is enough to manufacture a winner — which the project's
own README lists as limitation 8.3.

The ledger comes from `roughvol.experiments.model_comparison.hedge_backtest`;
this module supplies the policies and the cost sweep.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import compat
from ..models import blackscholes as bs
from ..models.rough import RoughVolParams, chi, psi


def bs_delta_policy(spot_paths: np.ndarray, time_grid: np.ndarray, strike: float,
                    maturity: float, sigma: float, is_call: bool = True,
                    rate: float = 0.0) -> np.ndarray:
    """Black-Scholes delta at every (path, time). The transparent baseline.

    Time to expiry is floored at one second so the terminal column is finite
    rather than NaN — the ledger needs a number there to liquidate against.
    """
    tau = np.maximum(maturity - np.asarray(time_grid, float), 1.0 / (365 * 24 * 3600))
    tau = np.broadcast_to(tau, spot_paths.shape)
    return bs.delta(sigma, spot_paths, strike, tau, is_call, rate)


def rough_delta_policy(spot_paths: np.ndarray, time_grid: np.ndarray, strike: float,
                       maturity: float, params: RoughVolParams = RoughVolParams(),
                       is_call: bool = True, rate: float = 0.0) -> np.ndarray:
    """Minimum-variance delta under the Bergomi-Guyon smile.

    Equals the BS delta plus vega times the smile's spot-sensitivity. Causal by
    construction: every entry uses only the spot and the time at that node.
    """
    from ..models.rough import rough_delta

    tau = np.maximum(maturity - np.asarray(time_grid, float), 1.0 / (365 * 24 * 3600))
    tau = np.broadcast_to(tau, spot_paths.shape)
    return rough_delta(strike, spot_paths, tau, params, is_call, rate)


def compare_hedge_policies(spot_paths: np.ndarray, time_grid: np.ndarray,
                           strike: float, maturity: float, premium: float,
                           sigma: float, params: RoughVolParams = RoughVolParams(),
                           is_call: bool = True, rate: float = 0.0,
                           transaction_cost_bps: float = 0.0,
                           return_ledger: bool = False):
    """Run BS delta, rough delta and no-hedge on the same paths. One row per policy.

    Includes an unhedged policy on purpose: it is the reference that says how
    much risk the hedging removed at all, and occasionally it wins on cost.

    Returns the per-policy summary, or `(summary, ledger)` when `return_ledger`
    is set. The pathwise ledger is returned rather than attached to
    `summary.attrs`, because `pd.concat` compares attrs values to decide whether
    to propagate them, and comparing two DataFrames with `==` raises.
    """
    hb = compat.roughvol_module("roughvol.experiments.model_comparison.hedge_backtest")

    policies = {
        "bs_delta": bs_delta_policy(spot_paths, time_grid, strike, maturity,
                                    sigma, is_call, rate),
        "rough_delta": rough_delta_policy(spot_paths, time_grid, strike, maturity,
                                          params, is_call, rate),
        "no_hedge": np.zeros_like(spot_paths),
    }

    result = hb.run_delta_hedge_backtest(
        spot_paths=spot_paths, time_grid=time_grid, policy_deltas=policies,
        premium=premium, strike=strike, is_call=is_call, rate=rate,
        config=hb.HedgeBacktestConfig(transaction_cost_bps=transaction_cost_bps),
    )
    summary = result.summary.copy()
    summary.attrs["transaction_cost_bps"] = transaction_cost_bps
    return (summary, result.ledger) if return_ledger else summary


def cost_sensitivity(spot_paths: np.ndarray, time_grid: np.ndarray, strike: float,
                     maturity: float, premium: float, sigma: float,
                     params: RoughVolParams = RoughVolParams(),
                     cost_bps=(0.0, 1.0, 5.0, 10.0, 25.0),
                     is_call: bool = True, rate: float = 0.0) -> pd.DataFrame:
    """Re-run the comparison across transaction-cost levels.

    The point: a hedge that rebalances more aggressively can win at zero cost
    and lose at five basis points. A single cost assumption is a choice that
    picks the winner, so report the curve, not a point.
    """
    frames = []
    for bps in cost_bps:
        s = compare_hedge_policies(spot_paths, time_grid, strike, maturity, premium,
                                   sigma, params, is_call, rate, bps)
        s = s.reset_index() if s.index.name else s.copy()
        s["cost_bps"] = bps
        frames.append(s)
    return pd.concat(frames, ignore_index=True)


def hedge_error_stats(errors) -> dict:
    """RMSE, MAE, std, and the 5% CVaR of a hedge-error distribution.

    CVaR (mean of the worst 5%) is reported alongside RMSE because a hedge is
    bought for its tail behaviour. Two policies with equal RMSE and different
    CVaR are not equally good.
    """
    e = np.asarray(errors, float).ravel()
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {"rmse": np.nan, "mae": np.nan, "std": np.nan, "cvar_5pct": np.nan, "n": 0}
    tail = e[e <= np.quantile(e, 0.05)]
    return {
        "rmse": float(np.sqrt(np.mean(e**2))), "mae": float(np.mean(np.abs(e))),
        "std": float(np.std(e)), "cvar_5pct": float(tail.mean()) if tail.size else np.nan,
        "n": int(e.size),
    }


def smile_sensitivity(params: RoughVolParams, spot: float, strike: float,
                      T: float) -> dict:
    """The two terms of the rough delta, so their sizes can be compared directly.

    Worth checking the maturity behaviour rather than assuming it. The skew
    psi ~ T^(H-1/2) explodes as T -> 0, but vega ~ sqrt(T) vanishes, and the
    product scales as T^H — so at fixed log-moneyness the correction actually
    *shrinks* at short maturity. What explodes at the short end is the smile
    slope in vol space, not the delta adjustment in price space. Notebook 10
    plots both so the distinction is visible.
    """
    k = float(np.log(strike / spot))
    dsigma_dS = -(psi(T, params) + chi(T, params) * k) / spot
    from ..models.rough import bergomi_guyon_smile
    sigma_k = float(bergomi_guyon_smile(strike, spot, T, params))
    return {
        "T": T, "log_moneyness": k, "smile_vol": sigma_k,
        "bs_delta": float(bs.delta(sigma_k, spot, strike, T, True, 0.0)),
        "vega": float(bs.vega(sigma_k, spot, strike, T, 0.0)),
        "dsigma_dS": float(dsigma_dS),
        "correction": float(bs.vega(sigma_k, spot, strike, T, 0.0) * dsigma_dS),
    }


__all__ = [
    "bs_delta_policy", "rough_delta_policy", "compare_hedge_policies",
    "cost_sensitivity", "hedge_error_stats", "smile_sensitivity",
]
