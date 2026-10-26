"""
rvlab.features.realized
=======================
Realized-volatility features, and the variance risk premium that pairs them with
implied vol.

The HAR decomposition (Corsi 2009) is the workhorse: regress tomorrow's realized
variance on its own averages over one day, one week and one month. It is a
*long-memory* model built from three short-memory terms, which is precisely the
econometric fact that rough volatility offers a different explanation for — so
HAR is both a baseline and a rival story, not just a feature block.

    VRP = implied variance - expected realized variance

A positive VRP is the compensation an option seller earns for bearing variance
risk. It is the economic reason the strategy side of this project sells options
at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MINUTES_PER_DAY = 390


def log_returns(prices, clip: float | None = 0.2) -> np.ndarray:
    """Log returns with an optional absolute clip to kill bad-print outliers."""
    p = np.asarray(prices, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.diff(np.log(np.where(p > 0, p, np.nan)), prepend=np.nan)
    return np.clip(r, -clip, clip) if clip else r


def realized_variance(returns, window: int, annualize_from: int = MINUTES_PER_DAY,
                      min_periods: int | None = None) -> pd.Series:
    """Rolling annualized realized variance from intraday returns.

    `annualize_from` is the number of return observations in a trading day, so
    1-minute bars use 390. Getting this wrong rescales every variance in the
    study by a constant — one of the ten limitations the project README lists.
    """
    s = pd.Series(np.asarray(returns, dtype=float))
    rv = s.pow(2).rolling(window, min_periods=min_periods or max(2, window // 2)).mean()
    return rv * annualize_from * TRADING_DAYS


def realized_vol(returns, window: int, **kwargs) -> pd.Series:
    """Square root of `realized_variance` — annualized realized volatility."""
    return np.sqrt(realized_variance(returns, window, **kwargs))


def add_har_components(df: pd.DataFrame, rv_col: str = "rv",
                       windows=(1, 5, 22), shift: int = 1,
                       prefix: str = "har") -> pd.DataFrame:
    """Append HAR daily/weekly/monthly averages of realized variance.

    Every component is shifted by `shift` so it is strictly backward-looking:
    the "daily" term at t is the realized variance *ending at t-1*. Without that
    shift a HAR regression scores beautifully and forecasts nothing.
    """
    out = df.copy()
    names = {1: "d", 5: "w", 22: "m"}
    for w in windows:
        label = names.get(w, str(w))
        out[f"{prefix}_{label}"] = (
            out[rv_col].rolling(w, min_periods=1).mean().shift(shift))
    out.attrs.update(df.attrs)
    return out


def variance_risk_premium(implied_var, realized_var) -> np.ndarray:
    """Implied minus realized variance, in the same annualized units.

    Both arguments must be *variances*, not vols, and both annualized the same
    way. Mixing a total variance with an annualized one is the unit error the
    project README flags as limitation 8.5.
    """
    return np.asarray(implied_var, dtype=float) - np.asarray(realized_var, dtype=float)


def add_vrp(df: pd.DataFrame, implied_var_col: str = "atm_total_var",
            realized_var_col: str = "rv", annualize_implied_by_T: bool = True,
            T_col: str = "T", out_col: str = "vrp") -> pd.DataFrame:
    """Append a VRP column, converting total variance to annualized first.

    `atm_total_var` in this project is sigma^2 * T (total, not annualized), so
    dividing by T is required before it can be compared with an annualized
    realized variance. That conversion is the whole reason this helper exists.
    """
    out = df.copy()
    iv = out[implied_var_col].to_numpy(dtype=float)
    if annualize_implied_by_T:
        T = out[T_col].to_numpy(dtype=float)
        iv = np.where(T > 0, iv / T, np.nan)
    out[out_col] = variance_risk_premium(iv, out[realized_var_col].to_numpy(dtype=float))
    out.attrs.update(df.attrs)
    return out


def realized_variance_blocks(close: pd.Series, block_size: int = 30,
                             annualization: float = TRADING_DAYS * MINUTES_PER_DAY,
                             session_aware: bool = True, *,
                             deseasonalize_intraday: bool = False) -> pd.DataFrame:
    """Non-overlapping variance blocks with a stable schema and local sessions.

    The default uses observed returns without sample-wide seasonal estimation,
    so appending future prices cannot change completed blocks. Optional seasonal
    adjustment is an offline full-sample diagnostic, not a causal feature.
    ``realized_variance`` aliases annualized variance; ``block_end`` duplicates
    the timestamp index for older callers. Raw variance and volatility retain
    their distinct names and units.
    """
    from roughvol.analytics.roughness import realized_variance_blocks as blocks
    result = blocks(close, block_size=block_size, annualization=annualization,
                    session_aware=session_aware, deseasonalize_intraday=deseasonalize_intraday)
    result = result.copy()
    result["block_end"] = result.index
    result["realized_variance"] = result["annualized_variance"]
    return result


__all__ = [
    "log_returns", "realized_variance", "realized_vol", "add_har_components",
    "variance_risk_premium", "add_vrp", "realized_variance_blocks",
    "TRADING_DAYS", "MINUTES_PER_DAY",
]
