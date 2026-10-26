"""Descriptive roughness features under explicit scaling assumptions.

``hurst_structure_function`` estimates a time-series structure-function slope
with an optional logarithmic transformation and session grouping.
``skew_term_structure_fit`` fits a cross-sectional risk-reversal term structure
and reports the assumed mapping ``H=slope+1/2``. Physical-measure proxies and
risk-neutral smile slopes need not agree. Proxy noise, finite samples, lag
choice and an unjustified scaling approximation can all distort these results;
they do not by themselves identify a latent rough-volatility model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import compat


@dataclass
class HurstEstimate:
    """Result of a structure-function fit."""

    hurst: float
    intercept: float
    r_squared: float
    lags: np.ndarray
    structure_function: np.ndarray
    n_obs: int

    def __str__(self) -> str:
        return (f"H = {self.hurst:.4f}  (R2 = {self.r_squared:.3f}, "
                f"{len(self.lags)} lags, {self.n_obs:,} obs)")


def hurst_structure_function(series, min_lag: int = 1, max_lag: int = 32,
                             log_transform: bool = True) -> HurstEstimate:
    """Estimate H from E|X(t+lag) - X(t)| ~ lag^H.

    `series` should be a volatility path. With `log_transform=True` the
    increments are taken of log sigma, which is the quantity the rough
    literature models as fractional Brownian motion.

    Uses the bundled grouped estimator after the requested transformation.
    A DatetimeIndex retains its local session dates; increments never cross
    those dates. Regular spacing within each session is the caller’s responsibility.
    """
    from roughvol.analytics.roughness import estimate_hurst_exponent

    s = pd.Series(series, copy=False).astype(float)
    s = s.loc[np.isfinite(s.to_numpy())]
    if log_transform and (s <= 0).any():
        raise ValueError("log-transform roughness inputs must be strictly positive")
    values = np.log(s.to_numpy()) if log_transform else s.to_numpy()
    groups = s.index.normalize() if isinstance(s.index, pd.DatetimeIndex) else None
    longest = int(pd.Series(1, index=s.index).groupby(groups).size().max()) if groups is not None and len(s) else len(s)
    actual_max = min(int(max_lag), longest//4)
    if actual_max <= min_lag:
        raise ValueError("insufficient within-session observations for the requested lags")
    estimate = estimate_hurst_exponent(values, min_lag=min_lag, max_lag=actual_max, groups=groups)
    return HurstEstimate(hurst=float(estimate.hurst), intercept=float(estimate.intercept),
        r_squared=float(estimate.r_squared), lags=np.asarray(estimate.lags),
        structure_function=np.asarray(estimate.structure_function), n_obs=len(s))


def skew_term_structure_fit(cross_section: pd.DataFrame, skew_col: str = "rr25",
                            T_col: str = "T", min_points: int = 3) -> dict | None:
    """Fit log|RR25| ~ beta * log(T) on one timestamp's term structure.

    Returns {slope, intercept, r_squared, implied_hurst, n_points} or None when
    the slice has too few usable maturities. `implied_hurst = slope + 0.5`.
    """
    df = cross_section[[T_col, skew_col]].dropna()
    df = df[(df[T_col] > 0) & (df[skew_col].abs() > 1e-8)]
    if len(df) < min_points:
        return None

    x = np.log(df[T_col].to_numpy(dtype=float))
    y = np.log(np.abs(df[skew_col].to_numpy(dtype=float)))
    if np.ptp(x) < 1e-9:                    # all one maturity: slope undefined
        return None

    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - pred) ** 2)) / ss_tot if ss_tot > 1e-12 else np.nan

    return {"slope": float(slope), "intercept": float(intercept), "r_squared": r2,
            "implied_hurst": float(slope) + 0.5, "n_points": len(df)}


def skew_scaling_panel(df: pd.DataFrame, skew_col: str = "rr25",
                       min_points: int = 3, max_timestamps: int | None = None
                       ) -> pd.DataFrame:
    """Run `skew_term_structure_fit` at every timestamp; return one row each.

    This is the Gate 1 computation. `max_timestamps` subsamples for speed —
    notebooks use it to stay inside the 60-second budget, and say so.
    """
    stamps = df["ts"].drop_duplicates()
    if max_timestamps and len(stamps) > max_timestamps:
        stamps = stamps.iloc[:: max(1, len(stamps) // max_timestamps)]
    keep = df[df["ts"].isin(set(stamps))]

    rows = []
    for ts, grp in keep.groupby("ts", observed=True, sort=True):
        fit = skew_term_structure_fit(grp, skew_col=skew_col, min_points=min_points)
        if fit:
            rows.append({"ts": ts, **fit})

    out = pd.DataFrame(rows)
    out.attrs["skew_col"] = skew_col
    return out


def summarise_skew_scaling(fits: pd.DataFrame) -> pd.Series:
    """Headline numbers for a skew-scaling panel — comparable to Gate 1.

    Reports the slope's coefficient of variation, because a power law that fits
    well at every instant but whose exponent wanders is a description, not a
    structural constant.
    """
    slope = fits["slope"].dropna()
    return pd.Series({
        "n_timestamps": len(fits),
        "median_slope": float(slope.median()),
        "mean_r_squared": float(fits["r_squared"].mean()),
        "median_r_squared": float(fits["r_squared"].median()),
        "slope_cv": float(slope.std() / abs(slope.mean())) if slope.mean() else np.nan,
        "median_implied_hurst": float(fits["implied_hurst"].median()),
    })


__all__ = [
    "HurstEstimate", "hurst_structure_function", "skew_term_structure_fit",
    "skew_scaling_panel", "summarise_skew_scaling",
]
