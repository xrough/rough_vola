"""
rvlab.features.smile
====================
The smile observables and what they mean.

Three numbers summarise an option smile at one maturity:

    atm_iv  the level      — how expensive is volatility
    RR25    the slope      — IV(25d call) - IV(25d put); negative = crash fear
    BF25    the curvature  — mean(25d wings) - ATM; positive = fat tails priced

Everything downstream in this series is a function of those three plus the
maturity T. The structural coefficients alpha and gamma are RR25 and BF25 with
the rough-model maturity scaling divided out, so that a *correct* Hurst exponent
makes them maturity-invariant.

Chain-level extraction (raw quotes -> these numbers) is delegated to the
project's existing solver in demo/python/research/shared/smile_pipeline.py, so
notebooks and the research gates cannot disagree about what an IV is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import compat, config
from ..models import rough


def add_structural_coefficients(df: pd.DataFrame, hurst: float = config.H_PRIOR,
                                suffix: str = "") -> pd.DataFrame:
    """Append alpha/gamma computed at a *given* H — the core rough transform.

    The point of a suffix is comparing several H values side by side::

        for h in (0.05, 0.10, 0.20):
            df = add_structural_coefficients(df, h, suffix=f"_H{h:g}")

    If the rough model is right for this H, the resulting coefficients have no
    systematic dependence on T. Notebook 04 tests exactly that.
    """
    out = df.copy()
    out[f"alpha{suffix}"] = rough.structural_alpha(
        out["rr25"], out["T"], out["atm_iv"], hurst)
    out[f"gamma{suffix}"] = rough.structural_gamma(
        out["bf25"], out["T"], out["atm_total_var"], hurst)
    out.attrs.update(df.attrs)
    return out


def add_moneyness_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append the maturity transforms that regressions actually want.

    log_T is the regressor for every power-law fit; sqrt_T is the natural scale
    for vol; dte is what humans read.
    """
    out = df.copy()
    T = out["T"].to_numpy(dtype=float)
    out["log_T"] = np.log(np.where(T > 0, T, np.nan))
    out["sqrt_T"] = np.sqrt(np.clip(T, 0, None))
    out["dte"] = T * 365.0
    if "atm_iv" in out.columns and "atm_total_var" not in out.columns:
        out["atm_total_var"] = out["atm_iv"] ** 2 * T
    out.attrs.update(df.attrs)
    return out


def add_tenor_bucket(df: pd.DataFrame, edges=(0, 10, 21, 35, 60, 999),
                     labels=None) -> pd.DataFrame:
    """Append a categorical `tenor_bucket` in calendar days.

    Bucketing matters because the rough signature is a *maturity* effect: pooling
    7-day and 60-day options into one regression averages away the thing you are
    trying to measure. Mirrors `pool_by_tenor_bucket` in robustness_sweeps.py.
    """
    labels = labels or [f"{a}-{b}d" for a, b in zip(edges[:-1], edges[1:])]
    out = df.copy()
    dte = out["T"].to_numpy(dtype=float) * 365.0
    out["tenor_bucket"] = pd.cut(dte, bins=list(edges), labels=labels, right=False)
    out.attrs.update(df.attrs)
    return out


def cross_section(df: pd.DataFrame, ts) -> pd.DataFrame:
    """One timestamp's term structure — every expiry, sorted by maturity.

    This is the unit of the skew-scaling test: one regression per timestamp
    across maturities, never one regression pooled across time.
    """
    ts = pd.Timestamp(ts, tz="UTC") if not isinstance(ts, pd.Timestamp) else ts
    return df[df["ts"] == ts].sort_values("T").reset_index(drop=True)


def widest_cross_section(df: pd.DataFrame) -> pd.DataFrame:
    """The timestamp with the most expiries — the best single slice for a plot."""
    counts = df.groupby("ts", observed=True).size()
    return cross_section(df, counts.idxmax())


def summarise_panel(df: pd.DataFrame) -> pd.DataFrame:
    """A one-look description of a smile panel: shape, span, coverage, levels."""
    rows = {
        "rows": len(df),
        "timestamps": df["ts"].nunique() if "ts" in df else np.nan,
        "expiries": df["expiry"].nunique() if "expiry" in df else np.nan,
        "expiries_per_ts": round(len(df) / max(df["ts"].nunique(), 1), 2) if "ts" in df else np.nan,
        "dte_min": round(df["T"].min() * 365, 1),
        "dte_max": round(df["T"].max() * 365, 1),
        "atm_iv_median": round(float(df["atm_iv"].median()), 4),
        "rr25_median": round(float(df["rr25"].median()), 5),
        "bf25_median": round(float(df["bf25"].median()), 5),
    }
    return pd.DataFrame({"value": rows.values()}, index=list(rows)).rename_axis("metric")


# ── chain-level extraction (needs the research tree) ───────────────────────────
def extract_from_chain(chain_slice: pd.DataFrame, T: float) -> dict | None:
    """Raw option quotes -> {forward, atm_iv, atm_total_var, rr25, bf25}.

    Requires columns strike, is_call, mid, spread. Delegates to the project's
    own extractor so this never becomes a second, subtly different definition.
    Returns None when the slice is too thin to invert reliably.
    """
    sp = compat.research_module("smile_pipeline")
    cols = ["strike", "is_call", "mid"]
    frame = chain_slice[cols + (["spread"] if "spread" in chain_slice else [])].copy()
    if "spread" not in frame:
        frame["spread"] = 0.01
    return sp.extract_features(frame, T)


def panel_from_chain(chain_df: pd.DataFrame, hurst: float = config.H_PRIOR) -> pd.DataFrame:
    """Full option-chain panel -> the standard smile-feature panel.

    Input columns: ts, expiry, strike, is_call, mid (spread optional).
    """
    sp = compat.research_module("smile_pipeline")
    return pd.DataFrame(sp.process_panel_df(chain_df, hurst))


__all__ = [
    "add_structural_coefficients", "add_moneyness_features", "add_tenor_bucket",
    "cross_section", "widest_cross_section", "summarise_panel",
    "extract_from_chain", "panel_from_chain",
]
