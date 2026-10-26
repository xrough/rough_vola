"""
rvlab.features.panel
====================
Feature builders for (date x entity) panels.

Three families, and the distinction between them is the substance of panel
feature engineering:

**Per-entity, backward looking** — lags, multi-horizon returns, rolling
statistics. Every one must be computed *within* an entity, or it reaches
sideways into a neighbouring entity's row.

**Cross-sectional** — rank and de-meaning *within a date*. These use every
entity at the same instant, which is legitimate because all of it is observable
then. They matter enormously for a ranking objective: a +2% day means something
different when the market rose 2% than when it fell 2%, and a raw return cannot
express that difference while a cross-sectional rank can.

**Group-relative** — subtract the sector's mean on that date. A sharper version
of the same idea, using structure you already know.

Per-entity lag/rolling helpers live in `rvlab.features.timeseries` and work here
unchanged: pass `group=("SecuritiesCode",)` and `sort_by="Date"`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.panel import DATE, ENTITY


def add_calendar_features(df: pd.DataFrame, date_col: str = DATE) -> pd.DataFrame:
    """Year, month, weekday, month-end and week-of-year.

    Cheap, occasionally real (turn-of-month and Monday effects are documented),
    and a useful control: if a model leans hard on `dayofweek`, that is usually a
    sign the rest of the feature set is weak rather than a discovery.
    """
    out = df.copy()
    ts = pd.to_datetime(out[date_col])
    out["year"] = ts.dt.year
    out["month"] = ts.dt.month
    out["dayofweek"] = ts.dt.dayofweek
    out["dayofmonth"] = ts.dt.day
    out["weekofyear"] = ts.dt.isocalendar().week.astype(int)
    out["is_month_end"] = ts.dt.is_month_end.astype(int)
    out.attrs.update(df.attrs)
    return out


def add_price_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Same-bar OHLCV shape: all of it observable at the close of that bar.

    `close_location` — where the close sits inside the day's range — is the one
    worth knowing about. It separates "up on the day" from "up on the day and
    closed on the high", which behave differently.
    """
    out = df.copy()
    for c in ("Open", "High", "Low", "Close", "Volume", "AdjustmentFactor",
              "ExpectedDividend"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if {"Open", "Close"} <= set(out.columns):
        out["intraday_return"] = out["Close"] / out["Open"].replace(0, np.nan) - 1
    if {"High", "Low", "Close"} <= set(out.columns):
        out["range_pct"] = (out["High"] - out["Low"]) / out["Close"].replace(0, np.nan)
        span = (out["High"] - out["Low"]).replace(0, np.nan)
        out["close_location"] = (out["Close"] - out["Low"]) / span
    if "Volume" in out.columns:
        out["log_volume"] = np.log1p(out["Volume"].clip(lower=0))
    if {"Close", "Volume"} <= set(out.columns):
        out["log_dollar_volume"] = np.log1p(
            out["Close"].clip(lower=0) * out["Volume"].clip(lower=0))

    if "ExpectedDividend" in out.columns:
        div = out["ExpectedDividend"].fillna(0)
        out["dividend_flag"] = div.gt(0).astype(int)
        if "Close" in out.columns:
            out["expected_dividend_yield"] = div / out["Close"].replace(0, np.nan)
    if "SupervisionFlag" in out.columns:
        out["supervision_flag"] = out["SupervisionFlag"].fillna(False).astype(int)

    # A corporate action makes raw-price returns across it meaningless.
    if "AdjustmentFactor" in out.columns:
        out["corporate_action_flag"] = out["AdjustmentFactor"].fillna(1.0).ne(1.0).astype(int)
    else:
        out["corporate_action_flag"] = 0

    out = out.replace([np.inf, -np.inf], np.nan)
    out.attrs.update(df.attrs)
    return out


def add_horizon_returns(df: pd.DataFrame, horizons=(1, 2, 5, 10, 20, 60),
                        price_col: str = "Close", entity_col: str = ENTITY,
                        date_col: str = DATE, mask_events: bool = True) -> pd.DataFrame:
    """Multi-horizon returns per entity, masked where a corporate action intrudes.

    A 2-for-1 split halves the raw close. A 5-day return computed across it reads
    as −50% and is pure fiction — and because splits cluster in time, that
    fiction is *correlated across entities*, which is exactly the kind of
    structure a model will happily learn.

    `mask_events=True` NaNs any window containing an action, using only the flag
    at each date. The alternative — a reverse cumulative adjustment factor — is
    computed from *future* actions and leaks during historical validation.
    """
    out = df.sort_values([entity_col, date_col], kind="stable").copy()
    grouped = out.groupby(entity_col, sort=False, group_keys=False)

    if {"Open", price_col} <= set(out.columns):
        prev_close = grouped[price_col].shift(1)
        out["overnight_gap"] = out["Open"] / prev_close.replace(0, np.nan) - 1
        if mask_events and "corporate_action_flag" in out.columns:
            out.loc[out["corporate_action_flag"].eq(1), "overnight_gap"] = np.nan

    has_flag = mask_events and "corporate_action_flag" in out.columns
    for h in horizons:
        ret = grouped[price_col].pct_change(h)
        if has_flag:
            in_window = grouped["corporate_action_flag"].transform(
                lambda s, h=h: s.rolling(h + 1, min_periods=1).max())
            ret = ret.mask(in_window.gt(0))
        out[f"ret_{h}d"] = ret

    out = out.replace([np.inf, -np.inf], np.nan)
    out.attrs.update(df.attrs)
    return out


def add_entity_rolling(df: pd.DataFrame, columns, windows=(5, 20, 60),
                       stats=("mean", "std"), entity_col: str = ENTITY,
                       date_col: str = DATE, zscore_of=()) -> pd.DataFrame:
    """Rolling statistics within each entity, plus optional rolling z-scores.

    A rolling z-score — `(x - mean_w) / std_w` — is usually more useful than the
    level for a cross-entity model, because it puts every entity on a comparable
    scale. Volume is the standard case: 3 million shares is enormous for one
    stock and a quiet day for another.
    """
    out = df.sort_values([entity_col, date_col], kind="stable").copy()
    grouped = out.groupby(entity_col, sort=False, group_keys=False)

    for col in columns:
        if col not in out.columns:
            continue
        for w in windows:
            minp = max(2, w // 3)
            roll = grouped[col].rolling(w, min_periods=minp)
            for stat in stats:
                vals = getattr(roll, stat)()
                out[f"{col}_{stat}{w}"] = vals.reset_index(level=0, drop=True)

    for col in zscore_of:
        if col not in out.columns:
            continue
        for w in windows:
            mean_col, std_col = f"{col}_mean{w}", f"{col}_std{w}"
            if mean_col in out.columns and std_col in out.columns:
                out[f"{col}_z{w}"] = ((out[col] - out[mean_col])
                                      / out[std_col].replace(0, np.nan))

    out = out.replace([np.inf, -np.inf], np.nan)
    out.attrs.update(df.attrs)
    return out


def add_cross_sectional(df: pd.DataFrame, columns, date_col: str = DATE,
                        rank: bool = True, demean: bool = True,
                        zscore: bool = False) -> pd.DataFrame:
    """Rank, de-mean and/or z-score each column *within its date*.

    The single highest-value transform for a ranking objective. `cs_rank_x` is
    bounded in [0, 1], immune to the fat tails that dominate raw returns, and
    directly comparable across dates with wildly different volatility.

    De-meaning uses the *median* rather than the mean, so one bad print cannot
    shift the whole cross-section.
    """
    out = df.copy()
    present = [c for c in columns if c in out.columns]
    by_date = out.groupby(date_col, observed=True)

    for col in present:
        if rank:
            out[f"cs_rank_{col}"] = by_date[col].rank(pct=True, method="average")
        if demean:
            out[f"cs_demean_{col}"] = out[col] - by_date[col].transform("median")
        if zscore:
            std = by_date[col].transform("std").replace(0, np.nan)
            out[f"cs_z_{col}"] = (out[col] - by_date[col].transform("mean")) / std

    out = out.replace([np.inf, -np.inf], np.nan)
    out.attrs.update(df.attrs)
    return out


def add_group_relative(df: pd.DataFrame, columns, group_col: str = "Sector",
                       date_col: str = DATE, suffix: str = "_vs_group") -> pd.DataFrame:
    """Subtract the group's same-date mean — sector-relative momentum and the like.

    Still causal: every entity in the group is observable on that date. This is
    a sharper control than cross-sectional de-meaning when you already know the
    grouping that drives most of the common variation.
    """
    out = df.copy()
    if group_col not in out.columns:
        return out
    for col in [c for c in columns if c in out.columns]:
        group_mean = out.groupby([date_col, group_col], observed=True)[col].transform("mean")
        out[f"{col}{suffix}"] = out[col] - group_mean
    out.attrs.update(df.attrs)
    return out


def make_panel_target(df: pd.DataFrame, target_col: str = "Target",
                      date_col: str = DATE, demean: bool = True,
                      out_col: str = "ModelTarget") -> pd.DataFrame:
    """Copy the label, optionally removing each date's cross-sectional mean.

    When the objective is *ranking*, the market's daily move is pure nuisance —
    it shifts every entity's target together and cannot change their order. De-
    meaning removes it, which typically improves rank correlation even though it
    makes the regression's RMSE look worse.

    This uses labels only within a date, and is not needed at inference time:
    ranking a set of predictions is invariant to adding a constant to all of them.
    """
    out = df.copy()
    out[out_col] = out[target_col]
    if demean:
        out[out_col] = out[target_col] - out.groupby(date_col, observed=True)[target_col].transform("mean")
    out.attrs.update(df.attrs)
    return out


def winsorize_by_date(df: pd.DataFrame, columns, date_col: str = DATE,
                      lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Clip each column to per-date quantiles.

    Per-date rather than global: a 10% move is an outlier on a calm day and
    unremarkable in a crash, and clipping to a global quantile removes most of a
    volatile period's cross-section rather than its outliers.
    """
    out = df.copy()
    by_date = out.groupby(date_col, observed=True)
    for col in [c for c in columns if c in out.columns]:
        lo = by_date[col].transform(lambda s: s.quantile(lower))
        hi = by_date[col].transform(lambda s: s.quantile(upper))
        out[col] = out[col].clip(lo, hi)
    out.attrs.update(df.attrs)
    return out


def build_panel_features(panel: pd.DataFrame, horizons=(1, 5, 20),
                         windows=(5, 20), group_col: str = "Sector",
                         demean_target: bool = True) -> pd.DataFrame:
    """The whole standard pipeline in one call — the notebooks spell it out step by step.

    calendar -> price geometry -> horizon returns (event-masked) -> entity
    rolling stats and volume z-scores -> cross-sectional rank/de-mean ->
    group-relative -> model target.
    """
    out = add_price_geometry(add_calendar_features(panel))
    out = add_horizon_returns(out, horizons=horizons)
    out = add_entity_rolling(out, ["ret_1d", "log_volume"], windows=windows,
                             zscore_of=["log_volume"])
    cs_cols = [c for c in [f"ret_{h}d" for h in horizons]
               + ["intraday_return", "range_pct", "log_dollar_volume",
                  "close_location"] if c in out.columns]
    out = add_cross_sectional(out, cs_cols)
    out = add_group_relative(out, [f"ret_{h}d" for h in horizons], group_col=group_col)
    if "Target" in out.columns:
        out = make_panel_target(out, demean=demean_target)
    return out.sort_values([DATE, ENTITY], kind="stable").reset_index(drop=True)


__all__ = [
    "add_calendar_features", "add_price_geometry", "add_horizon_returns",
    "add_entity_rolling", "add_cross_sectional", "add_group_relative",
    "make_panel_target", "winsorize_by_date", "build_panel_features",
]
