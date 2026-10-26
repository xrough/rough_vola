"""
rvlab.features.timeseries
=========================
Leakage-safe lag, rolling and target builders.

Every function here takes a `group` argument. That is not decoration: a smile
panel interleaves many expiries at the same timestamp, so a naive
`df["atm_iv"].shift(1)` silently pulls the previous *expiry's* value rather than
the previous *bar's* value for the same expiry. Grouping is the difference
between a feature and a bug.

The other rule enforced here: a feature at time t uses only information at or
before t, and a target at time t uses only information strictly after t. Every
function states which side of that line it is on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_GROUP = ("expiry",)


def _grouped(df: pd.DataFrame, col: str, group):
    """Return a (possibly grouped) Series ready for shift/rolling."""
    if not group:
        return df[col]
    return df.groupby(list(group), observed=True, sort=False)[col]


def add_lags(df: pd.DataFrame, columns, lags=(1, 2, 3), group=DEFAULT_GROUP,
             sort_by: str = "ts") -> pd.DataFrame:
    """Append `{col}_lag{k}` columns. Backward-looking: safe as features.

    Sorts by `sort_by` within the frame first — shift() is meaningless on an
    unsorted panel and gives no warning.
    """
    out = df.sort_values([*group, sort_by] if group else [sort_by], kind="stable").copy()
    for col in columns:
        for k in lags:
            out[f"{col}_lag{k}"] = _grouped(out, col, group).shift(k)
    return out


def add_diffs(df: pd.DataFrame, columns, periods: int = 1, group=DEFAULT_GROUP,
              sort_by: str = "ts", log: bool = False) -> pd.DataFrame:
    """Append `{col}_d{periods}` changes. Backward-looking: safe as features."""
    out = df.sort_values([*group, sort_by] if group else [sort_by], kind="stable").copy()
    for col in columns:
        s = np.log(out[col].where(out[col] > 0)) if log else out[col]
        out[f"{col}_{'ld' if log else 'd'}{periods}"] = (
            s.groupby([out[g] for g in group], observed=True, sort=False).diff(periods)
            if group else s.diff(periods))
    return out


def add_rolling(df: pd.DataFrame, columns, windows=(5, 20), stats=("mean", "std"),
                group=DEFAULT_GROUP, sort_by: str = "ts",
                min_periods: int | None = None) -> pd.DataFrame:
    """Append `{col}_{stat}{window}` rolling statistics.

    Backward-looking and *closed on the right*: the window ending at t includes
    t itself. That is correct for a feature describing the state at t, and wrong
    if you meant "the last w bars before t" — in that case shift the result by 1.
    """
    out = df.sort_values([*group, sort_by] if group else [sort_by], kind="stable").copy()
    for col in columns:
        g = _grouped(out, col, group)
        for w in windows:
            roll = g.rolling(w, min_periods=min_periods or max(2, w // 2))
            for stat in stats:
                vals = getattr(roll, stat)()
                out[f"{col}_{stat}{w}"] = vals.reset_index(level=list(range(len(group))),
                                                           drop=True) if group else vals
    return out


def add_expanding(df: pd.DataFrame, columns, stat: str = "median",
                  group=DEFAULT_GROUP, sort_by: str = "ts",
                  min_periods: int = 30, shift: int = 1) -> pd.DataFrame:
    """Append `{col}_exp_{stat}` expanding statistics, shifted by `shift` bars.

    `shift=1` by default because an expanding statistic that includes the current
    bar has already seen the row you are about to predict. This is the single
    most common leak in "use all history so far" features.
    """
    out = df.sort_values([*group, sort_by] if group else [sort_by], kind="stable").copy()
    for col in columns:
        g = _grouped(out, col, group)
        vals = getattr(g.expanding(min_periods=min_periods), stat)()
        if group:
            vals = vals.reset_index(level=list(range(len(group))), drop=True)
        out[f"{col}_exp_{stat}"] = vals.groupby(
            [out[g_] for g_ in group], observed=True, sort=False).shift(shift) \
            if group else vals.shift(shift)
    return out


def make_target(df: pd.DataFrame, column: str, horizon: int = 1,
                group=DEFAULT_GROUP, sort_by: str = "ts",
                name: str | None = None) -> pd.DataFrame:
    """Append the *forward* value `{column}_fwd{horizon}`. Forward-looking: a target.

    Never put a column produced by this function into X. It is `shift(-horizon)`,
    i.e. information from the future, which is exactly what you are trying to
    predict.
    """
    out = df.sort_values([*group, sort_by] if group else [sort_by], kind="stable").copy()
    target = name or f"{column}_fwd{horizon}"
    out[target] = _grouped(out, column, group).shift(-horizon)
    out.attrs["target"] = target
    return out


def supervised_frame(df: pd.DataFrame, target_col: str, feature_cols,
                     dropna: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Split a prepared frame into (X, y), dropping rows with missing values.

    Returns X as a DataFrame (not an array) because the baseline estimators in
    `rvlab.models.baselines` look columns up by name.
    """
    cols = [c for c in feature_cols if c in df.columns]
    frame = df[[*dict.fromkeys([*cols, target_col])]]
    if dropna:
        frame = frame.dropna()
    return frame[cols], frame[target_col]


def resample_panel(df: pd.DataFrame, step_minutes: int, ts_col: str = "ts",
                   group=DEFAULT_GROUP) -> pd.DataFrame:
    """Thin an intraday panel to every `step_minutes`, keeping whole bars.

    Subsampling rather than aggregating: the smile at 09:40 is a real
    observation, whereas a 5-minute *average* smile is an object that never
    traded. Mirrors `resample_panel` in shared/robustness_sweeps.py.
    """
    if step_minutes <= 1:
        return df.copy()
    ts = pd.to_datetime(df[ts_col], utc=True)
    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    keep = (minute_of_day % step_minutes) == 0
    out = df[keep].copy()
    out.attrs.update(df.attrs)
    return out


__all__ = [
    "add_lags", "add_diffs", "add_rolling", "add_expanding", "make_target",
    "supervised_frame", "resample_panel", "build_design_matrix", "DEFAULT_GROUP",
]


def build_design_matrix(panel: pd.DataFrame, target: str, horizon: int = 1,
                        step_minutes: int = 5, hurst: float = 0.10,
                        lag_cols=("atm_iv", "rr25", "bf25", "atm_total_var"),
                        lags=(1, 2), roll_windows=(5,),
                        extra_features=()) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """The standard preparation used by notebooks 06, 07 and 09, in one call.

    Resample -> structural coefficients -> grouped lags -> grouped rolling means
    -> forward target -> drop incomplete rows.

    Returns `(X, y, feature_names)` where `X` still carries `ts` and `expiry`
    (the panel-aware splitters and the baseline estimators need them) while
    `feature_names` lists only the columns a learned model should see. Nothing
    matching `_fwd` can appear in `feature_names` — that filter is the last line
    of defence against the most embarrassing kind of leak.
    """
    from .smile import add_structural_coefficients

    thinned = resample_panel(panel, step_minutes)
    with_alpha = add_structural_coefficients(thinned, hurst=hurst)
    lagged = add_lags(with_alpha, list(lag_cols), lags)
    rolled = add_rolling(lagged, list(lag_cols), windows=list(roll_windows), stats=("mean",))
    labelled = make_target(rolled, target, horizon)
    target_col = labelled.attrs["target"]

    feature_names = [c for c in labelled.columns
                     if c not in {"ts", "expiry", target_col}
                     and "_fwd" not in c
                     and labelled[c].dtype.kind == "f"]
    if extra_features:
        feature_names += [c for c in extra_features if c in labelled.columns
                          and c not in feature_names]

    X, y = supervised_frame(labelled, target_col, [*feature_names, "ts", "expiry"])
    return X.reset_index(drop=True), y.reset_index(drop=True), feature_names
