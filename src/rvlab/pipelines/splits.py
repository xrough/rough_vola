"""
rvlab.pipelines.splits
======================
Cross-validation splitters that do not leak.

`TimeSeriesSplit` gets the *direction* of time right — train always precedes
test — but it still leaks in two ways that matter here:

1. **Feature overlap.** A rolling 20-bar feature on the last training row was
   computed from bars that also sit in the test fold. The embargo fixes this by
   dropping bars between train and test.
2. **Panel duplication.** A smile panel has ~12 rows per timestamp, one per
   expiry. A row-wise split puts the 30-day expiry of 09:35 in train and the
   60-day expiry of the *same instant* in test. `GroupTimeSeriesSplit` splits on
   the timestamp instead, keeping whole instants together.

Both are drop-in for any sklearn `cv=` argument.

> ⚠️ **Leakage:** `GridSearchCV` accepts a leaky splitter without complaint and
> reports an excellent score. Nothing warns you. The only defence is choosing
> the splitter deliberately and testing it — see `rvlab/tests/test_splits.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


def _positions(X, time_col: str | None) -> np.ndarray:
    """Ordering key for the rows of X, as a sortable numpy array."""
    if time_col is None:
        if isinstance(X, pd.DataFrame) and isinstance(X.index, pd.DatetimeIndex):
            return X.index.to_numpy()
        return np.arange(len(X))
    if not isinstance(X, pd.DataFrame) or time_col not in X.columns:
        raise KeyError(
            f"time_col={time_col!r} not found. Pass a DataFrame containing that "
            "column, or time_col=None to use a DatetimeIndex / row order."
        )
    return X[time_col].to_numpy()


class PurgedTimeSeriesSplit(BaseCrossValidator):
    """Expanding-window CV with a gap between train and test.

    Parameters
    ----------
    n_splits : number of folds.
    embargo  : rows (int) or duration (pd.Timedelta) dropped from the end of each
               training fold. Must be at least your longest feature lookback.
    time_col : column holding the timestamp; None uses row order.
    max_train_size : cap the training window (rolling instead of expanding).

    Example
    -------
    >>> cv = PurgedTimeSeriesSplit(n_splits=5, embargo=pd.Timedelta("30min"))
    >>> for tr, te in cv.split(df): ...
    """

    def __init__(self, n_splits: int = 5, embargo=0, time_col: str | None = "ts",
                 max_train_size: int | None = None):
        self.n_splits = n_splits
        self.embargo = embargo
        self.time_col = time_col
        self.max_train_size = max_train_size

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        times = _positions(X, self.time_col)
        n = len(times)
        order = np.argsort(times, kind="stable")
        times_sorted = times[order]

        fold_size = n // (self.n_splits + 1)
        if fold_size < 1:
            raise ValueError(f"{n} rows is too few for {self.n_splits} splits")

        for i in range(1, self.n_splits + 1):
            split_at = fold_size * i
            test_end = min(split_at + fold_size, n)
            train_idx = order[:split_at]
            test_idx = order[split_at:test_end]
            if len(test_idx) == 0:
                continue

            # ── purge: drop the tail of train that is within `embargo` of test ──
            if isinstance(self.embargo, pd.Timedelta):
                boundary = pd.Timestamp(times_sorted[split_at]) - self.embargo
                keep = pd.to_datetime(times[train_idx], utc=True) <= boundary
                train_idx = train_idx[keep]
            elif self.embargo:
                train_idx = train_idx[: max(0, len(train_idx) - int(self.embargo))]

            if len(train_idx):
                yield train_idx, test_idx


class GroupTimeSeriesSplit(BaseCrossValidator):
    """Like `PurgedTimeSeriesSplit`, but folds are made of whole timestamps.

    Use this on any panel where one timestamp contributes several rows — which
    is every smile panel in this series. Splitting rows instead of instants lets
    the model see a different expiry of the same instant it is being tested on.
    """

    def __init__(self, n_splits: int = 5, embargo=0, time_col: str = "ts"):
        self.n_splits = n_splits
        self.embargo = embargo
        self.time_col = time_col

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        times = pd.Series(_positions(X, self.time_col))
        unique = np.sort(times.unique())
        n_u = len(unique)

        fold_size = n_u // (self.n_splits + 1)
        if fold_size < 1:
            raise ValueError(
                f"{n_u} distinct timestamps is too few for {self.n_splits} splits")

        idx = np.arange(len(times))
        for i in range(1, self.n_splits + 1):
            split_at = fold_size * i
            test_stamps = unique[split_at: min(split_at + fold_size, n_u)]
            train_stamps = unique[:split_at]

            if isinstance(self.embargo, pd.Timedelta) and len(test_stamps):
                boundary = pd.Timestamp(test_stamps[0]) - self.embargo
                train_stamps = train_stamps[pd.to_datetime(train_stamps, utc=True) <= boundary]
            elif self.embargo:
                train_stamps = train_stamps[: max(0, len(train_stamps) - int(self.embargo))]

            train_idx = idx[times.isin(set(train_stamps)).to_numpy()]
            test_idx = idx[times.isin(set(test_stamps)).to_numpy()]
            if len(train_idx) and len(test_idx):
                yield train_idx, test_idx


class WalkForwardSplit(BaseCrossValidator):
    """Fixed-length rolling train window followed by a fixed test window.

    Closest to how a model is actually used in production: retrain on the last
    `train_size` observations, trade the next `test_size`, roll forward by
    `step`. Unlike the expanding splitters, every fold has the same training
    size, so fold-to-fold score differences are about the market, not about how
    much data the model happened to get.
    """

    def __init__(self, train_size: int = 1_000, test_size: int = 250,
                 step: int | None = None, embargo: int = 0, time_col: str | None = "ts"):
        self.train_size = train_size
        self.test_size = test_size
        self.step = step
        self.embargo = embargo
        self.time_col = time_col

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        n = len(X)
        step = self.step or self.test_size
        return max(0, 1 + (n - self.train_size - self.test_size) // step)

    def split(self, X, y=None, groups=None):
        times = _positions(X, self.time_col)
        order = np.argsort(times, kind="stable")
        n = len(order)
        step = self.step or self.test_size

        start = 0
        while start + self.train_size + self.test_size <= n:
            tr_end = start + self.train_size
            train_idx = order[start: max(start, tr_end - self.embargo)]
            test_idx = order[tr_end: tr_end + self.test_size]
            if len(train_idx) and len(test_idx):
                yield train_idx, test_idx
            start += step


def describe_folds(cv, X, time_col: str | None = "ts") -> pd.DataFrame:
    """Tabulate what a splitter actually produced — sizes, spans, and the gap.

    Print this before trusting any CV score. The `gap` column is the embargo you
    think you configured; seeing it as `0 days 00:00:00` is how you discover you
    passed an int where a Timedelta was needed.
    """
    rows = []
    times = _positions(X, time_col)
    # tz-aware timestamps arrive as object dtype from .to_numpy(), so probe a
    # value rather than the dtype.
    try:
        times_idx = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
        is_time = True
    except (TypeError, ValueError):
        times_idx, is_time = None, False

    for i, (tr, te) in enumerate(cv.split(X)):
        row = {"fold": i, "n_train": len(tr), "n_test": len(te)}
        if is_time:
            t_tr, t_te = times_idx[tr], times_idx[te]
            row |= {"train_end": t_tr.max(), "test_start": t_te.min(),
                    "gap": t_te.min() - t_tr.max()}
            row["overlap"] = bool(t_tr.max() >= t_te.min())
        else:
            row["overlap"] = bool(set(tr) & set(te))
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "PurgedTimeSeriesSplit", "GroupTimeSeriesSplit", "WalkForwardSplit",
    "describe_folds", "purged_date_splits", "describe_date_folds",
]


def purged_date_splits(df, n_splits: int = 5, gap_days: int = 0,
                       date_col: str = "Date", max_train_dates: int | None = None):
    """The canonical competition CV idiom, as an explicit list of row indices.

    Split the **unique dates**, apply `TimeSeriesSplit(gap=...)` to those, then
    map each fold's dates back to integer row positions:

        cv = purged_date_splits(df, n_splits=4, gap_days=2)
        GridSearchCV(pipe, grid, cv=cv).fit(X, y)

    `gap_days` must be at least the label horizon. When the target at date *t*
    is built from prices at *t+1* and *t+2*, the last two training dates carry
    labels containing information from the validation period — a gap of 2 drops
    them.

    Returns a plain list of `(train_idx, val_idx)` arrays. sklearn accepts that
    anywhere a splitter is expected, and unlike a splitter object it is
    inspectable: print the fold spans before trusting any score.
    """
    from sklearn.model_selection import TimeSeriesSplit

    dates = np.sort(pd.Series(df[date_col]).unique())
    date_of_row = df[date_col].to_numpy()
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap_days)

    folds = []
    for train_pos, val_pos in splitter.split(dates):
        train_dates = dates[train_pos]
        if max_train_dates:
            train_dates = train_dates[-max_train_dates:]
        train_idx = np.flatnonzero(np.isin(date_of_row, train_dates))
        val_idx = np.flatnonzero(np.isin(date_of_row, dates[val_pos]))
        if len(train_idx) and len(val_idx):
            folds.append((train_idx, val_idx))
    return folds


def describe_date_folds(folds, df, date_col: str = "Date"):
    """Summarise `purged_date_splits` output: spans, sizes and the realised gap."""
    dates = pd.to_datetime(pd.Series(df[date_col]).to_numpy())
    rows = []
    for i, (train_idx, val_idx) in enumerate(folds):
        tr, va = dates[train_idx], dates[val_idx]
        rows.append({
            "fold": i, "n_train": len(train_idx), "n_test": len(val_idx),
            "train_end": tr.max().date(), "test_start": va.min().date(),
            "test_end": va.max().date(),
            "gap_days": (va.min() - tr.max()).days,
            "overlap": bool(tr.max() >= va.min()),
        })
    return pd.DataFrame(rows)
