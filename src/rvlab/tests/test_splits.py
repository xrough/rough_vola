"""The property that matters most: splitters must not leak.

Every test here is a leak the author has personally shipped at some point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rvlab.pipelines.splits import (
    GroupTimeSeriesSplit, PurgedTimeSeriesSplit, WalkForwardSplit, describe_folds,
)


class TestNoOverlap:
    def test_purged_train_and_test_are_disjoint(self, intraday):
        for tr, te in PurgedTimeSeriesSplit(n_splits=4).split(intraday):
            assert not set(tr) & set(te)

    def test_grouped_train_and_test_are_disjoint(self, intraday):
        for tr, te in GroupTimeSeriesSplit(n_splits=4).split(intraday):
            assert not set(tr) & set(te)

    def test_train_always_precedes_test(self, intraday):
        ts = intraday["ts"]
        for tr, te in GroupTimeSeriesSplit(n_splits=4).split(intraday):
            assert ts.iloc[tr].max() < ts.iloc[te].min()


class TestGroupIntegrity:
    def test_grouped_split_never_splits_a_timestamp(self, intraday):
        """The panel leak: one instant must be wholly in train or wholly in test."""
        ts = intraday["ts"]
        for tr, te in GroupTimeSeriesSplit(n_splits=4).split(intraday):
            assert not set(ts.iloc[tr]) & set(ts.iloc[te])

    def test_row_wise_split_leaks_on_an_uneven_panel(self):
        """Documents *why* GroupTimeSeriesSplit exists.

        A row-wise split only stays clean when every fold boundary happens to
        land between timestamps. On an even panel (a fixed number of expiries
        per bar, with a fold size that divides it) it can look perfectly safe.
        A real smile panel quotes a *variable* number of expiries per bar, so
        boundaries land mid-timestamp and the same instant appears on both
        sides of the split.
        """
        rng = np.random.default_rng(0)
        stamps = pd.date_range("2026-01-05 09:30", periods=300, freq="1min", tz="UTC")
        counts = rng.integers(2, 8, size=len(stamps))          # uneven, as in reality
        df = pd.DataFrame({
            "ts": np.repeat(stamps, counts),
            "expiry": [f"e{i}" for c in counts for i in range(c)],
            "x": rng.standard_normal(int(counts.sum())),
        })

        ts = df["ts"]
        leaked = any(
            set(ts.iloc[tr]) & set(ts.iloc[te])
            for tr, te in PurgedTimeSeriesSplit(n_splits=4, embargo=0).split(df)
        )
        assert leaked, "expected the row-wise splitter to straddle a timestamp"

        # ...and the grouped splitter must stay clean on exactly the same panel.
        assert not any(
            set(ts.iloc[tr]) & set(ts.iloc[te])
            for tr, te in GroupTimeSeriesSplit(n_splits=4).split(df)
        )


class TestEmbargo:
    @pytest.mark.parametrize("minutes", [15, 30, 120])
    def test_timedelta_embargo_creates_at_least_that_gap(self, intraday, minutes):
        embargo = pd.Timedelta(minutes=minutes)
        folds = describe_folds(GroupTimeSeriesSplit(n_splits=3, embargo=embargo), intraday)
        assert (folds["gap"] >= embargo).all()
        assert not folds["overlap"].any()

    def test_larger_embargo_shrinks_training_sets(self, intraday):
        small = describe_folds(
            GroupTimeSeriesSplit(n_splits=3, embargo=pd.Timedelta(minutes=15)), intraday)
        large = describe_folds(
            GroupTimeSeriesSplit(n_splits=3, embargo=pd.Timedelta(minutes=120)), intraday)
        assert (large["n_train"] < small["n_train"]).all()

    def test_integer_embargo_drops_that_many_rows(self, intraday):
        none = describe_folds(PurgedTimeSeriesSplit(n_splits=3, embargo=0), intraday)
        some = describe_folds(PurgedTimeSeriesSplit(n_splits=3, embargo=100), intraday)
        assert ((none["n_train"] - some["n_train"]) == 100).all()


class TestWalkForward:
    def test_every_fold_has_the_same_training_size(self, intraday):
        cv = WalkForwardSplit(train_size=400, test_size=100)
        sizes = {len(tr) for tr, _ in cv.split(intraday)}
        assert sizes == {400}

    def test_get_n_splits_matches_what_split_yields(self, intraday):
        cv = WalkForwardSplit(train_size=400, test_size=100)
        assert cv.get_n_splits(intraday) == len(list(cv.split(intraday)))


class TestErrors:
    def test_missing_time_column_raises_a_readable_error(self):
        df = pd.DataFrame({"x": np.arange(100)})
        with pytest.raises(KeyError, match="time_col"):
            list(PurgedTimeSeriesSplit(time_col="ts").split(df))

    def test_too_few_rows_raises(self):
        df = pd.DataFrame({"ts": pd.date_range("2026-01-01", periods=3, tz="UTC")})
        with pytest.raises(ValueError, match="too few"):
            list(PurgedTimeSeriesSplit(n_splits=5).split(df))


class TestIntegrationWithSklearn:
    def test_works_as_a_cv_argument(self, intraday):
        """A splitter that sklearn silently rejects is worse than none."""
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_score

        X = intraday[["x"]].copy()
        X["ts"] = intraday["ts"].to_numpy()
        y = intraday["x"].shift(-1).fillna(0.0)
        cv = list(GroupTimeSeriesSplit(n_splits=3).split(X))
        scores = cross_val_score(Ridge(), X[["x"]], y, cv=cv, scoring="neg_mean_squared_error")
        assert len(scores) == 3 and np.isfinite(scores).all()
