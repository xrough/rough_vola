"""Merges, as-of joins, interval joins and reshaping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rvlab.features.joins import (
    aggregate_and_join, asof_join, dates_in_columns_to_long, flatten_multiindex,
    interval_join, long_to_wide, merge_report, reconcile, safe_merge,
)


class TestMergeSafety:
    @staticmethod
    def _frames():
        left = pd.DataFrame({"k": [1, 2, 3], "x": [10, 20, 30]})
        clean = pd.DataFrame({"k": [1, 2, 3], "y": ["a", "b", "c"]})
        dupes = pd.DataFrame({"k": [1, 1, 2, 3], "y": list("abcd")})
        return left, clean, dupes

    def test_clean_merge_passes(self):
        left, clean, _ = self._frames()
        out = safe_merge(left, clean, on="k", validate="1:1")
        assert len(out) == 3

    def test_fan_out_is_refused_by_default(self):
        left, _, dupes = self._frames()
        with pytest.raises(ValueError, match="fanned out"):
            safe_merge(left, dupes, on="k")

    def test_fan_out_is_allowed_when_declared(self):
        left, _, dupes = self._frames()
        assert len(safe_merge(left, dupes, on="k", expect="any")) == 4

    def test_expect_same_catches_a_dropping_inner_join(self):
        left, clean, _ = self._frames()
        with pytest.raises(ValueError, match="changed row count"):
            safe_merge(left, clean.head(2), on="k", how="inner", expect="same")

    def test_validate_still_fires_for_cardinality(self):
        left, _, dupes = self._frames()
        with pytest.raises(pd.errors.MergeError):
            safe_merge(left, dupes, on="k", validate="1:1", expect="any")


class TestMergeReport:
    def test_predicts_a_fan_out_before_it_happens(self):
        left = pd.DataFrame({"k": [1, 2], "x": [1, 2]})
        right = pd.DataFrame({"k": [1, 1, 2, 2], "y": [1, 2, 3, 4]})
        report = merge_report(left, right, on="k")
        assert report.attrs["summary"]["fans_out"]
        assert report.attrs["summary"]["predicted_rows"] == 4

    def test_reports_a_dtype_mismatch_without_crashing(self):
        """The failure that returns an empty frame with no error — and which
        pandas raises on if you try the merge, so the report must not try it."""
        left = pd.DataFrame({"k": ["1", "2"], "x": [1, 2]})
        right = pd.DataFrame({"k": [1, 2], "y": [3, 4]})
        report = merge_report(left, right, on="k")
        assert not report.iloc[0]["dtype_match"]
        assert report.iloc[0]["shared"] == 0
        assert report.attrs["summary"]["dtype_mismatch"]

    def test_reports_key_overlap(self):
        left = pd.DataFrame({"k": [1, 2, 3]})
        right = pd.DataFrame({"k": [2, 3, 4]})
        row = merge_report(left, right, on="k").iloc[0]
        assert row["shared"] == 2 and row["left_only"] == 1 and row["right_only"] == 1


class TestReconcile:
    def test_counts_rows_and_lost_keys(self):
        before = pd.DataFrame({"k": [1, 2, 3], "v": [1, 2, 3]})
        after = before[before["k"] != 3]
        out = reconcile(before, after, "filter", key="k")
        assert out["row_delta"] == -1 and out["keys_lost"] == 1


class TestAsofJoin:
    @staticmethod
    def _series():
        left = pd.DataFrame({"t": pd.to_datetime(["2025-01-02", "2025-01-05"]), "x": [1, 2]})
        right = pd.DataFrame({"t": pd.to_datetime(["2025-01-01", "2025-01-03"]), "y": [10, 30]})
        return left, right

    def test_backward_never_matches_the_future(self):
        left, right = self._series()
        out = asof_join(left, right, on="t")
        assert list(out["y"]) == [10, 30]

    def test_forward_does_reach_forward(self):
        """Documented so the default is a deliberate choice, not an accident."""
        left, right = self._series()
        out = asof_join(left, right, on="t", direction="forward")
        assert out["y"].iloc[0] == 30            # matched to a *later* observation

    def test_tolerance_blocks_a_stale_match(self):
        """2 Jan finds 1 Jan (1 day, inside tolerance); 5 Jan would have to reach
        back to 3 Jan (2 days) and is left unmatched instead of carrying a stale
        value forward."""
        left, right = self._series()
        out = asof_join(left, right, on="t", tolerance=pd.Timedelta(days=1))
        assert out["y"].notna().iloc[0]
        assert out["y"].isna().iloc[1]

    def test_bad_direction_is_rejected(self):
        left, right = self._series()
        with pytest.raises(ValueError, match="backward"):
            asof_join(left, right, on="t", direction="sideways")

    def test_by_group_keeps_entities_separate(self):
        left = pd.DataFrame({"t": pd.to_datetime(["2025-01-02"] * 2), "g": ["a", "b"]})
        right = pd.DataFrame({"t": pd.to_datetime(["2025-01-01"] * 2),
                              "g": ["a", "b"], "y": [1, 2]})
        out = asof_join(left, right, on="t", by="g")
        assert list(out.sort_values("g")["y"]) == [1, 2]


class TestIntervalJoin:
    @staticmethod
    def _link_data():
        obs = pd.DataFrame({"id": [1, 1, 1],
                            "date": pd.to_datetime(["2024-01-15", "2024-06-15", "2024-12-15"]),
                            "v": [1, 2, 3]})
        links = pd.DataFrame({
            "id": [1, 1],
            "code": ["OLD", "NEW"],
            "start": pd.to_datetime(["2020-01-01", "2024-07-01"]),
            "end": pd.to_datetime(["2024-06-30", None]),
        })
        return obs, links

    def test_each_row_gets_the_mapping_valid_at_its_own_date(self):
        obs, links = self._link_data()
        out = interval_join(obs, links, on="id", left_time="date",
                            start="start", end="end")
        assert len(out) == 3
        assert list(out.sort_values("date")["code"]) == ["OLD", "OLD", "NEW"]

    def test_naive_id_join_would_have_doubled_the_rows(self):
        obs, links = self._link_data()
        assert len(obs.merge(links, on="id")) == 6      # the bug this prevents

    def test_open_ended_link_is_kept(self):
        obs, links = self._link_data()
        out = interval_join(obs, links, on="id", left_time="date",
                            start="start", end="end", end_na_is_open=True)
        assert (out["code"] == "NEW").sum() == 1

    def test_open_ended_link_dropped_when_disabled(self):
        obs, links = self._link_data()
        out = interval_join(obs, links, on="id", left_time="date",
                            start="start", end="end", end_na_is_open=False)
        assert (out["code"] == "NEW").sum() == 0

    def test_no_duplicate_keys_survive(self):
        obs, links = self._link_data()
        out = interval_join(obs, links, on="id", left_time="date",
                            start="start", end="end")
        assert not out.duplicated(["id", "date"]).any()


class TestAggregateAndJoin:
    def test_summarises_instead_of_fanning_out(self):
        left = pd.DataFrame({"d": [1, 2], "x": [10, 20]})
        right = pd.DataFrame({"d": [1, 1, 2, 2], "p": [1.0, 3.0, 5.0, 7.0]})
        out = aggregate_and_join(left, right, "d", {"p": ["mean", "max"]}, prefix="t_")
        assert len(out) == 2
        assert out.loc[out["d"] == 1, "t_p_mean"].iloc[0] == pytest.approx(2.0)


class TestReshaping:
    def test_flatten_multiindex_columns(self):
        df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1.0, 2.0, 3.0]})
        agg = df.groupby("g").agg({"v": ["mean", "std"]})
        assert list(flatten_multiindex(agg).columns) == ["v_mean", "v_std"]

    def test_dates_in_columns_melt_to_long(self):
        """The canonical untidy shape: one column per period."""
        df = pd.DataFrame({"name": ["a", "b"], "2026-04-07": [1.0, 2.0],
                           "2026-04-08": [3.0, 4.0]})
        out = dates_in_columns_to_long(df, "name")
        assert len(out) == 4
        assert str(out["date"].dtype).startswith("datetime64")
        assert set(out.columns) == {"name", "date", "value"}

    def test_melt_raises_when_no_date_columns_match(self):
        with pytest.raises(ValueError, match="no columns matched"):
            dates_in_columns_to_long(pd.DataFrame({"a": [1], "b": [2]}), "a")

    def test_long_to_wide_handles_duplicates_via_aggfunc(self):
        """`pivot` would raise here; `pivot_table` forces you to say what to do."""
        df = pd.DataFrame({"i": [1, 1, 2], "c": ["x", "x", "y"], "v": [1.0, 3.0, 5.0]})
        out = long_to_wide(df, index="i", columns="c", values="v", aggfunc="mean")
        assert out.loc[out["i"] == 1, "x"].iloc[0] == pytest.approx(2.0)
