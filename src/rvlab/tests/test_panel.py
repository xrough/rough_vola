"""Panel loading, feature builders, ranking metrics and the inference contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rvlab.data.panel import (
    audit_panel, concat_folders, leakage_audit, load_panel, synthetic_equity_panel,
)
from rvlab.evaluate.ranking import (
    add_prediction_rank, blend, cross_sectional_zscore, daily_rank_ic,
    information_ratio, ranking_report, spread_return_one_day, spread_sharpe,
)
from rvlab.features.panel import (
    add_cross_sectional, add_group_relative, add_horizon_returns,
    build_panel_features, make_panel_target,
)
from rvlab.pipelines.inference import (
    RollingHistoryBuffer, align_feature_columns, fill_missing_predictions,
    rank_for_submission,
)
from rvlab.pipelines.splits import describe_date_folds, purged_date_splits


@pytest.fixture(scope="module")
def panel():
    return synthetic_equity_panel(n_entities=60, n_dates=120, seed=0)


@pytest.fixture(scope="module")
def features(panel):
    return build_panel_features(panel)


class TestPanelData:
    def test_shape_and_keys_are_unique(self, panel):
        assert len(panel) == 60 * 120
        assert not panel.duplicated(["Date", "SecuritiesCode"]).any()

    def test_audit_passes_on_a_clean_panel(self, panel):
        assert audit_panel(panel).ok

    def test_audit_flags_duplicate_keys(self, panel):
        doubled = pd.concat([panel, panel.head(10)], ignore_index=True)
        audit = audit_panel(doubled)
        assert not audit.ok and "duplicate" in audit.problems[0]

    def test_planted_signal_is_predictable_from_the_past(self):
        """The reversal must be visible in ret_1d(t), not only in ret_1d(t+1).

        This is the property the generator exists for: a signal that loads on
        t+1 is real structure and useless, because it cannot be traded.
        """
        f = build_panel_features(synthetic_equity_panel(200, 300, seed=1,
                                                        reversal_strength=0.15))
        f = f.dropna(subset=["Target", "ret_1d"])
        ic = daily_rank_ic(f.assign(Prediction=-f["ret_1d"]))
        assert ic.mean() > 0.03, f"planted reversal not detectable: IC={ic.mean():.4f}"

    def test_null_panel_has_no_within_date_signal(self):
        """reversal_strength=0 must produce a panel with nothing to find.

        Checked *within date*: the pooled correlation is contaminated by the
        common market factor, whose lag-2 autocorrelation is nonzero by chance
        in any finite sample.
        """
        f = build_panel_features(synthetic_equity_panel(200, 300, seed=2,
                                                        reversal_strength=0.0))
        f = f.dropna(subset=["Target", "ret_1d"])
        ic = daily_rank_ic(f.assign(Prediction=-f["ret_1d"]))
        assert abs(ic.mean()) < 0.02, f"null panel shows IC={ic.mean():.4f}"

    def test_leakage_audit_finds_label_shaped_names(self):
        found = leakage_audit(["Close", "Target", "ret_1d", "bf25_fwd1", "y_label"])
        assert set(found) == {"Target", "bf25_fwd1", "y_label"}

    def test_leakage_audit_respects_the_allow_list(self):
        assert leakage_audit(["Target"], allow=("Target",)) == []

    def test_concat_folders_returns_none_when_absent(self, tmp_path):
        assert concat_folders(tmp_path, "stock_prices.csv") is None

    def test_load_panel_stamps_provenance(self):
        assert load_panel(n_entities=20, n_dates=40).attrs["provenance"]["synthetic"] is True

    def test_real_jpx_layout_canonicalizes_sector_and_builds_group_features(self, tmp_path):
        """The real JPX names must satisfy the same feature contract as fallback data."""
        source = synthetic_equity_panel(n_entities=4, n_dates=8, seed=7)
        train = tmp_path / "train_files"
        train.mkdir()

        static_cols = ["SecuritiesCode", "Sector", "MarketCapitalization", "IssuedShares"]
        static = source[static_cols].drop_duplicates("SecuritiesCode")
        static = static.rename(columns={"Sector": "33SectorName"})
        static.to_csv(tmp_path / "stock_list.csv", index=False)
        source.drop(columns=["Sector", "MarketCapitalization", "IssuedShares"]).to_csv(
            train / "stock_prices.csv", index=False)

        loaded = load_panel(root=tmp_path)
        assert loaded.attrs["provenance"]["synthetic"] is False
        assert loaded["Sector"].equals(loaded["33SectorName"])

        built = build_panel_features(loaded, horizons=(1,), windows=(2,))
        assert "ret_1d_vs_group" in built
        grouped = built.dropna(subset=["ret_1d_vs_group"]).groupby(["Date", "Sector"])
        assert grouped["ret_1d_vs_group"].mean().abs().max() < 1e-12


class TestPanelFeatures:
    def test_horizon_returns_are_within_entity(self, panel):
        """A 1-day return must never span two securities."""
        out = add_horizon_returns(panel, horizons=(1,))
        first_rows = out.sort_values(["SecuritiesCode", "Date"]).groupby(
            "SecuritiesCode").head(1)
        assert first_rows["ret_1d"].isna().all()

    def test_corporate_actions_mask_the_windows_they_touch(self):
        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=6),
            "SecuritiesCode": 1,
            "Close": [100.0, 101, 102, 51, 52, 53],
            "AdjustmentFactor": [1.0, 1, 1, 0.5, 1, 1],
        })
        from rvlab.features.panel import add_price_geometry
        out = add_horizon_returns(add_price_geometry(df), horizons=(1, 2))
        assert out["ret_1d"].isna().iloc[3], "return across the split was not masked"
        assert out["ret_2d"].isna().iloc[4], "2-day window containing the split not masked"

    def test_cross_sectional_rank_is_within_date(self, features):
        by_date = features.dropna(subset=["cs_rank_ret_1d"]).groupby("Date")["cs_rank_ret_1d"]
        assert by_date.max().max() <= 1.0 and by_date.min().min() >= 0.0
        # each date's ranks should span most of [0, 1]
        assert (by_date.max() - by_date.min()).median() > 0.9

    def test_group_relative_sums_to_zero_within_group_and_date(self, features):
        grouped = features.dropna(subset=["ret_1d_vs_group"]).groupby(["Date", "Sector"])
        assert grouped["ret_1d_vs_group"].mean().abs().max() < 1e-9

    def test_model_target_is_demeaned_within_date(self, features):
        assert features.groupby("Date")["ModelTarget"].mean().abs().max() < 1e-12

    def test_demeaning_preserves_the_within_date_ordering(self, features):
        """Ranking is invariant to removing a per-date constant — the reason
        de-meaning the target is safe for a ranking objective."""
        day = features[features["Date"] == features["Date"].iloc[500]].dropna(
            subset=["Target", "ModelTarget"])
        assert (day["Target"].rank().to_numpy()
                == day["ModelTarget"].rank().to_numpy()).all()


class TestRankingMetrics:
    @staticmethod
    def _frame(n_dates=40, n_entities=50, seed=0):
        rng = np.random.default_rng(seed)
        dates = np.repeat(pd.date_range("2024-01-01", periods=n_dates), n_entities)
        target = rng.normal(size=n_dates * n_entities)
        return pd.DataFrame({"Date": dates, "Target": target})

    def test_perfect_ranking_scores_ic_one(self):
        df = self._frame()
        rep = ranking_report(df, {"oracle": df["Target"]}, portfolio_size=10)
        assert rep.loc["oracle", "mean_ic"] == pytest.approx(1.0)
        assert rep.loc["oracle", "spread_sharpe"] > 0

    def test_inverted_ranking_is_the_exact_negative(self):
        df = self._frame()
        rep = ranking_report(df, {"good": df["Target"], "bad": -df["Target"]},
                             portfolio_size=10)
        assert rep.loc["bad", "mean_ic"] == pytest.approx(-rep.loc["good", "mean_ic"])

    def test_random_predictions_score_near_zero(self):
        df = self._frame()
        noise = np.random.default_rng(9).normal(size=len(df))
        assert abs(ranking_report(df, {"noise": noise}, portfolio_size=10)
                   .loc["noise", "mean_ic"]) < 0.1

    def test_ranking_metrics_ignore_a_monotone_rescaling(self):
        """The property that makes RMSE the wrong metric here."""
        df = self._frame()
        rep = ranking_report(df, {"raw": df["Target"],
                                  "scaled": 100 * df["Target"] + 7},
                             portfolio_size=10)
        assert rep.loc["raw", "mean_ic"] == pytest.approx(rep.loc["scaled", "mean_ic"])
        assert rep.loc["raw", "spread_sharpe"] == pytest.approx(
            rep.loc["scaled", "spread_sharpe"])
        assert rep.loc["raw", "rmse"] != pytest.approx(rep.loc["scaled", "rmse"])

    def test_rank_is_unique_and_zero_based(self):
        df = self._frame(n_dates=3, n_entities=20)
        df["Prediction"] = 0.0                      # every prediction identical
        ranked = add_prediction_rank(df)
        for _, day in ranked.groupby("Date"):
            assert sorted(day["Rank"]) == list(range(20))

    def test_spread_return_rewards_the_right_side(self):
        day = pd.DataFrame({"Target": [5.0, 4, 3, -3, -4, -5], "Rank": range(6)})
        assert spread_return_one_day(day, portfolio_size=2) > 0
        flipped = day.assign(Rank=day["Rank"].to_numpy()[::-1])
        assert spread_return_one_day(flipped, portfolio_size=2) < 0

    def test_information_ratio_fields(self):
        ic = pd.Series(np.random.default_rng(0).normal(0.05, 0.1, 200))
        stats = information_ratio(ic)
        assert stats["icir"] == pytest.approx(ic.mean() / ic.std(ddof=1))
        assert 0 <= stats["hit_rate"] <= 1

    def test_zscore_is_per_date_and_blend_is_scale_free(self):
        df = self._frame(n_dates=5, n_entities=30)
        z = cross_sectional_zscore(df["Target"], df["Date"])
        assert abs(pd.Series(z).groupby(df["Date"].to_numpy()).mean()).max() < 1e-9
        big = blend({"a": df["Target"], "b": 1000 * df["Target"]}, df["Date"])
        small = blend({"a": df["Target"], "b": df["Target"]}, df["Date"])
        assert np.allclose(big, small)


class TestPurgedDateSplits:
    def test_gap_is_respected_and_folds_do_not_overlap(self, panel):
        folds = purged_date_splits(panel, n_splits=3, gap_days=2)
        table = describe_date_folds(folds, panel)
        assert not table["overlap"].any()
        assert (table["gap_days"] >= 2).all()

    def test_larger_gap_shrinks_training(self, panel):
        small = describe_date_folds(purged_date_splits(panel, 3, gap_days=0), panel)
        large = describe_date_folds(purged_date_splits(panel, 3, gap_days=10), panel)
        assert (large["n_train"] <= small["n_train"]).all()
        assert (large["n_train"] < small["n_train"]).any()

    def test_no_date_appears_in_both_sides(self, panel):
        dates = panel["Date"].to_numpy()
        for train_idx, val_idx in purged_date_splits(panel, 3, gap_days=2):
            assert not set(dates[train_idx]) & set(dates[val_idx])


class TestInference:
    def test_align_adds_missing_and_orders_columns(self):
        df = pd.DataFrame({"b": [1], "a": [2], "extra": [3]})
        out = align_feature_columns(df, ["a", "b", "c"])
        assert list(out.columns) == ["a", "b", "c"]
        assert out["c"].isna().all()

    def test_buffer_trims_to_max_days(self, panel):
        buffer = RollingHistoryBuffer(panel, max_days=10)
        assert buffer.n_dates == 10

    def test_buffer_append_keeps_the_cap_and_dedupes(self, panel):
        early = panel[panel["Date"] < panel["Date"].unique()[100]]
        buffer = RollingHistoryBuffer(early, max_days=5)
        new_day = panel[panel["Date"] == panel["Date"].unique()[100]]
        buffer.append(new_day)
        assert buffer.n_dates == 5
        buffer.append(new_day)                       # a re-send must not duplicate
        assert not buffer.frame.duplicated(["Date", "SecuritiesCode"]).any()

    def test_submission_ranks_are_unique_and_preserve_row_order(self):
        sample = pd.DataFrame({"Date": ["2024-01-01"] * 5,
                               "SecuritiesCode": [5, 3, 1, 4, 2]})
        out = rank_for_submission(sample, [0.1, 0.9, 0.5, 0.2, 0.7])
        assert list(out["SecuritiesCode"]) == [5, 3, 1, 4, 2]
        assert sorted(out["Rank"]) == list(range(5))
        assert out.loc[out["Rank"] == 0, "SecuritiesCode"].iloc[0] == 3

    def test_missing_predictions_fall_back_to_the_daily_median(self):
        dates = ["a", "a", "a", "b", "b"]
        filled = fill_missing_predictions([1.0, 3.0, np.nan, np.nan, np.nan], dates)
        assert filled.iloc[2] == pytest.approx(2.0)
        assert filled.iloc[3] == 0.0


class TestSyntheticOHLC:
    """The bars must contain a coherent intraday path, not noise around the close.

    Independently-drawn High/Low satisfy High >= max(Open, Close) while still being
    far too narrow for their own returns — so every range-based volatility estimator
    understates by a large factor, silently. These assertions are what make
    notebook 18's estimator comparison meaningful.
    """

    @staticmethod
    def _bars(seed=1, n_entities=60, n_dates=500):
        return synthetic_equity_panel(n_entities=n_entities, n_dates=n_dates, seed=seed)

    def test_ohlc_bounds_hold_exactly(self):
        p = self._bars()
        assert (p["High"] >= p[["Open", "Close"]].max(axis=1) - 1e-12).all()
        assert (p["Low"] <= p[["Open", "Close"]].min(axis=1) + 1e-12).all()
        assert (p["High"] >= p["Low"]).all()

    def test_range_is_wide_enough_for_its_own_returns(self):
        """log(High/Low) should be ~1.5-2.5x the absolute daily return.

        A Brownian path has E[range]/E|endpoint| = sqrt(8/pi)/sqrt(2/pi) = 2. The
        old generator produced 0.52, which is the defect this guards against.
        """
        p = self._bars().sort_values(["SecuritiesCode", "Date"])
        ret = np.abs(np.log1p(p.groupby("SecuritiesCode")["Close"].pct_change())).dropna()
        ratio = np.log(p["High"] / p["Low"]).median() / ret.median()
        assert 1.2 < ratio < 3.0, f"range/|return| = {ratio:.2f}, bars are inconsistent"

    def test_yang_zhang_recovers_close_to_close_volatility(self):
        """Yang-Zhang is the only estimator here that models the overnight gap, so
        it is the only one expected to land near the close-to-close volatility."""
        ratios = []
        for _, g in self._bars().groupby("SecuritiesCode"):
            g = g.sort_values("Date")
            O, H, L, C = (g[c].to_numpy() for c in ("Open", "High", "Low", "Close"))
            Cp, O, H, L, C = C[:-1], O[1:], H[1:], L[1:], C[1:]
            n = len(C)
            cc = np.log(C / Cp).std(ddof=1)
            rs = np.mean(np.log(H / C) * np.log(H / O) + np.log(L / C) * np.log(L / O))
            k = 0.34 / (1.34 + (n + 1) / (n - 1))
            yz = np.sqrt(np.log(O / Cp).var(ddof=1) + k * np.log(C / O).var(ddof=1)
                         + (1 - k) * rs)
            ratios.append(yz / cc)
        median = float(np.median(ratios))
        assert 0.85 < median < 1.15, f"Yang-Zhang / close-to-close = {median:.3f}"

    def test_pure_range_estimators_sit_below_yang_zhang(self):
        """Parkinson/Rogers-Satchell see only the intraday leg, so they must come in
        below Yang-Zhang, which adds the overnight variance. Ordering, not level."""
        park, rogers, yang = [], [], []
        for _, g in self._bars().groupby("SecuritiesCode"):
            g = g.sort_values("Date")
            O, H, L, C = (g[c].to_numpy() for c in ("Open", "High", "Low", "Close"))
            Cp, O, H, L, C = C[:-1], O[1:], H[1:], L[1:], C[1:]
            n = len(C)
            park.append(np.sqrt(np.mean(np.log(H / L) ** 2) / (4 * np.log(2))))
            rs = np.mean(np.log(H / C) * np.log(H / O) + np.log(L / C) * np.log(L / O))
            rogers.append(np.sqrt(rs))
            k = 0.34 / (1.34 + (n + 1) / (n - 1))
            yang.append(np.sqrt(np.log(O / Cp).var(ddof=1)
                                + k * np.log(C / O).var(ddof=1) + (1 - k) * rs))
        assert np.median(park) < np.median(yang)
        assert np.median(rogers) < np.median(yang)

    def test_close_is_invariant_to_the_ohlc_parameters(self):
        """Open/High/Low derive from Close; changing how they are built must not
        move Close itself, or every Track B result silently shifts underneath.

        Asserted by varying `gap_share`, which reshapes the intraday path
        completely while leaving the daily return sequence alone.
        """
        base = synthetic_equity_panel(n_entities=20, n_dates=60, seed=3)
        wider = synthetic_equity_panel(n_entities=20, n_dates=60, seed=3, gap_share=0.6)

        assert np.allclose(base["Close"], wider["Close"], rtol=0, atol=0)
        assert np.allclose(base["Target"].fillna(0), wider["Target"].fillna(0),
                           rtol=0, atol=0)
        # ...while the bars themselves genuinely differ.
        assert not np.allclose(base["Open"], wider["Open"])
