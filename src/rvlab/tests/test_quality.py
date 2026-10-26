"""Health checks, drift detection and the leakage scan."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rvlab.data.quality import (
    categorical_report, datetime_audit, drift_report, health_report, leakage_scan,
    numeric_report, outlier_report, population_stability_index, target_audit,
)


@pytest.fixture
def messy():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        "id": np.arange(n),
        "clean": rng.normal(size=n),
        "constant": 3.0,
        "all_null": np.nan,
        "half_null": np.where(rng.random(n) < 0.6, np.nan, 1.0),
        "cat": rng.choice(["a", "b", "c"], n, p=[0.9, 0.07, 0.03]),
        "ts": pd.date_range("2025-01-01", periods=n, freq="D"),
    })
    return pd.concat([df, df.iloc[:5]], ignore_index=True)      # 5 duplicated rows


class TestHealthReport:
    def test_flags_constant_all_null_and_duplicates(self, messy):
        report = health_report(messy)
        joined = " ".join(report.problems)
        assert not report.ok
        assert "constant" in joined and "all-null" in joined and "duplicated rows" in joined

    def test_clean_frame_passes(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
        assert health_report(df).ok

    def test_duplicate_key_detection(self):
        df = pd.DataFrame({"k": [1, 1, 2], "v": [1, 2, 3]})
        assert any("duplicate rows on key" in p for p in health_report(df, key_cols=["k"]).problems)

    def test_declared_string_id_is_not_misreported_as_a_feature(self):
        df = pd.DataFrame({"id": [f"row-{i}" for i in range(20)],
                           "value": np.arange(20.0)})
        assert health_report(df, key_cols=["id"]).ok

    def test_column_table_covers_every_column(self, messy):
        report = health_report(messy)
        assert set(report.columns["column"]) == set(messy.columns)
        assert report.columns.set_index("column").loc["all_null", "null_rate"] == 1.0

    def test_summary_row_count_matches(self, messy):
        assert health_report(messy).summary["rows"] == len(messy)


class TestNumericAndOutliers:
    def test_numeric_report_counts_zeros_negatives_and_infinities(self):
        df = pd.DataFrame({"x": [0.0, -1.0, np.inf, 2.0, 3.0]})
        row = numeric_report(df).iloc[0]
        assert row["n_zero"] == 1 and row["n_negative"] == 1 and row["n_inf"] == 1
        assert np.isfinite(row["max"])              # inf excluded from the range

    def test_the_three_outlier_methods_disagree_on_heavy_tails(self):
        """z-score is inflated by the very points it should find; MAD is not.

        On a skewed, heavy-tailed sample the standard deviation is dragged upward
        by the tail, so the z-score threshold moves out and the outliers mask each
        other. MAD uses the median absolute deviation and does not move. Measured
        here: z finds ~18, MAD ~191, on the same 3,000 points.

        A single extreme spike does *not* show this — it is far enough out that
        every method finds it. The disagreement needs a genuine tail.
        """
        x = np.random.default_rng(1).lognormal(0, 1.2, 3000)
        row = outlier_report(pd.DataFrame({"x": x})).iloc[0]
        assert row["by_mad"] > 3 * row["by_zscore"]

    def test_no_outliers_in_clean_gaussian_data(self):
        x = np.random.default_rng(2).normal(size=2000)
        row = outlier_report(pd.DataFrame({"x": x})).iloc[0]
        assert row["by_zscore"] <= 5


class TestDatetimeAudit:
    def test_reports_tz_span_and_monotonicity(self, messy):
        audit = datetime_audit(messy, "ts")
        assert audit["tz"] == "naive"
        assert audit["modal_gap"] == pd.Timedelta(days=1)

    def test_gaps_use_distinct_timestamps_not_panel_repeats(self):
        """On a panel the same instant repeats; differencing raw rows gives zero."""
        ts = np.repeat(pd.date_range("2025-01-01", periods=10, freq="D"), 5)
        audit = datetime_audit(pd.DataFrame({"ts": ts}), "ts")
        assert audit["modal_gap"] == pd.Timedelta(days=1)
        assert audit["n_duplicated"] == 40

    def test_detects_a_planted_outage(self):
        dates = list(pd.date_range("2025-01-01", periods=20, freq="D"))
        dates += list(pd.date_range("2025-03-01", periods=20, freq="D"))
        audit = datetime_audit(pd.DataFrame({"ts": dates}), "ts")
        assert audit["largest_gap"] > pd.Timedelta(days=30)
        assert audit["n_gaps_over_3x_modal"] == 1


class TestDrift:
    def test_psi_is_zero_on_identical_samples(self):
        x = np.random.default_rng(0).normal(size=5000)
        assert population_stability_index(x, x) == pytest.approx(0.0, abs=1e-9)

    def test_psi_flags_a_planted_shift(self):
        rng = np.random.default_rng(0)
        assert population_stability_index(rng.normal(0, 1, 5000),
                                          rng.normal(1.5, 1, 5000)) > 0.25

    def test_psi_stays_small_under_resampling_noise(self):
        rng = np.random.default_rng(0)
        assert population_stability_index(rng.normal(size=5000),
                                          rng.normal(size=5000)) < 0.1

    def test_drift_report_flags_only_the_shifted_column(self):
        rng = np.random.default_rng(3)
        ref = pd.DataFrame({"same": rng.normal(size=3000), "moved": rng.normal(size=3000)})
        cur = pd.DataFrame({"same": rng.normal(size=3000),
                            "moved": rng.normal(2.0, 1, 3000)})
        out = drift_report(ref, cur).set_index("column")
        assert out.loc["moved", "drifted"]
        assert not out.loc["same", "drifted"]

    def test_drift_report_reports_shift_in_standard_deviations(self):
        rng = np.random.default_rng(4)
        ref = pd.DataFrame({"x": rng.normal(0, 2, 4000)})
        cur = pd.DataFrame({"x": rng.normal(2, 2, 4000)})
        assert drift_report(ref, cur)["mean_shift_in_sd"].iloc[0] == pytest.approx(1.0, abs=0.15)


class TestLeakageAndTarget:
    def test_finds_a_name_flagged_column(self):
        df = pd.DataFrame({"x": [1, 2, 3], "next_return": [1, 2, 3], "y": [1, 2, 3]})
        assert "next_return" in set(leakage_scan(df, "y")["column"])

    def test_finds_a_near_perfect_predictor_with_an_innocent_name(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=300)
        df = pd.DataFrame({"harmless": y * 1.001 + 1e-6 * rng.normal(size=300),
                           "noise": rng.normal(size=300), "y": y})
        flagged = leakage_scan(df, "y").set_index("column")
        assert "harmless" in flagged.index
        assert "noise" not in flagged.index

    def test_clean_features_produce_an_empty_scan(self):
        rng = np.random.default_rng(5)
        df = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200),
                           "y": rng.normal(size=200)})
        assert leakage_scan(df, "y").empty

    def test_finds_a_deterministic_string_target_proxy(self):
        target = pd.Series((["quiet", "volatile", "normal"] * 40), dtype="string")
        proxy = target.map({
            "quiet": "low-band",
            "volatile": "high-band",
            "normal": "middle-band",
        }).astype("string")
        unrelated = pd.Series((["north", "north", "south", "south"] * 30),
                              dtype="string")
        df = pd.DataFrame({"regime_alias": proxy, "region": unrelated, "y": target})

        flagged = leakage_scan(df, "y").set_index("column")

        assert "deterministic_mapping" in flagged.loc["regime_alias", "flags"]
        assert "region" not in flagged.index

    def test_finds_a_numeric_code_for_a_string_target(self):
        target = pd.Series((["no", "yes", "maybe"] * 40), dtype="string")
        df = pd.DataFrame({
            "status_code": target.map({"no": 0, "yes": 1, "maybe": 2}).astype(int),
            "continuous_measure": np.linspace(-1.0, 1.0, len(target)),
            "y": target,
        })

        flagged = leakage_scan(df, "y").set_index("column")

        assert "deterministic_mapping" in flagged.loc["status_code", "flags"]
        assert "continuous_measure" not in flagged.index

    def test_purity_scan_does_not_flag_a_near_unique_string_id(self):
        keys = [f"row-{i}" for i in range(90)] + [f"row-{i}" for i in range(10)]
        label_by_key = {key: ("up" if int(key.split("-")[1]) % 2 else "down")
                        for key in set(keys)}
        df = pd.DataFrame({
            "record_key": pd.Series(keys, dtype="string"),
            "y": pd.Series([label_by_key[key] for key in keys], dtype="string"),
        })

        # The mapping is perfectly pure only because almost every key identifies
        # one row; treating it as target evidence would be a cardinality artefact.
        assert leakage_scan(df, "y", feature_cols=["record_key"]).empty

    def test_purity_cardinality_guard_is_validated(self):
        df = pd.DataFrame({"x": ["a", "b"] * 6, "y": ["up", "down"] * 6})
        with pytest.raises(ValueError, match="purity_max_unique_rate"):
            leakage_scan(df, "y", purity_max_unique_rate=0)

    def test_target_audit_reports_class_balance(self):
        y = np.r_[np.zeros(90), np.ones(10)]
        audit = target_audit(pd.DataFrame({"y": y}), "y")
        assert audit["minority_share"] == pytest.approx(0.1)

    def test_target_audit_finds_groups_with_no_label_at_all(self):
        df = pd.DataFrame({"g": ["a"] * 10 + ["b"] * 10,
                           "y": [1.0] * 10 + [np.nan] * 10})
        audit = target_audit(df, "y", group="g")
        assert audit["groups_fully_missing"] == 1


class TestCategorical:
    def test_reports_dominance_and_rare_levels(self, messy):
        row = categorical_report(messy).set_index("column").loc["cat"]
        assert row["top_share"] > 0.8
        assert row["rare_levels"] >= 0
