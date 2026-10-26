"""Fold-safe encoders and classification diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score

from rvlab.evaluate.classification import (
    best_threshold, calibration_report, decile_table, imbalance_report, threshold_sweep,
)
from rvlab.features.encoding import (
    CVTargetEncoder, CyclicalEncoder, FrequencyEncoder, RareCategoryGrouper,
)


class TestCVTargetEncoder:
    def test_out_of_fold_encoding_leaks_nothing(self):
        """The property the class exists for.

        A high-cardinality id against a *random* target contains no information.
        A naive full-sample target encoding nonetheless produces strong
        cross-validated R², because each row's encoding contains its own label.
        The out-of-fold version must score ~0.
        """
        rng = np.random.default_rng(0)
        n = 2000
        cat = pd.DataFrame({"id": rng.integers(0, 600, n).astype(str)})
        y = rng.normal(size=n)

        leaky = cat.assign(enc=pd.Series(y).groupby(cat["id"]).transform("mean"))[["enc"]]
        safe = CVTargetEncoder(columns=["id"], cv=5, random_state=0).fit_transform(cat, y)

        assert cross_val_score(Ridge(), leaky, y, cv=5).mean() > 0.15
        assert abs(cross_val_score(Ridge(), safe, y, cv=5).mean()) < 0.05

    def test_encoding_recovers_a_real_category_effect(self):
        """It must not be so conservative that it destroys genuine signal."""
        rng = np.random.default_rng(1)
        n = 3000
        ids = rng.integers(0, 30, n)
        effect = rng.normal(0, 2.0, 30)
        y = effect[ids] + rng.normal(0, 0.5, n)
        cat = pd.DataFrame({"id": ids.astype(str)})

        encoded = CVTargetEncoder(columns=["id"], cv=5, random_state=0).fit_transform(cat, y)
        assert cross_val_score(LinearRegression(), encoded, y, cv=5).mean() > 0.7

    def test_smoothing_shrinks_rare_categories_toward_the_mean(self):
        y = np.r_[np.full(200, 0.0), [10.0]]
        cat = pd.DataFrame({"id": ["common"] * 200 + ["seen_once"]})
        enc = CVTargetEncoder(columns=["id"], smoothing=50.0, cv=3, random_state=0).fit(cat, y)
        # The rare category's full-training mapping is pulled toward the global mean.
        assert enc.mapping_["id"]["seen_once"] < 1.0

    def test_unseen_category_falls_back_to_the_global_mean(self):
        rng = np.random.default_rng(2)
        cat = pd.DataFrame({"id": rng.choice(list("abc"), 300)})
        y = rng.normal(size=300)
        enc = CVTargetEncoder(columns=["id"], cv=3, random_state=0).fit(cat, y)
        out = enc.transform(pd.DataFrame({"id": ["never_seen"]}))
        assert out["id"].iloc[0] == pytest.approx(enc.global_mean_)

    def test_fit_transform_and_transform_differ_on_training_rows(self):
        """fit_transform must return out-of-fold values, not the full mapping."""
        rng = np.random.default_rng(3)
        cat = pd.DataFrame({"id": rng.integers(0, 50, 500).astype(str)})
        y = rng.normal(size=500)
        enc = CVTargetEncoder(columns=["id"], cv=5, random_state=0)
        oof = enc.fit_transform(cat, y)
        full = enc.transform(cat.reset_index(drop=True).copy().set_index(
            pd.RangeIndex(1, 501)))
        assert not np.allclose(oof["id"].to_numpy(), full["id"].to_numpy())


class TestOtherEncoders:
    def test_rare_grouper_folds_infrequent_levels(self):
        df = pd.DataFrame({"c": ["a"] * 90 + ["b"] * 8 + list("cdefghij")[:2]})
        out = RareCategoryGrouper(columns=["c"], min_frequency=0.05).fit_transform(df)
        assert set(out["c"].unique()) == {"a", "b", "__rare__"}

    def test_rare_grouper_maps_unseen_levels_to_rare(self):
        train = pd.DataFrame({"c": ["a"] * 50 + ["b"] * 50})
        grouper = RareCategoryGrouper(columns=["c"], min_frequency=0.1).fit(train)
        assert grouper.transform(pd.DataFrame({"c": ["zzz"]}))["c"].iloc[0] == "__rare__"

    def test_frequency_encoder_is_target_free(self):
        df = pd.DataFrame({"c": ["a"] * 75 + ["b"] * 25})
        out = FrequencyEncoder(columns=["c"]).fit_transform(df)
        assert out["c"].iloc[0] == pytest.approx(0.75)
        assert out["c"].iloc[-1] == pytest.approx(0.25)

    def test_cyclical_encoder_keeps_december_next_to_january(self):
        """The whole point: an integer encoding says they are 11 apart."""
        df = pd.DataFrame({"month": [12, 1, 6]})
        out = CyclicalEncoder({"month": 12}).fit_transform(df)
        dist = lambda i, j: np.hypot(out["month_sin"].iloc[i] - out["month_sin"].iloc[j],
                                     out["month_cos"].iloc[i] - out["month_cos"].iloc[j])
        assert dist(0, 1) < dist(0, 2)
        assert "month" not in out.columns


class TestClassificationDiagnostics:
    @staticmethod
    def _problem(weight=0.9, seed=0):
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        X, y = make_classification(4000, 10, weights=[weight], random_state=seed)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=seed)
        proba = LogisticRegression(max_iter=1000).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
        return y_te, proba

    def test_imbalance_report_is_all_numeric(self):
        """So it can be rounded and concatenated like data."""
        report = imbalance_report(self._problem()[0])
        assert report.dtype.kind == "f"
        assert report["majority_accuracy"] > 0.8

    def test_cost_asymmetry_moves_the_threshold_down(self):
        """If a miss costs ten times a false alarm, the cut is not 0.5."""
        y, p = self._problem()
        symmetric = best_threshold(y, p, "expected_cost", 1.0, 1.0)["threshold"]
        asymmetric = best_threshold(y, p, "expected_cost", 1.0, 10.0)["threshold"]
        assert asymmetric < symmetric

    def test_threshold_sweep_recall_is_monotone(self):
        y, p = self._problem()
        sweep = threshold_sweep(y, p)
        assert sweep["recall"].is_monotonic_decreasing

    def test_calibration_slope_is_one_for_a_calibrated_model(self):
        y, p = self._problem()
        assert calibration_report(y, p)["calibration_slope"] == pytest.approx(1.0, abs=0.25)

    def test_calibration_slope_detects_overconfidence(self):
        """Inflating the log-odds by 3 must show a slope near 1/3."""
        y, p = self._problem()
        p = np.clip(p, 1e-6, 1 - 1e-6)
        over = 1 / (1 + np.exp(-3 * np.log(p / (1 - p))))
        slope = calibration_report(y, over)["calibration_slope"]
        assert slope == pytest.approx(1 / 3, abs=0.15)

    def test_decile_lift_is_high_in_the_top_bin(self):
        y, p = self._problem()
        table = decile_table(y, p)
        assert table["lift"].iloc[-1] > 2.0
        assert table["cumulative_capture"].iloc[-1] == pytest.approx(1.0, abs=0.05)
