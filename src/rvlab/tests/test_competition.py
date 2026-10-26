"""Executable contracts for the competition-analysis backbone."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rvlab.competition as competition_module
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rvlab.competition import (
    ProblemSpec,
    adversarial_validation_report,
    baseline_ledger,
    build_run_manifest,
    code_fingerprint,
    data_fingerprint,
    export_run_bundle,
    fold_diagnostics,
    heldout_permutation_importance,
    make_competition_demo,
    make_final_holdout,
    make_folds,
    materialize_folds,
    null_control_report,
    oof_ledger,
    resolve_notebook_source,
    score_ledger,
    score_predictions,
    smoke_test_condition_matrix,
    validate_competition_data,
    validate_development_target_proxies,
    validate_folds,
    validate_submission,
)


def _demo_spec(profile: str, *, n_splits: int = 4) -> ProblemSpec:
    common = dict(target="target", id_columns=("row_id",), n_splits=n_splits)
    if profile == "iid":
        return ProblemSpec(geometry="iid", task="regression", **common)
    if profile == "grouped":
        return ProblemSpec(
            geometry="grouped", task="binary", group_column="group", **common)
    if profile == "time":
        return ProblemSpec(
            geometry="time",
            task="regression",
            time_column="time",
            test_time_relation="future",
            label_horizon=1,
            feature_lookback=2,
            **common,
        )
    if profile == "panel":
        return ProblemSpec(
            geometry="panel",
            task="ranking",
            time_column="time",
            group_column="group",
            test_time_relation="future",
            label_horizon=1,
            feature_lookback=2,
            submission_transform="rank_descending_zero",
            ranking_target_order="higher_is_better",
            **common,
        )
    raise AssertionError(profile)


class TestProblemSpec:
    def test_is_frozen_and_derives_safe_defaults(self):
        spec = ProblemSpec(
            geometry="time",
            task="regression",
            time_column="time",
            test_time_relation="future",
            label_horizon=2,
            feature_lookback=5,
            id_columns=["row_id"],
        )
        assert spec.id_columns == ("row_id",)
        assert spec.purge_gap == 2
        assert spec.metric == "root_mean_squared_error"
        assert spec.prediction_column == "target"
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.n_splits = 3

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"geometry": "grouped", "task": "binary"}, "group_column"),
            ({"geometry": "time", "task": "regression"}, "time_column"),
            ({"geometry": "panel", "task": "ranking", "time_column": "time"},
             "group_column"),
            ({"geometry": "iid", "task": "ranking"}, "ranking"),
            ({"geometry": "time", "task": "ranking", "time_column": "time",
              "test_time_relation": "future"},
             "grouped or panel"),
            ({"geometry": "grouped", "task": "regression", "group_column": "entity",
              "id_columns": ("entity",)}, "role columns must be distinct"),
            ({"geometry": "iid", "task": "regression", "feature_lookback": 1},
             "only apply"),
        ],
    )
    def test_invalid_designs_fail_at_construction(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            ProblemSpec(**kwargs)

    def test_explicit_gap_cannot_understate_overlap(self):
        with pytest.raises(ValueError, match="unsafe"):
            ProblemSpec(
                geometry="time",
                task="regression",
                time_column="time",
                test_time_relation="future",
                label_horizon=5,
                feature_lookback=20,
                gap=4,
            )

    def test_past_only_lookback_does_not_inflate_the_purge(self):
        spec = ProblemSpec(
            geometry="time",
            task="regression",
            time_column="time",
            test_time_relation="future",
            label_horizon=2,
            feature_lookback=50,
        )
        assert spec.purge_gap == 2

    def test_metric_and_direction_are_a_checked_pair(self):
        spec = ProblemSpec(
            geometry="iid", task="binary", metric="average_precision")
        assert spec.metric_direction == "maximize"
        with pytest.raises(ValueError, match="unsupported"):
            ProblemSpec(geometry="iid", task="binary", metric="accuracy")
        with pytest.raises(ValueError, match="must be minimize"):
            ProblemSpec(
                geometry="iid",
                task="regression",
                metric="root_mean_squared_error",
                metric_direction="maximize",
            )

    def test_rank_submission_transform_is_ranking_only(self):
        with pytest.raises(ValueError, match="task='ranking'"):
            ProblemSpec(
                geometry="iid",
                task="regression",
                submission_transform="rank_descending_zero",
            )

    def test_chronology_and_ranking_order_must_be_explicit(self):
        with pytest.raises(ValueError, match="test_time_relation"):
            ProblemSpec(
                geometry="time", task="regression", time_column="time")
        with pytest.raises(ValueError, match="ranking_target_order"):
            ProblemSpec(
                geometry="grouped", task="ranking", group_column="query")

    def test_numeric_positive_label_must_be_finite(self):
        with pytest.raises(ValueError, match="positive_label must be finite"):
            ProblemSpec(
                geometry="iid", task="binary", positive_label=np.inf)
        with pytest.raises(ValueError, match="applies only"):
            ProblemSpec(
                geometry="iid", task="regression",
                ranking_target_order="higher_is_better")


class TestDemoProfiles:
    @pytest.mark.parametrize("profile", ["iid", "grouped", "time", "panel"])
    def test_demo_has_stable_swap_data_contract(self, profile):
        train, test, sample = make_competition_demo(profile, random_state=7)
        again = make_competition_demo(profile, random_state=7)

        assert len(train) == 240 and len(test) == len(sample) == 80
        assert {"row_id", "x_num", "x_aux", "category", "target"} <= set(train)
        assert "target" not in test
        assert sample.columns.tolist() == ["row_id", "target"]
        assert sample["row_id"].tolist() == test["row_id"].tolist()
        assert train["x_aux"].isna().any() and test["x_aux"].isna().any()
        assert train["category"].isna().any() and test["category"].isna().any()
        assert train.attrs["synthetic"] is True
        pd.testing.assert_frame_equal(train, again[0])
        pd.testing.assert_frame_equal(test, again[1])

        if profile in {"grouped", "panel"}:
            assert "group" in train and "group" in test
        if profile in {"time", "panel"}:
            assert "time" in train and "time" in test
            assert train["time"].max() < test["time"].min()


class TestSplitRouting:
    @pytest.mark.parametrize("profile", ["iid", "grouped", "time", "panel"])
    def test_all_geometries_route_to_valid_folds(self, profile):
        train, _, _ = make_competition_demo(profile, random_state=11)
        spec = _demo_spec(profile)
        folds = make_folds(train, spec)
        report = fold_diagnostics(train, folds, spec)

        assert len(folds) == spec.n_splits
        assert (report["row_overlap"] == 0).all()
        validation_rows = np.concatenate([valid for _, valid in folds])
        assert len(np.unique(validation_rows)) == len(validation_rows)

        if profile in {"iid", "grouped"}:
            assert sorted(validation_rows.tolist()) == list(range(len(train)))
        if profile == "grouped":
            assert (report["group_overlap"] == 0).all()
        if profile in {"time", "panel"}:
            assert (report["time_overlap"] == 0).all()
            assert (report["train_time_end"] < report["valid_time_start"]).all()
            assert (report["gap"] >= spec.purge_gap).all()
        if profile == "panel":
            # Entities are expected on both sides; dates, not entities, define
            # the independent unit for panel validation.
            assert (report["group_overlap"] > 0).all()

    def test_group_overlap_is_rejected_even_when_rows_are_disjoint(self):
        train, _, _ = make_competition_demo("grouped")
        spec = _demo_spec("grouped")
        folds = list(make_folds(train, spec))
        fold_train, fold_valid = (part.copy() for part in folds[0])
        validation_groups = set(train.iloc[fold_valid]["group"])
        candidate = next(
            pos for pos in fold_train
            if train.iloc[pos]["group"] not in validation_groups)
        fold_train = fold_train[fold_train != candidate]
        fold_valid = np.r_[fold_valid, candidate]
        folds[0] = (fold_train, fold_valid)

        with pytest.raises(ValueError, match="groups"):
            validate_folds(train, folds, spec)

    def test_future_training_row_is_rejected(self):
        train, _, _ = make_competition_demo("time")
        spec = _demo_spec("time")
        folds = list(make_folds(train, spec))
        fold_train, fold_valid = (part.copy() for part in folds[0])
        future_row = int(folds[-1][1][-1])
        fold_train = np.r_[fold_train, future_row]
        folds[0] = (fold_train, fold_valid)

        with pytest.raises(ValueError, match="not chronological"):
            validate_folds(train, folds, spec)

    def test_too_small_gap_is_rejected(self):
        train, _, _ = make_competition_demo("time")
        spec = _demo_spec("time")
        folds = list(make_folds(train, spec))
        fold_train, fold_valid = (part.copy() for part in folds[0])
        first_valid_time = train.iloc[fold_valid]["time"].min()
        times = pd.Index(train["time"].drop_duplicates()).sort_values()
        immediate_predecessor = times[times.get_loc(first_valid_time) - 1]
        leaked_gap_rows = np.flatnonzero(train["time"].to_numpy() == immediate_predecessor)
        fold_train = np.r_[fold_train, leaked_gap_rows]
        folds[0] = (fold_train, fold_valid)

        with pytest.raises(ValueError, match="smaller than required"):
            validate_folds(train, folds, spec)

    def test_repeated_validation_row_is_rejected(self):
        train, _, _ = make_competition_demo("iid")
        spec = _demo_spec("iid")
        folds = list(make_folds(train, spec))
        repeated = int(folds[0][1][0])
        second_train, second_valid = (part.copy() for part in folds[1])
        second_train = second_train[second_train != repeated]
        second_valid = np.r_[second_valid, repeated]
        folds[1] = (second_train, second_valid)
        with pytest.raises(ValueError, match="more than one fold"):
            validate_folds(train, folds, spec)

    def test_time_axis_rejects_lexical_dates(self):
        train, _, _ = make_competition_demo("time")
        spec = _demo_spec("time")
        train = train.assign(time=train["time"].dt.strftime("%-m/%-d/%Y"))
        with pytest.raises(TypeError, match="datetime or numeric dtype"):
            make_folds(train, spec)

        train = train.assign(
            time=np.arange(len(train), dtype=float).astype(complex) + 1j)
        with pytest.raises(TypeError, match="datetime or numeric dtype"):
            make_folds(train, spec)

    def test_train_test_time_relation_is_explicitly_enforced_or_waived(self):
        train, test, sample = make_competition_demo("time")
        future_spec = _demo_spec("time")
        overlapping = test.copy()
        overlapping["time"] = train["time"].iloc[:len(overlapping)].to_numpy()
        availability = {"x_num": True, "x_aux": True}

        with pytest.raises(ValueError, match="test_time_relation='future'"):
            validate_competition_data(
                train, overlapping, sample, ["x_num", "x_aux"], future_spec,
                availability=availability)
        unrestricted = dataclasses.replace(
            future_spec, test_time_relation="unrestricted")
        report = validate_competition_data(
            train, overlapping, sample, ["x_num", "x_aux"], unrestricted,
            availability=availability)
        assert len(report) == 2

        timezone_aware = test.assign(time=test["time"].dt.tz_localize("UTC"))
        with pytest.raises(TypeError, match="timezone-naive.*timezone-aware"):
            validate_competition_data(
                train, timezone_aware, sample, ["x_num", "x_aux"],
                unrestricted, availability=availability)

        numeric_train = train.assign(time=np.arange(len(train), dtype=np.int64))
        numeric_test = test.assign(
            time=np.arange(len(train), len(train) + len(test), dtype=np.int64))
        validate_competition_data(
            numeric_train, numeric_test, sample, ["x_num", "x_aux"], future_spec,
            availability=availability)


class TestFinalHoldout:
    @pytest.mark.parametrize("profile", ["iid", "grouped", "time", "panel"])
    def test_holdout_respects_geometry(self, profile):
        train, _, _ = make_competition_demo(profile)
        spec = _demo_spec(profile)
        development, holdout = make_final_holdout(train, spec, fraction=0.2)

        assert len(development) and len(holdout)
        assert not set(development) & set(holdout)
        if profile == "grouped":
            assert not set(train.iloc[development]["group"]) & set(
                train.iloc[holdout]["group"])
        if profile in {"time", "panel"}:
            assert (train.iloc[development]["time"].max()
                    < train.iloc[holdout]["time"].min())
            times = pd.Index(train["time"].drop_duplicates()).sort_values()
            realised = (times.get_loc(train.iloc[holdout]["time"].min())
                        - times.get_loc(train.iloc[development]["time"].max()) - 1)
            assert realised >= spec.purge_gap

    def test_grouped_binary_holdout_contains_both_classes(self):
        train, _, _ = make_competition_demo("grouped")
        spec = _demo_spec("grouped")
        development, holdout = make_final_holdout(train, spec)
        assert train.iloc[development]["target"].nunique() == 2
        assert train.iloc[holdout]["target"].nunique() == 2


class TestPredictionLedgers:
    def test_mixed_pipeline_accepts_pandas_string_missing_values(self):
        from rvlab.pipelines import make_column_transformer

        frame = pd.DataFrame({
            "number": [1.0, np.nan, 3.0],
            "category": pd.array(["a", pd.NA, "b"], dtype="string"),
        })
        transformed = make_column_transformer(
            numeric=["number"], categorical=["category"]).fit_transform(frame)
        assert transformed.shape[0] == len(frame)
        assert not transformed.isna().any(axis=None)

    def test_binary_ledger_uses_positive_class_probability(self):
        train, _, _ = make_competition_demo("grouped", random_state=3)
        spec = _demo_spec("grouped")
        estimator = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=500, random_state=0),
        )
        ledger = oof_ledger(estimator, train, ["x_num", "x_aux"], spec)

        assert len(ledger) == len(train)
        assert ledger["row_position"].tolist() == list(range(len(train)))
        assert ledger["row_id"].tolist() == train["row_id"].tolist()
        assert ledger["prediction"].between(0, 1).all()
        assert not set(ledger["prediction"].unique()) <= {0.0, 1.0}

    def test_regression_ledger_allows_chronological_partial_coverage(self):
        train, _, _ = make_competition_demo("time")
        spec = _demo_spec("time")
        estimator = make_pipeline(SimpleImputer(strategy="median"), Ridge())
        ledger = oof_ledger(estimator, train, ["x_num", "x_aux"], spec)
        expected = sorted(np.concatenate([valid for _, valid in make_folds(train, spec)]))
        assert ledger["row_position"].tolist() == expected
        assert np.isfinite(ledger["prediction"]).all()

    def test_target_column_is_refused_as_a_feature(self):
        train, _, _ = make_competition_demo("iid")
        spec = _demo_spec("iid")
        with pytest.raises(ValueError, match="target leakage"):
            oof_ledger(Ridge(), train, ["x_num", "target"], spec)

    def test_baseline_is_fit_on_training_target_only(self):
        train, _, _ = make_competition_demo("iid")
        spec = _demo_spec("iid")
        folds = make_folds(train, spec)
        ledger = baseline_ledger(train, spec, folds=folds)
        for fold_number, (fit, valid) in enumerate(folds):
            expected = train.iloc[fit]["target"].mean()
            observed = ledger.loc[ledger["fold"] == fold_number, "prediction"]
            assert np.allclose(observed, expected)
            assert set(ledger.loc[ledger["fold"] == fold_number, "row_position"]) == set(valid)

    def test_score_report_keeps_fold_and_pooled_auc_separate(self):
        train, _, _ = make_competition_demo("grouped")
        spec = _demo_spec("grouped")
        baseline = baseline_ledger(train, spec)
        report = score_ledger(baseline, spec)
        assert report.fold_scores == (0.5, 0.5, 0.5, 0.5)
        assert report.cv_mean == 0.5
        assert report.pooled_oof != pytest.approx(report.cv_mean)
        assert report.direction == "maximize"

    @pytest.mark.parametrize(
        "truth, message",
        [([0, 1, None, 0], "missing"),
         ([0, 1, 2, 0], "exactly two classes"),
         ([0.0, 1.0, -np.inf, 0.0], "infinity"),
         ([np.inf, "yes", np.inf, "yes"], "infinity")],
    )
    def test_binary_scorer_rejects_invalid_truth_before_mapping(
            self, truth, message):
        spec = ProblemSpec(geometry="iid", task="binary", positive_label=1)
        frame = pd.DataFrame({
            "y_true": truth,
            "prediction": [0.1, 0.9, 0.2, 0.3],
        })
        with pytest.raises(ValueError, match=message):
            score_predictions(frame, spec)

    def test_ranking_score_retains_query_column(self):
        train, _, _ = make_competition_demo("panel")
        spec = _demo_spec("panel")
        folds = make_folds(train, spec)
        ledger = baseline_ledger(train, spec, folds=folds)
        assert score_predictions(ledger, spec) == 0.0
        report = score_ledger(ledger, spec)
        assert report.cv_mean == 0.0

    def test_ranking_score_rejects_queries_without_comparable_truth(self):
        spec = ProblemSpec(
            geometry="grouped", task="ranking", group_column="query",
            ranking_target_order="higher_is_better")
        frame = pd.DataFrame({
            "query": ["a", "a", "b", "b"],
            "y_true": [1.0, 1.0, 2.0, 2.0],
            "prediction": [0.1, 0.2, 0.3, 0.4],
        })
        with pytest.raises(ValueError, match="varying truth"):
            score_predictions(frame, spec)

    @pytest.mark.parametrize("profile", ["iid", "grouped", "time", "panel"])
    def test_repeated_null_controls_do_not_beat_fold_local_baseline(self, profile):
        train, _, _ = make_competition_demo(profile)
        spec = _demo_spec(profile)
        estimator = make_pipeline(SimpleImputer(strategy="median"), Ridge())
        # Use only numeric columns so one estimator covers all task branches;
        # logistic regression is required for the binary profile.
        if spec.task == "binary":
            estimator = make_pipeline(
                SimpleImputer(strategy="median"),
                LogisticRegression(max_iter=500, random_state=0),
            )
        report = null_control_report(
            estimator,
            train,
            ["x_num", "x_aux"],
            spec,
            n_repeats=3,
            tolerance=0.20,
        )
        assert len(report) == 3
        assert report.attrs["mean_improvement_over_baseline"] <= 0.20

    def test_time_null_recomputes_its_fold_local_baseline_each_repeat(self):
        train, _, _ = make_competition_demo("time")
        spec = _demo_spec("time")
        report = null_control_report(
            make_pipeline(SimpleImputer(strategy="median"), Ridge()),
            train,
            ["x_num", "x_aux"],
            spec,
            n_repeats=3,
            tolerance=10.0,
        )
        assert report["baseline_cv_mean"].nunique() > 1

    def test_ranking_safe_heldout_importance_runs_within_query(self):
        train, _, _ = make_competition_demo("panel")
        spec = _demo_spec("panel")
        result = heldout_permutation_importance(
            make_pipeline(SimpleImputer(strategy="median"), Ridge()),
            train,
            ["x_num", "x_aux"],
            spec,
            n_repeats=2,
        )
        assert set(result["feature"]) == {"x_num", "x_aux"}
        assert np.isfinite(result[["importance", "std", "base_score"]]).all(axis=None)

    def test_heldout_importance_rejects_an_overlapping_fold_plan(self):
        train, _, _ = make_competition_demo("iid")
        spec = _demo_spec("iid")
        folds = list(make_folds(train, spec))
        fit, valid = folds[0]
        folds[0] = (np.r_[fit, valid[0]], valid)
        with pytest.raises(ValueError, match="leaks .* rows"):
            heldout_permutation_importance(
                make_pipeline(SimpleImputer(strategy="median"), Ridge()),
                train, ["x_num", "x_aux"], spec, folds=folds,
                n_repeats=1)

    def test_heldout_importance_rejects_future_training_rows(self):
        train, _, _ = make_competition_demo("time")
        spec = _demo_spec("time")
        folds = list(make_folds(train, spec))
        fit, valid = folds[0]
        future_row = int(folds[-1][1][-1])
        folds[0] = (np.r_[fit, future_row], valid)
        with pytest.raises(ValueError, match="not chronological"):
            heldout_permutation_importance(
                make_pipeline(SimpleImputer(strategy="median"), Ridge()),
                train, ["x_num", "x_aux"], spec, folds=folds,
                n_repeats=1)

    def test_time_trend_null_breaks_absolute_structure_signal(self):
        n_rows = 240
        trend = np.linspace(-2.0, 2.0, n_rows)
        data = pd.DataFrame({
            "row_id": [f"r{index:04d}" for index in range(n_rows)],
            "time": np.arange(n_rows, dtype=np.int64),
            "trend": trend,
            "target": trend,
        })
        spec = ProblemSpec(
            geometry="time", task="regression", time_column="time",
            test_time_relation="future", id_columns=("row_id",), n_splits=4)
        folds = make_folds(data, spec)
        estimator = make_pipeline(StandardScaler(), Ridge(alpha=1e-6))

        # The former one-row circular shift leaves a linear trend almost
        # perfectly learnable and therefore easily beats the constant null.
        old_null = data.assign(target=np.roll(data["target"].to_numpy(), 1))
        old_model = score_ledger(
            oof_ledger(estimator, old_null, ["trend"], spec, folds=folds), spec)
        old_baseline = score_ledger(
            baseline_ledger(old_null, spec, folds=folds), spec)
        assert old_baseline.cv_mean - old_model.cv_mean > 0.5

        report = null_control_report(
            estimator, data, ["trend"], spec, folds=folds, n_repeats=4,
            random_state=27, tolerance=0.20)
        assert report.attrs["strategy"] == "exact_structure_breaking"
        assert report.attrs["mean_improvement_over_baseline"] <= 0.20

    def test_unequal_block_null_never_draws_from_its_own_structure(self):
        groups = np.repeat(["a", "b", "c", "d"], [5, 3, 2, 1])
        data = pd.DataFrame({
            "group": groups,
            "target": np.arange(len(groups), dtype=float),
        })
        spec = ProblemSpec(
            geometry="grouped", task="regression", group_column="group")
        permuted = competition_module._geometry_preserving_permutation(
            data, spec, np.random.default_rng(5))
        assert sorted(permuted.tolist()) == data["target"].tolist()
        for positions in data.groupby("group", sort=False).indices.values():
            positions = np.asarray(positions, dtype=int)
            assert set(permuted[positions]).isdisjoint(data["target"].iloc[positions])

    def test_dominant_block_null_fails_when_exact_assignment_is_impossible(self):
        groups = np.repeat(["dominant", "b", "c", "d"], [100, 1, 1, 1])
        data = pd.DataFrame({"group": groups, "target": np.arange(len(groups))})
        spec = ProblemSpec(
            geometry="grouped", task="regression", group_column="group")
        with pytest.raises(ValueError, match="exact structure-breaking permutation"):
            competition_module._geometry_preserving_permutation(
                data, spec, np.random.default_rng(5))


class TestDriftAndSubmission:
    def test_data_boundary_guard_checks_availability_and_missingness(self):
        train, test, sample = make_competition_demo("time")
        spec = _demo_spec("time")
        availability = {"x_num": True, "x_aux": True, "category": True}
        report = validate_competition_data(
            train,
            test,
            sample,
            ["x_num", "x_aux", "category"],
            spec,
            availability=availability,
        )
        assert report["available_at_prediction"].all()
        assert set(report["column"]) == set(availability)
        assert {"train_null_rate", "test_null_rate", "null_rate_delta"} <= set(report)

    def test_data_boundary_guard_rejects_roles_and_unavailable_features(self):
        train, test, sample = make_competition_demo("time")
        spec = _demo_spec("time")
        with pytest.raises(ValueError, match="raw features"):
            validate_competition_data(
                train,
                test,
                sample,
                ["time", "x_num"],
                spec,
                availability={"time": True, "x_num": True},
            )
        with pytest.raises(ValueError, match="not explicitly available"):
            validate_competition_data(
                train,
                test,
                sample,
                ["x_num", "x_aux"],
                spec,
                availability={"x_num": True, "x_aux": False},
            )

    def test_data_boundary_guard_rejects_test_labels_and_nonfinite_targets(self):
        train, test, sample = make_competition_demo("iid")
        spec = _demo_spec("iid")
        labelled_test = test.assign(target=0.0)
        with pytest.raises(ValueError, match="test contains target"):
            validate_competition_data(
                train, labelled_test, sample, ["x_num"], spec,
                availability={"x_num": True})
        broken_train = train.copy()
        broken_train.loc[0, "target"] = np.inf
        with pytest.raises(ValueError, match="infinity"):
            validate_competition_data(
                broken_train, test, sample, ["x_num"], spec,
                availability={"x_num": True})

        binary_train, binary_test, binary_sample = make_competition_demo(
            "iid", task="binary")
        binary_spec = ProblemSpec(
            geometry="iid", task="binary", id_columns=("row_id",))
        binary_train["target"] = binary_train["target"].astype(float)
        binary_train.loc[0, "target"] = -np.inf
        with pytest.raises(ValueError, match="binary target.*infinity"):
            validate_competition_data(
                binary_train, binary_test, binary_sample, ["x_num"],
                binary_spec, availability={"x_num": True})

    def test_target_proxy_scan_uses_development_labels_only(self):
        train, test, sample = make_competition_demo("iid")
        spec = _demo_spec("iid")
        train = train.assign(signal_copy=train[spec.target])
        test = test.assign(signal_copy=np.linspace(-1.0, 1.0, len(test)))

        # The train/test boundary is target-independent and therefore does not
        # inspect the future holdout labels.
        report = validate_competition_data(
            train, test, sample, ["signal_copy"], spec,
            availability={"signal_copy": True})
        assert report["column"].tolist() == ["signal_copy"]

        development_positions, _ = make_final_holdout(train, spec)
        development = train.iloc[development_positions]
        with pytest.raises(ValueError, match="development features"):
            validate_development_target_proxies(
                development, ["signal_copy"], spec)

    def test_adversarial_report_handles_mixed_missing_features(self):
        train, test, _ = make_competition_demo("iid", n_train=320, n_test=160)
        report = adversarial_validation_report(
            train,
            test,
            ["x_num", "x_aux", "category"],
            n_splits=4,
            random_state=9,
        )
        repeated = adversarial_validation_report(
            train,
            test,
            ["x_num", "x_aux", "category"],
            n_splits=4,
            random_state=9,
        )

        assert report == repeated
        assert report.auc > 0.60
        assert len(report.fold_auc) == 4
        assert report.to_frame().columns.tolist() == ["fold", "auc"]

    def test_submission_preserves_sample_schema_ids_and_order(self):
        _, test, sample = make_competition_demo("iid")
        spec = ProblemSpec(
            geometry="iid",
            task="regression",
            target="target",
            id_columns=("row_id",),
        )
        predictions = pd.Series(np.linspace(-1, 1, len(test)), index=test.index[::-1])
        result = validate_submission(test, predictions, sample, spec)

        assert result.columns.tolist() == sample.columns.tolist()
        assert result["row_id"].tolist() == test["row_id"].tolist()
        assert np.allclose(result["target"], predictions.to_numpy())
        assert (sample["target"] == 0).all(), "the sample must not be mutated"

    def test_submission_refuses_reordered_ids(self):
        _, test, sample = make_competition_demo("iid")
        spec = ProblemSpec(
            geometry="iid", task="regression", id_columns=("row_id",))
        reordered = sample.iloc[::-1].reset_index(drop=True)
        with pytest.raises(ValueError, match="different order"):
            validate_submission(test, np.zeros(len(test)), reordered, spec)

    def test_submission_refuses_unpopulated_extra_targets(self):
        _, test, sample = make_competition_demo("iid")
        spec = _demo_spec("iid")
        sample["another_target"] = 0.0
        with pytest.raises(ValueError, match="extra columns"):
            validate_submission(test, np.zeros(len(test)), sample, spec)

    def test_binary_submission_refuses_class_labels_outside_probability_range(self):
        _, test, sample = make_competition_demo("grouped")
        spec = ProblemSpec(
            geometry="grouped",
            task="binary",
            group_column="group",
            id_columns=("row_id",),
        )
        bad = np.zeros(len(test))
        bad[0] = 1.1
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            validate_submission(test, bad, sample, spec)

    def test_ranking_submission_is_unique_zero_based_within_query(self):
        _, test, sample = make_competition_demo("panel")
        spec = _demo_spec("panel")
        raw = np.zeros(len(test))  # ties exercise stable, positional ordering
        submission = validate_submission(test, raw, sample, spec)
        joined = test[["time"]].assign(rank=submission["target"].to_numpy())
        for _, group in joined.groupby("time", observed=True):
            assert sorted(group["rank"].tolist()) == list(range(len(group)))

    def test_ranking_order_matches_zero_best_scoring_and_tie_contract(self):
        test = pd.DataFrame({
            "row_id": ["a", "b", "c"],
            "query": ["q", "q", "q"],
        })
        sample = test[["row_id"]].assign(target=0)
        lower = ProblemSpec(
            geometry="grouped", task="ranking", group_column="query",
            id_columns=("row_id",), submission_transform="rank_descending_zero",
            ranking_target_order="lower_is_better")
        frame = pd.DataFrame({
            "query": test["query"],
            "y_true": [0.0, 1.0, 2.0],
            "prediction": [0.0, 1.0, 2.0],
        })
        assert score_predictions(frame, lower) == pytest.approx(1.0)
        assert validate_submission(
            test, frame["prediction"], sample, lower)["target"].tolist() == [0, 1, 2]
        reversed_frame = frame.assign(prediction=[2.0, 1.0, 0.0])
        assert score_predictions(reversed_frame, lower) == pytest.approx(-1.0)

        higher = dataclasses.replace(lower, ranking_target_order="higher_is_better")
        assert score_predictions(frame, higher) == pytest.approx(1.0)
        assert validate_submission(
            test, frame["prediction"], sample, higher)["target"].tolist() == [2, 1, 0]
        tied = validate_submission(test, [1.0, 1.0, 0.0], sample, higher)
        assert tied["target"].tolist() == [0, 1, 2]


class TestRunIdentity:
    def test_notebook_source_resolution_is_portable_and_explicit(
            self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert resolve_notebook_source("analysis.ipynb") is None
        notebook = tmp_path / "analysis.ipynb"
        notebook.write_text("{}", encoding="utf-8")
        assert resolve_notebook_source("analysis.ipynb") == notebook.resolve()
        with pytest.raises(FileNotFoundError, match="configured notebook source"):
            resolve_notebook_source(
                "analysis.ipynb", tmp_path / "elsewhere.ipynb")

    def test_archived_notebook_source_resolves_from_repository_root(
            self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        archived = tmp_path / "notebooks" / "legacy" / "analysis.ipynb"
        archived.parent.mkdir(parents=True)
        archived.write_text("{}", encoding="utf-8")
        assert resolve_notebook_source("analysis.ipynb") == archived.resolve()

        # Existing launch locations retain priority over the archive fallback.
        notebook = tmp_path / "notebooks" / "analysis.ipynb"
        notebook.write_text("{}", encoding="utf-8")
        assert resolve_notebook_source("analysis.ipynb") == notebook.resolve()
        assert resolve_notebook_source(
            "analysis.ipynb", archived) == archived.resolve()

    def test_notebook_code_fingerprint_ignores_outputs_but_tracks_source(self, tmp_path):
        path = tmp_path / "analysis.ipynb"
        notebook = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "cells": [{"cell_type": "code", "id": "stable", "source": ["x = 1"],
                       "metadata": {}, "execution_count": 1,
                       "outputs": [{"output_type": "stream", "name": "stdout",
                                    "text": ["old"]}]}],
            "metadata": {},
        }
        path.write_text(json.dumps(notebook))
        original = code_fingerprint(path)
        notebook["cells"][0]["execution_count"] = 99
        notebook["cells"][0]["outputs"][0]["text"] = ["new"]
        path.write_text(json.dumps(notebook))
        assert code_fingerprint(path) == original
        notebook["cells"][0]["metadata"]["collapsed"] = True
        path.write_text(json.dumps(notebook))
        assert code_fingerprint(path) == original
        notebook["cells"][0]["metadata"]["tags"] = ["parameters", "setup"]
        path.write_text(json.dumps(notebook))
        tagged = code_fingerprint(path)
        assert tagged != original
        notebook["cells"][0]["metadata"]["tags"].reverse()
        path.write_text(json.dumps(notebook))
        assert code_fingerprint(path) == tagged
        notebook["cells"][0]["metadata"] = {}
        notebook["cells"][0]["source"] = ["x = 2"]
        path.write_text(json.dumps(notebook))
        assert code_fingerprint(path) != original

    def test_fingerprint_is_stable_and_order_sensitive(self):
        train, _, _ = make_competition_demo("iid")
        original = data_fingerprint(train)
        assert original == data_fingerprint(train.copy())

        changed = train.copy()
        changed.loc[0, "x_num"] += 0.001
        assert data_fingerprint(changed) != original
        assert data_fingerprint(train.iloc[::-1]) != original
        assert data_fingerprint(train[train.columns[::-1]]) != original

    def test_fingerprint_supports_multiindex_schema(self):
        index = pd.MultiIndex.from_arrays(
            [["a", "a", "b"], pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01"])],
            names=["entity", "time"],
        )
        frame = pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=index)
        fingerprint = data_fingerprint(frame)
        assert fingerprint == data_fingerprint(frame.copy())
        assert fingerprint != data_fingerprint(frame.reorder_levels([1, 0]))

    def test_manifest_is_deterministic_json_and_tracks_data(self):
        train, test, _ = make_competition_demo("time")
        spec = _demo_spec("time")
        kwargs = dict(
            feature_columns=["x_num", "x_aux", "category"],
            git_commit="abc123",
            extra={"budget": 10},
        )
        first = build_run_manifest(spec, train, test, **kwargs)
        second = build_run_manifest(spec, train.copy(), test.copy(), **kwargs)

        assert first == second
        assert len(first["manifest_id"]) == 64
        assert first["data"]["train"]["synthetic"] is True
        assert first["spec"]["geometry"] == "time"
        assert set(first["environment"]) >= {
            "python", "platform", "joblib", "numpy", "pandas", "rvlab",
            "scipy", "sklearn",
        }
        assert first["environment"]["joblib"] == competition_module.joblib.__version__
        assert first["environment"]["scipy"] == competition_module.scipy.__version__
        assert first["environment"]["rvlab"] == competition_module.rvlab_version
        json.dumps(first, sort_keys=True, allow_nan=False)

        changed = train.copy()
        changed.loc[0, "x_num"] += 1
        assert build_run_manifest(spec, changed, test, **kwargs)["manifest_id"] != first["manifest_id"]
        with pytest.raises(ValueError, match="test contains target"):
            build_run_manifest(spec, train, test.assign(target=0.0), **kwargs)

    def test_materialized_folds_preserve_source_positions_and_roles(self):
        train, _, _ = make_competition_demo("time")
        spec = _demo_spec("time", n_splits=3)
        development_positions, _ = make_final_holdout(train, spec)
        development = train.iloc[development_positions].reset_index(drop=True)
        folds = make_folds(development, spec)
        table = materialize_folds(
            development, folds, spec, source_positions=development_positions)

        assert {"fold", "role", "development_position", "source_position",
                "row_id", "time"} <= set(table)
        assert set(table["role"]) == {"train", "validation"}
        assert np.array_equal(
            table["source_position"],
            development_positions[table["development_position"].to_numpy()],
        )

    def test_export_bundle_hashes_and_revalidates_every_artifact(self, tmp_path):
        train, test, sample = make_competition_demo("iid", n_train=120, n_test=40)
        spec = ProblemSpec(
            geometry="iid", task="regression", id_columns=("row_id",), n_splits=3)
        features = ["x_num", "x_aux"]
        development_positions, _ = make_final_holdout(train, spec)
        development = train.iloc[development_positions].reset_index(drop=True)
        folds = make_folds(development, spec)
        estimator = make_pipeline(SimpleImputer(strategy="median"), Ridge())
        ledger = oof_ledger(estimator, development, features, spec, folds=folds)
        fitted = estimator.fit(train[features], train[spec.target])
        submission = validate_submission(test, fitted.predict(test[features]), sample, spec)
        semantic = pd.DataFrame(
            [[1.2, 0.2], [1.0, 0.1]],
            index=pd.Index(["rmse", "mae"], name="statistic"),
            columns=pd.MultiIndex.from_tuples(
                [("score", "mean"), ("score", "std")],
                names=("measure", "summary")),
        )
        semantic.attrs.update({"scope": "development_only", "tolerance": 0.1})
        categorical = pd.DataFrame({
            "bucket": pd.Categorical(
                ["low", "high"], categories=["low", "mid", "high"],
                ordered=True),
        }, index=pd.CategoricalIndex(
            ["train", "valid"], categories=["train", "valid", "test"],
            ordered=False, name="split"))
        tables = {
            "oof_predictions.csv": ledger.assign(candidate="model"),
            "semantic_diagnostic.csv": semantic,
            "categorical_diagnostic.csv": categorical,
        }
        model_card = {"scope": "test", "score": score_ledger(ledger, spec).cv_mean}

        with pytest.raises(ValueError, match="test contains target"):
            export_run_bundle(
                tmp_path, "labelled", spec, train, test.assign(target=0.0),
                sample, features, fitted, submission, development=development,
                folds=folds, source_positions=development_positions)
        assert not list(tmp_path.iterdir())

        destination = export_run_bundle(
            tmp_path, "run", spec, train, test, sample, features, fitted, submission,
            development=development, folds=folds,
            source_positions=development_positions,
            tables=tables,
            model_card=model_card,
            code_paths=[Path(__file__)],
        )
        required = {"split_diagnostics.csv", "fold_assignments.csv",
                    "oof_predictions.csv", "semantic_diagnostic.csv",
                    "categorical_diagnostic.csv",
                    "submission.csv", "model_card.json", "model.joblib",
                    "table_metadata.json", "manifest.json"}
        assert required <= {path.name for path in destination.iterdir()}
        manifest = json.loads((destination / "manifest.json").read_text())
        assert manifest["extra"]["sample_submission_fingerprint"] == data_fingerprint(sample)
        assert Path(__file__).name in manifest["extra"]["code_identity"]
        payload = {key: value for key, value in manifest.items() if key != "manifest_id"}
        expected_id = hashlib.sha256(json.dumps(
            payload, sort_keys=True, allow_nan=False,
            separators=(",", ":")).encode()).hexdigest()
        assert manifest["manifest_id"] == expected_id
        assert destination.name.endswith(expected_id[:12])
        saved_semantic = pd.read_csv(destination / "semantic_diagnostic.csv")
        assert saved_semantic.columns.tolist() == [
            "statistic", "score | mean", "score | std"]
        assert saved_semantic["statistic"].tolist() == ["rmse", "mae"]
        table_metadata = json.loads(
            (destination / "table_metadata.json").read_text())
        semantic_metadata = table_metadata["semantic_diagnostic.csv"]
        assert semantic_metadata["attrs"] == {
            "scope": "development_only", "tolerance": 0.1}
        assert semantic_metadata["index"]["preserved"] is True
        assert semantic_metadata["index"]["exported_columns"] == ["statistic"]
        assert semantic_metadata["columns"]["class"].endswith("MultiIndex")
        assert semantic_metadata["columns"]["names"][0]["value"] == "measure"
        categorical_metadata = table_metadata["categorical_diagnostic.csv"]
        assert categorical_metadata["index"]["categoricals"][0]["categories"][-1]["value"] == "test"
        assert categorical_metadata["columns"]["categoricals"][0]["ordered"] is True
        assert [item["value"] for item in
                categorical_metadata["columns"]["categoricals"][0]["categories"]] == [
                    "low", "mid", "high"]
        for filename, expected in manifest["artifacts_sha256"].items():
            observed = hashlib.sha256((destination / filename).read_bytes()).hexdigest()
            assert observed == expected

        repeated = export_run_bundle(
            tmp_path, "run", spec, train, test, sample, features, fitted, submission,
            development=development, folds=folds,
            source_positions=development_positions, tables=tables,
            model_card=model_card, code_paths=[Path(__file__)])
        assert repeated == destination
        assert not list(tmp_path.glob(".run.*"))

        manifest_bytes = (destination / "manifest.json").read_bytes()
        (destination / "stale.csv").write_text("old\n", encoding="utf-8")
        with pytest.raises(FileExistsError, match="different bytes"):
            export_run_bundle(
                tmp_path, "run", spec, train, test, sample, features,
                fitted, submission, development=development, folds=folds,
                source_positions=development_positions, tables=tables,
                model_card=model_card, code_paths=[Path(__file__)])
        assert (destination / "manifest.json").read_bytes() == manifest_bytes
        assert (destination / "stale.csv").read_text(encoding="utf-8") == "old\n"
        assert not list(tmp_path.glob(".run.*"))

    def test_export_derives_positions_and_preserves_leading_zero_ids(self, tmp_path):
        train, test, sample = make_competition_demo(
            "iid", n_train=100, n_test=30)
        train["row_id"] = [f"{index:06d}" for index in range(len(train))]
        test["row_id"] = [
            f"{index:06d}" for index in range(len(train), len(train) + len(test))]
        sample["row_id"] = test["row_id"].copy()
        sample.index = pd.Index(
            np.arange(5_000, 5_000 + len(sample)), name="not_submission_data")
        spec = ProblemSpec(
            geometry="iid", task="regression", id_columns=("row_id",),
            n_splits=3)
        features = ["x_num", "x_aux"]
        development_positions, _ = make_final_holdout(train, spec)
        development = train.iloc[development_positions].reset_index(drop=True)
        folds = make_folds(development, spec)
        fitted = make_pipeline(
            SimpleImputer(strategy="median"), Ridge()).fit(
                train[features], train[spec.target])
        submission = validate_submission(
            test, fitted.predict(test[features]), sample, spec)

        destination = export_run_bundle(
            tmp_path, "leading_zero", spec, train, test, sample, features,
            fitted, submission, development=development, folds=folds)
        saved = pd.read_csv(
            destination / "submission.csv", converters={"row_id": str})
        assert saved.columns.tolist() == ["row_id", "target"]
        assert saved["row_id"].tolist() == test["row_id"].tolist()
        metadata = json.loads(
            (destination / "table_metadata.json").read_text())
        submission_index = metadata["submission.csv"]["index"]
        assert submission_index["preserved"] is False
        assert submission_index["nondefault_index_excluded"] is True
        assert submission_index["names"][0]["value"] == "not_submission_data"
        assignments = pd.read_csv(
            destination / "fold_assignments.csv", converters={"row_id": str})
        expected = train.iloc[assignments["source_position"]]["row_id"].to_numpy()
        assert assignments["row_id"].to_numpy().tolist() == expected.tolist()

    def test_export_roundtrips_datetime_identifier_tokens(self, tmp_path):
        train, test, _ = make_competition_demo(
            "iid", n_train=100, n_test=30)
        train["event_time"] = pd.date_range(
            "2024-01-01", periods=len(train), freq="h")
        test["event_time"] = pd.date_range(
            "2025-01-01", periods=len(test), freq="h")
        sample = test[["event_time"]].copy()
        sample["target"] = 0.0
        spec = ProblemSpec(
            geometry="iid", task="regression", id_columns=("event_time",),
            n_splits=3)
        features = ["x_num", "x_aux"]
        development_positions, _ = make_final_holdout(train, spec)
        development = train.iloc[development_positions].reset_index(drop=True)
        folds = make_folds(development, spec)
        fitted = make_pipeline(
            SimpleImputer(strategy="median"), Ridge()).fit(
                train[features], train[spec.target])
        submission = validate_submission(
            test, fitted.predict(test[features]), sample, spec)

        destination = export_run_bundle(
            tmp_path, "datetime_id", spec, train, test, sample, features,
            fitted, submission, development=development, folds=folds)
        saved = pd.read_csv(
            destination / "submission.csv", dtype=str, keep_default_na=False)
        assert saved["event_time"].tolist() == test["event_time"].astype(str).tolist()

    @pytest.mark.parametrize(
        "first_two, message",
        [((1, "1"), "collide after CSV serialization"),
         (("", "1"), "empty token after CSV serialization")],
    )
    def test_export_rejects_lossy_csv_id_tokens(
            self, tmp_path, first_two, message):
        train, test, sample = make_competition_demo(
            "iid", n_train=100, n_test=30)
        ambiguous = pd.Series(
            [*first_two, *[f"id_{index}" for index in range(2, len(test))]],
            dtype=object)
        test["row_id"] = ambiguous
        sample["row_id"] = ambiguous.copy()
        spec = ProblemSpec(
            geometry="iid", task="regression", id_columns=("row_id",),
            n_splits=3)
        features = ["x_num", "x_aux"]
        development_positions, _ = make_final_holdout(train, spec)
        development = train.iloc[development_positions].reset_index(drop=True)
        folds = make_folds(development, spec)
        fitted = make_pipeline(
            SimpleImputer(strategy="median"), Ridge()).fit(
                train[features], train[spec.target])
        submission = validate_submission(
            test, fitted.predict(test[features]), sample, spec)

        with pytest.raises(ValueError, match=message):
            export_run_bundle(
                tmp_path, "ambiguous_id", spec, train, test, sample, features,
                fitted, submission, development=development, folds=folds,
                source_positions=development_positions)
        assert not list(tmp_path.iterdir())

    def test_export_rejects_notebook_local_model_in_fresh_interpreter(
            self, tmp_path, monkeypatch):
        train, test, sample = make_competition_demo(
            "iid", n_train=100, n_test=30)
        spec = ProblemSpec(
            geometry="iid", task="regression", id_columns=("row_id",),
            n_splits=3)
        features = ["x_num", "x_aux"]
        development_positions, _ = make_final_holdout(train, spec)
        development = train.iloc[development_positions].reset_index(drop=True)
        folds = make_folds(development, spec)
        fitted = make_pipeline(
            SimpleImputer(strategy="median"), Ridge()).fit(
                train[features], train[spec.target])

        def predict(instance, values):
            return instance.delegate.predict(values)

        local_type = type(
            "NotebookLocalEstimator", (),
            {"__module__": "__main__", "predict": predict})
        monkeypatch.setattr(
            sys.modules["__main__"], "NotebookLocalEstimator", local_type,
            raising=False)
        local_model = local_type()
        local_model.delegate = fitted
        submission = validate_submission(
            test, local_model.predict(test[features]), sample, spec)

        with pytest.raises(TypeError, match="fresh interpreter.*importable module"):
            export_run_bundle(
                tmp_path, "local_model", spec, train, test, sample, features,
                local_model, submission, development=development, folds=folds,
                source_positions=development_positions)
        assert not list(tmp_path.iterdir())

    def test_export_rejects_flattened_evidence_column_collisions(self, tmp_path):
        train, test, sample = make_competition_demo(
            "iid", n_train=100, n_test=30)
        spec = ProblemSpec(
            geometry="iid", task="regression", id_columns=("row_id",),
            n_splits=3)
        features = ["x_num", "x_aux"]
        development_positions, _ = make_final_holdout(train, spec)
        development = train.iloc[development_positions].reset_index(drop=True)
        folds = make_folds(development, spec)
        fitted = make_pipeline(
            SimpleImputer(strategy="median"), Ridge()).fit(
                train[features], train[spec.target])
        submission = validate_submission(
            test, fitted.predict(test[features]), sample, spec)
        ambiguous = pd.DataFrame(
            [[1.0, 2.0]],
            columns=pd.MultiIndex.from_tuples([("a", "b"), ("a", "b")]))

        with pytest.raises(ValueError, match="collide after.*CSV flattening"):
            export_run_bundle(
                tmp_path, "ambiguous_columns", spec, train, test, sample,
                features, fitted, submission, development=development,
                folds=folds, source_positions=development_positions,
                tables={"ambiguous.csv": ambiguous})
        assert not list(tmp_path.iterdir())

    def test_materialize_folds_requires_an_explicit_source_mapping(self):
        train, _, _ = make_competition_demo("iid")
        spec = _demo_spec("iid")
        with pytest.raises(ValueError, match="source_positions is required"):
            materialize_folds(train, make_folds(train, spec), spec)

    def test_condition_matrix_runs_every_supported_pair_end_to_end(self):
        matrix = smoke_test_condition_matrix(random_state=7)
        expected = {
            "iid/regression", "iid/binary",
            "grouped/regression", "grouped/binary", "grouped/ranking",
            "time/regression", "time/binary",
            "panel/regression", "panel/binary", "panel/ranking",
        }
        assert set(matrix["condition"]) == expected
        assert (matrix["folds"] == 3).all()
        assert (matrix["max_row_overlap"] == 0).all()
        assert (matrix["submission_rows"] == 60).all()
        assert np.isfinite(matrix[["cv_mean", "cv_std", "worst_fold",
                                   "pooled_oof", "final_holdout_score"]]).all(axis=None)
