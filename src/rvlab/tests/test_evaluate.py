"""Metrics, statistical tests, and sweep plumbing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rvlab.evaluate.metrics import (
    bias, directional_accuracy, mae, r2, regression_report, rmse, skill_vs_baseline,
)
from rvlab.evaluate.sweeps import grid_points, summarise_sweep, sweep_matrix
from rvlab.evaluate.tests import (
    benjamini_hochberg, block_bootstrap_ci, bonferroni, diebold_mariano,
    paired_permutation_test,
)


class TestMetrics:
    def test_perfect_forecast_scores_zero_error(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0 and mae(y, y) == 0.0 and bias(y, y) == 0.0
        assert r2(y, y) == pytest.approx(1.0)

    def test_rmse_matches_the_definition(self):
        y, p = np.array([1.0, 2.0]), np.array([1.0, 4.0])
        assert rmse(y, p) == pytest.approx(np.sqrt(4.0 / 2))

    def test_bias_is_signed(self):
        y, p = np.array([1.0, 1.0]), np.array([2.0, 2.0])
        assert bias(y, p) == pytest.approx(1.0)
        assert bias(p, y) == pytest.approx(-1.0)

    def test_non_finite_pairs_are_dropped_not_propagated(self):
        y = np.array([1.0, 2.0, np.nan]); p = np.array([1.0, 2.0, 5.0])
        assert rmse(y, p) == 0.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            rmse(np.zeros(3), np.zeros(4))

    def test_skill_is_zero_against_itself_and_positive_when_better(self):
        y = np.arange(50.0)
        base = y + 1.0
        good = y + 0.1
        assert skill_vs_baseline(y, base, base) == pytest.approx(0.0)
        assert skill_vs_baseline(y, good, base) > 0.8
        assert skill_vs_baseline(y, y + 5.0, base) < 0

    def test_directional_accuracy_ignores_zero_move_forecasts(self):
        prev = np.array([1.0, 2.0]); act = np.array([1.5, 1.5])
        assert np.isnan(directional_accuracy(act, prev, prev))   # carry: no call
        assert directional_accuracy(act, act, prev) == 1.0

    def test_regression_report_ranks_by_skill(self):
        y = np.arange(100.0)
        rep = regression_report(y, {"carry": y + 1.0, "good": y + 0.1, "bad": y + 3.0})
        assert list(rep.index) == ["good", "carry", "bad"]
        assert rep.loc["carry", "skill_vs_carry"] == pytest.approx(0.0)

    def test_regression_report_rejects_unknown_baseline(self):
        with pytest.raises(KeyError):
            regression_report(np.arange(5.0), {"a": np.arange(5.0)}, baseline="nope")


class TestDieboldMariano:
    def test_detects_a_clearly_better_forecast(self):
        rng = np.random.default_rng(0)
        y = np.cumsum(rng.normal(0, 1, 400))
        res = diebold_mariano(y, y + rng.normal(0, 0.2, 400), y + rng.normal(0, 2.0, 400))
        assert res.statistic < 0 and res.p_value < 0.01
        assert res.favours == "model A"

    def test_is_symmetric_under_swapping(self):
        rng = np.random.default_rng(1)
        y = np.cumsum(rng.normal(0, 1, 300))
        a, b = y + rng.normal(0, 0.3, 300), y + rng.normal(0, 1.0, 300)
        assert diebold_mariano(y, a, b).statistic == pytest.approx(
            -diebold_mariano(y, b, a).statistic, rel=1e-9)

    def test_identical_forecasts_are_not_significant(self):
        rng = np.random.default_rng(2)
        y = np.cumsum(rng.normal(0, 1, 300)); f = y + rng.normal(0, 1, 300)
        assert diebold_mariano(y, f, f.copy()).favours.startswith("neither")

    def test_too_few_observations_returns_nan(self):
        assert np.isnan(diebold_mariano(np.zeros(4), np.zeros(4), np.ones(4)).statistic)


class TestBootstrapAndMultiplicity:
    def test_ci_brackets_a_known_mean(self):
        x = np.random.default_rng(0).normal(5.0, 1.0, 800)
        ci = block_bootstrap_ci(x, n_boot=400)
        assert ci["lo"] < 5.0 < ci["hi"]
        assert ci["point"] == pytest.approx(x.mean())

    def test_block_size_defaults_to_cube_root(self):
        assert block_bootstrap_ci(np.random.default_rng(0).normal(size=1000),
                                  n_boot=100)["block_size"] == 10

    def test_excludes_zero_flag_is_correct(self):
        rng = np.random.default_rng(4)
        assert block_bootstrap_ci(rng.normal(10.0, 0.5, 500), n_boot=300)["excludes_zero"]
        assert not block_bootstrap_ci(rng.normal(0.0, 1.0, 500), n_boot=300)["excludes_zero"]

    def test_bh_is_less_conservative_than_bonferroni(self):
        p = np.array([0.001, 0.008, 0.02, 0.04, 0.2, 0.6, 0.9])
        assert benjamini_hochberg(p)["reject"].sum() >= bonferroni(p)["reject"].sum()

    def test_bh_rejects_the_expected_set(self):
        p = np.array([0.001, 0.008, 0.02, 0.04, 0.2, 0.6, 0.9])
        out = benjamini_hochberg(p, alpha=0.05).sort_values("p_value")
        assert list(out["reject"]) == [True, True, True, False, False, False, False]

    def test_bh_rejects_nothing_under_the_null(self):
        p = np.random.default_rng(0).uniform(0, 1, 200)
        assert benjamini_hochberg(p)["reject"].sum() <= 2

    def test_permutation_test_agrees_with_dm_on_a_clear_case(self):
        rng = np.random.default_rng(5)
        y = np.cumsum(rng.normal(0, 1, 300))
        res = paired_permutation_test(y, y + rng.normal(0, 0.2, 300),
                                      y + rng.normal(0, 2.0, 300), n_perm=500)
        assert res["mean_loss_diff"] < 0 and res["p_value"] < 0.05


class TestSweeps:
    def test_grid_points_is_the_cartesian_product(self):
        pts = grid_points({"h": [0.05, 0.1], "step": [1, 5]})
        assert len(pts) == 4
        assert {"h": 0.05, "step": 5} in pts

    def test_summarise_reports_pass_rate(self):
        df = pd.DataFrame({"h": [0.05] * 4, "skill": [0.1, 0.2, -0.1, -0.3]})
        s = summarise_sweep(df, "skill")
        assert s.loc[0, "n_cells"] == 4 and s.loc[0, "n_pass"] == 2
        assert s.loc[0, "pass_rate"] == pytest.approx(0.5)

    def test_sweep_matrix_pivots(self):
        df = pd.DataFrame({"h": [0.05, 0.05, 0.1, 0.1], "step": [1, 5, 1, 5],
                           "skill": [0.1, 0.2, 0.3, 0.4]})
        m = sweep_matrix(df, "h", "step", "skill")
        assert m.shape == (2, 2) and m.loc[0.1, 5] == pytest.approx(0.4)
