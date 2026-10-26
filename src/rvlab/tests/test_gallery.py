"""Every gallery chart must render without error and return (fig, axes).

Chart code fails in ways unit tests rarely catch — an empty group, a constant
column, a single observation. These tests exercise each function on plausible
awkward input, because a notebook that dies on cell 40 of 54 is worse than one
that never had the chart.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from rvlab.plotting import gallery as G


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 600
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "x": rng.normal(size=n),
        "y": rng.normal(size=n) * 0.5 + np.linspace(0, 2, n),
        "heavy": rng.lognormal(0, 1.1, n),
        "group": rng.choice(list("abcd"), n),
        "ret": rng.normal(0.0004, 0.01, n),
    }, index=idx)


@pytest.fixture(scope="module")
def classification():
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X, y = make_classification(1200, 8, weights=[0.85], random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)
    model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    return y_te, model.predict_proba(X_te)[:, 1], model.predict(X_te)


def _ok(result):
    fig, _ = result
    assert fig is not None and len(fig.axes) >= 1
    matplotlib.pyplot.close(fig)


class TestDistributions:
    def test_hist_kde(self, data):
        _ok(G.hist_kde(data["x"]))

    def test_ecdf_with_and_without_groups(self, data):
        _ok(G.ecdf_plot(data, "x"))
        _ok(G.ecdf_plot(data, "x", group="group"))

    def test_qq(self, data):
        _ok(G.qq_plot(data["heavy"]))

    def test_violin_strip(self, data):
        _ok(G.violin_strip(data, "x", "group"))

    def test_ridgeline(self, data):
        _ok(G.ridgeline(data, "x", "group"))

    def test_ridgeline_survives_a_degenerate_group(self, data):
        """A group with no variance would break a KDE; it must be skipped."""
        flat = pd.concat([data, pd.DataFrame({"x": [1.0] * 3, "group": ["z"] * 3})])
        _ok(G.ridgeline(flat, "x", "group"))


class TestRelationships:
    def test_scatter_fit(self, data):
        _ok(G.scatter_fit(data, "x", "y"))

    def test_hexbin(self, data):
        _ok(G.hexbin_plot(data, "x", "y"))

    def test_joint(self, data):
        _ok(G.joint_plot(data, "x", "y"))

    def test_pair_grid(self, data):
        _ok(G.pair_grid(data, ["x", "y", "heavy"]))

    def test_correlation_contrast_returns_three_panels(self, data):
        fig, axes = G.correlation_contrast(data[["x", "y", "heavy", "ret"]])
        assert len(np.atleast_1d(axes)) == 3
        matplotlib.pyplot.close(fig)

    def test_cluster_map(self, data):
        _ok(G.cluster_map(data[["x", "y", "heavy", "ret"]]))


class TestTimeSeries:
    def test_rolling_band(self, data):
        _ok(G.rolling_band(data["ret"], window=20))

    def test_calendar_heatmap(self, data):
        _ok(G.calendar_heatmap(data["ret"]))

    def test_acf_pacf(self, data):
        _ok(G.acf_pacf(data["ret"], lags=20))

    def test_drawdown(self, data):
        _ok(G.drawdown_underwater(data["ret"]))

    def test_event_study(self, data):
        events = data["ret"].nlargest(8).index
        frame = data.reset_index().rename(columns={"index": "ts"})
        _ok(G.event_study(frame, "y", "ts", events, window=6))

    def test_event_study_with_no_usable_events(self, data):
        """Events at the very edge have no complete window — must not raise."""
        frame = data.reset_index().rename(columns={"index": "ts"})
        _ok(G.event_study(frame, "y", "ts", [frame["ts"].iloc[0]], window=50))

    def test_multi_series(self, data):
        levels = data[["x", "y", "heavy"]].abs() + 1
        _ok(G.multi_series(levels, ["x", "y", "heavy"]))


class TestCategorical:
    def test_facet_grid(self, data):
        _ok(G.facet_grid(data, "x", "y", "group"))

    def test_annotated_bar(self, data):
        values = pd.Series({"a": 1.2, "b": 0.9, "c": 1.05})
        _ok(G.annotated_bar(values, pd.Series({"a": 0.1, "b": 0.08, "c": 0.09})))

    def test_decile_lift(self, data):
        _ok(G.decile_lift(data["y"], data["x"]))


class TestDiagnostics:
    def test_residual_panel_returns_four(self, data):
        fig, axes = G.residual_panel(data["y"], data["y"] + data["x"] * 0.1)
        assert np.atleast_1d(axes).size == 4
        matplotlib.pyplot.close(fig)

    def test_learning_curve(self):
        from sklearn.datasets import make_regression
        from sklearn.linear_model import Ridge
        X, y = make_regression(300, 5, noise=10, random_state=0)
        _ok(G.learning_curve_plot(Ridge(), X, y, cv=3))

    def test_validation_curve(self):
        from sklearn.datasets import make_regression
        from sklearn.linear_model import Ridge
        X, y = make_regression(300, 5, noise=10, random_state=0)
        _ok(G.validation_curve_plot(Ridge(), X, y, "alpha", [0.1, 1.0, 10.0], cv=3))

    def test_calibration(self, classification):
        y_true, proba, _ = classification
        _ok(G.calibration_plot(y_true, proba))

    def test_roc_pr(self, classification):
        y_true, proba, _ = classification
        _ok(G.roc_pr_plot(y_true, proba))

    def test_confusion_matrix(self, classification):
        y_true, _, y_pred = classification
        _ok(G.confusion_matrix_plot(y_true, y_pred))

    def test_pdp_grid(self):
        from sklearn.datasets import make_regression
        from sklearn.linear_model import Ridge
        X, y = make_regression(300, 4, noise=10, random_state=0)
        frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
        _ok(G.pdp_grid(Ridge().fit(frame, y), frame, ["f0", "f1"]))


class TestCraft:
    def test_palette_preview(self):
        _ok(G.palette_preview())

    def test_colorblind_check_alters_the_colours(self):
        """The simulation must actually change something, or it checks nothing."""
        from matplotlib.colors import to_rgb
        from rvlab.plotting.style import PALETTE
        matrix = np.array([[0.625, 0.375, 0.0], [0.700, 0.300, 0.0], [0.0, 0.300, 0.7]])
        shifted = [tuple(np.clip(matrix @ np.array(to_rgb(c)), 0, 1)) for c in PALETTE]
        assert any(s != to_rgb(c) for s, c in zip(shifted, PALETTE))
        _ok(G.colorblind_check())

    def test_save_publication_writes_the_file(self, data, tmp_path, monkeypatch):
        from rvlab import config
        monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
        fig, _ = G.hist_kde(data["x"])
        paths = G.save_publication(fig, "unit_test", dpi=72)
        assert paths[0].exists() and paths[0].stat().st_size > 0
        matplotlib.pyplot.close(fig)
