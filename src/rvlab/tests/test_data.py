"""Loaders, contracts, and the synthetic fallback that keeps notebooks runnable."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rvlab.data import (
    CHAIN_PANEL, SMILE_PANEL, SchemaSpec, available, describe_provenance,
    load_chain_panel, load_smile_panel, provenance,
)

SMILE_COLUMNS = {"ts", "expiry", "T", "atm_iv", "atm_total_var", "rr25", "bf25",
                 "alpha", "gamma"}


class TestLoaders:
    def test_synthetic_panel_has_the_real_schema(self, panel):
        """The fallback must be indistinguishable in shape, or notebooks branch."""
        assert SMILE_COLUMNS <= set(panel.columns)
        assert len(panel) > 100

    def test_synthetic_panel_passes_its_own_contract(self, panel):
        assert SMILE_PANEL.validate(panel).ok

    def test_chain_panel_passes_its_contract(self):
        assert CHAIN_PANEL.validate(load_chain_panel()).ok

    def test_provenance_records_the_fallback(self, panel):
        p = provenance(panel)
        assert p["synthetic"] is True
        assert p["fallback_reason"] == "RVLAB_FORCE_SYNTHETIC"
        assert "SYNTHETIC" in describe_provenance(panel)

    def test_dte_filters_are_applied(self):
        df = load_smile_panel(min_dte=10, max_dte=30)
        assert df["T"].min() * 365 >= 10 - 1e-9
        assert df["T"].max() * 365 <= 30 + 1e-9

    def test_timestamps_are_utc_and_sorted(self, panel):
        assert str(panel["ts"].dt.tz) == "UTC"
        assert panel["ts"].is_monotonic_increasing

    def test_unknown_dataset_raises(self):
        with pytest.raises(KeyError, match="unknown dataset"):
            load_smile_panel("does_not_exist")

    def test_available_lists_every_catalogued_dataset(self):
        cat = available()
        assert {"smile_90d", "smile_127d", "chain_panel"} <= set(cat["dataset_id"])
        assert cat["present"].dtype == bool


class TestSchemaSpec:
    @staticmethod
    def _spec():
        return SchemaSpec(name="t", columns={"a": "float", "b": "int"},
                          non_null=("a",), ranges={"a": (0.0, 1.0)},
                          unique=(("b",),), sorted_by="b")

    def test_clean_frame_passes(self):
        df = pd.DataFrame({"a": [0.1, 0.5, 0.9], "b": [1, 2, 3]})
        assert self._spec().validate(df).ok

    def test_missing_column_is_an_error(self):
        rep = self._spec().validate(pd.DataFrame({"a": [0.5]}))
        assert not rep.ok and "missing columns" in rep.errors[0]

    def test_out_of_range_is_an_error(self):
        df = pd.DataFrame({"a": [0.5, 7.0], "b": [1, 2]})
        rep = self._spec().validate(df)
        assert not rep.ok and any("outside" in e for e in rep.errors)

    def test_nulls_in_non_null_column_are_an_error(self):
        df = pd.DataFrame({"a": [0.5, np.nan], "b": [1, 2]})
        assert any("null" in e for e in self._spec().validate(df).errors)

    def test_duplicates_are_an_error(self):
        df = pd.DataFrame({"a": [0.5, 0.6], "b": [1, 1]})
        assert any("duplicate" in e for e in self._spec().validate(df).errors)

    def test_unsorted_is_a_warning_not_an_error(self):
        """Order is recoverable; the frame is still usable. Warn, do not fail."""
        df = pd.DataFrame({"a": [0.5, 0.6], "b": [2, 1]})
        rep = self._spec().validate(df)
        assert rep.ok and any("sorted" in w for w in rep.warnings)

    def test_raise_if_failed_lists_every_violation(self):
        df = pd.DataFrame({"a": [np.nan, 7.0], "b": [1, 1]})
        with pytest.raises(ValueError) as exc:
            self._spec().validate(df).raise_if_failed()
        assert str(exc.value).count("  - ") >= 3


class TestSynthetic:
    def test_fbm_paths_have_the_right_shape(self):
        from rvlab.data.synthetic import fbm_variance_paths
        t, vol = fbm_variance_paths(n_paths=8, n_steps=64, hurst=0.1)
        assert t.shape == (65,) and vol.shape == (8, 65)
        assert np.isfinite(vol).all() and (vol > 0).all()

    @pytest.mark.parametrize("hurst", [0.05, 0.1, 0.3, 0.45])
    def test_fbm_variance_equals_t_to_the_2h(self, hurst):
        """Var(W_H(t)) = t^(2H) exactly, for every H.

        This is the property a midpoint or left-endpoint discretisation of the
        singular kernel silently violates — and when it does, the simulated
        paths have H ~ 0.5 whatever value was requested, which makes every
        downstream validation meaningless. Worth asserting directly.
        """
        from rvlab.data.synthetic import fractional_brownian_motion
        t, W = fractional_brownian_motion(n_paths=4000, n_steps=192,
                                          horizon=1.0, hurst=hurst, seed=1)
        # ~2.3% Monte Carlo standard error on a variance from 4000 draws.
        assert W[:, -1].var() == pytest.approx(1.0, rel=0.08)
        mid = len(t) // 2
        assert W[:, mid].var() == pytest.approx(t[mid] ** (2 * hurst), rel=0.10)

    def test_hurst_estimate_tracks_the_simulated_hurst(self):
        """The estimator must be monotone in the H that was simulated.

        Absolute accuracy is not asserted: the structure-function estimator is
        genuinely biased downward on Riemann-Liouville fBM, whose increments are
        not stationary. Monotonicity is the property that makes it usable, and
        the bias is measured rather than assumed in notebook 04.
        """
        from rvlab.data.synthetic import fbm_variance_paths
        from rvlab.features import hurst_structure_function

        estimates = []
        for h in (0.05, 0.15, 0.30, 0.45):
            _, vol = fbm_variance_paths(n_paths=1, n_steps=2048, horizon=8.0,
                                        hurst=h, seed=42)
            estimates.append(hurst_structure_function(pd.Series(vol[0]), max_lag=64).hurst)

        assert estimates == sorted(estimates), f"not monotone in H: {estimates}"
        assert estimates[0] < 0.15 and estimates[-1] > 0.30

    def test_rougher_paths_are_more_jagged(self):
        """H is not a decoration: lower H must produce visibly rougher paths."""
        from rvlab.data.synthetic import fbm_variance_paths
        _, rough_v = fbm_variance_paths(n_paths=24, n_steps=256, hurst=0.05, seed=3)
        _, smooth_v = fbm_variance_paths(n_paths=24, n_steps=256, hurst=0.45, seed=3)
        jag = lambda v: np.mean(np.abs(np.diff(np.log(v), axis=1)))
        assert jag(rough_v) > jag(smooth_v)
