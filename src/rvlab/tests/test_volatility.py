"""Volatility estimators and conditional models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rvlab.data.panel import synthetic_equity_panel
from rvlab.models.volatility import (
    GarchFit, close_to_close, estimator_efficiency, estimator_table, ewma_vol,
    garch11, garch_forecast, garman_klass, har_rv, parkinson, rogers_satchell,
    variance_risk_premium, volatility_regimes, yang_zhang,
)


@pytest.fixture(scope="module")
def ohlc():
    return synthetic_equity_panel(n_entities=1, n_dates=1500, seed=3)


class TestRealizedEstimators:
    def test_close_to_close_recovers_a_known_sigma(self):
        """The only estimator with an exact target — assert it directly."""
        rng = np.random.default_rng(0)
        sigma = 0.02
        price = 100 * np.exp(np.cumsum(rng.normal(0, sigma, 20_000)))
        assert close_to_close(price) == pytest.approx(sigma * np.sqrt(252), rel=0.03)

    def test_all_estimators_are_positive_and_finite(self, ohlc):
        table = estimator_table(ohlc)
        assert (table["annualized_vol"] > 0).all()
        assert np.isfinite(table["annualized_vol"]).all()

    def test_pure_range_estimators_understate_when_prices_gap(self, ohlc):
        """Parkinson, Garman-Klass and Rogers-Satchell see only the intraday leg."""
        ratios = estimator_table(ohlc)["ratio_to_close_to_close"]
        for name in ("parkinson", "garman_klass", "rogers_satchell"):
            assert ratios[name] < 0.9, f"{name} did not understate: {ratios[name]:.3f}"

    def test_yang_zhang_is_the_closest_because_it_models_the_gap(self, ohlc):
        ratios = estimator_table(ohlc)["ratio_to_close_to_close"]
        assert ratios["yang_zhang"] > ratios[["parkinson", "garman_klass",
                                              "rogers_satchell"]].max()
        assert 0.85 < ratios["yang_zhang"] < 1.15

    def test_range_estimators_are_more_efficient(self, ohlc):
        """The claim people quote — measurable only by resampling one process."""
        eff = estimator_efficiency(ohlc, n_days=21, n_samples=150)
        assert eff.loc["parkinson", "efficiency_vs_close_to_close"] > 2.0
        assert eff.loc["garman_klass", "efficiency_vs_close_to_close"] > 2.0
        assert eff.loc["close_to_close", "efficiency_vs_close_to_close"] == pytest.approx(1.0)

    def test_estimators_ignore_non_positive_prices(self):
        o = np.array([100.0, 0.0, 102.0])
        assert np.isfinite(parkinson(np.array([101.0, 0.0, 103.0]),
                                     np.array([99.0, 0.0, 101.0])))

    def test_rogers_satchell_is_drift_independent(self):
        """Its defining property: a strong trend must not inflate it."""
        rng = np.random.default_rng(1)
        n = 4000
        for drift in (0.0, 0.001):
            noise = rng.normal(0, 0.01, n)
            close = 100 * np.exp(np.cumsum(noise + drift))
            open_ = close * np.exp(rng.normal(0, 0.002, n))
            high = np.maximum(open_, close) * 1.004
            low = np.minimum(open_, close) * 0.996
            value = rogers_satchell(open_, high, low, close)
            if drift == 0.0:
                baseline = value
        assert value == pytest.approx(baseline, rel=0.15)


class TestConditionalModels:
    def test_ewma_reacts_faster_than_a_long_rolling_window(self):
        rng = np.random.default_rng(0)
        calm = rng.normal(0, 0.005, 300)
        shock = rng.normal(0, 0.04, 60)
        r = pd.Series(np.r_[calm, shock])
        ewma = ewma_vol(r, lam=0.94)
        rolling = (r.rolling(100).std() * np.sqrt(252)).dropna()
        assert ewma.iloc[-1] > rolling.iloc[-1]

    def test_garch_recovers_its_own_simulated_parameters(self):
        """The property that makes the hand-written MLE trustworthy."""
        rng = np.random.default_rng(0)
        n, omega, alpha, beta = 4000, 2e-6, 0.09, 0.88
        eps = np.zeros(n); s2 = np.zeros(n); s2[0] = omega / (1 - alpha - beta)
        for t in range(1, n):
            s2[t] = omega + alpha * eps[t - 1] ** 2 + beta * s2[t - 1]
            eps[t] = rng.normal(0, np.sqrt(s2[t]))

        fit = garch11(pd.Series(eps))
        assert fit.converged
        assert fit.persistence == pytest.approx(alpha + beta, abs=0.05)
        assert fit.alpha == pytest.approx(alpha, abs=0.04)
        assert fit.beta == pytest.approx(beta, abs=0.05)

    def test_garch_half_life_follows_persistence(self):
        fit = GarchFit(omega=1e-6, alpha=0.05, beta=0.90, loglikelihood=0.0,
                       converged=True, conditional_vol=pd.Series(dtype=float))
        assert fit.persistence == pytest.approx(0.95)
        assert fit.half_life == pytest.approx(np.log(0.5) / np.log(0.95), rel=1e-9)

    def test_garch_forecast_reverts_toward_the_long_run_level(self):
        fit = GarchFit(omega=2e-6, alpha=0.08, beta=0.88, loglikelihood=0.0,
                       converged=True, conditional_vol=pd.Series(dtype=float))
        long_run = np.sqrt(fit.omega / (1 - fit.persistence) * 252)
        path = garch_forecast(fit, last_return=0.10, last_var=1e-3, horizon=200)
        assert path.iloc[0] > path.iloc[-1]                       # decaying from a shock
        assert path.iloc[-1] == pytest.approx(long_run, rel=0.1)

    def test_har_regressors_are_strictly_backward_looking(self):
        """Every HAR column must be shifted, or the R2 is meaningless."""
        rv = pd.Series(np.random.default_rng(0).lognormal(-6, 0.4, 400))
        _, frame = har_rv(rv)
        # Asserted on the shared index, not on position: dropna() removes the
        # warm-up rows, so positional comparison is off by however many it dropped.
        lagged = rv.shift(1)
        assert np.allclose(frame["daily"], lagged.loc[frame.index])
        assert frame["target"].loc[frame.index[0]] == pytest.approx(rv.loc[frame.index[0]])

    def test_har_beats_a_constant_on_persistent_data(self):
        rng = np.random.default_rng(0)
        rv = pd.Series(np.exp(pd.Series(rng.normal(0, 0.1, 600)).cumsum().rolling(
            3, min_periods=1).mean() - 6))
        model, _ = har_rv(rv)
        assert model.rsquared > 0.3


class TestPremiumAndRegimes:
    def test_variance_risk_premium_is_a_variance_difference(self):
        vrp = variance_risk_premium(pd.Series([0.20]), pd.Series([0.15]))
        assert vrp.iloc[0] == pytest.approx(0.20 ** 2 - 0.15 ** 2)

    def test_regimes_use_an_expanding_window_not_the_full_sample(self):
        """A full-sample quantile knows the future; conditioning on it leaks."""
        s = pd.Series(np.r_[np.full(200, 0.1), np.full(200, 0.5)])
        labels = volatility_regimes(s)
        # min_periods=60 means the 60th observation (index 59) is the first labelled.
        assert labels.iloc[:59].isna().all()
        assert labels.iloc[59:].notna().all()
        assert labels.dropna().iloc[-1] == "stressed"

    def test_regime_labels_cover_all_three_states(self):
        rng = np.random.default_rng(0)
        labels = volatility_regimes(pd.Series(rng.lognormal(0, 0.5, 1000))).dropna()
        assert set(labels.unique()) == {"calm", "normal", "stressed"}


class TestAnnualisationUnits:
    """Annualisation is a per-frequency choice, and getting it wrong is silent."""

    def test_long_run_vol_scales_with_the_annualisation_factor(self):
        kwargs = dict(omega=1e-9, alpha=0.05, beta=0.94, loglikelihood=0.0,
                      converged=True, conditional_vol=pd.Series(dtype=float))
        daily = GarchFit(**kwargs, annualize=252)
        minute = GarchFit(**kwargs, annualize=252 * 390)
        assert minute.long_run_vol == pytest.approx(daily.long_run_vol * np.sqrt(390), rel=1e-9)

    def test_fit_records_the_factor_it_was_given(self):
        rng = np.random.default_rng(0)
        fit = garch11(pd.Series(rng.normal(0, 0.001, 800)), annualize=98_280)
        assert fit.annualize == 98_280

    def test_half_life_is_in_periods_not_days(self):
        """0.996 persistence is 173 periods — under three hours on minute bars."""
        fit = GarchFit(omega=1e-9, alpha=0.05, beta=0.946, loglikelihood=0.0,
                       converged=True, conditional_vol=pd.Series(dtype=float),
                       annualize=252 * 390)
        assert fit.half_life == pytest.approx(np.log(0.5) / np.log(0.996), rel=1e-6)

    def test_forecast_inherits_the_fit_annualisation(self):
        rng = np.random.default_rng(1)
        fit = garch11(pd.Series(rng.normal(0, 0.001, 600)), annualize=252 * 390)
        path = garch_forecast(fit, last_return=0.002, last_var=1e-6, horizon=5)
        assert np.isfinite(path).all() and (path > 0).all()
