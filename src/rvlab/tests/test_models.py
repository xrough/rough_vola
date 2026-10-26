"""Black-Scholes identities, the rough scaling laws, and baseline degeneracy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rvlab.models import blackscholes as bs
from rvlab.models import rough
from rvlab.models.baselines import (
    AR1Forecaster, CarryConditionedRough, CarryForecaster, RoughStructuralForecaster,
)


class TestBlackScholes:
    def test_implied_vol_round_trip(self):
        """price -> implied_vol must recover the input vol to ~machine epsilon."""
        F, T = 600.0, 0.25
        K = np.array([480.0, 550.0, 600.0, 660.0, 720.0])
        for sigma in (0.08, 0.20, 0.45, 0.90):
            p = bs.price(sigma, F, K, T, True)
            assert np.allclose(bs.implied_vol(p, F, K, T, True), sigma, atol=1e-8)

    def test_put_call_parity(self):
        F, K, T, s, r = 600.0, np.array([550.0, 600.0, 650.0]), 0.3, 0.22, 0.053
        c, p = bs.price(s, F, K, T, True, r), bs.price(s, F, K, T, False, r)
        assert np.allclose(c - p, np.exp(-r * T) * (F - K), atol=1e-10)

    def test_delta_matches_numerical_derivative(self):
        F, K, T, s = 600.0, 610.0, 0.25, 0.2
        h = 1e-4
        numeric = (bs.price(s, F + h, K, T) - bs.price(s, F - h, K, T)) / (2 * h)
        assert bs.delta(s, F, K, T) == pytest.approx(float(numeric), abs=1e-6)

    def test_vega_matches_numerical_derivative(self):
        F, K, T, s = 600.0, 610.0, 0.25, 0.2
        h = 1e-6
        numeric = (bs.price(s + h, F, K, T) - bs.price(s - h, F, K, T)) / (2 * h)
        assert bs.vega(s, F, K, T) == pytest.approx(float(numeric), rel=1e-5)

    def test_implied_vol_nan_outside_arbitrage_bounds(self):
        """A price above the forward is unattainable; the solver must say so."""
        assert np.isnan(bs.implied_vol(np.array([1e6]), 600.0, 600.0, 0.25, True))


class TestRoughScaling:
    def test_hurst_slope_inverse(self):
        for h in (0.03, 0.1, 0.25, 0.49):
            assert rough.implied_hurst_from_slope(rough.skew_exponent(h)) == pytest.approx(h)

    def test_structural_alpha_inverts_the_scaling_law(self):
        """alpha recovered from a synthetic RR25 must equal the alpha used to build it."""
        h, T, atm_iv, alpha = 0.10, np.array([0.02, 0.05, 0.16]), 0.18, -0.17
        rr25 = alpha * T ** (h - 0.5) * atm_iv
        assert np.allclose(rough.structural_alpha(rr25, T, atm_iv, h), alpha)

    def test_structural_gamma_inverts_the_scaling_law(self):
        h, T, atv, gamma = 0.10, np.array([0.02, 0.05, 0.16]), 0.004, 0.55
        bf25 = gamma * T ** (2 * h - 1) * atv
        assert np.allclose(rough.structural_gamma(bf25, T, atv, h), gamma)

    def test_alpha_is_maturity_invariant_only_at_the_true_hurst(self):
        """The whole empirical test in notebook 04, as an assertion."""
        h_true, T, atm_iv, alpha = 0.10, np.array([0.02, 0.05, 0.10, 0.16]), 0.18, -0.17
        rr25 = alpha * T ** (h_true - 0.5) * atm_iv
        spread_right = np.ptp(rough.structural_alpha(rr25, T, atm_iv, h_true))
        spread_wrong = np.ptp(rough.structural_alpha(rr25, T, atm_iv, 0.40))
        assert spread_right < 1e-12
        assert spread_wrong > 0.05

    def test_smile_is_downward_sloping_when_rho_negative(self):
        p = rough.RoughVolParams(rho=-0.7)
        low = rough.bergomi_guyon_smile(90.0, 100.0, 0.25, p)
        high = rough.bergomi_guyon_smile(110.0, 100.0, 0.25, p)
        assert low > high

    def test_atm_vol_equals_sqrt_xi0(self):
        p = rough.RoughVolParams(xi0=0.0625)
        assert float(rough.bergomi_guyon_smile(100.0, 100.0, 0.25, p)) == pytest.approx(0.25)

    def test_skew_explodes_as_maturity_shrinks(self):
        """|psi| must increase as T -> 0. This is the rough signature."""
        p = rough.RoughVolParams(H=0.10)
        assert abs(rough.psi(1 / 365, p)) > abs(rough.psi(30 / 365, p)) > abs(rough.psi(1.0, p))


class TestBaselines:
    @staticmethod
    def _frame(n=200, seed=0):
        rng = np.random.default_rng(seed)
        bf25 = 0.004 + np.cumsum(rng.normal(0, 1e-4, n))
        return pd.DataFrame({
            "T": np.full(n, 0.08), "atm_iv": np.full(n, 0.18),
            "atm_total_var": np.full(n, 0.18**2 * 0.08), "rr25": -0.03 + rng.normal(0, 1e-3, n),
            "bf25": bf25,
        }), pd.Series(np.roll(bf25, -1))

    def test_carry_predicts_the_current_level(self):
        X, y = self._frame()
        assert np.allclose(CarryForecaster("bf25").fit(X, y).predict(X), X["bf25"])

    def test_ar1_recovers_a_known_linear_map(self):
        X, _ = self._frame()
        y = 0.5 * X["bf25"] + 0.001
        m = AR1Forecaster("bf25").fit(X, y)
        assert m.coef_ == pytest.approx(0.5, abs=1e-8)
        assert m.intercept_ == pytest.approx(0.001, abs=1e-8)

    def test_rough_structural_recovers_the_median_coefficient(self):
        X, _ = self._frame()
        y = 0.55 * X["T"] ** (2 * 0.1 - 1) * X["atm_total_var"]
        m = RoughStructuralForecaster("bf25", hurst=0.1).fit(X, y)
        assert m.coefficient_ == pytest.approx(0.55, rel=1e-9)

    def test_conditioned_rough_degenerates_to_carry_when_rough_is_useless(self):
        """beta ~ 0 must make the model reproduce carry, not merely resemble it."""
        X, _ = self._frame()
        y = X["bf25"].copy()                      # perfect carry: nothing to add
        m = CarryConditionedRough("bf25", hurst=0.1).fit(X, y)
        assert m.beta_ == pytest.approx(0.0, abs=1e-6)
        assert np.allclose(m.predict(X), X["bf25"], atol=1e-9)

    def test_rejects_numpy_input_with_a_readable_error(self):
        X, y = self._frame()
        with pytest.raises(TypeError, match="DataFrame"):
            CarryForecaster("bf25").fit(X.to_numpy(), y)

    def test_rejects_unknown_target(self):
        X, y = self._frame()
        with pytest.raises(ValueError, match="rr25.*bf25"):
            RoughStructuralForecaster(target="atm_iv").fit(X, y)
