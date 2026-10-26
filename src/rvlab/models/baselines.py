"""
rvlab.models.baselines
======================
The bar any smile forecast has to clear, packaged as scikit-learn estimators so
they drop straight into a `Pipeline` / `GridSearchCV` / `cross_val_predict`.

    carry   y_hat(t+1) = y(t)                    "nothing changes"
    AR(1)   y_hat(t+1) = a + b*y(t)              "one lag of mean reversion"
    rough   y_hat(t+1) = alpha_med * T^(H-1/2) * atm_iv(t)
    rough-conditioned carry
            y_hat(t+1) = y(t) + a + b*(rough(t) - y(t))

Why carry is the honest benchmark: a smile observed a minute ago is an
extraordinarily good forecast of the smile a minute from now. Any model that
cannot beat "no change" is not earning the turnover, model risk and complexity
it costs. Notebook 07 shows the raw rough forecast losing this contest, and the
conditioned form winning it narrowly.

Every estimator here follows the sklearn contract: `fit(X, y)` where `X` is a
DataFrame carrying the named columns each model needs, and `predict(X)` returns
a plain array.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


def _column(X, name: str) -> np.ndarray:
    """Pull a named column out of a DataFrame, with a directive error message."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"{type(X).__name__} given; these estimators need a DataFrame so they "
            f"can find the column '{name}'. Use .set_output(transform='pandas') "
            "on any upstream transformer."
        )
    if name not in X.columns:
        raise KeyError(f"column '{name}' not in X (have: {list(X.columns)[:12]})")
    return X[name].to_numpy(dtype=float)


class CarryForecaster(BaseEstimator, RegressorMixin):
    """Predict the current level. The random walk; the benchmark to beat.

    Parameters
    ----------
    level_col : the column of `X` holding the current observation of the target.
    """

    def __init__(self, level_col: str = "atm_total_var"):
        self.level_col = level_col

    def fit(self, X, y=None):
        _column(X, self.level_col)          # fail fast on a bad column name
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        return _column(X, self.level_col)


class AR1Forecaster(BaseEstimator, RegressorMixin):
    """y_hat(t+1) = a + b*y(t), fitted by OLS on the training fold."""

    def __init__(self, level_col: str = "atm_total_var"):
        self.level_col = level_col

    def fit(self, X, y):
        x = _column(X, self.level_col)
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3:
            self.coef_, self.intercept_ = 1.0, 0.0          # degenerate -> carry
        else:
            self.coef_, self.intercept_ = np.polyfit(x[ok], y[ok], 1)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.intercept_ + self.coef_ * _column(X, self.level_col)


class RoughStructuralForecaster(BaseEstimator, RegressorMixin):
    """The raw rough forecast — the thing this project set out to test.

    Fit takes the median structural coefficient over the training fold; predict
    reassembles the observable from the scaling law at each row's own maturity:

        rr25_hat = median(alpha) * T^(H - 1/2) * atm_iv
        bf25_hat = median(gamma) * T^(2H - 1)  * atm_total_var

    The median (not the mean) is deliberate: the structural coefficients have
    heavy tails wherever the IV solver struggled.

    target : "rr25" or "bf25" — selects which scaling law applies.
    """

    def __init__(self, target: str = "rr25", hurst: float = 0.10,
                 T_col: str = "T", atm_iv_col: str = "atm_iv",
                 atm_total_var_col: str = "atm_total_var"):
        self.target = target
        self.hurst = hurst
        self.T_col = T_col
        self.atm_iv_col = atm_iv_col
        self.atm_total_var_col = atm_total_var_col

    # ── internals ─────────────────────────────────────────────────────────────
    def _exponent(self) -> float:
        return self.hurst - 0.5 if self.target == "rr25" else 2 * self.hurst - 1.0

    def _scale(self, X) -> np.ndarray:
        col = self.atm_iv_col if self.target == "rr25" else self.atm_total_var_col
        return _column(X, self.T_col) ** self._exponent() * _column(X, col)

    # ── sklearn API ───────────────────────────────────────────────────────────
    def fit(self, X, y):
        if self.target not in ("rr25", "bf25"):
            raise ValueError(f"target must be 'rr25' or 'bf25', got {self.target!r}")
        scale = self._scale(X)
        y = np.asarray(y, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            coeffs = np.where(np.abs(scale) > 1e-12, y / scale, np.nan)
        finite = coeffs[np.isfinite(coeffs)]
        self.coefficient_ = float(np.median(finite)) if finite.size else 0.0
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.coefficient_ * self._scale(X)


class CarryConditionedRough(BaseEstimator, RegressorMixin):
    """Rough as a *correction to* carry, rather than a replacement for it.

    Regresses the carry error on the rough-minus-carry spread::

        y(t+1) - y(t) = a + b * (rough(t) - y(t))
        y_hat(t+1)    = y(t) + a + b * (rough(t) - y(t))

    b is the honest measure of incremental information: b = 0 means rough adds
    nothing beyond carry, and the model degenerates to carry exactly. This is
    the form that passes in notebook 07 where the raw form fails — and the
    reason is visible right here, in that the model can only ever *adjust* a
    forecast that was already good.
    """

    def __init__(self, target: str = "bf25", hurst: float = 0.10,
                 level_col: str | None = None, T_col: str = "T",
                 atm_iv_col: str = "atm_iv", atm_total_var_col: str = "atm_total_var"):
        self.target = target
        self.hurst = hurst
        self.level_col = level_col
        self.T_col = T_col
        self.atm_iv_col = atm_iv_col
        self.atm_total_var_col = atm_total_var_col

    def _level_name(self) -> str:
        return self.level_col or self.target

    def _rough(self, X) -> np.ndarray:
        return self.rough_.predict(X)

    def fit(self, X, y):
        self.rough_ = RoughStructuralForecaster(
            target=self.target, hurst=self.hurst, T_col=self.T_col,
            atm_iv_col=self.atm_iv_col, atm_total_var_col=self.atm_total_var_col,
        ).fit(X, y)

        carry = _column(X, self._level_name())
        spread = self._rough(X) - carry
        resid = np.asarray(y, dtype=float) - carry

        ok = np.isfinite(spread) & np.isfinite(resid)
        if ok.sum() < 3:
            self.intercept_, self.beta_ = 0.0, 0.0          # degenerate -> carry
        else:
            self.beta_, self.intercept_ = np.polyfit(spread[ok], resid[ok], 1)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        carry = _column(X, self._level_name())
        return carry + self.intercept_ + self.beta_ * (self._rough(X) - carry)


__all__ = [
    "CarryForecaster", "AR1Forecaster", "RoughStructuralForecaster",
    "CarryConditionedRough",
]
