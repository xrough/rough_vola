"""
rvlab.pipelines.transformers
============================
Custom scikit-learn transformers for smile panels.

The reason to write a transformer instead of a function: a transformer is
*fitted*, so any statistic it needs (a winsorizing quantile, a median for
imputation, a scaling constant) is learned on the training fold only and then
applied unchanged to the test fold. The same computation done with pandas before
splitting has already seen the test set.

All transformers here support `set_output(transform="pandas")` and are safe
inside `ColumnTransformer` and `Pipeline`.

> ⚠️ **Leakage:** the rule is not "clean the data, then split". It is
> "split, then fit the cleaning on train". If a step has any `fit`-time state,
> it belongs in the pipeline, not in a preparation cell.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def _as_frame(X, columns=None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(np.asarray(X), columns=columns)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to training-fold quantiles.

    Prefer this to dropping outliers: an option-chain outlier is usually a bad
    quote, not a missing observation, and deleting the row deletes the *whole
    smile* at that instant including the columns that were fine.

    Parameters
    ----------
    lower, upper : quantiles learned at fit time (default 1% / 99%).
    """

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        df = _as_frame(X)
        self.feature_names_in_ = np.asarray(df.columns, dtype=object)
        self.n_features_in_ = df.shape[1]
        self.lower_bounds_ = df.quantile(self.lower)
        self.upper_bounds_ = df.quantile(self.upper)
        return self

    def transform(self, X):
        check_is_fitted(self)
        df = _as_frame(X, self.feature_names_in_)
        return df.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return np.asarray(self.feature_names_in_, dtype=object)


class StructuralCoefficients(BaseEstimator, TransformerMixin):
    """Append rough structural coefficients alpha and gamma at a given H.

    Stateless in the statistical sense — it learns nothing at fit time — but
    written as a transformer so `H` becomes a tunable hyperparameter::

        GridSearchCV(pipe, {"rough__hurst": [0.03, 0.05, 0.10, 0.20]})

    That is the honest way to sweep H: inside the CV loop, scored out of sample,
    rather than picking the H that looked best on the whole sample.
    """

    def __init__(self, hurst: float = 0.10, drop_inputs: bool = False):
        self.hurst = hurst
        self.drop_inputs = drop_inputs

    def fit(self, X, y=None):
        df = _as_frame(X)
        missing = {"T", "atm_iv", "atm_total_var", "rr25", "bf25"} - set(df.columns)
        if missing:
            raise KeyError(f"StructuralCoefficients needs columns {sorted(missing)}")
        self.feature_names_in_ = np.asarray(df.columns, dtype=object)
        self.n_features_in_ = df.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self)
        from ..models import rough

        df = _as_frame(X, self.feature_names_in_).copy()
        df["alpha"] = rough.structural_alpha(df["rr25"], df["T"], df["atm_iv"], self.hurst)
        df["gamma"] = rough.structural_gamma(df["bf25"], df["T"],
                                             df["atm_total_var"], self.hurst)
        if self.drop_inputs:
            df = df.drop(columns=["rr25", "bf25"])
        self.feature_names_out_ = np.asarray(df.columns, dtype=object)
        return df

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return np.asarray(getattr(self, "feature_names_out_", self.feature_names_in_),
                          dtype=object)


class LagFeatures(BaseEstimator, TransformerMixin):
    """Append lagged copies of selected columns, grouped so lags stay within an expiry.

    Rows whose lag is undefined (the first `max(lags)` bars of each group) come
    back as NaN — chain an imputer after this, or drop them before fitting.
    """

    def __init__(self, columns=None, lags=(1, 2), group=("expiry",), sort_by: str = "ts"):
        self.columns = columns
        self.lags = lags
        self.group = group
        self.sort_by = sort_by

    def fit(self, X, y=None):
        df = _as_frame(X)
        self.feature_names_in_ = np.asarray(df.columns, dtype=object)
        self.n_features_in_ = df.shape[1]
        self.columns_ = list(self.columns) if self.columns is not None else \
            [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        return self

    def transform(self, X):
        check_is_fitted(self)
        from ..features.timeseries import add_lags

        df = _as_frame(X, self.feature_names_in_)
        group = tuple(g for g in (self.group or ()) if g in df.columns)
        sort_by = self.sort_by if self.sort_by in df.columns else None
        if sort_by is None:
            out = df.copy()
            for col in self.columns_:
                for k in self.lags:
                    out[f"{col}_lag{k}"] = out[col].shift(k)
        else:
            out = add_lags(df, self.columns_, self.lags, group=group, sort_by=sort_by)
        out = out.reindex(df.index)
        self.feature_names_out_ = np.asarray(out.columns, dtype=object)
        return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return np.asarray(getattr(self, "feature_names_out_", self.feature_names_in_),
                          dtype=object)


class LogTransform(BaseEstimator, TransformerMixin):
    """Signed log transform: sign(x) * log1p(|x|).

    Ordinary `np.log` is unusable on RR25, which is negative most of the time.
    This version is defined on the whole real line, is monotone, and leaves
    values near zero almost unchanged — so it tames the tails without inventing
    structure near the middle of the distribution.
    """

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        df = _as_frame(X)
        self.feature_names_in_ = np.asarray(df.columns, dtype=object)
        self.n_features_in_ = df.shape[1]
        self.columns_ = list(self.columns) if self.columns is not None else list(df.columns)
        return self

    def transform(self, X):
        check_is_fitted(self)
        df = _as_frame(X, self.feature_names_in_).copy()
        for col in self.columns_:
            v = df[col].to_numpy(dtype=float)
            df[col] = np.sign(v) * np.log1p(np.abs(v))
        return df

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return np.asarray(self.feature_names_in_, dtype=object)


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Keep named columns, in order, and fail loudly when one is missing.

    Exists because a silent column drop inside a `ColumnTransformer` is very
    hard to spot: the model still fits, just without the feature you thought was
    doing the work.
    """

    def __init__(self, columns, errors: str = "raise"):
        self.columns = columns
        self.errors = errors

    def fit(self, X, y=None):
        df = _as_frame(X)
        missing = [c for c in self.columns if c not in df.columns]
        if missing and self.errors == "raise":
            raise KeyError(f"ColumnSelector: missing {missing}; have {list(df.columns)[:12]}")
        self.columns_ = [c for c in self.columns if c in df.columns]
        self.feature_names_in_ = np.asarray(df.columns, dtype=object)
        self.n_features_in_ = df.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self)
        return _as_frame(X, self.feature_names_in_)[self.columns_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return np.asarray(self.columns_, dtype=object)


__all__ = [
    "Winsorizer", "StructuralCoefficients", "LagFeatures", "LogTransform",
    "ColumnSelector",
]
