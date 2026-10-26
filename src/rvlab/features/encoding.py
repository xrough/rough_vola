"""
rvlab.features.encoding
=======================
Categorical encoders that are safe inside cross-validation.

Target encoding — replacing a category with the mean outcome for that category —
is the highest-value encoding for high-cardinality features and the easiest to
get catastrophically wrong. Computed on the whole training set, it leaks the
target into the features: a category appearing once takes that row's own label as
its encoding, and the model reads the answer off its own input.

`CVTargetEncoder` fixes it the only way that works: each row's encoding is
computed from *other* folds, never its own. The test in
`rvlab/tests/test_encoding.py` asserts the property directly — encoding a random
target must produce no cross-validated skill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted


def _as_frame(X, columns=None) -> pd.DataFrame:
    return X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X), columns=columns)


class CVTargetEncoder(BaseEstimator, TransformerMixin):
    """Out-of-fold target encoding with smoothing toward the global mean.

    For each column, a category's encoding is the mean target of the rows in
    *other* folds, shrunk toward the global mean in proportion to how rare the
    category is:

        encoding = (n * category_mean + m * global_mean) / (n + m)

    `smoothing` is `m`. It is what stops a category seen twice from taking the
    average of those two labels as gospel — the shrinkage is strongest exactly
    where the estimate is least reliable.

    At `transform` time on unseen data, the full-training-set mapping is used;
    unseen categories fall back to the global mean.

    Pass a `cv` that matches your outer validation. On time-series data an
    ordinary `KFold` here will still leak *across time* even though it does not
    leak the target directly — use one of the purged splitters.
    """

    def __init__(self, columns=None, smoothing: float = 20.0, cv=5,
                 random_state: int | None = None):
        self.columns = columns
        self.smoothing = smoothing
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y):
        df = _as_frame(X)
        y = pd.Series(np.asarray(y, dtype=float), index=df.index)

        self.feature_names_in_ = np.asarray(df.columns, dtype=object)
        self.n_features_in_ = df.shape[1]
        self.columns_ = list(self.columns) if self.columns is not None else [
            c for c in df.columns
            if df[c].dtype == object or str(df[c].dtype) == "category"]
        self.global_mean_ = float(y.mean())

        # Full-training mapping, used at transform time on new data.
        self.mapping_ = {}
        for col in self.columns_:
            stats = y.groupby(df[col].astype(object), observed=True).agg(["mean", "count"])
            self.mapping_[col] = ((stats["mean"] * stats["count"]
                                   + self.global_mean_ * self.smoothing)
                                  / (stats["count"] + self.smoothing))

        # Out-of-fold encoding for the training rows themselves.
        splitter = (self.cv if hasattr(self.cv, "split")
                    else KFold(n_splits=self.cv, shuffle=True,
                               random_state=self.random_state))
        self.oof_ = pd.DataFrame(index=df.index, columns=self.columns_, dtype=float)
        for train_idx, test_idx in splitter.split(df):
            fold_y = y.iloc[train_idx]
            fold_mean = float(fold_y.mean())
            for col in self.columns_:
                stats = fold_y.groupby(df[col].astype(object).iloc[train_idx],
                                       observed=True).agg(["mean", "count"])
                enc = ((stats["mean"] * stats["count"] + fold_mean * self.smoothing)
                       / (stats["count"] + self.smoothing))
                self.oof_.iloc[test_idx, self.oof_.columns.get_loc(col)] = (
                    df[col].astype(object).iloc[test_idx].map(enc).fillna(fold_mean).to_numpy())
        self._fit_index = df.index
        return self

    def transform(self, X):
        check_is_fitted(self)
        df = _as_frame(X, self.feature_names_in_)
        out = df.copy()

        # Rows seen at fit time get their out-of-fold encoding; anything else gets
        # the full-training mapping. This is what keeps `fit_transform` honest.
        same = df.index.equals(self._fit_index)
        for col in self.columns_:
            if same:
                out[col] = self.oof_[col].to_numpy()
            else:
                out[col] = (df[col].astype(object).map(self.mapping_[col])
                            .fillna(self.global_mean_).to_numpy())
        return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return np.asarray(self.feature_names_in_, dtype=object)


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Fold categories below a frequency threshold into a single `"__rare__"` level.

    Two problems solved at once: hundreds of near-empty one-hot columns, and
    categories that appear only in the test set. Everything rare becomes one level
    that the model has actually seen.
    """

    def __init__(self, columns=None, min_frequency: float = 0.01,
                 rare_label: str = "__rare__"):
        self.columns = columns
        self.min_frequency = min_frequency
        self.rare_label = rare_label

    def fit(self, X, y=None):
        df = _as_frame(X)
        self.feature_names_in_ = np.asarray(df.columns, dtype=object)
        self.n_features_in_ = df.shape[1]
        self.columns_ = list(self.columns) if self.columns is not None else [
            c for c in df.columns
            if df[c].dtype == object or str(df[c].dtype) == "category"]
        self.keep_ = {c: set(df[c].value_counts(normalize=True)
                             .loc[lambda s: s >= self.min_frequency].index)
                      for c in self.columns_}
        return self

    def transform(self, X):
        check_is_fitted(self)
        df = _as_frame(X, self.feature_names_in_).copy()
        for col in self.columns_:
            df[col] = df[col].where(df[col].isin(self.keep_[col]), self.rare_label)
        return df

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return np.asarray(self.feature_names_in_, dtype=object)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace a category with how often it occurs.

    Cheap, leak-free (it uses no target), and often surprisingly strong — how
    common something is frequently carries real information, and unlike target
    encoding it cannot leak.
    """

    def __init__(self, columns=None, normalize: bool = True):
        self.columns = columns
        self.normalize = normalize

    def fit(self, X, y=None):
        df = _as_frame(X)
        self.feature_names_in_ = np.asarray(df.columns, dtype=object)
        self.n_features_in_ = df.shape[1]
        self.columns_ = list(self.columns) if self.columns is not None else [
            c for c in df.columns
            if df[c].dtype == object or str(df[c].dtype) == "category"]
        self.frequencies_ = {c: df[c].value_counts(normalize=self.normalize)
                             for c in self.columns_}
        return self

    def transform(self, X):
        check_is_fitted(self)
        df = _as_frame(X, self.feature_names_in_).copy()
        for col in self.columns_:
            df[col] = df[col].map(self.frequencies_[col]).fillna(0.0).astype(float)
        return df

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return np.asarray(self.feature_names_in_, dtype=object)


class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """Encode a periodic integer as a sine/cosine pair.

    Month, weekday and hour are circular: December is adjacent to January, and an
    integer encoding tells the model they are eleven apart. Two columns fix it,
    and the pair keeps the distance between any two points correct.
    """

    def __init__(self, columns: dict | None = None):
        self.columns = columns or {}

    def fit(self, X, y=None):
        df = _as_frame(X)
        self.feature_names_in_ = np.asarray(df.columns, dtype=object)
        self.n_features_in_ = df.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self)
        df = _as_frame(X, self.feature_names_in_).copy()
        for col, period in self.columns.items():
            if col not in df.columns:
                continue
            radians = 2 * np.pi * df[col].astype(float) / period
            df[f"{col}_sin"] = np.sin(radians)
            df[f"{col}_cos"] = np.cos(radians)
            df = df.drop(columns=col)
        self.feature_names_out_ = np.asarray(df.columns, dtype=object)
        return df

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return np.asarray(getattr(self, "feature_names_out_", self.feature_names_in_),
                          dtype=object)


__all__ = ["CVTargetEncoder", "RareCategoryGrouper", "FrequencyEncoder", "CyclicalEncoder"]
