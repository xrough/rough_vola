"""
rvlab.pipelines.build
=====================
Pipeline factories, so a notebook writes one line instead of fifteen.

Every pipeline returned here has the same shape::

    impute -> winsorize -> scale -> estimator

with `set_output(transform="pandas")` already applied, so feature names survive
all the way to `permutation_importance` and the coefficient tables in
notebook 08. Losing column names at the first `StandardScaler` is the reason so
many interpretability plots are labelled `x0, x1, x2`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet, HuberRegressor, Lasso, LinearRegression, LogisticRegression, Ridge,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from .transformers import Winsorizer

REGRESSORS = {
    "ols": LinearRegression(),
    "ridge": Ridge(alpha=1.0),
    "lasso": Lasso(alpha=1e-4, max_iter=5_000),
    "elasticnet": ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=5_000),
    "huber": HuberRegressor(alpha=1e-4, max_iter=500),
    "rf": RandomForestRegressor(n_estimators=200, min_samples_leaf=20,
                                n_jobs=-1, random_state=42),
    "hgb": HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08,
                                         random_state=42),
}


# Gradient boosters are optional and constructed lazily, so importing rvlab does
# not pay their (substantial) import cost and a missing one is not fatal.
OPTIONAL_BOOSTERS = ("xgb", "lgbm", "catboost")


def available_regressors() -> list[str]:
    """Names accepted by `make_regression_pipeline` on this machine."""
    names = list(REGRESSORS)
    for name, module in (("xgb", "xgboost"), ("lgbm", "lightgbm"),
                         ("catboost", "catboost")):
        try:
            __import__(module)
            names.append(name)
        except ImportError:
            pass
    return names


def _booster(name: str, **overrides):
    """Construct an optional gradient booster with sane panel-data defaults.

    The shared defaults matter more than the library: shallow trees, a low
    learning rate, generous `min_child_samples`/`min_child_weight`, and column
    subsampling. Financial panels have a very low signal-to-noise ratio, and a
    booster left on its defaults will fit the noise beautifully.
    """
    if name == "xgb":
        from xgboost import XGBRegressor
        params = dict(n_estimators=400, max_depth=5, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, min_child_weight=50,
                      reg_lambda=1.0, random_state=42, n_jobs=-1,
                      tree_method="hist")
        return XGBRegressor(**{**params, **overrides})
    if name == "lgbm":
        from lightgbm import LGBMRegressor
        params = dict(n_estimators=400, num_leaves=31, learning_rate=0.05,
                      min_child_samples=100, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.8, reg_lambda=1.0, random_state=42,
                      n_jobs=-1, verbosity=-1)
        return LGBMRegressor(**{**params, **overrides})
    if name == "catboost":
        from catboost import CatBoostRegressor
        params = dict(iterations=400, depth=6, learning_rate=0.05,
                      l2_leaf_reg=3.0, random_seed=42, verbose=0,
                      allow_writing_files=False)
        return CatBoostRegressor(**{**params, **overrides})
    raise KeyError(name)


def _estimator(model):
    """Resolve a name to an estimator instance; pass estimators through."""
    if not isinstance(model, str):
        return model
    if model in OPTIONAL_BOOSTERS:
        return _booster(model)
    if model not in REGRESSORS:
        raise KeyError(f"unknown model {model!r}; choose from {available_regressors()}")
    from sklearn.base import clone
    return clone(REGRESSORS[model])


def make_regression_pipeline(model="ridge", scale: bool = True, winsorize: bool = True,
                             impute: str = "median", quantiles=(0.01, 0.99)) -> Pipeline:
    """impute -> winsorize -> scale -> regressor, with pandas output throughout.

    Parameters
    ----------
    model     : a name from `available_regressors()`, or any sklearn estimator.
    scale     : RobustScaler when winsorizing (medians and IQR), else StandardScaler.
    winsorize : clip to training-fold quantiles before scaling.
    impute    : "median", "mean", or None.

    Tree models ignore scaling, but leaving it in costs nothing and keeps the
    pipeline identical across model families — which is what makes the
    comparison in notebook 06 a fair one.
    """
    steps = []
    if impute:
        steps.append(("impute", SimpleImputer(strategy=impute)))
    if winsorize:
        steps.append(("winsor", Winsorizer(lower=quantiles[0], upper=quantiles[1])))
    if scale:
        steps.append(("scale", RobustScaler() if winsorize else StandardScaler()))
    steps.append(("model", _estimator(model)))

    return Pipeline(steps).set_output(transform="pandas")


def make_classification_pipeline(model="logistic", scale: bool = True,
                                 impute: str = "median", class_weight="balanced"
                                 ) -> Pipeline:
    """The same shape for a classification target (e.g. sign of the next move).

    `class_weight="balanced"` by default: direction targets are rarely 50/50,
    and an unweighted classifier that always predicts the majority class scores
    well on accuracy while being useless. Notebook 07 scores these with ROC-AUC
    and average precision for the same reason.
    """
    from sklearn.base import clone
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

    table = {
        "logistic": LogisticRegression(max_iter=2_000, class_weight=class_weight),
        "rf": RandomForestClassifier(n_estimators=200, min_samples_leaf=20,
                                     n_jobs=-1, random_state=42,
                                     class_weight=class_weight),
        "hgb": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                              random_state=42),
    }
    est = clone(table[model]) if isinstance(model, str) else model

    steps = []
    if impute:
        steps.append(("impute", SimpleImputer(strategy=impute)))
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", est))
    return Pipeline(steps).set_output(transform="pandas")


def make_column_transformer(numeric=None, categorical=None,
                            passthrough=None) -> ColumnTransformer:
    """Numeric and categorical branches with sensible defaults.

    Numeric: median impute -> robust scale. Categorical: most-frequent impute ->
    one-hot with `handle_unknown="ignore"`, which is what stops an unseen tenor
    bucket in the test fold from raising at predict time.
    """
    from sklearn.preprocessing import OneHotEncoder

    branches = []
    if numeric:
        branches.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
        ]), list(numeric)))
    if categorical:
        branches.append(("cat", Pipeline([
            # `missing_values=pd.NA` handles pandas StringDtype, object columns
            # containing pd.NA, and ordinary np.nan.  sklearn's np.nan default
            # raises "boolean value of NA is ambiguous" on StringDtype input.
            ("impute", SimpleImputer(strategy="most_frequent", missing_values=pd.NA)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), list(categorical)))
    if passthrough:
        branches.append(("pass", "passthrough", list(passthrough)))

    return ColumnTransformer(branches, remainder="drop",
                             verbose_feature_names_out=False).set_output(transform="pandas")


def make_panel_pipeline(model="lgbm", numeric=None, categorical=None,
                        scale: bool | None = None) -> Pipeline:
    """A `ColumnTransformer` + estimator pipeline for a mixed-dtype panel.

    Numeric columns are median-imputed (with an indicator, because "this feature
    was missing" is itself informative on a panel where coverage varies) and
    scaled only for models that need it. Categorical columns are one-hot encoded
    with `handle_unknown="ignore"` — the setting that stops a security code
    appearing for the first time in the test set from raising at predict time.

    Scaling is skipped automatically for tree models, where it is pure cost.
    """
    from sklearn.preprocessing import OneHotEncoder

    est = _estimator(model)
    is_tree = isinstance(model, str) and model in (*OPTIONAL_BOOSTERS, "rf", "hgb")
    scale = (not is_tree) if scale is None else scale

    numeric_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        numeric_steps.append(("scale", RobustScaler()))

    branches = []
    if numeric:
        branches.append(("num", Pipeline(numeric_steps), list(numeric)))
    if categorical:
        branches.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", missing_values=pd.NA)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                     min_frequency=0.001)),
        ]), list(categorical)))

    prep = ColumnTransformer(branches, remainder="drop",
                             verbose_feature_names_out=False)
    return Pipeline([("prep", prep), ("model", est)]).set_output(transform="pandas")


def rough_hurst_grid(model_step: str = "model", hurst_values=None) -> dict:
    """A parameter grid that tunes H *inside* cross-validation.

    Returns a dict for `GridSearchCV(param_grid=...)` assuming a pipeline with a
    `StructuralCoefficients` step named "rough".
    """
    hurst_values = list(hurst_values or [0.03, 0.05, 0.07, 0.10, 0.15, 0.20])
    return {"rough__hurst": hurst_values}


__all__ = [
    "make_regression_pipeline", "make_classification_pipeline",
    "make_column_transformer", "make_panel_pipeline", "available_regressors",
    "rough_hurst_grid", "REGRESSORS", "OPTIONAL_BOOSTERS",
]
