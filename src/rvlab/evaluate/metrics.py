"""
rvlab.evaluate.metrics
======================
Scoring, with the emphasis on *skill against a baseline* rather than raw error.

An RMSE of 0.0003 on a smile forecast sounds excellent and means nothing on its
own: carry achieves 0.00031. The number that carries information is the ratio.

    skill = 1 - RMSE(model) / RMSE(baseline)

Positive skill beats the baseline; zero ties it; negative loses. Every headline
number in notebooks 06-09 is a skill score, and the baseline is always named.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clean(y_true, y_pred):
    """Align, cast and drop non-finite pairs. Returns (y_true, y_pred)."""
    a = np.asarray(y_true, dtype=float).ravel()
    b = np.asarray(y_pred, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: y_true {a.shape} vs y_pred {b.shape}")
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]


def rmse(y_true, y_pred) -> float:
    a, b = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2))) if a.size else np.nan


def mae(y_true, y_pred) -> float:
    a, b = _clean(y_true, y_pred)
    return float(np.mean(np.abs(a - b))) if a.size else np.nan


def bias(y_true, y_pred) -> float:
    """Mean signed error. A model can have great RMSE and a fatal bias."""
    a, b = _clean(y_true, y_pred)
    return float(np.mean(b - a)) if a.size else np.nan


def r2(y_true, y_pred) -> float:
    a, b = _clean(y_true, y_pred)
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    return 1.0 - float(np.sum((a - b) ** 2)) / ss_tot if ss_tot > 0 else np.nan


def directional_accuracy(y_true, y_pred, y_prev) -> float:
    """Fraction of moves whose *direction* was called correctly.

    Needs `y_prev` — the value the move is measured from — because a forecast of
    a level says nothing about direction until you know where it started.

    Rows where either the actual move or the predicted move is exactly zero are
    excluded, not scored as wrong. That matters for the carry baseline, whose
    predicted move is identically zero: carry makes no directional call at all,
    so its directional accuracy is undefined (NaN), not 0%. Scoring it as 0%
    would make every other model look good by comparison.
    """
    a, b = np.asarray(y_true, float).ravel(), np.asarray(y_pred, float).ravel()
    p = np.asarray(y_prev, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(p)
    actual, predicted = a[ok] - p[ok], b[ok] - p[ok]
    scored = (actual != 0) & (predicted != 0)
    if not scored.any():
        return np.nan
    return float(np.mean(np.sign(actual[scored]) == np.sign(predicted[scored])))


def skill_vs_baseline(y_true, y_pred, y_baseline, metric=rmse) -> float:
    """1 - metric(model) / metric(baseline). Positive means the model wins.

    Scored on the intersection of rows where *all three* series are finite, so
    a model is never credited for quietly skipping the hard observations.
    """
    a = np.asarray(y_true, float).ravel()
    m = np.asarray(y_pred, float).ravel()
    z = np.asarray(y_baseline, float).ravel()
    ok = np.isfinite(a) & np.isfinite(m) & np.isfinite(z)
    base = metric(a[ok], z[ok])
    if not np.isfinite(base) or base == 0:
        return np.nan
    return 1.0 - metric(a[ok], m[ok]) / base


def regression_report(y_true, predictions: dict, baseline: str | None = None,
                      y_prev=None) -> pd.DataFrame:
    """One row per model: rmse, mae, bias, r2, skill, directional accuracy.

    `predictions` maps a model name to its predicted array. `baseline` names the
    key to measure skill against — default is the first key, which by convention
    in this series is "carry".
    """
    names = list(predictions)
    baseline = baseline or names[0]
    if baseline not in predictions:
        raise KeyError(f"baseline {baseline!r} not among {names}")

    rows = []
    for name, pred in predictions.items():
        row = {"model": name, "rmse": rmse(y_true, pred), "mae": mae(y_true, pred),
               "bias": bias(y_true, pred), "r2": r2(y_true, pred),
               f"skill_vs_{baseline}": skill_vs_baseline(
                   y_true, pred, predictions[baseline])}
        if y_prev is not None:
            row["dir_acc"] = directional_accuracy(y_true, pred, y_prev)
        rows.append(row)

    return (pd.DataFrame(rows).set_index("model")
            .sort_values(f"skill_vs_{baseline}", ascending=False))


def classification_report_table(y_true, scores: dict, threshold: float = 0.5
                                ) -> pd.DataFrame:
    """Accuracy, balanced accuracy, ROC-AUC, average precision, Brier score.

    `scores` maps a model name to predicted *probabilities* of the positive
    class, not hard labels — threshold-free metrics (ROC-AUC, average precision)
    are the honest ones when the classes are unbalanced, and you cannot compute
    them from labels.
    """
    from sklearn.metrics import (
        accuracy_score, average_precision_score, balanced_accuracy_score,
        brier_score_loss, roc_auc_score,
    )

    y = np.asarray(y_true).ravel().astype(int)
    rows = []
    for name, prob in scores.items():
        p = np.asarray(prob, float).ravel()
        ok = np.isfinite(p)
        yy, pp = y[ok], p[ok]
        pred = (pp >= threshold).astype(int)
        rows.append({
            "model": name,
            "accuracy": accuracy_score(yy, pred),
            "balanced_acc": balanced_accuracy_score(yy, pred),
            "roc_auc": roc_auc_score(yy, pp) if len(set(yy)) > 1 else np.nan,
            "avg_precision": average_precision_score(yy, pp) if len(set(yy)) > 1 else np.nan,
            "brier": brier_score_loss(yy, pp),
            "base_rate": float(yy.mean()),
        })
    return pd.DataFrame(rows).set_index("model").sort_values("roc_auc", ascending=False)


__all__ = [
    "rmse", "mae", "bias", "r2", "directional_accuracy", "skill_vs_baseline",
    "regression_report", "classification_report_table", "out_of_fold_predictions",
]


def out_of_fold_predictions(models: dict, X, y, cv, feature_subsets: dict | None = None):
    """Fit every model on every training fold and predict its test fold.

    Returns `{name: array}` with NaN wherever a row was never in a test fold.
    Collecting predictions once, from one loop, is what guarantees that every
    later diagnostic scores exactly the same rows — the alternative, re-running
    the split per model, quietly compares models on different samples.

    `feature_subsets` maps a model name to the columns it should see; anything
    not listed gets all of `X`, which is what the baseline estimators need
    because they look their inputs up by name.

    `cv` may be a splitter object *or* a plain list of `(train_idx, test_idx)`
    pairs — the form `rvlab.pipelines.purged_date_splits` returns, and the form
    sklearn accepts anywhere a splitter is expected.
    """
    import numpy as np
    from sklearn.base import clone

    feature_subsets = feature_subsets or {}
    oof = {name: np.full(len(X), np.nan) for name in models}
    folds = cv.split(X) if hasattr(cv, "split") else cv

    for train_idx, test_idx in folds:
        for name, model in models.items():
            cols = feature_subsets.get(name)
            X_tr = X.iloc[train_idx][cols] if cols else X.iloc[train_idx]
            X_te = X.iloc[test_idx][cols] if cols else X.iloc[test_idx]
            oof[name][test_idx] = clone(model).fit(X_tr, y.iloc[train_idx]).predict(X_te)

    return oof
