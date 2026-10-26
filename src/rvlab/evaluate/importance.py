"""
rvlab.evaluate.importance
=========================
Getting interpretable names out of a fitted `Pipeline`.

After a `ColumnTransformer` with one-hot encoding, the model sees a few thousand
anonymous columns. `get_feature_names_out()` recovers the names — provided every
step supports it, which is the practical reason to prefer sklearn transformers
over hand-rolled numpy in a pipeline.

Two caveats that decide how much these numbers are worth:

  * **Tree gain importance is biased** toward high-cardinality features. A
    one-hot entity id with 2,000 levels will look important because it offers
    many places to split, not because it predicts.
  * **Linear coefficients are only comparable after scaling**, and collinear
    features split credit arbitrarily between themselves.

Permutation importance on held-out data (`sklearn.inspection`) avoids both and
is the one to trust when they disagree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pipeline_feature_names(pipeline, n_features: int | None = None) -> np.ndarray:
    """Feature names after preprocessing, with a positional fallback."""
    for step in ("prep", "preprocessor", "columntransformer"):
        if hasattr(pipeline, "named_steps") and step in pipeline.named_steps:
            try:
                return np.asarray(pipeline.named_steps[step].get_feature_names_out())
            except Exception:
                break
    try:
        return np.asarray(pipeline[:-1].get_feature_names_out())
    except Exception:
        n = n_features or 0
        return np.asarray([f"feature_{i}" for i in range(n)])


def pipeline_feature_importance(pipeline, top_n: int = 25,
                                normalise: bool = True) -> pd.DataFrame:
    """Rank a fitted pipeline's features by gain importance or |coefficient|.

    Returns a frame with `feature`, `importance` and `kind`, where `kind`
    records which of the two it actually is — they are not comparable, and
    labelling the column generically is how a plot ends up mislabelled.
    """
    model = pipeline.named_steps["model"] if hasattr(pipeline, "named_steps") else pipeline

    if hasattr(model, "feature_importances_"):
        values, kind = np.asarray(model.feature_importances_, float), "gain"
    elif hasattr(model, "coef_"):
        values, kind = np.abs(np.ravel(model.coef_)), "abs_coefficient"
    else:
        raise AttributeError(
            f"{type(model).__name__} exposes neither feature_importances_ nor coef_; "
            "use sklearn.inspection.permutation_importance instead")

    names = pipeline_feature_names(pipeline, n_features=len(values))
    if len(names) != len(values):
        names = np.asarray([f"feature_{i}" for i in range(len(values))])

    out = pd.DataFrame({"feature": names, "importance": values, "kind": kind})
    if normalise and out["importance"].sum() > 0:
        out["importance"] = out["importance"] / out["importance"].sum()
    return out.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def group_importance(importance: pd.DataFrame, prefixes: dict) -> pd.DataFrame:
    """Aggregate one-hot expanded importances back to their source columns.

    A categorical with 2,000 levels contributes 2,000 rows of tiny importance
    that individually look negligible and jointly may dominate. Summing them by
    prefix is the only way to compare a categorical against a numeric feature
    fairly.
    """
    rows = []
    for label, prefix in prefixes.items():
        mask = importance["feature"].str.contains(prefix, regex=False)
        rows.append({"group": label, "importance": float(importance.loc[mask, "importance"].sum()),
                     "n_features": int(mask.sum())})
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


__all__ = ["pipeline_feature_names", "pipeline_feature_importance", "group_importance"]
