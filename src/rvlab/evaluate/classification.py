"""
rvlab.evaluate.classification
=============================
Scoring a classifier when the classes are unbalanced and the errors cost
different amounts — which is the normal case, not the exception.

Three things a default workflow gets wrong:

* **Accuracy on imbalanced data is meaningless.** At a 5% positive rate, "always
  say no" scores 95%. Report balanced accuracy, ROC-AUC and average precision,
  and note that the last of those has a baseline equal to the positive rate.
* **0.5 is not a threshold, it is a default.** The right cut depends on what a
  false positive costs relative to a false negative, and that ratio is a business
  input rather than a statistical one.
* **Ranking well and being calibrated are different.** A model can order perfectly
  and still be systematically overconfident. If you act on the probability rather
  than the ordering, calibration is the property that matters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def imbalance_report(y_true) -> pd.Series:
    """Class balance, and what a trivial classifier would score.

    The `majority_accuracy` row is the point: any accuracy at or below it is
    evidence of nothing at all.

    Every value is numeric so the result can be `.round()`ed and concatenated —
    a report that mixes a string note into a float Series becomes object dtype and
    stops behaving like data.
    """
    y = pd.Series(np.asarray(y_true).ravel())
    counts = y.value_counts(normalize=True).sort_index()
    majority = float(counts.max())
    return pd.Series({
        "n": len(y),
        "n_classes": int(y.nunique()),
        "positive_rate": float(counts.get(1, counts.iloc[-1])),
        "minority_share": float(counts.min()),
        "imbalance_ratio": float(counts.max() / counts.min()) if counts.min() else np.inf,
        "majority_accuracy": majority,
    })


def threshold_sweep(y_true, proba, thresholds=None,
                    cost_false_positive: float = 1.0,
                    cost_false_negative: float = 1.0) -> pd.DataFrame:
    """Precision, recall, F1 and expected cost across candidate thresholds.

    The cost columns are what turn threshold selection from a taste question into
    an arithmetic one. If a missed positive costs ten times a false alarm, pass
    `cost_false_negative=10` and read off the minimum — it will not be 0.5.
    """
    from sklearn.metrics import confusion_matrix

    y = np.asarray(y_true).ravel().astype(int)
    p = np.asarray(proba, dtype=float).ravel()
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    grid = np.asarray(thresholds if thresholds is not None else np.linspace(0.02, 0.98, 49))

    rows = []
    for t in grid:
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({
            "threshold": float(t), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision": precision, "recall": recall, "f1": f1,
            "predicted_positive_rate": float(pred.mean()),
            "expected_cost": float(fp * cost_false_positive + fn * cost_false_negative),
        })
    return pd.DataFrame(rows)


def best_threshold(y_true, proba, objective: str = "f1",
                   cost_false_positive: float = 1.0,
                   cost_false_negative: float = 1.0) -> dict:
    """The threshold optimising `f1`, `expected_cost`, or `balanced` accuracy.

    Choose it on a **validation** split, not the test set — a threshold picked on
    the data you report is one more parameter fitted to it, and it flatters every
    metric downstream.
    """
    sweep = threshold_sweep(y_true, proba, cost_false_positive=cost_false_positive,
                            cost_false_negative=cost_false_negative)
    if objective == "expected_cost":
        row = sweep.loc[sweep["expected_cost"].idxmin()]
    elif objective == "balanced":
        specificity = sweep["tn"] / (sweep["tn"] + sweep["fp"]).replace(0, np.nan)
        row = sweep.loc[((sweep["recall"] + specificity) / 2).idxmax()]
    else:
        row = sweep.loc[sweep["f1"].idxmax()]
    return {"objective": objective, **row.to_dict()}


def calibration_report(y_true, proba, n_bins: int = 10) -> pd.Series:
    """Brier score, expected calibration error, and the calibration slope.

    **Slope < 1 means overconfident** — predictions too far from the base rate —
    which is the usual direction for a boosted tree. A slope near 1 with a
    non-zero intercept means a systematic shift instead. ROC-AUC detects neither.

    All values are numeric, so the result rounds and concatenates cleanly.
    """
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss, roc_auc_score

    y = np.asarray(y_true).ravel().astype(int)
    p = np.clip(np.asarray(proba, dtype=float).ravel(), 1e-6, 1 - 1e-6)
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]

    frac, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    weights = np.histogram(p, bins=np.r_[0, np.quantile(p, np.linspace(0, 1, n_bins + 1))[1:]])[0]
    weights = weights[:len(frac)] / max(weights[:len(frac)].sum(), 1)
    ece = float(np.sum(weights * np.abs(frac - mean_pred)))

    # Cox calibration slope: regress the outcome on the predicted log-odds with a
    # *logistic* link. A linear fit here returns probability-per-logit units, which
    # are never near 1 and so cannot be read against the perfect-calibration value.
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    if len(set(y)) > 1:
        from sklearn.linear_model import LogisticRegression
        # C=1e12 rather than np.inf: sklearn maps inf to penalty=None and warns.
        cox = LogisticRegression(C=1e12, max_iter=1000).fit(logit, y)
        slope, intercept = float(cox.coef_[0, 0]), float(cox.intercept_[0])
    else:
        slope, intercept = np.nan, np.nan

    return pd.Series({
        "brier": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else np.nan,
        "expected_calibration_error": ece,
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "mean_predicted": float(p.mean()),
        "observed_rate": float(y.mean()),
    })


def decile_table(y_true, score, n_bins: int = 10) -> pd.DataFrame:
    """Outcome rate and lift by predicted decile.

    Lift is the decile's outcome rate divided by the overall rate — how many times
    better than random that slice is. It is the form a non-technical reader
    understands immediately, and a monotone lift column is the most convincing
    evidence a ranking model can offer.
    """
    frame = pd.DataFrame({"y": np.asarray(y_true, dtype=float).ravel(),
                          "s": np.asarray(score, dtype=float).ravel()}).dropna()
    frame["decile"] = pd.qcut(frame["s"].rank(method="first"), n_bins,
                              labels=range(1, n_bins + 1))
    overall = frame["y"].mean()
    out = (frame.groupby("decile", observed=True)["y"]
           .agg(n="size", outcome_rate="mean").reset_index())
    out["lift"] = out["outcome_rate"] / overall if overall else np.nan
    out["cumulative_capture"] = (out.sort_values("decile", ascending=False)["outcome_rate"]
                                 .mul(out["n"]).cumsum().sort_index()
                                 / (frame["y"].sum() or 1))
    return out


__all__ = ["imbalance_report", "threshold_sweep", "best_threshold",
           "calibration_report", "decile_table"]
