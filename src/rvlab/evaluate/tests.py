"""
rvlab.evaluate.tests
====================
Statistical tests for "is this difference real?".

Three tools, for three distinct failure modes:

* **Diebold-Mariano** — is model A's forecast error genuinely smaller than
  model B's, given that the two error series are serially correlated? A paired
  t-test on autocorrelated errors reports significance that is not there; DM
  uses a HAC variance and does not.

* **Block bootstrap** — a confidence interval for a statistic computed on
  dependent data. Resampling individual rows destroys the serial dependence and
  produces intervals that are far too narrow. Resampling *blocks* preserves it.

* **Benjamini-Hochberg** — when a sweep tests 30 cells and 4 "pass" at p < 0.05,
  roughly 1.5 of those are expected by chance alone. BH controls the expected
  proportion of false discoveries among the rejections.

The third is the one this project most needs: its own README records that the
H4 result selected feature, method, H, horizon and regime on a single sample,
and labels the finding hypothesis-generating for exactly this reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DMResult:
    """Diebold-Mariano outcome. `statistic` < 0 means model A has lower loss."""

    statistic: float
    p_value: float
    n_obs: int
    horizon: int
    loss: str

    @property
    def favours(self) -> str:
        if not np.isfinite(self.p_value) or self.p_value > 0.05:
            return "neither (not significant at 5%)"
        return "model A" if self.statistic < 0 else "model B"

    def __str__(self) -> str:
        return (f"DM = {self.statistic:+.3f}, p = {self.p_value:.4f} "
                f"({self.n_obs:,} obs, {self.loss} loss) -> favours {self.favours}")


def diebold_mariano(y_true, pred_a, pred_b, horizon: int = 1,
                    loss: str = "squared") -> DMResult:
    """Test equal predictive accuracy of two forecasts. Negative favours A.

    Uses the Newey-West HAC variance with lag `horizon - 1`, plus the
    Harvey-Leybourne-Newbold small-sample correction, which matters at the fold
    sizes used in this series.
    """
    a = np.asarray(y_true, float).ravel()
    fa = np.asarray(pred_a, float).ravel()
    fb = np.asarray(pred_b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(fa) & np.isfinite(fb)
    a, fa, fb = a[ok], fa[ok], fb[ok]

    ea, eb = a - fa, a - fb
    if loss == "squared":
        d = ea**2 - eb**2
    elif loss == "absolute":
        d = np.abs(ea) - np.abs(eb)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")

    n = len(d)
    if n < 8:
        return DMResult(np.nan, np.nan, n, horizon, loss)

    d_bar = d.mean()
    # Newey-West long-run variance
    gamma0 = np.sum((d - d_bar) ** 2) / n
    lrv = gamma0
    for lag in range(1, horizon):
        cov = np.sum((d[lag:] - d_bar) * (d[:-lag] - d_bar)) / n
        lrv += 2.0 * (1.0 - lag / horizon) * cov
    if lrv <= 0:
        return DMResult(np.nan, np.nan, n, horizon, loss)

    dm = d_bar / np.sqrt(lrv / n)
    # Harvey-Leybourne-Newbold small-sample correction
    k = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    dm *= k
    p = 2.0 * (1.0 - stats.t.cdf(abs(dm), df=n - 1))
    return DMResult(float(dm), float(p), n, horizon, loss)


def block_bootstrap_ci(values, statistic=np.mean, block_size: int | None = None,
                       n_boot: int = 1_000, alpha: float = 0.05,
                       seed: int = 42) -> dict:
    """Moving-block bootstrap CI for a statistic of a dependent series.

    `block_size` defaults to n^(1/3), the usual rule of thumb. Compare the width
    against an i.i.d. bootstrap (block_size=1) to see how much of your apparent
    precision was an artefact of ignoring autocorrelation.
    """
    x = np.asarray(values, float).ravel()
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return {"point": np.nan, "lo": np.nan, "hi": np.nan, "n": n,
                "block_size": block_size, "n_boot": n_boot}

    b = int(block_size or max(1, round(n ** (1 / 3))))
    n_blocks = int(np.ceil(n / b))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, max(1, n - b + 1), size=(n_boot, n_blocks))

    draws = np.empty(n_boot)
    for i in range(n_boot):
        sample = np.concatenate([x[s: s + b] for s in starts[i]])[:n]
        draws[i] = statistic(sample)

    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {"point": float(statistic(x)), "lo": float(lo), "hi": float(hi),
            "n": n, "block_size": b, "n_boot": n_boot,
            "excludes_zero": bool(lo > 0 or hi < 0)}


def benjamini_hochberg(p_values, alpha: float = 0.05) -> pd.DataFrame:
    """Control the false discovery rate across a family of tests.

    Returns a frame with the original p-values, their rank, the BH threshold and
    a `reject` flag. Use it on every sweep: 30 cells tested at 5% will produce
    about 1.5 spurious passes even when nothing is real.
    """
    p = np.asarray(p_values, float).ravel()
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = alpha * np.arange(1, n + 1) / n

    below = ranked <= thresholds
    k = np.max(np.nonzero(below)[0]) + 1 if below.any() else 0
    reject_sorted = np.zeros(n, dtype=bool)
    reject_sorted[:k] = True

    reject = np.empty(n, dtype=bool)
    reject[order] = reject_sorted
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(1, n + 1)

    return pd.DataFrame({
        "p_value": p, "rank": rank,
        "bh_threshold": alpha * rank / n, "reject": reject,
    }).sort_values("p_value")


def bonferroni(p_values, alpha: float = 0.05) -> pd.DataFrame:
    """The conservative alternative to BH — controls family-wise error instead.

    Shown alongside BH in notebook 09 so the cost of the stricter criterion is
    visible rather than asserted.
    """
    p = np.asarray(p_values, float).ravel()
    return pd.DataFrame({"p_value": p, "threshold": alpha / len(p),
                         "reject": p <= alpha / len(p)}).sort_values("p_value")


def paired_permutation_test(y_true, pred_a, pred_b, n_perm: int = 2_000,
                            seed: int = 42, loss=lambda e: e**2) -> dict:
    """Distribution-free alternative to DM: permute the sign of the loss difference.

    Makes no assumption about the error distribution, but does assume the loss
    differences are exchangeable — which serial correlation violates. Use it as
    a cross-check on DM, not a replacement.
    """
    a = np.asarray(y_true, float).ravel()
    fa, fb = np.asarray(pred_a, float).ravel(), np.asarray(pred_b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(fa) & np.isfinite(fb)
    d = loss(a[ok] - fa[ok]) - loss(a[ok] - fb[ok])

    observed = float(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = (signs * d).mean(axis=1)
    p = float((np.abs(null) >= abs(observed)).mean())
    return {"mean_loss_diff": observed, "p_value": p, "n": len(d), "n_perm": n_perm}


__all__ = [
    "DMResult", "diebold_mariano", "block_bootstrap_ci", "benjamini_hochberg",
    "bonferroni", "paired_permutation_test",
]
