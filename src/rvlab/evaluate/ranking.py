"""
rvlab.evaluate.ranking
======================
Scoring a cross-sectional ranking model.

Most time-series competitions do not ask for an accurate forecast — they ask for
a good *ordering*, each day, and score a portfolio built from the extremes. That
changes what a metric should measure:

  * **RMSE is nearly irrelevant.** Adding a constant to every prediction on a
    date leaves the ranking — and the portfolio — untouched while moving RMSE a
    lot.
  * **Rank IC** (daily Spearman correlation between prediction and outcome) is
    the natural accuracy measure. Its mean says whether the ordering is right;
    the ratio of its mean to its standard deviation (the *ICIR*) says whether it
    is reliably right, which matters more.
  * **Top-k spread return** is what the portfolio actually earns: go long the k
    best-ranked, short the k worst, and take the difference. Its Sharpe ratio is
    the JPX competition's official score and a good general summary.

A model can have a positive mean rank IC and a negative spread Sharpe — that
happens when it orders the middle of the cross-section well and the extremes
badly, and only the extremes are traded. Report both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_prediction_rank(df: pd.DataFrame, prediction_col: str = "Prediction",
                        date_col: str = "Date", rank_col: str = "Rank") -> pd.DataFrame:
    """Dense 0-based rank within each date, 0 = most attractive.

    `method="first"` breaks ties by row order, which guarantees a *unique* rank
    per entity — competitions usually require that, and an average-rank tie
    produces non-integer ranks that fail submission validation.
    """
    out = df.copy()
    out[rank_col] = (out.groupby(date_col, observed=True)[prediction_col]
                     .rank(ascending=False, method="first").astype(int) - 1)
    return out


def daily_rank_ic(df: pd.DataFrame, target_col: str = "Target",
                  prediction_col: str = "Prediction", date_col: str = "Date",
                  method: str = "spearman") -> pd.Series:
    """Per-date rank correlation between prediction and outcome.

    Spearman rather than Pearson: the objective is an ordering, and returns have
    tails heavy enough that a handful of names would otherwise dominate.
    """
    def _ic(group):
        if group[target_col].nunique() < 2 or group[prediction_col].nunique() < 2:
            return np.nan
        return group[target_col].corr(group[prediction_col], method=method)

    return (df.groupby(date_col, observed=True)[[target_col, prediction_col]]
            .apply(_ic).rename("rank_ic").dropna())


def information_ratio(ic: pd.Series, periods_per_year: int = 252) -> dict:
    """Mean IC, its volatility, the ICIR, and a t-statistic.

    ICIR = mean(IC) / std(IC). It is the Sharpe ratio of the signal's accuracy,
    and it is the number that distinguishes a signal you can size from one that
    happens to average positive.
    """
    ic = pd.Series(ic).dropna()
    if len(ic) < 2 or ic.std(ddof=1) == 0:
        return {"mean_ic": np.nan, "ic_std": np.nan, "icir": np.nan,
                "t_stat": np.nan, "hit_rate": np.nan, "n_days": len(ic)}
    icir = float(ic.mean() / ic.std(ddof=1))
    return {
        "mean_ic": float(ic.mean()),
        "ic_std": float(ic.std(ddof=1)),
        "icir": icir,
        "t_stat": icir * np.sqrt(len(ic)),
        "hit_rate": float((ic > 0).mean()),
        "n_days": len(ic),
    }


def spread_return_one_day(day: pd.DataFrame, target_col: str = "Target",
                          rank_col: str = "Rank", portfolio_size: int = 200,
                          toprank_weight_ratio: float = 2.0) -> float:
    """Long the top k, short the bottom k, linearly weighted by rank.

    This is the JPX formula. Weights run from `toprank_weight_ratio` down to 1.0,
    so the best-ranked name carries twice the weight of the k-th. On the short
    leg the *worst*-ranked name gets the largest weight, which is why the short
    slice is reversed before weighting — getting that backwards silently halves
    the measured edge.
    """
    d = day.sort_values(rank_col)
    k = min(portfolio_size, len(d) // 2)
    if k < 1:
        return np.nan

    weights = np.linspace(toprank_weight_ratio, 1.0, k)
    long_leg = d[target_col].to_numpy()[:k]
    short_leg = d[target_col].to_numpy()[-k:][::-1]

    purchase = np.nansum(long_leg * weights) / weights.mean()
    short = np.nansum(short_leg * weights) / weights.mean()
    return float(purchase - short)


def spread_return_series(ranked: pd.DataFrame, target_col: str = "Target",
                         rank_col: str = "Rank", date_col: str = "Date",
                         portfolio_size: int = 200,
                         toprank_weight_ratio: float = 2.0) -> pd.Series:
    """Daily long-short spread return. The portfolio's actual P&L path."""
    values = {
        date: spread_return_one_day(group, target_col, rank_col,
                                    portfolio_size, toprank_weight_ratio)
        for date, group in ranked.groupby(date_col, observed=True, sort=True)
    }
    return pd.Series(values, name="spread_return").dropna()


def spread_sharpe(ranked: pd.DataFrame, **kwargs) -> float:
    """Mean over standard deviation of the daily spread return. The JPX score.

    Not annualised — the competition does not annualise it, and multiplying by
    sqrt(252) does not change any ranking of models.
    """
    r = spread_return_series(ranked, **kwargs)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return np.nan
    return float(r.mean() / r.std(ddof=1))


def ranking_report(eval_df: pd.DataFrame, predictions: dict,
                   target_col: str = "Target", date_col: str = "Date",
                   portfolio_size: int = 200) -> pd.DataFrame:
    """One row per model: RMSE, mean rank IC, ICIR, hit rate and spread Sharpe.

    RMSE is included precisely so you can watch it disagree with the ranking
    metrics. Cross-sectional de-meaning of the target usually makes RMSE worse
    and the ranking better; if you select on RMSE you will discard the model you
    wanted.
    """
    from .metrics import rmse

    rows = []
    for name, pred in predictions.items():
        tmp = eval_df[[date_col, target_col]].copy()
        tmp["Prediction"] = np.asarray(pred)
        tmp = tmp.dropna(subset=[target_col, "Prediction"])
        ranked = add_prediction_rank(tmp, date_col=date_col)

        stats = information_ratio(daily_rank_ic(ranked, target_col, "Prediction", date_col))
        rows.append({
            "model": name,
            "rmse": rmse(ranked[target_col], ranked["Prediction"]),
            "mean_ic": stats["mean_ic"],
            "icir": stats["icir"],
            "ic_hit_rate": stats["hit_rate"],
            "spread_sharpe": spread_sharpe(ranked, target_col=target_col,
                                           date_col=date_col,
                                           portfolio_size=portfolio_size),
        })
    return pd.DataFrame(rows).set_index("model").sort_values("spread_sharpe",
                                                             ascending=False)


def cross_sectional_zscore(predictions, dates) -> np.ndarray:
    """Standardise predictions within each date. Use before blending models.

    Ridge and a gradient-booster produce predictions on different scales, so a
    plain average is dominated by whichever has the larger spread. Z-scoring
    per date makes the average a genuine consensus — and, because ranking is
    invariant to a per-date affine transform, it costs nothing.
    """
    tmp = pd.DataFrame({"date": np.asarray(dates), "p": np.asarray(predictions, float)})
    grouped = tmp.groupby("date", observed=True)["p"]
    std = grouped.transform("std").replace(0, np.nan)
    return ((tmp["p"] - grouped.transform("mean")) / std).fillna(0.0).to_numpy()


def blend(predictions: dict, dates, weights: dict | None = None) -> np.ndarray:
    """Weighted average of per-date z-scored predictions."""
    weights = weights or {name: 1.0 for name in predictions}
    total = sum(weights[name] for name in predictions)
    return sum(weights[name] * cross_sectional_zscore(pred, dates)
               for name, pred in predictions.items()) / total


__all__ = [
    "add_prediction_rank", "daily_rank_ic", "information_ratio",
    "spread_return_one_day", "spread_return_series", "spread_sharpe",
    "ranking_report", "cross_sectional_zscore", "blend",
]
