"""
rvlab.models.volatility
=======================
Estimating and forecasting volatility — the general toolkit, of which rough
volatility (notebooks 03 and 04) is one corner.

Three families, and they answer different questions:

* **Realized estimators** measure what volatility *was*, from prices. The
  close-to-close estimator uses one number per day and throws away everything
  that happened in between; the range estimators (Parkinson, Garman-Klass,
  Rogers-Satchell, Yang-Zhang) use the high and low as well and are several times
  more efficient for it — at the cost of new assumptions, each of which the next
  estimator in the list relaxes.
* **Conditional models** — EWMA, HAR, GARCH — say what volatility *will be*, by
  exploiting the one robust fact about it: it clusters.
* **Implied** volatility says what the market *charges* for it. The gap between
  implied and subsequent realized is the variance risk premium, and it is the
  economic reason anyone sells options.

> ⚠️ **Two traps.** Range estimators are biased **low** when the bar is built
> from finitely many observations, because a bar cannot see the true continuous
> maximum — and they miss overnight gaps entirely, which is why Yang-Zhang exists.
> Both effects are demonstrated in notebook 18 rather than assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ── realized volatility estimators ─────────────────────────────────────────────
def close_to_close(close, annualize: int = TRADING_DAYS) -> float:
    """The textbook estimator: standard deviation of log returns.

    Unbiased and universally understood, and it discards the high and low
    entirely — roughly five times less efficient than the range estimators below
    for the same number of days.
    """
    c = np.asarray(close, float)
    r = np.diff(np.log(c[c > 0]))
    return float(np.std(r, ddof=1) * np.sqrt(annualize)) if r.size > 1 else np.nan


def parkinson(high, low, annualize: int = TRADING_DAYS) -> float:
    """Uses the high-low range only. About 5x more efficient than close-to-close.

    Assumes no drift and continuous observation. It therefore ignores overnight
    gaps completely, which on equities is a material share of total variance.
    """
    h, l = np.asarray(high, float), np.asarray(low, float)
    ok = (h > 0) & (l > 0)
    rng = np.log(h[ok] / l[ok]) ** 2
    return float(np.sqrt(np.mean(rng) / (4 * np.log(2)) * annualize))


def garman_klass(open_, high, low, close, annualize: int = TRADING_DAYS) -> float:
    """Adds the open and close to the range. More efficient than Parkinson.

    Still assumes zero drift and no overnight gap, so it inherits Parkinson's
    downward bias on gapping assets.
    """
    o, h, l, c = (np.asarray(x, float) for x in (open_, high, low, close))
    ok = (o > 0) & (h > 0) & (l > 0) & (c > 0)
    o, h, l, c = o[ok], h[ok], l[ok], c[ok]
    value = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    return float(np.sqrt(np.mean(value) * annualize))


def rogers_satchell(open_, high, low, close, annualize: int = TRADING_DAYS) -> float:
    """Drift-independent: correct even when the asset trends.

    The first estimator here that does not assume zero drift, which matters over
    long samples. Still blind to the overnight gap.
    """
    o, h, l, c = (np.asarray(x, float) for x in (open_, high, low, close))
    ok = (o > 0) & (h > 0) & (l > 0) & (c > 0)
    o, h, l, c = o[ok], h[ok], l[ok], c[ok]
    value = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    return float(np.sqrt(np.mean(value) * annualize))


def yang_zhang(open_, high, low, close, annualize: int = TRADING_DAYS) -> float:
    """Drift-independent **and** gap-aware — the most complete of the five.

    Combines overnight variance, open-to-close variance and Rogers-Satchell with
    a weight `k` chosen to minimise total variance. It is the only estimator here
    that can recover close-to-close volatility on an asset that gaps, which is
    every equity.
    """
    o, h, l, c = (np.asarray(x, float) for x in (open_, high, low, close))
    ok = (o > 0) & (h > 0) & (l > 0) & (c > 0)
    o, h, l, c = o[ok], h[ok], l[ok], c[ok]
    n = len(c)
    if n < 3:
        return np.nan

    prev_close = c[:-1]
    o, h, l, c = o[1:], h[1:], l[1:], c[1:]
    overnight = np.log(o / prev_close)
    open_to_close = np.log(c / o)
    rs = np.mean(np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o))

    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    total = overnight.var(ddof=1) + k * open_to_close.var(ddof=1) + (1 - k) * rs
    return float(np.sqrt(max(total, 0.0) * annualize))


ESTIMATORS = {
    "close_to_close": lambda d: close_to_close(d["Close"]),
    "parkinson": lambda d: parkinson(d["High"], d["Low"]),
    "garman_klass": lambda d: garman_klass(d["Open"], d["High"], d["Low"], d["Close"]),
    "rogers_satchell": lambda d: rogers_satchell(d["Open"], d["High"], d["Low"], d["Close"]),
    "yang_zhang": lambda d: yang_zhang(d["Open"], d["High"], d["Low"], d["Close"]),
}


def estimator_table(ohlc: pd.DataFrame, annualize: int = TRADING_DAYS) -> pd.DataFrame:
    """All five estimators on one OHLC frame, with each one's ratio to close-to-close.

    The ratio column is the diagnostic. On gapping data the three pure-range
    estimators come in low and Yang-Zhang does not — that ordering is the whole
    argument for using it.
    """
    values = {name: fn(ohlc) for name, fn in ESTIMATORS.items()}
    base = values["close_to_close"]
    return pd.DataFrame({
        "annualized_vol": pd.Series(values),
        "ratio_to_close_to_close": pd.Series({k: v / base for k, v in values.items()}),
    }).round(4)


def estimator_efficiency(ohlc: pd.DataFrame, n_days: int = 21, n_samples: int = 200,
                         seed: int = 42) -> pd.DataFrame:
    """Sampling variability of each estimator over repeated short windows.

    This is the claim people quote — "Parkinson is 5x more efficient" — and it can
    only be measured by *resampling the same process*, not by comparing across
    assets with genuinely different volatilities. Non-overlapping windows keep the
    samples independent.
    """
    rng = np.random.default_rng(seed)
    n = len(ohlc)
    if n < n_days * 4:
        return pd.DataFrame()

    starts = rng.choice(n - n_days, size=min(n_samples, n - n_days), replace=False)
    rows = {name: [] for name in ESTIMATORS}
    for start in starts:
        window = ohlc.iloc[start: start + n_days]
        for name, fn in ESTIMATORS.items():
            rows[name].append(fn(window))

    out = pd.DataFrame({
        "mean": {k: np.nanmean(v) for k, v in rows.items()},
        "sd": {k: np.nanstd(v, ddof=1) for k, v in rows.items()},
    })
    out["coefficient_of_variation"] = out["sd"] / out["mean"]
    base = out.loc["close_to_close", "coefficient_of_variation"]
    out["efficiency_vs_close_to_close"] = (base / out["coefficient_of_variation"]) ** 2
    return out.round(4)


# ── conditional volatility models ──────────────────────────────────────────────
def ewma_vol(returns, lam: float = 0.94, annualize: int = TRADING_DAYS) -> pd.Series:
    """RiskMetrics exponentially weighted volatility.

    One parameter, no fitting, and a surprisingly strong baseline — any GARCH you
    fit should be compared against it. `lam=0.94` is the RiskMetrics daily
    convention; its half-life is about 11 days.
    """
    r = pd.Series(returns).dropna()
    var = r.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var * annualize)


def har_rv(realized_variance, horizons=(1, 5, 22), fit_intercept: bool = True):
    """Heterogeneous AutoRegressive model (Corsi 2009) on realized variance.

    Regresses tomorrow's realized variance on its own averages over one day, one
    week and one month. A long-memory model built from three short-memory terms —
    and a strong benchmark that any more elaborate volatility model has to beat.

    Returns `(fitted_ols, design_frame)`; the design frame's columns are shifted
    so every regressor is known before the target it predicts.
    """
    import statsmodels.api as sm

    rv = pd.Series(realized_variance).dropna()
    design = pd.DataFrame(index=rv.index)
    names = {1: "daily", 5: "weekly", 22: "monthly"}
    for h in horizons:
        design[names.get(h, f"h{h}")] = rv.rolling(h, min_periods=1).mean().shift(1)
    design["target"] = rv

    frame = design.dropna()
    X = sm.add_constant(frame.drop(columns="target")) if fit_intercept \
        else frame.drop(columns="target")
    return sm.OLS(frame["target"], X).fit(), frame


@dataclass
class GarchFit:
    """Fitted GARCH(1,1). `persistence` = alpha + beta drives everything.

    `annualize` records the number of periods per year for the data that was
    fitted — 252 for daily bars, 252*390 for one-minute bars. It is stored rather
    than assumed because half-life and the long-run level are otherwise reported
    in the wrong units, and the error is invisible: a persistence of 0.996 on
    minute data is a half-life of 176 *bars*, which is under an hour, not 176 days.
    """

    omega: float
    alpha: float
    beta: float
    loglikelihood: float
    converged: bool
    conditional_vol: pd.Series
    annualize: int = TRADING_DAYS

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def half_life(self) -> float:
        """**Periods** for a shock to decay by half — bars, not days.

        Infinite as persistence approaches 1, which is why a fit on the constraint
        boundary is a warning rather than a result.
        """
        p = self.persistence
        return float(np.log(0.5) / np.log(p)) if 0 < p < 1 else np.inf

    @property
    def long_run_vol(self) -> float:
        """Unconditional volatility the process reverts to, annualised."""
        p = self.persistence
        return float(np.sqrt(self.omega / (1 - p) * self.annualize)) if p < 1 else np.nan

    def __str__(self) -> str:
        return (f"GARCH(1,1)  omega={self.omega:.3e}  alpha={self.alpha:.4f}  "
                f"beta={self.beta:.4f}\n  persistence {self.persistence:.4f} · "
                f"half-life {self.half_life:.1f} periods · "
                f"long-run vol {self.long_run_vol:.4f} (annualised by {self.annualize:,})")

    __repr__ = __str__


def garch11(returns, annualize: int = TRADING_DAYS) -> GarchFit:
    """Fit GARCH(1,1) by maximum likelihood.

        sigma2[t] = omega + alpha * eps[t-1]^2 + beta * sigma2[t-1]

    Written out rather than taken from a library, because the `arch` package is
    not always available and because the recursion is the part worth
    understanding: today's variance is a weighted average of a long-run level, the
    latest surprise, and yesterday's variance.

    Constrained to `alpha + beta < 1`, without which the process has no finite
    unconditional variance. Fitted volatilities near that boundary should be
    treated as a warning, not a result.
    """
    from scipy.optimize import minimize

    r = pd.Series(returns).dropna()
    eps = (r - r.mean()).to_numpy(float)
    n = len(eps)
    var0 = float(np.var(eps, ddof=1))

    def negative_loglik(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
            return 1e10
        sigma2 = np.empty(n)
        sigma2[0] = var0
        for t in range(1, n):
            sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        if not np.all(np.isfinite(sigma2)) or np.any(sigma2 <= 0):
            return 1e10
        return 0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + eps ** 2 / sigma2)

    start = [var0 * 0.05, 0.08, 0.88]
    result = minimize(negative_loglik, start, method="L-BFGS-B",
                      bounds=[(1e-12, None), (0.0, 0.999), (0.0, 0.999)])
    omega, alpha, beta = result.x

    sigma2 = np.empty(n)
    sigma2[0] = var0
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]

    return GarchFit(omega=float(omega), alpha=float(alpha), beta=float(beta),
                    loglikelihood=float(-result.fun), converged=bool(result.success),
                    conditional_vol=pd.Series(np.sqrt(sigma2 * annualize), index=r.index),
                    annualize=annualize)


def garch_forecast(fit: GarchFit, last_return: float, last_var: float,
                   horizon: int = 20, annualize: int | None = None) -> pd.Series:
    """Multi-step variance forecast, reverting toward the long-run level.

    The forecast decays geometrically at rate `persistence` — which is why the
    half-life is the number to report. A model with persistence 0.99 barely
    reverts within any horizon you care about.
    """
    annualize = fit.annualize if annualize is None else annualize
    long_run = fit.omega / (1 - fit.persistence) if fit.persistence < 1 else np.nan
    path = []
    var = fit.omega + fit.alpha * last_return ** 2 + fit.beta * last_var
    for _ in range(horizon):
        path.append(var)
        var = long_run + fit.persistence * (var - long_run)
    return pd.Series(np.sqrt(np.array(path) * annualize),
                     index=range(1, horizon + 1), name="forecast_vol")


# ── implied volatility and the premium ─────────────────────────────────────────
def vol_term_structure(panel: pd.DataFrame, ts_col: str = "ts", t_col: str = "T",
                       vol_col: str = "atm_iv") -> pd.DataFrame:
    """Average implied volatility by maturity — the term structure, pooled over time.

    Upward sloping in calm markets, inverted in stressed ones. The slope is a
    tradeable signal in its own right and a sanity check on any surface model.
    """
    out = (panel.groupby(pd.cut(panel[t_col] * 365, [0, 10, 21, 35, 60, 120, 365]),
                         observed=True)[vol_col]
           .agg(["mean", "std", "count"]).dropna())
    out.index.name = "days_to_expiry"
    return out.round(4)


def variance_risk_premium(implied_vol, realized_vol, annualized: bool = True) -> pd.Series:
    """Implied variance minus realized variance, in matching units.

    Positive on average — that premium is the compensation an option seller earns
    for bearing variance risk, and it is the economic reason the strategy side of
    this project sells options at all.

    Both inputs must be **volatilities** and annualised the same way. Mixing a
    total variance with an annualised one is the unit error the project README
    flags as limitation 8.5.
    """
    iv = pd.Series(implied_vol, dtype=float)
    rv = pd.Series(realized_vol, dtype=float)
    return iv ** 2 - rv ** 2


def volatility_regimes(vol_series, quantiles=(0.33, 0.67),
                       labels=("calm", "normal", "stressed")) -> pd.Series:
    """Label each observation by volatility tercile, using an expanding window.

    Expanding rather than full-sample: a regime label computed from the whole
    history knows the future, and conditioning any analysis on it leaks.
    """
    s = pd.Series(vol_series).dropna()
    lo = s.expanding(min_periods=60).quantile(quantiles[0])
    hi = s.expanding(min_periods=60).quantile(quantiles[1])
    out = pd.Series(labels[1], index=s.index, dtype=object)
    out[s <= lo] = labels[0]
    out[s >= hi] = labels[2]
    out[lo.isna()] = np.nan
    return out


__all__ = [
    "close_to_close", "parkinson", "garman_klass", "rogers_satchell", "yang_zhang",
    "ESTIMATORS", "estimator_table", "estimator_efficiency", "ewma_vol", "har_rv",
    "GarchFit", "garch11", "garch_forecast", "vol_term_structure",
    "variance_risk_premium", "volatility_regimes", "TRADING_DAYS",
]
