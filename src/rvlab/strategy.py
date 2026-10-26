"""
rvlab.strategy
==============
Turning a signal into a position, and finding out what it costs.

A forecast is not a strategy. Between the two sit four decisions that usually
matter more than the forecast itself:

  1. **Position construction** — how a score becomes a size. Cross-sectional
     ranking is scale-free and bounded; raw z-scores are neither.
  2. **Risk scaling** — a strategy whose volatility wanders is unrunnable
     regardless of its Sharpe, and volatility targeting is the standard fix.
  3. **Costs** — the difference between gross and net P&L, and the reason a
     signal that rebalances daily can be worse than one that does not.
  4. **Evaluation** — Sharpe alone hides the path, and the path is what decides
     whether anyone can hold the position through a bad year.

> ⚠️ Everything here is applied to a signal that is already causal. If the signal
> saw the future, none of the machinery below will tell you — it will simply
> report an excellent Sharpe. That check belongs upstream, in
> `rvlab.features.timeseries` and the purged splitters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ── 1. signal to position ──────────────────────────────────────────────────────
def signal_to_position(signal, dates=None, method: str = "zscore",
                       cap: float | None = 3.0, gross_target: float | None = None
                       ) -> pd.Series:
    """Convert a raw score into a position.

    `method`:
      * ``"zscore"`` — standardise, then cap. Keeps relative magnitude; sensitive
        to outliers, hence the cap.
      * ``"rank"`` — map to [-1, 1] by rank. Scale-free, bounded, and immune to a
        single extreme score. The default choice for a cross-sectional book.
      * ``"sign"`` — equal-weight long/short. Throws away conviction and is
        surprisingly hard to beat.

    Pass `dates` to work **within each date** (a cross-sectional book); leave it
    None for a single time series. Getting this wrong is the most common error
    here: a time-series z-score applied to a panel standardises across entities
    *and* time at once, which is neither.

    `gross_target` rescales so the absolute positions sum to a fixed gross
    exposure per date, which is what makes P&L comparable across dates when the
    universe size changes.
    """
    s = pd.Series(signal, dtype=float)
    grouper = pd.Series(np.asarray(dates)).to_numpy() if dates is not None else None

    def _one(x: pd.Series) -> pd.Series:
        x = x.astype(float)
        if method == "sign":
            return np.sign(x - x.median())
        if method == "rank":
            r = x.rank(pct=True)
            return (r - 0.5) * 2.0
        sd = x.std(ddof=0)
        z = (x - x.mean()) / sd if sd and np.isfinite(sd) else x * 0.0
        return z.clip(-cap, cap) if cap else z

    out = s.groupby(grouper, sort=False).transform(_one) if grouper is not None else _one(s)
    out = out.fillna(0.0)

    if gross_target:
        gross = (out.abs().groupby(grouper).transform("sum") if grouper is not None
                 else out.abs().sum())
        out = out * gross_target / gross.replace(0, np.nan)
    return out.fillna(0.0)


def vol_target(positions, returns, target_vol: float = 0.10, lookback: int = 60,
               max_leverage: float = 3.0, annualize: int = TRADING_DAYS) -> pd.Series:
    """Scale positions so realised volatility tracks `target_vol`.

    The scaling uses a **trailing** estimate of the strategy's own volatility,
    shifted by one period so today's scale does not use today's return. Without
    that shift the target is met exactly and the backtest is fiction.

    `max_leverage` matters more than it looks: as the volatility estimate falls,
    the implied leverage rises without bound, and an uncapped vol target
    concentrates its worst losses in exactly the calm periods that preceded a
    shock.
    """
    pos = pd.Series(positions, dtype=float)
    ret = pd.Series(returns, dtype=float).reindex(pos.index)

    gross_pnl = pos.shift(1) * ret
    realised = gross_pnl.rolling(lookback, min_periods=lookback // 2).std() * np.sqrt(annualize)
    scale = (target_vol / realised.shift(1)).clip(upper=max_leverage)
    return (pos * scale).fillna(0.0)


# ── 2. costs and P&L ───────────────────────────────────────────────────────────
def apply_costs(positions, cost_bps: float = 1.0, fixed_per_trade: float = 0.0,
                dates=None, tolerance: float = 0.0) -> pd.DataFrame:
    """Turnover and transaction cost implied by a position path.

    Cost is charged on the **change** in position, not the level — holding is
    free, trading is not. That is why a signal with a high hit rate and a high
    turnover can be worse after costs than a weaker, stickier one.

    Returns a frame with `turnover` and `cost` aligned to `positions`.
    """
    pos = pd.Series(positions, dtype=float)
    if dates is not None:
        grouper = pd.Series(np.asarray(dates)).to_numpy()
        prev = pos.groupby(grouper, sort=False).shift(1)
    else:
        prev = pos.shift(1)

    turnover = (pos - prev.fillna(0.0)).abs()
    traded = turnover > tolerance
    cost = turnover * cost_bps / 1e4 + traded.astype(float) * fixed_per_trade
    return pd.DataFrame({"turnover": turnover, "cost": cost})


def pnl_series(positions, returns, cost_bps: float = 0.0, dates=None,
               fixed_per_trade: float = 0.0) -> pd.DataFrame:
    """Gross P&L, cost and net P&L for a position path.

    Positions are lagged by one period: the position you hold *into* a return is
    the one decided before it. Forgetting that lag is the single most common way
    a backtest reports impossible performance.

    With `dates`, P&L is aggregated to one row per date — the portfolio's return,
    not each name's.
    """
    pos = pd.Series(positions, dtype=float)
    ret = pd.Series(returns, dtype=float).reindex(pos.index)

    if dates is not None:
        grouper = pd.Series(np.asarray(dates)).to_numpy()
        lagged = pos.groupby(grouper, sort=False).shift(0)      # panel: already t-1 signal
        gross = lagged * ret
        costs = apply_costs(pos, cost_bps, fixed_per_trade, dates=grouper)
        frame = pd.DataFrame({"gross": gross, "cost": costs["cost"],
                              "turnover": costs["turnover"]})
        out = frame.groupby(grouper, sort=True).sum()
        out.index.name = "date"
    else:
        gross = pos.shift(1) * ret
        costs = apply_costs(pos, cost_bps, fixed_per_trade)
        out = pd.DataFrame({"gross": gross, "cost": costs["cost"],
                            "turnover": costs["turnover"]}).dropna()

    out["net"] = out["gross"] - out["cost"]
    return out


# ── 3. evaluation ──────────────────────────────────────────────────────────────
@dataclass
class StrategyMetrics:
    """Summary of a P&L series. Read `sharpe` and `max_drawdown` together."""

    values: pd.Series

    def to_series(self) -> pd.Series:
        return self.values

    def __str__(self) -> str:
        return self.values.to_string()

    __repr__ = __str__


def strategy_metrics(pnl, annualize: int = TRADING_DAYS, cost: pd.Series | None = None,
                     turnover: pd.Series | None = None) -> pd.Series:
    """Sharpe, Sortino, Calmar, drawdown, hit rate and turnover, in one place.

    Three ratios because they penalise different things: **Sharpe** treats upside
    and downside volatility alike, **Sortino** only counts the downside, and
    **Calmar** divides return by the worst drawdown, which is what actually
    determines whether a position survives a committee.

    A large Sharpe-Sortino gap means the volatility is mostly upside — good news
    that Sharpe understates. A large Sharpe-Calmar gap means one long, deep
    drawdown, which Sharpe hides entirely.
    """
    r = pd.Series(pnl, dtype=float).dropna()
    if len(r) < 3:
        return pd.Series(dtype=float)

    mean, sd = r.mean(), r.std(ddof=1)
    downside = r[r < 0].std(ddof=1)
    cum = r.cumsum()
    drawdown = cum - cum.cummax()
    max_dd = float(-drawdown.min())

    ann_return = mean * annualize
    out = {
        "n_periods": len(r),
        "mean_return": float(mean),
        "annualised_return": float(ann_return),
        "annualised_vol": float(sd * np.sqrt(annualize)) if sd else np.nan,
        "sharpe": float(mean / sd * np.sqrt(annualize)) if sd else np.nan,
        "sortino": float(mean / downside * np.sqrt(annualize)) if downside else np.nan,
        "max_drawdown": max_dd,
        "calmar": float(ann_return / max_dd) if max_dd > 0 else np.nan,
        "hit_rate": float((r > 0).mean()),
        "best": float(r.max()),
        "worst": float(r.min()),
        "skew": float(r.skew()),
        "pct_underwater": float((drawdown < 0).mean()),
    }
    if turnover is not None:
        t = pd.Series(turnover).reindex(r.index).dropna()
        out["mean_turnover"] = float(t.mean())
    if cost is not None:
        c = pd.Series(cost).reindex(r.index).dropna()
        out["total_cost"] = float(c.sum())
        out["cost_share_of_gross"] = (float(c.sum() / (r.sum() + c.sum()))
                                      if (r.sum() + c.sum()) else np.nan)
    return pd.Series(out)


def cost_sweep(positions, returns, cost_levels=(0.0, 0.5, 1.0, 2.0, 5.0, 10.0),
               dates=None, annualize: int = TRADING_DAYS) -> pd.DataFrame:
    """Sharpe against transaction cost — the curve, not a point.

    A single cost assumption silently picks the winner. The number that matters
    is the **break-even cost**: the level at which the strategy stops making
    money. If it is below what you actually pay, there is no strategy.
    """
    rows = []
    for bps in cost_levels:
        pnl = pnl_series(positions, returns, cost_bps=bps, dates=dates)
        metrics = strategy_metrics(pnl["net"], annualize=annualize,
                                   turnover=pnl["turnover"])
        rows.append({"cost_bps": bps, "sharpe": metrics.get("sharpe", np.nan),
                     "annualised_return": metrics.get("annualised_return", np.nan),
                     "mean_turnover": metrics.get("mean_turnover", np.nan)})
    return pd.DataFrame(rows)


def break_even_cost(sweep: pd.DataFrame, metric: str = "annualised_return") -> float:
    """Interpolate the cost level at which the strategy stops making money."""
    s = sweep.dropna(subset=[metric])
    positive = s[s[metric] > 0]
    if positive.empty:
        return 0.0
    if (s[metric] > 0).all():
        return float(np.inf)
    last_good = positive["cost_bps"].max()
    first_bad = s[s["cost_bps"] > last_good]["cost_bps"].min()
    y0 = float(s.loc[s["cost_bps"] == last_good, metric].iloc[0])
    y1 = float(s.loc[s["cost_bps"] == first_bad, metric].iloc[0])
    return float(last_good + (first_bad - last_good) * y0 / (y0 - y1))


def walk_forward_strategy(frame: pd.DataFrame, signal_fn, return_col: str,
                          date_col: str = "date", train_periods: int = 250,
                          test_periods: int = 50, cost_bps: float = 1.0) -> pd.DataFrame:
    """Refit the signal on a rolling window and trade the block that follows.

    The evaluation closest to how a strategy is actually run. `signal_fn` receives
    the training slice and the test slice and returns positions for the test slice
    only — so it cannot see beyond its own window by construction.
    """
    dates = np.sort(frame[date_col].unique())
    rows = []
    start = 0
    while start + train_periods + test_periods <= len(dates):
        train_dates = dates[start: start + train_periods]
        test_dates = dates[start + train_periods: start + train_periods + test_periods]
        train = frame[frame[date_col].isin(train_dates)]
        test = frame[frame[date_col].isin(test_dates)]

        positions = pd.Series(signal_fn(train, test), index=test.index).fillna(0.0)
        pnl = pnl_series(positions, test[return_col], cost_bps=cost_bps,
                         dates=test[date_col])
        pnl["fold_start"] = pd.Timestamp(test_dates[0])
        rows.append(pnl)
        start += test_periods

    return pd.concat(rows) if rows else pd.DataFrame()


def tearsheet(pnl: pd.DataFrame, title: str = "", annualize: int = TRADING_DAYS,
              cost_col: str = "cost"):
    """Four panels: cumulative net P&L with drawdown, gross vs net, and turnover.

    The gross-versus-net panel is the one that changes decisions. A strategy whose
    two curves diverge steadily is being eaten by turnover, and no amount of
    signal improvement fixes that — only trading less does.
    """
    import matplotlib.pyplot as plt

    from .plotting.style import C_BASELINE, C_MODEL, C_WARN, finish

    net = pnl["net"].dropna()
    cum_net = net.cumsum()
    cum_gross = pnl["gross"].dropna().cumsum()
    drawdown = cum_net - cum_net.cummax()

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 6.6))
    axes[0, 0].plot(cum_net.index, cum_net.to_numpy(), color=C_MODEL, lw=1.8)
    axes[0, 0].axhline(0, color="#4a5568", lw=1)
    metrics = strategy_metrics(net, annualize=annualize)
    finish(axes[0, 0], "Cumulative net P&L", "", "cumulative",
           f"Sharpe {metrics['sharpe']:.2f} · Calmar {metrics['calmar']:.2f}")

    axes[0, 1].fill_between(drawdown.index, drawdown.to_numpy(), 0, color=C_WARN, alpha=0.35)
    finish(axes[0, 1], "Drawdown", "", "",
           f"max {metrics['max_drawdown']:.4f} · "
           f"{metrics['pct_underwater']:.0%} of periods underwater")

    axes[1, 0].plot(cum_gross.index, cum_gross.to_numpy(), color=C_BASELINE, lw=1.6,
                    label="gross")
    axes[1, 0].plot(cum_net.index, cum_net.to_numpy(), color=C_MODEL, lw=1.8, label="net")
    axes[1, 0].legend()
    eaten = float(cum_gross.iloc[-1] - cum_net.iloc[-1]) if len(cum_gross) else np.nan
    finish(axes[1, 0], "Gross against net", "", "cumulative",
           f"costs consumed {eaten:.4f}")

    if cost_col in pnl:
        axes[1, 1].plot(pnl.index, pnl["turnover"].to_numpy(), color=C_BASELINE, lw=0.8)
        finish(axes[1, 1], "Turnover", "", "per period",
               f"mean {pnl['turnover'].mean():.3f}")
    fig.suptitle(title or "Strategy tearsheet", x=0.005, ha="left",
                 fontsize=12, fontweight="600")
    fig.tight_layout()
    return fig, axes


__all__ = [
    "signal_to_position", "vol_target", "apply_costs", "pnl_series",
    "strategy_metrics", "StrategyMetrics", "cost_sweep", "break_even_cost",
    "walk_forward_strategy", "tearsheet", "TRADING_DAYS",
]
