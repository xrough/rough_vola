"""
rvlab.plotting.charts
=====================
The figures this series uses, each in one call.

Conventions every chart follows:
  * observed data is blue, the baseline is grey, the model under test is orange;
  * the title states the finding, not the mechanic;
  * axis labels carry units;
  * a `note` argument prints sample size or caveats under the axes.

Each function returns `(fig, ax)` so a notebook can add one annotation without
reimplementing the chart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .style import (
    C_ALT, C_BASELINE, C_GRID, C_MODEL, C_OBSERVED, C_WARN, DIVERGING, PALETTE, finish,
)


def _new(figsize=(8.0, 4.2)):
    import matplotlib.pyplot as plt
    return plt.subplots(figsize=figsize)


def smile_plot(k, implied_vol, label: str = "observed", ax=None,
               title: str = "", note: str = "", color: str | None = None):
    """Implied vol against log-moneyness — one maturity."""
    fig, ax = (_new() if ax is None else (ax.figure, ax))
    ax.plot(np.asarray(k), np.asarray(implied_vol) * 100, marker="o",
            color=color or C_OBSERVED, label=label)
    ax.axvline(0.0, color=C_GRID, lw=1, zorder=0)
    ax.legend()
    finish(ax, title or "Implied volatility smile", "log-moneyness  log(K/F)",
           "implied vol (%)", note)
    return fig, ax


def term_structure_plot(df: pd.DataFrame, x: str = "T", y: str = "rr25",
                        by: str | None = None, loglog: bool = False, ax=None,
                        title: str = "", note: str = "", ylabel: str = ""):
    """A smile observable against maturity, optionally on log-log axes.

    `loglog=True` with y="rr25" is the Gate 1 picture: a straight line means a
    power law, and its slope is the rough exponent H - 1/2.
    """
    fig, ax = (_new() if ax is None else (ax.figure, ax))
    plot = ax.loglog if loglog else ax.plot

    if by:
        for i, (key, grp) in enumerate(df.groupby(by, observed=True)):
            g = grp.sort_values(x)
            vals = np.abs(g[y]) if loglog else g[y]
            plot(g[x], vals, marker="o", ms=3, label=str(key),
                 color=PALETTE[i % len(PALETTE)])
        ax.legend(title=by, ncols=2)
    else:
        g = df.sort_values(x)
        vals = np.abs(g[y]) if loglog else g[y]
        plot(g[x], vals, marker="o", ms=3, color=C_OBSERVED)

    finish(ax, title or f"{y} term structure",
           "maturity T (years, log)" if loglog else "maturity T (years)",
           ylabel or (f"|{y}| (log)" if loglog else y), note)
    return fig, ax


def scaling_fit_plot(T, values, slope: float, intercept: float,
                     implied_hurst: float | None = None, r_squared: float | None = None,
                     ax=None, title: str = "", note: str = ""):
    """log|value| vs log(T) with the fitted line — the estimator, made visible."""
    fig, ax = (_new() if ax is None else (ax.figure, ax))
    lt = np.log(np.asarray(T, float))
    ly = np.log(np.abs(np.asarray(values, float)))

    ax.scatter(lt, ly, s=18, color=C_OBSERVED, alpha=0.75, label="observed")
    xs = np.linspace(lt.min(), lt.max(), 50)
    lbl = f"fit: slope = {slope:+.3f}"
    if implied_hurst is not None:
        lbl += f"  ->  H = {implied_hurst:.3f}"
    ax.plot(xs, intercept + slope * xs, color=C_MODEL, lw=2, label=lbl)
    ax.legend()

    if r_squared is not None and not note:
        note = f"R2 = {r_squared:.3f}"
    finish(ax, title or "Skew follows a power law in maturity",
           "log T", "log |RR25|", note)
    return fig, ax


def distribution_plot(values, ax=None, bins: int = 60, reference: float | None = None,
                      reference_label: str = "prior", title: str = "",
                      xlabel: str = "", note: str = ""):
    """A histogram with an optional vertical reference line.

    Used for the distribution of fitted slopes against the value the rough prior
    implies — the comparison that decides Gate 1.
    """
    fig, ax = (_new() if ax is None else (ax.figure, ax))
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    ax.hist(v, bins=bins, color=C_OBSERVED, alpha=0.85, edgecolor="white", lw=0.4)
    ax.axvline(float(np.median(v)), color=C_MODEL, lw=2,
               label=f"median = {np.median(v):+.4f}")
    if reference is not None:
        ax.axvline(reference, color=C_WARN, lw=2, ls="--",
                   label=f"{reference_label} = {reference:+.4f}")
    ax.legend()
    finish(ax, title or "Distribution", xlabel, "count",
           note or f"n = {len(v):,}")
    return fig, ax


def forecast_diagnostics(y_true, predictions: dict, baseline: str | None = None,
                         max_points: int = 400, title: str = "", note: str = ""):
    """Three panels: predicted vs actual, error distributions, cumulative squared error.

    The third panel is the one that matters. A model can win on aggregate RMSE
    while losing over most of the sample; the cumulative-error curves show *when*
    each model earned its score.
    """
    import matplotlib.pyplot as plt

    names = list(predictions)
    baseline = baseline or names[0]
    y = np.asarray(y_true, float).ravel()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))

    step = max(1, len(y) // max_points)
    for i, name in enumerate(names):
        p = np.asarray(predictions[name], float).ravel()
        colour = C_BASELINE if name == baseline else PALETTE[(i + 2) % len(PALETTE)]
        axes[0].scatter(y[::step], p[::step], s=8, alpha=0.5, color=colour, label=name)
        axes[1].hist(y - p, bins=50, histtype="step", lw=1.6, color=colour, label=name)
        axes[2].plot(np.cumsum((y - p) ** 2), color=colour, lw=1.6, label=name)

    lims = [np.nanmin(y), np.nanmax(y)]
    axes[0].plot(lims, lims, color=C_GRID, lw=1, zorder=0)
    axes[0].legend(fontsize=8)
    finish(axes[0], "Predicted vs actual", "actual", "predicted")
    finish(axes[1], "Error distribution", "actual - predicted", "count")
    finish(axes[2], "Cumulative squared error", "observation", "cumulative SE")
    axes[2].legend(fontsize=8)

    if title:
        fig.suptitle(title, x=0.008, ha="left", fontsize=12, fontweight="600")
    if note:
        fig.text(0.008, -0.04, note, fontsize=8.5, color="#718096")
    fig.tight_layout()
    return fig, axes


def sweep_heatmap(matrix: pd.DataFrame, ax=None, center: float = 0.0,
                  title: str = "", xlabel: str = "", ylabel: str = "",
                  fmt: str = "{:+.3f}", note: str = "", cmap: str = DIVERGING):
    """A parameter-sweep matrix, diverging around `center` (default: zero skill).

    Diverging rather than sequential on purpose: for a skill score the sign is
    the finding, and a sequential map hides where a cell crosses zero.
    """
    fig, ax = (_new(figsize=(1.15 * matrix.shape[1] + 3.2,
                             0.55 * matrix.shape[0] + 2.4)) if ax is None
               else (ax.figure, ax))
    vals = matrix.to_numpy(dtype=float)
    lim = np.nanmax(np.abs(vals - center)) or 1.0

    im = ax.imshow(vals, cmap=cmap, vmin=center - lim, vmax=center + lim, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]), [str(c) for c in matrix.columns])
    ax.set_yticks(range(matrix.shape[0]), [str(i) for i in matrix.index])
    ax.grid(False)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = vals[i, j]
            if np.isfinite(v):
                shade = "white" if abs(v - center) > 0.6 * lim else "#1a202c"
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=8.5, color=shade)

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    finish(ax, title or "Parameter sweep", xlabel or str(matrix.columns.name or ""),
           ylabel or str(matrix.index.name or ""), note)
    return fig, ax


def pnl_plot(summary: pd.DataFrame, value_col: str = "rmse",
             policy_col: str = "policy", ax=None, title: str = "",
             ylabel: str = "", note: str = "", lower_is_better: bool = True):
    """Bar chart of a per-policy backtest statistic, best-first."""
    fig, ax = (_new(figsize=(7.0, 3.8)) if ax is None else (ax.figure, ax))
    df = summary.sort_values(value_col, ascending=lower_is_better)
    colours = [C_MODEL if i == 0 else C_BASELINE for i in range(len(df))]

    ax.bar(df[policy_col].astype(str), df[value_col], color=colours, width=0.6)
    for x, v in zip(df[policy_col].astype(str), df[value_col]):
        ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    finish(ax, title or f"{value_col} by hedge policy", "",
           ylabel or value_col,
           note or ("lower is better" if lower_is_better else "higher is better"))
    return fig, ax


def missingness_map(df: pd.DataFrame, max_rows: int = 2_000, ax=None,
                    title: str = "", note: str = ""):
    """Where the holes are — columns across, rows down, missing in red.

    Look for *vertical stripes* (a column that fails everywhere) versus
    *horizontal bands* (whole timestamps lost). They call for different fixes:
    the first is a broken feature, the second a data outage.
    """
    fig, ax = (_new(figsize=(9.0, 3.6)) if ax is None else (ax.figure, ax))
    step = max(1, len(df) // max_rows)
    mask = df.iloc[::step].isna().to_numpy(dtype=float)

    ax.imshow(mask, aspect="auto", cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(df.shape[1]), list(df.columns), rotation=60, ha="right", fontsize=8)
    ax.set_yticks([])
    ax.grid(False)
    pct = 100 * df.isna().to_numpy().mean()
    finish(ax, title or f"Missing values: {pct:.2f}% of cells", "", "rows (subsampled)",
           note or f"showing every {step} row of {len(df):,}")
    return fig, ax


def correlation_heatmap(df: pd.DataFrame, method: str = "spearman", ax=None,
                        title: str = "", note: str = ""):
    """Correlation matrix, Spearman by default.

    Spearman rather than Pearson because smile features have heavy tails, and a
    handful of bad quotes can create or destroy a Pearson correlation on their
    own.
    """
    corr = df.select_dtypes("number").corr(method=method)
    n = len(corr)
    fig, ax = (_new(figsize=(0.55 * n + 3.0, 0.5 * n + 2.4)) if ax is None
               else (ax.figure, ax))

    im = ax.imshow(corr.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n), corr.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(n), corr.index, fontsize=8)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            v = corr.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > 0.6 else "#1a202c")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    finish(ax, title or f"{method.title()} correlation", "", "", note)
    return fig, ax


def path_fan(t, paths: np.ndarray, n_show: int = 30, ax=None, title: str = "",
             xlabel: str = "time (years)", ylabel: str = "", note: str = ""):
    """A sample of simulated paths plus the 5/50/95 percentile envelope."""
    fig, ax = (_new() if ax is None else (ax.figure, ax))
    t = np.asarray(t, float)
    P = np.asarray(paths, float)

    for row in P[:n_show]:
        ax.plot(t, row, color=C_OBSERVED, alpha=0.18, lw=0.9)
    q05, q50, q95 = np.nanpercentile(P, [5, 50, 95], axis=0)
    ax.fill_between(t, q05, q95, color=C_MODEL, alpha=0.18, label="5-95%")
    ax.plot(t, q50, color=C_MODEL, lw=2, label="median")
    ax.legend()
    finish(ax, title or "Simulated paths", xlabel, ylabel,
           note or f"{P.shape[0]:,} paths, showing {min(n_show, len(P))}")
    return fig, ax


__all__ = [
    "smile_plot", "term_structure_plot", "scaling_fit_plot", "distribution_plot",
    "forecast_diagnostics", "sweep_heatmap", "pnl_plot", "missingness_map",
    "correlation_heatmap", "path_fan", "coverage_plot", "target_distribution_plot",
    "group_box_plot", "ic_timeseries_plot", "cumulative_return_plot",
    "importance_plot", "model_comparison_plot",
]


# ── seaborn-backed panel charts ────────────────────────────────────────────────
# seaborn is optional; each of these degrades to matplotlib when it is absent.
def _sns():
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "#a0aec0",
                                             "grid.color": C_GRID})
        return sns
    except ImportError:
        return None


def coverage_plot(df, date_col="Date", entity_col="SecuritiesCode", ax=None,
                  title: str = "", note: str = ""):
    """Entities present on each date — the first panel health check.

    A step down is a delisting wave or a truncated file; a step up is new
    listings. Either way the cross-section changed, and any metric averaged
    across the break is averaging two different universes.
    """
    fig, ax = (_new() if ax is None else (ax.figure, ax))
    counts = df.groupby(date_col, observed=True)[entity_col].nunique()
    ax.plot(counts.index, counts.to_numpy(), color=C_OBSERVED, lw=1.4)
    ax.fill_between(counts.index, 0, counts.to_numpy(), color=C_OBSERVED, alpha=0.12)
    ax.set_ylim(0, counts.max() * 1.08)
    finish(ax, title or "Cross-sectional coverage over time", "date", "entities",
           note or f"median {counts.median():.0f}, min {counts.min():.0f}")
    return fig, ax


def target_distribution_plot(df, target_col="Target", clip_q=(0.005, 0.995),
                             ax=None, title: str = "", note: str = ""):
    """Target histogram with a KDE, clipped for display only.

    Clipping the *display* keeps a handful of extreme returns from compressing
    the visible range into one bar. The clip is never applied to the data.
    """
    fig, ax = (_new() if ax is None else (ax.figure, ax))
    values = df[target_col].dropna()
    lo, hi = values.quantile(list(clip_q))
    sns = _sns()
    if sns is not None:
        sns.histplot(values.clip(lo, hi), bins=80, kde=True, color=C_OBSERVED,
                     ax=ax, edgecolor="white", linewidth=0.3)
    else:
        ax.hist(values.clip(lo, hi), bins=80, color=C_OBSERVED)
    ax.axvline(0, color=C_WARN, lw=1.2, ls="--")
    finish(ax, title or f"{target_col} distribution", target_col, "count",
           note or f"n = {len(values):,}, display clipped to "
                   f"{clip_q[0]:.1%}-{clip_q[1]:.1%}")
    return fig, ax


def group_box_plot(df, value_col, group_col, ax=None, max_groups: int = 14,
                   title: str = "", note: str = ""):
    """Distribution of a value by category — sectors, regimes, tenor buckets.

    Ordered by median so the comparison is readable. A boxplot beats a bar of
    means here because the *spread* per group is usually the finding.
    """
    fig, ax = (_new(figsize=(9.5, 4.4)) if ax is None else (ax.figure, ax))
    data = df[[value_col, group_col]].dropna()
    order = (data.groupby(group_col, observed=True)[value_col].median()
             .sort_values(ascending=False).head(max_groups).index)
    data = data[data[group_col].isin(order)]

    sns = _sns()
    if sns is not None:
        sns.boxplot(data=data, x=group_col, y=value_col, order=order, ax=ax,
                    showfliers=False, color=C_OBSERVED, width=0.6,
                    linecolor="#2d3748", linewidth=1.0)
    else:
        ax.boxplot([data.loc[data[group_col] == g, value_col] for g in order],
                   tick_labels=list(order), showfliers=False)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    finish(ax, title or f"{value_col} by {group_col}", "", value_col,
           note or "boxes ordered by median; outliers hidden")
    return fig, ax


def ic_timeseries_plot(ic, window: int = 20, ax=None, title: str = "", note: str = ""):
    """Daily rank IC with a rolling mean — is the signal steady or episodic?

    The rolling line is the one to read. A signal whose IC is positive on average
    but spends long stretches negative is far harder to run than the mean
    suggests.
    """
    fig, ax = (_new() if ax is None else (ax.figure, ax))
    ic = pd.Series(ic).dropna()
    ax.bar(ic.index, ic.to_numpy(), width=1.0,
           color=np.where(ic.to_numpy() > 0, C_OBSERVED, C_WARN), alpha=0.45)
    ax.plot(ic.index, ic.rolling(window, min_periods=window // 2).mean(),
            color=C_MODEL, lw=2, label=f"{window}-day mean")
    ax.axhline(0, color="#4a5568", lw=1)
    ax.axhline(ic.mean(), color=C_ALT, lw=1.2, ls="--",
               label=f"mean {ic.mean():+.4f}")
    ax.legend()
    finish(ax, title or "Daily rank IC", "date", "Spearman IC",
           note or f"positive on {(ic > 0).mean():.0%} of {len(ic):,} days")
    return fig, ax


def cumulative_return_plot(returns, ax=None, title: str = "", note: str = "",
                           label: str = "strategy"):
    """Cumulative sum of a daily return series, with drawdown shaded.

    Cumulative *sum* rather than product: these are spread returns on a metric
    scale, not compounding portfolio returns, and compounding them would imply a
    reinvestment story the metric does not make.
    """
    fig, ax = (_new() if ax is None else (ax.figure, ax))
    r = pd.Series(returns).dropna()
    cum = r.cumsum()
    peak = cum.cummax()

    ax.plot(cum.index, cum.to_numpy(), color=C_MODEL, lw=1.8, label=label)
    ax.fill_between(cum.index, cum.to_numpy(), peak.to_numpy(), color=C_WARN,
                    alpha=0.18, label="drawdown")
    ax.axhline(0, color="#4a5568", lw=1)
    ax.legend()
    sharpe = r.mean() / r.std(ddof=1) if r.std(ddof=1) else np.nan
    finish(ax, title or "Cumulative spread return", "date", "cumulative return",
           note or f"Sharpe {sharpe:.2f} over {len(r):,} days, "
                   f"max drawdown {(peak - cum).max():.3f}")
    return fig, ax


def importance_plot(importance: pd.DataFrame, ax=None, title: str = "",
                    note: str = "", highlight=()):
    """Horizontal bars from `rvlab.evaluate.importance.pipeline_feature_importance`."""
    fig, ax = (_new(figsize=(8.0, max(3.6, 0.26 * len(importance)))) if ax is None
               else (ax.figure, ax))
    data = importance.iloc[::-1]
    colours = [C_MODEL if any(h in f for h in highlight) else C_BASELINE
               for f in data["feature"]]
    ax.barh(data["feature"], data["importance"], color=colours)
    kind = data["kind"].iloc[0] if "kind" in data.columns else "importance"
    finish(ax, title or "Feature importance", kind.replace("_", " "), "",
           note or (f"orange = {', '.join(highlight)}" if highlight else ""))
    return fig, ax


def model_comparison_plot(report: pd.DataFrame, metric: str = "spread_sharpe",
                          ax=None, title: str = "", note: str = ""):
    """Bar chart of one metric across models, best first, winner highlighted."""
    fig, ax = (_new(figsize=(7.5, 3.8)) if ax is None else (ax.figure, ax))
    data = report[metric].sort_values(ascending=False)
    colours = [C_MODEL if i == 0 else C_BASELINE for i in range(len(data))]
    ax.bar(data.index.astype(str), data.to_numpy(), color=colours, width=0.62)
    ax.axhline(0, color="#4a5568", lw=1)
    for x, v in zip(data.index.astype(str), data.to_numpy()):
        ax.text(x, v, f"{v:.3f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8.5)
    ax.tick_params(axis="x", rotation=20)
    finish(ax, title or f"{metric} by model", "", metric, note)
    return fig, ax
