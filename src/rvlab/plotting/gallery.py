"""
rvlab.plotting.gallery
======================
The chart set, as one visual system.

Each function makes *one* point, labels its axes with units, and takes a `title`
that should state the finding rather than the mechanic — "BF25 curvature carries
the signal", not "feature importances". Every one returns `(fig, ax)` so a
notebook can add a single annotation without reimplementing the chart.

Choosing the form is most of the work, and the heuristic is short:

  * comparing **distributions** — ECDF beats a histogram whenever you have more
    than two groups, because histograms occlude and ECDFs do not;
  * showing a **relationship** — scatter until the points overlap, then hexbin;
  * showing **change over time** — a line with a rolling band, and a separate
    drawdown panel if the level matters;
  * comparing **categories** — a box or a strip, never a bar of means, because a
    bar hides the spread that usually is the finding;
  * more than ~7 series — facet instead of overplotting.

seaborn is used where it genuinely shortens the code (KDE, ECDF, facets,
clustermaps) and matplotlib elsewhere. Both are driven by the same palette from
`rvlab.plotting.style`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .style import (
    C_ALT, C_BASELINE, C_GRID, C_MODEL, C_OBSERVED, C_WARN, DIVERGING, PALETTE,
    SEQUENTIAL, finish,
)


def _new(figsize=(8.0, 4.2)):
    import matplotlib.pyplot as plt
    return plt.subplots(figsize=figsize)


def _sns():
    """seaborn with the house grid, or None when it is not installed."""
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid",
                      rc={"axes.edgecolor": "#a0aec0", "grid.color": C_GRID})
        return sns
    except ImportError:
        return None


# ── 1. distributions ───────────────────────────────────────────────────────────
def hist_kde(values, bins: int = 60, ax=None, title: str = "", xlabel: str = "",
             note: str = "", clip_q=(0.001, 0.999)):
    """Histogram with a kernel density overlay. The default first look at one variable.

    Clip the *display*, never the data: a handful of extreme values otherwise
    compresses the whole histogram into a single bar.
    """
    fig, ax = (_new() if ax is None else (ax.figure, ax))
    v = pd.Series(values).dropna()
    lo, hi = v.quantile(list(clip_q))
    sns = _sns()
    if sns is not None:
        sns.histplot(v.clip(lo, hi), bins=bins, kde=True, ax=ax, color=C_OBSERVED,
                     edgecolor="white", linewidth=0.3)
    else:
        ax.hist(v.clip(lo, hi), bins=bins, color=C_OBSERVED)
    ax.axvline(float(v.median()), color=C_MODEL, lw=1.8, ls="--",
               label=f"median {v.median():.4g}")
    ax.legend()
    finish(ax, title or "Distribution", xlabel, "count",
           note or f"n = {len(v):,}, display clipped to {clip_q[0]:.1%}-{clip_q[1]:.1%}")
    return fig, ax


def ecdf_plot(df: pd.DataFrame, value: str, group: str | None = None, ax=None,
              title: str = "", note: str = ""):
    """Empirical CDF — the right way to compare more than two distributions.

    Histograms of several groups occlude each other and depend on bin width; ECDFs
    do neither, and vertical distance between two curves is exactly the KS
    statistic.
    """
    fig, ax = (_new() if ax is None else (ax.figure, ax))
    sns = _sns()
    if sns is not None:
        sns.ecdfplot(data=df, x=value, hue=group, ax=ax,
                     palette=PALETTE[:df[group].nunique()] if group else None,
                     color=None if group else C_OBSERVED)
    else:
        for i, (key, part) in enumerate(df.groupby(group) if group else [(None, df)]):
            v = np.sort(part[value].dropna())
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post",
                    color=PALETTE[i % len(PALETTE)], label=str(key))
        if group:
            ax.legend()
    finish(ax, title or f"ECDF of {value}", value, "cumulative proportion",
           note or "vertical distance between curves is the KS statistic")
    return fig, ax


def qq_plot(values, dist: str = "norm", ax=None, title: str = "", note: str = ""):
    """Quantile-quantile plot against a reference distribution.

    Reads at a glance: an S-shape means heavy tails, a bend at one end means skew,
    a straight line means the reference fits. Far more informative about tails
    than a histogram, which spends its resolution on the middle.
    """
    from scipy import stats

    fig, ax = (_new(figsize=(5.4, 5.0)) if ax is None else (ax.figure, ax))
    v = pd.Series(values).dropna().to_numpy()
    (osm, osr), (slope, intercept, r) = stats.probplot(v, dist=dist, fit=True)

    ax.scatter(osm, osr, s=10, color=C_OBSERVED, alpha=0.6)
    ax.plot(osm, slope * osm + intercept, color=C_MODEL, lw=1.8,
            label=f"{dist} fit (R² = {r ** 2:.3f})")
    ax.legend()
    finish(ax, title or f"Quantiles against a {dist} distribution",
           "theoretical quantiles", "sample quantiles",
           note or f"n = {len(v):,}; an S-shape means heavier tails than {dist}")
    return fig, ax


def violin_strip(df: pd.DataFrame, value: str, group: str, ax=None, max_groups: int = 10,
                 title: str = "", note: str = "", max_points: int = 400):
    """Violin (the shape) with the observations overlaid (the evidence).

    A violin alone can imply density a small sample does not support. Overlaying
    the points shows how many observations produced that shape — which is the
    honest version.
    """
    fig, ax = (_new(figsize=(9.0, 4.4)) if ax is None else (ax.figure, ax))
    data = df[[value, group]].dropna()
    order = (data.groupby(group, observed=True)[value].median()
             .sort_values(ascending=False).head(max_groups).index)
    data = data[data[group].isin(order)]

    sns = _sns()
    if sns is not None:
        sns.violinplot(data=data, x=group, y=value, order=order, ax=ax,
                       color=C_OBSERVED, inner=None, cut=0, linewidth=0.8)
        per_group = max_points // max(len(order), 1)
        thin = pd.concat([g.sample(min(len(g), per_group), random_state=0)
                          for _, g in data.groupby(group, observed=True)])
        sns.stripplot(data=thin, x=group, y=value, order=order, ax=ax,
                      color="#1a202c", size=1.8, alpha=0.45, jitter=0.25)
    else:
        ax.boxplot([data.loc[data[group] == g, value] for g in order],
                   tick_labels=list(order))
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    finish(ax, title or f"{value} by {group}", "", value,
           note or "shape from the violin, evidence from the points")
    return fig, ax


def ridgeline(df: pd.DataFrame, value: str, group: str, max_groups: int = 10,
              title: str = "", note: str = "", overlap: float = 0.6):
    """Stacked densities, one per group — many distributions in one panel.

    The form to reach for when you have 5-15 groups and care about the *shape* of
    each. Beyond ~15 it becomes unreadable; use an ECDF or a facet grid instead.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    data = df[[value, group]].dropna()
    order = (data.groupby(group, observed=True)[value].median()
             .sort_values().tail(max_groups).index.tolist())

    fig, ax = plt.subplots(figsize=(8.0, 0.55 * len(order) + 2.0))
    grid = np.linspace(data[value].quantile(0.005), data[value].quantile(0.995), 256)

    for i, key in enumerate(order):
        v = data.loc[data[group] == key, value].to_numpy()
        if v.size < 5 or np.ptp(v) == 0:
            continue
        density = stats.gaussian_kde(v)(grid)
        density = density / density.max() * (1 + overlap)
        ax.fill_between(grid, i, i + density, color=PALETTE[i % len(PALETTE)],
                        alpha=0.72, lw=0.8, edgecolor="white", zorder=len(order) - i)
        ax.text(grid[0], i + 0.12, str(key), fontsize=8.5, va="bottom", ha="left")

    ax.set_yticks([])
    finish(ax, title or f"{value} by {group}", value, "",
           note or f"{len(order)} groups, densities scaled to equal height")
    return fig, ax


# ── 2. relationships ───────────────────────────────────────────────────────────
def scatter_fit(df: pd.DataFrame, x: str, y: str, ax=None, order: int = 1,
                max_points: int = 3000, title: str = "", note: str = ""):
    """Scatter with a fitted line and its confidence band.

    Subsamples for display when the data is large — the fit still uses everything,
    but 300,000 overlapping markers communicate nothing and take ten seconds to
    render.
    """
    fig, ax = (_new() if ax is None else (ax.figure, ax))
    data = df[[x, y]].dropna()
    shown = data.sample(min(len(data), max_points), random_state=0)

    sns = _sns()
    if sns is not None:
        sns.regplot(data=shown, x=x, y=y, ax=ax, order=order,
                    scatter_kws={"s": 9, "alpha": 0.35, "color": C_OBSERVED},
                    line_kws={"color": C_MODEL, "lw": 2})
    else:
        ax.scatter(shown[x], shown[y], s=9, alpha=0.35, color=C_OBSERVED)

    r = data[x].corr(data[y])
    rho = data[x].corr(data[y], method="spearman")
    finish(ax, title or f"{y} against {x}", x, y,
           note or f"n = {len(data):,} (showing {len(shown):,}) · "
                   f"Pearson {r:+.3f}, Spearman {rho:+.3f}")
    return fig, ax


def hexbin_plot(df: pd.DataFrame, x: str, y: str, ax=None, gridsize: int = 40,
                title: str = "", note: str = ""):
    """Density-binned scatter — the fix for overplotting.

    Once markers overlap, a scatter shows you the *outline* of the data and
    nothing about where the mass is. Hexbin shows the mass. Use it above roughly
    5,000 points.
    """
    fig, ax = (_new() if ax is None else (ax.figure, ax))
    data = df[[x, y]].dropna()
    hb = ax.hexbin(data[x], data[y], gridsize=gridsize, cmap=SEQUENTIAL,
                   mincnt=1, linewidths=0.2)
    fig.colorbar(hb, ax=ax, label="count", fraction=0.04, pad=0.02)
    finish(ax, title or f"Where the mass of ({x}, {y}) actually is", x, y,
           note or f"n = {len(data):,}")
    return fig, ax


def joint_plot(df: pd.DataFrame, x: str, y: str, kind: str = "hex",
               title: str = "", note: str = ""):
    """Bivariate view with both marginals attached.

    Use when the marginals matter as much as the relationship — a joint plot
    catches the case where a correlation is driven by one clump in a bimodal
    margin, which a bare scatter hides.
    """
    sns = _sns()
    if sns is None:
        return hexbin_plot(df, x, y, title=title, note=note)
    data = df[[x, y]].dropna()
    grid = sns.jointplot(data=data, x=x, y=y, kind=kind, height=5.2,
                         color=C_OBSERVED, marginal_kws={"color": C_OBSERVED})
    grid.figure.suptitle(title or f"{y} against {x}, with marginals",
                         x=0.02, ha="left", fontsize=11, fontweight="600")
    grid.figure.tight_layout()
    return grid.figure, grid.ax_joint


def pair_grid(df: pd.DataFrame, columns, hue: str | None = None,
              max_points: int = 1500, title: str = ""):
    """Every pairwise relationship at once — the fastest way to see structure.

    Keep it to 4-6 columns. Beyond that the panels are too small to read and the
    render cost grows quadratically.
    """
    sns = _sns()
    cols = list(columns) + ([hue] if hue else [])
    data = df[cols].dropna()
    data = data.sample(min(len(data), max_points), random_state=0)
    if sns is None:
        return correlation_contrast(data[list(columns)])

    grid = sns.pairplot(data, vars=list(columns), hue=hue, corner=True,
                        plot_kws={"s": 10, "alpha": 0.4},
                        diag_kind="kde", palette=PALETTE[:data[hue].nunique()] if hue else None)
    grid.figure.suptitle(title or "Pairwise structure", x=0.02, ha="left",
                         fontsize=11, fontweight="600")
    grid.figure.tight_layout()
    return grid.figure, grid.axes


def correlation_contrast(df: pd.DataFrame, title: str = "", note: str = ""):
    """Pearson and Spearman side by side, plus their difference.

    The third panel is the point. Where the two disagree, the relationship is
    monotone but not linear, or a few extreme points are driving Pearson — and
    either way a linear model is being told something misleading.
    """
    import matplotlib.pyplot as plt

    numeric = df.select_dtypes("number")
    pearson, spearman = numeric.corr(), numeric.corr("spearman")
    delta = pearson - spearman
    n = len(pearson)

    fig, axes = plt.subplots(1, 3, figsize=(3.7 * 3 + 1.2, 0.42 * n + 2.6))
    for ax, (mat, label, lim) in zip(axes, [(pearson, "Pearson", 1),
                                            (spearman, "Spearman", 1),
                                            (delta, "Pearson − Spearman", 0.3)]):
        im = ax.imshow(mat.to_numpy(), cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(n), mat.columns, rotation=60, ha="right", fontsize=7.5)
        ax.set_yticks(range(n), mat.index, fontsize=7.5)
        ax.grid(False)
        finish(ax, label, "", "")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    if note:
        fig.text(0.005, -0.03, note, fontsize=8.5, color="#718096")
    fig.suptitle(title or "Where the two correlations disagree, the relationship is not linear",
                 x=0.005, ha="left", fontsize=11.5, fontweight="600")
    fig.tight_layout()
    return fig, axes


def cluster_map(df: pd.DataFrame, columns=None, title: str = ""):
    """Correlation heatmap with the rows and columns reordered by similarity.

    The reordering is the value: it groups features that move together, which a
    fixed-order heatmap cannot show. Use it to find redundant feature blocks
    before deciding what to drop.
    """
    sns = _sns()
    numeric = df[list(columns)] if columns else df.select_dtypes("number")
    corr = numeric.corr()
    if sns is None:
        return correlation_contrast(numeric, title=title)
    grid = sns.clustermap(corr, cmap=DIVERGING, vmin=-1, vmax=1, center=0,
                          figsize=(0.42 * len(corr) + 3.5, 0.42 * len(corr) + 3.5),
                          annot=len(corr) <= 12, fmt=".2f", annot_kws={"size": 7},
                          cbar_pos=(0.02, 0.83, 0.03, 0.14))
    grid.figure.suptitle(title or "Feature blocks, reordered by similarity",
                         x=0.02, ha="left", fontsize=11, fontweight="600")
    return grid.figure, grid.ax_heatmap


__all__ = [
    "hist_kde", "ecdf_plot", "qq_plot", "violin_strip", "ridgeline",
    "scatter_fit", "hexbin_plot", "joint_plot", "pair_grid",
    "correlation_contrast", "cluster_map",
]


# ── 3. time series ─────────────────────────────────────────────────────────────
def rolling_band(series, window: int = 20, k: float = 2.0, ax=None,
                 title: str = "", ylabel: str = "", note: str = ""):
    """A line with its rolling mean and a ±k·σ band — level and volatility together.

    The band is what makes the line interpretable: a move of 0.02 means something
    different when the recent standard deviation is 0.005 than when it is 0.05.
    """
    fig, ax = (_new() if ax is None else (ax.figure, ax))
    s = pd.Series(series).dropna()
    mean = s.rolling(window, min_periods=window // 2).mean()
    sd = s.rolling(window, min_periods=window // 2).std()

    ax.plot(s.index, s.to_numpy(), color=C_OBSERVED, lw=0.9, alpha=0.55, label="observed")
    ax.plot(mean.index, mean.to_numpy(), color=C_MODEL, lw=2, label=f"{window}-period mean")
    ax.fill_between(s.index, mean - k * sd, mean + k * sd, color=C_MODEL, alpha=0.15,
                    label=f"±{k:g}σ")
    ax.legend()
    finish(ax, title or "Level with its rolling volatility band", "", ylabel,
           note or f"{len(s):,} observations, window {window}")
    return fig, ax


def calendar_heatmap(series, ax=None, title: str = "", note: str = "", cmap=DIVERGING):
    """Value by weekday and week — seasonality you cannot see in a line chart.

    Day-of-week and turn-of-month effects are real in financial data and invisible
    on a time axis, because they are periodic at a scale a line chart compresses
    away.
    """
    fig, ax = (_new(figsize=(11.0, 3.4)) if ax is None else (ax.figure, ax))
    s = pd.Series(series).dropna()
    s.index = pd.to_datetime(s.index)

    frame = pd.DataFrame({"v": s.to_numpy(), "week": s.index.isocalendar().week.to_numpy(),
                          "year": s.index.year.to_numpy(), "dow": s.index.dayofweek})
    frame["yw"] = frame["year"].astype(str) + "-" + frame["week"].astype(str).str.zfill(2)
    grid = frame.pivot_table(index="dow", columns="yw", values="v", aggfunc="mean")

    lim = np.nanmax(np.abs(grid.to_numpy()))
    im = ax.imshow(grid.to_numpy(), cmap=cmap, aspect="auto", vmin=-lim, vmax=lim)
    ax.set_yticks(range(len(grid.index)),
                  [["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d] for d in grid.index])
    step = max(1, grid.shape[1] // 14)
    ax.set_xticks(range(0, grid.shape[1], step), grid.columns[::step], rotation=60,
                  ha="right", fontsize=7.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    finish(ax, title or "Value by weekday and week", "", "",
           note or f"{len(s):,} observations")
    return fig, ax


def acf_pacf(series, lags: int = 40, title: str = "", note: str = ""):
    """Autocorrelation and partial autocorrelation, with significance bands.

    ACF shows total dependence at each lag, PACF shows dependence *after removing*
    the shorter lags. The pair is how you tell an AR process (PACF cuts off) from
    an MA one (ACF cuts off) — and how you find out that your residuals are not
    white noise.
    """
    import matplotlib.pyplot as plt

    s = pd.Series(series).dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.6))
    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        plot_acf(s, lags=lags, ax=axes[0], color=C_OBSERVED, vlines_kwargs={"colors": C_OBSERVED})
        plot_pacf(s, lags=lags, ax=axes[1], method="ywm", color=C_OBSERVED,
                  vlines_kwargs={"colors": C_OBSERVED})
    except ImportError:
        for ax, fn in zip(axes, ("acf", "pacf")):
            vals = [s.autocorr(l) for l in range(1, lags + 1)]
            ax.bar(range(1, lags + 1), vals, color=C_OBSERVED)
    band = 1.96 / np.sqrt(len(s))
    for ax, label in zip(axes, ("Autocorrelation", "Partial autocorrelation")):
        ax.axhspan(-band, band, color=C_GRID, alpha=0.7, zorder=0)
        finish(ax, label, "lag", "correlation")
    fig.suptitle(title or "Serial dependence", x=0.005, ha="left",
                 fontsize=11.5, fontweight="600")
    if note:
        fig.text(0.005, -0.04, note, fontsize=8.5, color="#718096")
    fig.tight_layout()
    return fig, axes


def drawdown_underwater(returns, ax=None, title: str = "", note: str = "",
                        compound: bool = False):
    """Cumulative performance with the drawdown shaded beneath it.

    A Sharpe ratio says nothing about the *path*. The underwater shading answers
    the question that decides whether a strategy is runnable: how long, and how
    deep, were the bad stretches.
    """
    import matplotlib.pyplot as plt

    r = pd.Series(returns).dropna()
    cum = (1 + r).cumprod() - 1 if compound else r.cumsum()
    peak = cum.cummax()
    under = cum - peak

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 5.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    axes[0].plot(cum.index, cum.to_numpy(), color=C_MODEL, lw=1.8)
    axes[0].axhline(0, color="#4a5568", lw=1)
    sharpe = r.mean() / r.std(ddof=1) if r.std(ddof=1) else np.nan
    finish(axes[0], title or "Cumulative performance and its drawdowns", "",
           "cumulative return", "")

    axes[1].fill_between(under.index, under.to_numpy(), 0, color=C_WARN, alpha=0.35)
    axes[1].plot(under.index, under.to_numpy(), color=C_WARN, lw=1)
    finish(axes[1], "", "", "drawdown",
           note or f"Sharpe {sharpe:.2f} · max drawdown {under.min():.4f} · "
                   f"{(under < 0).mean():.0%} of periods underwater")
    fig.tight_layout()
    return fig, axes


def event_study(df: pd.DataFrame, value: str, time: str, event_times,
                window: int = 10, ax=None, title: str = "", note: str = ""):
    """Average path around a set of events, with a confidence band.

    Aligns every event at t=0 and averages. The band is the standard error across
    events — without it an event study will show you a pattern in three
    observations.
    """
    fig, ax = (_new() if ax is None else (ax.figure, ax))
    s = df.set_index(time)[value].sort_index()
    paths = []
    for stamp in pd.to_datetime(pd.Series(list(event_times))):
        pos = s.index.searchsorted(stamp)
        if pos - window < 0 or pos + window >= len(s):
            continue
        segment = s.iloc[pos - window: pos + window + 1].to_numpy(dtype=float)
        paths.append(segment - segment[window])          # re-base at the event

    if not paths:
        finish(ax, title or "Event study", "periods from event", value,
               "no event had a complete window")
        return fig, ax

    stack = np.vstack(paths)
    offsets = np.arange(-window, window + 1)
    mean = stack.mean(axis=0)
    err = stack.std(axis=0, ddof=1) / np.sqrt(len(stack))

    ax.plot(offsets, mean, color=C_MODEL, lw=2, marker="o", ms=3)
    ax.fill_between(offsets, mean - 1.96 * err, mean + 1.96 * err, color=C_MODEL, alpha=0.18)
    ax.axvline(0, color=C_WARN, lw=1.4, ls="--")
    ax.axhline(0, color="#4a5568", lw=1)
    finish(ax, title or f"{value} around the event", "periods from event",
           f"change in {value}", note or f"{len(stack)} events, 95% band")
    return fig, ax


def multi_series(df: pd.DataFrame, columns, rebase: bool = True, ax=None,
                 title: str = "", ylabel: str = "", note: str = ""):
    """Several series on one axis, rebased to a common start.

    Rebasing is what makes them comparable. Plotting raw levels with different
    scales on one axis — or worse, on twin axes — invites the reader to see a
    relationship that is an artefact of the scaling.
    """
    fig, ax = (_new() if ax is None else (ax.figure, ax))
    for i, col in enumerate(columns):
        s = df[col].dropna()
        v = s / s.iloc[0] if rebase and s.iloc[0] else s
        ax.plot(s.index, v.to_numpy(), color=PALETTE[i % len(PALETTE)], lw=1.6, label=col)
    ax.legend(ncols=min(len(list(columns)), 4))
    finish(ax, title or "Series comparison", "",
           ylabel or ("rebased to 1.0 at the start" if rebase else ""),
           note or ("twin axes were avoided on purpose — rebasing is honest, "
                    "a second y-axis is not"))
    return fig, ax


# ── 4. categorical and facets ──────────────────────────────────────────────────
def facet_grid(df: pd.DataFrame, x: str, y: str, col: str, col_wrap: int = 4,
               kind: str = "scatter", title: str = ""):
    """Small multiples — one panel per category, shared axes.

    The answer to "too many series on one chart". Above about seven groups,
    overplotting hides everything; faceting keeps every group readable and makes
    differences in *shape* obvious.
    """
    sns = _sns()
    data = df[[x, y, col]].dropna()
    if sns is None:
        return scatter_fit(data, x, y, title=title)
    grid = sns.relplot(data=data, x=x, y=y, col=col, col_wrap=col_wrap, kind=kind,
                       height=2.3, s=10, alpha=0.5, color=C_OBSERVED,
                       facet_kws={"sharex": True, "sharey": True})
    grid.figure.suptitle(title or f"{y} against {x}, by {col}", x=0.02, ha="left",
                         fontsize=11, fontweight="600")
    grid.figure.tight_layout()
    return grid.figure, grid.axes


def annotated_bar(values, errors=None, ax=None, title: str = "", ylabel: str = "",
                  note: str = "", highlight_best: bool = True, higher_is_better: bool = True):
    """Bar chart with value labels and error bars, winner highlighted.

    A bar of means with no error bar is the least informative chart in common use:
    it asserts a difference without showing whether one exists. If you have the
    uncertainty, plot it.
    """
    fig, ax = (_new(figsize=(7.6, 3.9)) if ax is None else (ax.figure, ax))
    s = pd.Series(values).sort_values(ascending=not higher_is_better)
    err = pd.Series(errors).reindex(s.index) if errors is not None else None

    best = 0 if highlight_best else -1
    colours = [C_MODEL if i == best else C_BASELINE for i in range(len(s))]
    ax.bar(s.index.astype(str), s.to_numpy(), yerr=None if err is None else err.to_numpy(),
           color=colours, width=0.62, capsize=4, error_kw={"lw": 1, "ecolor": "#4a5568"})
    for x, v in zip(s.index.astype(str), s.to_numpy()):
        ax.text(x, v, f"{v:.3g}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5)
    ax.axhline(0, color="#4a5568", lw=1)
    ax.tick_params(axis="x", rotation=20)
    finish(ax, title or "Comparison", "", ylabel, note)
    return fig, ax


def decile_lift(y_true, score, n_bins: int = 10, ax=None, title: str = "",
                ylabel: str = "", note: str = ""):
    """Mean outcome by predicted decile — does the ordering actually work?

    The most direct answer to "is this model useful". A monotone staircase means
    the ranking is real; a flat or ragged one means the aggregate metric is being
    carried by something other than a usable ordering.
    """
    fig, ax = (_new(figsize=(7.6, 4.0)) if ax is None else (ax.figure, ax))
    frame = pd.DataFrame({"y": np.asarray(y_true, float),
                          "s": np.asarray(score, float)}).dropna()
    frame["bin"] = pd.qcut(frame["s"].rank(method="first"), n_bins,
                           labels=range(1, n_bins + 1))
    grouped = frame.groupby("bin", observed=True)["y"]
    mean, err = grouped.mean(), grouped.sem()

    colours = [C_MODEL if i in (0, n_bins - 1) else C_BASELINE for i in range(n_bins)]
    ax.bar(mean.index.astype(int), mean.to_numpy(), yerr=err.to_numpy(), color=colours,
           width=0.7, capsize=3, error_kw={"lw": 1, "ecolor": "#4a5568"})
    ax.axhline(float(frame["y"].mean()), color=C_WARN, ls="--", lw=1.4,
               label=f"overall mean {frame['y'].mean():.4g}")
    ax.legend()
    spread = float(mean.iloc[-1] - mean.iloc[0])
    finish(ax, title or "Outcome by predicted decile", "decile of prediction (1 = lowest)",
           ylabel or "mean outcome",
           note or f"top-minus-bottom spread {spread:+.4g}, n = {len(frame):,}")
    return fig, ax


__all__ += [
    "rolling_band", "calendar_heatmap", "acf_pacf", "drawdown_underwater",
    "event_study", "multi_series", "facet_grid", "annotated_bar", "decile_lift",
]


# ── 5. model diagnostics ───────────────────────────────────────────────────────
def residual_panel(y_true, y_pred, title: str = "", note: str = ""):
    """Four residual views: vs fitted, distribution, QQ, and vs observation order.

    Each panel tests a different assumption, and the fourth is the one that is
    almost always omitted and almost always informative on time-series data:
    residuals plotted in observation order reveal serial correlation, which
    invalidates every standard error the model reported.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    y = np.asarray(y_true, float).ravel()
    p = np.asarray(y_pred, float).ravel()
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    resid = y - p

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0))
    step = max(1, len(y) // 4000)

    axes[0, 0].scatter(p[::step], resid[::step], s=8, alpha=0.35, color=C_OBSERVED)
    axes[0, 0].axhline(0, color=C_WARN, lw=1.2)
    finish(axes[0, 0], "Residual against fitted", "fitted", "residual")

    axes[0, 1].hist(resid, bins=60, color=C_OBSERVED, edgecolor="white", linewidth=0.3)
    axes[0, 1].axvline(float(resid.mean()), color=C_MODEL, lw=1.8,
                       label=f"mean {resid.mean():+.3g}")
    axes[0, 1].legend()
    finish(axes[0, 1], "Residual distribution", "residual", "count")

    (osm, osr), (slope, inter, r) = stats.probplot(resid, dist="norm", fit=True)
    axes[1, 0].scatter(osm[::step], osr[::step], s=8, alpha=0.4, color=C_OBSERVED)
    axes[1, 0].plot(osm, slope * osm + inter, color=C_MODEL, lw=1.6)
    finish(axes[1, 0], f"Normal QQ (R² = {r ** 2:.3f})", "theoretical", "sample")

    axes[1, 1].plot(np.arange(len(resid))[::step], resid[::step], color=C_OBSERVED,
                    lw=0.6, alpha=0.8)
    axes[1, 1].axhline(0, color=C_WARN, lw=1.2)
    ac1 = pd.Series(resid).autocorr(1)
    finish(axes[1, 1], f"Residual in order (lag-1 autocorrelation {ac1:+.3f})",
           "observation", "residual")

    fig.suptitle(title or "Residual diagnostics", x=0.005, ha="left",
                 fontsize=12, fontweight="600")
    if note:
        fig.text(0.005, -0.02, note, fontsize=8.5, color="#718096")
    fig.tight_layout()
    return fig, axes


def learning_curve_plot(estimator, X, y, cv=None, scoring=None, ax=None,
                        train_sizes=None, n_jobs: int = 1, title: str = "",
                        note: str = ""):
    """Score against training-set size — does more data help, or more model?

    The shape is the diagnosis. Curves that converge to a low score together mean
    **bias**: a bigger model, not more data. A wide persistent gap means
    **variance**: more data or more regularisation. Curves still separating at the
    right edge mean more data would genuinely help.
    """
    from sklearn.model_selection import learning_curve

    fig, ax = (_new() if ax is None else (ax.figure, ax))
    sizes, train, test = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs,
        train_sizes=train_sizes if train_sizes is not None else np.linspace(0.15, 1.0, 6))

    for values, colour, label in ((train, C_BASELINE, "training"),
                                  (test, C_MODEL, "validation")):
        mean, sd = values.mean(axis=1), values.std(axis=1)
        ax.plot(sizes, mean, color=colour, lw=2, marker="o", ms=4, label=label)
        ax.fill_between(sizes, mean - sd, mean + sd, color=colour, alpha=0.15)
    ax.legend()
    gap = float(train.mean(axis=1)[-1] - test.mean(axis=1)[-1])
    finish(ax, title or "Learning curve", "training examples", scoring or "score",
           note or f"final train-validation gap {gap:+.4f} — "
                   f"{'variance' if abs(gap) > 0.05 else 'bias'}-dominated")
    return fig, ax


def validation_curve_plot(estimator, X, y, param_name: str, param_range, cv=None,
                          scoring=None, ax=None, logx: bool = True, n_jobs: int = 1,
                          title: str = "", note: str = ""):
    """Score against one hyperparameter — where the bias-variance trade-off sits.

    Read the *shape*, not just the peak. A flat curve means the parameter does not
    matter and tuning it was theatre; a sharp peak means the choice is real and
    the value should be selected inside cross-validation, not by eye.
    """
    from sklearn.model_selection import validation_curve

    fig, ax = (_new() if ax is None else (ax.figure, ax))
    train, test = validation_curve(estimator, X, y, param_name=param_name,
                                   param_range=list(param_range), cv=cv,
                                   scoring=scoring, n_jobs=n_jobs)
    xs = np.asarray(list(param_range), dtype=float)

    for values, colour, label in ((train, C_BASELINE, "training"),
                                  (test, C_MODEL, "validation")):
        mean, sd = values.mean(axis=1), values.std(axis=1)
        ax.plot(xs, mean, color=colour, lw=2, marker="o", ms=4, label=label)
        ax.fill_between(xs, mean - sd, mean + sd, color=colour, alpha=0.15)
    if logx:
        ax.set_xscale("log")
    best = xs[int(np.argmax(test.mean(axis=1)))]
    ax.axvline(best, color=C_WARN, ls="--", lw=1.4, label=f"best {best:g}")
    ax.legend()
    finish(ax, title or f"Validation curve over {param_name}", param_name,
           scoring or "score", note)
    return fig, ax


def calibration_plot(y_true, proba, n_bins: int = 10, ax=None, title: str = "",
                     note: str = ""):
    """Predicted probability against observed frequency, with the histogram beneath.

    A model can rank perfectly and still be badly calibrated — excellent ROC-AUC,
    probabilities that are systematically too confident. If you will *act* on the
    number rather than the ordering (position sizing, expected value), this is the
    plot that matters, not the AUC.
    """
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    y = np.asarray(y_true).ravel().astype(int)
    p = np.asarray(proba, float).ravel()
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]

    fig, axes = plt.subplots(2, 1, figsize=(6.0, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    frac, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    axes[0].plot([0, 1], [0, 1], color=C_GRID, lw=1.5, ls="--", label="perfect")
    axes[0].plot(mean_pred, frac, color=C_MODEL, lw=2, marker="o", ms=5, label="model")
    axes[0].legend()
    finish(axes[0], title or "Calibration", "", "observed frequency")

    axes[1].hist(p, bins=30, color=C_OBSERVED, edgecolor="white", linewidth=0.3)
    finish(axes[1], "", "predicted probability", "count",
           note or f"Brier score {brier_score_loss(y, p):.4f} (lower is better), "
                   f"base rate {y.mean():.3f}")
    fig.tight_layout()
    return fig, axes


def roc_pr_plot(y_true, proba, title: str = "", note: str = ""):
    """ROC and precision-recall side by side — and why the second one matters.

    ROC-AUC is nearly insensitive to class imbalance, which sounds like a virtue
    and is not: on a 2%-positive problem a model can have ROC-AUC 0.9 and be
    useless, because the curve is dominated by the abundant negatives. Average
    precision, whose baseline is the positive rate, tells you the truth.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import (average_precision_score, precision_recall_curve,
                                 roc_auc_score, roc_curve)

    y = np.asarray(y_true).ravel().astype(int)
    p = np.asarray(proba, float).ravel()
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    fpr, tpr, _ = roc_curve(y, p)
    axes[0].plot(fpr, tpr, color=C_MODEL, lw=2, label=f"AUC {roc_auc_score(y, p):.3f}")
    axes[0].plot([0, 1], [0, 1], color=C_GRID, ls="--", lw=1.5, label="chance")
    axes[0].legend()
    finish(axes[0], "ROC", "false positive rate", "true positive rate")

    prec, rec, _ = precision_recall_curve(y, p)
    base = float(y.mean())
    axes[1].plot(rec, prec, color=C_MODEL, lw=2,
                 label=f"AP {average_precision_score(y, p):.3f}")
    axes[1].axhline(base, color=C_WARN, ls="--", lw=1.5, label=f"base rate {base:.3f}")
    axes[1].legend()
    finish(axes[1], "Precision-recall", "recall", "precision")

    fig.suptitle(title or "Two views of the same classifier", x=0.005, ha="left",
                 fontsize=11.5, fontweight="600")
    if note:
        fig.text(0.005, -0.03, note, fontsize=8.5, color="#718096")
    fig.tight_layout()
    return fig, axes


def confusion_matrix_plot(y_true, y_pred, labels=None, normalize: str = "true",
                          ax=None, title: str = "", note: str = ""):
    """Confusion matrix, normalised by row so the classes are comparable.

    Normalising by true class turns each row into "of the actual X, what did we
    say" — which is recall per class, and the question you usually mean. A raw
    count matrix on imbalanced data is dominated by the majority class.
    """
    from sklearn.metrics import confusion_matrix

    fig, ax = (_new(figsize=(4.8, 4.2)) if ax is None else (ax.figure, ax))
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    names = labels or sorted(pd.unique(np.asarray(y_true).ravel()))

    im = ax.imshow(cm, cmap=SEQUENTIAL, vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > 0.5 else "#1a202c")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    finish(ax, title or "Confusion matrix", "predicted", "actual",
           note or f"normalised by {normalize}; rows are per-class recall")
    return fig, ax


def pdp_grid(estimator, X, features, kind: str = "average", n_cols: int = 3,
             title: str = ""):
    """Partial dependence for several features on a shared layout.

    > ⚠️ **Extrapolation:** partial dependence evaluates the model at feature
    > combinations that may never occur. Where a feature is strongly correlated
    > with others, the tails of these curves are model artefacts, not findings.
    """
    import matplotlib.pyplot as plt
    from sklearn.inspection import PartialDependenceDisplay

    features = list(features)
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.2 * n_rows))
    flat = np.atleast_1d(axes).ravel()
    PartialDependenceDisplay.from_estimator(estimator, X, features, ax=flat[:len(features)],
                                            kind=kind, line_kw={"color": C_MODEL})
    for extra in flat[len(features):]:
        extra.set_visible(False)
    fig.suptitle(title or "Partial dependence", x=0.005, ha="left",
                 fontsize=11.5, fontweight="600")
    fig.tight_layout()
    return fig, axes


# ── 6. craft ───────────────────────────────────────────────────────────────────
def palette_preview(palette=None, title: str = ""):
    """Show the palette as swatches, in colour and in greyscale.

    The greyscale row is the test that matters. A palette whose colours collapse
    to the same grey will be unreadable when printed, photocopied, or read by
    anyone with reduced colour vision — and you cannot tell by looking at the
    colour row.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    colours = list(palette or PALETTE)
    fig, axes = plt.subplots(2, 1, figsize=(1.15 * len(colours) + 1.2, 2.6))
    for row, (ax, mode) in enumerate(zip(axes, ("colour", "greyscale"))):
        for i, c in enumerate(colours):
            rgb = np.array(to_rgb(c))
            if mode == "greyscale":
                rgb = np.repeat(float(rgb @ [0.299, 0.587, 0.114]), 3)
            ax.add_patch(plt.Rectangle((i, 0), 0.92, 1, color=rgb))
            if row == 0:
                ax.text(i + 0.46, -0.18, str(i), ha="center", fontsize=8)
        ax.set_xlim(0, len(colours)); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_ylabel(mode, fontsize=9, rotation=0, ha="right", va="center")
    fig.suptitle(title or "Palette, and how it survives greyscale",
                 x=0.02, ha="left", fontsize=11, fontweight="600")
    fig.tight_layout()
    return fig, axes


def colorblind_check(palette=None, title: str = ""):
    """Simulate deuteranopia and protanopia over the palette.

    Roughly 8% of men have reduced red-green vision, which is exactly the axis
    most default palettes rely on. Two colours that look distinct here and
    identical below are two series a reader cannot tell apart.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    # Brettel/Machado-style linear approximations, adequate for a design check.
    matrices = {
        "normal": np.eye(3),
        "deuteranopia": np.array([[0.625, 0.375, 0.0],
                                  [0.700, 0.300, 0.0],
                                  [0.000, 0.300, 0.7]]),
        "protanopia": np.array([[0.567, 0.433, 0.0],
                                [0.558, 0.442, 0.0],
                                [0.000, 0.242, 0.758]]),
    }
    colours = list(palette or PALETTE)
    fig, axes = plt.subplots(len(matrices), 1,
                             figsize=(1.15 * len(colours) + 1.6, 1.0 * len(matrices) + 1.2))
    for ax, (label, matrix) in zip(axes, matrices.items()):
        for i, c in enumerate(colours):
            shifted = np.clip(matrix @ np.array(to_rgb(c)), 0, 1)
            ax.add_patch(plt.Rectangle((i, 0), 0.92, 1, color=shifted))
        ax.set_xlim(0, len(colours)); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_ylabel(label, fontsize=8.5, rotation=0, ha="right", va="center")
    fig.suptitle(title or "The same palette under two kinds of colour blindness",
                 x=0.02, ha="left", fontsize=11, fontweight="600")
    fig.tight_layout()
    return fig, axes


def annotate_point(ax, x, y, text: str, dx: float = 45, dy: float = 30,
                   colour: str = C_WARN):
    """Label one point with an arrow — say what the reader should look at.

    A chart with no annotation asks the reader to find the finding. One arrow and
    six words is usually the difference between a figure that communicates and a
    figure that merely displays.

    `dx`/`dy` are offsets in **points**, not data units. That is deliberate: a
    data-unit offset breaks on a datetime axis (a Timestamp plus a float is not a
    Timestamp) and has to be re-tuned whenever the axis range changes. Points work
    on every axis type and mean the same thing at any zoom.
    """
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=9, color=colour, fontweight="600",
                ha="left", va="bottom",
                arrowprops={"arrowstyle": "->", "color": colour, "lw": 1.3})
    return ax


def save_publication(fig, name: str, dpi: int = 300, formats=("png",), subdir="figures"):
    """Write a figure at print resolution, returning the paths.

    300 dpi for raster, and PDF when the figure will be scaled — a vector format
    survives being enlarged in a slide deck, a 150-dpi PNG does not. `bbox_inches`
    is set globally in the house style so captions are never clipped.
    """
    from ..config import OUTPUT_DIR

    out = OUTPUT_DIR / subdir
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = out / f"{name}.{fmt}"
        fig.savefig(path, dpi=dpi, format=fmt)
        paths.append(path)
    return paths


__all__ += [
    "residual_panel", "learning_curve_plot", "validation_curve_plot",
    "calibration_plot", "roc_pr_plot", "confusion_matrix_plot", "pdp_grid",
    "palette_preview", "colorblind_check", "annotate_point", "save_publication",
]


def line_with_reference(values, reference: float | None = None, ax=None,
                        reference_label: str = "long-run", title: str = "",
                        xlabel: str = "", ylabel: str = "", note: str = "",
                        marker: str = "o"):
    """A single path against a horizontal reference level.

    The shape for anything that decays toward, or diverges from, a known level —
    a variance forecast reverting to its unconditional value, a metric against a
    baseline, a convergence diagnostic. The reference line is what turns "the
    number went down" into "the number is most of the way back".
    """
    fig, ax = (_new() if ax is None else (ax.figure, ax))
    s = pd.Series(values).dropna()
    ax.plot(s.index, s.to_numpy(), color=C_MODEL, lw=2, marker=marker, ms=3.5)
    if reference is not None and np.isfinite(reference):
        ax.axhline(reference, color=C_BASELINE, ls="--", lw=1.6,
                   label=f"{reference_label} {reference:.4g}")
        ax.legend()
    finish(ax, title or "Path against its reference level", xlabel, ylabel, note)
    return fig, ax


__all__ += ["line_with_reference"]
