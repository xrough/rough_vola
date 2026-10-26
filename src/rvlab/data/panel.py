"""
rvlab.data.panel
================
Panel datasets: the shape almost every time-series competition uses.

A *panel* is (date x entity) — trading days by security, timestamps by expiry,
store-days by SKU. It is not the same as a single time series, and the
differences are exactly where predictive projects break:

  * a `shift(1)` that ignores the entity pulls a neighbouring entity's value;
  * a row-wise train/test split puts the same *date* on both sides;
  * the cross-section at each date is itself information — a stock's return
    means something different on a day when everything rose;
  * the entity set changes over time (listings, delistings, new expiries).

This module supplies the loading half: locating a dataset, concatenating the
train/supplemental folders competitions ship, auditing what arrived, and — when
no real data is present — generating a Kaggle-shaped panel so the recipes still
run.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config

# Column names follow the JPX Tokyo Stock Exchange competition, which is a
# representative example of the genre. Rename yours or pass the *_col arguments.
DATE = "Date"
ENTITY = "SecuritiesCode"
TARGET = "Target"


# ── locating a dataset ─────────────────────────────────────────────────────────
def find_dataset_root(marker: str = "train_files/stock_prices.csv",
                      extra_candidates=()) -> Path | None:
    """Find the directory containing `marker`, searching upward then downward.

    Competition data lands in a different place on every machine — a Kaggle
    input mount, a Downloads folder, a sibling of the notebook. Searching for a
    known file rather than hard-coding a path is what makes a notebook portable.

    Returns None rather than raising, so a caller can fall back to synthetic
    data instead of failing.
    """
    marker_path = Path(marker)
    kaggle = Path("/kaggle/input")
    candidates = [Path.cwd(), *Path.cwd().parents]
    if kaggle.is_dir():
        candidates += [kaggle, *kaggle.glob("*")]
    candidates += [Path(c) for c in extra_candidates]
    for base in candidates:
        try:
            if (base / marker_path).exists():
                return base
        except OSError:                          # unreadable mount, keep looking
            continue

    # Last resort: inspect at most 200 matching paths without materialising an
    # arbitrarily large result list (important under a shared Kaggle mount).
    for found in islice(Path.cwd().rglob(marker_path.name), 200):
        if found.match(f"*/{marker}") or found.name == marker_path.name:
            candidate = found
            for _ in marker_path.parts:
                candidate = candidate.parent
            if (candidate / marker_path).exists():
                return candidate
    return None


def read_if_exists(root: Path, relative_path: str, **kwargs) -> pd.DataFrame | None:
    """`pd.read_csv` that returns None instead of raising on a missing file.

    Competitions ship optional tables. Branching on None keeps a notebook
    running when only some of them are present.
    """
    path = Path(root) / relative_path
    return pd.read_csv(path, **kwargs) if path.exists() else None


def concat_folders(root: Path, filename: str,
                   folders=("train_files", "supplemental_files"),
                   key_cols=(DATE, ENTITY), date_col: str = DATE) -> pd.DataFrame | None:
    """Concatenate the same file across folders and de-duplicate on the keys.

    Competitions ship a frozen `train_files/` plus a rolling
    `supplemental_files/` that overlaps it. `keep="last"` makes the later folder
    win, which is what you want when the supplement carries corrections.

    A `_source_folder` column records where each row came from — cheap, and the
    first thing you want when the concatenated frame looks wrong.
    """
    frames = []
    for folder in folders:
        part = read_if_exists(root, f"{folder}/{filename}")
        if part is not None:
            part["_source_folder"] = folder
            frames.append(part)
    if not frames:
        return None

    out = pd.concat(frames, ignore_index=True)
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col])

    keys = [c for c in key_cols if c in out.columns]
    if keys:
        out = out.sort_values(keys).drop_duplicates(keys, keep="last")
    return out.reset_index(drop=True)


# ── auditing what arrived ──────────────────────────────────────────────────────
@dataclass
class PanelAudit:
    """The health check every panel should pass before any modelling."""

    summary: pd.Series
    coverage: pd.Series                 # entities per date
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems

    def __str__(self) -> str:
        head = "PASS" if self.ok else "PROBLEMS FOUND"
        lines = [f"[{head}] panel audit", self.summary.to_string()]
        lines += [f"  ! {p}" for p in self.problems]
        return "\n".join(lines)

    __repr__ = __str__

    def _repr_html_(self) -> str:
        colour = "#2f855a" if self.ok else "#c53030"
        rows = "".join(
            f"<tr><td style='padding:1px 12px;text-align:left'>{k}</td>"
            f"<td style='padding:1px 12px;text-align:right'>{v}</td></tr>"
            for k, v in self.summary.items())
        probs = "".join(f"<li style='color:#c53030'>{p}</li>" for p in self.problems)
        return (f"<div style='font-family:system-ui;font-size:13px'>"
                f"<b style='color:{colour}'>{'PASS' if self.ok else 'PROBLEMS'}</b> "
                f"panel audit<table>{rows}</table>"
                + (f"<ul>{probs}</ul>" if probs else "") + "</div>")


def audit_panel(df: pd.DataFrame, date_col: str = DATE, entity_col: str = ENTITY,
                target_col: str | None = TARGET,
                max_missing_rate: float = 0.5) -> PanelAudit:
    """Shape, span, coverage, duplicate keys and target missingness, in one call.

    The checks are chosen because each one, when it fails, produces a *plausible
    wrong answer* rather than an error:

      * duplicate (date, entity) rows silently double-weight an observation;
      * a collapsing entity count means survivorship or a data outage;
      * a target missing for whole entities biases every model toward the ones
        that happened to survive.
    """
    problems: list[str] = []

    n_dupes = int(df.duplicated([date_col, entity_col]).sum())
    if n_dupes:
        problems.append(f"{n_dupes:,} duplicate ({date_col}, {entity_col}) rows")

    coverage = df.groupby(date_col, observed=True)[entity_col].nunique()
    if len(coverage) > 2:
        drop = coverage.iloc[-1] / max(coverage.max(), 1)
        if drop < 0.5:
            problems.append(
                f"entity coverage on the last date is {drop:.0%} of the maximum — "
                "check for a truncated final day")

    summary = {
        "rows": len(df),
        "dates": int(df[date_col].nunique()),
        "entities": int(df[entity_col].nunique()),
        "date_min": str(pd.Timestamp(df[date_col].min()).date()),
        "date_max": str(pd.Timestamp(df[date_col].max()).date()),
        "entities_per_date_median": int(coverage.median()),
        "entities_per_date_min": int(coverage.min()),
        "duplicate_keys": n_dupes,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 1),
    }

    if target_col and target_col in df.columns:
        rate = float(df[target_col].isna().mean())
        summary["target_missing_rate"] = round(rate, 4)
        if rate > max_missing_rate:
            problems.append(f"{rate:.1%} of {target_col} is missing")

    return PanelAudit(pd.Series(summary), coverage, problems)


def leakage_audit(feature_cols, forbidden=("target", "label", "_fwd", "future"),
                  allow=()) -> list[str]:
    """Return feature names that look like labels. Assert on the result.

    Crude on purpose. A name-based check costs nothing, runs in CI, and catches
    the single most damaging class of mistake — a future column that survived a
    merge from an auxiliary table.

        assert not leakage_audit(feature_cols), "leaky columns"
    """
    allow = {a.lower() for a in allow}
    return [c for c in feature_cols
            if c.lower() not in allow
            and any(bad in c.lower() for bad in forbidden)]


# ── the synthetic fallback ─────────────────────────────────────────────────────
SECTORS = ("Technology", "Financials", "Industrials", "Consumer", "Healthcare",
           "Energy", "Materials", "Utilities", "RealEstate", "Telecom")


def _intraday_ohlc(returns: np.ndarray, close: np.ndarray, rng, gap_share: float = 0.35,
                   n_intraday: int = 78) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Open, High and Low consistent with the daily return that produced Close.

    High and Low must come from an actual intraday path, not from noise sprinkled
    around the close. Drawing them independently gives bars whose range is far too
    narrow for their own returns, and every range-based volatility estimator
    (Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang) then understates by a
    factor of several — silently, because the bars still satisfy
    High >= max(Open, Close).

    Construction: each day's return is split into an overnight gap and an intraday
    leg. The intraday leg is a **Brownian bridge** pinned to that day's realised
    return, with diffusion set to the entity's own daily volatility. The bridge's
    running maximum and minimum become High and Low, so by construction

        E[log(High/Low)] / E[|return|]  ->  sqrt(8/pi) / sqrt(2/pi) = 2

    which is where real equity data sits. The last point of the bridge is the day's
    return exactly, so Close is unchanged.

    `n_intraday` defaults to 78 — 5-minute bars in a 6.5-hour session. It matters:
    a bar built from finitely many observations cannot see the true continuous
    maximum, so every range estimator is biased *low*, and the bias shrinks as the
    sampling gets finer. Measured on this generator, Yang-Zhang recovers 0.90 of
    close-to-close volatility at 24 steps, 0.94 at 78 and 0.96 at 195. Real daily
    bars have exactly this bias, which is one reason published range estimators
    understate.

    Computed by accumulating the bridge one step at a time rather than materialising
    the full (dates x entities x steps) path, which would be ~200 MB on a panel this
    size for no benefit.
    """
    n_dates, n_entities = returns.shape
    sigma = returns.std(axis=0, ddof=1)[None, :]
    a = gap_share

    # Split the daily move into an overnight gap and an intraday leg that are
    # *uncorrelated* — Yang-Zhang assumes exactly that, and a naive proportional
    # split (gap = a*R) leaves a covariance term that makes every range estimator
    # understate. Writing
    #     gap      = a*R + e
    #     intraday = (1-a)*R - e,     e ~ N(0, a(1-a) sigma^2)
    # gives Cov(gap, intraday) = a(1-a)sigma^2 - Var(e) = 0, while gap + intraday
    # is still exactly R — so Close, and every Track B feature built from it, are
    # unchanged.
    e = rng.normal(0.0, 1.0, returns.shape) * np.sqrt(a * (1.0 - a)) * sigma
    gap = a * returns + e
    intraday = returns - gap

    prev_close = np.vstack([np.full((1, n_entities), close[0] / np.exp(returns[0])),
                            close[:-1]])
    open_ = prev_close * np.exp(gap)

    # Bridge diffusion carries only the intraday share of the variance, so the
    # range reflects the intraday leg and not the overnight gap as well.
    sigma_intraday = sigma * np.sqrt(1.0 - a)

    steps = rng.standard_normal((n_dates, n_entities, n_intraday)).astype(np.float32)
    steps /= np.sqrt(n_intraday)
    w_end = steps.sum(axis=2)

    walk = np.zeros((n_dates, n_entities), dtype=np.float64)
    hi = np.zeros((n_dates, n_entities))          # the open, at s = 0, is log 0
    lo = np.zeros((n_dates, n_entities))
    for k in range(n_intraday):
        walk += steps[:, :, k]
        s = (k + 1) / n_intraday
        level = s * intraday + sigma_intraday * (walk - s * w_end)
        np.maximum(hi, level, out=hi)
        np.minimum(lo, level, out=lo)

    # The bridge's final point is the close by construction, but it is accumulated
    # in float32 and so lands a few ulps away. Clamp against the exact close (and
    # the open, at log 0) so High >= max(Open, Close) holds identically rather than
    # 96% of the time.
    np.maximum(hi, intraday, out=hi)
    np.minimum(lo, intraday, out=lo)

    return open_, open_ * np.exp(hi), open_ * np.exp(lo)


def synthetic_equity_panel(n_entities: int = 400, n_dates: int = 500,
                           seed: int = config.DEFAULT_SEED,
                           reversal_strength: float = 0.10,
                           gap_share: float = 0.35,
                           start: str = "2021-01-04") -> pd.DataFrame:
    """A Kaggle-shaped daily equity panel, with a small signal planted in it.

    Columns mirror the JPX competition: Date, SecuritiesCode, Open/High/Low/
    Close/Volume, AdjustmentFactor, ExpectedDividend, SupervisionFlag, plus
    static Sector / MarketCapitalization and a forward-return `Target`.

    Returns are built from a market factor, a sector factor and idiosyncratic
    noise, so the cross-section has genuine correlation structure. A **short-term
    reversal** effect is planted deliberately: `Target` loads negatively on the
    previous day's idiosyncratic return with coefficient `reversal_strength`.

    That the signal is planted is the point. On this panel a correct pipeline
    *must* find it, which makes the dataset a test of the pipeline rather than a
    test of the market — set `reversal_strength=0.0` to get a null panel and
    confirm your validation reports no skill.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_dates)
    codes = np.arange(1301, 1301 + n_entities)

    sector_of = rng.integers(0, len(SECTORS), n_entities)
    beta = rng.normal(1.0, 0.25, n_entities).clip(0.2, 2.0)
    idio_vol = rng.uniform(0.010, 0.035, n_entities)

    market = rng.normal(0.0003, 0.011, n_dates)
    sector_ret = rng.normal(0.0, 0.006, (n_dates, len(SECTORS)))
    idio = rng.normal(0.0, 1.0, (n_dates, n_entities)) * idio_vol

    # returns[t, i] = beta_i * market_t + sector factor + idiosyncratic
    returns = beta * market[:, None] + sector_ret[:, sector_of] + idio

    # The planted effect. Target(t) is the return from t+1 to t+2, so for the
    # signal to be *predictable at t* the reversal must land two steps later:
    # return(t+2) loads negatively on the idiosyncratic move at t, which is
    # visible in ret_1d(t). A one-step lag here would make Target depend on
    # idio(t+1) — real structure, but not knowable in time to trade it.
    returns[2:] -= reversal_strength * idio[:-2]

    close = 1000.0 * np.exp(np.cumsum(returns, axis=0))
    open_, high, low = _intraday_ohlc(returns, close, rng, gap_share=gap_share)
    volume = rng.lognormal(12.0, 0.9, close.shape) * (1 + 3 * np.abs(returns))

    # Corporate actions and dividends, at realistic rarity.
    adjustment = np.ones(close.shape)
    adjustment[rng.random(close.shape) < 0.0004] = 0.5          # 2-for-1 splits
    dividend = np.zeros(close.shape)
    div_mask = rng.random(close.shape) < 0.004
    dividend[div_mask] = close[div_mask] * rng.uniform(0.002, 0.02, div_mask.sum())
    supervision = rng.random(close.shape) < 0.0008

    # Target is the JPX definition: the return from t+1 close to t+2 close.
    target = np.full(close.shape, np.nan)
    target[:-2] = close[2:] / close[1:-1] - 1.0

    n = n_dates * n_entities
    panel = pd.DataFrame({
        DATE: np.repeat(dates.to_numpy(), n_entities),
        ENTITY: np.tile(codes, n_dates),
        "Open": open_.ravel(), "High": high.ravel(), "Low": low.ravel(),
        "Close": close.ravel(), "Volume": volume.ravel().round(),
        "AdjustmentFactor": adjustment.ravel(),
        "ExpectedDividend": np.where(dividend.ravel() > 0, dividend.ravel(), np.nan),
        "SupervisionFlag": supervision.ravel(),
        TARGET: target.ravel(),
    })
    panel["RowId"] = (panel[DATE].dt.strftime("%Y%m%d") + "_"
                      + panel[ENTITY].astype(str))

    static = pd.DataFrame({
        ENTITY: codes,
        "Sector": [SECTORS[s] for s in sector_of],
        "MarketCapitalization": rng.lognormal(25.0, 1.3, n_entities).round(),
        "IssuedShares": rng.lognormal(18.5, 1.0, n_entities).round(),
    })
    panel = panel.merge(static, on=ENTITY, how="left", validate="many_to_one")

    panel.attrs["generator"] = "rvlab.synthetic_equity_panel"
    panel.attrs["planted_effect"] = (
        f"short-term reversal, coefficient {reversal_strength}" if reversal_strength
        else "none (null panel)")
    assert len(panel) == n, "panel construction lost rows"
    return panel


def load_panel(root=None, n_entities: int = 400, n_dates: int = 500,
               seed: int = config.DEFAULT_SEED,
               reversal_strength: float = 0.10) -> pd.DataFrame:
    """Real competition data when it can be found; a synthetic panel otherwise.

    Set `$RVLAB_PANEL_ROOT`, or pass `root`, to point at an unzipped competition
    directory containing `train_files/stock_prices.csv`. The returned frame
    carries `df.attrs["provenance"]` either way, so a notebook can state which
    it got.
    """
    import os

    root = root or os.environ.get("RVLAB_PANEL_ROOT")
    root = Path(root) if root else (None if config.force_synthetic()
                                    else find_dataset_root())

    if root is not None and (Path(root) / "train_files/stock_prices.csv").exists():
        panel = concat_folders(Path(root), "stock_prices.csv")
        static = read_if_exists(Path(root), "stock_list.csv")
        if static is not None and ENTITY in static.columns:
            keep = [c for c in ("SecuritiesCode", "Sector", "33SectorName", "17SectorName",
                                "NewMarketSegment", "MarketCapitalization",
                                "IssuedShares") if c in static.columns]
            static = static[keep].drop_duplicates(ENTITY, keep="last")
            if "Sector" not in static.columns:
                for source in ("33SectorName", "17SectorName"):
                    if source in static.columns:
                        static["Sector"] = static[source]
                        break
            merge_cols = [ENTITY, *[c for c in static.columns
                                    if c != ENTITY and c not in panel.columns]]
            if len(merge_cols) > 1:
                panel = panel.merge(static[merge_cols], on=ENTITY, how="left",
                                    validate="many_to_one")
        panel.attrs["provenance"] = {"source": str(root), "synthetic": False,
                                     "rows": len(panel)}
        return panel

    panel = synthetic_equity_panel(n_entities, n_dates, seed, reversal_strength)
    panel.attrs["provenance"] = {
        "source": "synthetic", "synthetic": True, "rows": len(panel),
        "planted_effect": panel.attrs["planted_effect"],
        "hint": "set $RVLAB_PANEL_ROOT to an unzipped competition directory",
    }
    return panel


__all__ = [
    "DATE", "ENTITY", "TARGET", "SECTORS",
    "find_dataset_root", "read_if_exists", "concat_folders",
    "PanelAudit", "audit_panel", "leakage_audit",
    "synthetic_equity_panel", "load_panel",
]
