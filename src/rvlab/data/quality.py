"""
rvlab.data.quality
==================
The health check to run before any modelling, and again after any pipeline that
reshapes data.

The checks here were chosen on one criterion: **each one, when it fails, produces
a plausible wrong answer rather than an error.** A duplicated key silently
double-weights an observation. A constant column costs a feature slot and teaches
nothing. A train/test distribution shift makes a validated model fail in
production with no warning at all. None of these raise; all of them change your
conclusions.

    report = health_report(df)          # per-column, with flags
    report.problems                     # just the ones that matter

Drift and leakage get their own functions because they need two inputs — a
reference frame, or a target — and because they are the two most expensive
mistakes to find late.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ── frame-level ────────────────────────────────────────────────────────────────
@dataclass
class HealthReport:
    """Per-column diagnostics plus the frame-level problems worth acting on."""

    columns: pd.DataFrame
    summary: pd.Series
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def __str__(self) -> str:
        head = "PASS" if self.ok else f"{len(self.problems)} PROBLEMS"
        return "\n".join([f"[{head}] health report", self.summary.to_string(),
                          *(f"  ! {p}" for p in self.problems)])

    __repr__ = __str__

    def _repr_html_(self) -> str:
        colour = "#2f855a" if self.ok else "#c53030"
        rows = "".join(f"<tr><td style='padding:1px 12px;text-align:left'>{k}</td>"
                       f"<td style='padding:1px 12px;text-align:right'>{v}</td></tr>"
                       for k, v in self.summary.items())
        probs = "".join(f"<li style='color:#c53030'>{p}</li>" for p in self.problems)
        return (f"<div style='font-family:system-ui;font-size:13px'>"
                f"<b style='color:{colour}'>{'PASS' if self.ok else 'PROBLEMS'}</b>"
                f" health report<table>{rows}</table>"
                + (f"<ul>{probs}</ul>" if probs else "") + "</div>")


def health_report(df: pd.DataFrame, key_cols=None, max_null_rate: float = 0.5,
                  high_cardinality: float = 0.9) -> HealthReport:
    """One-call audit: dtypes, nulls, duplicates, constants, cardinality, memory.

    `key_cols` names the columns that should uniquely identify a row; duplicates
    on them are reported as a problem rather than a curiosity.
    """
    problems: list[str] = []
    n = len(df)

    rows = []
    for col in df.columns:
        s = df[col]
        n_null = int(s.isna().sum())
        n_unique = int(s.nunique(dropna=True))
        non_null = s.dropna()
        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "null_rate": round(n_null / n, 4) if n else np.nan,
            "n_unique": n_unique,
            "unique_rate": round(n_unique / n, 4) if n else np.nan,
            "constant": n_unique <= 1,
            "memory_mb": round(s.memory_usage(deep=True) / 1e6, 3),
            "sample": str(non_null.iloc[0])[:28] if len(non_null) else "",
        })
    columns = pd.DataFrame(rows)

    constants = columns.loc[columns["constant"], "column"].tolist()
    if constants:
        problems.append(f"{len(constants)} constant column(s): {constants[:6]}")

    all_null = columns.loc[columns["null_rate"] == 1.0, "column"].tolist()
    if all_null:
        problems.append(f"{len(all_null)} all-null column(s): {all_null[:6]}")

    very_null = columns.loc[columns["null_rate"].between(max_null_rate, 1.0, "left"),
                            "column"].tolist()
    if very_null:
        problems.append(f"{len(very_null)} column(s) over {max_null_rate:.0%} null: "
                        f"{very_null[:6]}")

    n_dupe_rows = int(df.duplicated().sum())
    if n_dupe_rows:
        problems.append(f"{n_dupe_rows:,} fully duplicated rows")

    if key_cols:
        present = [c for c in key_cols if c in df.columns]
        if len(present) == len(key_cols):
            n_dupe_keys = int(df.duplicated(present).sum())
            if n_dupe_keys:
                problems.append(f"{n_dupe_keys:,} duplicate rows on key {list(key_cols)}")

    declared_keys = set(key_cols or ())
    object_cols = columns[(columns["dtype"] == "object")
                          & (columns["unique_rate"] > high_cardinality)
                          & ~columns["column"].isin(declared_keys)]
    if len(object_cols):
        problems.append(f"{len(object_cols)} near-unique object column(s) — likely ids, "
                        f"not features: {object_cols['column'].tolist()[:4]}")

    summary = pd.Series({
        "rows": n,
        "columns": df.shape[1],
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "duplicate_rows": n_dupe_rows,
        "constant_columns": len(constants),
        "columns_with_nulls": int((columns["null_rate"] > 0).sum()),
        "total_null_rate": round(float(df.isna().to_numpy().mean()), 4) if n else np.nan,
    })
    return HealthReport(columns=columns, summary=summary, problems=problems)


# ── numeric ────────────────────────────────────────────────────────────────────
def numeric_report(df: pd.DataFrame) -> pd.DataFrame:
    """Range, zeros, negatives and non-finite counts for every numeric column.

    The three columns people skip and then regret: `n_inf` (an infinity propagates
    through every aggregate and turns a mean into NaN), `n_zero` (a spike at zero
    usually means a fill, not a measurement) and `n_negative` on a quantity that
    cannot be negative.
    """
    rows = []
    for col in df.select_dtypes("number").columns:
        v = df[col].to_numpy(dtype=float)
        finite = v[np.isfinite(v)]
        rows.append({
            "column": col,
            "min": finite.min() if finite.size else np.nan,
            "p01": np.percentile(finite, 1) if finite.size else np.nan,
            "median": np.median(finite) if finite.size else np.nan,
            "p99": np.percentile(finite, 99) if finite.size else np.nan,
            "max": finite.max() if finite.size else np.nan,
            "mean": finite.mean() if finite.size else np.nan,
            "std": finite.std(ddof=1) if finite.size > 1 else np.nan,
            "n_zero": int((finite == 0).sum()),
            "n_negative": int((finite < 0).sum()),
            "n_inf": int(np.isinf(v).sum()),
            "skew": float(pd.Series(finite).skew()) if finite.size > 2 else np.nan,
        })
    return pd.DataFrame(rows)


def outlier_report(df: pd.DataFrame, columns=None, z_threshold: float = 4.0,
                   iqr_multiple: float = 3.0, mad_threshold: float = 5.0) -> pd.DataFrame:
    """Count outliers three ways, because the three disagree and the gap is the point.

    * **z-score** uses the mean and standard deviation, both of which the outliers
      themselves inflate — so a few extreme points hide each other. It finds the
      fewest.
    * **IQR** uses quartiles and is unaffected by the tails.
    * **MAD** (median absolute deviation, scaled by 1.4826) is the most robust and
      usually flags the most.

    A column where z-score finds 2 and MAD finds 400 has heavy tails, and that is
    a modelling decision — not a cleaning one.
    """
    rows = []
    for col in (columns or df.select_dtypes("number").columns):
        v = df[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size < 3:
            continue
        mean, std = v.mean(), v.std(ddof=1)
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        median = np.median(v)
        mad = np.median(np.abs(v - median)) * 1.4826

        rows.append({
            "column": col,
            "n": v.size,
            "by_zscore": int((np.abs(v - mean) > z_threshold * std).sum()) if std else 0,
            "by_iqr": int(((v < q1 - iqr_multiple * iqr)
                           | (v > q3 + iqr_multiple * iqr)).sum()) if iqr else 0,
            "by_mad": int((np.abs(v - median) > mad_threshold * mad).sum()) if mad else 0,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["disagreement"] = out[["by_zscore", "by_iqr", "by_mad"]].max(axis=1) - \
                              out[["by_zscore", "by_iqr", "by_mad"]].min(axis=1)
        out = out.sort_values("disagreement", ascending=False).reset_index(drop=True)
    return out


# ── datetime ───────────────────────────────────────────────────────────────────
def datetime_audit(df: pd.DataFrame, column: str, expect_freq: str | None = None
                   ) -> pd.Series:
    """Span, timezone, monotonicity, duplicates and the largest gaps.

    `largest_gap` is the diagnostic that matters. On daily data the modal gap is
    one day and the large ones are weekends and holidays — but a gap of 40 days
    is a data outage, and it will quietly become a 40-day-wide rolling window.

    Gaps are measured between **distinct** timestamps. On a panel the same instant
    appears once per entity, and differencing the raw column would report a modal
    gap of zero — true, and useless. `n_duplicated` still reports the repetition,
    which is what tells you it is a panel in the first place.
    """
    s = pd.to_datetime(df[column])
    distinct = pd.Series(s.dropna().unique()).sort_values()
    deltas = distinct.diff().dropna()
    modal = deltas.mode()

    out = {
        "n": len(s),
        "n_unique": int(s.nunique()),
        "n_null": int(s.isna().sum()),
        "tz": str(s.dt.tz) if hasattr(s.dt, "tz") and s.dt.tz is not None else "naive",
        "min": s.min(),
        "max": s.max(),
        "monotonic_increasing": bool(s.is_monotonic_increasing),
        "n_duplicated": int(s.duplicated().sum()),
        "modal_gap": modal.iloc[0] if len(modal) else pd.NaT,
        "largest_gap": deltas.max() if len(deltas) else pd.NaT,
        "n_gaps_over_3x_modal": int((deltas > 3 * modal.iloc[0]).sum())
        if len(modal) and modal.iloc[0] > pd.Timedelta(0) else 0,
    }
    if expect_freq:
        expected = pd.date_range(s.min(), s.max(), freq=expect_freq)
        out["missing_vs_expected"] = int(len(set(expected) - set(distinct)))
    return pd.Series(out)


# ── categorical ────────────────────────────────────────────────────────────────
def categorical_report(df: pd.DataFrame, columns=None, rare_threshold: float = 0.01
                       ) -> pd.DataFrame:
    """Cardinality, the dominant level, and how much mass sits in rare levels.

    `rare_share` is what decides whether one-hot encoding is safe. A column with
    2,000 levels where 40% of rows sit in levels seen fewer than 1% of the time
    will produce hundreds of near-empty dummies and a category the test set has
    never seen.
    """
    columns = columns or df.select_dtypes(include=["object", "category", "bool"]).columns
    rows = []
    for col in columns:
        s = df[col].dropna()
        if not len(s):
            continue
        counts = s.value_counts(normalize=True)
        rows.append({
            "column": col,
            "n_unique": int(s.nunique()),
            "top_level": str(counts.index[0])[:24],
            "top_share": round(float(counts.iloc[0]), 4),
            "rare_levels": int((counts < rare_threshold).sum()),
            "rare_share": round(float(counts[counts < rare_threshold].sum()), 4),
        })
    return pd.DataFrame(rows)


# ── drift ──────────────────────────────────────────────────────────────────────
def population_stability_index(expected, actual, bins: int = 10) -> float:
    """PSI between two samples. <0.1 stable, 0.1-0.25 moderate, >0.25 a real shift.

    Binned on the *reference* sample's quantiles so the comparison is like for
    like, with a small floor on each bin so an empty bin does not send the
    logarithm to infinity.
    """
    expected = np.asarray(expected, float)
    actual = np.asarray(actual, float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if expected.size < bins or actual.size < bins:
        return np.nan

    edges = np.unique(np.percentile(expected, np.linspace(0, 100, bins + 1)))
    if edges.size < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    e = np.histogram(expected, edges)[0] / expected.size
    a = np.histogram(actual, edges)[0] / actual.size
    floor = 1e-6
    e, a = np.maximum(e, floor), np.maximum(a, floor)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, columns=None,
                 psi_warn: float = 0.25, ks_alpha: float = 0.01) -> pd.DataFrame:
    """Per-column distribution shift between two frames, by PSI and the KS test.

    The canonical use is train versus test, or an old month versus a new one. Two
    measures because they see different things: KS is sensitive to a shift
    anywhere in the distribution and, on large samples, flags differences too
    small to matter; PSI is bucketed and reads as an effect size. Trust the pair.

    > A column that drifts is not automatically a column to drop. It is a column
    > whose relationship with the target you should re-check on recent data.
    """
    from scipy import stats

    columns = columns or [c for c in reference.select_dtypes("number").columns
                          if c in current.columns]
    rows = []
    for col in columns:
        a = reference[col].to_numpy(float)
        b = current[col].to_numpy(float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if a.size < 20 or b.size < 20:
            continue
        ks = stats.ks_2samp(a, b)
        psi = population_stability_index(a, b)
        rows.append({
            "column": col,
            "ref_mean": a.mean(), "cur_mean": b.mean(),
            "mean_shift_in_sd": (b.mean() - a.mean()) / a.std(ddof=1) if a.std(ddof=1) else np.nan,
            "sd_ratio": b.std(ddof=1) / a.std(ddof=1) if a.std(ddof=1) else np.nan,
            "ks_stat": ks.statistic, "ks_pvalue": ks.pvalue,
            "psi": psi,
            "drifted": bool(psi > psi_warn or ks.pvalue < ks_alpha),
        })
    out = pd.DataFrame(rows)
    return out.sort_values("psi", ascending=False).reset_index(drop=True) if len(out) else out


# ── leakage ────────────────────────────────────────────────────────────────────
def leakage_scan(df: pd.DataFrame, target: str, feature_cols=None,
                 name_patterns=("target", "label", "_fwd", "future", "next_", "outcome"),
                 corr_warn: float = 0.95,
                 purity_max_unique_rate: float = 0.5) -> pd.DataFrame:
    """Cheap leakage signatures over the candidate features.

    * **name** — a column whose name says it is a label. Crude, free, and catches
      the common case: a future column that survived a merge.
    * **correlation / monotone** — absolute Pearson or Spearman correlation with
      the target above `corr_warn`. Spearman catches monotone numeric aliases that
      Pearson can miss; neither is a general single-feature model.
    * **deterministic mapping** — every observed feature value maps to exactly one
      target value. This catches string labels copied or renamed into a feature,
      as well as numeric recodings of non-numeric labels.

    Conditional purity is trivially 100% for a row identifier, so it is only
    evaluated when ``n_unique / n_complete <= purity_max_unique_rate``. The
    default therefore requires at least two complete observations per level on
    average; continuous numeric columns and near-unique IDs are therefore not
    declared pure merely because their observed values identify individual rows.
    This remains a heuristic: a leak with an innocent name and a modest association
    — yesterday's label merged on the wrong date — still needs a time-aware audit,
    not a scan.
    """
    if not 0 < purity_max_unique_rate <= 1:
        raise ValueError("purity_max_unique_rate must lie in (0, 1]")

    feature_cols = feature_cols or [c for c in df.columns if c != target]
    y = df[target]
    y_num = pd.to_numeric(y, errors="coerce")

    rows = []
    for col in feature_cols:
        flags = []
        low = col.lower()
        if any(p in low for p in name_patterns):
            flags.append("name")

        corr = np.nan
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() > 10 and y_num.notna().sum() > 10:
            ok = s.notna() & y_num.notna()
            if ok.sum() > 10 and s[ok].nunique() > 1 and y_num[ok].nunique() > 1:
                corr = float(np.corrcoef(s[ok], y_num[ok])[0, 1])
                if abs(corr) >= corr_warn:
                    flags.append("correlation")
                spearman = float(pd.Series(s[ok]).corr(y_num[ok], method="spearman"))
                if abs(spearman) >= corr_warn and "correlation" not in flags:
                    flags.append("monotone")

        pairs = pd.DataFrame({"_feature": df[col], "_target": y}).dropna()
        n_complete = len(pairs)
        n_feature_levels = int(pairs["_feature"].nunique())
        n_target_levels = int(pairs["_target"].nunique())
        repeated_levels = (
            n_complete > 10
            and 1 < n_feature_levels
            and 1 < n_target_levels
            and n_feature_levels / n_complete <= purity_max_unique_rate
        )
        if repeated_levels:
            joint_counts = pairs.groupby(
                ["_feature", "_target"], observed=True, sort=False
            ).size()
            correctly_mapped = int(
                joint_counts.groupby(level=0, observed=True, sort=False).max().sum()
            )
            if correctly_mapped == n_complete:
                flags.append("deterministic_mapping")

        if flags:
            rows.append({"column": col, "flags": "+".join(flags),
                         "abs_corr": abs(corr) if np.isfinite(corr) else np.nan})

    out = pd.DataFrame(rows, columns=["column", "flags", "abs_corr"])
    return out.sort_values("abs_corr", ascending=False, na_position="last").reset_index(drop=True)


def target_audit(df: pd.DataFrame, target: str, group: str | None = None) -> pd.Series:
    """Missingness, balance and tail weight of the label itself.

    Worth its own function because a target problem invalidates everything, and
    because missingness *by group* is the version that matters: a label missing
    at random costs you rows, while a label missing for whole entities biases the
    model toward whichever entities happened to survive.
    """
    y = df[target]
    out = {
        "n": len(y),
        "n_missing": int(y.isna().sum()),
        "missing_rate": round(float(y.isna().mean()), 4),
        "n_unique": int(y.nunique(dropna=True)),
    }
    numeric = pd.to_numeric(y, errors="coerce").dropna()
    if len(numeric) and out["n_unique"] > 10:
        out |= {"mean": float(numeric.mean()), "std": float(numeric.std(ddof=1)),
                "skew": float(numeric.skew()),
                "p99_over_sd": float(numeric.quantile(0.99) / numeric.std(ddof=1))
                if numeric.std(ddof=1) else np.nan}
    elif out["n_unique"] <= 10:
        counts = y.value_counts(normalize=True, dropna=True)
        out |= {"class_balance": ", ".join(f"{k}:{v:.1%}" for k, v in counts.head(5).items()),
                "minority_share": round(float(counts.min()), 4)}

    if group and group in df.columns:
        by_group = df.groupby(group, observed=True)[target].apply(lambda s: s.isna().mean())
        out |= {"groups": int(by_group.size),
                "groups_fully_missing": int((by_group == 1.0).sum()),
                "worst_group_missing_rate": round(float(by_group.max()), 4)}
    return pd.Series(out)


__all__ = [
    "HealthReport", "health_report", "numeric_report", "outlier_report",
    "datetime_audit", "categorical_report", "population_stability_index",
    "drift_report", "leakage_scan", "target_audit",
]
