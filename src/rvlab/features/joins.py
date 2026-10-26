"""
rvlab.features.joins
====================
Combining datasets, and knowing you did it right.

A join is the operation most likely to silently corrupt an analysis, because the
three ways it fails all produce a frame that looks fine:

  * **it explodes** — duplicate keys on the right side multiply rows, and every
    subsequent mean is weighted wrong;
  * **it evaporates** — a dtype mismatch (`datetime64[ns, UTC]` against a string,
    `int64` against `object`) matches nothing and returns an empty frame, or worse,
    a mostly-empty one;
  * **it reaches forward** — an as-of join with the wrong `direction` matches each
    row to data recorded *after* it, which is a leak that improves your score.

The remedy in every case is the same and is what this module enforces: state what
you expect the join to do, then check it. `merge_report` before, `safe_merge`
during, `reconcile` after.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── before: will this join do what I think? ────────────────────────────────────
def merge_report(left: pd.DataFrame, right: pd.DataFrame, on, how: str = "inner"
                 ) -> pd.DataFrame:
    """Diagnose a join *before* running it: dtypes, key overlap, duplication.

    The row you look at first is `predicted_rows`. If it exceeds
    `max(len(left), len(right))`, the join will fan out and you almost certainly
    meant to aggregate one side first.
    """
    keys = [on] if isinstance(on, str) else list(on)
    rows = []
    for key in keys:
        in_left, in_right = key in left.columns, key in right.columns
        entry = {"key": key, "in_left": in_left, "in_right": in_right,
                 "left_dtype": str(left[key].dtype) if in_left else "-",
                 "right_dtype": str(right[key].dtype) if in_right else "-"}
        if in_left and in_right:
            lv, rv = set(left[key].dropna()), set(right[key].dropna())
            entry |= {
                "dtype_match": str(left[key].dtype) == str(right[key].dtype),
                "left_unique": len(lv), "right_unique": len(rv),
                "shared": len(lv & rv),
                "left_only": len(lv - rv), "right_only": len(rv - lv),
            }
        rows.append(entry)
    report = pd.DataFrame(rows)

    dup_left = int(left.duplicated([k for k in keys if k in left.columns]).sum())
    dup_right = int(right.duplicated([k for k in keys if k in right.columns]).sum())

    usable = [k for k in keys if k in left.columns and k in right.columns]
    # Only predict when the dtypes actually match — attempting the merge otherwise
    # raises, and this function exists precisely to diagnose that case rather than
    # to fall over on it.
    compatible = usable and all(str(left[k].dtype) == str(right[k].dtype) for k in usable)

    if compatible:
        counts_right = right.groupby(usable, observed=True).size()
        predicted = int(left.merge(counts_right.rename("_n").reset_index(),
                                   on=usable, how="left")["_n"].fillna(
                            0 if how == "inner" else 1).sum())
    else:
        predicted = np.nan

    report.attrs["summary"] = pd.Series({
        "how": how, "left_rows": len(left), "right_rows": len(right),
        "duplicate_keys_left": dup_left, "duplicate_keys_right": dup_right,
        "predicted_rows": predicted,
        # Measured against the LEFT frame: an enrichment join should return one row
        # per left row, and anything more is the right side duplicating them.
        "fans_out": bool(predicted > len(left)) if compatible else False,
        "dtype_mismatch": not compatible and bool(usable),
    })
    return report


def safe_merge(left: pd.DataFrame, right: pd.DataFrame, on, how: str = "left",
               validate: str | None = None, expect: str = "same_or_fewer",
               **kwargs) -> pd.DataFrame:
    """`pd.merge` that refuses to change the row count in a way you did not ask for.

    `expect` is the contract:
      * ``"same"``           — the row count must not change at all
      * ``"same_or_fewer"``  — a left join that may drop nothing, an inner that may
        drop (the default, and the right one for enrichment joins)
      * ``"any"``            — a fan-out is intended; say so explicitly

    `validate` is passed to pandas (`"1:1"`, `"m:1"`, `"1:m"`) and is the stronger
    check when you can state the cardinality. Use both: `validate` catches a
    duplicated key, `expect` catches the row-count consequence.
    """
    before = len(left)
    merged = left.merge(right, on=on, how=how, validate=validate, **kwargs)
    after = len(merged)

    if expect == "same" and after != before:
        raise ValueError(f"merge changed row count {before:,} -> {after:,}; "
                         "pass expect='any' if that is intended")
    if expect == "same_or_fewer" and after > before:
        raise ValueError(
            f"merge fanned out {before:,} -> {after:,} rows — the right frame has "
            f"duplicate keys on {on}. Aggregate it first, or pass expect='any'.")

    merged.attrs["merge"] = {"before": before, "after": after, "how": how, "on": on}
    return merged


def reconcile(before: pd.DataFrame, after: pd.DataFrame, name: str = "step",
              key=None) -> pd.Series:
    """Row and key accounting across any transformation. Print it; do not skip it.

    Chains of merges are where row counts drift without anyone noticing. One line
    per step, checked as you go, costs nothing and localises the damage.
    """
    out = {"step": name, "rows_before": len(before), "rows_after": len(after),
           "row_delta": len(after) - len(before)}
    if key:
        keys = [key] if isinstance(key, str) else list(key)
        if all(k in before.columns for k in keys) and all(k in after.columns for k in keys):
            kb = set(map(tuple, before[keys].dropna().to_numpy()))
            ka = set(map(tuple, after[keys].dropna().to_numpy()))
            out |= {"keys_before": len(kb), "keys_after": len(ka), "keys_lost": len(kb - ka)}
    return pd.Series(out)


# ── as-of and interval joins ───────────────────────────────────────────────────
def asof_join(left: pd.DataFrame, right: pd.DataFrame, on: str, by=None,
              tolerance=None, direction: str = "backward",
              suffixes=("", "_right")) -> pd.DataFrame:
    """Match each left row to the most recent right row at or before it.

    The join for combining series on different clocks — a 1-minute option panel
    against a daily macro series, a trade against the prevailing quote.

    > ⚠️ **Look-ahead:** `direction` defaults to `"backward"` here and *must* stay
    > that way for anything predictive. `"forward"` and `"nearest"` will happily
    > match a row to data recorded after it. That is a leak, it improves your
    > validation score, and nothing warns you.

    `tolerance` is the other half. Without it, a gap in the right frame silently
    carries a stale value forward — potentially for months.
    """
    if direction not in ("backward", "forward", "nearest"):
        raise ValueError(f"direction must be backward/forward/nearest, got {direction!r}")

    by_cols = [by] if isinstance(by, str) else (list(by) if by else None)
    sort_cols = ([*by_cols, on] if by_cols else [on])
    left_sorted = left.sort_values(on, kind="stable")
    right_sorted = right.sort_values(on, kind="stable")

    merged = pd.merge_asof(left_sorted, right_sorted, on=on, by=by_cols,
                           tolerance=tolerance, direction=direction, suffixes=suffixes)

    matched_col = [c for c in right.columns if c not in (by_cols or []) and c != on]
    match_rate = (merged[matched_col[0]].notna().mean() if matched_col else np.nan)
    merged.attrs["asof"] = {"direction": direction, "tolerance": str(tolerance),
                            "match_rate": round(float(match_rate), 4)
                            if np.isfinite(match_rate) else None,
                            "sorted_on": sort_cols}
    return merged


def interval_join(left: pd.DataFrame, right: pd.DataFrame, on: str,
                  left_time: str, start: str, end: str,
                  closed: str = "both", end_na_is_open: bool = True) -> pd.DataFrame:
    """Join where the left row's time falls inside a right row's validity window.

    The shape of every *link table*: an identifier mapping that is only valid
    between two dates. A security changes issuer, a customer changes segment, a
    CRSP `permno` maps to a Compustat `gvkey` for a period. Joining on the id
    alone attaches every historical mapping to every row.

    `end_na_is_open=True` treats a missing end date as "still valid", which is how
    link tables encode a live record — and forgetting it silently drops every
    current mapping.
    """
    merged = left.merge(right, on=on, how="left", suffixes=("", "_link"))

    t = pd.to_datetime(merged[left_time])
    lo = pd.to_datetime(merged[start])
    hi = pd.to_datetime(merged[end])
    if end_na_is_open:
        hi = hi.fillna(pd.Timestamp.max.tz_localize(t.dt.tz) if t.dt.tz else pd.Timestamp.max)

    after = t >= lo if closed in ("both", "left") else t > lo
    before = t <= hi if closed in ("both", "right") else t < hi
    keep = merged[(after & before) | lo.isna()]

    out = keep.reset_index(drop=True)
    out.attrs["interval_join"] = {
        "left_rows": len(left), "after_id_merge": len(merged), "after_window": len(out),
        "unmatched": int(len(left) - out[on].notna().sum()) if on in out else None,
    }
    return out


# ── aggregate and join ─────────────────────────────────────────────────────────
def aggregate_and_join(left: pd.DataFrame, right: pd.DataFrame, group_keys,
                       aggs: dict, on=None, prefix: str = "", how: str = "left"
                       ) -> pd.DataFrame:
    """Summarise the right frame to one row per key, then join it on safely.

    The fix for a fan-out. When the right frame has many rows per key, you almost
    never want the cross product — you want a summary. Doing it in one call keeps
    the aggregation and the join together, so the `validate="m:1"` below is always
    true by construction.

        aggregate_and_join(daily, ticks, "date",
                           {"price": ["mean", "std"], "size": "sum"}, prefix="tick_")
    """
    keys = [group_keys] if isinstance(group_keys, str) else list(group_keys)
    summary = right.groupby(keys, observed=True).agg(aggs)
    summary.columns = [f"{prefix}{col}_{stat}" if isinstance(col, tuple) is False else
                       f"{prefix}{col[0]}_{col[1]}"
                       for col in summary.columns.to_flat_index()]
    summary = summary.reset_index()
    return safe_merge(left, summary, on=on or keys, how=how,
                      validate="m:1", expect="same_or_fewer")


# ── reshaping ──────────────────────────────────────────────────────────────────
def flatten_multiindex(df: pd.DataFrame, sep: str = "_") -> pd.DataFrame:
    """Turn a MultiIndex column axis into flat strings.

    Any `groupby().agg()` with several statistics produces one, and almost every
    downstream tool — `ColumnTransformer`, `to_csv`, plotting — is happier without
    it.
    """
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [sep.join(str(p) for p in col if p not in ("", None))
                       for col in out.columns.to_flat_index()]
    return out


def dates_in_columns_to_long(df: pd.DataFrame, id_vars, date_pattern=r"^\d{4}-\d{2}-\d{2}$",
                             var_name: str = "date", value_name: str = "value"
                             ) -> pd.DataFrame:
    """Melt a frame whose *columns are dates* into tidy long form.

    The single most common untidy shape in delivered data — one column per period,
    which makes every time operation impossible until it is fixed. Detects the
    date-like columns by pattern so you do not have to list them.
    """
    import re

    date_cols = [c for c in df.columns if re.match(date_pattern, str(c))]
    if not date_cols:
        raise ValueError(f"no columns matched {date_pattern!r}; got {list(df.columns)[:8]}")
    ids = [id_vars] if isinstance(id_vars, str) else list(id_vars)
    out = df.melt(id_vars=ids, value_vars=date_cols,
                  var_name=var_name, value_name=value_name)
    out[var_name] = pd.to_datetime(out[var_name])
    return out.sort_values([*ids, var_name]).reset_index(drop=True)


def long_to_wide(df: pd.DataFrame, index, columns, values,
                 aggfunc="mean", flatten: bool = True) -> pd.DataFrame:
    """Long → wide via `pivot_table`, flattened and index-reset by default.

    `pivot_table` over `pivot` on purpose: `pivot` raises on duplicate
    index/column pairs, and real data has them. Passing `aggfunc` forces you to
    decide what a duplicate means instead of discovering it as an exception.
    """
    out = df.pivot_table(index=index, columns=columns, values=values, aggfunc=aggfunc)
    if flatten:
        out = flatten_multiindex(out).reset_index()
        out.columns.name = None
    return out


__all__ = [
    "merge_report", "safe_merge", "reconcile", "asof_join", "interval_join",
    "aggregate_and_join", "flatten_multiindex", "dates_in_columns_to_long",
    "long_to_wide",
]
