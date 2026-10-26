"""
rvlab.evaluate.sweeps
=====================
Run a function over a parameter grid, in parallel, with results cached to disk.

The pattern this replaces is the one in
`demo/python/research/shared/robustness_sweeps.py`: nested loops over H,
resampling frequency and move threshold, each writing a timestamped CSV. That
works but is hard to read and impossible to re-enter — rerunning costs the full
compute again.

`run_grid` gives the same sweep as one call returning a tidy frame, memoised by
`joblib.Memory`, so a notebook re-executes in seconds after the first run.

> ⚠️ **Multiplicity:** a grid with 30 cells is 30 hypothesis tests. Pass the
> resulting p-values through `rvlab.evaluate.tests.benjamini_hochberg` before
> calling any cell a result. `pass_rate` below is descriptive, not inferential.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed

from ..config import CACHE_DIR


def _memory(enabled: bool) -> Memory:
    """A joblib cache rooted at notebooks/_output/cache, or a no-op cache."""
    if not enabled:
        return Memory(location=None, verbose=0)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return Memory(location=str(CACHE_DIR), verbose=0)


def grid_points(grid: Mapping[str, Sequence]) -> list[dict]:
    """Cartesian product of a parameter grid, as a list of kwargs dicts.

    >>> grid_points({"hurst": [0.05, 0.1], "step_min": [1, 5]})
    [{'hurst': 0.05, 'step_min': 1}, ..., {'hurst': 0.1, 'step_min': 5}]
    """
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def run_grid(fn: Callable[..., Mapping], grid: Mapping[str, Sequence],
             n_jobs: int = -1, cache: bool = True, verbose: int = 0,
             **fixed) -> pd.DataFrame:
    """Evaluate `fn(**point, **fixed)` at every grid point; return a tidy frame.

    `fn` must return a mapping of scalar results (or None to skip the cell). The
    parameters are prepended as columns, so the output is one row per cell with
    the settings and the outcome side by side — directly plottable as a heatmap
    via `rvlab.plotting.charts.sweep_heatmap`.

    Results are cached on the *arguments*, so changing the grid only computes
    the new cells. Clear with `rvlab.evaluate.sweeps.clear_cache()`.
    """
    points = grid_points(grid)
    cached_fn = _memory(cache).cache(fn)

    def _one(point):
        try:
            result = cached_fn(**point, **fixed)
        except Exception as exc:                    # one bad cell must not kill a sweep
            return {**point, "error": f"{type(exc).__name__}: {exc}"}
        return None if result is None else {**point, **dict(result)}

    rows = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(_one)(p) for p in points)
    out = pd.DataFrame([r for r in rows if r is not None])
    out.attrs["grid"] = dict(grid)
    out.attrs["fixed"] = {k: str(v)[:80] for k, v in fixed.items()}
    out.attrs["n_cells"] = len(points)
    return out


def clear_cache() -> None:
    """Drop the on-disk sweep cache. Call after changing `fn`'s definition."""
    _memory(True).clear(warn=False)


def summarise_sweep(results: pd.DataFrame, value_col: str,
                    by: Sequence[str] | None = None,
                    pass_when: Callable[[pd.Series], pd.Series] | None = None
                    ) -> pd.DataFrame:
    """Aggregate a sweep: cell count, pass rate, and the distribution of `value_col`.

    `pass_when` defines what counts as a pass — by default, a positive value.
    Report the pass rate as "k of n cells", never as evidence of an effect.
    """
    df = results.dropna(subset=[value_col]).copy()
    df["_pass"] = (pass_when(df[value_col]) if pass_when else df[value_col] > 0)

    if not by:
        return pd.DataFrame([{
            "n_cells": len(df), "n_pass": int(df["_pass"].sum()),
            "pass_rate": float(df["_pass"].mean()),
            "median": float(df[value_col].median()),
            "p25": float(df[value_col].quantile(0.25)),
            "p75": float(df[value_col].quantile(0.75)),
        }])

    return (df.groupby(list(by), observed=True)
            .agg(n_cells=(value_col, "size"), n_pass=("_pass", "sum"),
                 pass_rate=("_pass", "mean"), median=(value_col, "median"),
                 p25=(value_col, lambda s: s.quantile(0.25)),
                 p75=(value_col, lambda s: s.quantile(0.75)))
            .reset_index())


def sweep_matrix(results: pd.DataFrame, row: str, col: str, value: str,
                 agg: str = "mean") -> pd.DataFrame:
    """Pivot a sweep into a row x col matrix — the input to a heatmap."""
    return results.pivot_table(index=row, columns=col, values=value, aggfunc=agg)


__all__ = [
    "run_grid", "grid_points", "summarise_sweep", "sweep_matrix", "clear_cache",
]
