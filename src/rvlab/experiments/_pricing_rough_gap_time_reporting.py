"""Read-only displays of sealed stopping and physical-time refinement evidence.

The collector owns statistics, selection and gates. These displays retain saved
rows and never reprice, pool historical confirmation paths or rank by final-path outcomes.
"""
from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd


_SELECTED = "selected_policy"
_CONTROLLED = "stopped martingale"


def _grid_figure(tables):
    import matplotlib.pyplot as plt

    table = tables.get("delta_grid_estimates", pd.DataFrame())
    required = {"policy", "estimator", "bump_relative", "mean", "ci_half_width",
        "delta_allowance", "grid_role", "steps", "new_paths", "paths"}
    if table.empty or not required.issubset(table.columns):
        return {}
    data = table.loc[table.policy.eq(_SELECTED)]
    if data.empty:
        return {}
    roles = data[["grid_role", "steps", "new_paths", "paths"]].drop_duplicates()
    roles = roles.sort_values(["steps", "new_paths"], kind="stable").reset_index(drop=True)
    x = {role: i for i, role in enumerate(roles.grid_role)}
    labels = [f"{r.steps:g} steps\n"
        + ("new" if r.new_paths else "historical") + f" · n={int(r.paths):,}"
        + ("\nprecision only" if not r.new_paths and r.steps == 256 else "")
        for r in roles.itertuples()]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8))
    bumps = sorted(data.bump_relative.unique(), reverse=True)
    for i, bump in enumerate(bumps):
        color = ("#2878a5", "#9a6599")[i % 2]
        offset = (i - (len(bumps)-1)/2)*.12
        selected = data.loc[data.bump_relative.eq(bump)]
        controlled = selected.loc[selected.estimator.eq(_CONTROLLED)]
        axes[0].errorbar([x[r]+offset for r in controlled.grid_role], controlled["mean"],
            yerr=controlled.ci_half_width, fmt="o", capsize=4, color=color,
            label=f"{100*bump:g}% stock bump")
        for estimator, rows in selected.groupby("estimator", sort=False):
            axes[1].plot([x[r]+offset for r in rows.grid_role], rows.ci_half_width/rows.delta_allowance,
                ls="none", marker="x" if estimator == "raw" else "o", color=color,
                label=f"{100*bump:g}% · {'raw' if estimator == 'raw' else 'stopped control'}")
    for axis in axes:
        axis.set_xticks(np.arange(len(roles)), labels, fontsize=9)
    axes[0].set(title="New frozen rule: keep each grid and path pool separate",
        ylabel="Controlled stock sensitivity\nmarginal 95% interval")
    axes[0].legend(loc="upper center", bbox_to_anchor=(.5, -.24), frameon=False, fontsize=9, ncol=2)
    axes[1].axhline(1., color="#268577", ls="--")
    axes[1].set(title="How much sampling uncertainty remains?", yscale="log",
        ylabel="95% sampling half-width / unchanged allowance")
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.24), frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Tighter time-refinement evidence with distinct historical and fresh pools", y=1.01)
    fig.tight_layout(rect=(0, .10, 1, 1))
    return {"delta_grid_precision": fig}


def _time_figure(tables):
    import matplotlib.pyplot as plt

    frames = [tables.get(name, pd.DataFrame()) for name in
        ("delta_time_comparisons", "delta_finer_time_comparisons")]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return {}
    table = pd.concat(frames, ignore_index=True)
    required = {"policy", "estimator", "bump_relative", "coarse_steps", "fine_steps", "mean",
        "ci_half_width", "change_plus_uncertainty", "delta_allowance", "comparison_passed"}
    if not required.issubset(table.columns):
        return {}
    data = table.loc[table.policy.eq(_SELECTED) & table.estimator.eq(_CONTROLLED)]
    data = data.sort_values(["coarse_steps", "bump_relative"], ascending=[True, False], kind="stable")
    if data.empty:
        return {}
    labels = [f"{r.coarse_steps:g} → {r.fine_steps:g} steps\n{100*r.bump_relative:g}% stock bump"
        for r in data.itertuples()]
    y = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, max(4.8, .75*len(data))))
    axes[0].errorbar(data["mean"], y, xerr=data.ci_half_width, fmt="o", capsize=4, color="#2878a5")
    axes[0].axvline(0., color="#666666", lw=1.)
    axes[0].set(yticks=y, yticklabels=labels, title="Independent grid changes retain both variances",
        xlabel="Fine minus coarse sensitivity · marginal 95% interval")
    ratio = data.change_plus_uncertainty/data.delta_allowance
    axes[1].barh(y, ratio, color=np.where(data.comparison_passed, "#268577", "#bf6b4e"))
    axes[1].axvline(1., color="#268577", ls="--")
    axes[1].set(yticks=y, yticklabels=labels, title="A smaller interval must still meet the same gate",
        xlabel="(|change| + 95% half-width) / unchanged allowance")
    for axis in axes:
        axis.invert_yaxis()
    fig.suptitle("New selected rule · independent physical-time checks", y=1.01)
    fig.tight_layout()
    return {"delta_independent_time_checks": fig}


def _stopping_figure(tables):
    import matplotlib.pyplot as plt

    data = tables.get("stopping_comparison", pd.DataFrame())
    required = {"control", "gap", "gap_se", "price_error_budget", "price_allowance", "price_budget_pass"}
    if data.empty or not required.issubset(data.columns):
        return {}
    names = {"stable baseline": "Prior selected local control",
        "grid 1 bins penalty0": "Stock-grid · 1 bin · no penalty",
        "grid 1 bins penalty0.0001": "Stock-grid · 1 bin · penalty 1e−4",
        "grid 4 bins penalty0": "Stock-grid · 4 bins · no penalty",
        "grid 4 bins penalty0.0001": "Stock-grid · 4 bins · penalty 1e−4"}
    labels = [textwrap.fill(names.get(r.control, str(r.control).replace("_", " ")), 30)
        + ("\n[selected before final paths]" if getattr(r, "selected_before_final", False) is True else "")
        for r in data.itertuples()]
    y = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, max(5.5, .85*len(data))))
    axes[0].errorbar(data.gap, y, xerr=1.96*data.gap_se, fmt="o", capsize=4, color="#2878a5")
    axes[0].axvline(0., color="#666666", lw=1.)
    axes[0].set(yticks=y, yticklabels=labels, title="Every candidate remains visible",
        xlabel="Upper minus lower ($) · marginal 95% interval")
    ratio = data.price_error_budget/data.price_allowance
    axes[1].barh(y, ratio, color=np.where(data.price_budget_pass, "#268577", "#bf6b4e"))
    axes[1].axvline(1., color="#268577", ls="--")
    axes[1].set(yticks=y, yticklabels=labels, title="Stopping slack and sampling both enter the budget",
        xlabel="Combined price budget / unchanged allowance")
    for axis in axes:
        axis.invert_yaxis()
    fig.suptitle("Frozen exercise rule · further independent stopping-control trials", y=1.01)
    fig.tight_layout()
    return {"stopping_gap_and_budget": fig}


def _interpolation_figure(tables):
    import matplotlib.pyplot as plt

    data = tables.get("stopping_interpolation_audit", pd.DataFrame())
    required = {"date_index", "positive_nodes", "rms_interpolation_error", "max_absolute_interpolation_error"}
    if data.empty or not required.issubset(data.columns):
        return {}
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    for date, rows in data.groupby("date_index", sort=True):
        rows = rows.sort_values("positive_nodes")
        for axis, field in zip(axes, ("rms_interpolation_error", "max_absolute_interpolation_error")):
            axis.plot(np.arange(len(rows)), rows[field], "o-", label=f"Exercise index {date:g}")
            axis.set_xticks(np.arange(len(rows)), [str(n) for n in rows.positive_nodes])
    for axis in axes:
        axis.set(xlabel="Positive stock knots in the proxy", ylabel="Control-function error ($)", yscale="log")
    axes[0].set_title("RMS interpolation error falls on the spent states")
    axes[1].set_title("Local errors near kinks remain larger")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False, fontsize=9)
    fig.suptitle("Deterministic control-function probe · no new acceptance paths", y=1.01)
    fig.tight_layout(rect=(0, .08, 1, 1))
    return {"stopping_interpolation": fig}


def add_gap_time_reporting(result):
    """Attach figures while preserving the exact saved tables and manifest."""
    figures = dict(result.get("figures", {}))
    for build in (_grid_figure, _time_figure, _stopping_figure, _interpolation_figure):
        figures.update(build(result.get("tables", {})))
    return {**result, "figures": figures}


def latest_gap_time_state(result):
    """Use the recorded selection and checks; never choose a final-path winner."""
    manifest = result.get("manifest", {}).get("rough_gap_time_refinement")
    if not manifest:
        return pd.DataFrame()
    count = manifest.get("individual_delta_check_count")
    passed = manifest.get("new_time_delta_individual_checks_passed")
    checks = (f"{count} correlated checks · {'passed' if passed else 'not all passed'}"
        if count is not None and passed is not None else "No completed individual-check summary")
    stopping = result.get("tables", {}).get("rough_gap_time_refinement_stopping_comparison", pd.DataFrame())
    if "selected_before_final" in stopping:
        stopping = stopping.loc[stopping.selected_before_final.eq(True)]
    else:
        stopping = pd.DataFrame()
    required = {"price_error_budget", "price_allowance", "price_budget_pass"}
    if len(stopping) == 1 and required.issubset(stopping.columns):
        row = stopping.iloc[0]
        price = (f"${row.price_error_budget:.6f} vs ${row.price_allowance:.6f} · "
            + ("passed" if row.price_budget_pass else "failed"))
    else:
        price = "No unique completed selected-control price check"
    return pd.DataFrame([
        {"Question": "Physical-time stock sensitivity", "Current evidence": checks},
        {"Question": "Selected stopping estimator", "Current evidence": price},
        {"Question": "New price control's delta sampling", "Current evidence":
            str(manifest.get("new_stopping_control_delta_validated", "unreported"))},
        {"Question": "Readiness flags", "Current evidence":
            f"American price: {manifest.get('american_pricing_ready', 'unreported')}; "
            f"stock delta: {manifest.get('stock_delta_ready', 'unreported')}"},
    ])
