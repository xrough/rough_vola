"""Read-only figures for the frozen Bellman rule's sensitivities and bounds."""
from __future__ import annotations

import numpy as np
import pandas as pd

FINE_BUMP = .0025
POLICY_LABELS = {"baseline": "Original rule", "bellman": "GH16 full rule"}


def bellman_delta_sampling_summary(tables):
    data = tables.get("delta_sampling", pd.DataFrame())
    if not {"policy", "estimator", "physical_steps", "bump_relative"}.issubset(data.columns):
        return pd.DataFrame()
    return data.loc[data.policy.eq("bellman") & data.estimator.eq("controlled")].sort_values(
        ["physical_steps", "bump_relative"], ascending=[True, False]).copy()


def bellman_delta_refinement_summary(tables):
    """Display aliases for the recorded paired-bump and independent-time checks."""
    rows = []
    for name, kind in (("bump_refinement", "Paired bump"), ("time_refinement", "Independent time")):
        data = tables.get(name, pd.DataFrame())
        if not {"policy", "estimator"}.issubset(data.columns):
            continue
        for _, original in data.loc[data.policy.eq("bellman") & data.estimator.eq("controlled")].iterrows():
            row = original.copy()
            row["check_family"] = kind
            if name == "bump_refinement":
                row["display_check"] = f"Paired bump · {int(row.physical_steps)} steps"
                row["recorded_check_pass"] = row.bump_check_pass
            else:
                row["display_check"] = f"Independent time · {100*row.bump_relative:g}% bump"
                row["estimate"] = row.delta_change
                row["recorded_check_pass"] = row.time_check_pass
            rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()


def bellman_gap_summary(tables):
    """Keep every recorded final control, in its sealed candidate order."""
    data = tables.get("gap_comparison", pd.DataFrame())
    required = {"control", "selected_control", "paths", "gap", "gap_se",
        "price_error_budget", "price_allowance", "price_budget_pass"}
    return data.copy() if required.issubset(data.columns) else pd.DataFrame()


def bellman_gap_selected_changes(tables):
    """Read paired changes for the training-selected control, without reselecting."""
    data = tables.get("gap_paired_changes", pd.DataFrame())
    required = {"selected_control", "metric", "estimate", "interval_low", "interval_high"}
    if not required.issubset(data.columns):
        return pd.DataFrame()
    return data.loc[data.selected_control.eq(True) & data.metric.isin(
        ["gap_gain", "upper_decrease", "lower_change"])].copy()


def _sampling_figure(tables):
    data = tables.get("delta_sampling", pd.DataFrame())
    fields = {"policy", "estimator", "physical_steps", "paths", "bump_relative", "estimate",
        "interval_low", "interval_high", "half_width", "delta_allowance"}
    if data.empty or not fields.issubset(data.columns):
        return {}
    fine = data.loc[data.bump_relative.eq(FINE_BUMP)].copy()
    controlled = fine.loc[fine.estimator.eq("controlled")].sort_values(["physical_steps", "policy"])
    if controlled.empty:
        return {}
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    fig, axes = plt.subplots(1, 2, figsize=(12.5, max(4.7, .7*len(controlled))))
    labels = [f"{POLICY_LABELS.get(row.policy,row.policy)} · {int(row.physical_steps)} steps\nN = {int(row.paths):,}"
        for _, row in controlled.iterrows()]
    y = np.arange(len(controlled))
    values = controlled.estimate.to_numpy()
    errors = np.vstack((values-controlled.interval_low, controlled.interval_high-values))
    axes[0].errorbar(values, y, xerr=errors, fmt="o", capsize=4, color="#2878a5")
    axes[0].set(yticks=y, yticklabels=labels, xlabel="Stock delta · marginal 95% interval",
        title="Controlled estimates at the declared 0.25% bump")
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda value,_: f"{value:.4f}"))
    axes[0].xaxis.set_major_locator(MaxNLocator(nbins=4))
    for estimator, offset, color in (("raw",-.16,"#9b9b9b"),("controlled",.16,"#2878a5")):
        positions, ratios = [], []
        for i, (_, row) in enumerate(controlled.iterrows()):
            match = fine.loc[fine.policy.eq(row.policy) & fine.physical_steps.eq(row.physical_steps)
                & fine.estimator.eq(estimator)]
            if len(match) == 1:
                positions.append(i+offset)
                ratios.append(float(match.iloc[0].half_width/match.iloc[0].delta_allowance))
        axes[1].barh(positions, ratios, height=.3, color=color,
            label="Raw" if estimator == "raw" else "Stopped martingale")
        for position, ratio in zip(positions,ratios):
            if ratio == 0.:
                axes[1].annotate("0", (0.,position), xycoords=("axes fraction","data"),
                    xytext=(3,0),textcoords="offset points",va="center")
    axes[1].axvline(1.,color="#2a9d8f",ls="--")
    axes[1].set(xscale="log",yticks=y,yticklabels=labels,
        xlabel="95% half-width / unchanged delta allowance · log scale",
        title="Variance reduction does not certify total delta error")
    axes[1].legend(frameon=False,loc="upper center",bbox_to_anchor=(.5,-.20),ncol=2)
    for axis in axes:axis.invert_yaxis();axis.grid(alpha=.18)
    fig.suptitle("Fixed rules; independent physical-time pools remain separate")
    fig.tight_layout()
    return {"delta_sampling_vs_allowance":fig}


def _refinement_figure(tables):
    data = bellman_delta_refinement_summary(tables)
    fields = {"estimate", "interval_low", "interval_high", "change_plus_uncertainty", "delta_allowance"}
    if data.empty or not fields.issubset(data.columns):
        return {}
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    fig, axes = plt.subplots(1,2,figsize=(12.5,max(4.7,.7*len(data))))
    y = np.arange(len(data))
    for i, row in data.iterrows():
        color = "#2878a5" if row.check_family == "Paired bump" else "#9a6599"
        axes[0].errorbar(row.estimate,i,xerr=np.array([[row.estimate-row.interval_low],
            [row.interval_high-row.estimate]]),fmt="o",capsize=4,color=color)
    axes[0].axvline(0.,color="#555555",lw=1)
    axes[0].set(xlabel="Delta change (10⁻³) · marginal 95% interval",
        title="Bump changes are paired; time changes are independent")
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda value,_: f"{value/1e-3:g}"))
    axes[1].barh(y,data.change_plus_uncertainty/data.delta_allowance,
        color=["#2a9d8f" if flag else "#bd6c50" for flag in data.recorded_check_pass])
    axes[1].axvline(1.,color="#2a9d8f",ls="--")
    axes[1].set(xlabel="Recorded |change| + uncertainty / unchanged allowance",
        title="Individual checks retain their original budgets")
    for axis in axes:
        axis.set(yticks=y,yticklabels=data.display_check.tolist());axis.invert_yaxis();axis.grid(alpha=.18)
    fig.suptitle("GH16 full rule: these correlated diagnostics are not a combined certificate")
    fig.tight_layout()
    return {"delta_refinement":fig}


def _gap_figure(tables):
    data = bellman_gap_summary(tables)
    if data.empty:
        return {}
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    labels = []
    for _, row in data.iterrows():
        label = row.control.replace("latest selected tail control", "Prior tail control")
        label = label.replace("target 1 bins penalty0.0001", "1 bin · penalty 10⁻⁴")
        label = label.replace("target 4 bins penalty0.0001", "4 bins · penalty 10⁻⁴")
        label = label.replace("target 1 bins penalty0", "1 bin · no penalty")
        label = label.replace("target 4 bins penalty0", "4 bins · no penalty")
        labels.append(label + ("\nSelected before final paths" if row.selected_control else ""))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, max(4.7, .75*len(data))))
    y = np.arange(len(data))
    for i, (_, row) in enumerate(data.iterrows()):
        axes[0].errorbar(row.gap, i, xerr=1.96*row.gap_se, fmt="o", capsize=4,
            color="#2878a5" if row.selected_control else "#8a8a8a")
    axes[0].set(xlabel="Stopping gap ($) · marginal 95% interval",
        title="Same GH16 rule and identical final paths")
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda value,_: f"{value:.4f}"))
    axes[0].xaxis.set_major_locator(MaxNLocator(nbins=4))
    axes[1].barh(y, data.price_error_budget/data.price_allowance,
        color=["#2a9d8f" if flag else "#bd6c50" for flag in data.price_budget_pass])
    axes[1].axvline(1., color="#2a9d8f", ls="--")
    axes[1].set(xlabel="Recorded price budget / unchanged price allowance",
        title="A smaller gap must still meet the original budget")
    for axis in axes:
        axis.set(yticks=y, yticklabels=labels)
        axis.invert_yaxis()
        axis.grid(alpha=.18)
    for axis in axes:
        for tick, selected in zip(axis.get_yticklabels(), data.selected_control):
            if selected:
                tick.set_fontweight("bold")
    fig.suptitle("Control selection is frozen before independent valuation; all alternatives remain visible")
    fig.tight_layout()
    return {"stopping_gap_and_budget": fig}


def add_bellman_delta_reporting(result):
    """Attach displays without mutating samples, tables, selection or gates."""
    tables = result.get("tables",{})
    figures = dict(result.get("figures",{}))
    figures.update(_sampling_figure(tables))
    figures.update(_refinement_figure(tables))
    figures.update(_gap_figure(tables))
    return {**result,"figures":figures}


def latest_bellman_delta_state(result):
    """Read recorded finite-rule checks without creating an overall certificate."""
    manifest=result.get("manifest",{}).get("rough_bellman_delta")
    if not manifest:
        return pd.DataFrame()
    prefix="rough_bellman_delta_"
    tables={name.removeprefix(prefix):table for name,table in result.get("tables",{}).items()
        if name.startswith(prefix)}
    sampling=bellman_delta_sampling_summary(tables)
    rows=[{"Question":"Frozen decision function","Current evidence":
        f"GH16 full · operator {manifest.get('frozen_operator_steps','unreported')} physical steps; 64 exercise intervals"}]
    for steps in (128,256):
        match=(sampling.loc[sampling.physical_steps.eq(steps)&sampling.bump_relative.eq(FINE_BUMP)]
            if not sampling.empty else pd.DataFrame())
        if len(match)==1 and {"estimate","half_width","delta_allowance","paths"}.issubset(match.columns):
            row=match.iloc[0]
            value=f"N={int(row.paths):,}; delta {row.estimate:.6f} ± {row.half_width:.6f}; allowance {row.delta_allowance:.6f}"
        else:
            value="No unique completed sampling record"
        rows.append({"Question":f"GH16 {steps}-step paths · 95% half-width","Current evidence":value})
    checks=[]
    if "sampling_pass" in sampling:
        checks.extend(sampling.sampling_pass.tolist())
    for name,column in (("bump_refinement","bump_check_pass"),("time_refinement","time_check_pass")):
        table=tables.get(name,pd.DataFrame())
        if {"policy","estimator",column}.issubset(table.columns):
            checks.extend(table.loc[table.policy.eq("bellman")&table.estimator.eq("controlled"),column].tolist())
    rows.extend([
        {"Question":"GH16 individual delta checks","Current evidence":
            f"{sum(bool(flag) for flag in checks)}/{len(checks)} recorded; expected 8, correlated" if checks else "No completed individual checks"},
        {"Question":"Optimal delta ready","Current evidence":str(manifest.get("optimal_delta_ready","unreported"))}])
    gap = bellman_gap_summary(tables)
    selected = gap.loc[gap.selected_control.eq(True)] if not gap.empty else pd.DataFrame()
    if len(selected) == 1:
        row = selected.iloc[0]
        rows.extend([
            {"Question":"Selected stopping control","Current evidence":f"{row.control}; N={int(row.paths):,}"},
            {"Question":"Selected price budget / allowance","Current evidence":
                f"{row.price_error_budget:.6f} / {row.price_allowance:.6f}; recorded pass: {bool(row.price_budget_pass)}"},
            {"Question":"New price-control delta validated","Current evidence":str(row.get("new_control_delta_validated","unreported"))}])
    rows.extend([
        {"Question":"Readiness flags","Current evidence":f"American price: {manifest.get('american_pricing_ready','unreported')}; stock delta: {manifest.get('stock_delta_ready','unreported')}"}])
    return pd.DataFrame(rows)
