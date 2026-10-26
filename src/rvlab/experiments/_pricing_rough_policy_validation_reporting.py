"""Read-only figures for the newly selected policy validation study.

This module never prices, selects an estimator, changes a sampling unit or
recomputes an acceptance decision. Saved data remain the scientific record.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


_POLICY_LABELS = {"original_policy": "Previous dense policy", "selected_policy": "New policy (ridge 1e−5)"}


def _delta_figures(tables):
    import matplotlib.pyplot as plt

    estimates = tables.get("delta_estimates", pd.DataFrame())
    required = {"policy", "estimator", "bump_relative", "mean", "ci_half_width", "delta_allowance"}
    if estimates.empty or not required.issubset(estimates.columns):
        return {}
    # The result carries the predeclared pooled rows. No candidate or seed is
    # chosen here; row sorting only gives an ascending horizontal bump axis.
    policies = list(dict.fromkeys(estimates.policy))
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.3))
    for j, policy in enumerate(policies):
        data = estimates.loc[estimates.policy.eq(policy)]
        color = ("#2878a5", "#9a6599")[j % 2]
        controlled = data.loc[data.estimator.eq("stopped martingale")].sort_values("bump_relative")
        if not controlled.empty:
            axes[0].errorbar(100*controlled.bump_relative, controlled["mean"],
                yerr=controlled.ci_half_width, fmt="o-" if j == 0 else "D-",
                color=color, capsize=4, label=_POLICY_LABELS.get(policy, policy))
        for estimator, group in data.groupby("estimator", sort=False):
            group = group.sort_values("bump_relative")
            style = ":" if estimator == "raw" else "-"
            marker = "x" if estimator == "raw" else ("o" if j == 0 else "D")
            short_policy = "Previous dense" if policy == "original_policy" else "New ridge1e-5"
            short_estimator = "raw" if estimator == "raw" else "stopped control"
            axes[1].plot(100*group.bump_relative, group.ci_half_width/group.delta_allowance,
                linestyle=style, marker=marker, color=color,
                label=f"{short_policy} · {short_estimator}")
    axes[0].set(title="Compare both frozen rules directly", ylabel="Controlled stock sensitivity",
        xlabel="Symmetric stock bump (%)")
    axes[0].legend(frameon=False)
    axes[1].axhline(1., color="#268577", ls="--")
    axes[1].set(title="Retain raw and controlled sampling checks", yscale="log",
        ylabel="95% half-width / unchanged delta allowance", xlabel="Symmetric stock bump (%)")
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.17), frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Independent stock-bump checks for the newly selected rule", y=1.01)
    fig.tight_layout(rect=(0, .06, 1, 1))
    figures = {"policy_delta_uncertainty": fig}

    comparisons = tables.get("delta_comparisons", pd.DataFrame())
    required = {"comparison", "policy", "estimator", "coarse_bump", "fine_bump", "mean", "ci_half_width"}
    if not comparisons.empty and required.issubset(comparisons.columns):
        chosen = comparisons.loc[comparisons.comparison.eq("bump refinement") &
            comparisons.estimator.eq("stopped martingale")]
        time = tables.get("delta_time_estimates", pd.DataFrame())
        if not chosen.empty:
            fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
            rows = chosen.reset_index(drop=True)
            labels = [f"{_POLICY_LABELS.get(r.policy, r.policy)}\n{100*r.coarse_bump:g}% → {100*r.fine_bump:g}%"
                for r in rows.itertuples()]
            axes[0].errorbar(rows["mean"], np.arange(len(rows)), xerr=rows.ci_half_width,
                fmt="o", capsize=3, color="#2878a5")
            axes[0].axvline(0., color="#666666", lw=1.)
            axes[0].set(yticks=np.arange(len(rows)), yticklabels=labels,
                xlabel="Fine minus coarse stock sensitivity", title="Retain paired bump changes")
            axes[0].invert_yaxis()
            if not time.empty and {"policy", "estimator", "bump_relative", "mean", "ci_half_width"}.issubset(time.columns):
                fine = time.loc[time.estimator.eq("stopped martingale")]
                for j, policy in enumerate(policies):
                    base = estimates.loc[estimates.policy.eq(policy) & estimates.estimator.eq("stopped martingale")].sort_values("bump_relative")
                    fine_policy = fine.loc[fine.policy.eq(policy)].sort_values("bump_relative")
                    color = ("#2878a5", "#9a6599")[j % 2]
                    for frame, linestyle, label in ((base, "-", "128 microsteps"), (fine_policy, "--", "256 microsteps")):
                        if not frame.empty:
                            axes[1].errorbar(100*frame.bump_relative, frame["mean"], yerr=frame.ci_half_width,
                                fmt="o"+linestyle, color=color, capsize=3,
                                label=f"{_POLICY_LABELS.get(policy, policy)} · {label}")
                axes[1].set(xlabel="Symmetric stock bump (%)", ylabel="Frozen-policy stock sensitivity",
                    title="Independent time-refinement paths")
                axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.17), frameon=False, fontsize=8)
            else:
                axes[1].text(.5, .5, "No completed independent time-refinement result",
                    ha="center", va="center", transform=axes[1].transAxes)
                axes[1].set_axis_off()
            fig.suptitle("A precise frozen-rule sensitivity still needs independent checks", y=1.01)
            fig.tight_layout(rect=(0, .04, 1, 1))
            figures["delta_bump_and_time"] = fig
    return figures


def _gap_figure(tables):
    import matplotlib.pyplot as plt

    table = tables.get("stopping_comparison", pd.DataFrame())
    required = {"control", "policy", "gap", "gap_se", "price_error_budget", "price_allowance"}
    if table.empty or not required.issubset(table.columns):
        return {}
    control_labels = {"stable baseline": "Stable baseline", "local 1 bins penalty0": "1 bin · no penalty",
        "local 1 bins penalty0.0001": "1 bin · penalty 1e−4", "local 4 bins penalty0": "4 bins · no penalty",
        "local 4 bins penalty0.0001": "4 bins · penalty 1e−4"}
    fixed_policy = table.policy.nunique() == 1
    labels = [control_labels.get(r.control, str(r.control).replace("_", " "))
        + (" · selected" if getattr(r, "selected_before_final", False) is True else "")
        + ("" if fixed_policy else "\n"+str(r.policy).replace("_", " ")) for r in table.itertuples()]
    y = np.arange(len(table))
    fig, axes = plt.subplots(1, 2, figsize=(12.8, max(4.8, .48*len(table))))
    axes[0].errorbar(table.gap, y, xerr=1.96*table.gap_se, fmt="o", capsize=3, color="#2878a5")
    axes[0].axvline(0., color="#666666", lw=1.)
    axes[0].set(yticks=y, yticklabels=labels, xlabel="Upper minus lower ($), marginal 95% interval",
        title="The remaining discrete stopping gap")
    ratios = table.price_error_budget/table.price_allowance
    axes[1].barh(y, ratios, color=np.where(ratios <= 1., "#268577", "#bf6b4e"))
    axes[1].axvline(1., color="#268577", ls="--")
    axes[1].set(yticks=y, yticklabels=labels, xlabel="Combined price budget / unchanged allowance",
        title="Sampling and stopping error both count")
    for ax in axes:
        ax.invert_yaxis()
    if fixed_policy:
        fig.suptitle("Same frozen exercise rule; different martingale controls", y=1.01)
    fig.tight_layout()
    return {"stopping_gap_and_budget": fig}


def add_policy_validation_reporting(result):
    """Attach figures without mutating tables, manifests or acceptance results."""
    figures = dict(result.get("figures", {}))
    figures.update(_gap_figure(result["tables"]))
    figures.update(_delta_figures(result["tables"]))
    return {**result, "figures": figures}


def latest_policy_validation_state(result):
    """A compact opening panel using recorded gates, never a model ranking."""
    manifest = result.get("manifest", {}).get("rough_policy_validation")
    if not manifest:
        return pd.DataFrame()
    count = manifest.get("individual_delta_check_count")
    checked = manifest.get("new_policy_individual_delta_checks_passed")
    if count is None or checked is None:
        delta = "No completed individual-check summary"
    else:
        delta = f"{count} correlated checks · {'passed' if checked else 'not all passed'}"
    stopping = result.get("tables", {}).get("rough_policy_validation_stopping_comparison", pd.DataFrame())
    if not stopping.empty and "selected_before_final" in stopping:
        stopping = stopping.loc[stopping.selected_before_final.eq(True)]
    else:
        stopping = pd.DataFrame()
    if len(stopping) == 1 and {"price_error_budget", "price_allowance", "price_budget_pass"}.issubset(stopping.columns):
        row = stopping.iloc[0]
        price = (f"${row.price_error_budget:.6f} vs ${row.price_allowance:.6f} · "
            + ("passed" if row.price_budget_pass else "failed"))
    else:
        price = "No unique completed selected-control price check"
    readiness = (f"American price: {manifest.get('american_pricing_ready', 'unreported')}; "
        f"stock delta: {manifest.get('stock_delta_ready', 'unreported')}")
    return pd.DataFrame([
        {"Question": "New selected rule", "Current evidence": manifest.get("new_policy_name", "unreported"),
            "Interpretation": "Explicit frozen coefficients; preceding-rule Greeks are separate"},
        {"Question": "Finite-policy stock sensitivity", "Current evidence": delta,
            "Interpretation": "Individual sampling, bump and time checks; no total error certificate"},
        {"Question": "Selected stopping estimator", "Current evidence": price,
            "Interpretation": "Combined price budget against unchanged allowance"},
        {"Question": "New control stock-bump estimator",
            "Current evidence": f"Validated: {manifest.get('new_stopping_control_delta_validated', 'unreported')}",
            "Interpretation": "The delta study uses the stable martingale; new-control sampling is separate"},
        {"Question": "Readiness flags", "Current evidence": readiness,
            "Interpretation": "Optimal-value Greek and historical SPY inputs remain separate requirements"}])
