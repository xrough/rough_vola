"""Read-only figures for frozen-policy stopping and stock-delta experiments.

This module never prices, selects an estimator, changes a sampling unit or
recomputes an acceptance decision. Saved data remain the scientific record.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


_POLICY_LABELS = {"original_policy": "Original policy", "selected_policy": "Dense selected policy"}
_ESTIMATOR_LABELS = {"raw": "Raw payoff", "stopped martingale": "Stopped martingale"}
_COLORS = {"raw": "#a6611a", "stopped martingale": "#2878a5"}


def _delta_figures(tables):
    import matplotlib.pyplot as plt

    estimates = tables.get("delta_estimates", pd.DataFrame())
    required = {"policy", "estimator", "bump_relative", "mean", "ci_half_width", "delta_allowance"}
    if estimates.empty or not required.issubset(estimates.columns):
        return {}
    # The result carries the predeclared pooled rows. No candidate or seed is
    # chosen here; row sorting only gives an ascending horizontal bump axis.
    policies = list(dict.fromkeys(estimates.policy))
    fig, axes = plt.subplots(2, len(policies), figsize=(6.3*len(policies), 8.0), squeeze=False)
    for column, policy in enumerate(policies):
        data = estimates.loc[estimates.policy.eq(policy)]
        for estimator, group in data.groupby("estimator", sort=False):
            group = group.sort_values("bump_relative")
            x = 100*np.asarray(group.bump_relative, float)
            color = _COLORS.get(estimator, "#737b83")
            label = _ESTIMATOR_LABELS.get(estimator, estimator)
            axes[0, column].errorbar(x, group["mean"], yerr=group.ci_half_width,
                fmt="o-", color=color, capsize=4, label=label)
            axes[1, column].plot(x, group.ci_half_width/group.delta_allowance,
                "o-", color=color, label=label)
        axes[0, column].set(title=_POLICY_LABELS.get(policy, policy),
            ylabel="Frozen-policy stock sensitivity", xlabel="Symmetric stock bump (%)")
        axes[1, column].axhline(1., color="#268577", ls="--", label="Unchanged allowance")
        axes[1, column].set(yscale="log", ylabel="95% sampling half-width / delta allowance",
            xlabel="Symmetric stock bump (%)")
        axes[0, column].legend(frameon=False)
    fig.suptitle("A stopped martingale reduces stock-bump sampling noise", y=1.01)
    fig.tight_layout()
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
            fig.suptitle("Small sampling error does not establish optimal-value delta", y=1.01)
            fig.tight_layout(rect=(0, .04, 1, 1))
            figures["delta_bump_and_time"] = fig
    return figures


def _gap_figure(tables):
    import matplotlib.pyplot as plt

    table = tables.get("stopping_comparison", pd.DataFrame())
    required = {"control", "policy", "gap", "gap_se", "price_error_budget", "price_allowance"}
    if table.empty or not required.issubset(table.columns):
        return {}
    labels = [f"{str(r.control).replace('_', ' ')}\n{str(r.policy).replace('_', ' ')}" for r in table.itertuples()]
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
    fig.tight_layout()
    return {"stopping_gap_and_budget": fig}


def add_stopping_delta_reporting(result):
    """Attach figures without mutating tables, manifests or acceptance results."""
    figures = dict(result.get("figures", {}))
    figures.update(_gap_figure(result["tables"]))
    figures.update(_delta_figures(result["tables"]))
    return {**result, "figures": figures}
