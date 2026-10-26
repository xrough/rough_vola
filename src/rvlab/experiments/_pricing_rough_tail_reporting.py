"""Read-only figures for independent policy/dual tail-attribution evidence.

Recorded selection and dominance decisions belong to the audited collector.
These displays never price, select a final-path winner or alter a gate.
"""
from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd


_BASE_POLICY = "frozen selected policy"
_BASE_CONTROL = "current baseline"


def curated_tail_comparisons(tables, manifest):
    """Four declared comparisons, preserving fallback coincidences explicitly."""
    data = tables.get("comparison", pd.DataFrame())
    if data.empty or not {"policy", "control"}.issubset(data.columns):
        return pd.DataFrame()
    policy, control = manifest.get("selected_policy"), manifest.get("selected_control")
    if policy is None or control is None:
        return pd.DataFrame()
    base_policy = manifest.get("baseline_policy", _BASE_POLICY)
    base_control = manifest.get("baseline_control", _BASE_CONTROL)
    roles = (("Baseline", base_policy, base_control),
        ("Policy challenge · common control", policy, base_control),
        ("Dual challenge · baseline rule", base_policy, control),
        ("Both training-selected choices", policy, control))
    rows = []
    for role, p, c in roles:
        row = data.loc[data.policy.eq(p) & data.control.eq(c)]
        if len(row) == 1:
            rows.append(row.assign(display_role=role))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _component_figure(tables):
    import matplotlib.pyplot as plt

    data = tables.get("dominance", pd.DataFrame())
    fields = {"challenge", "bound_improvement", "improvement_low", "improvement_high",
        "majority_margin", "majority_margin_low", "majority_margin_high", "majority_dominance_proved"}
    if data.empty or not fields.issubset(data.columns):
        return {}
    labels = [{"dual": "Upper-estimator challenge", "policy": "Admissible-policy challenge"}.get(v, str(v))
        for v in data.challenge]
    y = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6))
    axes[0].errorbar(data.bound_improvement, y,
        xerr=np.vstack((data.bound_improvement-data.improvement_low, data.improvement_high-data.bound_improvement)),
        fmt="o", capsize=4, color="#2878a5")
    axes[1].errorbar(data.majority_margin, y,
        xerr=np.vstack((data.majority_margin-data.majority_margin_low, data.majority_margin_high-data.majority_margin)),
        fmt="o", capsize=4, color="#9a6599")
    for axis in axes:
        axis.axvline(0., color="#666666", lw=1.)
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
    axes[0].set(title="How much removable error is resolved?", xlabel="Bound improvement ($) · marginal 95% interval")
    axes[1].set(title="Does it establish a majority of the baseline gap?",
        xlabel="Improvement − half baseline gap ($) · paired 95% interval")
    fig.suptitle("Paired component bounds retain uncertainty in the baseline gap", y=1.01)
    fig.tight_layout()
    return {"component_bounds": fig}


def _gap_figure(tables, manifest):
    import matplotlib.pyplot as plt

    data = curated_tail_comparisons(tables, manifest)
    fields = {"gap", "gap_se", "price_error_budget", "price_allowance", "price_budget_pass"}
    if data.empty or not fields.issubset(data.columns):
        return {}
    labels = [textwrap.fill(label, 28) for label in data.display_role]
    y = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1))
    axes[0].errorbar(data.gap, y, xerr=1.96*data.gap_se, fmt="o", capsize=4, color="#2878a5")
    axes[0].axvline(0., color="#666666", lw=1.)
    axes[0].set(title="Compare fixed scientific contrasts", xlabel="Upper minus lower ($) · marginal 95% interval")
    axes[1].barh(y, data.price_error_budget/data.price_allowance,
        color=np.where(data.price_budget_pass, "#268577", "#bf6b4e"))
    axes[1].axvline(1., color="#268577", ls="--")
    axes[1].set(title="The original price gate still applies", xlabel="Combined price budget / unchanged allowance")
    for axis in axes:
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
    fig.suptitle("Policy and control choices are frozen before final valuation", y=1.01)
    fig.tight_layout()
    return {"gap_and_price_budget": fig}


def _tail_figure(tables, manifest):
    import matplotlib.pyplot as plt

    data = tables.get("gap_distribution", pd.DataFrame())
    fields = {"policy", "control", "gap_p90", "gap_p99", "gap_maximum",
        "largest_decile_gap_share", "largest_percentile_gap_share"}
    if data.empty or not fields.issubset(data.columns):
        return {}
    policy, control = manifest.get("selected_policy"), manifest.get("selected_control")
    if policy is None or control is None:
        return {}
    base_policy = manifest.get("baseline_policy", _BASE_POLICY)
    base_control = manifest.get("baseline_control", _BASE_CONTROL)
    roles = (("Baseline", base_policy, base_control),
        ("Selected control · baseline rule", base_policy, control),
        ("Both selected choices", policy, control))
    selected, seen = [], set()
    for label, p, c in roles:
        if (p, c) in seen:
            continue
        row = data.loc[data.policy.eq(p) & data.control.eq(c)]
        if len(row) == 1:
            selected.append((label, row.iloc[0]))
            seen.add((p, c))
    if not selected:
        return {}
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1))
    colors = ("#2878a5", "#9a6599", "#bf6b4e")
    width = .7/len(selected)
    for i, (label, row) in enumerate(selected):
        values = np.array([row.gap_p90, row.gap_p99, row.gap_maximum])
        # Exact zeros are omitted from a log ordinate, never replaced by a
        # positive floor or removed from the saved distribution/statistics.
        mask = values > 0
        axes[0].plot(np.arange(3)[mask], values[mask], "o-", color=colors[i%3], label=label)
        axes[1].bar(np.arange(2)+(i-(len(selected)-1)/2)*width,
            [row.largest_decile_gap_share, row.largest_percentile_gap_share], width,
            color=colors[i%3], label=label)
    axes[0].set(xticks=np.arange(3), xticklabels=("90th percentile", "99th percentile", "Maximum"),
        yscale="log", ylabel="Pathwise gap ($)", title="Keep remaining rare-path gaps visible")
    axes[1].set(xticks=np.arange(2), xticklabels=("Largest 10% of paths", "Largest 1% of paths"),
        ylabel="Share of total gap", title="Concentration is descriptive")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1 if len(labels)>2 else 2, frameon=False, fontsize=9)
    fig.suptitle("Final-path tail diagnostics · every path retained", y=1.01)
    fig.tight_layout(rect=(0, .1 if len(labels)>2 else .06, 1, 1))
    return {"residual_tail_diagnostics": fig}


def _spent_figure(tables):
    import matplotlib.pyplot as plt

    data = tables.get("spent_state_summary", pd.DataFrame())
    if data.empty or not {"group", "path_fraction", "gap_share"}.issubset(data.columns):
        return {}
    y = np.arange(len(data))
    fig, axis = plt.subplots(figsize=(10.8, max(4.9, .58*len(data))))
    axis.barh(y-.16, 100*data.path_fraction, .3, label="Share of all spent paths", color="#2878a5")
    axis.barh(y+.16, 100*data.gap_share, .3, label="Share of their total gap", color="#9a6599")
    axis.set(yticks=y, yticklabels=[textwrap.fill(str(v), 32) for v in data.group],
        xlabel="Percent", title="Overlapping spent-path groups guide prospective challenges")
    axis.invert_yaxis()
    axis.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return {"spent_state_description": fig}


def add_tail_reporting(result):
    """Attach figures without changing any recorded data or decisions."""
    tables, manifest = result.get("tables", {}), result.get("manifest", {})
    figures = dict(result.get("figures", {}))
    figures.update(_component_figure(tables))
    figures.update(_gap_figure(tables, manifest))
    figures.update(_tail_figure(tables, manifest))
    figures.update(_spent_figure(tables))
    return {**result, "figures": figures}


def latest_tail_state(result):
    """Opening status using explicit training selection and recorded gates."""
    manifest = result.get("manifest", {}).get("rough_tail_attribution")
    if not manifest:
        return pd.DataFrame()
    data = result.get("tables", {}).get("rough_tail_attribution_comparison", pd.DataFrame())
    columns = {"selected_policy", "selected_control", "price_error_budget", "price_allowance", "price_budget_pass"}
    selected = (data.loc[data.selected_policy.eq(True) & data.selected_control.eq(True)]
        if columns.issubset(data.columns) else pd.DataFrame())
    if len(selected) == 1:
        row = selected.iloc[0]
        price = f"${row.price_error_budget:.6f} vs ${row.price_allowance:.6f} · " + ("passed" if row.price_budget_pass else "failed")
    else:
        price = "No unique completed training-selected pair"
    return pd.DataFrame([
        {"Question": "Selected exercise rule", "Current evidence": manifest.get("selected_policy", "unreported")},
        {"Question": "Selected control", "Current evidence": manifest.get("selected_control", "unreported")},
        {"Question": "Majority attribution", "Current evidence": manifest.get("dominant_source", "No completed majority conclusion")},
        {"Question": "Selected-pair price gate", "Current evidence": price},
        {"Question": "New-rule stock delta validated", "Current evidence": str(manifest.get("new_rule_stock_delta_validated", "unreported"))},
        {"Question": "Readiness flags", "Current evidence": f"American price: {manifest.get('american_pricing_ready', 'unreported')}; stock delta: {manifest.get('stock_delta_ready', 'unreported')}"},
    ])
