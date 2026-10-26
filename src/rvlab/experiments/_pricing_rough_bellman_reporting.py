"""Read-only graphics for preselected Bellman policy challenges."""
from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd


BASELINE = "frozen selected policy"


def _identities(manifest):
    baseline = manifest.get("baseline_policy", BASELINE)
    accepted, challenger = manifest.get("selected_policy"), manifest.get("selected_challenger")
    if not accepted or not challenger:
        return []
    return [("Frozen baseline", baseline), ("Accepted rule · training gate", accepted),
        ("Nonbaseline challenger · training only", challenger)]


def curated_bellman_comparisons(tables, manifest):
    """Match explicit frozen identities; coincident roles keep their original values."""
    table = tables.get("comparison", pd.DataFrame())
    if "policy" not in table:
        return pd.DataFrame()
    rows = []
    for role, name in _identities(manifest):
        match = table.loc[table.policy.eq(name)]
        if len(match) == 1:
            row = match.iloc[0].copy()
            row["display_role"] = role
            rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()


def _labels(table):
    labels = []
    for _, row in table.iterrows():
        suffix = []
        if row.get("selected_policy", False):
            suffix.append("accepted")
        if row.get("selected_challenger", False):
            suffix.append("challenger")
        name = str(row.policy) + (" · " + ", ".join(suffix) if suffix else "")
        labels.append(textwrap.fill(name, 28))
    return labels


def _policy_gain_figure(tables):
    data = tables.get("policy_error", pd.DataFrame())
    fields = {"policy", "identified_policy_gain", "gain_interval_low", "gain_interval_high",
        "majority_margin", "majority_margin_low", "majority_margin_high"}
    if data.empty or not fields.issubset(data.columns):
        return {}
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    fig, axes = plt.subplots(1, 2, figsize=(12.5, max(5., .64*len(data))))
    y = np.arange(len(data))
    for axis, mean, low, high, title, xlabel in (
            (axes[0], "identified_policy_gain", "gain_interval_low", "gain_interval_high",
                "Does a different stopping rule improve the lower bound?", "Paired lower gain ($) · marginal 95% interval"),
            (axes[1], "majority_margin", "majority_margin_low", "majority_margin_high",
                "Does the challenge explain a majority of the gap?", "Gain − half baseline gap ($) · paired 95% interval")):
        values = data[mean].to_numpy()
        errors = np.vstack((values-data[low], data[high]-values))
        axis.errorbar(values, y, xerr=errors, fmt="o", capsize=4, color="#2878a5")
        axis.axvline(0., color="#555555", lw=1)
        axis.set(yticks=y, yticklabels=_labels(data), xlabel=xlabel.replace("($)", "(10⁻⁴ $)"), title=title)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value/1e-4:g}"))
        axis.invert_yaxis();axis.grid(alpha=.18)
    fig.suptitle("Accepted policy and designated challenger are frozen before final paths")
    fig.tight_layout()
    return {"policy_gain_and_majority": fig}


def _gap_figure(tables):
    data = tables.get("comparison", pd.DataFrame())
    fields = {"policy", "gap", "gap_se", "price_error_budget", "price_allowance", "price_budget_pass"}
    if data.empty or not fields.issubset(data.columns):
        return {}
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.5, max(5., .64*len(data))))
    y = np.arange(len(data))
    axes[0].errorbar(data.gap, y, xerr=1.96*data.gap_se, fmt="o", capsize=4, color="#2878a5")
    axes[0].axvline(0., color="#555555", lw=1)
    ratios = data.price_error_budget/data.price_allowance
    axes[1].barh(y, ratios, color=["#2a9d8f" if flag else "#bd6c50" for flag in data.price_budget_pass])
    axes[1].axvline(1., color="#2a9d8f", linestyle="--")
    for axis in axes:
        axis.set(yticks=y, yticklabels=_labels(data));axis.invert_yaxis();axis.grid(alpha=.18)
    axes[0].set(title="Common upper; only stopping decisions change", xlabel="Upper minus lower ($) · marginal 95% interval")
    axes[1].set(title="The unchanged price allowance remains binding", xlabel="Combined price budget / unchanged allowance")
    fig.suptitle("Every declared policy remains visible, including rejected challenges")
    fig.tight_layout()
    return {"gap_and_price_budget": fig}


def _exercise_and_tail_figure(tables, manifest):
    exercise = tables.get("exercise_times", pd.DataFrame())
    tails = tables.get("tail_distribution", pd.DataFrame())
    if (exercise.empty or tails.empty or not {"policy", "stop_index", "fraction"}.issubset(exercise.columns)
            or not {"policy", "gap_p90", "gap_p99", "gap_maximum"}.issubset(tails.columns)):
        return {}
    cases, seen = [], set()
    for role, policy in _identities(manifest):
        if policy in seen:
            continue
        stops, tail = exercise.loc[exercise.policy.eq(policy)], tails.loc[tails.policy.eq(policy)]
        if len(stops) != 65 or len(tail) != 1 or set(stops.stop_index) != set(range(65)):
            continue
        cases.append((role, stops, tail.iloc[0]));seen.add(policy)
    if not cases:
        return {}
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.))
    bins = ((0, 15), (16, 31), (32, 47), (48, 63), (64, 64))
    colors = ("#2878a5", "#539ab5", "#81b9c5", "#9a6599", "#bdc3c7")
    left = np.zeros(len(cases))
    for (first, last), color in zip(bins, colors):
        mass = np.array([float(stops.loc[stops.stop_index.between(first,last), "fraction"].sum())
            for _, stops, _ in cases])
        axes[0].barh(np.arange(len(cases)), mass, left=left, color=color,
            label=f"{first}–{last}" if first != last else "64 · expiry")
        left += mass
    axes[0].set(yticks=np.arange(len(cases)), yticklabels=[textwrap.fill(role,24) for role, _, _ in cases],
        xlabel="Fraction of all final paths", title="Recorded stop dates, including expiry", xlim=(0.,1.))
    axes[0].invert_yaxis();axes[0].legend(title="Exercise date index", frameon=False, ncol=3,
        loc="upper center", bbox_to_anchor=(.5,-.15))
    for role, _, row in cases:
        values = np.array([row.gap_p90,row.gap_p99,row.gap_maximum],float)
        positive = values > 0
        axes[1].plot(np.arange(3)[positive],values[positive],"o-",label=textwrap.fill(role,28))
    axes[1].set(xticks=np.arange(3),xticklabels=["90th percentile","99th percentile","Maximum"],
        yscale="log",ylabel="Pathwise gap ($)",title="All-path tails remain descriptive")
    axes[1].legend(frameon=False,loc="upper left")
    for axis in axes:axis.grid(alpha=.18)
    fig.suptitle("Coincident policy roles are shown once; no path is removed")
    fig.tight_layout()
    return {"exercise_and_tails":fig}


def add_bellman_reporting(result):
    """Attach graphics without changing data, policy selection or numerical gates."""
    tables, manifest = result.get("tables",{}), result.get("manifest",{})
    figures = dict(result.get("figures",{}))
    figures.update(_policy_gain_figure(tables))
    figures.update(_gap_figure(tables))
    figures.update(_exercise_and_tail_figure(tables,manifest))
    return {**result,"figures":figures}


def latest_bellman_state(result):
    """Opening status reads explicit training identities and recorded intervals."""
    manifest = result.get("manifest",{}).get("rough_bellman_policy")
    if not manifest:
        return pd.DataFrame()
    tables = result.get("tables",{})
    comparison = tables.get("rough_bellman_policy_comparison",pd.DataFrame())
    errors = tables.get("rough_bellman_policy_policy_error",pd.DataFrame())
    accepted = (comparison.loc[comparison.policy.eq(manifest.get("selected_policy"))]
        if "policy" in comparison else pd.DataFrame())
    challenger = (errors.loc[errors.policy.eq(manifest.get("selected_challenger"))]
        if "policy" in errors else pd.DataFrame())
    gate, gain, majority = "No unique completed accepted rule", "No unique completed challenger", "Unreported"
    if len(accepted)==1 and {"price_error_budget","price_allowance","price_budget_pass"}.issubset(accepted.columns):
        row=accepted.iloc[0]
        gate=f"${row.price_error_budget:.6f} vs ${row.price_allowance:.6f} · "+("passed" if row.price_budget_pass else "failed")
    if len(challenger)==1 and {"identified_policy_gain","gain_interval_low","gain_interval_high","policy_majority_proved"}.issubset(challenger.columns):
        row=challenger.iloc[0]
        gain=f"${row.identified_policy_gain:+.6f} [${row.gain_interval_low:+.6f}, ${row.gain_interval_high:+.6f}]"
        majority=str(bool(row.policy_majority_proved))
    return pd.DataFrame([
        {"Question":"Accepted rule · training gate","Current evidence":manifest.get("selected_policy","unreported")},
        {"Question":"Designated nonbaseline challenger","Current evidence":manifest.get("selected_challenger","unreported")},
        {"Question":"Challenger paired gain · marginal 95%","Current evidence":gain},
        {"Question":"Challenger policy-majority check","Current evidence":majority},
        {"Question":"Accepted rule price gate","Current evidence":gate},
        {"Question":"New-rule stock delta validated","Current evidence":str(manifest.get("new_rule_stock_delta_validated","unreported"))},
        {"Question":"Readiness flags","Current evidence":f"American price: {manifest.get('american_pricing_ready','unreported')}; stock delta: {manifest.get('stock_delta_ready','unreported')}"}])
