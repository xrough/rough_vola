"""Whole-path statistics for every predeclared policy/control combination."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pricing_rough_policy_validation_stopping import _summary


def selection_statistics(training, controls, policies, *, training_seed):
    """Reconstruct held-out selection; no final-path outcomes enter these scores."""
    control_rows, policy_rows = [], []
    for name, control in controls.items():
        value = training["values"][name]
        lower = value["lower"][2048:, 0]
        gap = value["upper"][2048:]-lower
        price, pse, _, _ = _summary([lower])
        mean, se, _, _ = _summary([gap])
        control_rows.append({"control": name, "family": control["family"], "heldout_paths": 2048,
            "price": price, "price_se": pse, "gap": mean, "gap_se": se, "target_final_paths": 8192,
            "selection_score": mean+1.96*.5*(se+pse), "final_paths_used_for_selection": False})
    control_table = pd.DataFrame(control_rows)
    selected_control = control_table.loc[control_table.selection_score.idxmin(), "control"]
    control_table["selected_before_final"] = control_table.control == selected_control
    lower = training["values"]["current baseline"]["lower"][2048:]
    stops = training["stops"][2048:]
    chosen, best = policies[0], 0.
    for p, policy in enumerate(policies):
        mean, se, _, _ = _summary([lower[:, p]-lower[:, 0]])
        score = mean-1.96*se
        policy_rows.append({"policy": policy, "paired_lower_gain": mean, "paired_lower_gain_se": se,
            "selection_score": score, "exercise_change_fraction": float(np.mean(stops[:, p] != stops[:, 0])),
            "selection_seed": training_seed, "final_paths_used_for_selection": False})
        if score > best:
            chosen, best = policy, score
    policy_table = pd.DataFrame(policy_rows)
    policy_table["selected_before_final"] = policy_table.policy == chosen
    families = {}
    for family in ("poststop", "extended"):
        eligible = control_table.loc[control_table.family.isin(("fallback", family))]
        families[family] = eligible.loc[eligible.selection_score.idxmin(), "control"]
    return control_table, policy_table, selected_control, chosen, families


def final_statistics(streams, controls, policies, selected_control, selected_policy):
    from ._pricing_rough_tail_attribution import _dominance_statistics
    rows, paired, per_seed, tails = [], [], [], []
    for name, control in controls.items():
        for p, policy in enumerate(policies):
            lower = [s["values"][name]["lower"][:, p] for s in streams]
            upper = [s["values"][name]["upper"] for s in streams]
            gaps = [u-l for u, l in zip(upper, lower)]
            price, pse, _, _ = _summary(lower)
            up, use, _, _ = _summary(upper)
            gap, gse, _, _ = _summary(gaps)
            allowance, budget = .001+1e-4*abs(price), gap+1.96*(pse+gse)
            flags = {"control": name, "family": control["family"], "policy": policy,
                "selected_control": name == selected_control, "selected_policy": policy == selected_policy}
            rows.append({**flags, "paths": 8192, "streams": 2, "price": price, "price_se": pse,
                "upper": up, "upper_se": use, "gap": gap, "gap_se": gse,
                "price_error_budget": budget, "price_allowance": allowance, "price_budget_pass": budget <= allowance,
                "american_pricing_ready": False, "new_rule_stock_delta_validated": False})
            sequences = {"lower_change": [], "upper_change": [], "gap_change": []}
            for stream, lo, hi in zip(streams, lower, upper):
                baseline = stream["values"]["current baseline"]
                bl, bu = baseline["lower"][:, 0], baseline["upper"]
                sequences["lower_change"].append(lo-bl)
                sequences["upper_change"].append(hi-bu)
                sequences["gap_change"].append((hi-lo)-(bu-bl))
            row = flags.copy()
            for key, chunks in sequences.items():
                mean, se, low, high = _summary(chunks)
                row.update({key: mean, key+"_se": se, key+"_low": low, key+"_high": high})
            paired.append(row)
            for stream, lo, hi in zip(streams, lower, upper):
                value, se, _, _ = _summary([lo])
                gm, gs, _, _ = _summary([hi-lo])
                per_seed.append({**flags, "seed": stream["seed"], "paths": len(lo), "price": value,
                    "price_se": se, "upper": float(hi.mean()), "gap": gm, "gap_se": gs})
            if p == 0 or (name == selected_control and policy == selected_policy):
                all_gaps = np.concatenate(gaps)
                total, ordered = float(all_gaps.sum()), np.sort(all_gaps)
                tails.append({**flags, "paths": len(all_gaps), "zero_gap_fraction": float(np.mean(all_gaps <= 1e-12)),
                    "gap_p90": float(np.quantile(all_gaps, .9)), "gap_p99": float(np.quantile(all_gaps, .99)),
                    "gap_maximum": float(all_gaps.max()),
                    "largest_decile_gap_share": float(ordered[-int(np.ceil(.1*len(all_gaps))):].sum()/total) if total else 0.,
                    "largest_percentile_gap_share": float(ordered[-int(np.ceil(.01*len(all_gaps))):].sum()/total) if total else 0.,
                    "used_for_selection": False, "acceptance_input": False})
    contrasts = _dominance_statistics(streams, policies.index(selected_policy), selected_control,
        baseline_policy=0, baseline_control="current baseline")
    dominance = []
    for challenge, gain_key, majority_key in (("dual", "dual upper decrease", "dual decrease minus half baseline gap"),
            ("policy", "policy lower gain", "policy gain minus half baseline gap")):
        gain = contrasts.loc[contrasts.contrast == gain_key].iloc[0]
        majority = contrasts.loc[contrasts.contrast == majority_key].iloc[0]
        dominance.append({"challenge": challenge, "selected_control": selected_control, "selected_policy": selected_policy,
            "bound_improvement": gain["mean"], "bound_improvement_se": gain.standard_error,
            "improvement_low": gain.interval_low, "improvement_high": gain.interval_high,
            "majority_margin": majority["mean"], "majority_margin_se": majority.standard_error,
            "majority_margin_low": majority.interval_low, "majority_margin_high": majority.interval_high,
            "majority_dominance_proved": bool(majority.resolved_positive),
            "remaining_source_unresolved": not bool(majority.resolved_positive), "confidence_multiplier": 1.96})
    return {"comparison": pd.DataFrame(rows), "paired_changes": pd.DataFrame(paired),
        "per_seed": pd.DataFrame(per_seed), "gap_distribution": pd.DataFrame(tails),
        "dominance": pd.DataFrame(dominance), "majority_contrasts": contrasts}


__all__ = ["selection_statistics", "final_statistics"]
