"""Whole-path paired statistics for frozen Bellman-policy challenges."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pricing_rough_policy_validation_stopping import _summary


def selection_statistics(stream, policy_specs, *, seed):
    names = list(policy_specs)
    rows = []
    selected, best = names[0], 0.
    challenger, challenger_score = None, -np.inf
    for index, name in enumerate(names):
        gain = stream["lower"][:, index] - stream["lower"][:, 0]
        mean, se, low, high = _summary([gain])
        rows.append({"policy": name, **policy_specs[name], "paths": len(gain),
            "paired_lower_gain": mean, "paired_lower_gain_se": se,
            "gain_interval_low": low, "gain_interval_high": high, "selection_score": low,
            "exercise_change_fraction": float(np.mean(stream["stops"][:, index] != stream["stops"][:, 0])),
            "selection_seed": seed, "final_paths_used_for_selection": False})
        if low > best:
            selected, best = name, low
        if index and low > challenger_score:
            challenger, challenger_score = name, low
    if challenger is None:
        raise ValueError("Bellman selection needs a nonbaseline challenger")
    for row in rows:
        row.update(selected_policy=row["policy"] == selected,
            selected_challenger=row["policy"] == challenger)
    return pd.DataFrame(rows), selected, challenger


def final_statistics(streams, policy_specs, *, selected_policy, selected_challenger):
    """Common-M improvements and covariance-aware half-gap contrasts.

    Equal stream sizes are predeclared by the campaign. Standard errors retain
    the independent-stream stratification rather than pooling between seeds.
    No final statistic chooses a policy.
    """
    names = list(policy_specs)
    comparison, paired, per_seed, policy_error, tails, exercise = [], [], [], [], [], []
    total = sum(len(s["lower"]) for s in streams)
    for index, name in enumerate(names):
        flags = {"policy": name, **policy_specs[name], "selected_policy": name == selected_policy,
            "selected_challenger": name == selected_challenger}
        lower = [s["lower"][:, index] for s in streams]
        upper = [s["upper"] for s in streams]
        gaps = [u-l for u, l in zip(upper, lower)]
        price, pse, _, _ = _summary(lower)
        u, use, _, _ = _summary(upper)
        gap, gse, _, _ = _summary(gaps)
        allowance = .001+1e-4*abs(price)
        budget = gap+1.96*(pse+gse)
        comparison.append({**flags, "paths": total, "streams": len(streams),
            "price": price, "price_se": pse, "upper": u, "upper_se": use,
            "gap": gap, "gap_se": gse, "price_error_budget": budget, "price_allowance": allowance,
            "price_budget_pass": budget <= allowance, "common_upper_identical": True,
            "american_pricing_ready": False, "new_rule_stock_delta_validated": False})
        gains = [s["lower"][:, index]-s["lower"][:, 0] for s in streams]
        raw_gains = [s["raw"][:, index]-s["raw"][:, 0] for s in streams]
        margins = [gain-.5*(s["upper"]-s["lower"][:, 0]) for s, gain in zip(streams, gains)]
        row = dict(flags)
        for label, chunks in (("lower_gain", gains), ("raw_gain", raw_gains),
                ("gap_change", [-gain for gain in gains])):
            mean, se, low, high = _summary(chunks)
            row.update({label: mean, label+"_se": se, label+"_low": low, label+"_high": high})
        row["exercise_change_fraction"] = float(np.mean(np.concatenate([
            s["stops"][:, index] != s["stops"][:, 0] for s in streams])))
        paired.append(row)
        gain, gainse, low, high = _summary(gains)
        margin, marginse, ml, mh = _summary(margins)
        policy_error.append({**flags, "identified_policy_gain": gain,
            "gain_standard_error": gainse, "gain_interval_low": low, "gain_interval_high": high,
            "improvement_resolved": low > 0., "majority_margin": margin,
            "majority_margin_se": marginse, "majority_margin_low": ml, "majority_margin_high": mh,
            "policy_majority_proved": ml > 0., "confidence_multiplier": 1.96,
            "full_policy_error_identified": False, "same_martingale": True})
        for s, l, g, gain_seq, raw_gain in zip(streams, lower, gaps, gains, raw_gains):
            lm, ls, _, _ = _summary([l]); gm, gs, _, _ = _summary([g])
            mean, se, gl, gh = _summary([gain_seq])
            rm, rs, rl, rh = _summary([raw_gain])
            per_seed.append({**flags, "seed": s["seed"], "paths": len(l),
                "price": lm, "price_se": ls, "upper": float(s["upper"].mean()),
                "gap": gm, "gap_se": gs, "lower_gain": mean, "lower_gain_se": se,
                "lower_gain_low": gl, "lower_gain_high": gh,
                "raw_gain": rm, "raw_gain_se": rs, "raw_gain_low": rl, "raw_gain_high": rh})
        all_gaps = np.concatenate(gaps)
        baseline_gaps = np.concatenate([s["upper"]-s["lower"][:, 0] for s in streams])
        changes = all_gaps-baseline_gaps
        ordered, mass = np.sort(all_gaps), all_gaps.sum()
        tails.append({**flags, "paths": total, "zero_gap_fraction": float(np.mean(all_gaps <= 1e-12)),
            "gap_p90": float(np.quantile(all_gaps, .9)), "gap_p99": float(np.quantile(all_gaps, .99)),
            "gap_maximum": float(all_gaps.max()),
            "largest_decile_gap_share": float(ordered[-int(np.ceil(.1*total)):].sum()/mass) if mass else 0.,
            "largest_percentile_gap_share": float(ordered[-int(np.ceil(.01*total)):].sum()/mass) if mass else 0.,
            "path_gaps_worsened_fraction": float(np.mean(changes > 1e-12)),
            "path_gaps_improved_fraction": float(np.mean(changes < -1e-12)),
            "maximum_path_gap_increase": float(changes.max()),
            "used_for_selection": False, "acceptance_input": False})
        stops = np.concatenate([s["stops"][:, index] for s in streams])
        for date in range(65):
            exercise.append({**flags, "stop_index": date, "paths": int(np.sum(stops == date)),
                "fraction": float(np.mean(stops == date)), "descriptive_only": True})
    result = {"comparison": pd.DataFrame(comparison), "paired_changes": pd.DataFrame(paired),
        "per_seed": pd.DataFrame(per_seed), "policy_error": pd.DataFrame(policy_error),
        "tail_distribution": pd.DataFrame(tails), "exercise_times": pd.DataFrame(exercise)}
    if {s.get("order") for s in policy_specs.values()} >= {8,16,32}:
        refinements = []
        lookup = {(s["order"],s["blend"]):i for i,s in enumerate(policy_specs.values())}
        for blend in (.5,1.):
            for coarse,fine in ((8,16),(16,32)):
                a,b = lookup[(coarse,blend)],lookup[(fine,blend)]
                mean,se,low,high = _summary([s["lower"][:,b]-s["lower"][:,a] for s in streams])
                price,_,_,_ = _summary([s["lower"][:,b] for s in streams])
                allowance = .001+1e-4*abs(price)
                refinements.append({"coarse_order":coarse,"fine_order":fine,"blend":blend,
                    "price_change":mean,"price_change_se":se,"change_low":low,"change_high":high,
                    "change_plus_uncertainty":abs(mean)+1.96*se,"price_allowance":allowance,
                    "finite_policy_price_comparison_pass":abs(mean)+1.96*se <= allowance,
                    "Bellman_quadrature_certified":False})
        result["quadrature_policy_refinement"] = pd.DataFrame(refinements)
    return result


def continuation_diagnostics(streams, orders):
    rows = []
    for coarse in range(len(orders)-1):
        differences, decisions, values = [], [], []
        for s in streams:
            mask = s["computed_mask"]
            a, b = s["continuation"][..., coarse][mask], s["continuation"][..., coarse+1][mask]
            p = s["intrinsic"][mask]
            differences.append(b-a)
            decisions.append((p >= a) != (p >= b))
            values.append(b)
        diff, changed, fine = map(np.concatenate, (differences, decisions, values))
        rows.append({"coarse_order": orders[coarse], "fine_order": orders[coarse+1],
            "computed_states": len(diff), "mean_signed_change": float(np.mean(diff)),
            "mean_absolute_change": float(np.mean(abs(diff))),
            "absolute_change_p99": float(np.quantile(abs(diff), .99)),
            "maximum_absolute_change": float(np.max(abs(diff))),
            "one_step_decision_change_fraction": float(np.mean(changed)),
            "within_original_price_allowance_fraction": float(np.mean(abs(diff) <= .001+1e-4*abs(fine))),
            "descriptive_only": True, "certified_operator_error_bound": False,
            "used_for_selection": False})
    return pd.DataFrame(rows)
