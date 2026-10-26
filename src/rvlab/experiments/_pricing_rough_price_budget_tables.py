"""Independent price-budget attribution using complete, independent seed strata.

The additive gate is deliberately unchanged. A common exactly centred shift of
both endpoints may reduce sampling error; it cannot reduce their stopping gap.
The gap combines policy error with dual slack, which this experiment cannot
separately identify.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def statistics(chunks):
    if not chunks:
        raise ValueError("Independent whole-path strata required")
    values = [np.asarray(a, float) for a in chunks]
    if any(a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all() for a in values):
        raise ValueError("Every stratum must retain at least two finite path samples")
    n = sum(map(len, values))
    estimate = float(sum(a.sum() for a in values)/n)
    se = float(np.sqrt(sum(len(a)*np.var(a, ddof=1) for a in values))/n)
    return {"estimate": estimate, "standard_error": se,
        "half_width": 1.96*se, "interval_low": estimate-1.96*se,
        "interval_high": estimate+1.96*se}


def price_budget(lowers, uppers, *, label, role):
    if len(lowers) != len(uppers) or any(np.shape(l) != np.shape(u) for l,u in zip(lowers,uppers)):
        raise ValueError("Contract-aligned lower/upper path samples required")
    if any(np.any(np.asarray(u) < np.asarray(l)-2e-11) for l,u in zip(lowers,uppers)):
        raise ValueError("The common price control changed endpoint ordering")
    p = statistics(lowers); u = statistics(uppers)
    g = statistics([np.asarray(b)-np.asarray(a) for a,b in zip(lowers,uppers)])
    allowance = .001+1e-4*abs(p["estimate"])
    mc = p["half_width"]+g["half_width"]
    budget = g["estimate"]+mc
    return {"estimator": label, "role": role, "paths": sum(map(len,lowers)),
        "price": p["estimate"], "price_se": p["standard_error"],
        "upper": u["estimate"], "upper_se": u["standard_error"],
        "gap": g["estimate"], "gap_se": g["standard_error"],
        "lower_sampling_term": p["half_width"], "gap_sampling_term": g["half_width"],
        "total_sampling_term": mc, "price_error_budget": budget,
        "price_allowance": allowance, "budget_minus_allowance": budget-allowance,
        "gap_below_allowance": g["estimate"] <= allowance,
        "gap_upper_95_below_allowance": g["interval_high"] <= allowance,
        "price_budget_pass": budget <= allowance,
        "policy_and_dual_slack_separately_identified": False,
        "american_pricing_ready": False, "stock_delta_ready": False}


def summarize_price_streams(streams, *, primary="shared control", baseline="unchanged control"):
    if not streams or len({s["seed"] for s in streams}) != len(streams):
        raise ValueError("Unique independent valuation seeds required")
    names = list(streams[0]["values"])
    if primary not in names or baseline not in names or any(list(s["values"]) != names for s in streams):
        raise ValueError("The frozen estimator order and primary must match in every stream")
    comparison, per_seed, paired, decomposition = [], [], [], []
    for name in names:
        lowers = [s["values"][name]["lower"] for s in streams]
        uppers = [s["values"][name]["upper"] for s in streams]
        record = price_budget(lowers,uppers,label=name,role="independent final valuation")
        record.update(primary=name==primary, baseline=name==baseline)
        comparison.append(record)
        for component,column in (("Mean stopping gap","gap"),
                ("Lower-price sampling uncertainty","lower_sampling_term"),
                ("Gap sampling uncertainty","gap_sampling_term")):
            decomposition.append({"estimator":name,"primary":name==primary,
                "component":component,"contribution":record[column],
                "allowance":record["price_allowance"],"budget":record["price_error_budget"],
                "price_budget_pass":record["price_budget_pass"],"paths":record["paths"]})
        for s,lower,upper in zip(streams,lowers,uppers):
            row=price_budget([lower],[upper],label=name,role="one independent seed")
            row.update(seed=s["seed"],primary=name==primary)
            per_seed.append(row)
        for metric,chunks in (("lower change",[s["values"][name]["lower"]-
                    s["values"][baseline]["lower"] for s in streams]),
                ("upper change",[s["values"][name]["upper"]-
                    s["values"][baseline]["upper"] for s in streams]),
                ("gap change",[(s["values"][name]["upper"]-s["values"][name]["lower"])-
                    (s["values"][baseline]["upper"]-s["values"][baseline]["lower"]) for s in streams])):
            paired.append({"estimator":name,"primary":name==primary,"metric":metric,
                **statistics(chunks),"paths":record["paths"]})
    frame=pd.DataFrame(comparison)
    base=frame.loc[frame.baseline].iloc[0]
    frame["price_variance_ratio_vs_baseline"]=(frame.price_se/base.price_se)**2 if base.price_se else np.nan
    frame["gap_changed_by_common_control"]=False
    for s in streams:
        base=s["values"][baseline]
        for name,values in s["values"].items():
            if not np.allclose(values["upper"]-values["lower"],base["upper"]-base["lower"],rtol=0.,atol=2e-13):
                raise ValueError("A shared estimator shift changed the pathwise stopping gap")
    return {"price_comparison":frame,"budget_decomposition":pd.DataFrame(decomposition),
        "per_seed":pd.DataFrame(per_seed),"paired_changes":pd.DataFrame(paired)}


def diagnose_spent_budget(record):
    """A planning projection only; it is never a fresh-path acceptance result."""
    slack=record["price_allowance"]-record["gap"]
    required_se=slack/1.96-record["gap_se"]
    projection=(record["paths"]*(record["total_sampling_term"]/slack)**2
        if slack>0 else np.inf)
    return {"role":"spent development evidence; no new acceptance",
        "gap":record["gap"],"price_allowance":record["price_allowance"],
        "lower_sampling_term":record["lower_sampling_term"],
        "gap_sampling_term":record["gap_sampling_term"],
        "total_sampling_term":record["total_sampling_term"],
        "price_error_budget":record["price_error_budget"],
        "allowance_left_after_mean_gap":slack,
        "required_lower_se_at_current_paths":required_se,
        "current_lower_se":record["price_se"],
        "projected_paths_with_unchanged_variance":projection,
        "variance_and_gap_held_fixed_in_projection":True,
        "sampling_causes_recorded_budget_failure":record["gap"]<record["price_allowance"]<record["price_error_budget"],
        "optimal_policy_error_identified":False}
