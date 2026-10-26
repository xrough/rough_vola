"""Paired bump checks and independent time-law checks for fixed Bellman rules."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pricing_rough_policy_validation_stopping import _summary

POLICIES = ("baseline", "bellman")
ESTIMATORS = ("raw", "controlled")
BUMPS = (.005, .0025)


def delta_samples(stream, policy, estimator, bump):
    relative = np.asarray(stream["relative_spots"])
    a = int(np.flatnonzero(relative == -bump)[0])
    b = int(np.flatnonzero(relative == bump)[0])
    return (stream[policy][estimator][:, b] - stream[policy][estimator][:, a]) / (200. * bump)


def _row(chunks):
    mean, se, low, high = _summary(chunks)
    return {"estimate": mean, "standard_error": se, "interval_low": low,
        "interval_high": high, "half_width": 1.96 * se}


def summarize_delta_streams(streams):
    """All paths and fixed strata enter; final data cannot choose a bump or rule."""
    if not streams:
        raise ValueError("No fixed-rule delta streams supplied")
    seeds = [s["seed"] for s in streams]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Independent streams cannot reuse a valuation seed")
    estimates, bumps, controls, policies, costs = [], [], [], [], []
    for steps in sorted({s["steps"] for s in streams}):
        group = [s for s in streams if s["steps"] == steps]
        count = sum(len(s["baseline"]["raw"]) for s in group)
        flags = {"physical_steps": steps, "paths": count, "streams": len(group),
            "exercise_intervals": 64, "policy_operator_steps": 128,
            "optimal_delta_ready": False}
        for policy in POLICIES:
            for estimator in ESTIMATORS:
                for bump in BUMPS:
                    chunks = [delta_samples(s, policy, estimator, bump) for s in group]
                    row = {**flags, "policy": policy, "estimator": estimator,
                        "bump_relative": bump, **_row(chunks)}
                    allowance = .002 + .005 * abs(row["estimate"])
                    row.update(delta_allowance=allowance,
                        sampling_pass=row["half_width"] <= allowance)
                    estimates.append(row)
                diff = [delta_samples(s, policy, estimator, BUMPS[1]) -
                    delta_samples(s, policy, estimator, BUMPS[0]) for s in group]
                row = {**flags, "policy": policy, "estimator": estimator,
                    "coarse_bump": BUMPS[0], "fine_bump": BUMPS[1], **_row(diff)}
                fine = _summary([delta_samples(s, policy, estimator, BUMPS[1]) for s in group])[0]
                budget = abs(row["estimate"]) + row["half_width"]
                allowance = .002 + .005 * abs(fine)
                row.update(change_plus_uncertainty=budget, delta_allowance=allowance,
                    bump_check_pass=budget <= allowance)
                bumps.append(row)
            for bump in BUMPS:
                row = {**flags, "policy": policy, "bump_relative": bump,
                    **_row([delta_samples(s, policy, "raw", bump) -
                        delta_samples(s, policy, "controlled", bump) for s in group])}
                ref = _summary([delta_samples(s, policy, "controlled", bump) for s in group])[0]
                row.update(delta_allowance=.002+.005*abs(ref),
                    tolerance_bias_resolved=abs(row["estimate"])+row["half_width"] <= .002+.005*abs(ref),
                    zero_in_interval=row["interval_low"] <= 0 <= row["interval_high"])
                controls.append(row)
        for estimator in ESTIMATORS:
            for bump in BUMPS:
                policies.append({**flags, "estimator": estimator, "bump_relative": bump,
                    **_row([delta_samples(s, "bellman", estimator, bump) -
                        delta_samples(s, "baseline", estimator, bump) for s in group])})
        for s in group:
            for policy in POLICIES:
                for bump in BUMPS:
                    relative = np.asarray(s["relative_spots"])
                    a, b = [int(np.flatnonzero(relative == x)[0]) for x in (-bump, bump)]
                    costs.append({"physical_steps": steps, "seed": s["seed"],
                        "paths": len(s[policy]["raw"]), "policy": policy, "bump_relative": bump,
                        "exercise_change_fraction": float(np.mean(s[policy]["stops"][:, a] != s[policy]["stops"][:, b])),
                        **_row([delta_samples(s, policy, "controlled", bump)])})
    sampling = pd.DataFrame(estimates)
    time = []
    resolutions = sorted(sampling.physical_steps.unique())
    for coarse, fine in zip(resolutions[:-1], resolutions[1:]):
        for policy in POLICIES:
            for estimator in ESTIMATORS:
                for bump in BUMPS:
                    match = sampling[(sampling.policy == policy) & (sampling.estimator == estimator)
                        & (sampling.bump_relative == bump)]
                    a, b = [match[match.physical_steps == n].iloc[0] for n in (coarse, fine)]
                    change = float(b.estimate - a.estimate)
                    se = float(np.hypot(a.standard_error, b.standard_error))
                    allowance = .002+.005*abs(float(b.estimate))
                    time.append({"policy": policy, "estimator": estimator, "bump_relative": bump,
                        "coarse_steps": int(coarse), "fine_steps": int(fine),
                        "coarse_paths": int(a.paths), "fine_paths": int(b.paths),
                        "delta_change": change, "standard_error": se,
                        "interval_low": change-1.96*se, "interval_high": change+1.96*se,
                        "change_plus_uncertainty": abs(change)+1.96*se,
                        "delta_allowance": allowance, "time_check_pass": abs(change)+1.96*se <= allowance,
                        "independent_resolutions": True, "same_frozen_rule": True,
                        "same_exercise_calendar": True, "optimal_delta_ready": False})
    return {"delta_sampling": sampling, "bump_refinement": pd.DataFrame(bumps),
        "raw_control": pd.DataFrame(controls), "policy_delta_change": pd.DataFrame(policies),
        "delta_per_seed": pd.DataFrame(costs), "time_refinement": pd.DataFrame(time)}
