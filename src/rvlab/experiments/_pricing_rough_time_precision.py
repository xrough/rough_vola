"""Fresh weak-time precision checks with two finest stock bumps.

Each physical resolution has independent paths.  The old 128-step campaign is
an explicitly sealed historical baseline.  The earlier 256-step pilot never
enters the fresh 256-step estimate.  All results are finite differences of a
fixed lower policy, without an optimal-American-delta certificate.
"""
from __future__ import annotations

from numbers import Integral
import numpy as np
import pandas as pd

from ._pricing_rough_stock_delta import _summary, frozen_policy_martingale_delta_samples

POLICIES = ("original_policy", "selected_policy")
LEVELS = (.005, .0025)
RELATIVE_SPOTS = np.array([-.005, -.0025, 0., .0025, .005])
TOLERANCES = {"price_atol": .001, "price_rtol": .0001,
    "delta_atol": .002, "delta_rtol": .005,
    "confidence_multiplier": 1.96, "spread_fraction": .1}


def new_policy_time_samples(bundle, *, seed, paths, steps):
    if bundle["manifest"]["selected_name"] != bundle["selected_name"]:
        raise ValueError("Selected policy differs from the sealed identity")
    if seed in bundle["manifest"]["all_selection_training_seeds"]:
        raise ValueError("Time-precision paths must be independent of fitting and selection")
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps not in (256, 512):
        raise ValueError("Fresh time precision uses 256 or 512 physical steps")
    frozen = {"proxy": bundle["proxy"], "weights": bundle["weights"],
        "original_policy": bundle["previous_selected_policy"], "selected_policy": bundle["selected_policy"]}
    result = frozen_policy_martingale_delta_samples(frozen, seed=seed, paths=paths,
        steps=int(steps), bump_relatives=LEVELS)
    result["diagnostics"].update(selected_policy_name=bundle["selected_name"],
        previous_policy_name="previous dense tensor", policy_selection_used_delta_paths=False,
        optimal_value_delta_certified=False, old_256_pilot_pooled=False,
        five_stock_bumps=True, fresh_resolutions_independent=True)
    return result


def flatten_samples(result):
    arrays = {policy + "__" + key: values
        for policy, samples in result["samples"].items() for key, values in samples.items()}
    arrays.update({policy + "__stops": values for policy, values in result["stops"].items()})
    arrays["relative_spots"] = result["relative_spots"]
    return arrays


def validate_time_arrays(arrays, *, paths, spot=100.):
    if not np.array_equal(arrays.get("relative_spots"), RELATIVE_SPOTS):
        raise ValueError("Time precision requires the five predeclared stock levels")
    expected = {"relative_spots"}
    for policy in POLICIES:
        expected.update(policy + "__" + key for key in ("raw_prices", "controlled_prices", "stops"))
        stops = arrays.get(policy + "__stops", np.array([]))
        if (stops.shape != (paths, 5) or not np.issubdtype(stops.dtype, np.integer)
                or np.any(stops < 0) or np.any(stops > 64)):
            raise ValueError("Invalid five-stock stopping indices")
        for estimator, label in (("raw", "raw_prices"), ("stopped martingale", "controlled_prices")):
            prices = arrays.get(policy + "__" + label, np.array([]))
            if (prices.shape != (paths, 5) or not np.isfinite(prices).all()
                    or (estimator == "raw" and np.any(prices < 0))):
                raise ValueError("Invalid five-stock payoff samples")
            for level in LEVELS:
                key = policy + "__" + f"{estimator}:{level:g}"
                expected.add(key)
                value = arrays.get(key, np.array([]))
                lo = int(np.flatnonzero(RELATIVE_SPOTS == -level)[0])
                hi = int(np.flatnonzero(RELATIVE_SPOTS == level)[0])
                if (value.shape != (paths,) or not np.isfinite(value).all()
                        or not np.allclose(value, (prices[:, hi] - prices[:, lo]) / (2 * spot * level),
                            atol=2e-14, rtol=0.)):
                    raise ValueError("Time delta differs from its paired stock payoffs")
    if set(arrays) != expected:
        raise ValueError("Missing or unexpected time-precision arrays")


def summarize_time_arrays(arrays):
    estimates, comparisons = [], []
    for policy in POLICIES:
        for estimator in ("raw", "stopped martingale"):
            for level in LEVELS:
                values = arrays[policy + "__" + f"{estimator}:{level:g}"]
                summary = _summary(values); allowance = .002 + .005 * abs(summary["mean"])
                estimates.append({"policy": policy, "estimator": estimator,
                    "bump_relative": level, "paths": len(values), **summary,
                    "delta_allowance": allowance, "sampling_passed": summary["ci_half_width"] <= allowance,
                    "optimal_delta_ready": False})
            fine = arrays[policy + "__" + f"{estimator}:0.0025"]
            coarse = arrays[policy + "__" + f"{estimator}:0.005"]
            summary = _summary(fine - coarse); allowance = .002 + .005 * abs(float(fine.mean()))
            budget = abs(summary["mean"]) + summary["ci_half_width"]
            comparisons.append({"comparison": "bump refinement", "policy": policy,
                "estimator": estimator, "coarse_bump": .005, "fine_bump": .0025,
                **summary, "delta_allowance": allowance, "change_plus_uncertainty": budget,
                "comparison_passed": budget <= allowance})
    for estimator in ("raw", "stopped martingale"):
        for level in LEVELS:
            key = f"{estimator}:{level:g}"
            values = arrays["selected_policy__" + key] - arrays["original_policy__" + key]
            comparisons.append({"comparison": "selected minus original", "policy": "paired policies",
                "estimator": estimator, "coarse_bump": np.nan, "fine_bump": level, **_summary(values)})
    return pd.DataFrame(estimates), pd.DataFrame(comparisons)


def independent_resolution_comparison(coarse, fine, *, coarse_steps, fine_steps):
    if (isinstance(coarse_steps, bool) or isinstance(fine_steps, bool)
            or not isinstance(coarse_steps, Integral) or not isinstance(fine_steps, Integral)
            or not 0 < coarse_steps < fine_steps):
        raise ValueError("Independent physical resolutions must strictly increase")
    keys = ["policy", "estimator", "bump_relative"]
    if coarse.duplicated(keys).any() or fine.duplicated(keys).any() or len(coarse) != 8 or len(fine) != 8:
        raise ValueError("Independent resolution tables require eight unique matching rows")
    rows = []
    for _, f in fine.iterrows():
        match = coarse[(coarse.policy == f.policy) & (coarse.estimator == f.estimator)
            & (coarse.bump_relative == f.bump_relative)]
        if len(match) != 1:
            raise ValueError("Independent resolution tables have unmatched contracts")
        c = match.iloc[0]; change = float(f["mean"] - c["mean"])
        se = float(np.hypot(f.standard_error, c.standard_error))
        allowance = .002 + .005 * abs(float(f["mean"]))
        budget = abs(change) + 1.96 * se
        rows.append({"policy": f.policy, "estimator": f.estimator,
            "bump_relative": f.bump_relative, "coarse_steps": int(coarse_steps),
            "fine_steps": int(fine_steps), "coarse_paths": int(c.paths), "fine_paths": int(f.paths),
            "mean": change, "standard_error": se, "ci_half_width": 1.96 * se,
            "delta_allowance": allowance, "change_plus_uncertainty": budget,
            "comparison_passed": budget <= allowance, "independent_streams": True,
            "same_exercise_calendar": True, "optimal_delta_ready": False})
    return pd.DataFrame(rows)
