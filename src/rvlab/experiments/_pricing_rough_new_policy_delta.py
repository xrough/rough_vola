"""Independent stock sensitivities of the explicitly sealed larger-fit policy.

This adapter keeps the audited stopped-martingale estimator unchanged.  The
new rule and its predecessor are reapplied at each stock bump.  None of these
finite differences certifies the derivative of the optimal American value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pricing_rough_stock_delta import (
    _summary, frozen_policy_martingale_delta_samples)


POLICIES = ("original_policy", "selected_policy")
LEVELS = (.01, .005, .0025)
TOLERANCES = {"price_atol": .001, "price_rtol": .0001,
    "delta_atol": .002, "delta_rtol": .005,
    "confidence_multiplier": 1.96, "spread_fraction": .1}


def new_policy_delta_samples(bundle, *, seed, paths, steps):
    """Replay both frozen rules with one independent multistock martingale."""
    record = bundle["manifest"]
    if record["selected_name"] != bundle["selected_name"]:
        raise ValueError("The new selected policy differs from its sealed identity")
    if seed in record["all_selection_training_seeds"]:
        raise ValueError("New-policy delta paths must be independent of selection and training")
    policies = {"proxy": bundle["proxy"], "weights": bundle["weights"],
        "original_policy": bundle["previous_selected_policy"],
        "selected_policy": bundle["selected_policy"]}
    result = frozen_policy_martingale_delta_samples(policies, seed=seed,
        paths=paths, steps=steps, bump_relatives=LEVELS)
    result["diagnostics"].update({"selected_policy_name": bundle["selected_name"],
        "previous_policy_name": "previous dense tensor",
        "policy_selection_used_delta_paths": False,
        "optimal_value_delta_certified": False,
        "method_reference_checks": "historical, separately labelled; not new-policy measurements"})
    return result


def flatten_samples(result):
    arrays = {name + "__" + key: values
        for name, samples in result["samples"].items() for key, values in samples.items()}
    arrays.update({name + "__stops": values for name, values in result["stops"].items()})
    arrays["relative_spots"] = result["relative_spots"]
    return arrays


def validate_arrays(arrays, *, paths, spot=100., exercise_steps=64):
    """Validate that saved finite differences really use all bumped payoffs."""
    relative = np.array([-.01, -.005, -.0025, 0., .0025, .005, .01])
    np.testing.assert_array_equal(arrays["relative_spots"], relative)
    expected = {"relative_spots"}
    for policy in POLICIES:
        for key in ("raw_prices", "controlled_prices", "stops"):
            expected.add(policy + "__" + key)
        for estimator in ("raw", "stopped martingale"):
            for level in LEVELS:
                expected.add(policy + "__" + f"{estimator}:{level:g}")
    if set(arrays) != expected:
        raise ValueError("New-policy sample archive has missing or unexpected arrays")
    for policy in POLICIES:
        stops = arrays[policy + "__stops"]
        if (stops.shape != (paths, 7) or not np.issubdtype(stops.dtype, np.integer)
                or np.any(stops < 0) or np.any(stops > exercise_steps)):
            raise ValueError("Invalid new-policy stopping indices")
        for estimator, key in (("raw", "raw_prices"), ("stopped martingale", "controlled_prices")):
            prices = arrays[policy + "__" + key]
            if prices.shape != (paths, 7) or not np.isfinite(prices).all():
                raise ValueError("Invalid new-policy bumped price samples")
            for level in LEVELS:
                values = arrays[policy + "__" + f"{estimator}:{level:g}"]
                if values.shape != (paths,) or not np.isfinite(values).all():
                    raise ValueError("Invalid new-policy delta samples")
                lo = int(np.flatnonzero(relative == -level)[0])
                hi = int(np.flatnonzero(relative == level)[0])
                np.testing.assert_allclose(values, (prices[:, hi] - prices[:, lo]) / (2 * spot * level),
                    rtol=0., atol=2e-14)


def summarize_arrays(arrays):
    """Whole-path uncertainty, paired bump changes and paired policy changes."""
    rows, comparisons = [], []
    for policy in POLICIES:
        for estimator in ("raw", "stopped martingale"):
            for level in LEVELS:
                sample = arrays[policy + "__" + f"{estimator}:{level:g}"]
                summary = _summary(sample)
                allowance = .002 + .005 * abs(summary["mean"])
                rows.append({"policy": policy, "estimator": estimator,
                    "bump_relative": level, "paths": len(sample), **summary,
                    "delta_allowance": allowance,
                    "sampling_passed": summary["ci_half_width"] <= allowance,
                    "optimal_delta_ready": False})
            for coarse, fine in zip(LEVELS[:-1], LEVELS[1:]):
                values = (arrays[policy + "__" + f"{estimator}:{fine:g}"]
                    - arrays[policy + "__" + f"{estimator}:{coarse:g}"])
                summary = _summary(values)
                fine_mean = float(arrays[policy + "__" + f"{estimator}:{fine:g}"].mean())
                allowance = .002 + .005 * abs(fine_mean)
                comparisons.append({"comparison": "bump refinement", "policy": policy,
                    "estimator": estimator, "coarse_bump": coarse, "fine_bump": fine,
                    **summary, "delta_allowance": allowance,
                    "change_plus_uncertainty": abs(summary["mean"]) + summary["ci_half_width"],
                    "comparison_passed": abs(summary["mean"]) + summary["ci_half_width"] <= allowance})
    for estimator in ("raw", "stopped martingale"):
        for level in LEVELS:
            values = (arrays[POLICIES[1] + "__" + f"{estimator}:{level:g}"]
                - arrays[POLICIES[0] + "__" + f"{estimator}:{level:g}"])
            comparisons.append({"comparison": "selected minus original", "policy": "paired policies",
                "estimator": estimator, "coarse_bump": np.nan, "fine_bump": level,
                **_summary(values)})
    return pd.DataFrame(rows), pd.DataFrame(comparisons)


def independent_time_comparison(base, refined):
    """Different physical laws are independent, never falsely paired/pooled."""
    rows = []
    for _, fine in refined.iterrows():
        match = base[(base.policy == fine.policy) & (base.estimator == fine.estimator)
            & (base.bump_relative == fine.bump_relative)]
        if len(match) != 1:
            raise ValueError("Independent time comparison requires unique matching contracts")
        coarse = match.iloc[0]
        change = float(fine["mean"] - coarse["mean"])
        se = float(np.hypot(fine.standard_error, coarse.standard_error))
        allowance = .002 + .005 * abs(float(fine["mean"]))
        rows.append({"policy": fine.policy, "estimator": fine.estimator,
            "bump_relative": fine.bump_relative, "coarse_steps": 128, "fine_steps": 256,
            "mean": change, "standard_error": se, "ci_half_width": 1.96 * se,
            "delta_allowance": allowance, "change_plus_uncertainty": abs(change) + 1.96 * se,
            "comparison_passed": abs(change) + 1.96 * se <= allowance,
            "independent_streams": True, "same_exercise_calendar": True,
            "optimal_delta_ready": False})
    return pd.DataFrame(rows)
