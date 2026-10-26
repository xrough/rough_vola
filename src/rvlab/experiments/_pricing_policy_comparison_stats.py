"""Conditional, paired statistics for the predeclared exercise-policy study.

Training replicas are averaged *within each valuation path* before Monte Carlo
uncertainty is calculated.  These intervals condition on the frozen fits; two
training replicas do not estimate general training-sample uncertainty.  The
common upper estimator is shared across policies, so a paired gap reduction is
exactly a controlled policy-value gain, not independent dual improvement.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd


POLICIES = ("cashflow_lsm", "fourier_premium_lsm", "bellman_lookahead")
CONTROLS = ("raw", "european", "european_premium")
CONTRASTS = (("fourier_premium_lsm", "cashflow_lsm"),
    ("bellman_lookahead", "fourier_premium_lsm"))
CONFIDENCE_MULTIPLIER = 1.96
PRICE_ATOL = .001
PRICE_RTOL = 1e-4


def _moments(chunks):
    """Weighted mean and independent-stream, within-stream variance estimate.

    This is the same fixed-stream convention used by the existing benchmark:
    Var(mean) = sum(n_s * sample_variance_s) / sum(n_s)**2.  Independent streams
    may have different declared path counts; neither stream is given the weight
    of a single path or mistaken for a training replication.
    """
    arrays = [np.asarray(x, float) for x in chunks]
    if not arrays or any(x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all() for x in arrays):
        raise ValueError("finite independent valuation streams with at least two paths are required")
    n = sum(len(x) for x in arrays)
    mean = float(sum(np.sum(x) for x in arrays)/n)
    variance = float(sum(len(x)*np.var(x, ddof=1) for x in arrays)/n)
    se = float(np.sqrt(variance/n))
    half = CONFIDENCE_MULTIPLIER*se
    return {"paths": n, "estimate": mean, "standard_error": se,
        "interval_low": mean-half, "interval_high": mean+half,
        "half_width": half, "within_stream_variance": variance}


def _validate(samples, replica_ids):
    if (not isinstance(samples, (list, tuple)) or len(samples) != 2
            or not isinstance(replica_ids, (list, tuple)) or len(replica_ids) != 2
            or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) or x < 0
                for x in replica_ids) or len(set(replica_ids)) != 2):
        raise ValueError("exactly two valuation streams and two distinct nonnegative training replica IDs are required")
    result, reference_times = [], None
    for source in samples:
        if not isinstance(source, dict):
            raise ValueError("each valuation stream must be an explicit sample dictionary")
        try:
            times = np.asarray(source["times"], float)
            stops = np.asarray(source["stop_indices"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("each stream needs its exercise calendar and own stopping indices") from error
        if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
                or times[0] != 0 or not np.all(np.diff(times) > 0)):
            raise ValueError("the exercise calendar must start at zero and increase strictly")
        if reference_times is None:
            reference_times = times
        elif not np.array_equal(times, reference_times):
            raise ValueError("both streams must share exactly the same exercise calendar")
        if (stops.ndim != 4 or stops.shape[0] < 2 or stops.shape[1:] != (2, 3, 1)
                or stops.dtype.kind not in "iu" or np.any(stops < 0) or np.any(stops >= len(times))):
            raise ValueError("own stopping indices must have shape (paths, 2 replicas, 3 policies, 1 stock)")
        if "policy_names" in source and tuple(np.asarray(source["policy_names"]).tolist()) != POLICIES:
            raise ValueError("the declared policy order changed")
        if "replica_ids" in source and tuple(np.asarray(source["replica_ids"]).tolist()) != tuple(replica_ids):
            raise ValueError("the declared training replica order changed")
        stream = {"times": times, "stop_indices": stops}
        for control in CONTROLS:
            try:
                lower, upper = (np.asarray(source[control+suffix], float)
                    for suffix in ("_lower", "_upper"))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("all three explicit lower/common-upper control samples are required") from error
            if (lower.shape != stops.shape or upper.shape != (len(stops), 1)
                    or not np.isfinite(lower).all() or not np.isfinite(upper).all()
                    or np.any(lower > upper[:, None, None, :]+2e-11)):
                raise ValueError("finite pathwise-ordered endpoints with a single shared upper are required")
            stream[control+"_lower"], stream[control+"_upper"] = lower[..., 0], upper[:, 0]
        result.append(stream)
    return result


def _scope(case_id, arm, replica_ids):
    if not isinstance(case_id, str) or not case_id or not isinstance(arm, str) or not arm:
        raise ValueError("nonempty case and budget-arm identities are required")
    return {"case_id": case_id, "arm": arm, "training_replicas": len(replica_ids),
        "conditional_on_frozen_training": True, "training_uncertainty_in_interval": False,
        "pointwise_interval": True, "used_for_policy_selection": False,
        "optimal_continuous_price_certified": False}


def summarize_case(samples, *, case_id, arm, replica_ids=(0, 1), training_metrics=None):
    """Summarize all policies and controls without choosing a winner.

    ``samples`` contains exactly two independent valuation-stream dictionaries.
    Lower samples have axes (path, training replica, policy, one inception
    stock); each control has one upper per path shared by every candidate.
    Training metrics are accepted for orchestration compatibility, but walltime
    accounting belongs to the runner and never changes these statistical gates.
    """
    streams = _validate(samples, replica_ids)
    scope = _scope(case_id, arm, replica_ids)
    if training_metrics is not None and not isinstance(training_metrics, (dict, list, tuple)):
        raise ValueError("training metrics must be an explicit mapping or sequence")
    values, gains, controls, variation, exercise = [], [], [], [], []
    averaged = {control: [s[control+"_lower"].mean(axis=1) for s in streams]
        for control in CONTROLS}
    for p, policy in enumerate(POLICIES):
        raw = [x[:, p] for x in averaged["raw"]]
        raw_summary = _moments(raw)
        for control in CONTROLS:
            lower = [x[:, p] for x in averaged[control]]
            upper = [s[control+"_upper"] for s in streams]
            lo, hi = _moments(lower), _moments(upper)
            gap = _moments([u-l for u, l in zip(upper, lower)])
            allowance = PRICE_ATOL+PRICE_RTOL*abs(lo["estimate"])
            budget = gap["estimate"]+CONFIDENCE_MULTIPLIER*(lo["standard_error"]+gap["standard_error"])
            identity = {**scope, "policy": policy, "control": control,
                "primary": control == "european_premium"}
            values.append({**identity, "paths": lo["paths"],
                "price": lo["estimate"], "price_se": lo["standard_error"],
                "price_interval_low": lo["interval_low"], "price_interval_high": lo["interval_high"],
                "upper": hi["estimate"], "upper_se": hi["standard_error"],
                "gap": gap["estimate"], "gap_se": gap["standard_error"],
                "price_error_budget": budget, "price_allowance": allowance,
                "price_budget_pass": budget <= allowance,
                "upper_is_shared_dual_maximum": True,
                "replicas_averaged_within_path": True})
            shift = _moments([x-y for x, y in zip(lower, raw)])
            raw_var, controlled_var = raw_summary["within_stream_variance"], lo["within_stream_variance"]
            ratio = controlled_var/raw_var if raw_var > 0 else np.nan
            controls.append({**identity, "paths": lo["paths"],
                "price_se": lo["standard_error"], "raw_price_se": raw_summary["standard_error"],
                "variance_ratio_to_raw": ratio,
                "variance_reduction_fraction": 1-ratio,
                "paired_shift_from_raw": shift["estimate"],
                "paired_shift_se": shift["standard_error"],
                "paired_shift_interval_low": shift["interval_low"],
                "paired_shift_interval_high": shift["interval_high"],
                "shift_zero_in_interval": shift["interval_low"] <= 0 <= shift["interval_high"],
                "shift_plus_uncertainty": abs(shift["estimate"])+shift["half_width"],
                "same_frozen_policy": True,
                "control_unbiasedness_established_by_this_test": False})
            for r, replica_id in enumerate(replica_ids):
                individual = _moments([s[control+"_lower"][:, r, p] for s in streams])
                variation.append({**identity, "record_type": "individual_replica",
                    "replica_id": replica_id, **individual,
                    "between_training_population_uncertainty_estimated": False})
            disagreement = _moments([s[control+"_lower"][:, 1, p]-s[control+"_lower"][:, 0, p]
                for s in streams])
            variation.append({**identity, "record_type": "paired_replica_difference",
                "replica_id": None, "reference_replica_id": replica_ids[0],
                "comparison_replica_id": replica_ids[1], **disagreement,
                "between_training_population_uncertainty_estimated": False})
        for r, replica_id in enumerate(replica_ids):
            chunks = [s["stop_indices"][:, r, p, 0] for s in streams]
            joined = np.concatenate(chunks)
            times = streams[0]["times"]
            exercise.append({**scope, "policy": policy, "replica_id": replica_id,
                "record_type": "replica_summary", "paths": len(joined),
                "inception_probability": float(np.mean(joined == 0)),
                "early_exercise_probability": float(np.mean(joined < len(times)-1)),
                "expiry_probability": float(np.mean(joined == len(times)-1)),
                "mean_stop_years": float(np.mean(times[joined]))})
        stops = np.concatenate([s["stop_indices"][:, :, p, 0] for s in streams])
        times = streams[0]["times"]
        n = len(stops)
        exercise.append({**scope, "policy": policy, "replica_id": None,
            "record_type": "replica_average", "paths": n,
            "inception_probability": float(np.mean(stops == 0)),
            "early_exercise_probability": float(np.mean(stops < len(times)-1)),
            "expiry_probability": float(np.mean(stops == len(times)-1)),
            "mean_stop_years": float(np.mean(times[stops]))})
        for j, time in enumerate(times):
            exercise.append({**scope, "policy": policy, "replica_id": None,
                "record_type": "exercise_date_distribution", "paths": n,
                "exercise_index": j, "exercise_time": float(time),
                "probability": float(np.mean(stops == j))})
    for candidate, reference in CONTRASTS:
        a, b = POLICIES.index(candidate), POLICIES.index(reference)
        for control in CONTROLS:
            paired = _moments([x[:, a]-x[:, b] for x in averaged[control]])
            gains.append({**scope, "candidate_policy": candidate, "reference_policy": reference,
                "control": control, "primary": control == "european_premium", **paired,
                "zero_in_interval": paired["interval_low"] <= 0 <= paired["interval_high"],
                "positive_interval": paired["interval_low"] > 0,
                "same_paths_same_upper": True,
                "gap_reduction_equals_controlled_gain": True,
                "independent_dual_improvement": False})
    return {"values": pd.DataFrame(values), "paired_policy_gains": pd.DataFrame(gains),
        "controls": pd.DataFrame(controls), "training_variation": pd.DataFrame(variation),
        "exercise": pd.DataFrame(exercise)}


__all__ = ["summarize_case", "POLICIES", "CONTROLS", "CONTRASTS"]
