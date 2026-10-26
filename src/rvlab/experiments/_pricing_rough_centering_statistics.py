"""Paired, independent-run summaries for a frozen martingale audit.

The experimental unit is a simulated path.  Exercise dates along a path are
not independent observations.  A run can contain several paired quantities,
but every quantity must retain the same path ordering within that run.  These
summaries make no policy choice, alter no pricing tolerance and cannot prove
centering by failing to reject a zero mean.
"""
from __future__ import annotations

from numbers import Integral, Real

import numpy as np
import pandas as pd


def stopped_martingale_contrast(martingale, original_stops, selected_stops):
    """Reconstruct a paired stopped-control contrast with predictable gates.

    If ``j`` indexes the right endpoint of an increment, ``tau >= j`` means
    the policy had not stopped by its left endpoint.  Thus the signed gate is
    adapted even though this retrospective diagnostic obtains it from stored
    stopping indices.  Calendar increments on a path remain correlated data;
    inferential summaries must use each path's complete gated sum.
    """
    values = np.asarray(martingale, dtype=float)
    original, selected = np.asarray(original_stops), np.asarray(selected_stops)
    if (values.ndim != 2 or len(values) < 2 or values.shape[1] < 2
            or not np.isfinite(values).all() or np.any(values[:, 0] != 0)):
        raise ValueError("Martingale samples must be finite (path,time) arrays starting at zero")
    for stops in (original, selected):
        if (stops.shape != (len(values),) or stops.dtype.kind not in "iu"
                or np.any(stops < 0) or np.any(stops >= values.shape[1])):
            raise ValueError("Stopping indices must align with paths and calendar dates")
    endpoints = np.arange(1, values.shape[1])[None, :]
    gate = ((selected[:, None] >= endpoints).astype(np.int8)
        -(original[:, None] >= endpoints).astype(np.int8))
    gated = gate*np.diff(values, axis=1)
    path = np.arange(len(values))
    endpoint = values[path, selected]-values[path, original]
    reconstructed = gated.sum(axis=1)
    return {"endpoint_contrast": endpoint, "gated_increment_sums": reconstructed,
        "signed_left_gate": gate, "gated_increments": gated,
        "max_reconstruction_error": float(np.max(np.abs(reconstructed-endpoint))),
        "path_is_sampling_unit": True,
        "gate_scope": "not stopped at the left date; signed disagreement of frozen stopping rules"}


def _interval(estimate, standard_error, multiplier):
    """Preserve signed estimates and degenerate, nonzero contrasts."""
    low = estimate-multiplier*standard_error
    high = estimate+multiplier*standard_error
    z_score = (estimate/standard_error if standard_error > 0
        else (0. if estimate == 0 else None))
    return {"estimate": float(estimate), "standard_error": float(standard_error),
        "confidence_low": float(low), "confidence_high": float(high),
        "z_score": z_score, "zero_consistent": bool(low <= 0 <= high)}


def summarize_independent_centering_runs(runs, *, confidence_multiplier=1.96,
        forbidden_seeds=()):
    """Return per-stream and pooled signed-mean diagnostics.

    Each run is ``{"seed": int, "samples": {name: one_dimensional_array}}``.
    Streams must be independent and quantities within a run must be paired.
    Independence is a design requirement; distinct recorded seed integers do
    not establish it mathematically.  ``forbidden_seeds`` provides a firewall
    for spent valuation and training streams.

    The pooled estimate weights each stream by its path count.  Its standard
    error is ``sqrt(sum(n_i * s_i**2)) / sum(n_i)``: independent strata need
    not have equal variances.  No inverse-variance weights are fitted to noisy
    samples.  A common-distribution pooled standard error and seed dispersion
    are retained as diagnostics.  Intervals are marginal normal intervals,
    not simultaneous statements across quantities or dates.
    """
    if (isinstance(confidence_multiplier, bool)
            or not isinstance(confidence_multiplier, Real)
            or not np.isfinite(confidence_multiplier) or confidence_multiplier <= 0):
        raise ValueError("confidence_multiplier must be finite and positive")
    if not isinstance(runs, (list, tuple)) or not runs:
        raise ValueError("Declare a nonempty independent-run list")
    if not isinstance(forbidden_seeds, (list, tuple, set, frozenset)):
        raise TypeError("forbidden_seeds must be an explicit collection")
    forbidden = set()
    for seed in forbidden_seeds:
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("Forbidden seeds must be nonnegative integers")
        forbidden.add(int(seed))
    seen, quantities, prepared = set(), None, []
    for run in runs:
        if not isinstance(run, dict) or set(run) != {"seed", "samples"}:
            raise ValueError("Each run must declare exactly seed and samples")
        seed = run["seed"]
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("Run seeds must be nonnegative integers")
        seed = int(seed)
        if seed in seen or seed in forbidden:
            raise ValueError("Run seeds overlap another run or a forbidden stream")
        seen.add(seed)
        samples = run["samples"]
        if (not isinstance(samples, dict) or not samples or
                any(not isinstance(name, str) or not name for name in samples)):
            raise ValueError("Samples require nonempty named paired quantities")
        if quantities is None:
            quantities = tuple(samples)
        if set(samples) != set(quantities):
            raise ValueError("Every run must report the same quantities")
        arrays, size = {}, None
        for name in quantities:
            values = np.asarray(samples[name], dtype=float)
            if (values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all()
                    or (size is not None and len(values) != size)):
                raise ValueError("Each run requires finite aligned path samples, with at least two paths")
            size, arrays[name] = len(values), values
        prepared.append((seed, size, arrays))
    shared = {"confidence_multiplier": float(confidence_multiplier),
        "marginal_inference_only": True, "sampling_noise_confirmed": False,
        "centering_error_identified": False, "acceptance_input": False,
        "american_pricing_ready": False, "stock_delta_ready": False}
    per_seed, pooled = [], []
    for name in quantities:
        counts, means, variances = [], [], []
        for seed, size, arrays in prepared:
            values = arrays[name]
            mean, variance = float(np.mean(values)), float(np.var(values, ddof=1))
            counts.append(size)
            means.append(mean)
            variances.append(variance)
            per_seed.append({"quantity": name, "seed": seed, "paths": size,
                **_interval(mean, np.sqrt(variance/size), confidence_multiplier),
                **shared})
        counts = np.asarray(counts, dtype=float)
        means, variances = np.asarray(means), np.asarray(variances)
        total = int(sum(counts))
        mean = float(np.dot(counts, means)/total)
        stratified_se = float(np.sqrt(np.dot(counts, variances))/total)
        # The second term includes observed variation of independent seed means;
        # do not discard it when reporting a common-law pooled diagnostic.
        sum_squares = float(np.dot(counts-1, variances)+
            np.dot(counts, (means-mean)**2))
        iid_se = float(np.sqrt(sum_squares/(total-1)/total))
        inverse_mean_variance = np.divide(counts, variances,
            out=np.zeros_like(counts), where=variances > 0)
        q = (float(np.dot(inverse_mean_variance, (means-mean)**2))
            if np.all(variances > 0) else None)
        pooled.append({"quantity": name, "streams": len(prepared), "paths": total,
            **_interval(mean, stratified_se, confidence_multiplier),
            "iid_pooled_standard_error": iid_se,
            "seed_mean_min": float(min(means)), "seed_mean_max": float(max(means)),
            "seed_dispersion_q": q, "seed_dispersion_degrees_of_freedom": len(prepared)-1,
            "seed_dispersion_is_diagnostic": True,
            "standard_error_method": "independent_strata_path_count_weighted",
            **shared})
    return {"per_seed": pd.DataFrame(per_seed), "pooled": pd.DataFrame(pooled),
        "metadata": {"stream_count": len(prepared), "seeds": [r[0] for r in prepared],
            "quantities": list(quantities), "forbidden_seed_count": len(forbidden),
            "path_is_sampling_unit": True, "temporal_increments_treated_independent": False,
            "zero_mean_nonrejection_proves_centering": False, "multiple_comparison_adjustment": False,
            "confidence_multiplier": float(confidence_multiplier),
            "scope": "Fixed-policy, fixed-martingale independent-path mean diagnostics; no tolerance or readiness change"}}
