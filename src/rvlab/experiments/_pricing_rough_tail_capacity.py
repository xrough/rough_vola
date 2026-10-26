"""All-path capacity bound for controls that change only after a fixed stop.

If M1=M0 through tau, its dual upper is at least the maximum of P-M0 through
tau. The resulting upper-decrease ceiling limits this control family. It does
not identify exercise-policy error or the total error of the original dual.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def poststop_capacity(payoff, martingale, stops, *, eligible=None):
    payoff, martingale, stops = map(np.asarray, (payoff, martingale, stops))
    if (payoff.ndim != 2 or len(payoff) == 0 or payoff.shape[1] == 0
            or martingale.shape != payoff.shape or stops.shape != (len(payoff),)
            or stops.dtype.kind not in "iu" or np.any(stops < 0) or np.any(stops >= payoff.shape[1])
            or not np.isfinite(payoff).all() or np.any(payoff < 0)
            or not np.isfinite(martingale).all() or np.any(martingale[:, 0] != 0)):
        raise ValueError("Finite aligned vanilla payoffs, zero-start control and valid fixed stops required")
    eligible = np.ones(payoff.shape[1], bool) if eligible is None else np.asarray(eligible)
    if (eligible.dtype.kind != "b" or eligible.shape != (payoff.shape[1],)
            or not eligible[-1] or not np.all(eligible[stops])):
        raise ValueError("Eligibility must include maturity and every fixed stopping date")
    adjusted = np.where(eligible, payoff - martingale, -np.inf)
    prefix = np.maximum.accumulate(adjusted, axis=1)
    path = np.arange(len(payoff))
    lower = adjusted[path, stops]
    upper = np.max(adjusted, axis=1)
    prefix_upper = prefix[path, stops]
    capacity = upper - prefix_upper
    gap = upper - lower
    remaining = prefix_upper - lower
    if (not all(np.isfinite(v).all() for v in (lower, upper, prefix_upper, capacity, gap, remaining))
            or np.any(capacity < 0) or np.any(remaining < 0)
            or not np.allclose(capacity + remaining, gap, rtol=0., atol=2e-11)):
        raise FloatingPointError("Poststop family ceiling failed its pathwise prefix identity")
    return {"baseline_lower": lower, "baseline_upper": upper, "baseline_gap": gap,
        "prefix_upper": prefix_upper, "capacity": capacity, "remaining_prefix_gap": remaining}


def check_poststop_upper(capacity, candidate_upper):
    """Check the necessary upper bound after the caller verifies family membership."""
    upper = np.asarray(candidate_upper, float)
    if (upper.shape != capacity["prefix_upper"].shape or not np.isfinite(upper).all()
            or np.any(upper < capacity["prefix_upper"] - 2e-11)):
        raise ValueError("A claimed poststop control crosses the unchanged prefix upper bound")
    return capacity["baseline_upper"] - upper


def _summary(chunks):
    if (not isinstance(chunks, (tuple, list)) or not chunks
            or any(np.asarray(v).ndim != 1 or len(v) < 2 or not np.isfinite(v).all() for v in chunks)):
        raise ValueError("Capacity uncertainty needs complete finite independent path vectors")
    count = sum(map(len, chunks))
    mean = float(np.concatenate(chunks).mean())
    se = float(np.sqrt(sum(len(v) * np.var(v, ddof=1) for v in chunks)) / count)
    return mean, se


def summarize_poststop_capacity(chunks):
    """Propagate the random baseline gap in the half-gap capacity contrast.

    The claim is limited to this family and fixed baseline rule. Cohorts
    chosen by future gaps must not be passed as all-path acceptance evidence.
    This function is diagnostic; no numerical tolerance or selector changes.
    """
    keys = {"baseline_lower", "baseline_upper", "baseline_gap", "prefix_upper", "capacity", "remaining_prefix_gap"}
    if not isinstance(chunks, (tuple, list)) or not chunks:
        raise ValueError("Provide independent capacity path chunks")
    for chunk in chunks:
        if set(chunk) != keys:
            raise ValueError("Incomplete poststop capacity path fields")
        n = len(chunk["capacity"])
        if (any(np.asarray(v).shape != (n,) or not np.isfinite(v).all() for v in chunk.values())
                or np.any(chunk["capacity"] < 0) or np.any(chunk["baseline_gap"] < 0)
                or np.any(chunk["remaining_prefix_gap"] < 0)
                or not np.allclose(chunk["baseline_upper"] - chunk["baseline_lower"], chunk["baseline_gap"],
                    atol=2e-11, rtol=0.)
                or not np.allclose(chunk["prefix_upper"] - chunk["baseline_lower"], chunk["remaining_prefix_gap"],
                    atol=2e-11, rtol=0.)
                or not np.allclose(chunk["baseline_upper"] - chunk["prefix_upper"], chunk["capacity"],
                    atol=2e-11, rtol=0.)
                or not np.allclose(chunk["capacity"] + chunk["remaining_prefix_gap"], chunk["baseline_gap"],
                    atol=2e-11, rtol=0.)):
            raise ValueError("Capacity path fields change the fixed prefix identity")
    ceiling, se = _summary([c["capacity"] for c in chunks])
    gap, gse = _summary([c["baseline_gap"] for c in chunks])
    remaining, rse = _summary([c["remaining_prefix_gap"] for c in chunks])
    margin, mse = _summary([c["capacity"] - .5 * c["baseline_gap"] for c in chunks])
    low, high = margin - 1.96 * mse, margin + 1.96 * mse
    return pd.DataFrame([{"paths": sum(len(c["capacity"]) for c in chunks), "streams": len(chunks),
        "mean_ceiling": ceiling, "ceiling_standard_error": se,
        "baseline_gap": gap, "baseline_gap_standard_error": gse,
        "remaining_prefix_gap": remaining, "prefix_gap_standard_error": rse,
        "ceiling_gap_fraction": ceiling / gap if gap > 0 else np.nan,
        "half_gap_margin": margin, "half_gap_margin_standard_error": mse,
        "half_gap_margin_interval_low": low, "half_gap_margin_interval_high": high,
        "poststop_family_cannot_remove_half": high < 0.,
        "positive_capacity_fraction": float(np.mean(np.concatenate([c["capacity"] for c in chunks]) > 1e-12)),
        "confidence_multiplier": 1.96, "paired_whole_paths": True,
        "acceptance_input": False, "used_for_selection": False,
        "causal_policy_or_dual_error_identified": False,
        "scope": "strictly poststop controls; fixed policy and unchanged pre-stop martingale"}])
