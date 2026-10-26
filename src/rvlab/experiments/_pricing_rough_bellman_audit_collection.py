"""Reconstruct the sealed scalar Bellman audit without repeating integration."""
from __future__ import annotations

import ast
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SEEDS = (1054005042, 1055005042)
ORDERS = (8, 16, 32)
DATES = (16, 32, 48, 63)


def required_audit_sources():
    names = {str(p.relative_to(ROOT)) for p in (ROOT/"src/rvlab/pricing").glob("*.py")}
    names.add("examples/rough_bellman_adaptive_audit.py")
    names.update("src/rvlab/experiments/"+name for name in ("_pricing_rough_bellman_audit.py",
        "_pricing_rough_bellman_policy.py", "_pricing_rough_exercise_block.py",
        "_pricing_rough_large_policy.py", "_pricing_rough_frozen_policy.py"))
    return names


def required_audit_inputs():
    model = "rough_larger_policy_v1/32768_paths_two_starts/"
    spent = "rough_tail_attribution_v1/joint_switch_v1/"
    return {model+"frozen_policy_before_final.json", model+"frozen_policy_coefficients.npz",
        spent+"manifest.json", spent+"selection_before_final.json"} | {
        spent+f"final_{i}_{suffix}" for i in range(2) for suffix in ("manifest.json", "samples.npz")}


def _source_panel(folder):
    """Reapply the declared spent-state selector without looking at new Q."""
    from ._pricing_rough_large_policy import load_larger_policy_bundle
    output = folder.parents[1]
    bundle = load_larger_policy_bundle(output/"rough_larger_policy_v1/32768_paths_two_starts")
    proxy = bundle["proxy"]; weights = np.asarray(proxy.kernel.weights)
    rows = []
    for index, seed in enumerate(SEEDS):
        with np.load(output/f"rough_tail_attribution_v1/joint_switch_v1/final_{index}_samples.npz", allow_pickle=False) as z:
            stocks, factors, margins, times = (z[k] for k in ("stocks", "factors", "tail_margin", "times"))
            if (stocks.shape != (4096, 65, 1) or factors.shape != (4096, 65, 4)
                    or margins.shape != (4096, 65) or not np.array_equal(times, proxy.exercise_times)):
                raise ValueError("The scalar audit source changed its complete spent states or calendar")
            for j in DATES:
                intrinsic = np.maximum((1 if proxy.is_call else -1)*(stocks[:, j, 0]-proxy.strike), 0.)
                available = np.flatnonzero(intrinsic > 0)
                if len(available) < 4:
                    raise ValueError("The declared scalar panel lacks four distinct ITM states")
                first, second = map(int, available[:2])
                remaining = available[2:]
                near = int(min(remaining, key=lambda p: (abs(margins[p, j]), int(p))))
                remaining = remaining[remaining != near]
                variance = proxy.params.v0+factors[:, j]@weights
                high = int(min(remaining, key=lambda p: (-variance[p], int(p))))
                for role, path in (("first ITM", first), ("second ITM", second),
                        ("nearest frozen boundary", near), ("largest variance", high)):
                    rows.append({"state_index": len(rows), "seed": seed, "source_index": index,
                        "path_index": path, "exercise_index": j, "time": float(times[j]), "role": role,
                        "stock": float(stocks[path, j, 0]), "variance": float(variance[path]),
                        "frozen_margin": float(margins[path, j]), "factors": factors[path, j].tolist()})
    return rows


def _number(value, *, nonnegative=False, integer=False):
    kind = Integral if integer else Real
    return (not isinstance(value, (bool, np.bool_)) and isinstance(value, kind)
        and np.isfinite(value) and (not nonnegative or value >= 0))


def _close(left, right):
    return bool(np.isfinite(left) and np.isfinite(right) and np.isclose(left, right, atol=2e-13, rtol=2e-13))


def _compare(actual, expected, keys):
    if (set(actual) != set(expected) or len(actual) != len(expected)
            or actual.duplicated(keys).any() or expected.duplicated(keys).any()):
        raise ValueError("An independent scalar-audit table omitted, duplicated or added a field/state")
    left = actual.sort_values(keys).reset_index(drop=True).copy()
    right = expected.sort_values(keys).reset_index(drop=True).copy()
    for name in ("warnings", "factors"):
        if name in right:
            normalize = lambda v: json.dumps(ast.literal_eval(v) if isinstance(v, str) else v, sort_keys=True)
            left[name] = left[name].map(normalize); right[name] = right[name].map(normalize)
    try:
        pd.testing.assert_frame_equal(left[list(right)], right, check_dtype=False,
            check_exact=False, atol=2e-13, rtol=2e-13)
    except (AssertionError, TypeError, ValueError) as error:
        raise ValueError("A scalar-audit table differs from its sealed per-state evidence") from error


def _validate_record(record, state, design_sha):
    if (not isinstance(record, dict) or record.get("design_sha256") != design_sha or record.get("state") != state
            or record.get("status") not in ("completed", "failed")
            or not _number(record.get("elapsed_seconds"), nonnegative=True)
            or not isinstance(record.get("rows"), list) or len(record["rows"]) != 3
            or any(not isinstance(row, dict) for row in record["rows"])):
        raise ValueError("A scalar state changed its design, identity, completion status or resolution count")
    status = record["status"]; rows = record["rows"]
    for order, row in zip(ORDERS, rows):
        identity = {key: state[key] for key in ("state_index", "seed", "path_index", "exercise_index", "role")}
        if (any(row.get(k) != v for k, v in identity.items()) or row.get("order") != order
                or row.get("status") != status or row.get("used_for_formal_policy_selection") is not False):
            raise ValueError("A scalar result changed state/order identity or entered formal policy selection")
        if status == "failed":
            if (not isinstance(record.get("error_type"), str) or not record["error_type"]
                    or not isinstance(record.get("error_message"), str)
                    or row.get("error_type") != record["error_type"] or row.get("error_message") != record["error_message"]
                    or set(row) != set(identity)|{"order", "status", "error_type", "error_message", "used_for_formal_policy_selection"}):
                raise ValueError("A failed scalar integration was hidden or replaced with numerical results")
            continue
        required = set(identity)|{"order", "status", "reference_continuation", "cv_continuation", "direct_continuation",
            "cv_minus_reference", "direct_minus_reference", "quadrature_error_estimate", "omitted_tail_value_bound",
            "price_allowance", "cv_within_dollar_allowance", "direct_within_dollar_allowance", "terminal_exact_payoff",
            "quadrature_error_is_rigorous_bound", "used_for_formal_policy_selection"}
        numeric = ("reference_continuation", "cv_continuation", "direct_continuation", "cv_minus_reference",
            "direct_minus_reference", "quadrature_error_estimate", "omitted_tail_value_bound", "price_allowance")
        if (set(row) != required or any(not _number(row.get(k)) for k in numeric)
                or any(row[k] < 0 for k in ("reference_continuation", "cv_continuation", "direct_continuation",
                    "quadrature_error_estimate", "omitted_tail_value_bound"))
                or row["quadrature_error_is_rigorous_bound"] is not False
                or row["terminal_exact_payoff"] is not (state["exercise_index"] == 63)):
            raise ValueError("A completed scalar result has invalid values, error estimates or interpretation flags")
        reference = row["reference_continuation"]; allowance = .001+1e-4*abs(reference)
        if (not _close(row["price_allowance"], allowance)
                or not _close(row["cv_minus_reference"], row["cv_continuation"]-reference)
                or not _close(row["direct_minus_reference"], row["direct_continuation"]-reference)):
            raise ValueError("A scalar result changes comparison arithmetic or the unchanged acceptance allowance")
        for label in ("cv", "direct"):
            passed = abs(row[label+"_minus_reference"])+row["quadrature_error_estimate"]+row["omitted_tail_value_bound"] <= allowance
            if row[label+"_within_dollar_allowance"] is not passed:
                raise ValueError("A scalar price diagnostic hid a failed unchanged dollar allowance")
    if status == "completed":
        common = ("reference_continuation", "quadrature_error_estimate", "omitted_tail_value_bound", "price_allowance")
        if any(row[k] != rows[0][k] for row in rows for k in common):
            raise ValueError("Quadrature orders do not share the same independent scalar reference")
        diagnostic = record.get("numerical_diagnostics", {})
        difference = np.asarray(diagnostic.get("max_cv_direct_difference", []), float)
        flags = {"orders": list(ORDERS), "physical_increments": 2, "only_premium_capped": True,
            "actual_future_state_used": False, "all_factor_branches_and_stock_orderings": True,
            "terminal_exact_payoff": state["exercise_index"] == 63}
        if (any(diagnostic.get(k) != v for k, v in flags.items())
                or not _number(diagnostic.get("max_probability_error"), nonnegative=True)
                or diagnostic["max_probability_error"] > 2e-12
                or difference.shape != (3,) or not np.isfinite(difference).all()
                or not np.allclose(difference,
                    [abs(row["cv_continuation"]-row["direct_continuation"]) for row in rows], atol=2e-13, rtol=2e-13)
                or any(not _number(record.get(k), nonnegative=True, integer=True) for k in ("detected_snell_crossings", "function_evaluations"))
                or any(not _number(record.get(k), nonnegative=True) for k in ("envelope_constant", "envelope_stock"))
                or not isinstance(record.get("warnings"), list) or any(not isinstance(w, str) for w in record["warnings"])):
            raise ValueError("A scalar conditional-law diagnostic changed centering, endpoints or numerical scope")
        if state["exercise_index"] == 63 and any(not _close(row[k], rows[0]["cv_continuation"])
                for row in rows for k in ("cv_continuation", "direct_continuation")):
            raise ValueError("The scalar audit terminal interval no longer uses the common exact payoff integral")


def validate_adaptive_audit(folder, design, manifest, panel, checks, diagnostics):
    """Validate all success/failure records; no requirement that every check pass."""
    folder = Path(folder).resolve()
    design_path = folder/"design_before_evaluation.json"
    design_sha = hashlib.sha256(design_path.read_bytes()).hexdigest()
    expected = {"version": "rough_bellman_independent_adaptive_audit_v1", "states": 32, "orders": list(ORDERS),
        "dates": list(DATES), "spent_seeds": list(SEEDS), "absolute_tolerance": 1e-9, "relative_tolerance": 1e-10,
        "normal_limit": 12., "state_wall_limit_seconds": 180., "new_outer_paths": 0, "new_fits": 0,
        "used_for_formal_policy_selection": False, "quadrature_error_is_rigorous_bound": False,
        "only_premium_capped": True, "proxy_policy_improvement_guaranteed": False,
        "allowance": ".001+1e-4*abs(C)", "tolerances_changed": False}
    if (json.loads(design_path.read_text()) != design or any(design.get(k) != v for k, v in expected.items())
            or any(manifest.get(k) != v for k, v in design.items()) or manifest.get("complete") is not True
            or manifest.get("design_sha256") != design_sha or manifest.get("all_predeclared_states_retained") is not True):
        raise ValueError("An independent scalar audit changed its sealed design, completion or scientific role")
    sources, inputs = design.get("source_sha256", {}), design.get("input_sha256", {})
    # Exact source-set equality already enforces its cardinality. The producer
    # and validator both include every current pricing module; a fixed count
    # would reject a complete closure after a new public module is added.
    if (set(sources) != required_audit_sources()
            or set(inputs) != required_audit_inputs() or len(inputs) != 8):
        raise ValueError("An independent scalar audit omitted a required numerical source or spent/model input")
    expected_panel = _source_panel(folder)
    if design.get("state_panel") != expected_panel:
        raise ValueError("The scalar audit panel differs from its predeclared spent-state selector")
    _compare(panel, pd.DataFrame(expected_panel), ["state_index"])
    required_paths = {folder/f"state_{i:02d}.json" for i in range(32)}
    if set(folder.glob("state_*.json")) != required_paths:
        raise ValueError("An independent scalar integration state was omitted or added")
    records = []
    for i, state in enumerate(expected_panel):
        record = json.loads((folder/f"state_{i:02d}.json").read_text())
        _validate_record(record, state, design_sha); records.append(record)
    successes = sum(record["status"] == "completed" for record in records)
    if manifest.get("successful_states") != successes or manifest.get("failed_states") != 32-successes:
        raise ValueError("A scalar audit completion summary hid a failed integration")
    result_checks = pd.DataFrame([row for record in records for row in record["rows"]])
    result_diagnostics = pd.DataFrame([{key: value for key, value in record.items()
        if key not in ("rows", "state", "numerical_diagnostics")} | {key: value for key, value in record["state"].items()
        if key != "factors"} for record in records])
    _compare(checks, result_checks, ["state_index", "order"])
    _compare(diagnostics, result_diagnostics, ["state_index"])
    return {"spent_scalar_checks": result_checks, "spent_scalar_states": pd.DataFrame(expected_panel),
        "spent_scalar_diagnostics": result_diagnostics}


__all__ = ["validate_adaptive_audit"]
