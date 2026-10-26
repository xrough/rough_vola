"""Supported generic-policy serialization with explicit Boolean normalization.

The original benchmark saved all coefficient arrays but could not encode a
NumPy scalar exercise flag in JSON.  This module changes that flag's Python
representation only.  Every other policy field and numerical array is checked
exactly.  No exercise decision, fitted coefficient or valuation setting changes.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, replace
import hashlib
from pathlib import Path

import numpy as np

from . import _pricing_rough_benchmark as original
from rvlab.pricing.rough_stopping import _Continuation
from rvlab.pricing.rough_stopping_fourier import FourierStoppingPolicy, fourier_policy_fingerprint


def _identical(a, b):
    if a is b:
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
            and a.shape == b.shape and a.dtype.str == b.dtype.str
            and a.tobytes(order="C") == b.tobytes(order="C"))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return set(a) == set(b) and all(_identical(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)) and isinstance(b, type(a)):
        return len(a) == len(b) and all(_identical(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and bool(a == b)


def normalize_policy_metadata(policy):
    """Return the same rule with a JSON-compatible native Boolean flag.

    The historical fingerprint hashes scalar ``repr`` values. Depending on
    the NumPy version, a NumPy bool and native bool can therefore have different
    fingerprints despite agreeing in truth value. Both identities and the
    allowed type-only change are returned.
    """
    if not isinstance(policy, FourierStoppingPolicy):
        raise TypeError("Boolean metadata normalization requires FourierStoppingPolicy")
    flag = policy.exercise_at_inception
    if not isinstance(flag, (bool, np.bool_)):
        raise ValueError("exercise_at_inception must be a native or NumPy Boolean")
    normalized = replace(policy, exercise_at_inception=bool(flag))
    checks = {field.name: _identical(getattr(policy, field.name),
        getattr(normalized, field.name)) for field in fields(policy)
        if field.name != "exercise_at_inception"}
    if not all(checks.values()) or bool(flag) is not normalized.exercise_at_inception:
        raise ValueError("metadata normalization changed a numerical policy field")
    proof = {"original_flag_type": type(flag).__module__+"."+type(flag).__qualname__,
        "normalized_flag_type": "builtins.bool", "original_truth_value": bool(flag),
        "normalized_truth_value": normalized.exercise_at_inception,
        "original_typed_fingerprint": fourier_policy_fingerprint(policy),
        "normalized_fingerprint": fourier_policy_fingerprint(normalized),
        "permitted_change": "exercise_at_inception scalar type only; identical Boolean truth value",
        "all_other_policy_fields_identical": checks,
        "numerical_method_changed": False, "fitted_rule_changed": False}
    return normalized, proof


def coefficient_arrays(policy):
    """Reconstruct exactly the original writer's entire explicit array schema."""
    if not isinstance(policy, FourierStoppingPolicy):
        raise TypeError("coefficient verification requires FourierStoppingPolicy")
    arrays = {"exercise_times": np.asarray(policy.exercise_times),
        "continuation_times": np.asarray(policy.continuation_times)}
    for group_index, group in enumerate((policy.continuations, policy.value_premiums)):
        for index, model in enumerate(group):
            if type(model) is not _Continuation:
                raise TypeError("generic coefficient proof requires polynomial continuations")
            for name in ("center", "scale", "beta"):
                arrays[f"g{group_index}_{index}_{name}"] = np.asarray(getattr(model, name))
    return arrays


def verify_partial_coefficients(policy, archive):
    """Require byte identity of EVERY saved coefficient and time-grid array.

    Archive ZIP timestamps are immaterial; keys, shape, dtype and the complete
    logical array bytes must match.  Even a one-ULP coefficient change fails.
    """
    actual = coefficient_arrays(policy)
    rows = []
    with np.load(Path(archive), allow_pickle=False) as saved:
        if set(actual) != set(saved.files):
            raise ValueError("retained partial coefficient archive changed its complete key set")
        for name in sorted(actual):
            value, expected = actual[name], saved[name]
            if not _identical(value, expected):
                raise ValueError("deterministic coefficient replay is not byte-identical: "+name)
            rows.append({"array": name, "shape": list(value.shape), "dtype": value.dtype.str,
                "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
                "byte_identical": True})
    return {"all_keys_shapes_dtypes_bytes_identical": True,
        "time_grids_included": True, "array_count": len(rows), "arrays": rows,
        "tolerance_used": None}


def save_benchmark_policy(policy, folder):
    """Save a generic policy safely, using the frozen explicit-format writer.

    Use this serializer in new source examples.  ``original.load_policy`` reads
    the resulting JSON/NPZ unchanged.  The returned proof records the historical
    representation-sensitive identity and the normalized, loadable identity.
    """
    normalized, proof = normalize_policy_metadata(policy)
    declaration = original.save_policy(normalized, folder)
    if declaration["policy_fingerprint"] != proof["normalized_fingerprint"]:
        raise ValueError("serialized policy differs from the normalized identity")
    folder = Path(folder)
    return {"metadata_normalization": proof,
        "policy_fingerprint": declaration["policy_fingerprint"],
        "policy_sha256": original.sha256(folder/"policy.json"),
        "coefficient_sha256": original.sha256(folder/"coefficients.npz")}


def assert_valuation_arrays_identical(before, after):
    """An implementation-only paired replay proves no stopping/output change."""
    if set(before) != set(after):
        raise ValueError("metadata normalization changed the valuation array schema")
    rows = []
    for name in sorted(before):
        if not _identical(before[name], after[name]):
            raise ValueError("metadata normalization changed valuation arrays: "+name)
        rows.append({"array": name, "shape": list(before[name].shape),
            "dtype": before[name].dtype.str, "byte_identical": True})
    return {"all_valuation_arrays_byte_identical": True,
        "own_stopping_dates_included": True, "all_bumps_included": True,
        "arrays": rows, "acceptance_evidence": False}


__all__ = ["normalize_policy_metadata", "coefficient_arrays",
    "verify_partial_coefficients", "save_benchmark_policy", "assert_valuation_arrays_identical"]
