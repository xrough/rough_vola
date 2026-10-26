"""Recovery is scalar metadata normalization, never a numerical refit choice."""
from dataclasses import replace

import numpy as np
import pytest

from rvlab.experiments import _pricing_rough_benchmark as original
from rvlab.experiments import _pricing_rough_benchmark_serialization as safe
from rvlab.pricing import RoughHestonParams
from rvlab.pricing.rough_estimators import RoughEuropeanControlConfig
from rvlab.pricing.rough_lift import ExponentialKernel
from rvlab.pricing.rough_stopping_fourier import train_fourier_stopping_policy
from rvlab.pricing.vanilla_types import DiscountCurve, DividendSchedule, SpotMarket


@pytest.fixture(scope="module")
def policy():
    config = replace(original.benchmark_config(training_seed=321123,
        valuation_seed=321124, training_paths=32, valuation_paths=4,
        steps=8, exercise_steps=4),
        premium_control=RoughEuropeanControlConfig(8, 40., 32))
    return train_fourier_stopping_policy(RoughHestonParams(.04, .04, 1., 0., -.6, .15),
        SpotMarket(100., DiscountCurve.flat(.04, 1.), DividendSchedule((), 1.)),
        ExponentialKernel((1.,), (1.,), .15, 1.), strike=100., maturity=30/365,
        is_call=False, config=config, control_config=config.premium_control, premium_cap=.7)


@pytest.mark.parametrize("flag", [np.bool_(True), np.bool_(False)])
def test_numpy_boolean_failure_reproduced_then_normalized_roundtrip(policy, flag, tmp_path):
    typed = replace(policy, exercise_at_inception=flag)
    partial = tmp_path/"original_failure"
    with pytest.raises(TypeError, match="JSON serializable"):
        original.save_policy(typed, partial)
    assert (partial/"coefficients.npz").is_file()
    assert not (partial/"policy.json").exists()
    proof = safe.verify_partial_coefficients(typed, partial/"coefficients.npz")
    assert proof["all_keys_shapes_dtypes_bytes_identical"]
    normalized, identity = safe.normalize_policy_metadata(typed)
    assert type(normalized.exercise_at_inception) is bool
    assert normalized.exercise_at_inception == bool(flag)
    assert all(identity["all_other_policy_fields_identical"].values())
    assert (identity["original_typed_fingerprint"] != identity["normalized_fingerprint"]) == (
        repr(flag) != repr(bool(flag)))
    assert identity["original_truth_value"] == identity["normalized_truth_value"]
    saved = safe.save_benchmark_policy(typed, tmp_path/"corrected")
    restored = original.load_policy(tmp_path/"corrected")
    assert saved["policy_fingerprint"] == identity["normalized_fingerprint"]
    safe.verify_partial_coefficients(restored, partial/"coefficients.npz")
    before, _ = original.value_benchmark_policy(typed, typed.config)
    after, _ = original.value_benchmark_policy(restored, restored.config)
    checked = safe.assert_valuation_arrays_identical(before, after)
    assert checked["own_stopping_dates_included"] and checked["all_bumps_included"]


@pytest.mark.parametrize("flag", [True, False])
def test_native_booleans_preserve_original_identity(policy, flag):
    normalized, proof = safe.normalize_policy_metadata(replace(policy, exercise_at_inception=flag))
    assert normalized.exercise_at_inception is flag
    assert proof["original_typed_fingerprint"] == proof["normalized_fingerprint"]


@pytest.mark.parametrize("defect", ["one_ulp", "dtype", "shape", "missing_key", "extra_key", "time_grid"])
def test_full_partial_archive_byte_comparison_rejects_any_change(policy, defect, tmp_path):
    arrays = {k: a.copy() for k, a in safe.coefficient_arrays(policy).items()}
    key = "g0_1_beta"
    if defect == "one_ulp": arrays[key][0] = np.nextafter(arrays[key][0], np.inf)
    elif defect == "dtype": arrays[key] = arrays[key].astype(np.float32)
    elif defect == "shape": arrays[key] = arrays[key][None]
    elif defect == "missing_key": arrays.pop(key)
    elif defect == "extra_key": arrays["unsealed"] = np.zeros(1)
    else: arrays["exercise_times"][1] = np.nextafter(arrays["exercise_times"][1], np.inf)
    path = tmp_path/"partial.npz"
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError):
        safe.verify_partial_coefficients(policy, path)


def test_normalization_rejects_non_boolean_and_valuation_tamper(policy):
    with pytest.raises(ValueError, match="Boolean"):
        safe.normalize_policy_metadata(replace(policy, exercise_at_inception=1))
    before = {"stop_indices": np.array([[1, 2]], dtype=int)}
    after = {"stop_indices": np.array([[1, 1]], dtype=int)}
    with pytest.raises(ValueError, match="valuation arrays"):
        safe.assert_valuation_arrays_identical(before, after)
