"""A metadata repair cannot become another adaptive numerical experiment."""
from copy import deepcopy

import numpy as np
import pytest

from rvlab.experiments import _pricing_rough_benchmark_recovery_collection as r
from rvlab.experiments._pricing_rough_benchmark import TOLERANCES


@pytest.fixture
def protocol():
    parent = {"trials": [{"index": i, "valuation_seeds": [1147005042 + 3000000 * i,
        1148005042 + 3000000 * i]} for i in range(10)]}
    design = {"version": "rough_generic_benchmark_serialization_repair_v1",
        "indices": [2, 5], "trials": [deepcopy(parent["trials"][i]) for i in (2, 5)],
        "original_four_never_drawn_valuation_seeds": [1153005042, 1154005042, 1162005042, 1163005042],
        "valuation_paths_per_stream": 2048, "total_recovered_valuation_paths": 8192,
        "original_predeclared_total_paths_after_composition": 40960,
        "maximum_workers": 2, "new_or_adaptive_acceptance_seeds": False,
        "no_new_valuation_selection": True, "no_empirical_valuation_recentering": True,
        "original_failures_remain_immutable": True,
        "both_identity_replays_required_before_any_recovered_valuation": True,
        "every_partial_coefficient_key_shape_dtype_and_byte_must_match": True,
        "all_other_policy_fields_must_be_identical": True,
        "original_training_streams_replayed": True, "training_path_budget_changed": False,
        "all_failures_retained": True, "price_or_optimal_delta_readiness_promoted": False,
        "implementation_fixture": {"seed": 1095005042, "paths_per_case": 4,
            "same_paths_replayed_before_and_after_normalization": True,
            "changed_N_old_seed_label": True, "acceptance_evidence": False}, **TOLERANCES}
    return design, parent


def test_metadata_only_recovery_keeps_the_original_fixed_pool(protocol):
    design, parent = protocol
    r.validate_recovery_protocol(design, parent)
    assert design["valuation_paths_per_stream"] * 4 + 32768 == 40960


@pytest.mark.parametrize("key,value", [
    ("price_atol", .002), ("delta_atol", .004), ("confidence_multiplier", 1.),
    ("valuation_paths_per_stream", 4096), ("total_recovered_valuation_paths", 16384),
    ("new_or_adaptive_acceptance_seeds", True), ("no_new_valuation_selection", False),
    ("no_empirical_valuation_recentering", False), ("original_failures_remain_immutable", False),
    ("all_failures_retained", False), ("training_path_budget_changed", True),
    ("both_identity_replays_required_before_any_recovered_valuation", False),
    ("every_partial_coefficient_key_shape_dtype_and_byte_must_match", False),
    ("price_or_optimal_delta_readiness_promoted", True)])
def test_repair_cannot_relax_accuracy_or_change_observations(protocol, key, value):
    design, parent = protocol
    design[key] = value
    with pytest.raises(ValueError, match="scope or tolerance"):
        r.validate_recovery_protocol(design, parent)


@pytest.mark.parametrize("change", ["case", "seed", "extra", "reorder"])
def test_repair_cannot_reselect_the_failed_attempts(protocol, change):
    design, parent = protocol
    if change == "case": design["indices"][0] = 1
    elif change == "seed": design["trials"][0]["valuation_seeds"][0] += 1
    elif change == "extra": design["original_four_never_drawn_valuation_seeds"].append(123)
    else: design["trials"].reverse()
    with pytest.raises(ValueError, match="failed cases or seed"):
        r.validate_recovery_protocol(design, parent)


def test_small_identity_fixture_is_never_acceptance_evidence(protocol):
    design, parent = protocol
    design["implementation_fixture"]["acceptance_evidence"] = True
    with pytest.raises(ValueError, match="Implementation-only"):
        r.validate_recovery_protocol(design, parent)


def test_recovery_array_loader_rejects_omitted_own_stopping_data(tmp_path):
    path = tmp_path / "samples.npz"
    np.savez(path, raw_lower=np.ones((4, 5)))
    with pytest.raises(ValueError, match="numerical fields"):
        r.load_arrays(path, {})


def test_recovery_array_loader_rejects_changed_dtype(tmp_path):
    path = tmp_path / "samples.npz"
    arrays = {key: np.ones(2) for key in r.ARRAY_KEYS}
    np.savez(path, **arrays)
    schema = {key: {"shape": [2], "dtype": "float64"} for key in arrays}
    schema["stop_indices"]["dtype"] = "int64"
    with pytest.raises(ValueError, match="shape or dtype"):
        r.load_arrays(path, schema)


@pytest.mark.parametrize("key,value", [("seed", 1095005043), ("paths", 2048),
    ("steps", 256), ("strike", 100.)])
def test_identity_fixture_cannot_coherently_change_its_saved_metadata(key, value):
    trial = {"training_seed": 1152005042, "steps": 128, "exercise_steps": 64,
             "factors": 4, "case": {"strike": 110., "maturity": 30 / 365}}
    spec = {"trial_id": "serialization_implementation_2", "seed": 1095005042,
        "training_seed": 1152005042, "paths": 4, "steps": 128,
        "exercise_steps": 64, "factors": 4, "strike": 110., "maturity": 30 / 365}
    r.validate_implementation_spec(spec, trial, 2)
    spec[key] = value
    with pytest.raises(ValueError, match="predeclared seed, N or transition"):
        r.validate_implementation_spec(spec, trial, 2)
