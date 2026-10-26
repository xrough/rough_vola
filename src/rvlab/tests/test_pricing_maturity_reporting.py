"""Maturity comparisons retain common horizons and honest execution counts."""

import numpy as np
import pandas as pd
import pytest

from rvlab.experiments._pricing_maturity_study import (
    _parameter_recovery_table, _price_recovery_tables, _timing_summary,
    maturity_contracts,
)


def test_common_horizon_comparison_uses_identical_contracts_and_separate_new_tenors():
    contracts = maturity_contracts()
    frames = []
    for design, sample in (("short", contracts.iloc[:18]), ("augmented", contracts)):
        frames.append(sample.assign(design=design, attempt_id=design+"_1", start=1,
            stage="exploratory", price_residual=np.where(sample.maturity <= 60/365, .01, .08),
            clean_price_error=np.where(sample.maturity <= 60/365, .02, .16), scaled_residual=1.))
    per_tenor, cohorts = _price_recovery_tables(pd.concat(frames, ignore_index=True))
    assert len(per_tenor) == 18  # Two splits per tenor, three short plus six augmented.
    shared = cohorts.loc[cohorts.horizon_cohort.eq("common_14_30_60")]
    assert set(shared.design) == {"short", "augmented"}
    assert set(shared.loc[shared.split.eq("train"), "contract_count"]) == {12}
    assert set(shared.loc[shared.split.eq("heldout"), "contract_count"]) == {6}
    np.testing.assert_allclose(shared.clean_rmse, .02)
    long_only = cohorts.loc[cohorts.horizon_cohort.eq("added_90_180_365")]
    assert long_only.design.eq("augmented").all()
    np.testing.assert_allclose(long_only.observed_rmse, .08)
    np.testing.assert_allclose(long_only.clean_rmse, .16)
    # Overall augmented RMSE would mix these deliberately different errors;
    # shared-horizon scores must retain the original short comparison.
    assert len(cohorts) == 6


def test_cached_and_imported_work_is_not_counted_as_current_numerical_execution():
    summary = pd.DataFrame([
        {"checkpoint_reused": True, "prior_profile_reused": False,
         "elapsed_seconds": 100., "pricing_evaluations": 50, "pricing_evaluations_complete": True},
        {"checkpoint_reused": False, "prior_profile_reused": True,
         "elapsed_seconds": 200., "pricing_evaluations": 80, "pricing_evaluations_complete": True},
        {"checkpoint_reused": False, "prior_profile_reused": False,
         "elapsed_seconds": 30., "pricing_evaluations": 20, "pricing_evaluations_complete": False},
    ])
    result = _timing_summary(summary)
    assert result["recorded_attempt_seconds"] == 330.
    assert result["recorded_pricing_evaluations"] == 150
    assert result["current_attempt_seconds"] == 30.
    assert result["current_pricing_evaluations"] == 20
    assert result["reused_attempt_count"] == 1
    assert result["imported_prior_attempt_count"] == 1
    assert result["executed_attempt_count"] == 1
    assert not result["pricing_evaluations_complete"]
    assert "parallel wall time" in result["timing_scope"]


def test_parameter_errors_use_matching_design_and_name_not_row_order():
    estimates = pd.DataFrame([
        {"design": "short", "parameter": "theta_2", "fitted": .005, "generating": .045},
        {"design": "augmented", "parameter": "kappa", "fitted": 1.6, "generating": 1.5},
        {"design": "augmented", "parameter": "theta_2", "fitted": .046, "generating": .045},
    ])
    geometry = pd.DataFrame([
        {"design": "augmented", "parameter": "theta_2", "local_noise_sd": .002, "nuisance_projected_fraction": .5},
        {"design": "short", "parameter": "theta_2", "local_noise_sd": .08, "nuisance_projected_fraction": .01},
        {"design": "augmented", "parameter": "kappa", "local_noise_sd": .2, "nuisance_projected_fraction": .2},
    ])
    table = _parameter_recovery_table(estimates, geometry)
    assert list(table.parameter) == ["theta_2", "kappa", "theta_2"]
    np.testing.assert_allclose(table.error_in_local_noise_sd, [-.5, .5, .5])
    assert table.interpretation.str.contains("prevent a z-test").all()
    duplicate = pd.concat([geometry, geometry.iloc[[0]]], ignore_index=True)
    with pytest.raises(pd.errors.MergeError):
        _parameter_recovery_table(estimates, duplicate)
