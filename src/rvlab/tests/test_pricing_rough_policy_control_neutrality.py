"""Centering diagnostics use paired contrasts without changing acceptance."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rvlab.experiments._pricing_rough_policy_reporting import add_policy_control_neutrality
from rvlab.experiments._pricing_rough_stopping_attribution import _save_sample_archive


def _fixture(tmp_path, *, selected="improved", nonfinite=False, negative=False):
    common = np.array([-2., -1., 0., 1., 2., 0.])
    noise = .01*common
    raw_gain = (-.02 if negative else .02)+noise
    controlled_gain = .003+.99*noise
    raw = np.column_stack((3.+common, 3.+common+raw_gain))
    controlled = np.column_stack((2.+.1*common, 2.+.1*common+controlled_gain))
    if nonfinite: raw[0, 0] = np.nan
    archive = _save_sample_archive(tmp_path/"paired_policy_endpoints.npz",
        raw_payoffs=raw, controlled_primals=controlled,
        dual_samples=controlled.max(axis=1)+1.)
    record = {"status": "completed", "selected_policy": selected,
        "selected_control": "frozen control", "policy_names": ["original", "improved"],
        "same_upper_samples": True, "sample_archives": [archive]}
    checkpoint = tmp_path/"checkpoint.json"
    checkpoint.write_text(json.dumps({"payload": record}))
    result = {"tables": {"trials": pd.DataFrame([{"case": "rough case",
        "status": "completed", "checkpoint": str(checkpoint)}]),
        "comparison": pd.DataFrame([{"discrete_price_gate_passed": False,
            "american_pricing_ready": False}])},
        "manifest": {"profile_fingerprint": "frozen", "acceptance": {"price_atol": .001}},
        "figures": {}}
    return result, checkpoint, record, raw_gain, controlled_gain


def test_control_neutrality_retains_paired_covariance_and_changes_no_frozen_evidence(tmp_path):
    result, checkpoint, record, raw_gain, controlled_gain = _fixture(tmp_path)
    sample = Path(record["sample_archives"][0]["path"])
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (checkpoint, sample)}
    comparison = result["tables"]["comparison"].copy(deep=True)
    output = add_policy_control_neutrality(result)
    row = output["tables"]["control_neutrality"].iloc[0]
    contrast = raw_gain-controlled_gain
    expected_se = contrast.std(ddof=1)/len(contrast)**.5
    assert row.raw_minus_controlled_gain == pytest.approx(contrast.mean())
    assert row.raw_minus_controlled_standard_error == pytest.approx(expected_se)
    assert row.z_score == pytest.approx(contrast.mean()/expected_se)
    assert row.confidence_low == pytest.approx(contrast.mean()-1.96*expected_se)
    assert row.confidence_high == pytest.approx(contrast.mean()+1.96*expected_se)
    unpaired_se = np.hypot(raw_gain.std(ddof=1), controlled_gain.std(ddof=1))/len(contrast)**.5
    assert expected_se < unpaired_se/100
    assert not row.zero_consistent
    assert row.marginal_inference_only and not row.bias_identified
    assert not row.sampling_noise_confirmed and not row.acceptance_input
    assert not row.american_pricing_ready and not row.stock_delta_ready
    assert "control_neutrality" not in result["tables"]
    pd.testing.assert_frame_equal(output["tables"]["comparison"], comparison)
    assert output["manifest"]["acceptance"] == {"price_atol": .001}
    assert output["manifest"]["profile_fingerprint"] == "frozen"
    assert output["manifest"]["policy_control_neutrality"]["pointwise_intervals_excluding_zero"] == 1
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in hashes.items())


def test_baseline_fallback_has_zero_contrast_without_undefined_zero_over_zero(tmp_path):
    result, _, _, _, _ = _fixture(tmp_path, selected="original")
    row = add_policy_control_neutrality(result)["tables"]["control_neutrality"].iloc[0]
    assert row.raw_minus_controlled_gain == 0.
    assert row.raw_minus_controlled_standard_error == 0.
    assert row.z_score == 0. and row.zero_consistent


def test_negative_contrast_is_preserved_without_clipping(tmp_path):
    result, _, _, _, _ = _fixture(tmp_path, negative=True)
    row = add_policy_control_neutrality(result)["tables"]["control_neutrality"].iloc[0]
    assert row.raw_minus_controlled_gain < 0 and row.confidence_high < 0
    assert row.z_score < 0 and not row.zero_consistent


@pytest.mark.parametrize("mode", ["empty", "failed", "missing_columns"])
def test_absent_completed_evidence_returns_unchanged_result(tmp_path, mode):
    result, _, _, _, _ = _fixture(tmp_path)
    if mode == "empty": result["tables"]["trials"] = pd.DataFrame()
    elif mode == "failed": result["tables"]["trials"]["status"] = "numerical_failure"
    else: result["tables"]["trials"] = pd.DataFrame({"case": ["rough"]})
    assert add_policy_control_neutrality(result) is result


def test_changed_samples_fail_seal_verification(tmp_path):
    result, _, record, _, _ = _fixture(tmp_path)
    sample = Path(record["sample_archives"][0]["path"])
    with sample.open("ab") as stream: stream.write(b"changed")
    with pytest.raises(ValueError, match="Changed paired-sample"):
        add_policy_control_neutrality(result)


def test_nonfinite_but_sealed_samples_are_rejected(tmp_path):
    result, _, _, _, _ = _fixture(tmp_path, nonfinite=True)
    with pytest.raises(ValueError, match="finite aligned"):
        add_policy_control_neutrality(result)


@pytest.mark.parametrize("mutation,error", [("status", "incomplete"), ("upper", "identical")])
def test_inconsistent_frozen_comparison_claims_are_rejected(tmp_path, mutation, error):
    result, checkpoint, record, _, _ = _fixture(tmp_path)
    if mutation == "status": record["status"] = "failed"
    else: record["same_upper_samples"] = False
    checkpoint.write_text(json.dumps({"payload": record}))
    with pytest.raises(ValueError, match=error):
        add_policy_control_neutrality(result)
