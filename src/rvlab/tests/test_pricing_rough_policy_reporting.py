"""Post-run reference diagnostics retain paired covariance and gate decisions."""
import json

import numpy as np
import pandas as pd
import pytest

from rvlab.experiments._pricing_rough_policy_reporting import add_policy_reference_pairs
from rvlab.experiments._pricing_rough_stopping_attribution import _save_sample_archive


def _fixture(tmp_path, *, upper_shape=False):
    common = np.array([-.1, .05, .025, -.025, .075, -.075])
    noise = np.array([-.0002, .0001, .00005, -.00005, .00015, -.00015])
    baseline = 3.+common
    selected = baseline+.001+noise
    dp = baseline+.0015+2*noise
    upper = dp+.0003-noise/2
    lower = np.column_stack((baseline, selected, dp))
    archive = _save_sample_archive(tmp_path/"paired_policy_endpoints.npz",
        controlled_primals=lower, dual_samples=upper[:, None] if upper_shape else upper)
    record = {"status": "completed", "reference": {"diagnostics": {
        "reference_change_diagnostic": .000014}}, "policy_names": [
            "original", "selected", "independent BS reference policy"],
        "selected_policy": "selected", "same_upper_samples": True,
        "sample_archives": [archive]}
    checkpoint = tmp_path/"case.json"
    checkpoint.write_text(json.dumps({"payload": record}))
    result = {"tables": {"trials": pd.DataFrame([{"case": "constant variance",
        "status": "completed", "checkpoint": str(checkpoint)}]),
        "comparison": pd.DataFrame([{"discrete_price_gate_passed": False,
            "american_pricing_ready": False, "price": 3.}])},
        "manifest": {"profile_fingerprint": "frozen-profile",
            "acceptance": {"price_atol": .001}}, "figures": {"original": object()}}
    return result, checkpoint, record, lower, upper


def test_reference_pairs_use_exact_paired_samples_and_preserve_frozen_results(tmp_path):
    result, _, _, lower, upper = _fixture(tmp_path)
    before = result["tables"]["comparison"].copy(deep=True)
    output = add_policy_reference_pairs(result)
    table = output["tables"]["reference_policy_pairs"]
    assert len(table) == 6 and set(table.policy) == {"original", "selected"}
    for policy, index in (("original", 0), ("selected", 1)):
        gain = lower[:, 2]-lower[:, index]
        gap = upper-lower[:, 2]
        for name, samples in (("dp_minus_policy", gain), ("upper_minus_dp", gap),
                ("upper_gap_minus_policy_gain", gap-gain)):
            row = table.loc[table.policy.eq(policy) & table.quantity.eq(name)].iloc[0]
            se = samples.std(ddof=1)/len(samples)**.5
            assert row.estimate == pytest.approx(samples.mean())
            assert row.standard_error == pytest.approx(se)
            assert row.confidence_low == pytest.approx(samples.mean()-1.96*se)
            assert row.confidence_high == pytest.approx(samples.mean()+1.96*se)
    assert table.approximate_dp_rule.all() and table.reference_change_is_empirical.all()
    assert not table.acceptance_input.any()
    assert not table.american_pricing_ready.any() and not table.stock_delta_ready.any()
    assert table.reference_mesh_change.eq(.000014).all()
    # The third quantity is signed; no false nonnegative bias interpretation.
    assert table.loc[table.quantity.eq("upper_gap_minus_policy_gain"), "estimate"].lt(0).all()
    pd.testing.assert_frame_equal(output["tables"]["comparison"], before)
    assert "reference_policy_pairs" not in result["tables"]
    assert output["manifest"]["profile_fingerprint"] == "frozen-profile"
    assert output["manifest"]["acceptance"] == {"price_atol": .001}
    assert not output["manifest"]["reference_policy_pair_diagnostics"]["mesh_change_included_in_intervals"]
    assert output["figures"] is result["figures"]


@pytest.mark.parametrize("mode", ["empty", "no_bs", "failed", "no_reference"])
def test_absent_reference_evidence_leaves_result_unchanged(tmp_path, mode):
    result, checkpoint, record, _, _ = _fixture(tmp_path)
    if mode == "empty": result["tables"]["trials"] = pd.DataFrame()
    elif mode == "no_bs": result["tables"]["trials"]["case"] = "central rough"
    elif mode == "failed": result["tables"]["trials"]["status"] = "numerical_failure"
    else:
        record["reference"] = None
        checkpoint.write_text(json.dumps({"payload": record}))
    assert add_policy_reference_pairs(result) is result


def test_reference_pairs_reject_changed_sealed_endpoint_bytes(tmp_path):
    result, _, record, _, _ = _fixture(tmp_path)
    path = record["sample_archives"][0]["path"]
    with open(path, "ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Changed paired-sample"):
        add_policy_reference_pairs(result)


def test_reference_pairs_reject_misaligned_but_sealed_samples(tmp_path):
    result, _, _, _, _ = _fixture(tmp_path, upper_shape=True)
    with pytest.raises(ValueError, match="aligned"):
        add_policy_reference_pairs(result)


@pytest.mark.parametrize("mutation,error", [
    ("status", "incomplete"), ("same_upper", "identical"), ("mesh", "mesh change")])
def test_reference_pairs_reject_inconsistent_checkpoint_claims(tmp_path, mutation, error):
    result, checkpoint, record, _, _ = _fixture(tmp_path)
    if mutation == "status": record["status"] = "numerical_failure"
    elif mutation == "same_upper": record["same_upper_samples"] = False
    else: record["reference"]["diagnostics"]["reference_change_diagnostic"] = -.01
    checkpoint.write_text(json.dumps({"payload": record}))
    with pytest.raises(ValueError, match=error):
        add_policy_reference_pairs(result)
