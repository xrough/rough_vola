"""Long-tenor experiments retain independent numerical acceptance gates."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rvlab.experiments import _pricing_maturity_accuracy as study
from rvlab.experiments._pricing_american_identification import joint_contracts, joint_market, joint_specifications
from rvlab.pricing import FDConfig, PricingError


def _audit(monkeypatch, *, change=0., ref_error=0., missing_axis=None, fail=False):
    contracts = joint_contracts().sample(frac=1., random_state=42).reset_index(drop=True)
    index = int(np.flatnonzero(contracts.half_spread.to_numpy() == .008)[0])
    captured = []
    original_build = study.build_grid
    def build(market, strikes, config, **kwargs):
        captured.append(np.asarray(strikes).copy())
        return original_build(market, strikes, config, **kwargs)
    def evaluate(params, **kwargs):
        if fail:
            raise PricingError("long-tenor divergence")
        axes = [axis for axis in ("spot", "time", "variance", "domain", "joint") if axis != missing_axis]
        differences = np.zeros(len(contracts)); differences[index] = change
        return SimpleNamespace(prices=np.ones(len(contracts)), deltas=np.full(len(contracts), .3),
            price_stable=True, delta_stable=True, diagnostics={"checks": pd.DataFrame([
                {"axis": axis, "status": "ok", "price_stable": True, "delta_stable": True,
                 "max_price_change": change, "max_delta_change": 0., "error": None} for axis in axes]),
                "price_changes": {axis: differences for axis in axes}, "pricing_evaluations": 6,
                "grid_sha256": "training-only"})
    monkeypatch.setattr(study, "build_grid", build)
    monkeypatch.setattr(study, "evaluate_vanilla", evaluate)
    reference = np.ones(len(contracts)); reference[index] += ref_error
    result = study.maturity_truth_accuracy(SimpleNamespace(progress=lambda m: None),
        joint_specifications()[0]["heston_curve"], contracts, joint_market(), FDConfig(240,64,600),
        reference, np.full(len(contracts), .3))
    return result, contracts, captured


def test_long_truth_screen_keeps_training_mesh_and_contract_order(monkeypatch):
    result, contracts, grids = _audit(monkeypatch)
    assert result["accepted"] and len(result["refinement"]) == 5
    np.testing.assert_array_equal(grids[0], contracts.loc[contracts.split.eq("train"), "strike"])
    assert pd.DataFrame(result["contracts"]).contract_id.tolist() == contracts.contract_id.tolist()
    assert len(result["by_maturity"]) == 3


@pytest.mark.parametrize("failure", ["spread", "reference", "missing_axis", "pricing"])
def test_long_truth_screen_cannot_relax_or_skip_unchanged_checks(monkeypatch, failure):
    options = {"spread": {"change": .00081}, "reference": {"ref_error": .00081},
        "missing_axis": {"missing_axis": "variance"}, "pricing": {"fail": True}}
    result, _, _ = _audit(monkeypatch, **options[failure])
    assert not result["accepted"]
    if failure == "pricing":
        assert "long-tenor divergence" in result["accuracy"][0]["error"]


def test_long_truth_screen_requires_aligned_finite_references():
    contracts = joint_contracts()
    with pytest.raises(ValueError, match="contract aligned"):
        study.maturity_truth_accuracy(SimpleNamespace(progress=lambda m: None),
            joint_specifications()[0]["heston_curve"], contracts, joint_market(), FDConfig(),
            np.ones(len(contracts)-1), np.ones(len(contracts)))


def test_long_truth_screen_rechecks_precomputed_axis_diagnostics(monkeypatch):
    contracts = joint_contracts()
    monkeypatch.setattr(study, "evaluate_vanilla", lambda *a, **kw: pytest.fail("unexpected PDE repeat"))
    axes = ("spot", "time", "variance", "domain", "joint")
    changes = np.zeros(len(contracts)); changes[1] = .00081
    saved = SimpleNamespace(prices=np.ones(len(contracts)), deltas=np.full(len(contracts), .3),
        price_stable=True, delta_stable=True, diagnostics={"checks": pd.DataFrame([
            {"axis": axis, "status": "ok", "price_stable": True, "delta_stable": True}
            for axis in axes]), "price_changes": {axis: changes for axis in axes},
            "pricing_evaluations": 6, "grid_sha256": "training-only"})
    result = study.maturity_truth_accuracy(SimpleNamespace(progress=lambda m: None),
        joint_specifications()[0]["heston_curve"], contracts, joint_market(), FDConfig(),
        np.ones(len(contracts)), np.full(len(contracts), .3), precomputed_result=saved)
    assert not result["accepted"]
    assert result["accuracy"][0]["price_stable"]
    assert not result["accuracy"][0]["spread_stable"]
