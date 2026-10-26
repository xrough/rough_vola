"""A cheap fitting grid cannot bypass the unchanged price/spread screen."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rvlab.experiments import _pricing_american_identification as joint
from rvlab.experiments import _pricing_spot_accuracy as study
from rvlab.pricing import FDConfig, PricingError


def _run(monkeypatch, *, refinement_change=0., reference_error=0., fail=False):
    contracts = joint.joint_contracts().sample(frac=1., random_state=42).reset_index(drop=True)
    observed = np.ones(len(contracts))
    reference = observed.copy()
    index = int(np.flatnonzero(contracts.half_spread.to_numpy() == .008)[0])
    reference[index] += reference_error
    calls = []
    def evaluate(params, **kwargs):
        calls.append(kwargs)
        if fail and kwargs["check_refinement"]:
            raise PricingError("selected-grid divergence")
        changes = np.zeros(len(contracts)); changes[index] = refinement_change
        checks = pd.DataFrame([{"axis": "spot", "status": "ok",
            "price_stable": True, "delta_stable": True,
            "max_price_change": refinement_change, "max_delta_change": 0., "error": None}])
        return SimpleNamespace(prices=observed, deltas=np.full(len(contracts), .3),
            price_stable=True, delta_stable=True, diagnostics={"checks": checks,
                "price_changes": {"spot": changes}, "pricing_evaluations": 1,
                "grid_sha256": "fixed-training-mesh"})
    monkeypatch.setattr(study, "evaluate_vanilla", evaluate)
    result = study.model_spot_accuracy(SimpleNamespace(progress=lambda message: None),
        "heston", joint.joint_specifications()[0]["heston"], contracts, joint.joint_market(),
        reference, np.full(len(contracts), .3), FDConfig(192, 48, 180, spot_concentration=.05))
    return result, contracts, calls


def test_truth_screen_preserves_contract_order_and_refines_only_declared_grid(monkeypatch):
    result, contracts, calls = _run(monkeypatch)
    assert result["accepted"] and len(result["accuracy"]) == 7
    assert sum(call["check_refinement"] for call in calls) == 1
    assert [call["config"].spot_steps for call in calls if call["check_refinement"]] == [192]
    details = pd.DataFrame(result["contracts"])
    for _, group in details.groupby("spot_steps"):
        assert group.contract_id.tolist() == contracts.contract_id.tolist()
    assert {row["spot_concentration"] for row in result["accuracy"]} == {.05}


@pytest.mark.parametrize("kind", ["independent refinement", "finer reference"])
def test_spread_guard_cannot_be_overridden_by_native_absolute_tolerance(monkeypatch, kind):
    kwargs = {"refinement_change": .00081} if kind == "independent refinement" else {"reference_error": .00081}
    result, _, _ = _run(monkeypatch, **kwargs)
    selected = next(row for row in result["accuracy"] if row["selected_grid"])
    assert not result["accepted"] and selected["price_stable"]
    if kind == "independent refinement":
        assert not selected["spread_stable"]
    else:
        assert not selected["reference_price_pass"]
        assert selected["max_price_error_ratio"] > 1.


def test_selected_grid_failure_does_not_silently_choose_another_resolution(monkeypatch):
    result, _, _ = _run(monkeypatch, fail=True)
    selected = next(row for row in result["accuracy"] if row["selected_grid"])
    assert not result["accepted"] and selected["status"] == "numerical_failure"
    assert "selected-grid divergence" in selected["error"]
    assert len(result["accuracy"]) == 7
