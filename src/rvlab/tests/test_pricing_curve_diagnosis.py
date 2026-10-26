"""Follow-up experiments must distinguish profiles, accepted fits and holdouts."""
import copy
from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from rvlab.experiments import _pricing_curve_diagnosis as diagnosis
from rvlab.experiments import _pricing_american_identification as joint
from rvlab.pricing import FDConfig, OptimizerConfig, parameter_values


@pytest.fixture
def linear_problem(monkeypatch):
    """A known compensating direction with six independently identifiable nuisances."""
    contracts = joint.joint_contracts()
    truth = joint.joint_specifications()[0]["heston_curve"]
    names = [name for name in parameter_values(truth) if name != "theta_2"]
    matrix = np.random.default_rng(123).normal(size=(len(contracts), len(names)))
    theta_column = matrix[:, names.index("theta_1")]
    true_values = parameter_values(truth)
    target = 5+matrix@np.array([true_values[name] for name in names])+.045*theta_column
    quotes = joint.synthetic_quotes(contracts, target, np.zeros(len(contracts)))["clean"]
    spec = {"dataset": "clean", "quotes": quotes.to_dict("records"),
            "fit_config": asdict(FDConfig(192, 48, 180, spot_concentration=.05)),
            "optimizer": asdict(OptimizerConfig(parameterization="asinh", ftol=1e-8, xtol=1e-8, gtol=1e-6)),
            "bounds": joint.joint_specifications()[2]["heston_curve"]}
    identifiers = {(r.strike, r.maturity): i for i, r in enumerate(contracts.itertuples())}

    def evaluate(params, *, strike, maturity, grid, **kwargs):
        p = parameter_values(params)
        indices = [identifiers[(k, t)] for k, t in zip(strike, maturity)]
        prices = 5+matrix[indices]@np.array([p[name] for name in names])+p["theta_2"]*theta_column[indices]
        digest = hashlib.sha256(grid.spot.tobytes()+grid.variance.tobytes()).hexdigest()
        return SimpleNamespace(prices=prices, diagnostics={"grid_sha256": digest, "pricing_evaluations": 1})

    monkeypatch.setattr(diagnosis, "evaluate_vanilla", evaluate)
    return {"specification": spec}, true_values


def test_profile_reoptimizes_nuisances_without_promoting_it_to_accepted_fit(linear_problem):
    record, initial = linear_problem
    result = diagnosis._profile_attempt(record, .02, initial, 40)
    row = result["summary"]
    assert row["optimizer_success"]
    assert row["objective"] < 1e-12
    assert row["params"]["theta_2"] == .02
    assert row["params"]["theta_1"] == pytest.approx(initial["theta_1"]+.025, abs=1e-6)
    assert not row["numerically_refined"] and not row["accepted_calibration"]
    assert len(result["predictions"]) == 18


def test_holdout_changes_do_not_enter_profile_objective(linear_problem):
    record, initial = linear_problem
    original = diagnosis._profile_attempt(record, .02, initial, 40)
    changed = copy.deepcopy(record)
    for quote in changed["specification"]["quotes"]:
        if quote["split"] == "heldout":
            quote["market_price"] += .5
            quote["bid"] += .5
            quote["ask"] += .5
    alternative = diagnosis._profile_attempt(changed, .02, initial, 40)
    assert alternative["summary"]["params"] == original["summary"]["params"]
    assert alternative["summary"]["objective"] == original["summary"]["objective"]


def test_zero_level_probe_preserves_original_bound_specification(linear_problem):
    record, initial = linear_problem
    saved = copy.deepcopy(record)
    result = diagnosis._profile_attempt(record, 0., initial, 40)["summary"]
    assert result["optimizer_success"]
    assert result["params"]["theta_2"] == 0.
    assert result["bound_case"] == "nonnegative-level sensitivity"
    assert record == saved
    assert record["specification"]["bounds"]["theta_2"] == (.0025, .25)


def test_profile_records_grid_mismatch_as_failure(linear_problem, monkeypatch):
    record, initial = linear_problem
    monkeypatch.setattr(diagnosis, "evaluate_vanilla", lambda *a, **k:
        SimpleNamespace(prices=np.ones(12), diagnostics={"grid_sha256": "changed", "pricing_evaluations": 1}))
    result = diagnosis._profile_attempt(record, .02, initial, 40)["summary"]
    assert result["status"] == "numerical_failure"
    assert "frozen training grid" in result["error"]
    assert not result["accepted_calibration"]


def test_baseline_loader_rejects_foreign_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(joint, "_engine_fingerprint", lambda: "current")
    spec = {"engine_sha256": "old"}
    record = {"specification": spec, "fingerprint": joint._fingerprint(spec), "payload": {}}
    (tmp_path/"heston_curve_clean_1_exploratory.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Incompatible source/input"):
        diagnosis._load_baseline(tmp_path, "clean")
