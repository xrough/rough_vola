"""A cached numerical trial cannot cross input, algorithm or source versions."""

from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rvlab.experiments import _pricing_curve_convergence as study
from rvlab.experiments._pricing_fits import FitAttempt
from rvlab.pricing import HestonCurveParams, MeanReversionCurve, OptimizerConfig, PricingConfig


@pytest.fixture
def trial(monkeypatch, tmp_path):
    quotes = pd.DataFrame({"contract_id": ["a", "b"], "strike": [99.,101.],
        "maturity": [.04,.08], "is_call": [False,True], "forward": 100.,
        "discount": .998, "market_price": [1.,2.], "split": ["train","heldout"]},
        index=[7,7])
    params = HestonCurveParams(.02,1.,.3,-.6,MeanReversionCurve((.04,),(.03,.04)))
    calls = []

    def fit(label, q, initial, **kwargs):
        calls.append((q.copy(deep=True), kwargs))
        prediction = q.copy(deep=True)
        prediction["model_price"] = [1.25, np.nan]
        prediction["residual"] = [.25,np.nan]
        return FitAttempt(label, SimpleNamespace(params=initial),
            {"status": "exhausted", "ready": False, "nfev": 1, "params": asdict(initial)}, prediction)

    monkeypatch.setattr(study, "fit_attempt", fit)
    monkeypatch.setattr(study, "_code_fingerprints", lambda: {"test_source": "one"})
    options = dict(bounds={"kappa": (.01,20.)}, config=PricingConfig(48,100.,64),
                   max_nfev=1, checkpoint_dir=tmp_path,
                   fit_options={"optimizer": OptimizerConfig(parameterization="asinh")},
                   metadata={"scenario": "raw"})
    return quotes, params, options, calls


def test_cache_preserves_failed_status_contract_order_and_missing_prices(trial):
    q, p, kwargs, calls = trial
    first = study.checkpointed_attempt("trial", q, p, **kwargs)
    second = study.checkpointed_attempt("trial", q, p, **kwargs)
    assert len(calls) == 1
    assert not first.checkpoint_reused and second.checkpoint_reused
    assert second.summary["status"] == "exhausted" and not second.summary["ready"]
    assert second.params == p
    pd.testing.assert_frame_equal(first.predictions, second.predictions)
    assert second.predictions.index.tolist() == [7,7]
    assert np.isnan(second.predictions.model_price.iloc[1])
    assert "model_price" not in q


@pytest.mark.parametrize("change", ["heldout", "order", "dtype", "bounds", "config",
                                    "budget", "optimizer", "metadata", "source", "environment"])
def test_cache_rejects_material_trial_changes(trial, monkeypatch, change):
    q, p, kwargs, calls = trial
    study.checkpointed_attempt("trial", q, p, **kwargs)
    q = q.copy(deep=True)
    if change == "heldout":
        q.iloc[1,q.columns.get_loc("market_price")] += .01
    elif change == "order":
        q = q.iloc[::-1]
    elif change == "dtype":
        q["strike"] = q.strike.astype("float32")
    elif change == "bounds":
        kwargs["bounds"] = {"kappa": (.01,80.)}
    elif change == "config":
        kwargs["config"] = replace(kwargs["config"], riccati_steps=128)
    elif change == "budget":
        kwargs["max_nfev"] = 2
    elif change == "optimizer":
        kwargs["fit_options"] = {"optimizer": OptimizerConfig(parameterization="log1p")}
    elif change == "metadata":
        kwargs["metadata"] = {"scenario": "exercise"}
    elif change == "source":
        monkeypatch.setattr(study, "_code_fingerprints", lambda: {"test_source": "two"})
    elif change == "environment":
        monkeypatch.setattr(study, "version", lambda name: "changed-version")
    result = study.checkpointed_attempt("trial", q, p, **kwargs)
    assert not result.checkpoint_reused
    assert len(calls) == 2


@pytest.mark.parametrize("damage", ["json", "shape", "params"])
def test_incomplete_cache_recomputes_instead_of_accepting_bad_diagnostics(trial, damage):
    q, p, kwargs, calls = trial
    study.checkpointed_attempt("trial", q, p, **kwargs)
    path = kwargs["checkpoint_dir"] / "trial.json"
    if damage == "json":
        path.write_text("{incomplete")
    else:
        data = json.loads(path.read_text())
        if damage == "shape":
            data["prediction_columns"]["model_price"] = [1.]
        else:
            data["params"]["curve"]["levels"] = [-1., .04]
        path.write_text(json.dumps(data))
    assert not study.checkpointed_attempt("trial", q, p, **kwargs).checkpoint_reused
    assert len(calls) == 2
