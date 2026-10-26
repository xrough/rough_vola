"""Joint American research preserves starts, weights and failed evidence."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rvlab.experiments import _pricing_american_identification as study
from rvlab.pricing import FDConfig, OptimizerConfig, PricingError, evaluate_vanilla, parameter_values


@pytest.fixture(autouse=True)
def isolate_spot_report_from_orchestration_tests(monkeypatch):
    """The accuracy report has independent numerical tests, not twelve mock fits."""
    monkeypatch.setattr(study, "_spot_accuracy_source_sha256", lambda: "0"*64)
    monkeypatch.setattr(study, "_spot_accuracy", lambda *args:
        {"accuracy": [], "refinement": [], "contracts": [], "accepted": True})


def _quotes():
    contracts = study.joint_contracts()
    return study.synthetic_quotes(contracts, np.full(len(contracts), .5),
                                  np.linspace(-.005, .005, len(contracts)),
                                  np.where(contracts.is_call, .3, -.3))["noisy"]


def _checks(change=0.):
    return pd.DataFrame([{"axis": "joint", "status": "priced", "price_stable": True,
        "delta_stable": True, "max_price_change": change, "max_delta_change": 0.,
        "error": None}])


def _fake_fit(params, quotes, success=True):
    return SimpleNamespace(params=params, optimizer_success=success, optimizer_status=1 if success else 0,
        optimizer_message="test convergence" if success else "evaluation budget exhausted",
        price_stable=True, delta_stable=True, nfev=2, njev=1,
        pricing_evaluations=12, optimizer_settings={}, active_bounds={}, objective=.2,
        optimality=.01, physical_optimality=.02, fixed_parameters={},
        diagnostics={"checks": _checks(), "spread_stable": True,
                     "grid_fingerprints": ["same-mesh"], "fixed_grid": True},
        fitted_quotes=quotes.assign(model_price=quotes.market_price.to_numpy()),
        max_price_change=0., max_delta_change=0.)


def _fake_values(quotes, change=0.):
    prices, deltas = quotes.market_price.to_numpy(), quotes.clean_delta.to_numpy()
    return SimpleNamespace(prices=prices, deltas=deltas, price_stable=True, delta_stable=True,
        diagnostics={"pricing_evaluations": 6, "checks": _checks(change),
            "price_changes": {"joint": np.full(len(quotes), change)},
            "reference_prices": prices, "reference_deltas": deltas,
            "grid_sha256": "same-mesh"})


def test_joint_design_has_disjoint_rows_and_all_curve_levels_before_last_tenor():
    contracts = study.joint_contracts()
    truths, starts, bounds = study.joint_specifications()
    assert len(contracts) == 18 and contracts.contract_id.is_unique
    assert contracts.split.value_counts().to_dict() == {"train": 12, "heldout": 6}
    assert set(contracts.loc[contracts.split.eq("heldout"), "strike"]) == {97., 103.}
    assert len(starts["heston"]) == len(starts["heston_curve"]) == 3
    assert len(parameter_values(truths["heston"])) == 5
    assert len(parameter_values(truths["heston_curve"])) == 7
    curve = truths["heston_curve"].curve
    assert contracts.maturity.min() < curve.knot_times[0] < 30/365 < curve.knot_times[1] < contracts.maturity.max()
    for model, truth in truths.items():
        assert set(bounds[model]) == set(parameter_values(truth))
        assert all(lower < upper for lower, upper in bounds[model].values())
        for initial in starts[model]:
            assert all(bounds[model][name][0] <= value <= bounds[model][name][1]
                       for name, value in parameter_values(initial).items())


def test_clean_and_noisy_use_same_spreads_and_noise_is_never_clipped():
    contracts = study.joint_contracts()
    noise = np.random.default_rng(42).normal(0, .005, len(contracts))
    pair = study.synthetic_quotes(contracts, np.full(len(contracts), .5), noise)
    np.testing.assert_array_equal(pair["noisy"].noise, noise)
    np.testing.assert_allclose(pair["noisy"].market_price-pair["clean"].market_price, noise)
    np.testing.assert_allclose(pair["clean"].ask-pair["clean"].bid,
                               pair["noisy"].ask-pair["noisy"].bid)
    assert pair["noisy"].effective_half_spread.nunique() > 3
    assert pair["noisy"].effective_half_spread.eq(.005).any()
    assert pair["clean"].noise.eq(0).all()
    with pytest.raises(ValueError, match="observations were not clipped"):
        study.synthetic_quotes(contracts, np.full(len(contracts), .001), noise)
    with pytest.raises(ValueError, match="aligned"):
        study.synthetic_quotes(contracts, np.ones(3), noise)


def test_selection_uses_training_objective_and_start_tie_break_only():
    rows = [
        {"start": 3, "training_objective": .1, "params": {}, "test_rmse": 0., "ready": True},
        {"start": 2, "training_objective": .01, "params": {}, "test_rmse": 50., "ready": False},
        {"start": 1, "training_objective": .01, "params": {}, "test_rmse": 100., "ready": False},
        {"start": 0, "training_objective": 0., "params": None},
    ]
    assert study.select_training_attempt(rows)["start"] == 1
    assert study.select_training_attempt([{"params": {}, "training_objective": np.nan}]) is None


@pytest.mark.parametrize("model, expected_count", [("heston", 5), ("heston_curve", 7)])
def test_attempt_joint_coordinates_weights_order_and_fixed_validation_mesh(monkeypatch, model, expected_count):
    quotes = _quotes()
    truths, starts, bounds = study.joint_specifications()
    seen = []
    def fit(frame, initial, **kwargs):
        seen.append((frame.copy(), kwargs))
        return _fake_fit(initial, frame)
    monkeypatch.setattr(study, "calibrate_vanilla_surface", fit)
    monkeypatch.setattr(study, "evaluate_vanilla", lambda *args, **kwargs: _fake_values(quotes))
    result = study._attempt(model, "noisy", 1, "exploratory", quotes, study.joint_market(),
        starts[model][0], bounds[model], truths[model], FDConfig(24, 8, 16), 3, OptimizerConfig())
    row = result["summary"]
    assert row["ready"] and row["hedge_delta_ready"]
    assert row["free_parameter_count"] == expected_count and row["fixed_parameters"] == {}
    assert row["validation_grid_matches_training"] and row["train_reprice_difference"] == 0.
    assert row["pricing_evaluations"] == 18
    assert row["test_clean_rmse"] > 0. and row["test_rmse"] == 0.
    assert row["test_delta_rmse"] == 0.
    assert [item["contract_id"] for item in result["predictions"]] == quotes.contract_id.tolist()
    frame, settings = seen[0]
    assert frame.split.eq("train").all() and len(frame) == 12
    assert settings["exercise_style"] == "american" and settings["price_scale_floor"] == .005
    assert settings["stability_atol"] == .001 and settings["stability_rtol"] == 1e-4
    assert settings["delta_atol"] == .002 and settings["delta_rtol"] == .005
    assert settings["spread_refinement_fraction"] == .1
    assert all(lower < upper for lower, upper in settings["bounds"].values())
    assert len(result["parameters"]) == expected_count
    assert {item["coverage"] for item in result["refinement"]} == {"training", "training and held-out"}


@pytest.mark.parametrize("case", ["exhausted", "spread_failure", "validation_failure"])
def test_attempt_failure_states_remain_visible(monkeypatch, case):
    quotes = _quotes()
    truths, starts, bounds = study.joint_specifications()
    monkeypatch.setattr(study, "calibrate_vanilla_surface",
        lambda frame, initial, **kwargs: _fake_fit(initial, frame, success=case != "exhausted"))
    def values(*args, **kwargs):
        if case == "validation_failure":
            raise PricingError("independent refinement failed")
        return _fake_values(quotes, .001 if case == "spread_failure" else 0.)
    monkeypatch.setattr(study, "evaluate_vanilla", values)
    result = study._attempt("heston", "noisy", 1, "exploratory", quotes, study.joint_market(),
        starts["heston"][0], bounds["heston"], truths["heston"], FDConfig(24, 8, 16), 3, OptimizerConfig())
    row = result["summary"]
    assert not row["ready"] and not row["hedge_delta_ready"]
    assert row["params"] is not None and len(result["parameters"]) == 5
    if case == "exhausted":
        assert row["status"] == "exhausted" and row["price_stable"]
    elif case == "spread_failure":
        assert row["status"] == "converged_unstable" and not row["spread_stable"]
    else:
        assert row["status"] == "numerical_failure" and row["optimizer_success"]
        assert "independent refinement failed" in row["error"]


def test_checkpoint_requires_exact_numerical_spec_and_retains_failure(tmp_path):
    path, calls = tmp_path/"attempt.json", []
    def compute():
        calls.append(1)
        return {"status": "numerical_failure", "value": np.float64(.3), "undefined": np.nan}
    first, reused = study._cached(path, {"grid": 80}, compute, True)
    assert not reused and first["undefined"] is None
    again, reused = study._cached(path, {"grid": 80}, compute, True)
    assert reused and again["status"] == "numerical_failure" and len(calls) == 1
    study._cached(path, {"grid": 160}, compute, True)
    assert len(calls) == 2


@pytest.mark.parametrize("setting, value", [("max_nfev", 0), ("max_nfev", 161),
                                           ("seed", True), ("run_continuation", "yes")])
def test_settings_reject_invalid_budgets_and_non_finer_generation(setting, value):
    args = {"fit": FDConfig(20, 6, 20), "generation": FDConfig(40, 12, 40),
            "continuation": FDConfig(24, 8, 24), "max_nfev": 2,
            "continuation_max_nfev": 2, "seed": 42, "run_continuation": True, "reuse": True}
    args[setting] = value
    with pytest.raises(ValueError):
        study._validate_settings(**args)
    args[setting] = {"max_nfev": 2, "seed": 42, "run_continuation": True}.get(setting, value)
    args["generation"] = args["continuation"]
    with pytest.raises(ValueError, match="strictly finer"):
        study._validate_settings(**args)


def test_larger_attempt_budgets_preserve_numerical_acceptance():
    study._validate_settings(FDConfig(20, 6, 20), FDConfig(40, 12, 40),
        FDConfig(24, 8, 24), 80, 160, 42, True, True)
    assert (study.PRICE_ATOL, study.PRICE_RTOL) == (.001, 1e-4)
    assert (study.DELTA_ATOL, study.DELTA_RTOL) == (.002, .005)
    assert study.SPREAD_REFINEMENT_FRACTION == .1


def test_default_study_declares_the_validated_spot_grid_and_same_grid_continuation(monkeypatch):
    monkeypatch.setattr(study, "evaluate_vanilla", lambda *args, **kwargs:
        (_ for _ in ()).throw(PricingError("stop after recording settings")))
    result = study.american_identification_study(
        SimpleNamespace(progress=lambda message: None), compute_figures=False)
    manifest = result["manifest"]
    assert manifest["fit_config"] == manifest["continuation_config"]
    assert {name: manifest["fit_config"][name] for name in
            ("spot_steps", "variance_steps", "time_steps", "spot_concentration")} == {
                "spot_steps": 192, "variance_steps": 48, "time_steps": 180,
                "spot_concentration": .05}
    assert (manifest["max_nfev"], manifest["continuation_max_nfev"]) == (80, 160)
    assert manifest["generation_config"]["spot_steps"] == 640
    assert manifest["generation_config"]["spot_concentration"] == .05


def test_one_real_joint_attempt_preserves_optimizer_exhaustion():
    contracts, market = study.joint_contracts(), study.joint_market()
    truths, starts, bounds = study.joint_specifications()
    generated = evaluate_vanilla(truths["heston"], market=market,
        strike=contracts.strike.to_numpy(), maturity=contracts.maturity.to_numpy(),
        is_call=contracts.is_call.to_numpy(), exercise_style="american", config=FDConfig(40, 12, 30))
    quotes = study.synthetic_quotes(contracts, generated.prices,
        np.random.default_rng(42).normal(0, .005, len(contracts)), generated.deltas)["noisy"]
    result = study._attempt("heston", "noisy", 1, "exploratory", quotes, market,
        starts["heston"][0], bounds["heston"], truths["heston"], FDConfig(24, 8, 16), 1,
        OptimizerConfig(parameterization="asinh", ftol=1e-8, xtol=1e-8, gtol=1e-6))
    row = result["summary"]
    assert row["status"] == "exhausted" and not row["ready"]
    assert row["nfev"] == 1 and row["pricing_evaluations"] > row["nfev"]
    assert row["free_parameter_count"] == 5 and len(result["predictions"]) == 18
    assert row["validation_grid_matches_training"] and row["train_reprice_difference"] < 1e-12


def test_complete_design_runs_twelve_starts_and_four_training_selected_continuations(tmp_path, monkeypatch):
    calls, generated = [], []
    def evaluate(params, **kwargs):
        generated.append(kwargs)
        return SimpleNamespace(prices=np.full(len(kwargs["strike"]), .5),
            deltas=np.zeros(len(kwargs["strike"])), diagnostics={"grid_sha256": "generation-grid"})
    def attempt(model, dataset, start, stage, quotes, market, initial, bounds, truth, config, budget, optimizer):
        calls.append((model, dataset, start, stage, quotes.copy(), budget))
        row = study._empty_attempt(model, dataset, start, stage, initial, bounds, truth, config, budget)
        row.update(params=parameter_values(initial), training_objective=float((start-2)**2),
            train_rmse=.01, test_rmse=.02, train_clean_rmse=.01, test_clean_rmse=.02,
            max_price_change=.03, max_delta_change=.01, status="exhausted", pricing_evaluations=10)
        return {"summary": row, "parameters": [], "predictions": [], "refinement": []}
    monkeypatch.setattr(study, "evaluate_vanilla", evaluate)
    monkeypatch.setattr(study, "_attempt", attempt)
    monkeypatch.setattr(study, "_figures", lambda tables: {})
    context = SimpleNamespace(output_dir=tmp_path, progress=lambda message: None)
    kwargs = {"fit_config": FDConfig(20, 6, 20), "continuation_config": FDConfig(24, 8, 24),
              "generation_config": FDConfig(40, 12, 40), "max_nfev": 3, "continuation_max_nfev": 2}
    result = study.american_identification_study(context, **kwargs)
    assert len(calls) == 16
    summary, selection = result["tables"]["summary"], result["tables"]["selection"]
    assert summary.stage.value_counts().to_dict() == {"exploratory": 12, "continuation": 4}
    assert selection.selected_start.eq(2).all() and not selection.heldout_used_for_selection.any()
    assert set(summary.loc[summary.model.eq("heston"), "free_parameter_count"]) == {5}
    assert set(summary.loc[summary.model.eq("heston_curve"), "free_parameter_count"]) == {7}
    assert summary.status.eq("exhausted").all() and not summary.ready.any()
    assert len(result["tables"]["truth_baseline"]) == 8
    noisy = [item[4].noise.to_numpy() for item in calls if item[1] == "noisy"]
    assert all(np.array_equal(noisy[0], values) for values in noisy)
    assert result["manifest"]["pricing_evaluations"] == 160
    assert result["manifest"]["current_execution_pricing_evaluations"] == 160
    assert result["manifest"]["executed_attempt_count"] == 16
    assert result["manifest"]["reused_attempt_count"] == 0
    assert len(result["manifest"]["experiment_numerics_sha256"]) == 64
    count = len(generated)
    replay = study.american_identification_study(context, **kwargs)
    assert len(calls) == 16 and len(generated) == count
    assert replay["tables"]["summary"].checkpoint_reused.all()
    assert replay["manifest"]["pricing_evaluations"] == 160
    assert replay["manifest"]["current_execution_pricing_evaluations"] == 0
    assert replay["manifest"]["executed_attempt_count"] == 0
    assert replay["manifest"]["reused_attempt_count"] == 16
    assert replay["manifest"]["recorded_attempt_seconds"] == result["manifest"]["recorded_attempt_seconds"]
    assert replay["manifest"]["recorded_generation_seconds"] == result["manifest"]["recorded_generation_seconds"]
    partial = study.american_identification_study(context, **kwargs,
        models=("heston_curve",), compute_figures=False)
    assert len(calls) == 16 and len(generated) == count
    assert partial["manifest"]["requested_models"] == ["heston_curve"]
    assert partial["manifest"]["checkpoint_directory"] == result["manifest"]["checkpoint_directory"]
    assert partial["tables"]["summary"].model.eq("heston_curve").all()
    assert partial["figures"] == {}
    partial_dataset = study.american_identification_study(context, **kwargs,
        models=("heston_curve",), datasets=("clean",), compute_figures=False)
    assert len(calls) == 16 and len(generated) == count
    assert partial_dataset["manifest"]["requested_datasets"] == ["clean"]
    assert partial_dataset["manifest"]["checkpoint_directory"] == result["manifest"]["checkpoint_directory"]
    assert partial_dataset["tables"]["summary"].dataset.eq("clean").all()
    assert partial_dataset["tables"]["summary"].checkpoint_reused.all()
    assert replay["manifest"]["acceptance"] == {
        "price_atol": .001, "price_rtol": 1e-4, "delta_atol": .002,
        "delta_rtol": .005, "spread_refinement_fraction": .1}
    previous_directory = result["manifest"]["checkpoint_directory"]
    previous_files = {path.name: path.read_bytes()
                      for path in (tmp_path/study.STUDY_VERSION).rglob("*.json")}
    rerun = study.american_identification_study(context, **{**kwargs, "max_nfev": 4})
    assert rerun["manifest"]["checkpoint_directory"] != previous_directory
    from pathlib import Path
    assert all((Path(previous_directory)/name).read_bytes() == value
               for name, value in previous_files.items())
    assert rerun["manifest"]["executed_attempt_count"] == 16


def test_converged_training_selected_fits_do_not_consume_a_continuation_budget(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(study, "evaluate_vanilla", lambda params, **kwargs:
        SimpleNamespace(prices=np.full(len(kwargs["strike"]), .5),
            deltas=np.zeros(len(kwargs["strike"])), diagnostics={"grid_sha256": "grid"}))
    def attempt(model, dataset, start, stage, quotes, market, initial, bounds, truth, config, budget, optimizer):
        calls.append((model, dataset, start, stage))
        row = study._empty_attempt(model, dataset, start, stage, initial, bounds, truth, config, budget)
        row.update(params=parameter_values(initial), training_objective=float((start-2)**2),
            optimizer_success=start == 2, status="converged" if start == 2 else "exhausted",
            train_clean_rmse=.01, test_clean_rmse=.02)
        return {"summary": row, "parameters": [], "predictions": [], "refinement": []}
    monkeypatch.setattr(study, "_attempt", attempt)
    result = study.american_identification_study(
        SimpleNamespace(output_dir=tmp_path, progress=lambda message: None),
        fit_config=FDConfig(20, 6, 20), continuation_config=FDConfig(24, 8, 24),
        generation_config=FDConfig(40, 12, 40), max_nfev=80,
        continuation_max_nfev=160, compute_figures=False)
    assert len(calls) == 12 and all(call[-1] == "exploratory" for call in calls)
    selection = result["tables"]["selection"]
    assert selection.selected_start.eq(2).all()
    assert not selection.continuation_executed.any()
    assert selection.continuation_reason.eq("Selected optimizer already converged").all()


@pytest.mark.parametrize("kwargs", [{"models": "heston"}, {"models": ()},
    {"models": ("heston", "heston")}, {"models": ("rough_heston",)},
    {"models": (["heston"],)}, {"compute_figures": 1}, {"run_calibration": "yes"},
    {"datasets": "clean"}, {"datasets": ()}, {"datasets": ("clean", "clean")},
    {"datasets": ("unknown",)}, {"datasets": (["clean"],)}])
def test_partial_execution_rejects_malformed_selection(kwargs):
    with pytest.raises(ValueError):
        study.american_identification_study(SimpleNamespace(progress=lambda message: None), **kwargs)


@pytest.mark.parametrize("accepted, run_calibration", [(False, True), (True, False)])
def test_truth_accuracy_gate_and_audit_only_mode_do_not_run_an_optimizer(
        tmp_path, monkeypatch, accepted, run_calibration):
    monkeypatch.setattr(study, "evaluate_vanilla", lambda params, **kwargs:
        SimpleNamespace(prices=np.full(len(kwargs["strike"]), .5),
            deltas=np.zeros(len(kwargs["strike"])), diagnostics={"grid_sha256": "grid"}))
    monkeypatch.setattr(study, "_spot_accuracy", lambda *args:
        {"accuracy": [{"model": args[1], "accepted": accepted}],
         "refinement": [], "contracts": [], "accepted": accepted})
    monkeypatch.setattr(study, "_attempt", lambda *args:
        pytest.fail("accuracy rejection or audit-only mode must not call calibration"))
    result = study.american_identification_study(
        SimpleNamespace(output_dir=tmp_path, progress=lambda message: None),
        fit_config=FDConfig(20, 6, 20), continuation_config=FDConfig(24, 8, 24),
        generation_config=FDConfig(40, 12, 40), compute_figures=False,
        run_calibration=run_calibration)
    assert result["tables"]["summary"].empty
    assert len(result["tables"]["spot_accuracy"]) == 2
    assert result["manifest"]["run_calibration"] == run_calibration
    if not accepted:
        assert result["tables"]["failures"].stage.eq("calibration blocked by truth-grid accuracy").all()
    else:
        assert result["tables"]["failures"].empty
        assert "audit only" in result["tables"]["findings"].finding.iloc[0]


def test_reference_generation_failure_produces_honest_no_fit_evidence(monkeypatch):
    monkeypatch.setattr(study, "evaluate_vanilla",
        lambda *args, **kwargs: (_ for _ in ()).throw(PricingError("reference failure")))
    result = study.american_identification_study(SimpleNamespace(progress=lambda message: None),
        fit_config=FDConfig(20, 6, 20), generation_config=FDConfig(40, 12, 40), run_continuation=False)
    assert result["tables"]["summary"].empty and result["tables"]["predictions"].empty
    failures = result["tables"]["failures"]
    assert len(failures) == 2 and failures.error.str.contains("reference failure").all()
    assert result["manifest"]["attempts"] == [] and result["manifest"]["pricing_evaluations"] == 0


def test_failed_pricing_counts_are_unknown_or_explicitly_incomplete_without_mutating_cache():
    payloads = [{"summary": {"status": status, "pricing_evaluations": count}}
                for status, count in (("numerical_failure", 0), ("numerical_failure", 17), ("exhausted", 20))]
    summary = study._reporting_summary(payloads)
    assert np.isnan(summary.pricing_evaluations.iloc[0])
    assert summary.pricing_evaluations.iloc[1] == 17
    assert summary.pricing_evaluations_complete.tolist() == [False, False, True]
    assert payloads[0]["summary"]["pricing_evaluations"] == 0
    assert "pricing_evaluations_complete" not in payloads[0]["summary"]


def test_all_numerical_failures_still_render_diagnostic_figures():
    pytest.importorskip("matplotlib").use("Agg", force=True)
    truths, starts, bounds = study.joint_specifications()
    rows = [study._empty_attempt(model, dataset, 1, "exploratory", starts[model][0],
        bounds[model], truth, FDConfig(24, 8, 16), 1)
        for model, truth in truths.items() for dataset in ("clean", "noisy")]
    figures = study._figures({"summary": pd.DataFrame(rows),
        "parameters": pd.DataFrame(), "predictions": pd.DataFrame()})
    assert {"recovery", "convergence"} == set(figures)
    import matplotlib.pyplot as plt
    for figure in figures.values():
        plt.close(figure)
