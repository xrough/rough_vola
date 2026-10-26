"""American market routing cannot reuse European SPY approximation evidence."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rvlab.experiments import _pricing_american as study
from rvlab.experiments import pricing
from rvlab.experiments.pricing_presentation import load_recorded_results


def blocked_inputs():
    return {"audit": pd.DataFrame([{"date": day, "ready": False, "contracts": 0,
        "blocker": "Independent stock prices; cash-dividend expectation coverage",
        "stock_observed": False, "dividend_expectations_available": False} for day in study.DATES]),
        "snapshots": {}, "quotes": pd.DataFrame(), "exclusions": pd.DataFrame(),
        "fingerprints": [], "processed_root": "missing"}


def test_all_spy_workflows_use_american_gate_without_fitting_when_blocked(monkeypatch, tmp_path):
    pytest.importorskip("matplotlib").use("Agg", force=True)
    monkeypatch.setattr(study, "load_american_spy_inputs", lambda *a, **k: blocked_inputs())
    def forbidden(*args, **kwargs):
        pytest.fail("calibration must not run with missing observed stock or dividend inputs")
    monkeypatch.setattr(study, "_fit_ready_surface", forbidden)
    runners = [pricing.run_spy_demo, pricing.run_accuracy_demo, pricing.run_attribution_demo,
               pricing.run_variance_curve_demo, pricing.run_curve_convergence_demo]
    for i, run in enumerate(runners):
        context = pricing.setup_demo(f"gate_{i}", output_dir=tmp_path / str(i))
        result = run(context)
        assert result["manifest"]["backend"] == study.AMERICAN_BACKEND
        assert result["manifest"]["exercise_style"] == "american"
        assert result["manifest"]["status"] == "blocked_inputs"
        assert not result["manifest"]["calibration_ready"]
        assert not result["manifest"]["hedging_ready"]
        assert "fit_summary" not in result["tables"]
        assert "calibration_residuals" not in result["figures"]
    import matplotlib.pyplot as plt
    plt.close("all")


def test_missing_canonical_archive_is_an_explicit_audit_not_a_synthetic_surface(tmp_path):
    result = study.load_american_spy_inputs(tmp_path)
    assert result["quotes"].empty and result["snapshots"] == {}
    assert not result["audit"].ready.any()
    assert result["audit"].blocker.str.contains("Missing canonical option observations").all()


def test_dividend_coverage_never_inherits_a_stale_readiness_declaration(tmp_path, monkeypatch):
    from rvlab.market_data import pipeline, snapshots
    directory = tmp_path / "options/session_date=2025-08-07"
    directory.mkdir(parents=True)
    (directory / "quotes.parquet").write_bytes(b"canonical fixture")
    (tmp_path / "readiness.json").write_text(json.dumps({"schema_version": 2,
        "calibration_backend": study.AMERICAN_BACKEND, "exercise_style": "american",
        "dividend_coverage_through": "2099-01-01T00:00:00Z"}))
    options = pd.DataFrame({"expiry_at": [pd.Timestamp("2025-08-22 20:00Z")], "strike": [100.]})
    monkeypatch.setattr(pipeline, "read_option_window", lambda *a, **k: options.copy())
    seen = []
    def build(quotes, as_of, **kwargs):
        seen.append(kwargs["dividend_coverage_through"])
        return SimpleNamespace(quotes=quotes, exclusions=pd.DataFrame(), provenance={}, ready=False,
            missing_inputs=("dividend expectation coverage missing",))
    monkeypatch.setattr(snapshots, "build_calibration_snapshot", build)
    study.load_american_spy_inputs(tmp_path, dates=["2025-08-07"])
    study.load_american_spy_inputs(tmp_path, dates=["2025-08-07"],
        dividend_coverage_through="2025-09-05T20:00:00Z")
    assert seen == [None, "2025-09-05T20:00:00Z"]


def test_default_output_paths_keep_superseded_evidence_separate(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.setattr(pricing, "ROOT", tmp_path)
    spy = pricing.setup_demo("02_spy_pricing")
    synthetic = pricing.setup_demo("03_calibration_identification")
    european = pricing.setup_demo("01_synthetic_pricing")
    assert spy.output_dir == tmp_path / "notebooks/_output/pricing/american_v1/02_spy_pricing"
    assert synthetic.output_dir == tmp_path / "notebooks/_output/pricing/synthetic_v2/03_calibration_identification"
    assert european.output_dir == tmp_path / "notebooks/_output/pricing/01_synthetic_pricing"


def test_presentation_rejects_old_spy_artifacts_before_loading_tables(tmp_path):
    path = tmp_path / "02_spy_pricing"
    path.mkdir()
    (path / "manifest.json").write_text(json.dumps({"notebook": path.name, "tables": {}, "figures": []}))
    with pytest.raises(ValueError, match="Superseded European"):
        load_recorded_results(path.name, artifact_root=tmp_path)
    current = {"notebook": path.name, "tables": {}, "figures": [],
        "backend": study.AMERICAN_BACKEND, "evidence_version": study.AMERICAN_EVIDENCE_VERSION,
        "exercise_style": "american", "status": "blocked_inputs"}
    (path / "manifest.json").write_text(json.dumps(current))
    assert load_recorded_results(path.name, artifact_root=tmp_path)["manifest"] == current


def test_presentation_rejects_mixed_spy_identification_artifacts(tmp_path):
    path = tmp_path / "03_calibration_identification"
    path.mkdir()
    manifest = {"notebook": path.name, "tables": {}, "figures": [], "spy_inputs": {}}
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="synthetic only"):
        load_recorded_results(path.name, artifact_root=tmp_path)


def test_preserved_american_notebooks_do_not_contain_old_empirical_fits():
    nbformat = pytest.importorskip("nbformat")
    for stem in study.SPY_NOTEBOOKS:
        nb = nbformat.read(pricing.ROOT / "notebooks/legacy/pricing" / f"{stem}.ipynb", 4)
        sources = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")
        assert "PricingConfig(" not in sources
        assert "implied_volatility(" not in sources
        assert "load_recorded_results(" not in sources
        assert "load_spy_demo(" not in sources
    nb = nbformat.read(pricing.ROOT / "notebooks/legacy/pricing/03_calibration_identification.ipynb", 4)
    assert "SPY_2025-08-07" not in "\n".join(cell.source for cell in nb.cells)


def test_contract_selection_is_deterministic_and_training_is_disjoint():
    from rvlab.pricing.vanilla_types import DiscountCurve, DividendSchedule, SpotMarket
    market = SpotMarket(100., DiscountCurve.flat(.04, 1.), DividendSchedule((), 1.))
    rows = [{"contract_id": f"{expiry}-{strike}-{call}", "expiry_at": f"{expiry} 20:00Z",
        "maturity": .1, "strike": strike, "is_call": call}
        for expiry in study.EXPIRIES for strike in (94., 96., 97., 98., 99., 101., 102., 103., 104., 106.)
        for call in (False, True)]
    class Snapshot:
        def __init__(self, quotes): self.quotes = quotes
        def calibration_inputs(self): return {"quotes": self.quotes, "market": market}
    quotes = pd.DataFrame(rows)
    first, actual_market = study.select_american_contracts(Snapshot(quotes))
    second, _ = study.select_american_contracts(Snapshot(quotes.sample(frac=1., random_state=42)))
    pd.testing.assert_frame_equal(first, second)
    assert actual_market is market
    assert len(first) == 18 and first.contract_id.is_unique
    assert first.split.value_counts().to_dict() == {"train": 12, "heldout": 6}
    assert set(first.loc[first.split.eq("train"), "contract_id"]).isdisjoint(
        first.loc[first.split.eq("heldout"), "contract_id"])
    assert first.exercise_style.eq("american").all()
    assert "forward" not in first and "anchors" not in first


def test_internal_legacy_demo_names_redirect_to_current_workflow(monkeypatch):
    import importlib
    seen = []
    monkeypatch.setattr(study, "american_spy_demo", lambda context, **kwargs: seen.append(kwargs["study"]))
    for module, name in [("_pricing_spy", "spy_demo"), ("_pricing_accuracy", "accuracy_demo"),
            ("_pricing_attribution", "attribution_demo"), ("_pricing_variance_curve", "variance_curve_demo"),
            ("_pricing_curve_convergence", "curve_convergence_demo")]:
        getattr(importlib.import_module(f"rvlab.experiments.{module}"), name)(None)
    assert seen == ["pricing", "accuracy", "attribution", "variance-curve", "curve-convergence"]


def test_american_synthetic_routes_requested_budgets_and_keeps_coarse_failures(monkeypatch, tmp_path):
    from rvlab import pricing as api
    from rvlab.experiments._pricing_american_synthetic import american_synthetic_demo
    evaluations, fits = [], []
    def evaluate(params, **kwargs):
        evaluations.append((type(params).__name__, kwargs["config"]))
        return SimpleNamespace(prices=np.zeros_like(np.asarray(kwargs["strike"], float)),
            deltas=np.zeros_like(np.asarray(kwargs["strike"], float)), exercise_boundaries=[])
    def fit(quotes, initial, **kwargs):
        fits.append((type(initial).__name__, kwargs["config"]))
        raise api.PricingError("retained calibration failure")
    monkeypatch.setattr(api, "evaluate_vanilla", evaluate)
    monkeypatch.setattr(api, "calibrate_vanilla_surface", fit)
    result = american_synthetic_demo(pricing.setup_demo("budget_test", output_dir=tmp_path), max_nfev=1)
    actual = [(name, settings.spot_steps, settings.variance_steps, settings.time_steps)
              for name, settings in fits]
    assert actual == [("BlackScholesParams", 100, 32, 100), ("BlackScholesParams", 400, 80, 400),
        ("HestonParams", 100, 32, 100), ("HestonParams", 240, 80, 300)]
    observed = {(name, settings.spot_steps, settings.variance_steps, settings.time_steps)
                for name, settings in evaluations}
    assert ("BlackScholesParams", 800, 160, 800) in observed
    assert ("HestonParams", 480, 160, 600) in observed
    attempts = result["tables"]["calibration_summary"]
    assert len(attempts) == 4 and attempts.status.eq("numerical_failure").all()
    assert set(attempts.study) == {"deliberately coarse", "requested starting budget"}
    assert result["manifest"]["calibration_configs"]["heston"]["spot_steps"] == 240
    import matplotlib.pyplot as plt
    plt.close("all")


def test_american_synthetic_rejects_generation_at_calibration_resolution(tmp_path):
    from rvlab.pricing import FDConfig
    from rvlab.experiments._pricing_american_synthetic import american_synthetic_demo
    configs = {"bs": FDConfig(20, 6, 20), "heston": FDConfig(20, 6, 20)}
    with pytest.raises(ValueError, match="strictly finer"):
        american_synthetic_demo(pricing.setup_demo("bad_budget", output_dir=tmp_path),
            config=FDConfig(20, 6, 20), calibration_configs=configs, generation_configs=configs)


@pytest.mark.parametrize("failed", [False, True])
def test_ready_path_fits_raw_american_prices_and_preserves_failed_attempts(monkeypatch, failed):
    from rvlab import pricing as api
    from rvlab.pricing.vanilla_types import DiscountCurve, DividendSchedule, SpotMarket, FDConfig
    market = SpotMarket(100., DiscountCurve.flat(.04, 1.), DividendSchedule((), 1.))
    quotes = pd.DataFrame({"contract_id": ["a", "b", "c"], "strike": [97., 101., 103.],
        "maturity": [.1, .1, .1], "is_call": [False, True, True],
        "market_price": [1., 2., 1.], "bid": [.98, 1.98, .98], "ask": [1.02, 2.02, 1.02],
        "split": ["train", "train", "heldout"], "exercise_style": "american"})
    monkeypatch.setattr(study, "select_american_contracts", lambda snapshot: (quotes.copy(), market))
    calls = []
    def fit(training, initial, **kwargs):
        calls.append((training.copy(), kwargs))
        assert kwargs["exercise_style"] == "american" and kwargs["market"] is market
        assert "forward" not in kwargs and "discount" not in kwargs
        if failed:
            raise api.PricingError("exercise solver failed")
        return SimpleNamespace(params=initial, optimizer_success=True, ready=True,
            nfev=2, pricing_evaluations=7, active_bounds={})
    monkeypatch.setattr(api, "calibrate_vanilla_surface", fit)
    monkeypatch.setattr(api, "evaluate_vanilla", lambda *a, **k: SimpleNamespace(
        prices=np.array([1., 2., 1.01]), deltas=np.array([-.2, .5, .4]),
        price_stable=True, delta_stable=True))
    tables, figures, manifest = {}, {}, {}
    study._fit_ready_surface(None, "pricing", FDConfig(20, 6, 20), 4, tables, figures, manifest)
    assert len(calls) == 2
    for frame, _ in calls:
        pd.testing.assert_series_equal(frame.market_price, quotes.iloc[:2].market_price)
        assert frame.exercise_style.eq("american").all()
    summary = tables["fit_summary"]
    if failed:
        assert summary.status.eq("numerical_failure").all()
        assert summary.error.str.contains("exercise solver failed").all()
        assert not summary.ready.any() and tables["fitted_prices"].empty
        assert "calibration_residuals" not in figures
    else:
        assert summary.status.eq("converged").all()
        np.testing.assert_allclose(summary.test_rmse, .01)
        assert len(tables["fitted_prices"]) == 6
        import matplotlib.pyplot as plt
        plt.close("all")
