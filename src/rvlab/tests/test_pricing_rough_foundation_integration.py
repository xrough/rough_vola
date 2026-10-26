"""The dynamics foundation stays separate from American pricing and SPY inputs."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rvlab.experiments import _pricing_research as research
from rvlab.experiments import _pricing_rough_foundation as foundation
from rvlab.pricing import (
    DiscountCurve, DividendSchedule, ExponentialKernel, FDConfig, PricingConfig,
    RoughHestonParams, RoughSimulationConfig, SpotMarket, UnsupportedModelError,
    approximate_fractional_kernel, european_payoff_samples, evaluate_vanilla,
    lifted_heston_price, simulate_rough_heston,
)


@pytest.mark.parametrize("kwargs,error", [
    ({"run_rough_foundation": 1}, ValueError),
    ({"rough_reuse_checkpoints": "yes"}, ValueError),
    ({"rough_study_kwargs": []}, TypeError),
    ({"rough_study_kwargs": {"seed": 17}}, ValueError),
    ({"rough_study_kwargs": {"reuse_checkpoints": False}}, ValueError),
    ({"rough_study_kwargs": {"reuse": False}}, ValueError),
    ({"run_rough_precision": 1}, ValueError),
    ({"rough_precision_kwargs": []}, TypeError),
    ({"rough_precision_kwargs": {"seed": 17}}, ValueError),
    ({"rough_precision_kwargs": {"reuse_checkpoints": False}}, ValueError),
    ({"rough_precision_kwargs": {"reuse": False}}, ValueError),
    ({"run_rough_precision": True, "run_rough_foundation": True,
      "rough_study_kwargs": {"paths": 17}}, ValueError),
])
def test_research_controls_reject_ambiguous_study_arguments(kwargs, error):
    with pytest.raises(error):
        research.pricing_research(SimpleNamespace(), **kwargs)


def test_notebook_bundle_namespaces_foundation_without_changing_main_backend(monkeypatch):
    plt = pytest.importorskip("matplotlib.pyplot")
    calls = []
    table = pd.DataFrame([{"accepted": False, "stage": "fixed-lift simulation"}])
    manifest = {"backend": "rough_heston_lifted_simulation_foundation",
                "american_pricing_ready": False, "stock_delta_ready": False}

    def study(context, **kwargs):
        calls.append(kwargs)
        return {"tables": {"summary": table}, "figures": {}, "manifest": manifest}

    monkeypatch.setattr(foundation, "rough_foundation_study", study)
    monkeypatch.setattr(research, "calibrate_vanilla_surface",
                        lambda *a, **k: pytest.fail("pricing gallery must not calibrate"))
    try:
        result = research.pricing_research(SimpleNamespace(progress=lambda _: None),
            fourier_config=PricingConfig(32, 40., 64), fd_config=FDConfig(24, 8, 20),
            run_rough_foundation=True, rough_reuse_checkpoints=False,
            rough_study_kwargs={"paths": 17, "compute_figures": False})
        assert calls == [{"seed": 42, "reuse_checkpoints": False,
                          "paths": 17, "compute_figures": False}]
        assert result["tables"]["rough_foundation_summary"] is table
        assert result["manifest"]["backend"] == research.MIXED_BACKEND
        assert result["manifest"]["rough_foundation_executed"]
        assert result["manifest"]["rough_foundation"] is manifest
        assert result["manifest"]["supplemental_backends"] == [manifest["backend"]]
        assert not manifest["american_pricing_ready"] and not manifest["stock_delta_ready"]
        assert set(result["tables"]["american_prices"].model) == {"Black–Scholes", "Heston"}
        assert result["manifest"]["optimizer_calls"] == 0
    finally:
        plt.close("all")


def test_rough_stopping_requires_explicit_kernel_and_preserves_foundation():
    params = RoughHestonParams(.04, .045, 1.5, .3, -.6, .15)
    market = SpotMarket(100., DiscountCurve.flat(.04, 1.), DividendSchedule((), 1.))
    with pytest.raises(ValueError, match="explicit ExponentialKernel"):
        evaluate_vanilla(params, market=market, strike=100., maturity=.25,
                         exercise_style="american")
    # The foundation remains usable through its explicit inception-only APIs.
    kernel = approximate_fractional_kernel(.15, 1., 2)
    assert isinstance(kernel, ExponentialKernel)
    result = simulate_rough_heston(params, market, kernel, observation_times=[0.],
        config=RoughSimulationConfig(paths=2, steps=1))
    np.testing.assert_array_equal(european_payoff_samples(result, strike=95., maturity=0.), 5.)
    assert lifted_heston_price(params, kernel=kernel, forward=100., discount=1.,
                               strike=95., maturity=0.) == 5.


def test_precision_workflow_keeps_historical_foundation_separate(monkeypatch):
    from rvlab.experiments import _pricing_rough_precision as precision
    plt = pytest.importorskip("matplotlib.pyplot")
    calls = []
    historical = {"tables": {"summary": pd.DataFrame([{"accepted": False}])},
        "figures": {}, "manifest": {"backend": "rough_heston_lifted_simulation_foundation",
            "historical_replay": True, "current_engine_execution": False}}
    current = {"tables": {"summary": pd.DataFrame([{"accepted": True}])},
        "figures": {}, "manifest": {"backend": "rough_heston_simulation_precision",
            "american_pricing_ready": False, "stock_delta_ready": False}}
    monkeypatch.setattr(foundation, "rough_foundation_study",
                        lambda *a, **k: pytest.fail("historical trials must not rerun under the new engine"))
    monkeypatch.setattr(precision, "load_historical_foundation", lambda context: historical)
    monkeypatch.setattr(precision, "rough_precision_study",
                        lambda context, **kwargs: calls.append(kwargs) or current, raising=False)
    try:
        result = research.pricing_research(SimpleNamespace(progress=lambda _: None),
            fourier_config=PricingConfig(32, 40., 64), fd_config=FDConfig(24, 8, 20),
            run_rough_foundation=True, run_rough_precision=True,
            rough_precision_kwargs={"paths": 17}, rough_reuse_checkpoints=False)
        assert calls == [{"seed": 42, "reuse_checkpoints": False, "paths": 17}]
        assert result["manifest"]["rough_foundation_included"]
        assert not result["manifest"]["rough_foundation_executed"]
        assert result["manifest"]["rough_foundation_execution_mode"] == "preserved_historical_records"
        assert result["manifest"]["rough_precision_executed"]
        assert result["manifest"]["rough_foundation"] is historical["manifest"]
        assert result["manifest"]["rough_precision"] is current["manifest"]
        assert result["tables"]["rough_foundation_summary"] is historical["tables"]["summary"]
        assert result["tables"]["rough_precision_summary"] is current["tables"]["summary"]
    finally:
        plt.close("all")


def test_export_of_frozen_figure_preserves_pixels_and_handles_same_path(tmp_path):
    import time
    from rvlab.experiments.pricing import DemoContext, _finish
    png = bytes.fromhex("89504e470d0a1a0a") + b"preserved image bytes"
    source = tmp_path/"original.png"
    source.write_bytes(png)
    context = DemoContext("test", tmp_path, time.perf_counter())
    bundle = {"tables": {}, "figures": {"copied": source}}
    _finish(context, bundle)
    assert (tmp_path/"copied.png").read_bytes() == png
    _finish(context, {"tables": {}, "figures": {"copied": tmp_path/"copied.png"}})
    assert source.read_bytes() == png
    with pytest.raises(FileNotFoundError):
        _finish(context, {"tables": {}, "figures": {"missing": tmp_path/"missing.png"}})


def test_display_of_frozen_figure_loads_original_bytes(tmp_path, monkeypatch):
    from rvlab.experiments.pricing import show_figures
    import IPython.display
    path = tmp_path/"frozen.png"
    pixels = bytes.fromhex("89504e470d0a1a0a") + b"original recorded bytes"
    path.write_bytes(pixels)
    shown = []
    monkeypatch.setattr(IPython.display, "Image", lambda **kwargs: kwargs["data"])
    monkeypatch.setattr(IPython.display, "display", shown.append)
    show_figures({"figures": {"history": path}})
    assert shown == [pixels] and path.read_bytes() == pixels
