"""Dual-tightening follow-ups preserve identity, pairing and failed gates."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from rvlab.experiments import _pricing_rough_dual_tightening as study


def _archive(tmp_path):
    archive = tmp_path/"snapshot"
    results = archive/"results"
    results.mkdir(parents=True)
    (results/"rough_stopping_attribution_matrix.csv").write_text("price,discrete_price_gate_passed\n2.856324123456789,False\n")
    (results/"rough_stopping_attribution_failures.csv").write_text("\n")
    (results/"rough_stopping_attribution_fixed_policy_bounds.png").write_bytes(b"original pixels")
    (archive/"source.py").write_text("# original source\n")
    manifest = {"evidence_version": "pricing_research_v1", "notebook": "01_pricing_and_exercise",
        "created_at": "original execution", "tables": {"rough_stopping_attribution_matrix": 1,
            "rough_stopping_attribution_failures": 0},
        "figures": ["rough_stopping_attribution_fixed_policy_bounds"],
        "rough_stopping_attribution": {"backend": "rough_heston_stopping_attribution",
            "profile_fingerprint": study.HISTORICAL_ATTRIBUTION_PROFILE,
            "source_elapsed_seconds": 500.123, "sources": {"original.py": "old-hash"}}}
    (results/"manifest.json").write_text(json.dumps(manifest))
    _inventory(archive)
    return archive


def _inventory(archive):
    records = [{"preserved": str(p.relative_to(archive)), "bytes": p.stat().st_size,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(archive.rglob('*')) if p.is_file() and p.name != "preservation_manifest.json"]
    (archive/"preservation_manifest.json").write_text(json.dumps({"files": records}))


def test_attribution_replay_preserves_original_profile_prices_and_execution_cost(tmp_path):
    archive = _archive(tmp_path)
    result = study.load_historical_attribution(archive=archive)
    manifest = result["manifest"]
    assert manifest["profile_fingerprint"] == study.HISTORICAL_ATTRIBUTION_PROFILE
    assert manifest["source_elapsed_seconds"] == 500.123
    assert manifest["sources"] == {"original.py": "old-hash"}
    assert manifest["historical_replay"] and not manifest["current_engine_execution"]
    assert not manifest["american_pricing_ready"] and not manifest["stock_delta_ready"]
    assert result["tables"]["failures"].empty
    assert result["tables"]["matrix"].price.item() == 2.856324123456789
    assert not result["tables"]["matrix"].discrete_price_gate_passed.any()
    assert result["figures"]["fixed_policy_bounds"].read_bytes() == b"original pixels"


@pytest.mark.parametrize("relative", ["source.py", "results/rough_stopping_attribution_matrix.csv",
    "results/rough_stopping_attribution_fixed_policy_bounds.png"])
def test_historical_replay_rejects_changed_source_prices_or_figures(tmp_path, relative):
    archive = _archive(tmp_path)
    (archive/relative).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed preserved"):
        study.load_historical_attribution(archive=archive)


def test_verified_different_profile_cannot_be_relabelled_as_historical_attribution(tmp_path):
    archive = _archive(tmp_path)
    path = archive/"results/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["rough_stopping_attribution"]["profile_fingerprint"] = "other"
    path.write_text(json.dumps(manifest))
    _inventory(archive)
    with pytest.raises(ValueError, match="declared six-case"):
        study.load_historical_attribution(archive=archive)


def _samples():
    shock = np.array([-.03, .02, .01, -.01, .025, -.015])
    raw = 2.+shock
    lower = raw[:, None]-shock[:, None]*[1., 2., 1.5]
    upper = np.broadcast_to([2.1, 2.05, 2.07], lower.shape).copy()
    return raw, lower, upper


def test_dual_changes_retain_pairing_lower_control_shift_and_unchanged_gate():
    raw, lower, upper = _samples()
    result = study.paired_dual_comparison(raw, lower, upper,
        control_names=["baseline", "stronger", "failed alternative"])
    comparison, changes = result["comparison"], result["paired_changes"]
    assert comparison.raw_price.nunique() == 1
    assert not comparison.discrete_price_gate_passed.any()
    assert changes.same_exercise_policy.all() and changes.raw_payoff_change.eq(0).all()
    assert changes.resolved_upper_reduction.all() and changes.resolved_gap_reduction.all()
    assert changes.lower_change_consistent_with_zero.all()
    expected = (upper[:, 1]-lower[:, 1])-(upper[:, 0]-lower[:, 0])
    first = changes.iloc[0]
    assert first.gap_change == pytest.approx(expected.mean())
    assert first.gap_change_standard_error == pytest.approx(expected.std(ddof=1)/np.sqrt(len(expected)))
    assert first.gap_change_interval_high == pytest.approx(first.gap_change+1.96*first.gap_change_standard_error)
    assert first.gap_reduction_fraction == pytest.approx(.5)
    assert not comparison.american_pricing_ready.any() and not comparison.stock_delta_ready.any()


def test_named_reference_reorders_all_sample_axes_consistently():
    raw, lower, upper = _samples()
    result = study.paired_dual_comparison(raw, lower, upper,
        control_names=["unit", "stronger", "other"], reference_control=1)
    assert result["comparison"].control.tolist() == ["stronger", "unit", "other"]
    assert result["paired_changes"].reference_control.eq("stronger").all()
    assert result["paired_changes"].dual_change.gt(0).all()


@pytest.mark.parametrize("bad", ["shape", "nonfinite", "domination", "names", "reference"])
def test_dual_comparison_rejects_malformed_or_invalid_control_samples(bad):
    raw, lower, upper = _samples()
    names, reference = ["unit", "stronger", "other"], 0
    if bad == "shape": raw = raw[:, None]
    elif bad == "nonfinite": lower[0, 0] = np.nan
    elif bad == "domination": upper[:] = 0.
    elif bad == "names": names = ["a"]
    elif bad == "reference": reference = True
    with pytest.raises(ValueError):
        study.paired_dual_comparison(raw, lower, upper,
            control_names=names, reference_control=reference)


@pytest.mark.parametrize("options,error", [
    ({"run_rough_dual_tightening": 1}, ValueError),
    ({"rough_dual_tightening_kwargs": []}, TypeError),
    ({"rough_dual_tightening_kwargs": {"seed": 7}}, ValueError),
    ({"run_rough_dual_tightening": True, "rough_stopping_attribution_kwargs": {"trial_specs": []}}, ValueError),
    ({"run_rough_dual_tightening": True, "rough_american_accuracy_kwargs": {"trial_specs": []}}, ValueError),
])
def test_workflow_rejects_ambiguous_tightening_and_historical_options(options, error):
    from types import SimpleNamespace
    from rvlab.experiments._pricing_research import pricing_research
    with pytest.raises(error):
        pricing_research(SimpleNamespace(), **options)


def test_tightening_workflow_replays_previous_attribution_without_current_source_repricing(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import matplotlib.pyplot as plt
    from rvlab.experiments import _pricing_research as research
    from rvlab.experiments import _pricing_rough_american as american
    from rvlab.experiments import _pricing_rough_stopping_attribution as previous
    from rvlab.pricing import PricingConfig, FDConfig
    old_american = {"tables": {}, "figures": {}, "manifest": {"backend": "old American", "historical_replay": True}}
    old_accuracy = {"tables": {}, "figures": {}, "manifest": {"backend": "old accuracy", "historical_replay": True}}
    old_attribution = {"tables": {"matrix": pd.DataFrame([{"price": 2.8563, "discrete_price_gate_passed": False}])},
        "figures": {}, "manifest": {"backend": "rough_heston_stopping_attribution",
            "historical_replay": True, "profile_fingerprint": study.HISTORICAL_ATTRIBUTION_PROFILE,
            "source_elapsed_seconds": 500.123}}
    current = {"tables": {"comparison": pd.DataFrame([{"control": "new"}])},
        "figures": {}, "manifest": {"backend": "rough_heston_dual_tightening"}}
    monkeypatch.setattr(american, "load_historical_american", lambda *a, **k: old_american)
    monkeypatch.setattr(previous, "load_historical_accuracy", lambda *a, **k: old_accuracy)
    monkeypatch.setattr(study, "load_historical_attribution", lambda *a, **k: old_attribution)
    monkeypatch.setattr(study, "add_dual_tightening_extras", lambda value: value)
    monkeypatch.setattr(previous, "rough_stopping_attribution_study", lambda *a, **k: pytest.fail("Must not reprice the historical six-case campaign"))
    calls = []
    monkeypatch.setattr(study, "rough_dual_tightening_study",
        lambda context, **options: calls.append(options) or current, raising=False)
    try:
        result = research.pricing_research(SimpleNamespace(progress=lambda _: None, output_dir=tmp_path),
            fourier_config=PricingConfig(32, 40., 64), fd_config=FDConfig(24, 8, 20),
            run_rough_stopping_attribution=True, run_rough_dual_tightening=True,
            rough_dual_tightening_kwargs={"trial_specs": []}, rough_reuse_checkpoints=False)
        manifest = result["manifest"]
        assert manifest["rough_stopping_attribution_included"]
        assert not manifest["rough_stopping_attribution_executed"]
        assert manifest["rough_stopping_attribution_execution_mode"] == "preserved_historical_records"
        assert manifest["rough_stopping_attribution"] is old_attribution["manifest"]
        assert manifest["rough_dual_tightening_executed"]
        assert result["tables"]["rough_stopping_attribution_matrix"] is old_attribution["tables"]["matrix"]
        assert calls == [{"seed": 42, "reuse_checkpoints": False, "trial_specs": []}]
    finally:
        plt.close("all")


def test_figures_keep_failed_acceptance_and_paired_upper_lower_changes_visible():
    import matplotlib.pyplot as plt
    raw, lower, upper = _samples()
    tables = study.paired_dual_comparison(raw, lower, upper,
        control_names=["unit", "stronger", "other"])
    for frame in tables.values(): frame.insert(0, "case", "finite rough target")
    figures = study._tightening_figures(tables)
    assert set(figures) == {"gap_and_acceptance", "paired_upper_changes"}
    assert not tables["comparison"].discrete_price_gate_passed.any()
    for figure in figures.values():
        figure.canvas.draw()
        plt.close(figure)
    assert study._tightening_figures({}) == {}


def _case(**updates):
    return {"case": "central rough", "contract_id": "rough_put_60_100", "factors": 4,
        "config": {"training_method": "fourier_premium", "premium_cap": .7,
            "training_paths": 8, "valuation_paths": 8, "steps": 4, "exercise_steps": 2},
        "dual_configs": {"soft boundary intrinsic": {"training_paths": 8, "candidate_bins": (1,)}},
        **updates}


def test_new_campaign_reserves_fresh_disjoint_policy_dual_proxy_and_valuation_streams():
    design = study._validated_tightening_design([_case(), _case(case="repeat")], 42)
    streams = [s for row in design for s in (row["config"].training_seed,
        row["dual_seed"], row["improvement_seed"], row["config"].valuation_seed)]
    assert len(set(streams)) == 8
    assert min(streams) > 160_000_000
    assert design[0]["config"].training_seed == 170_000_042
    assert design[0]["dual_seed"] == 180_000_042
    assert design[0]["improvement_seed"] == 190_000_042
    assert design[0]["config"].valuation_seed == 165_000_042
    assert all(not row["policy_variants"] for row in design)


@pytest.mark.parametrize("updates", [
    {"policy_variants": {"changed exercise": {"premium_floor": 0., "premium_cap": .7}}},
    {"dual_configs": {"new only": {"training_paths": 8, "candidate_bins": (1,)}}},
    {"dual_configs": {"soft boundary intrinsic": {"training_paths": 8, "candidate_bins": (1,), "selection_objective": "gap"}}},
])
def test_new_campaign_rejects_policy_changes_missing_old_control_or_changed_selection(updates):
    with pytest.raises(ValueError):
        study._validated_tightening_design([_case(**updates)], 42)


@pytest.mark.parametrize('historical_replay', [False, True])
def test_notebook_tightening_display_preserves_all_failed_cases_without_assuming_comparisons(historical_replay):
    import nbformat
    from pathlib import Path
    notebook = nbformat.read(Path(__file__).resolve().parents[3]/
        "notebooks/legacy/pricing/first_stage_pre_report_2026-10-20/01_pricing_and_exercise.ipynb", as_version=4)
    cells = [c.source for c in notebook.cells if c.cell_type == "code" and
        c.source.startswith('if results["manifest"].get("rough_dual_tightening_included"')]
    assert len(cells) == 3
    messages = []
    environment = {"pd": pd, "results": {"manifest": {"rough_dual_tightening_executed": not historical_replay,
        "rough_dual_tightening_included": True, "rough_dual_tightening": {}}, "figures": {}},
        "tables": {"rough_dual_tightening_failures": pd.DataFrame([{"error": "retained"}])},
        "display": messages.append, "Markdown": str, "present_table": lambda *a, **k: None}
    for cell in cells: exec(cell, environment)
    assert any("No completed dual-tightening" in value for value in messages)


def test_probe_loader_reads_both_inventory_schemas_without_rewriting_sealed_evidence(tmp_path):
    archives = []
    for key in ("path", "preserved"):
        folder = tmp_path/key
        folder.mkdir()
        payload = b"status,gap\nfailed,0.02\n"
        (folder/"valuation.csv").write_bytes(payload)
        records = [{key: "valuation.csv", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}]
        manifest = {"files": records, "inventory_sha256": hashlib.sha256(json.dumps(records,
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()}
        (folder/"preservation_manifest.json").write_text(json.dumps(manifest))
        archives.append(folder)
    before = {p: p.read_bytes() for folder in archives for p in folder.iterdir()}
    table, records = study._load_tightening_probes(archives)
    assert len(table) == len(records) == 2
    assert table.status.eq("failed").all() and table.exploratory_only.all()
    assert not table.independent_acceptance_evidence.any()
    assert all(path.read_bytes() == content for path, content in before.items())
    (archives[0]/"valuation.csv").write_text("changed")
    with pytest.raises(ValueError, match="Changed preserved"):
        study._load_tightening_probes(archives)


def test_after_stopping_control_changes_report_identical_lower_samples_pathwise():
    raw, lower, upper = _samples()
    lower[:, 1] = lower[:, 0]
    result = study.paired_dual_comparison(raw, lower, upper,
        control_names=["before", "post-stop only", "ordinary"])
    change = result["paired_changes"].iloc[0]
    assert change.controlled_lower_samples_identical
    assert change.controlled_lower_change == change.lower_change_standard_error == 0.
    assert change.gap_change == pytest.approx(change.dual_change, abs=1e-14)
    assert change.gap_change_standard_error == pytest.approx(change.dual_change_standard_error, abs=1e-14)
    assert result["comparison"].controlled_lower_samples_identical_to_reference.tolist() == [True, True, False]


def test_formal_design_declares_all_cases_meshes_and_eight_controls_before_valuation():
    specs = study.dual_tightening_trial_specs()
    design = study._validated_tightening_design(specs, 42)
    assert len(design) == 6
    assert [r["config"].training_paths for r in design] == [4096]*4+[8192,4096]
    assert [r["config"].steps for r in design] == [64]*5+[128]
    assert [r["config"].exercise_steps for r in design] == [32]*5+[64]
    assert all(len(r["obstacle_proxy"]["stock_knots"]) == 20 for r in design)
    assert all(len(r["dual_configs"])+len(r["obstacle_proxy"]["controls"])+2 == 8 for r in design)
    for row in design:
        pure = row["dual_configs"]["micro poststop only"]
        assert pure.european_bounds == pure.premium_bounds == (1.,1.)
        assert pure.components == ("european","premium","micro_poststop")
        assert pure.predictable_features == ()


def test_cross_proxy_selection_compares_actual_scores_and_records_different_unit_baselines():
    from types import SimpleNamespace
    def fit(baseline, selected):
        return SimpleNamespace(diagnostics={"selection_objective":"error_budget",
            "selected_score":selected,"selected_candidate":"1 bins",
            "selection_source":"held-out dual-training paths only",
            "candidates":[{"heldout_error_budget":baseline}]})
    fits={"old soft":("regression",fit(.01,.008)),
        "obstacle soft":("obstacle",fit(.009,.006)),
        "obstacle micro":("obstacle",fit(.009,.007))}
    selected, rows=study._select_proxy_controls(fits,{"regression":"unit","obstacle":"obstacle unit"})
    assert selected == "obstacle soft"
    assert len(rows) == 5
    assert {r['proxy_unit_training_score'] for r in rows} == {.01,.009}
    assert sum(r['selected_before_valuation'] for r in rows) == 1
    assert not any(r['valuation_used_for_selection'] for r in rows)


@pytest.mark.parametrize('field',['stocks','factors','discounted_payoffs','stop_indices'])
def test_family_trace_comparison_rejects_policy_or_simulated_path_changes(field):
    reference={key:np.zeros((2,3,1)) for key in
        ('times','eligible','stocks','factors','discounted_payoffs','stop_indices')}
    candidate={key:value.copy() for key,value in reference.items()}
    candidate[field][0,0,0]=1
    with pytest.raises(ValueError,match='shared simulation or frozen exercise policy'):
        study._assert_same_policy_traces(reference,candidate)


@pytest.mark.parametrize('options',[
    {'stock_knots':(.8,.9,1.),'controls':('missing',)},
    {'stock_knots':(.8,.8,1.),'controls':('soft boundary intrinsic',)},
    {'stock_knots':(.8,.9,1.),'controls':()},
    {'stock_knots':(.8,.9,1.),'controls':('soft boundary intrinsic',),'unexpected':True},
])
def test_obstacle_mesh_and_control_declarations_are_validated_before_simulation(options):
    with pytest.raises(ValueError):
        study._validated_tightening_design([_case(obstacle_proxy=options)],42)


def test_late_exploratory_extras_are_display_only_and_do_not_change_formal_profile(tmp_path):
    folder=tmp_path/'extra';folder.mkdir();content=b'status,gap\nfailed,0.03\n';(folder/'results.csv').write_bytes(content)
    records=[{'path':'results.csv','bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}]
    (folder/'preservation_inventory.json').write_text(json.dumps({'files':records}))
    original={'tables':{},'manifest':{'profile_fingerprint':'frozen','sources':{'original':'hash'},'preserved_probes':[]}}
    result=study.add_dual_tightening_extras(original,archives=(folder,))
    assert result['manifest']['profile_fingerprint']=='frozen'
    assert result['manifest']['sources']=={'original':'hash'} and result['manifest']['preserved_probes']==[]
    assert not result['manifest']['experimental_extras'][0]['formal_campaign_input']
    assert result['tables']['supplemental_probes'].status.eq('failed').all()
    assert not result['tables']['supplemental_probe_inventory'].formal_campaign_input.any()


def test_poststop_figure_shows_direct_upper_intervals_separately_from_selected_proxy_changes():
    import matplotlib.pyplot as plt
    raw, lower, upper=_samples()
    tables=study.paired_dual_comparison(raw,lower,upper,control_names=['unit','poststop','other'])
    for frame in tables.values(): frame.insert(0,'case','declared finite target')
    tables['poststop_vs_unit']=tables['paired_changes'].iloc[:1].copy()
    figures=study._tightening_figures(tables)
    assert 'poststop_upper_only' in figures
    for figure in figures.values():figure.canvas.draw();plt.close(figure)
