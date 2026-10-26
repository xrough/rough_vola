"""Preserved evidence and strict acceptance for the simulator follow-up."""
import hashlib
import json

import pytest

from rvlab.experiments import _pricing_rough_precision as precision


def _archive(tmp_path):
    """Tiny byte-inventoried archive for read-only historical replay tests."""
    root = tmp_path/"snapshot"
    results = root/"results"
    results.mkdir(parents=True)
    (results/"rough_foundation_summary.csv").write_text("stage,accepted\nold pilot,False\n")
    (results/"rough_foundation_failures.csv").write_text("\n")
    (results/"rough_foundation_kernel_and_prices.png").write_bytes(b"historical png pixels")
    recorded = {"evidence_version":"pricing_research_v1",
        "notebook":"01_pricing_and_exercise", "created_at":"2026-10-18T00:00:00Z",
        "tables":{"rough_foundation_summary":1,"rough_foundation_failures":0,
                  "unrelated_pricing_table":10},
        "figures":["rough_foundation_kernel_and_prices", "unrelated_plot"],
        "rough_foundation":{"backend":"rough_heston_lifted_simulation_foundation",
            "source_elapsed_seconds":55., "profile_fingerprint":"original-profile",
            "sources":{"rough_simulation.py":"historical-source"},
            "american_pricing_ready":False,"stock_delta_ready":False}}
    (results/"manifest.json").write_text(json.dumps(recorded))
    rows=[]
    for path in sorted(results.iterdir()):
        payload=path.read_bytes()
        rows.append({"source":"original/"+path.name,"preserved":str(path.relative_to(root)),
            "bytes":len(payload),"sha256":hashlib.sha256(payload).hexdigest()})
    (root/"preservation_manifest.json").write_text(json.dumps({"files":rows}))
    return root,recorded


def test_historical_loader_preserves_original_source_cost_and_failed_results(tmp_path):
    archive, original = _archive(tmp_path)
    result = precision.load_historical_foundation(archive=archive)
    assert set(result["tables"]) == {"summary","failures"}
    assert not result["tables"]["summary"].iloc[0].accepted
    assert result["tables"]["failures"].empty
    assert result["figures"]["kernel_and_prices"].read_bytes() == b"historical png pixels"
    manifest=result["manifest"]
    assert manifest["sources"] == original["rough_foundation"]["sources"]
    assert manifest["profile_fingerprint"] == "original-profile"
    assert manifest["source_elapsed_seconds"] == 55.
    assert manifest["historical_replay"] and not manifest["current_engine_execution"]
    assert not manifest["american_pricing_ready"] and not manifest["stock_delta_ready"]
    assert manifest["historical_archive_files"] == 4
    assert precision.load_historical_foundation(archive=archive,
        compute_figures=False)["figures"] == {}


@pytest.mark.parametrize("target",["rough_foundation_summary.csv","rough_foundation_kernel_and_prices.png"])
def test_historical_loader_rejects_changed_table_or_pixels(tmp_path,target):
    archive,_=_archive(tmp_path)
    (archive/"results"/target).write_bytes(b"later modified evidence")
    with pytest.raises(ValueError,match="Changed preserved"):
        precision.load_historical_foundation(archive=archive)


def test_historical_loader_checks_every_inventoried_source_not_only_displayed_rows(tmp_path):
    archive,_=_archive(tmp_path)
    path=archive/"source.py";path.write_text("# original numerical source\n")
    manifest_path=archive/"preservation_manifest.json"
    inventory=json.loads(manifest_path.read_text())
    payload=path.read_bytes()
    inventory["files"].append({"preserved":"source.py","bytes":len(payload),
        "sha256":hashlib.sha256(payload).hexdigest()})
    manifest_path.write_text(json.dumps(inventory))
    path.unlink()
    with pytest.raises(ValueError,match="Missing preserved"):
        precision.load_historical_foundation(archive=archive)


def test_historical_loader_rejects_paths_outside_snapshot(tmp_path):
    archive,_=_archive(tmp_path)
    manifest_path=archive/"preservation_manifest.json"
    manifest_path.write_text(json.dumps({"files":[{"preserved":"../outside.py"}]}))
    with pytest.raises(ValueError,match="remain inside"):
        precision.load_historical_foundation(archive=archive)


def test_historical_loader_rejects_other_study_identity_after_inventory_verification(tmp_path):
    archive,_=_archive(tmp_path)
    path=archive/"results/manifest.json"
    record=json.loads(path.read_text());record["rough_foundation"]["backend"]="new simulator"
    path.write_text(json.dumps(record))
    manifest_path=archive/"preservation_manifest.json"
    inventory=json.loads(manifest_path.read_text())
    payload=path.read_bytes()
    row=next(row for row in inventory["files"] if row["preserved"]=="results/manifest.json")
    row.update(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
    manifest_path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError,match="not the rough-Heston foundation"):
        precision.load_historical_foundation(archive=archive)


def test_independent_pooling_matches_concatenated_path_statistics():
    import numpy as np
    samples=[np.array([[1.,2.],[3.,7.],[4.,9.]]),
        np.array([[2.,1.],[4.,2.],[8.,4.],[9.,8.]])]
    means=[a.mean(axis=0) for a in samples]
    errors=[a.std(axis=0,ddof=1)/np.sqrt(len(a)) for a in samples]
    mean,error=precision.pooled_independent_summary(means,errors,[len(a) for a in samples])
    combined=np.concatenate(samples)
    assert mean == pytest.approx(combined.mean(axis=0))
    assert error == pytest.approx(combined.std(axis=0,ddof=1)/np.sqrt(len(combined)))
    # Averaging the two standard errors would ignore the unequal sample sizes
    # and the shift between independent sample means.
    assert np.all(error > 0)


@pytest.mark.parametrize("means,errors,counts",[
    ([1.,2.],[.1],[3,3]),([1.,2.],[.1,.1],[3]),
    ([1.,2.],[.1,.1],[3.,3.]),([1.,2.],[.1,.1],[True,True]),
    ([1.,2.],[.1,-.1],[3,3]),([1.,float('nan')],[.1,.1],[3,3]),
    ([1.,2.],[.1,.1],[1,3])])
def test_pooling_rejects_nonindependent_or_malformed_statistics(means,errors,counts):
    with pytest.raises(ValueError,match="Independent summaries"):
        precision.pooled_independent_summary(means,errors,counts)


def test_precision_acceptance_requires_same_lift_and_full_model_gates():
    accepted=precision.precision_acceptance(1.,.0001,lift_reference=1.,
        fractional_reference=1.0001,lift_stable=True,fractional_stable=True)
    assert accepted["accepted"]
    mismatch=precision.precision_acceptance(1.,.0001,lift_reference=1.,
        fractional_reference=1.01,lift_stable=True,fractional_stable=True)
    assert mismatch["same_lift_accepted"] and not mismatch["accepted"]
    unstable=precision.precision_acceptance(1.,0.,lift_reference=1.,
        fractional_reference=1.,lift_stable=False,fractional_stable=True)
    assert not unstable["accepted"]
    wide=precision.precision_acceptance(1.,.001,lift_reference=1.,
        fractional_reference=1.,lift_stable=True,fractional_stable=True)
    assert not wide["accepted"]


def _mock_campaign(monkeypatch,tmp_path,*,failed_curve_seed=None):
    from types import SimpleNamespace
    import numpy as np
    from rvlab.pricing import ExponentialKernel,PricingError
    from rvlab import pricing
    from rvlab.experiments import _pricing_rough_foundation as foundation
    monkeypatch.setattr(pricing,"approximate_fractional_kernel",lambda h,t,n,**kw:
        ExponentialKernel((0. if h==.5 else 1.,),(1.,),hurst=h,horizon=t))
    monkeypatch.setattr(foundation,"_reference_prices",lambda params,contracts,market,config:
        {"status":"completed","prices":[1.]*len(contracts)})
    monkeypatch.setattr(foundation,"_lift_prices",lambda params,contracts,market,kernel,config:
        {"status":"completed","prices":[1.]*len(contracts)})
    monkeypatch.setattr(precision,"_precision_limits",lambda *a,**k:([{"control":"constant","absolute_error":0.,"ci_half_width":0.,"conservative_error":0.,"price_allowance":.001,"accepted":True}],[]))
    # This mocked campaign tests pooling and failure retention. Exercise the real
    # helper guard against a tiny inventoried fixture, independently of omitted
    # historical report payloads; dedicated tests below cover guard tampering.
    archive,current,_=_shared_guard_fixture(tmp_path)
    actual_guard=precision._ensure_shared_helper_guard
    monkeypatch.setattr(precision,"_ensure_shared_helper_guard",lambda folder:
        actual_guard(folder,current_source=current,archive=archive))
    calls=[]
    def estimate(params,market,kernel,contracts,**kwargs):
        from rvlab.pricing import RoughHestonCurveParams
        calls.append((type(params).__name__,kwargs["stock_integral"],kwargs["steps"],kwargs["seed"]))
        if isinstance(params,RoughHestonCurveParams) and kwargs["seed"]==failed_curve_seed:
            raise PricingError("deliberate retained failed replication")
        means=[1.+.0001*(kwargs["seed"]-42)]*len(contracts)
        return {"status":"completed","prices":means,"standard_errors":[.00001]*9,
            "conditional_prices":means,"conditional_standard_errors":[.1]*9,
            "control_means":[0.]*9,"control_standard_errors":[.1]*9,
            "sample_covariances":np.zeros((9,3,3)).tolist(),"diagnostics":{}}
    monkeypatch.setattr(precision,"_estimate_trial",estimate)
    context=SimpleNamespace(output_dir=tmp_path,progress=lambda _:None)
    result=precision.rough_precision_study(context,paths=4,steps=(4,8),final_steps=8,
        replications=3,compute_figures=False)
    return result,calls


def test_campaign_pools_every_predeclared_seed_without_picking_the_best(monkeypatch,tmp_path):
    result,calls=_mock_campaign(monkeypatch,tmp_path)
    pooled=result["tables"]["prices"].query("stage == 'pooled final'")
    assert set(pooled.model)=={"flat","curve","Hurst .5"}
    assert pooled.paths.eq(12).all() and pooled.replications.eq(3).all()
    assert pooled.estimate.to_numpy()==pytest.approx([1.0001]*54)
    assert len(calls)==13
    assert result["manifest"]["tested_european_prices_accepted"]
    assert not result["manifest"]["american_pricing_ready"]
    assert not result["manifest"]["stock_delta_ready"]
    assert not result["tables"]["controls"].continuous_reference_used_to_center.any()


def test_failed_replication_is_retained_and_incomplete_campaign_is_not_pooled(monkeypatch,tmp_path):
    result,calls=_mock_campaign(monkeypatch,tmp_path,failed_curve_seed=43)
    pooled=result["tables"]["prices"].query("stage == 'pooled final'")
    assert "curve" not in set(pooled.model)
    assert not result["manifest"]["tested_european_prices_accepted"]
    failures=result["tables"]["failures"]
    assert failures.error.str.contains("deliberate retained").any()
    assert failures.error.str.contains("no incomplete campaign").any()
    assert result["tables"]["trials"].status.eq("numerical_failure").sum()==1


@pytest.mark.parametrize("enabled",[False,True])
def test_precision_notebook_displays_refresh_from_artifacts_without_parameter_globals(enabled):
    from pathlib import Path
    import numpy as np
    import pandas as pd
    nbformat=pytest.importorskip("nbformat")
    notebook=nbformat.read(Path(precision.__file__).resolve().parents[3]/
        "notebooks/legacy/pricing/first_stage_pre_report_2026-10-20/01_pricing_and_exercise.ipynb",as_version=4)
    cells=[cell for cell in notebook.cells if cell.cell_type=="code" and
        cell.metadata.get("rvlab_presentation") and "rough_precision_" in cell.source]
    assert len(cells)==3
    summary=pd.DataFrame({"stage":["time refinement","pooled final"],
        "model":["flat","flat"],"estimator":["discrete martingale control"]*2})
    tables={"rough_precision_summary":summary,
        "rough_precision_limit_summary":pd.DataFrame(),
        "rough_precision_prototypes":pd.DataFrame(),
        "rough_precision_cash_gains":pd.DataFrame(),
        "rough_precision_prices":pd.DataFrame({"stage":["pooled final"],
            "estimator":["discrete martingale control"],"accepted":[True]}),
        "rough_precision_failures":pd.DataFrame()}
    messages,figures,selected=[],[],[]
    result={"manifest":{"rough_precision_executed":enabled,
        "rough_precision":{"source_elapsed_seconds":123.}},"tables":tables}
    namespace={"results":result,"tables":tables,"np":np,"pd":pd,
        "Markdown":lambda value:value,"display":messages.append,
        "present_figure":lambda result,name,**kwargs:figures.append(name),
        "present_table":lambda table,**kwargs:selected.append(table)}
    for index,cell in enumerate(cells):
        exec(compile(cell.source,f"recorded-precision-display-{index}","exec"),namespace)
    if enabled:
        assert len(figures)==2 and len(selected)==5
        assert any("1/1 pooled final contracts" in message for message in messages)
        assert any("remain **false**" in message for message in messages)
    else:
        assert not figures and not selected
        assert any("disabled" in message for message in messages)


def test_foundation_notebook_labels_preserved_outputs_without_claiming_current_execution():
    from pathlib import Path
    import numpy as np
    import pandas as pd
    nbformat=pytest.importorskip("nbformat")
    notebook=nbformat.read(Path(precision.__file__).resolve().parents[3]/
        "notebooks/legacy/pricing/first_stage_pre_report_2026-10-20/01_pricing_and_exercise.ipynb",as_version=4)
    cells=[cell for cell in notebook.cells if cell.cell_type=="code" and
        cell.metadata.get("rvlab_presentation") and "rough_foundation_" in cell.source]
    summary=pd.DataFrame({"stage":["kernel approximation","fixed-lift simulation","limit"],
        "model":["flat","flat","control"],"factors":[8,8,1],"accepted":[True,False,True]})
    tables={"rough_foundation_summary":summary,
        "rough_foundation_candidate_selection":pd.DataFrame(),
        "rough_foundation_failures":pd.DataFrame()}
    messages=[]
    result={"manifest":{"rough_foundation_executed":False,"rough_foundation_included":True,
        "rough_foundation":{"historical_replay":True,"source_elapsed_seconds":55.}},"tables":tables}
    namespace={"results":result,"tables":tables,"np":np,"pd":pd,
        "Markdown":lambda value:value,"display":messages.append,
        "present_figure":lambda *a,**k:None,"present_table":lambda *a,**k:None}
    for cell in cells:exec(compile(cell.source,"historical-foundation-display","exec"),namespace)
    assert any("Preserved foundation study" in message for message in messages)
    assert not any("disabled" in message for message in messages)


def _shared_guard_fixture(tmp_path):
    archive=tmp_path/"archive"
    original=archive/"source/src/rvlab/experiments/_pricing_rough_foundation.py"
    original.parent.mkdir(parents=True)
    source="PRICE_ATOL=.001\nPRICE_RTOL=1e-4\nCONFIDENCE_MULTIPLIER=1.96\n"+"\n".join(
        f"def {name}():\n    return 1\n" for name in precision._SHARED_HELPER_FUNCTIONS)
    original.write_text(source+"\ndef _simulate_trial():\n    return 'original historical method'\n")
    payload=original.read_bytes()
    (archive/"preservation_manifest.json").write_text(json.dumps({"files":[{
        "preserved":str(original.relative_to(archive)),"bytes":len(payload),
        "sha256":hashlib.sha256(payload).hexdigest()}]}))
    current=tmp_path/"current.py"
    current.write_text(source+"\ndef _simulate_trial():\n    return 'explicit trapezoidal comparator'\n")
    return archive,current,tmp_path/"profile"


def test_late_shared_guard_proves_archived_bodies_and_preserves_valid_trials(tmp_path):
    archive,current,folder=_shared_guard_fixture(tmp_path)
    record=precision._ensure_shared_helper_guard(folder,current_source=current,archive=archive)
    guard=json.loads((folder/"shared_helper_guard.json").read_text())
    assert not record["existing_guard"]
    assert not guard["numerical_results_recomputed"]
    assert not guard["acceptance_tolerances_changed"]
    assert "Late provenance audit" in guard["audit_basis"]
    replay=precision._ensure_shared_helper_guard(folder,current_source=current,archive=archive)
    assert replay["existing_guard"] and replay["sha256"]==record["sha256"]
    # Changes in the intentionally unused historical simulation helper do not
    # modify the referenced numerical function/constant ASTs.
    current.write_text(current.read_text().replace("explicit trapezoidal comparator","later historical commentary"))
    assert precision._ensure_shared_helper_guard(folder,current_source=current,
        archive=archive)["existing_guard"]


def test_shared_guard_refuses_changed_numerical_dependency_before_cache_reuse(tmp_path):
    archive,current,folder=_shared_guard_fixture(tmp_path)
    precision._ensure_shared_helper_guard(folder,current_source=current,archive=archive)
    current.write_text(current.read_text().replace("def price_gate():\n    return 1",
        "def price_gate():\n    return 2"))
    with pytest.raises(ValueError,match="Shared numerical helper changed"):
        precision._ensure_shared_helper_guard(folder,current_source=current,archive=archive)


def test_shared_guard_rejects_tampered_sidecar_and_source_snapshot(tmp_path):
    archive,current,folder=_shared_guard_fixture(tmp_path)
    precision._ensure_shared_helper_guard(folder,current_source=current,archive=archive)
    path=folder/"shared_helper_guard.json"
    original=path.read_bytes()
    guard=json.loads(original);guard["audit_basis"]="changed later"
    path.write_text(json.dumps(guard))
    with pytest.raises(ValueError,match="Changed shared-helper guard"):
        precision._ensure_shared_helper_guard(folder,current_source=current,archive=archive)
    path.write_bytes(original)
    guard=json.loads(original)
    from pathlib import Path
    Path(guard["source_snapshot"]).write_text("# later altered source evidence\n")
    with pytest.raises(ValueError,match="Changed shared-helper source snapshot"):
        precision._ensure_shared_helper_guard(folder,current_source=current,archive=archive)


def test_initial_shared_guard_rejects_changed_archived_baseline(tmp_path):
    archive,current,folder=_shared_guard_fixture(tmp_path)
    current.write_text(current.read_text().replace("PRICE_ATOL=.001","PRICE_ATOL=.1"))
    with pytest.raises(ValueError,match="do not match the archived"):
        precision._ensure_shared_helper_guard(folder,current_source=current,archive=archive)


def test_actual_notebook_parameter_and_run_cells_obey_workflow_argument_contract(monkeypatch):
    """The current report uses the policy-comparison facade, not history flags."""
    from pathlib import Path
    from types import SimpleNamespace
    import inspect
    import nbformat
    from rvlab.pricing import PricingConfig,FDConfig
    from rvlab.experiments import pricing as public
    from rvlab.experiments._pricing_policy_comparison_report import policy_comparison_pricing
    notebook=nbformat.read(Path(precision.__file__).resolve().parents[3]/
        "notebooks/pricing/01_pricing_and_exercise.ipynb",as_version=4)
    parameter_cell=next(cell for cell in notebook.cells if cell.cell_type=="code"
        and 'context = setup_policy_comparison(' in cell.source)
    run_cell=next(cell for cell in notebook.cells if cell.cell_type=="code"
        and 'results = run_policy_comparison_pricing(' in cell.source)
    captured=[]
    class ValidatedArguments(Exception):pass
    def validate_report(context,**kwargs):
        inspect.signature(policy_comparison_pricing).bind(context,**kwargs)
        captured.append(kwargs)
        raise ValidatedArguments
    monkeypatch.setattr(public,"run_policy_comparison_pricing",validate_report)
    namespace={"PricingConfig":PricingConfig,"FDConfig":FDConfig,
        "setup_policy_comparison":lambda *a,**k:SimpleNamespace(progress=lambda _:None),
        "presentation_style":lambda:None}
    exec(compile(parameter_cell.source,"actual-report-parameters","exec"),namespace)
    with pytest.raises(ValidatedArguments):
        exec(compile(run_cell.source,"actual-report-run","exec"),namespace)
    assert len(captured)==1 and captured[0]["seed"]==42
    assert set(captured[0])=={"seed","fourier_config","fd_config"}
    assert notebook.metadata.rvlab_artifact_family=="first_stage_v2"


def test_shared_guard_resolves_local_snapshot_after_original_checkout_is_removed(tmp_path):
    import shutil
    from pathlib import Path
    original=tmp_path/"original_checkout"
    archive,current,folder=_shared_guard_fixture(original)
    precision._ensure_shared_helper_guard(folder,current_source=current,archive=archive)
    recorded=json.loads((folder/"shared_helper_guard.json").read_text())
    original_snapshot=Path(recorded["source_snapshot"])
    copied=tmp_path/"copied_checkout"
    shutil.copytree(original,copied)
    shutil.rmtree(original)
    assert not original_snapshot.exists()
    replay=precision._ensure_shared_helper_guard(copied/"profile",
        current_source=copied/"current.py",archive=copied/"archive")
    assert replay["existing_guard"]
    local=Path(replay["resolved_source_snapshot"])
    assert local.is_relative_to(copied) and local.is_file()
    assert hashlib.sha256(local.read_bytes()).hexdigest()==recorded["source_sha256"]
    # The old absolute location survives only as historical metadata.
    preserved=json.loads((copied/"profile/shared_helper_guard.json").read_text())
    assert preserved["source_snapshot"]==str(original_snapshot)


def test_execution_evidence_uses_copied_local_runners_after_original_is_removed(tmp_path):
    import shutil
    from pathlib import Path
    original=tmp_path/"original"/"rough_precision_v1"
    (original/"raw").mkdir(parents=True)
    (original/"prototypes").mkdir()
    def runner(relative):
        path=original/relative;path.write_text("# frozen local execution runner\n")
        return path,hashlib.sha256(path.read_bytes()).hexdigest()
    campaign,campaign_sha=runner("raw/run_precision_campaign.py")
    curve,curve_sha=runner("raw/prefill_curve_2048_44.py")
    hurst,hurst_sha=runner("prototypes/hurst_checkpoint_prefill.py")
    (original/"raw/campaign_scheduling_audit.json").write_text(json.dumps({
        "runner":str(campaign),"runner_sha256":campaign_sha}))
    (original/"raw/prefill_curve_2048_44.json").write_text(json.dumps({
        "runner_path":str(curve),"runner_sha256":curve_sha}))
    (original/"prototypes/hurst_checkpoint_prefill_completion.json").write_text(json.dumps({
        "records":[{"runner_sha256":hurst_sha}]}))
    copied=tmp_path/"copied"/"rough_precision_v1"
    shutil.copytree(original,copied);shutil.rmtree(original.parent)
    records=precision._execution_evidence(copied)
    assert len(records)==3
    for record in records:
        path=Path(record["runner_path"])
        assert path.is_relative_to(copied) and path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record["runner_sha256"]
    assert any(record["historical_runner_path"]==str(curve) for record in records)


def test_prototype_loader_resolves_copied_inventory_and_source_records_locally(tmp_path):
    import shutil
    original=tmp_path/"original"/"prototypes"
    original.mkdir(parents=True)
    paths=[]
    runner=original/"trial.py";runner.write_text("# preserved prototype runner\n");paths.append(runner)
    sample={"name":"flat","n":4,"steps":256,"seed":42,"seconds":1.,
        "reference":[1.],"controlled_mean":[1.],"controlled_se":[.0001]}
    for name in ("flat_256_4096.json","flat_256_4096_fourier.json","exact_flat_1024_4096.json"):
        path=original/name;path.write_text(json.dumps(sample));paths.append(path)
    inventory={"files":[{"path":str(path),"bytes":path.stat().st_size,
        "sha256":hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]}
    (original/"variance_reduction_manifest.json").write_text(json.dumps(inventory))
    runner_record={"path":str(runner),"sha256":hashlib.sha256(runner.read_bytes()).hexdigest()}
    (original/"naive_exact_drift_trials.json").write_text(json.dumps({
        "source_records":[runner_record],"runner":runner_record,"trials":[{
            "args":{"paths":4,"steps":256,"seed":42},"elapsed":1.,
            "rows":[{"estimate":1.,"reference":1.,"SE":.0001}]}]}))
    copied=tmp_path/"copied"/"prototypes"
    shutil.copytree(original,copied);shutil.rmtree(original.parent)
    table,provenance=precision._prototype_trials(copied)
    assert len(table)==4 and table.accepted.all()
    assert len(provenance)==2
