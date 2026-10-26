"""Collection preserves whole-path results and pins every byte actually read."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import rvlab.experiments._pricing_rough_stopping_delta as collection
from rvlab.experiments._pricing_rough_stopping_delta_validation import _time_deltas


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _entry(path):
    return {"path":str(path),"bytes":Path(path).stat().st_size,"sha256":_sha(path)}


def _write_json(path,payload):
    Path(path).write_text(json.dumps(payload,sort_keys=True,indent=2))


def _stats(values):
    values=np.asarray(values,float);mean=float(np.mean(values))
    error=float(np.sqrt(np.sum((values-mean)**2)/(len(values)-1)/len(values)))
    return {"mean":mean,"standard_error":error,"ci_half_width":1.96*error,
        "interval_low":mean-1.96*error,"interval_high":mean+1.96*error}


def _tables(arrays,*,seed=None,paths,steps=128):
    estimates=[]; comparisons=[]
    for policy in ("original_policy","selected_policy"):
        for estimator in ("raw","stopped martingale"):
            for level in (.01,.005,.0025):
                values=arrays[policy+"__"+f"{estimator}:{level:g}"]
                row={"policy":policy,"estimator":estimator,"bump_relative":level,
                    "paths":paths,**_stats(values),"optimal_delta_ready":False}
                row["delta_allowance"]=.002+.005*abs(row["mean"])
                row["sampling_passed"]=row["ci_half_width"]<=row["delta_allowance"]
                if seed is not None:row.update(seed=seed,steps=steps)
                estimates.append(row)
            for coarse,fine in ((.01,.005),(.005,.0025)):
                values=arrays[policy+"__"+f"{estimator}:{fine:g}"]-arrays[policy+"__"+f"{estimator}:{coarse:g}"]
                comparisons.append({"comparison":"bump refinement","policy":policy,"estimator":estimator,
                    "coarse_bump":coarse,"fine_bump":fine,**_stats(values)})
    for estimator in ("raw","stopped martingale"):
        for level in (.01,.005,.0025):
            values=arrays["selected_policy__"+f"{estimator}:{level:g}"]-arrays["original_policy__"+f"{estimator}:{level:g}"]
            comparisons.append({"comparison":"selected minus original","policy":"paired policies",
                "estimator":estimator,"coarse_bump":np.nan,"fine_bump":level,**_stats(values)})
    return pd.DataFrame(estimates),pd.DataFrame(comparisons)


@pytest.fixture()
def fixture(tmp_path,monkeypatch):
    project=tmp_path/"project"
    fake_module=project/"src/rvlab/experiments/_pricing_rough_stopping_delta.py"
    fake_module.parent.mkdir(parents=True);fake_module.write_text("# fixture collector identity\n")
    monkeypatch.setattr(collection,"__file__",str(fake_module))
    source=project/"numerical_source.py";source.write_text("PARAMETER = .04\n")
    folder=project/"output/rough_stock_delta_v1/frozen_joint_2x4096"
    folder.mkdir(parents=True)
    bundle=folder.parents[1]/"rough_martingale_centering_v1/frozen_joint_policy"
    bundle.mkdir(parents=True)
    np.savez_compressed(bundle/"coefficients.npz",coefficient=np.arange(3.))
    _write_json(bundle/"manifest.json",{"market":{"spot":100.},
        "coefficient_archive_sha256":_sha(bundle/"coefficients.npz")})
    design={"seeds":[710005042,711005042],"paths_per_stream":4096,
        "bump_relatives":[.01,.005,.0025],"steps":128,"exercise_steps":64,
        "price_atol":.001,"price_rtol":.0001,"delta_atol":.002,"delta_rtol":.005,
        "confidence_multiplier":1.96,"spread_fraction":.1,
        "source_sha256":{"numerical_source.py":_sha(source)},
        "frozen_bundle_manifest_sha256":_sha(bundle/"manifest.json"),
        "frozen_bundle_arrays_sha256":_sha(bundle/"coefficients.npz")}
    design_path=folder/"design_before_primary_valuation.json";_write_json(design_path,design)
    records=[];all_arrays=[];relative=np.array([-.01,-.005,-.0025,0.,.0025,.005,.01])
    for index,seed in enumerate(design["seeds"]):
        rng=np.random.default_rng(seed);arrays={"relative_spots":relative}
        for p,policy in enumerate(("original_policy","selected_policy")):
            raw_delta=-.4-.01*p+rng.normal(0,.03,4096)
            # A deliberately biased-looking finite-bump mean must remain in
            # the evidence; collection does not filter by a desired answer.
            controlled_delta=-.35-.01*p+rng.normal(0,.005,4096)
            for name,delta in (("raw_prices",raw_delta),("controlled_prices",controlled_delta)):
                arrays[policy+"__"+name]=10+delta[:,None]*(100*relative)[None,:]+.03*(100*relative)[None,:]**2
            arrays[policy+"__stops"]=np.full((4096,7),64,dtype=np.int64)
            for estimator,key in (("raw","raw_prices"),("stopped martingale","controlled_prices")):
                values=arrays[policy+"__"+key]
                for level in (.01,.005,.0025):
                    lo=np.flatnonzero(relative==-level)[0];hi=np.flatnonzero(relative==level)[0]
                    arrays[policy+"__"+f"{estimator}:{level:g}"]=(values[:,hi]-values[:,lo])/(200*level)
        estimates,comparisons=_tables(arrays,seed=seed,paths=4096)
        estimates.to_csv(folder/f"stream_{index}_estimates.csv",index=False)
        comparisons.to_csv(folder/f"stream_{index}_comparisons.csv",index=False)
        np.savez_compressed(folder/f"stream_{index}_samples.npz",**arrays)
        record={"index":index,"seed":seed,"paths":4096,"design_sha256":_sha(design_path),
            "source_sha256":design["source_sha256"],"files":[_entry(folder/f"stream_{index}_{suffix}")
                for suffix in ("estimates.csv","comparisons.csv","samples.npz")]}
        _write_json(folder/f"stream_{index}.json",record);records.append(record);all_arrays.append(arrays)
    joined={key:np.concatenate([a[key] for a in all_arrays]) for key in all_arrays[0] if key!="relative_spots"}
    estimates,comparisons=_tables(joined,paths=8192)
    estimates.to_csv(folder/"pooled_estimates.csv",index=False)
    comparisons.to_csv(folder/"pooled_comparisons.csv",index=False)
    manifest={"version":"rough_stock_delta_v1","complete":True,"paths":8192,"seed_count":2,
        "design_sha256":_sha(design_path),"records":records,"independent_paths":True,
        "all_streams_retained":True,"tolerances_changed":False,"optimal_delta_ready":False}
    _write_json(folder/"manifest.json",manifest)
    return {"folder":folder,"source":source,"bundle":bundle,"arrays":all_arrays,"records":records}


@pytest.fixture()
def time_fixture(fixture):
    folder=fixture["folder"].parent/"frozen_joint_time_refinement";folder.mkdir()
    arrays={key:(values if key=="relative_spots" else values[:2048])
        for key,values in fixture["arrays"][0].items()}
    estimates,comparisons=_tables(arrays,seed=712005042,paths=2048,steps=256)
    estimates.to_csv(folder/"estimates.csv",index=False)
    comparisons.to_csv(folder/"comparisons.csv",index=False)
    np.savez_compressed(folder/"samples.npz",**arrays)
    design={"seed":712005042,"paths":2048,"steps":256,"exercise_steps":64,
        "bump_relatives":[.01,.005,.0025],"source_sha256":{"numerical_source.py":_sha(fixture["source"])},
        "frozen_bundle_manifest_sha256":_sha(fixture["bundle"]/"manifest.json"),
        "price_atol":.001,"price_rtol":.0001,"delta_atol":.002,"delta_rtol":.005,
        "confidence_multiplier":1.96,"spread_fraction":.1}
    design_path=folder/"design_before_valuation.json";_write_json(design_path,design)
    manifest={"seed":712005042,"paths":2048,"steps":256,"design_sha256":_sha(design_path),
        "source_sha256":design["source_sha256"],"sources_unchanged":True,
        "independent_training_valuation":True,"valuation_used_for_fitting":False,
        "fixed_parameters_factors_discount_dividends":True,
        "decisions_reapplied_at_each_bump":True,"stopping_times_held_fixed":False,
        "same_common_martingale_for_policies":True,"optimal_delta_ready":False,"tolerances_changed":False,
        "files":[_entry(folder/name) for name in ("estimates.csv","comparisons.csv","samples.npz")]}
    _write_json(folder/"manifest.json",manifest)
    return {"folder":folder,"arrays":arrays,"primary":fixture}


def _reseal_time(fixture):
    folder=fixture["folder"];path=folder/"manifest.json";manifest=json.loads(path.read_text())
    manifest["files"]=[_entry(folder/name) for name in ("estimates.csv","comparisons.csv","samples.npz")]
    _write_json(path,manifest)


def _reseal_stream(fixture,index=0):
    folder=fixture["folder"];path=folder/f"stream_{index}.json"
    record=json.loads(path.read_text())
    record["files"]=[_entry(folder/f"stream_{index}_{suffix}")
        for suffix in ("estimates.csv","comparisons.csv","samples.npz")]
    _write_json(path,record)
    manifest_path=folder/"manifest.json";manifest=json.loads(manifest_path.read_text())
    manifest["records"][index]=record;_write_json(manifest_path,manifest)


def test_complete_fixture_passes_and_preserves_uncomfortable_means(fixture):
    pooled,paired,streams,entries,manifest=collection._primary_deltas(fixture["folder"])
    assert len(pooled)==12 and len(paired)==14 and len(streams)==24
    assert manifest["complete"]
    retained=pooled.query("policy=='selected_policy' and estimator=='stopped martingale'")
    assert np.all(retained["mean"]>-.37)
    assert not retained.optimal_delta_ready.any()
    assert {Path(row["path"]).name for row in entries}>={"stream_0_samples.npz","stream_1_samples.npz",
        "pooled_estimates.csv","pooled_comparisons.csv","design_before_primary_valuation.json"}


def test_changed_sample_archive_rejected_without_resealing(fixture):
    path=fixture["folder"]/"stream_0_samples.npz";path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(ValueError,match="Changed stopping/delta evidence"):
        collection._primary_deltas(fixture["folder"])


def test_changed_current_source_rejected(fixture):
    fixture["source"].write_text("PARAMETER = .05\n")
    with pytest.raises(ValueError,match="Changed current numerical source"):
        collection._primary_deltas(fixture["folder"])


@pytest.mark.parametrize("field,value",[("complete",False),("all_streams_retained",False),
    ("paths",4096),("seed_count",1),("independent_paths",False),("design_sha256","0"*64)])
def test_incomplete_or_contradictory_primary_manifest_rejected(fixture,field,value):
    path=fixture["folder"]/"manifest.json";manifest=json.loads(path.read_text());manifest[field]=value
    _write_json(path,manifest)
    with pytest.raises(ValueError):collection._primary_deltas(fixture["folder"])


def test_incomplete_path_count_rejected_even_after_artifact_resealing(fixture):
    arrays={key:(values if key=="relative_spots" else values[:-1]) for key,values in fixture["arrays"][0].items()}
    np.savez_compressed(fixture["folder"]/"stream_0_samples.npz",**arrays);_reseal_stream(fixture)
    with pytest.raises(ValueError,match="paths|contracts"):
        collection._primary_deltas(fixture["folder"])


def test_omitted_declared_bump_rejected(fixture):
    arrays=dict(fixture["arrays"][0]);arrays["relative_spots"]=arrays["relative_spots"][:-1]
    np.savez_compressed(fixture["folder"]/"stream_0_samples.npz",**arrays);_reseal_stream(fixture)
    with pytest.raises(ValueError,match="stock bumps"):
        collection._primary_deltas(fixture["folder"])


def test_saved_delta_must_equal_reapplied_paired_prices(fixture):
    arrays=dict(fixture["arrays"][0]);arrays["selected_policy__stopped martingale:0.0025"]=np.zeros(4096)
    np.savez_compressed(fixture["folder"]/"stream_0_samples.npz",**arrays);_reseal_stream(fixture)
    with pytest.raises(ValueError,match="paired stock-price differences"):
        collection._primary_deltas(fixture["folder"])


@pytest.mark.parametrize("table",["pooled_estimates.csv","pooled_comparisons.csv"])
def test_changed_pooled_csv_rejected_by_independent_sample_recomputation(fixture,table):
    path=fixture["folder"]/table;frame=pd.read_csv(path)
    frame.loc[0,["mean","interval_low","interval_high"]]+=.01
    if "delta_allowance" in frame:
        frame.loc[0,"delta_allowance"]=.002+.005*abs(frame.loc[0,"mean"])
    frame.to_csv(path,index=False)
    with pytest.raises(ValueError,match="Changed pooled|Changed paired"):
        collection._primary_deltas(fixture["folder"])


def test_changed_stream_csv_statistics_rejected_even_after_resealing(fixture):
    path=fixture["folder"]/"stream_0_estimates.csv";frame=pd.read_csv(path)
    frame.loc[0,["mean","interval_low","interval_high"]]+=.1
    frame.loc[0,"delta_allowance"]=.002+.005*abs(frame.loc[0,"mean"])
    frame.to_csv(path,index=False);_reseal_stream(fixture)
    with pytest.raises(ValueError,match="stream|statistics|statistics|Stream"):
        collection._primary_deltas(fixture["folder"])


def test_duplicate_paired_row_cannot_replace_missing_declared_comparison(fixture):
    path=fixture["folder"]/"pooled_comparisons.csv";frame=pd.read_csv(path)
    frame.iloc[-1]=frame.iloc[-2];frame.to_csv(path,index=False)
    with pytest.raises(ValueError,match="omit|duplicate|comparison"):
        collection._primary_deltas(fixture["folder"])


@pytest.mark.parametrize("field,value",[("delta_allowance",.02),("sampling_passed",False),
    ("optimal_delta_ready",True),("paths",4096)])
def test_pooled_acceptance_fields_must_follow_unchanged_tolerances(fixture,field,value):
    path=fixture["folder"]/"pooled_estimates.csv";frame=pd.read_csv(path)
    frame.loc[0,field]=value;frame.to_csv(path,index=False)
    with pytest.raises(ValueError,match="tolerance|acceptance|metadata|allowance|sampling|ready"):
        collection._primary_deltas(fixture["folder"])


@pytest.mark.parametrize("field,value",[("delta_allowance",.02),("sampling_passed",False),
    ("optimal_delta_ready",True)])
def test_stream_acceptance_fields_must_follow_unchanged_tolerances(fixture,field,value):
    path=fixture["folder"]/"stream_0_estimates.csv";frame=pd.read_csv(path)
    frame.loc[0,field]=value;frame.to_csv(path,index=False);_reseal_stream(fixture)
    with pytest.raises(ValueError,match="tolerance|acceptance|metadata|allowance|sampling|ready"):
        collection._primary_deltas(fixture["folder"])


@pytest.mark.parametrize("field,value",[("price_atol",.01),("price_rtol",.001),
    ("delta_atol",.02),("delta_rtol",.05),("confidence_multiplier",1.64),("spread_fraction",.2)])
def test_resealed_design_cannot_relax_promised_acceptance_tolerances(fixture,field,value):
    folder=fixture["folder"];path=folder/"design_before_primary_valuation.json"
    design=json.loads(path.read_text());design[field]=value;_write_json(path,design)
    manifest_path=folder/"manifest.json";manifest=json.loads(manifest_path.read_text())
    manifest["design_sha256"]=_sha(path)
    for index in (0,1):
        record_path=folder/f"stream_{index}.json";record=json.loads(record_path.read_text())
        record["design_sha256"]=_sha(path);_write_json(record_path,record);manifest["records"][index]=record
    _write_json(manifest_path,manifest)
    with pytest.raises(ValueError,match="tolerance|design|Design"):
        collection._primary_deltas(folder)


def test_uninventoried_samples_cannot_enter_fingerprint(fixture):
    folder=fixture["folder"];path=folder/"stream_0.json";record=json.loads(path.read_text())
    record["files"]=[row for row in record["files"] if not row["path"].endswith("samples.npz")]
    _write_json(path,record)
    with pytest.raises(ValueError,match="inventory|files|samples|Inventory"):
        collection._primary_deltas(folder)


def test_external_valid_archive_cannot_mask_changed_local_archive(fixture,tmp_path):
    folder=fixture["folder"];external=tmp_path/"external";external.mkdir()
    old=folder/"stream_0_samples.npz";copy=external/old.name;copy.write_bytes(old.read_bytes())
    path=folder/"stream_0.json";record=json.loads(path.read_text())
    for row in record["files"]:
        if row["path"].endswith("samples.npz"):row["path"]=str(copy)
    _write_json(path,record)
    # A valid external copy cannot mask changed bytes at the local file that
    # the collector actually reads. Relocation itself is harmless when the
    # canonical local file still matches the declared digest.
    old.write_bytes(old.read_bytes()+b"changed local bytes")
    with pytest.raises(ValueError,match="Changed stopping/delta evidence"):
        collection._primary_deltas(folder)


@pytest.mark.parametrize("values",[[1.],[1.,np.nan],[[1.,2.],[3.,4.]],[]])
def test_stat_rejects_missing_or_nonfinite_whole_paths(values):
    with pytest.raises(ValueError,match="whole-path"):
        collection._stat(values)


def test_stat_uses_whole_paths_and_retains_nonzero_bias():
    values=np.array([.1,.2,.4,.9])
    result=collection._stat(values)
    assert result["mean"]==pytest.approx(.4)
    assert result["standard_error"]==pytest.approx(np.std(values,ddof=1)/2)
    assert result["interval_low"]>0


def test_manifest_missing_experiment_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError,match="Complete the independent experiment"):
        collection._manifest(tmp_path)


def test_manifest_rejects_changed_current_numerical_source(fixture,tmp_path):
    folder=tmp_path/"manifest";folder.mkdir();data=folder/"data.csv";data.write_text("value\n1\n")
    _write_json(folder/"manifest.json",{"files":[_entry(data)],"source_sha256":{"numerical_source.py":"0"*64}})
    with pytest.raises(ValueError,match="Changed current numerical source"):
        collection._manifest(folder)


def test_manifest_pins_design_bytes_into_input_profile(fixture,tmp_path):
    folder=tmp_path/"manifest";folder.mkdir();data=folder/"data.csv";data.write_text("value\n1\n")
    design=folder/"design_before_valuation.json";design.write_text('{"seed":712005042}\n')
    _write_json(folder/"manifest.json",{"files":[_entry(data)],"design_sha256":_sha(design)})
    _,entries=collection._manifest(folder)
    assert str(design) in {row["path"] for row in entries}


def test_manifest_changed_design_rejected(fixture,tmp_path):
    folder=tmp_path/"manifest";folder.mkdir();data=folder/"data.csv";data.write_text("value\n1\n")
    design=folder/"design_before_valuation.json";design.write_text('{"seed":712005042}\n')
    _write_json(folder/"manifest.json",{"files":[_entry(data)],"design_sha256":_sha(design)})
    design.write_text('{"seed":712005043}\n')
    with pytest.raises(ValueError,match="design|Design"):
        collection._manifest(folder)


def test_preservation_inventory_cannot_escape_experiment_folder(tmp_path):
    outside=tmp_path/"outside.csv";outside.write_text("1\n")
    folder=tmp_path/"experiment";folder.mkdir();_write_json(folder/"manifest.json",{})
    row={**_entry(outside),"path":"../outside.csv"};rows=[row]
    digest=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    _write_json(folder/"preservation_inventory.json",{"files":rows,"inventory_sha256":digest})
    with pytest.raises(ValueError,match="escapes"):
        collection._manifest(folder,inventory=True)


def test_time_validator_recomputes_complete_independent_paths_without_filtering(time_fixture):
    frame,entries,manifest=_time_deltas(time_fixture["folder"])
    assert len(frame)==12 and manifest["seed"]==712005042
    assert np.all(frame.loc[frame.estimator=="stopped martingale","mean"]>-.37)
    assert not frame.optimal_delta_ready.any()
    assert {Path(row["path"]).name for row in entries}>={"samples.npz","estimates.csv",
        "design_before_valuation.json","coefficients.npz"}


@pytest.mark.parametrize("field,value",[("seed",712005043),("paths",1024),("steps",128),
    ("independent_training_valuation",False),("valuation_used_for_fitting",True),
    ("stopping_times_held_fixed",True),("optimal_delta_ready",True)])
def test_time_validator_rejects_changed_design_or_provenance(time_fixture,field,value):
    path=time_fixture["folder"]/"manifest.json";manifest=json.loads(path.read_text())
    manifest[field]=value;_write_json(path,manifest)
    with pytest.raises(ValueError,match="design|provenance"):
        _time_deltas(time_fixture["folder"])


@pytest.mark.parametrize("change",["omitted row","duplicate row","changed mean","changed gate"])
def test_time_validator_rejects_incomplete_or_changed_csv_after_resealing(time_fixture,change):
    path=time_fixture["folder"]/"estimates.csv";frame=pd.read_csv(path)
    if change=="omitted row":frame=frame.iloc[:-1]
    elif change=="duplicate row":frame.iloc[-1]=frame.iloc[-2]
    elif change=="changed mean":frame.loc[0,"mean"]+=.01
    else:frame.loc[0,"delta_allowance"]*=10
    frame.to_csv(path,index=False);_reseal_time(time_fixture)
    with pytest.raises(ValueError,match="omit|duplicate|statistics|allowance"):
        _time_deltas(time_fixture["folder"])


@pytest.mark.parametrize("change",["omitted path","omitted bump","changed derivative","fractional stop"])
def test_time_validator_rejects_broken_whole_path_protocol_after_resealing(time_fixture,change):
    arrays=dict(time_fixture["arrays"])
    if change=="omitted path":
        arrays={key:(value if key=="relative_spots" else value[:-1]) for key,value in arrays.items()}
    elif change=="omitted bump":arrays["relative_spots"]=arrays["relative_spots"][:-1]
    elif change=="changed derivative":arrays["selected_policy__stopped martingale:0.0025"]=np.zeros(2048)
    else:arrays["selected_policy__stops"]=arrays["selected_policy__stops"].astype(float)+.5
    np.savez_compressed(time_fixture["folder"]/"samples.npz",**arrays);_reseal_time(time_fixture)
    with pytest.raises(ValueError,match="paths|bumps|paired stock-price|stop indices"):
        _time_deltas(time_fixture["folder"])


def test_time_validator_rejects_changed_frozen_coefficients(time_fixture):
    path=time_fixture["primary"]["bundle"]/"coefficients.npz"
    path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(ValueError,match="coefficients changed"):
        _time_deltas(time_fixture["folder"])


def test_time_validator_requires_its_sample_archive_in_input_inventory(time_fixture):
    path=time_fixture["folder"]/"manifest.json";manifest=json.loads(path.read_text())
    manifest["files"]=[row for row in manifest["files"] if not row["path"].endswith("samples.npz")]
    _write_json(path,manifest)
    with pytest.raises(ValueError,match="required artifacts"):
        _time_deltas(time_fixture["folder"])


def test_time_validator_cannot_drop_current_numerical_source_checks(time_fixture):
    path=time_fixture["folder"]/"manifest.json";manifest=json.loads(path.read_text())
    del manifest["source_sha256"];_write_json(path,manifest)
    time_fixture["primary"]["source"].write_text("PARAMETER = .05\n")
    with pytest.raises(ValueError,match="numerical source"):
        _time_deltas(time_fixture["folder"])
