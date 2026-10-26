"""Reconstruct prior/local/grid controls independently before trusting endpoints."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rvlab.experiments import _pricing_rough_gap_time_stopping as study


def _direct(corrections, control):
    """Literal interval sums, independent of the collector's basis tensor."""
    paths,dates,features=corrections.shape
    result=np.zeros((paths,dates))
    boundaries=(0,64) if control["bins"]==1 else (0,16,32,48,64)
    for block,(left,right) in enumerate(zip(boundaries[:-1],boundaries[1:])):
        for feature in range(features):
            coefficient=control["weights"][block*features+feature]
            for date in range(left+1,dates):
                result[:,date]+=coefficient*(corrections[:,min(date,right),feature]-corrections[:,left,feature])
    return result


def _workspace(paths=4,seed=42):
    rng=np.random.default_rng(seed)
    payoff=np.maximum(3+rng.normal(size=(paths,65)),0.)
    stable=np.cumsum(rng.normal(0,.01,(paths,65)),axis=1);stable-=stable[:,:1]
    prior=np.cumsum(rng.normal(0,.02,(paths,65,3)),axis=1);prior-=prior[:,:1]
    grid=np.cumsum(rng.normal(0,.02,(paths,65,6)),axis=1);grid-=grid[:,:1]
    stops=np.resize(np.array([0,16,33,64]),paths)
    prior_control={"bins":4,"weights":[.1,.05,.02,.05,.15,.01,.12,.04,.08,.08,.05,.2]}
    controls={"stable baseline":{"bins":1,"penalty":0.,"weights":[0.]*6},
        "grid 1 bins penalty0":{"bins":1,"penalty":0.,"weights":[.05,.03,.06,.01,.02,.01]},
        "grid 4 bins penalty0":{"bins":4,"penalty":0.,"weights":[.02,.03,.01,.04,.05,.02]*4}}
    baseline=stable+_direct(prior,prior_control)
    raw=payoff[np.arange(paths),stops].copy()
    arrays={"raw":raw,"stops":stops,"payoff":payoff,"stable_old_martingale":stable,
        "prior_local_corrections":prior,"baseline_martingale":baseline,"grid_corrections":grid}
    for i,control in enumerate(controls.values()):
        martingale=baseline+_direct(grid,control)
        arrays[f"control{i}_lower"]=raw-martingale[np.arange(paths),stops]
        arrays[f"control{i}_upper"]=np.max(payoff-martingale,axis=1)
    return arrays,controls,prior_control


@pytest.fixture
def workspace():return _workspace()


def test_all_prior_and_new_features_reconstruct_fixed_rule_endpoints(workspace):
    arrays,controls,prior=workspace
    before=deepcopy(arrays)
    values=study._grid_samples(arrays,controls,prior,4)
    assert len(values)==3
    for i,(name,(lower,upper)) in enumerate(values.items()):
        np.testing.assert_allclose(lower,arrays[f"control{i}_lower"],rtol=0.,atol=2e-14)
        np.testing.assert_allclose(upper,arrays[f"control{i}_upper"],rtol=0.,atol=2e-14)
        assert len(lower)==4 and np.all(upper>=lower-2e-14)
    for key,value in before.items():np.testing.assert_array_equal(arrays[key],value)
    np.testing.assert_allclose(values["stable baseline"][0],
        arrays["raw"]-arrays["baseline_martingale"][np.arange(4),arrays["stops"]])


@pytest.mark.parametrize("change",["missing","extra","nonfinite","raw","negative_payoff","stops_float",
    "stops_range","path_count","prior_shape","grid_shape","baseline","lower","upper","endpoint_shape",
    "prior_budget","prior_negative","new_budget","new_negative","new_bins","fallback_missing","fallback_nonzero"])
def test_changed_samples_controls_and_path_counts_are_rejected(workspace,change):
    arrays,controls,prior=workspace;paths=4
    if change=="missing":arrays.pop("control0_lower")
    elif change=="extra":arrays["undeclared"]=np.zeros(4)
    elif change=="nonfinite":arrays["grid_corrections"][0,2,0]=np.inf
    elif change=="raw":arrays["raw"][0]+=1e-7
    elif change=="negative_payoff":arrays["payoff"][0,1]=-1e-9
    elif change=="stops_float":arrays["stops"]=arrays["stops"].astype(float)
    elif change=="stops_range":arrays["stops"][0]=65
    elif change=="path_count":paths=3
    elif change=="prior_shape":arrays["prior_local_corrections"]=arrays["prior_local_corrections"][:,:,:2]
    elif change=="grid_shape":arrays["grid_corrections"]=arrays["grid_corrections"][:,:,:5]
    elif change=="baseline":arrays["baseline_martingale"][0,2]+=1e-7
    elif change=="lower":arrays["control1_lower"][0]+=1e-7
    elif change=="upper":arrays["control1_upper"][0]-=1e-7
    elif change=="endpoint_shape":arrays["control1_upper"]=arrays["control1_upper"][:3]
    elif change=="prior_budget":prior["weights"][:3]=[.5,.5,.5]
    elif change=="prior_negative":prior["weights"][0]=-.001
    elif change=="new_budget":controls["grid 1 bins penalty0"]["weights"][:]=[.2]*6
    elif change=="new_negative":controls["grid 1 bins penalty0"]["weights"][0]=-.001
    elif change=="new_bins":controls["grid 1 bins penalty0"]["bins"]=True
    elif change=="fallback_missing":controls.pop("stable baseline")
    else:controls["stable baseline"]["weights"][0]=.001
    with pytest.raises(ValueError):study._grid_samples(arrays,controls,prior,paths)


@pytest.mark.parametrize("key",["stable_old_martingale","prior_local_corrections","baseline_martingale","grid_corrections"])
def test_each_martingale_must_start_at_zero(workspace,key):
    arrays,controls,prior=workspace
    arrays[key][0,0]=1e-9
    with pytest.raises(ValueError,match="origin"):
        study._grid_samples(arrays,controls,prior,4)


def _selection(tmp_path):
    _,available,_=_workspace()
    controls={"stable baseline":available["stable baseline"]}
    trials=[]
    for bins in (1,4):
        for penalty in (0.,1e-4):
            name=f"grid {bins} bins penalty{penalty:g}"
            completed=name!="grid 1 bins penalty0.0001"
            trials.append({"control":name,"status":"completed" if completed else "optimizer_failure",
                "error":"" if completed else "declared optimizer failed"})
            if completed:controls[name]={"bins":bins,"penalty":penalty,"weights":[0.]*(bins*6)}
    rows=[]
    selected="grid 4 bins penalty0"
    for index,name in enumerate(controls):
        gap=.002 if name==selected else .003+index*.0001
        gse,pse=.0001,.0002
        rows.append({"control":name,"heldout_paths":2048,"target_final_paths":8192,
            "heldout_gap":gap,"heldout_gap_se":gse,"heldout_price":3.,"heldout_price_se":pse,
            "selection_score":gap+1.96*.5*(gse+pse),"final_valuation_used_for_selection":False,
            "selection_source":"second half of independent control-training stream only",
            "selected_before_final":name==selected})
    selection,trials=pd.DataFrame(rows),pd.DataFrame(trials)
    selection.to_csv(tmp_path/"stopping_selection.csv",index=False)
    trials.to_csv(tmp_path/"stopping_trials.csv",index=False)
    return {"controls":controls,"selected":selected},selection,trials


def test_training_only_selection_retains_failed_fits_and_fallback(tmp_path):
    seal,expected,trials=_selection(tmp_path)
    entries=[study._entry(tmp_path/name) for name in ("stopping_selection.csv","stopping_trials.csv")]
    table,attempts=study._selection_tables(tmp_path,entries,seal)
    pd.testing.assert_frame_equal(table,expected)
    assert len(attempts)==4 and "optimizer_failure" in attempts.status.values
    assert attempts.loc[attempts.status.eq("optimizer_failure"),"error"].item()=="declared optimizer failed"
    assert "stable baseline" in seal["controls"] and len(entries)==2
    assert trials.loc[trials.status.eq("optimizer_failure"),"control"].item() not in seal["controls"]


@pytest.mark.parametrize("change",["omit_attempt","duplicate_attempt","hide_failure","unknown_status","omit_fallback",
    "omit_completed","add_failed_control","wrong_bins","wrong_penalty","heldout_count","final_count",
    "use_final","wrong_source","score","selected","selected_flag","budget",
    "negative_gap","negative_gap_se","negative_price_se","nonfinite_price","fallback_penalty"])
def test_changed_fit_attempts_or_selection_inputs_are_rejected(tmp_path,change):
    seal,selection,trials=_selection(tmp_path)
    if change=="omit_attempt":trials=trials.iloc[:-1]
    elif change=="duplicate_attempt":trials.loc[1,"control"]=trials.loc[0,"control"]
    elif change=="hide_failure":trials.loc[trials.status.eq("optimizer_failure"),"status"]="completed"
    elif change=="unknown_status":trials.loc[0,"status"]="silently skipped"
    elif change=="omit_fallback":seal["controls"].pop("stable baseline")
    elif change=="omit_completed":seal["controls"].pop("grid 4 bins penalty0")
    elif change=="add_failed_control":seal["controls"]["grid 1 bins penalty0.0001"]={"bins":1,"penalty":1e-4,"weights":[0.]*6}
    elif change=="wrong_bins":seal["controls"]["grid 4 bins penalty0"]["bins"]=1
    elif change=="wrong_penalty":seal["controls"]["grid 4 bins penalty0"]["penalty"]=.01
    elif change=="heldout_count":selection.loc[0,"heldout_paths"]=1024
    elif change=="final_count":selection.loc[0,"target_final_paths"]=16384
    elif change=="use_final":selection.loc[0,"final_valuation_used_for_selection"]=True
    elif change=="wrong_source":selection.loc[0,"selection_source"]="final valuation"
    elif change=="score":selection.loc[0,"selection_score"]+=.001
    elif change=="selected":seal["selected"]="stable baseline"
    elif change=="selected_flag":selection.loc[0,"selected_before_final"]=True
    elif change=="nonfinite_price":selection.loc[0,"heldout_price"]=np.nan
    elif change=="fallback_penalty":seal["controls"]["stable baseline"]["penalty"]=.001
    elif change in ("negative_gap","negative_gap_se","negative_price_se"):
        index=selection.index[selection.control.eq(seal["selected"])][0]
        column={"negative_gap":"heldout_gap","negative_gap_se":"heldout_gap_se",
            "negative_price_se":"heldout_price_se"}[change]
        selection.loc[index,column]=-.00001
        selection["selection_score"]=selection.heldout_gap+1.96*.5*(selection.heldout_gap_se+selection.heldout_price_se)
    else:seal["controls"]["grid 4 bins penalty0"]["weights"][:6]=[.2]*6
    selection.to_csv(tmp_path/"stopping_selection.csv",index=False)
    trials.to_csv(tmp_path/"stopping_trials.csv",index=False)
    entries=[study._entry(tmp_path/name) for name in ("stopping_selection.csv","stopping_trials.csv")]
    with pytest.raises(ValueError):study._selection_tables(tmp_path,entries,seal)


def test_final_budgets_preserve_whole_path_pairing_and_original_allowances():
    chunks=[]
    for seed in (1,2):
        arrays,controls,prior=_workspace(seed=seed)
        values=study._grid_samples(arrays,controls,prior,4)
        chunks.append({"seed":seed,"values":values})
    selected="grid 4 bins penalty0"
    comparison,changes,per_seed=study._final_tables(chunks,selected,"frozen rule")
    assert len(comparison)==3 and len(changes)==2 and len(per_seed)==6
    assert not comparison.american_pricing_ready.any() and not comparison.stock_delta_ready.any()
    for row in comparison.itertuples():
        lows=[chunk["values"][row.control][0] for chunk in chunks]
        gaps=[chunk["values"][row.control][1]-chunk["values"][row.control][0] for chunk in chunks]
        price=np.concatenate(lows).mean();gap=np.concatenate(gaps).mean()
        # Independent fixed-size streams retain their own within-stream variance.
        pse=np.sqrt(sum(len(a)*a.var(ddof=1) for a in lows))/sum(map(len,lows))
        gse=np.sqrt(sum(len(a)*a.var(ddof=1) for a in gaps))/sum(map(len,gaps))
        assert row.price==pytest.approx(price) and row.gap==pytest.approx(gap)
        assert row.price_se==pytest.approx(pse) and row.gap_se==pytest.approx(gse)
        assert row.price_allowance==pytest.approx(.001+.0001*abs(price))
        assert row.price_error_budget==pytest.approx(gap+1.96*(pse+gse))
        assert row.price_budget_pass==(row.price_error_budget<=row.price_allowance)
    assert changes.all_paths_retained.all()
    for row in changes.itertuples():
        differences=[(chunk["values"][row.control][1]-chunk["values"][row.control][0])-
            (chunk["values"]["stable baseline"][1]-chunk["values"]["stable baseline"][0]) for chunk in chunks]
        mean=np.concatenate(differences).mean()
        se=np.sqrt(sum(len(a)*a.var(ddof=1) for a in differences))/sum(map(len,differences))
        assert row.gap_change==pytest.approx(mean) and row.gap_change_se==pytest.approx(se)
        assert row.gap_change_low==pytest.approx(mean-1.96*se)
        assert row.gap_change_high==pytest.approx(mean+1.96*se)


def _inventory(folder):
    import hashlib,json
    rows=[]
    for path in sorted(folder.rglob('*')):
        if path.is_file() and path.name!='preservation_inventory.json':
            data=path.read_bytes()
            rows.append({"path":str(path.relative_to(folder)),"bytes":len(data),
                "sha256":hashlib.sha256(data).hexdigest()})
    record={"files":rows,"inventory_sha256":hashlib.sha256(
        json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
    (folder/'preservation_inventory.json').write_text(json.dumps(record))
    return record


def test_preserved_computational_trial_pins_all_nested_bytes_with_canonical_paths(tmp_path):
    folder=tmp_path/'trial';(folder/'source').mkdir(parents=True)
    (folder/'source/method.py').write_text('frozen_method = True\n')
    _inventory(folder)
    record,entries=study._preserved(folder)
    assert len(record['files'])==1 and len(entries)==2
    assert all(entry['path']==str(Path(entry['path']).resolve()) for entry in entries)


@pytest.mark.parametrize('change',['artifact','inventory','duplicate','alias','escape'])
def test_changed_or_escaping_computational_trial_is_rejected(tmp_path,change):
    import hashlib,json
    folder=tmp_path/'trial';(folder/'source').mkdir(parents=True)
    data=folder/'source/method.py';data.write_text('frozen_method = True\n')
    record=_inventory(folder)
    if change=='artifact':data.write_text('changed_method = True\n')
    else:
        if change=='inventory':record['inventory_sha256']='0'*64
        elif change=='duplicate':record['files'].append(dict(record['files'][0]))
        elif change=='alias':record['files'].append({**record['files'][0],'path':'source/../source/method.py'})
        else:
            outside=tmp_path/'outside.py';outside.write_bytes(data.read_bytes())
            record['files'][0]['path']='../outside.py'
        if change!='inventory':record['inventory_sha256']=hashlib.sha256(
            json.dumps(record['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        (folder/'preservation_inventory.json').write_text(json.dumps(record))
    with pytest.raises(ValueError):study._preserved(folder)


@pytest.mark.parametrize('filename',['stopping_selection.csv','stopping_trials.csv'])
def test_selection_tables_reject_modified_sealed_csv_bytes(tmp_path,filename):
    seal,_,_=_selection(tmp_path)
    entries=[study._entry(tmp_path/name) for name in ('stopping_selection.csv','stopping_trials.csv')]
    path=tmp_path/filename;path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='Changed stopping/delta evidence'):
        study._selection_tables(tmp_path,entries,seal)


def test_gap_tail_diagnostic_retains_every_path_and_the_prevaluation_selection():
    gap=np.r_[np.zeros(90),np.arange(1.,11.)]
    streams=[]
    for seed,part in enumerate((gap[:50],gap[50:])):
        streams.append({"seed":seed,"values":{
            "stable baseline":(np.zeros(50),np.zeros(50)),
            "selected before finals":(np.zeros(50),2*part),
            "smaller final mean":(np.zeros(50),part)}})
    before=deepcopy(streams)
    table=study._gap_distribution(streams,"selected before finals").set_index("control")
    row=table.loc["selected before finals"]
    assert row.paths==100 and row.selected_before_final
    assert not table.loc["smaller final mean","selected_before_final"]
    assert row.zero_gap_fraction_at_1e_12==pytest.approx(.9)
    assert row.gap_median==0. and row.gap_p90==pytest.approx(.2)
    assert row.gap_p99==pytest.approx(18.02) and row.gap_maximum==20.
    assert row.largest_decile_gap_share==1.
    assert row.largest_percentile_gap_share==pytest.approx(20/110)
    assert table.loc["stable baseline","largest_percentile_gap_share"]==0.
    assert not table.acceptance_input.any() and not table.used_for_selection.any()
    assert table.all_paths_retained.all()
    for old,new in zip(before,streams):
        for name in old["values"]:
            for a,b in zip(old["values"][name],new["values"][name]):np.testing.assert_array_equal(a,b)
