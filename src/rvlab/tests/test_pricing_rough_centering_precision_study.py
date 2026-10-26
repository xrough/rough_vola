"""The precision campaign preserves fixed streams, failures and paired evidence."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rvlab.experiments import _pricing_rough_centering_precision as study
from rvlab.experiments._pricing_rough_stopping_attribution import _save_sample_archive
from rvlab.pricing import PricingError


def test_primary_design_declares_eight_fresh_streams_and_fixed_path_budget():
    rows=study.precision_repeat_specs()
    assert [row["seed"] for row in rows]==[510005042+1000000*i for i in range(8)]
    assert all(row["paths"]==2048 for row in rows)
    assert sum(row["paths"] for row in rows)==16384


@pytest.fixture
def campaign(tmp_path,monkeypatch):
    oldfolder=tmp_path/"previous";oldfolder.mkdir()
    reference=oldfolder/"manifest.json"
    old={"forbidden_seeds":[31,32,33,34,350005042],
        "repeats":[{"seed":410005042,"paths":8192}],
        "frozen_identity":{"proxy":"frozen_proxy","original":"original","selected":"selected","bundle":"frozen_bundle"},
        "profile_fingerprint":"previous_complete_profile",
        "frozen_reconstruction":{"price_allowance":.0012860844852291117},
        "acceptance":{"price_atol":.001,"price_rtol":1e-4,"delta_atol":.002,"delta_rtol":.005,
            "ci_multiplier":1.96,"effective_half_spread_fraction":.1}}
    reference.write_text(json.dumps(old))
    pd.DataFrame([{"quantity":"contrast","paths":65536,"standard_error":.002991159211431192,
        "estimate":-.0010733824631053665}]).to_csv(oldfolder/"pooled.csv",index=False)
    frozen={"proxy":SimpleNamespace(exercise_times=np.array([0.,.5,1.]))}
    monkeypatch.setattr(study,"_frozen",lambda output:(frozen,old,reference))
    monkeypatch.setattr(study,"_sources",lambda:{"fixture_numerical_source":"a"*64})
    calls=[]

    def calculate(frozen,spec,folder,levels):
        calls.append(spec.copy())
        n=spec["paths"];fine=np.full(n,.003)
        raw=np.linspace(-1.,1.,n);counts=np.arange(n,dtype=int)%3
        sample=_save_sample_archive(folder/f"seed_{spec['seed']}_samples.npz",
            raw_contrast=raw,projected_coarse=fine-1e-7,projected_fine=fine,
            quadrature_error=np.full(n,2e-8),tail_error=np.full(n,3e-12),arithmetic_error=np.full(n,4e-12),
            absolute_gate_counts=counts,raw_gain=raw+.01,controlled_gain=np.full(n,.01),
            projected_by_level=np.tile(fine,(3,1)))
        trace=_save_sample_archive(folder/f"seed_{spec['seed']}_microstates.npz",times=np.array([0.,.5,1.]))
        destination=folder/f"seed_{spec['seed']}_conditional_states.csv"
        pd.DataFrame({"path":[0],"first_refinement":[1e-6],"last_refinement":[1e-7],"tail_envelope":[3e-12]}).to_csv(destination,index=False)
        raw_csv=destination.read_bytes()
        return {"status":"completed",**spec,"sample_archives":[sample,trace],
            "state_table":{"path":str(destination),"bytes":len(raw_csv),"sha256":hashlib.sha256(raw_csv).hexdigest()},
            "gated_microstates":int(counts.sum()),"disagreeing_paths":int(np.count_nonzero(counts)),
            "actual_integration_steps":4,"exercise_dates":3,
            "maximum_first_refinement":1e-6,"maximum_last_refinement":1e-7,"maximum_tail_envelope":3e-12}

    monkeypatch.setattr(study,"_calculate_repeat",calculate)
    context=SimpleNamespace(output_dir=tmp_path,progress=lambda message:None)
    repeats=[{"seed":51,"paths":4},{"seed":52,"paths":6}]
    return context,repeats,calls,calculate


def _run(campaign,**kwargs):
    context,repeats,_,_=campaign
    return study.centering_precision_study(context,repeat_specs=repeats,compute_figures=False,**kwargs)


def test_fixed_campaign_pools_all_paths_and_preserves_nonzero_bias(campaign):
    result=_run(campaign)
    manifest=result["manifest"]
    assert manifest["primary_complete"]
    assert manifest["completed_streams"]==2
    assert manifest["price_allowance"]==.0012860844852291117
    assert manifest["quadrature_nodes"]==[32,64,128]
    assert manifest["policy_optimizer_calls"]==manifest["dual_optimizer_calls"]==manifest["calibration_optimizer_calls"]==0
    assert not manifest["projection_mean_assumed_zero"] and not manifest["tolerances_changed"]
    assert not manifest["american_pricing_ready"] and not manifest["stock_delta_ready"]
    pooled=result["tables"]["pooled"].set_index("quantity")
    assert pooled.loc["projected_fine","paths"]==10
    assert pooled.loc["projected_fine","estimate"]==pytest.approx(.003)
    budget=result["tables"]["numerical_budget"].query("scope=='pooled'").iloc[0]
    assert not budget.endpoint_interval_within_price_allowance
    assert len(campaign[2])==2
    seal=Path(manifest["design_seal"]["path"])
    assert seal.is_file()
    assert json.loads(seal.read_text())["repeats"]==campaign[1]


def test_negative_controls_are_paired_predictable_gate_defects(campaign):
    result=_run(campaign)
    rows=result["tables"]["negative_controls"]
    counts=np.r_[np.arange(4)%3,np.arange(6)%3]
    assert len(rows)==2
    # The adaptive defect is coefficient*g, so g times that defect sums as |g|.
    coefficient=result["manifest"]["negative_control_shift_per_gated_step"]
    assert coefficient==.01
    np.testing.assert_allclose(sorted(rows.estimate),sorted([.003-coefficient*counts.mean(),.003+coefficient*counts.mean()]),atol=1e-15,rtol=0.)
    assert all(rows.paths==10)
    assert rows.same_path_artificial_transformation.all()


def test_checkpoint_replay_does_not_repeat_simulation_or_change_seal(campaign):
    first=_run(campaign)
    seal=Path(first["manifest"]["design_seal"]["path"]);before=seal.read_bytes()
    second=_run(campaign)
    assert len(campaign[2])==2
    assert second["manifest"]["checkpoint_replay"]
    assert second["tables"]["provenance"].reused.all()
    assert seal.read_bytes()==before
    pd.testing.assert_frame_equal(first["tables"]["pooled"],second["tables"]["pooled"])


@pytest.mark.parametrize("seed",[31,32,33,34,350005042,410005042,890005042,890105042,430005042])
def test_spent_fitting_valuation_and_audit_seeds_are_rejected(campaign,seed):
    context,_,calls,_=campaign
    with pytest.raises(ValueError,match="overlap"):
        study.centering_precision_study(context,repeat_specs=[{"seed":seed,"paths":4}],compute_figures=False)
    assert not calls


@pytest.mark.parametrize("repeats",[[],{},[{"seed":True,"paths":2}],[{"seed":-1,"paths":2}],
    [{"seed":51,"paths":1}],[{"seed":51,"paths":2,"extra":1}],
    [{"seed":51,"paths":2},{"seed":51,"paths":2}]])
def test_malformed_design_rejected_before_simulation(campaign,repeats):
    context,_,calls,_=campaign
    with pytest.raises(ValueError):
        study.centering_precision_study(context,repeat_specs=repeats,compute_figures=False)
    assert not calls


@pytest.mark.parametrize("indices",[[True],[-1],[2],[0,0],"all"])
def test_invalid_repeat_indices_rejected_before_simulation(campaign,indices):
    with pytest.raises(ValueError,match="indices"):
        _run(campaign,repeat_indices=indices)
    assert not campaign[2]


def test_partial_worker_remains_explicitly_incomplete(campaign):
    result=_run(campaign,repeat_indices=[1])
    assert not result["manifest"]["primary_complete"]
    assert result["manifest"]["planned_streams"]==2 and result["manifest"]["completed_streams"]==1
    assert campaign[2]==[campaign[1][1]]
    assert result["tables"]["pooled"].paths.eq(6).all()


@pytest.mark.parametrize("artifact",["samples","state_table"])
def test_replayed_paired_evidence_is_verified(campaign,artifact):
    result=_run(campaign)
    checkpoint=Path(result["tables"]["provenance"].iloc[0].checkpoint)
    payload=json.loads(checkpoint.read_text())["payload"]
    target=Path(payload["sample_archives"][0]["path"] if artifact=="samples" else payload["state_table"]["path"])
    target.write_bytes(target.read_bytes()+b"changed")
    with pytest.raises(ValueError,match="Changed"):
        _run(campaign)


def test_pricing_failure_is_saved_and_does_not_discard_other_stream(campaign,monkeypatch):
    calculate=campaign[3]
    def failed(frozen,spec,folder,levels):
        if spec["seed"]==51:raise PricingError("declared injected numerical failure")
        return calculate(frozen,spec,folder,levels)
    monkeypatch.setattr(study,"_calculate_repeat",failed)
    result=_run(campaign)
    assert not result["manifest"]["primary_complete"]
    assert result["manifest"]["completed_streams"]==1
    assert result["tables"]["failures"].error.tolist()==["declared injected numerical failure"]
    assert set(result["tables"]["repeats"].status)=={"numerical_failure","completed"}


def test_floating_point_failure_is_preserved_as_a_numerical_attempt(campaign,monkeypatch):
    def failed(*args):raise FloatingPointError("declared quadrature overflow")
    monkeypatch.setattr(study,"_calculate_repeat",failed)
    result=_run(campaign)
    assert not result["manifest"]["primary_complete"]
    assert result["manifest"]["completed_streams"]==0
    assert len(result["tables"]["failures"])==2
    assert result["tables"]["repeats"].status.eq("numerical_failure").all()


def test_source_changes_abort_before_a_fresh_attempt(campaign,monkeypatch):
    calls=iter(({"fixture_numerical_source":"a"*64},{"fixture_numerical_source":"b"*64}))
    monkeypatch.setattr(study,"_sources",lambda:next(calls))
    with pytest.raises(RuntimeError,match="sources changed before"):
        _run(campaign)
    assert not campaign[2]


@pytest.mark.parametrize("array,match",[("raw_contrast","primary samples"),
    ("absolute_gate_counts","gate counts"),("controlled_gain","policy gains")])
def test_saved_array_lengths_must_equal_declared_path_budget(campaign,monkeypatch,array,match):
    calculate=campaign[3]
    def shortened(frozen,spec,folder,levels):
        record=calculate(frozen,spec,folder,levels)
        path=Path(record["sample_archives"][0]["path"])
        with np.load(path,allow_pickle=False) as z:values={key:z[key].copy() for key in z.files}
        values[array]=values[array][:-1]
        # Re-seal valid saved bytes/schema to isolate declared-budget checking.
        path.unlink()
        record["sample_archives"][0]=_save_sample_archive(path,**values)
        return record
    monkeypatch.setattr(study,"_calculate_repeat",shortened)
    with pytest.raises(ValueError,match=match):
        _run(campaign)
