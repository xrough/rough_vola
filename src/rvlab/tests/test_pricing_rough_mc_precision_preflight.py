"""Preflight proofs must survive archive/manifest rehashing by an adversary."""
from pathlib import Path
import json
import shutil

import numpy as np
import pytest

import rvlab.experiments._pricing_rough_mc_precision_collection as audit


INDICES=np.r_[np.arange(251),511,512,1023,4095]
SOURCES={'src/rvlab/experiments/fake_frozen.py':'frozen-source'}
TIMES=('2026-10-20T00:00:00+00:00','2026-10-20T00:00:01+00:00',
       '2026-10-20T00:00:02+00:00','2026-10-20T00:00:03+00:00','2026-10-20T00:00:04+00:00')


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def load(path):
    with np.load(path,allow_pickle=False) as saved:return {k:saved[k].copy() for k in saved.files}


@pytest.fixture(scope='module')
def template(tmp_path_factory):
    root=tmp_path_factory.mktemp('precision_preflight_template');output=root/'output'
    folder=output/'rough_mc_precision_v1/spent_equivalence_v3';folder.mkdir(parents=True)
    old=output/'rough_bellman_validation_v1/bellman_control_v1';old.mkdir(parents=True)
    fit=output/'rough_price_budget_v1/spent_shared_fit_v1';fit.mkdir(parents=True)
    anchor=output/'rough_price_budget_v1/spent_terminal_anchor_v1';anchor.mkdir(parents=True)
    n=4096;inputs={};records=[];checks=[]
    for i,seed in enumerate((1094005042,1095005042)):
        original={'times':np.linspace(0.,1.,65),'discounts':np.ones(65),
            'eligible':np.ones(65,dtype=bool),'actual_time_grid':np.linspace(0.,1.,129),
            'stocks':np.zeros((n,65,1)),'factors':np.zeros((n,65,4)),
            'european_proxy':np.zeros((n,65,1)),'payoff':np.zeros((n,65)),
            'prior_local_corrections':np.zeros((n,65,3)),'grid_corrections':np.zeros((n,65,6)),
            'tail_corrections':np.zeros((n,65,7)),'tail_gates':np.zeros((n,64,7)),
            'tail_margin':np.zeros((n,65)),'target_martingales':np.zeros((n,65,4)),
            'target_expected_european':np.zeros((n,65)),
            'target_actual_premiums':np.zeros((n,65,4)),'target_expected_premiums':np.zeros((n,65,4)),
            'gh16_stops':np.full(n,64,dtype=np.int64),'gh16_raw':np.zeros(n),
            'gh16_continuation':np.full((n,65),np.nan),'gh16_computed_mask':np.zeros((n,65),dtype=bool),
            'candidate3_lower':np.zeros(n),'candidate3_upper':np.zeros(n)}
        original.update({k:np.zeros((n,65)) for k in ('european_martingale','premium_martingale',
            'intrinsic_martingale','micro_boundary_martingale','stable_old_martingale',
            'prior_martingale','current_martingale','latest_martingale')})
        selected={k:v if k in ('times','discounts','eligible','actual_time_grid') else v[INDICES]
                  for k,v in original.items()}
        original_path=old/f'final_{i}_samples.npz';np.savez_compressed(original_path,**original)
        referencefit={'gated_anchor':np.zeros((n,193)),'lower':np.zeros(n),'upper':np.zeros(n)}
        fit_path=fit/f'spent_{i}_basis.npz';np.savez_compressed(fit_path,**referencefit)
        selected.update(basis=referencefit['gated_anchor'][INDICES],unchanged_lower=np.zeros(len(INDICES)),
            unchanged_upper=np.zeros(len(INDICES)),shared_lower=np.zeros(len(INDICES)),
            shared_upper=np.zeros(len(INDICES)),shift=np.zeros(len(INDICES)),
            local_actual=np.zeros((len(INDICES),65)),local_expected=np.zeros((len(INDICES),65)),
            grid_actual=np.zeros((len(INDICES),65,2)),grid_expected=np.zeros((len(INDICES),65,2)))
        selected_path=folder/f'stream_{i}_selected_samples.npz';np.savez_compressed(selected_path,**selected)
        conditional={'absolute_path_indices':np.arange(n),'physical_time_grid':np.linspace(0.,1.,129),
            'conditional_european_payoff':np.zeros(n),'conditional_european_martingale':np.zeros(n),
            'conditional_european_estimate':np.zeros(n),'shared_control':np.zeros(n),
            'replay_physical_factors':np.zeros((n,129,4))}
        conditional.update({k:np.zeros((n,129)) for k in ('conditional_log_means','conditional_variances',
            'conditional_expected_next','conditional_actual_next','replay_physical_stocks')})
        anchor_path=anchor/f'stream_{i}_samples.npz';np.savez_compressed(anchor_path,**conditional)
        subset={k:v if k=='physical_time_grid' else v[INDICES] for k,v in conditional.items()}
        conditional_path=folder/f'stream_{i}_conditional_samples.npz';np.savez_compressed(conditional_path,**subset)
        for path in (original_path,fit_path,anchor_path):inputs[str(path.relative_to(output))]=audit.sha(path)
        records.append({'seed':seed,'paths':4096,'evaluated':len(INDICES),
            'selected_sha256':audit.sha(selected_path),'conditional_sha256':audit.sha(conditional_path),
            'started_at':TIMES[1],'completed_at':TIMES[2],'elapsed_seconds':1.,'cpu_seconds':.5})
        for k in selected:
            if k in original:
                checks.append({'stream':i,'field':k,'max_abs_error':0.,'same_nan_mask':True,'passed':True})
        for k in ('basis','unchanged_lower','unchanged_upper'):
            checks.append({'stream':i,'field':k,'max_abs_error':0.,'same_nan_mask':True,'passed':True})
        for k in subset:
            if k not in ('absolute_path_indices','physical_time_grid'):
                checks.append({'stream':i,'field':k,'max_abs_error':0.,'same_nan_mask':True,'passed':True})
    design={'version':'rough_mc_precision_spent_equivalence_v1','created_at':TIMES[0],
        'source_sha256':SOURCES,'input_sha256':inputs,'seeds':[1094005042,1095005042],
        'full_stream_paths':4096,'fresh_paths':0,'equivalence_atol':2e-13,'indices':INDICES.tolist(),
        'physical_random_batch_size':512,'work_batch_size':64,'european65_and129_kept_separate':True,
        'all_65_dates_and_129_physical_dates':True,'role':'spent implementation-equivalence and runtime only'}
    write(folder/'design_before_evaluation.json',design);write(folder/'equivalence_checks.json',checks)
    write(folder/'manifest.json',{**design,'complete':True,'all_checks_passed':True,'sources_unchanged':True,
        'completed_at':TIMES[3],'checks_sha256':audit.sha(folder/'equivalence_checks.json'),'records':records})
    return root


@pytest.fixture
def case(template,tmp_path,monkeypatch):
    shutil.copytree(template,tmp_path/'case')
    output=tmp_path/'case/output';folder=output/'rough_mc_precision_v1/spent_equivalence_v3'
    monkeypatch.setattr(audit,'_preserved',lambda path:({},[]))
    monkeypatch.setattr(audit,'verify_sources',lambda *args:None)
    monkeypatch.setattr(audit,'verify_inputs',lambda *args:[])
    return folder,output,{'created_at':TIMES[4],'source_sha256':SOURCES,'work_batch_size':64}


def reseal_sample(folder,kind,mutate):
    path=folder/f'stream_0_{kind}_samples.npz';arrays=load(path);mutate(arrays)
    np.savez_compressed(path,**arrays)
    m=audit.read(folder/'manifest.json');m['records'][0][kind+'_sha256']=audit.sha(path)
    # Retaining a claimed success and rehashing its own inventory is insufficient.
    checks=audit.read(folder/'equivalence_checks.json')
    for row in checks:row.update(max_abs_error=0.,passed=True)
    write(folder/'equivalence_checks.json',checks);m['checks_sha256']=audit.sha(folder/'equivalence_checks.json')
    write(folder/'manifest.json',m)


def test_complete_spent_arrays_are_compared_instead_of_trusting_saved_passed_flags(case):
    result=audit.validate_preflight(*case)
    assert len(result['tables']['implementation_checks'])>80
    assert result['tables']['implementation_checks'].acceptance_role.eq(False).all()


@pytest.mark.parametrize('kind,field,index',[
    ('selected','stocks',(0,10,0)),('selected','premium_martingale',(2,20)),
    ('selected','gh16_continuation',(0,1)),('selected','basis',(0,12)),
    ('conditional','conditional_variances',(0,20)),
    ('conditional','conditional_expected_next',(1,30)),
    ('conditional','replay_physical_factors',(0,32,0))])
def test_adjusted_array_plus_updated_archive_and_manifest_hashes_still_fails(case,kind,field,index):
    folder,_,_=case
    def mutate(a):a[field][index]=.001
    reseal_sample(folder,kind,mutate)
    with pytest.raises(ValueError):audit.validate_preflight(*case)


@pytest.mark.parametrize('kind,field',[
    ('selected','european_martingale'),('selected','gh16_continuation'),
    ('conditional','conditional_expected_next'),('conditional','conditional_european_estimate')])
def test_omitting_a_compared_field_then_rehashing_cannot_shrink_the_equivalence_proof(case,kind,field):
    reseal_sample(case[0],kind,lambda a:a.pop(field))
    with pytest.raises(ValueError):audit.validate_preflight(*case)


@pytest.mark.parametrize('kind',['negative_elapsed','start_before_design','finish_after_manifest','missing_input_pin'])
def test_preflight_chronology_and_reference_pin_coverage_are_mandatory(case,kind):
    folder,_,_=case;m=audit.read(folder/'manifest.json')
    if kind=='negative_elapsed':m['records'][0]['elapsed_seconds']=-1.
    elif kind=='start_before_design':m['records'][0]['started_at']='2026-10-19T23:59:59+00:00'
    elif kind=='finish_after_manifest':m['records'][0]['completed_at']=TIMES[4]
    else:
        d=audit.read(folder/'design_before_evaluation.json')
        key=next(iter(d['input_sha256']));del d['input_sha256'][key]
        m['input_sha256']=d['input_sha256'];write(folder/'design_before_evaluation.json',d)
    write(folder/'manifest.json',m)
    with pytest.raises(ValueError):audit.validate_preflight(*case)

