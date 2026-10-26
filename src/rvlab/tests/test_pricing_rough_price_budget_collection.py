"""Sealed independent price studies cannot silently change models or budgets."""
from pathlib import Path
import hashlib
from types import SimpleNamespace

import pandas as pd
import pytest

from rvlab.experiments import _pricing_rough_price_budget_collection as c


@pytest.fixture
def design():
    # Keep the archived numerical protocol guards without private path archives.
    return c.read(Path(__file__).with_name('fixtures')/'price_budget_protocol_2026_10_19.json')


def test_actual_sealed_design_uses_unchanged_tolerances(design):c.validate_design(design)


@pytest.mark.parametrize('key,value',[
    ('streams',[{'seed':1106005042,'paths':8192}]),('total_paths',8192),
    ('coefficient_fit_seeds',[1106005042,1107005042]),('coefficient_bound',3.),
    ('physical_steps',256),('exercise_intervals',128),('frozen_policy','GH32 blend1'),
    ('gap_improvement_claimed',True),('no_empirical_valuation_recentering',False),
    ('upper_is_Rogers_max_after_shift',True),('independent_reference_mean_used',True),
    ('continuous_European_price_used_to_center',True),('features',192),
    ('price_atol',.002),('price_rtol',.0002),('confidence_multiplier',1.64),
    ('delta_atol',.004),('spread_fraction',.2),('tolerances_changed',True),
    ('optimal_value_ready',True),('stock_delta_ready',True)])
def test_protocol_and_scope_cannot_change(design,key,value):
    design[key]=value
    with pytest.raises(ValueError):c.validate_design(design)


def test_final_seed_cannot_reenter_training_firewall(design):
    design['forbidden_prior_seeds'].append(1106005042)
    with pytest.raises(ValueError,match='overlap'):c.validate_design(design)


@pytest.mark.parametrize('options,error',[
    ({'seed':True},ValueError),({'seed':43},ValueError),
    ({'compute_figures':1},TypeError),({'unsealed_override':1},TypeError)])
def test_options_rejected_before_accessing_context(options,error):
    with pytest.raises(error):c.price_budget_study(None,**options)


def test_missing_study_does_not_invent_new_path_evidence(tmp_path):
    with pytest.raises(FileNotFoundError,match='Complete the independent experiment'):
        c.collect_price_budget_campaign(tmp_path/'validation'/'missing')


@pytest.mark.parametrize('mutation',['current','snapshot','omitted','extra'])
def test_changed_source_dependency_and_snapshot_rejected(tmp_path,monkeypatch,mutation):
    root=tmp_path/'root';folder=tmp_path/'study'
    for base in (root,folder/'source'):
        (base/'src').mkdir(parents=True);(base/'src/model.py').write_text('fixed=1\n')
    monkeypatch.setattr(c,'ROOT',root)
    digest=hashlib.sha256((root/'src/model.py').read_bytes()).hexdigest()
    design={'source_sha256':{'src/model.py':digest}}
    if mutation=='current':(root/'src/model.py').write_text('fixed=2\n')
    if mutation=='snapshot':(folder/'source/src/model.py').write_text('fixed=2\n')
    if mutation=='omitted':design['source_sha256'].clear()
    if mutation=='extra':design['source_sha256']['other.py']=digest
    with pytest.raises(ValueError):c.verify_sources(folder,design,{'src/model.py'})


def test_model_and_development_input_cannot_escape_or_change(tmp_path):
    (tmp_path/'input.json').write_text('fixed')
    digest=hashlib.sha256(b'fixed').hexdigest()
    assert len(c.verify_inputs({'input_sha256':{'input.json':digest}},tmp_path))==1
    (tmp_path/'input.json').write_text('changed')
    with pytest.raises(ValueError,match='input changed'):
        c.verify_inputs({'input_sha256':{'input.json':digest}},tmp_path)
    with pytest.raises(ValueError,match='input changed'):
        c.verify_inputs({'input_sha256':{'../escape.json':digest}},tmp_path)


def test_price_pass_does_not_certify_american_engine_or_stock_deltas(tmp_path,monkeypatch):
    row={'path':str(tmp_path/'full_samples.npz'),'bytes':123,'sha256':'pinned'}
    raw={'tables':{'price_comparison':pd.DataFrame([{'price_budget_pass':True}])},
        'manifest':{'complete':True},'input_files':[row,row]}
    monkeypatch.setattr(c,'collect_price_budget_campaign',lambda _:raw)
    result=c.price_budget_study(SimpleNamespace(output_dir=tmp_path,progress=lambda _:None),compute_figures=False)
    m=result['manifest']
    assert m['primary_complete'] and m['primary_paths']==49152
    assert len(m['input_files'])==1 and m['gap_pointwise_unchanged']
    assert not m['american_pricing_ready'] and not m['stock_delta_ready']
    assert not m['new_control_stock_delta_validated'] and not m['optimal_delta_ready']
    assert m['primary_policy_fits']==m['primary_dual_control_fits']==0
    assert m['spent_control_regressions']==51


def test_one_input_cannot_have_conflicting_fingerprints(tmp_path,monkeypatch):
    row={'path':str(tmp_path/'full_samples.npz'),'bytes':123,'sha256':'pinned'}
    raw={'tables':{},'manifest':{},'input_files':[row,{**row,'sha256':'different'}]}
    monkeypatch.setattr(c,'collect_price_budget_campaign',lambda _:raw)
    with pytest.raises(ValueError,match='Conflicting'):
        c.price_budget_study(SimpleNamespace(output_dir=tmp_path,progress=lambda _:None),compute_figures=False)
