"""Stage22 precision evidence preserves its narrow pass without validating new Greeks."""
import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from rvlab.experiments._pricing_rough_shared_delta_history import (
    MC_PRECISION_PROFILE, load_historical_mc_precision)


def _archive(tmp_path, **overrides):
    archive=tmp_path/'archive';results=archive/'results';results.mkdir(parents=True)
    names=['rough_mc_precision_table'+str(i) for i in range(17)]
    for i,name in enumerate(names):
        pd.DataFrame([{'index':i,'estimate':1.4e-14,'recorded_check':i%2==0}]).to_csv(results/(name+'.csv'),index=False)
    figures=['rough_mc_precision_figure'+str(i) for i in range(3)]
    for name in figures:(results/(name+'.png')).write_bytes(b'recorded pixels')
    stage={'backend':'rough_heston_mc_precision','profile_fingerprint':MC_PRECISION_PROFILE,
        'primary_complete':True,'original_elapsed_seconds':263.25495504192077,
        'sampling_gate_pass':True,'american_pricing_ready':False,'stock_delta_ready':False,
        'optimal_delta_ready':False,
        'input_files':[{'path':f'frozen input {i}'} for i in range(631)],
        'analysis_sources':{f'frozen_source{i}':'a'*64 for i in range(17)},'primary_paths':196608,
        'seed_count':3,'new_control_stock_delta_validated':False,**overrides}
    manifest={'notebook':'01_pricing_and_exercise','evidence_version':'pricing_research_v1',
        'created_at':'2026-10-20T02:32:06Z','rough_mc_precision':stage,
        'tables':{name:1 for name in names},'figures':figures}
    (results/'manifest.json').write_text(json.dumps(manifest))
    _inventory(archive)
    return archive,stage


def _inventory(archive):
    rows=[]
    for path in (archive/'results').iterdir():
        data=path.read_bytes();rows.append({'preserved':str(path.relative_to(archive)),
            'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    (archive/'preservation_manifest.json').write_text(json.dumps({'files':rows}))


def test_historical_precision_preserves_narrow_pass_statistics_pixels_and_identities(tmp_path):
    archive,stage=_archive(tmp_path);messages=[]
    result=load_historical_mc_precision(SimpleNamespace(progress=messages.append),archive=archive)
    assert len(result['tables'])==17 and len(result['figures'])==3
    assert result['tables']['table0'].estimate.item()==1.4e-14
    assert result['tables']['table1'].recorded_check.item()==False
    assert all(path.read_bytes()==b'recorded pixels' for path in result['figures'].values())
    m=result['manifest'];assert m['historical_replay'] and not m['current_engine_execution']
    assert not m['american_pricing_ready'] and not m['stock_delta_ready'] and not m['optimal_delta_ready']
    assert m['analysis_sources']==stage['analysis_sources'] and m['original_elapsed_seconds']==263.25495504192077
    assert (m['primary_paths'],m['seed_count'])==(196608,3)
    assert m['new_control_stock_delta_validated'] is False and m['sampling_gate_pass'] is True and len(messages)==1
    assert not load_historical_mc_precision(archive=archive,compute_figures=False)['figures']


@pytest.mark.parametrize('changes',[{'profile_fingerprint':'wrong'},{'primary_complete':False},
    {'backend':'unverified backend'},{'input_files':[]},
    {'input_files':[{'path':'duplicate'}]*631}, {'sampling_gate_pass':False},
    {'american_pricing_ready':True},{'stock_delta_ready':True},
    {'new_control_stock_delta_validated':True},{'optimal_delta_ready':True},
    {'primary_paths':49152},{'analysis_sources':{}}])
def test_changed_or_partial_stage_cannot_be_relabelled_complete(tmp_path,changes):
    archive,_=_archive(tmp_path,**changes)
    with pytest.raises(ValueError,match='declared complete'):
        load_historical_mc_precision(archive=archive)


@pytest.mark.parametrize('kind',['table','figure'])
def test_complete_output_set_is_required_even_with_unchecked_manifest_count(tmp_path,kind):
    archive,_=_archive(tmp_path);path=archive/'results/manifest.json';m=json.loads(path.read_text())
    if kind=='table':m['tables'].pop('rough_mc_precision_table0')
    else:m['figures'].pop()
    path.write_text(json.dumps(m));_inventory(archive)
    with pytest.raises(ValueError,match='set is incomplete'):
        load_historical_mc_precision(archive=archive)


def test_outputs_must_belong_to_byte_verified_inventory(tmp_path):
    archive,_=_archive(tmp_path);path=archive/'preservation_manifest.json';m=json.loads(path.read_text())
    m['files']=[row for row in m['files'] if row['preserved']!='results/rough_mc_precision_table0.csv']
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='outside its inventory'):
        load_historical_mc_precision(archive=archive)


def test_changed_preserved_bytes_are_rejected(tmp_path):
    archive,_=_archive(tmp_path)
    (archive/'results/rough_mc_precision_table0.csv').write_text('changed\n')
    with pytest.raises(ValueError,match='Changed preserved'):
        load_historical_mc_precision(archive=archive)


@pytest.mark.parametrize('value',[1,None,'yes'])
def test_nonboolean_figure_request_is_rejected_before_archive_access(value):
    with pytest.raises(TypeError,match='boolean'):
        load_historical_mc_precision(archive='missing',compute_figures=value)


def test_large_preserved_payloads_are_streamed_instead_of_read_as_one_object(tmp_path,monkeypatch):
    from pathlib import Path
    archive,_=_archive(tmp_path)
    original=Path.read_bytes
    def bounded_read(path):
        assert path.name=='preservation_manifest.json','Only the small inventory may use read_bytes'
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',bounded_read)
    result=load_historical_mc_precision(archive=archive,compute_figures=False)
    assert len(result['tables'])==17 and result['manifest']['sampling_gate_pass'] is True


def test_preserved_paths_cannot_escape_the_archive_even_with_matching_bytes(tmp_path):
    archive,_=_archive(tmp_path)
    outside=tmp_path/'outside.csv';outside.write_text('outside evidence')
    p=archive/'preservation_manifest.json';m=json.loads(p.read_text())
    data=outside.read_bytes();m['files'].append({'preserved':'../outside.csv','bytes':len(data),
        'sha256':hashlib.sha256(data).hexdigest()})
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='remain inside'):
        load_historical_mc_precision(archive=archive)
