"""First-stage presentation requires its declared namespace and preserves history."""
import base64
import hashlib
import json

import pandas as pd
import pytest

from rvlab.experiments import pricing
from rvlab.experiments.pricing_presentation import (
    load_recorded_results, present_archive, refresh_presentations)


def _write(root, stem='01_pricing_and_exercise', **updates):
    folder=root/stem;folder.mkdir(parents=True,exist_ok=True)
    manifest={'notebook':stem,'artifact_family':'first_stage_v1',
        'evidence_version':'pricing_first_stage_v1','data_kind':'synthetic',
        'backend':'mixed_european_fourier_american_fd','tables':{},'figures':[],
        'artifact_sha256':{},
        'elapsed_seconds':2.,'created_at':'report collection'}
    manifest.update(updates)
    (folder/'manifest.json').write_text(json.dumps(manifest))
    return manifest


def test_explicit_first_stage_namespace_never_falls_back_to_research_evidence(tmp_path,monkeypatch):
    monkeypatch.setattr(pricing,'ROOT',tmp_path)
    stem='01_pricing_and_exercise'
    old=tmp_path/'notebooks/_output/pricing/research_v1'
    _write(old,stem,artifact_family='research_v1',evidence_version='pricing_research_v1')
    with pytest.raises(FileNotFoundError):
        load_recorded_results(stem,family='first_stage_v1')
    new=tmp_path/'notebooks/_output/pricing/first_stage_v1'
    expected=_write(new,stem)
    assert load_recorded_results(stem,family='first_stage_v1')['manifest']==expected
    assert load_recorded_results(stem)['manifest']['evidence_version']=='pricing_research_v1'


@pytest.mark.parametrize('updates',[{'artifact_family':'research_v1'},
    {'evidence_version':'pricing_research_v1'},{'data_kind':'real_spy'},
    {'spy_inputs':{'source':'market'}}])
def test_first_stage_synthetic_artifacts_require_matching_identity(tmp_path,updates):
    _write(tmp_path,**updates)
    with pytest.raises(ValueError):
        load_recorded_results('01_pricing_and_exercise',artifact_root=tmp_path,family='first_stage_v1')


def test_first_stage_spy_still_requires_direct_american_evidence(tmp_path):
    stem='03_spy_calibration'
    _write(tmp_path,stem,data_kind='real_spy',exercise_style='european')
    with pytest.raises(ValueError,match='American backend'):
        load_recorded_results(stem,artifact_root=tmp_path,family='first_stage_v1')
    expected=_write(tmp_path,stem,data_kind='real_spy',backend='american_vanilla_fd',
        exercise_style='american',status='blocked_inputs')
    assert load_recorded_results(stem,artifact_root=tmp_path,family='first_stage_v1')['manifest']==expected


def test_archive_links_and_cost_text_use_the_report_namespace(monkeypatch):
    from IPython import display as display_module
    shown=[];monkeypatch.setattr(display_module,'display',shown.append)
    result={'manifest':{'notebook':'01_pricing_and_exercise','artifact_family':'first_stage_v1',
        'elapsed_seconds':2.,'created_at':'collection'},'tables':{'failures':pd.DataFrame()}}
    present_archive(result)
    html=shown[0].data
    assert '../_output/pricing/first_stage_v1/01_pricing_and_exercise/failures.csv' in html
    assert 'Report artifacts collected' in html and 'original experiment runtime 2.0' not in html
    assert 'cold computation' in html
    result['manifest']['artifact_family']='../research_v1'
    with pytest.raises(ValueError,match='artifact family'):present_archive(result)


@pytest.mark.parametrize('family',['first_stage_v1','first_stage_v2'])
def test_refresh_uses_explicit_notebook_family_and_isolates_exports(tmp_path,monkeypatch,family):
    nbformat=pytest.importorskip('nbformat')
    monkeypatch.setattr(pricing,'ROOT',tmp_path)
    path=tmp_path/'01_pricing_and_exercise.ipynb'
    computation=nbformat.v4.new_code_cell("raise RuntimeError('never recompute')",execution_count=4)
    display=nbformat.v4.new_code_cell("present_figure(results,'panel')",metadata={'rvlab_presentation':True})
    execution={'executed_at':'original','elapsed_seconds':200}
    nbformat.write(nbformat.v4.new_notebook(cells=[computation,display],
        metadata={'rvlab_artifact_family':family,'rvlab_execution':execution}),path)
    def render(source,**kwargs):
        n=nbformat.read(source,4)
        assert f"family={family!r}" in n.cells[0].source
        assert all('never recompute' not in c.source for c in n.cells)
        pixels=base64.b64encode(b'preserved figure bytes').decode()
        n.cells[1].outputs=[nbformat.v4.new_output('display_data',
            data={'text/html':f'<img src="data:image/png;base64,{pixels}">'},
            metadata={'rvlab_figure':'panel'})]
        nbformat.write(n,source)
    monkeypatch.setattr(pricing,'execute_notebook',render)
    refresh_presentations([path]);n=nbformat.read(path,4)
    assert n.cells[0]==computation and n.metadata.rvlab_execution==execution
    export=tmp_path/'notebooks/_output/pricing'/family/'presentation/01_pricing_and_exercise/panel.png'
    assert export.read_bytes()==b'preserved figure bytes'
    assert not (tmp_path/'notebooks/_output/pricing/presentation').exists()
    assert n.metadata.rvlab_presentation.artifact_family==family


@pytest.mark.parametrize('family',['../first_stage_v1','research_v1','unknown'])
def test_invalid_explicit_families_are_rejected_before_io(family,tmp_path):
    with pytest.raises(ValueError,match='artifact family'):
        load_recorded_results('01_pricing_and_exercise',artifact_root=tmp_path,family=family)


@pytest.mark.parametrize('changed',['csv','png','missing_map','extra_map'])
def test_first_stage_refresh_rejects_same_size_artifact_changes_and_inexact_digest_maps(tmp_path,changed):
    folder=tmp_path/'01_pricing_and_exercise';folder.mkdir()
    (folder/'data.csv').write_text('value\n1\n');(folder/'panel.png').write_bytes(b'figure1')
    digests={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
    _write(tmp_path,tables={'data':1},figures=['panel'],artifact_sha256=digests)
    if changed=='csv':(folder/'data.csv').write_text('value\n9\n')
    if changed=='png':(folder/'panel.png').write_bytes(b'figure9')
    if changed in ('missing_map','extra_map'):
        path=folder/'manifest.json';manifest=json.loads(path.read_text())
        if changed=='missing_map':manifest.pop('artifact_sha256')
        else:manifest['artifact_sha256']['extra.csv']='unrecorded'
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='changed after execution|digest map'):
        load_recorded_results('01_pricing_and_exercise',artifact_root=tmp_path,family='first_stage_v1')
