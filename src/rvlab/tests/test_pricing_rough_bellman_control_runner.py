"""Standalone stage completion cannot certify broken hash/chronology chains."""
import importlib.util
from pathlib import Path
import pytest


@pytest.fixture
def driver(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[3]/'examples/rough_bellman_control.py'
    spec = importlib.util.spec_from_file_location('test_control_driver', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'FOLDER', tmp_path)
    module.write(tmp_path/'design_before_training.json', {})
    design = {'created_at': '00', 'source_sha256': {'source': 'pin'},
        'final_streams': [{'seed': module.FINALS[0], 'paths': 4096}, {'seed': module.FINALS[1], 'paths': 4096}]}
    controls = {module.BASELINE: {'weights': [0.]*4}, 'target': {'weights': [1., 0., 0., 0.]}}
    module.write(tmp_path/'controls_before_selection.json', {'design_sha256': module.sha(tmp_path/'design_before_training.json'),
        'training_seed': module.TRAIN, 'selection_paths_generated_before_seal': False, 'controls': controls, 'sealed_at': '02'})
    seal = {'design_sha256': module.sha(tmp_path/'design_before_training.json'),
        'control_sha256': module.sha(tmp_path/'controls_before_selection.json'), 'selection_seed': module.SELECT,
        'controls': controls, 'control_order': list(controls), 'frozen_policy': 'GH16 blend1',
        'final_paths_generated_before_seal': False, 'selected_control': 'target', 'designated_challenger': 'target', 'sealed_at': '05'}
    module.write(tmp_path/'selection_before_final.json', seal)
    for stage, seed, paths, started, completed in [('training', module.TRAIN, 2048, '01', '03'),
            ('selection', module.SELECT, 2048, '04', '06'),
            ('final_0', module.FINALS[0], 4096, '07', '08'), ('final_1', module.FINALS[1], 4096, '07', '09')]:
        (tmp_path/f'{stage}_samples.npz').write_bytes(b'sealed sample')
        record = {'seed': seed, 'paths': paths, 'started_at': started, 'completed_at': completed, 'elapsed_seconds': 1.,
            'source_sha256': design['source_sha256'], 'sources_unchanged': True,
            'design_sha256': module.sha(tmp_path/'design_before_training.json'),
            'sample_sha256': module.sha(tmp_path/f'{stage}_samples.npz')}
        if stage == 'training': record['control_sha256'] = module.sha(tmp_path/'controls_before_selection.json')
        else: record['selection_sha256'] = module.sha(tmp_path/'selection_before_final.json')
        if stage.startswith('final'): record.update(index=int(stage[-1]), control_order=list(controls))
        module.write(tmp_path/f'{stage}_manifest.json', record)
    return module, design


def test_complete_chain_validates(driver):
    module, design = driver
    seal, records = module.validate_completed_stages(design)
    assert len(records) == 2 and seal['selected_control'] == 'target'


@pytest.mark.parametrize('stage,field,value', [('final_0', 'seed', 123), ('final_1', 'index', 0),
    ('final_0', 'paths', 2048), ('final_1', 'sample_sha256', 'bad'), ('final_1', 'selection_sha256', 'bad'),
    ('final_0', 'source_sha256', {}), ('final_0', 'control_order', ['target']),
    ('final_1', 'started_at', '04'), ('training', 'sample_sha256', 'bad'),
    ('selection', 'started_at', '01'), ('selection', 'design_sha256', 'bad')])
def test_bad_stage_record_rejected(driver, stage, field, value):
    module, design = driver; path = module.FOLDER/f'{stage}_manifest.json'
    record = module.read(path); record[field] = value; module.write(path, record)
    with pytest.raises(ValueError): module.validate_completed_stages(design)


def test_final_sample_mutation_is_rejected(driver):
    module, design = driver; (module.FOLDER/'final_0_samples.npz').write_bytes(b'altered')
    with pytest.raises(ValueError, match='final0'): module.validate_completed_stages(design)
