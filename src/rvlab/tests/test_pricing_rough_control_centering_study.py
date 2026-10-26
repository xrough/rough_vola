"""Fresh-path means cannot silently reuse the triggering stream or select seeds."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rvlab.experiments import _pricing_rough_control_centering as study


def test_repeat_design_has_exact_eight_new_streams_and_budget():
    rows = study._validated_repeats(study.control_centering_repeat_specs(), (350005042,430005042))
    assert len(rows) == 8 and sum(r['paths'] for r in rows) == 65536
    assert [r['seed'] for r in rows] == [410005042+1000000*i for i in range(8)]


@pytest.mark.parametrize('specs', [[], {}, [{'seed':True,'paths':2}], [{'seed':-1,'paths':2}],
    [{'seed':1,'paths':1}], [{'seed':1,'paths':2,'extra':0}],
    [{'seed':350005042,'paths':8}], [{'seed':1,'paths':8},{'seed':1,'paths':8}]])
def test_seed_firewall_and_input_validation(specs):
    with pytest.raises(ValueError): study._validated_repeats(specs,(350005042,))


def test_tail_diagnostics_keep_extremes_and_sparse_contributions():
    x = np.r_[np.zeros(98),-8.,12.]
    before = x.copy(); t=study._tails(x)
    np.testing.assert_array_equal(x,before)
    assert t['nonzero_fraction'] == .02
    assert t['minimum'] == -8 and t['maximum'] == 12
    assert t['largest_one_percent_absolute_mass_fraction'] == .6
    assert t['excess_kurtosis'] > 30
    assert study._tails(np.zeros(8))['skewness'] == 0


@pytest.mark.parametrize('options,error', [
    ({'run_rough_control_centering':1},ValueError),
    ({'rough_control_centering_kwargs':[]},TypeError),
    ({'rough_control_centering_kwargs':{'seed':1}},ValueError),
    ({'run_rough_control_centering':True,'rough_policy_improvement_kwargs':{'trial_specs':[]}},ValueError),
    ({'run_rough_control_centering':True,'rough_dual_tightening_kwargs':{'trial_specs':[]}},ValueError),
])
def test_wrapper_rejects_ambiguous_new_or_historical_options_before_pricing(options,error):
    from rvlab.experiments._pricing_research import pricing_research
    with pytest.raises(error): pricing_research(SimpleNamespace(),**options)


def _archive(tmp_path):
    root=tmp_path/'archive'; (root/'results').mkdir(parents=True)
    (root/'results/rough_policy_improvement_summary.csv').write_text('gap,ready\n0.0017935384380483655,False\n')
    (root/'results/rough_policy_improvement_failures.csv').write_text('\n')
    m={'evidence_version':'pricing_research_v1','notebook':'01_pricing_and_exercise','created_at':'originaltime',
        'tables':{'rough_policy_improvement_summary':1,'rough_policy_improvement_failures':0},'figures':[],
        'rough_policy_improvement':{'backend':'rough_heston_policy_improvement','profile_fingerprint':study.POLICY_PROFILE,
            'source_elapsed_seconds':2430.45,'sources':{'original.py':'unchanged'}}}
    (root/'results/manifest.json').write_text(json.dumps(m))
    files=[{'preserved':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(root.rglob('*')) if p.is_file()]
    (root/'preservation_manifest.json').write_text(json.dumps({'files':files}))
    return root


def test_historical_policy_replay_preserves_source_identity_prices_and_cost(tmp_path):
    root=_archive(tmp_path); r=study.load_historical_policy(archive=root)
    assert r['tables']['summary'].gap.item() == .0017935384380483655
    assert r['tables']['failures'].empty
    assert r['manifest']['historical_replay'] and not r['manifest']['current_engine_execution']
    assert r['manifest']['source_elapsed_seconds'] == 2430.45
    assert r['manifest']['sources'] == {'original.py':'unchanged'}
    assert not r['manifest']['american_pricing_ready'] and not r['manifest']['stock_delta_ready']
    (root/'results/rough_policy_improvement_summary.csv').write_text('changed')
    with pytest.raises(ValueError,match='Changed preserved'):study.load_historical_policy(archive=root)
