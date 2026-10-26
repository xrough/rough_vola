"""The Bellman campaign keeps the latest selected tail martingale common."""
from pathlib import Path
import importlib.util
import numpy as np
from rvlab.tests.test_pricing_rough_local_control import local_case
from rvlab.experiments._pricing_rough_tail_switch_guarded import tail_switch_corrections
from rvlab.experiments._pricing_rough_tail_policy import BASELINE_NAME

def test_latest_selected_tail_control_is_preserved_and_eight_arrays_are_added(local_case,monkeypatch):
    path=Path(__file__).resolve().parents[3]/'examples/rough_bellman_policy.py'
    spec=importlib.util.spec_from_file_location('bellman_runner_test',path);runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    proxy,selected,trace=local_case;m=trace['european_martingale'][:,:,0]
    correction=tail_switch_corrections(proxy,selected,trace,m)
    w={'current_martingale':m,'tail_corrections':correction['corrections'],'times':trace['times'],
        'payoff':trace['discounted_payoffs'][:,:,0],'stops':correction['stops']}
    frozen={'proxy':proxy,'selected_policy':selected}
    controls={'current baseline':{'family':'fallback','bins':1,'weights':np.zeros(7)},
        'selected test control':{'family':'extended','bins':1,'weights':np.array([0.,0.,0.,0.,.2,0.,0.])}}
    seal={'selected_policy':BASELINE_NAME,'selected_control':'selected test control','controls':controls}
    monkeypatch.setattr(runner.tail,'generate',lambda seed,paths:(frozen,trace,w))
    monkeypatch.setattr(runner,'latest_control',lambda:(seal,controls['selected test control']))
    _,_,data,result,diagnostics=runner.generate(trace['seed'],len(m))
    keys={'bellman_continuations','bellman_direct_continuations','bellman_computed_mask','baseline_continuation',
        'new_raw','new_stops','new_lower','common_upper'}
    assert keys.issubset(data) and len(keys)==8
    np.testing.assert_array_equal(data['new_lower'][:,0],data['control1_lower'][:,0])
    np.testing.assert_array_equal(data['common_upper'],data['control1_upper'])
    np.testing.assert_array_equal(data['new_stops'][:,0],data['stops'][:,0])
    assert diagnostics['common_upper_exact'] and result['new_lower'].shape==(len(m),7)
