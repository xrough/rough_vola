"""The staged runner preserves every shrinkage candidate and common bounds."""
from pathlib import Path
import importlib.util
import numpy as np
from rvlab.tests.test_pricing_rough_local_control import local_case
from rvlab.experiments._pricing_rough_tail_policy import build_margin_policies
from rvlab.experiments._pricing_rough_tail_switch_guarded import tail_switch_corrections

def runner():
    path=Path(__file__).resolve().parents[3]/'examples/rough_tail_attribution.py'
    spec=importlib.util.spec_from_file_location('rough_tail_runner_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def test_all_predeclared_shrunk_controls_are_convex_interpolations(monkeypatch):
    r=runner()
    def fit(payoff,m,stops,basis,*,bins,penalty):
        return np.full(basis.shape[-1],bins/basis.shape[-1]),{'status':'completed','bins':bins,'penalty':penalty}
    monkeypatch.setattr(r,'fit_correction_weights',fit)
    w={'tail_corrections':np.zeros((4,65,7)),'times':np.linspace(0.,1.,65),
        'payoff':np.zeros((4,65)),'current_martingale':np.zeros((4,65)),'stops':np.full(4,64)}
    controls,trials=r.fit_controls(w)
    assert len(trials)==8 and len(controls)==17
    for c in controls.values():
        if 'parent_control' in c:
            parent=controls[c['parent_control']]
            np.testing.assert_array_equal(c['weights'],parent['weights']*c['shrink_scale'])
        assert np.all(np.asarray(c['weights']).reshape(c['bins'],-1).sum(axis=1)<=1.+1e-14)

def test_cross_policy_bounds_share_each_upper_and_poststop_lower_is_fixed(local_case):
    r=runner();proxy,selected,trace=local_case
    m=trace['european_martingale'][:,:,0]+.7*trace['premium_martingale'][:,:,0]
    d=tail_switch_corrections(proxy,selected,trace,m);stops=d['stops']
    w={'current_martingale':m,'tail_corrections':d['corrections'],'times':trace['times'],
        'payoff':trace['discounted_payoffs'][:,:,0],'stops':stops}
    controls={'current baseline':{'family':'fallback','bins':1,'weights':np.zeros(7)},
        'poststop':{'family':'poststop','bins':1,'weights':np.array([.5,0.,0.,0.,0.,0.,0.])}}
    candidates,data=r.evaluate(w,trace,build_margin_policies(selected),controls)
    assert len(candidates)==7 and data['control0_lower'].shape==(len(m),7)
    np.testing.assert_array_equal(data['control0_lower'][:,0],data['control1_lower'][:,0])
    for i in (0,1):assert np.all(data[f'control{i}_upper'][:,None]>=data[f'control{i}_lower']-2e-11)
