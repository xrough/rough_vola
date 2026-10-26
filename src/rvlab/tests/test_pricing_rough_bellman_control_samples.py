"""Primitive, admissibility and convex-budget guards are independent of the runner."""
from types import SimpleNamespace
import numpy as np
import pytest
import rvlab.experiments._pricing_rough_bellman_control_samples as audit


@pytest.fixture
def sample(monkeypatch):
    n=3; arrays={'times':np.linspace(0,1,65),'discounts':np.ones(65),
        'payoff':np.full((n,65),2.),'stocks':np.full((n,65,1),98.),
        'current_martingale':np.zeros((n,65)),'tail_corrections':np.zeros((n,65,7)),
        'european_proxy':np.zeros((n,65,1)),'latest_martingale':np.zeros((n,65)),
        'target_martingales':np.zeros((n,65,4)),'target_expected_european':np.zeros((n,65)),
        'target_actual_premiums':np.zeros((n,65,4)),'target_expected_premiums':np.zeros((n,65,4)),
        'gh16_stops':np.ones(n,dtype=int),'gh16_raw':np.full(n,2.),
        'gh16_continuation':np.full((n,65),np.nan),'gh16_computed_mask':np.zeros((n,65),bool),
        'candidate0_lower':np.full(n,2.),'candidate0_upper':np.full(n,2.)}
    arrays['gh16_continuation'][:,1]=1.; arrays['gh16_computed_mask'][:,1]=True
    monkeypatch.setattr(audit,'validate_tail_samples',lambda *a,**kw: {})
    controls={audit.BASELINE:{'family':'fallback','bins':1,'penalty':0.,'weights':[0.]*4}}
    tail={'selected_control':'base','controls':{'base':{'bins':1,'weights':[0.]*7}}}
    bundle={'proxy':SimpleNamespace(strike=100.,is_call=False)}
    return arrays,controls,bundle,tail


def check(sample):
    a,c,b,t=sample
    return audit.validate_control_samples(a,c,b,None,None,t,paths=3)


def test_full_primitive_algebra_and_stops(sample):
    result=check(sample)
    assert len(result['raw'])==3 and len(result['pure_targets'])==4


@pytest.mark.parametrize('field,index,value', [('target_martingales',(0,10,2),.1),
    ('target_expected_european',(1,20),.1),('target_actual_premiums',(2,64,0),.1),
    ('target_expected_premiums',(0,10,0),-.1),('gh16_stops',(0,),2),
    ('gh16_raw',(0,),1.),('gh16_computed_mask',(0,2),True),
    ('gh16_continuation',(0,2),1.),('candidate0_upper',(0,),3.),('latest_martingale',(1,10),.1)])
def test_tampered_actual_conditional_policy_or_endpoint_rejected(sample,field,index,value):
    sample[0][field][index]=value
    with pytest.raises(ValueError):check(sample)


@pytest.mark.parametrize('weights', [[-.1,0,0,0],[.6,.6,0,0],[0]*3,[float('nan')]*4])
def test_convex_control_budgets(weights):
    controls={audit.BASELINE:{'family':'fallback','bins':1,'penalty':0.,'weights':[0.]*4},
        'target 1 bins penalty0':{'family':'target','bins':1,'penalty':0.,'weights':weights}}
    with pytest.raises(ValueError):audit.validate_specs(controls,{'target 1 bins penalty0'})


def test_only_predeclared_successful_candidates_allowed():
    controls={audit.BASELINE:{'family':'fallback','bins':1,'penalty':0.,'weights':[0.]*4}}
    audit.validate_specs(controls,set())
    controls['undeclared']={'family':'target','bins':1,'penalty':0.,'weights':[0.]*4}
    with pytest.raises(ValueError):audit.validate_specs(controls,set())
