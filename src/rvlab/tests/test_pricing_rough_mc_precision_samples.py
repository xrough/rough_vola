"""Selected-only identities without fabricating unused price tables."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import expit

import rvlab.experiments._pricing_rough_mc_precision_samples as audit


@pytest.fixture
def sample(monkeypatch):
    """Deterministic algebra fixture; RNG/reference-function tests are separate."""
    n=3;times=np.linspace(0.,1.,65);physical=np.linspace(0.,1.,129)
    class Model:
        center=np.zeros(4)
        scale=np.ones(4)
        def __init__(self,value):self.amount=value
        def value(self,stock,factors,strike):return np.full(np.shape(stock),self.amount)
    selected=SimpleNamespace(times=times,premium_cap=.6,
        continuations=tuple(Model(.6 if j<64 else 0.) for j in range(65)))
    market=SimpleNamespace(spot=98.,dividends=SimpleNamespace(events=()),
        discount_curve=SimpleNamespace(discount=lambda t:np.ones(np.shape(t))))
    proxy=SimpleNamespace(strike=100.,is_call=False,exercise_times=times,continuation_times=times,
        config=SimpleNamespace(steps=128,batch_size=512),kernel=SimpleNamespace(nodes=np.ones(4)),
        market=market,params=None)
    fit=SimpleNamespace(config=SimpleNamespace(predictable_features=(),
        components=('european','premium','micro_boundary')),weights=np.full((4,3),.2),
        time_edges=np.linspace(0.,1.,5))
    bundle={'proxy':proxy,'selected_policy':selected,'weights':fit}
    a={'times':times,'discounts':np.ones(65),'eligible':np.ones(65,dtype=bool),
       'actual_time_grid':physical,'stocks':np.full((n,65,1),98.),'factors':np.zeros((n,65,4)),
       'european_proxy':np.full((n,65,1),1.3),'payoff':np.full((n,65),2.)}
    a['european_proxy'][:,-1]=2.
    for name in ('european_martingale','premium_martingale','intrinsic_martingale',
                 'micro_boundary_martingale','stable_old_martingale','prior_martingale',
                 'current_martingale','latest_martingale','local_actual','local_expected'):
        a[name]=np.zeros((n,65))
    for name,features in (('prior_local_corrections',3),('grid_corrections',6),
            ('tail_corrections',7),('grid_actual',2),('grid_expected',2),
            ('target_martingales',4),('target_actual_premiums',4),('target_expected_premiums',4)):
        a[name]=np.zeros((n,65,features))
    a['target_expected_european']=a['european_proxy'][:,:,0].copy()
    a['target_expected_european'][:,0]=0.
    margin=np.full((n,65),2.-1.3-.6);margin[:,-1]=0.
    post=(np.arange(64)[None,:]>=np.ones((n,1))).astype(float);left=margin[:,:-1]
    a['tail_margin']=margin
    a['tail_gates']=np.stack((post,post*expit(left/.1),post*expit(left/.3),
        expit(left/.03),expit(left/.1),expit(left/.3),np.zeros((n,64))),axis=-1)
    a['gh16_continuation']=np.full((n,65),np.nan)
    a['gh16_computed_mask']=np.zeros((n,65),dtype=bool)
    stops=np.array([1,2,64])
    for path,stop in enumerate(stops):
        for right in range(1,min(stop+1,64)):
            a['gh16_computed_mask'][path,right]=True
            a['gh16_continuation'][path,right]=1. if right==stop else 3.
    a.update(gh16_stops=stops,gh16_raw=np.full(n,2.))
    for name in ('candidate3_lower','candidate3_upper','unchanged_lower','unchanged_upper'):
        a[name]=np.full(n,2.)
    anchor=np.array([-.02,.01,.03]);coefficients=np.linspace(-.1,.1,193)
    basis=np.zeros((n,193));basis[:,-1]=anchor;shift=basis@coefficients
    a.update(basis=basis,shared_control=anchor,shift=shift,shared_lower=2.-shift,shared_upper=2.-shift)
    prior={'bins':4,'weights':np.tile([.1,.2,.3],4).tolist()}
    grid={'bins':4,'weights':np.tile([.1,0.,.1,.1,0.,.1],4).tolist()}
    tail={'selected_control':'selected','controls':{'selected':{'bins':1,'weights':[.01]*7}}}
    target={'selected_control':'target 4 bins penalty0','controls':{
        'target 4 bins penalty0':{'bins':4,'weights':np.tile([.1,.2,.1,.2],4).tolist()}}}
    monkeypatch.setattr(audit,'_time_grid',lambda *args:physical)
    monkeypatch.setattr(audit,'_bounded_probes',lambda *args:[{'bounded':True}])
    monkeypatch.setattr(audit,'validate_conditional_european',lambda *args,**kwargs:{
        'shared_european_control':anchor,'probes':[],'max_stock_error':0.,
        'max_factor_error':0.,'max_log_mean_error':0.,'max_clock_error':0.,'max_probability_error':0.})
    return a,bundle,prior,grid,tail,target,coefficients


def validate(sample):
    a,bundle,prior,grid,tail,target,coefficients=sample
    return audit.validate_selected_samples(a,bundle,prior,grid,tail,target,
        paths=3,seed=1119005042,coefficients=coefficients,models=(None,)*65)


def test_only_retained_components_and_selected_endpoints_are_required(sample):
    result=validate(sample)
    assert result['checks']['basis_columns']==193
    assert result['checks']['no_unselected_endpoints_invented']
    assert set(result['values'])=={'unchanged control','shared control'}
    np.testing.assert_array_equal(result['stops'],[1,2,64])
    assert not any(k.startswith('control0_') or k.startswith('candidate0_') for k in sample[0])


@pytest.mark.parametrize('field,index',[
    ('local_actual',(0,10)),('local_expected',(1,20)),('grid_actual',(0,10,0)),
    ('grid_expected',(2,30,1)),('prior_local_corrections',(0,10,1)),
    ('grid_corrections',(1,20,4)),('stable_old_martingale',(0,10)),
    ('prior_martingale',(1,20)),('current_martingale',(2,30)),('tail_margin',(0,1)),
    ('tail_gates',(0,0,0)),('tail_corrections',(1,20,3)),('latest_martingale',(2,30)),
    ('target_martingales',(0,10,0)),('target_expected_european',(1,20)),
    ('target_actual_premiums',(2,30,0)),('target_expected_premiums',(0,64,1)),
    ('candidate3_lower',(0,)),('candidate3_upper',(1,)),('unchanged_lower',(0,)),
    ('shared_upper',(1,)),('shift',(0,)),('shared_control',(1,)),('basis',(2,0)),('payoff',(0,3))])
def test_tampered_used_primitive_component_or_endpoint_rejected(sample,field,index):
    sample[0][field][index]+=.001
    with pytest.raises(ValueError):validate(sample)


@pytest.mark.parametrize('kind',['mask','continuation_off_mask','stops','raw','omit_path',
    'omit_primitive','changed_coefficients','empirical_recentering','nonfinite','inception_memory'])
def test_own_stopping_full_path_and_unrecentered_shift_guards(sample,kind):
    a=sample[0]
    if kind=='mask':a['gh16_computed_mask'][0,2]=True
    elif kind=='continuation_off_mask':a['gh16_continuation'][0,2]=1.
    elif kind=='stops':a['gh16_stops'][1]=3
    elif kind=='raw':a['gh16_raw'][0]+=.1
    elif kind=='omit_path':a['stocks']=a['stocks'][:-1]
    elif kind=='omit_primitive':del a['local_expected']
    elif kind=='changed_coefficients':sample[-1][0]=2.1
    elif kind=='empirical_recentering':a['shift']-=a['shift'].mean()
    elif kind=='nonfinite':a['target_martingales'][0,10,1]=np.inf
    else:a['factors'][0,0,0]=.1
    with pytest.raises(ValueError):validate(sample)


def test_weighted_control_uses_each_increment_and_its_own_calendar_bin():
    correction=np.zeros((2,65,2));correction[:,:,0]=np.arange(65)*1e-3
    correction[:,:,1]=np.arange(65)**2*1e-3
    weights=np.array([[.1,.2],[.2,.3],[.3,.1],[.1,.4]])
    control={'bins':4,'weights':weights.ravel().tolist()}
    expected=np.zeros((2,65))
    for right in range(1,65):
        w=weights[(right-1)//16]
        expected[:,right]=expected[:,right-1]+1e-3*(w[0]+w[1]*(right**2-(right-1)**2))
    np.testing.assert_allclose(audit._weighted(correction,control),expected,atol=2e-13,rtol=0.)


def test_same_translation_preserves_every_original_gap(sample):
    result=validate(sample);old=result['values']['unchanged control'];new=result['values']['shared control']
    np.testing.assert_allclose(new['upper']-new['lower'],old['upper']-old['lower'],rtol=0.,atol=2e-13)
    assert result['checks']['all_path_conditional_means_repriced'] is False

