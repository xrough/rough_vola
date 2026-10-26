"""Independent own-bump sample identities; deterministic fixtures are not MC evidence."""
from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest

import rvlab.experiments._pricing_rough_shared_delta_samples as audit
from rvlab.experiments._pricing_rough_shared_delta_statistics import RELATIVE

ORIGINAL_RNG_ANCHOR=audit._rng_anchor


@pytest.fixture
def sample(monkeypatch):
    """Full algebra fixture; real RNG and conditional functions are tested separately."""
    n=3;times=np.linspace(0.,1.,65);physical=np.linspace(0.,1.,129)
    class Model:
        center=np.zeros(4)
        scale=np.ones(4)
        def __init__(self,value):self.amount=value
        def value(self,stocks,factors,strike):return np.full(np.shape(stocks),self.amount)
    selected=SimpleNamespace(times=times,premium_cap=.7,
        continuations=tuple(Model(.2 if j<64 else 0.) for j in range(65)))
    market=SimpleNamespace(spot=100.,dividends=SimpleNamespace(events=()),
        discount_curve=SimpleNamespace(discount=lambda t:np.ones(np.shape(t))))
    proxy=SimpleNamespace(strike=100.,is_call=False,exercise_times=times,continuation_times=times,
        config=SimpleNamespace(steps=128,batch_size=512),kernel=SimpleNamespace(nodes=np.ones(4)),
        market=market,params=None)
    fit=SimpleNamespace(config=SimpleNamespace(predictable_features=(),
        components=('european','premium','micro_boundary')),weights=np.full((4,3),.2),
        time_edges=np.linspace(0.,1.,5))
    bundle={'proxy':proxy,'selected_policy':selected,'weights':fit}
    spots=100.*(1+RELATIVE)
    stocks=np.broadcast_to(.98*spots,(n,65,5)).copy();stocks[:,0]=spots
    payoff=np.maximum(100.-stocks,0.)
    a=dict(relative_spots=RELATIVE.copy(),initial_spots=spots.copy(),times=times,
        discounts=np.ones(65),eligible=np.ones(65,dtype=bool),actual_time_grid=physical,
        operator_time_grid=physical.copy(),stocks=stocks,factors=np.zeros((n,65,4)),
        payoff=payoff,calendar_european=np.full((n,65,5),.5),
        gh16_continuation=np.full((n,65,5),np.nan),
        gh16_computed_mask=np.zeros((n,65,5),dtype=bool))
    a['calendar_european'][:,-1]=payoff[:,-1]
    stops=np.array([[1,2,3,4,64],[2,1,64,3,4],[64,4,3,2,1]])
    for p in range(n):
        for b in range(5):
            for j in range(1,min(stops[p,b]+1,64)):
                a['gh16_computed_mask'][p,j,b]=True
                a['gh16_continuation'][p,j,b]=payoff[p,j,b]+(-.1 if j==stops[p,b] else .1)
    a['gh16_stops']=stops
    anchor=np.arange(n*5).reshape(n,5)*.003-.02
    coefficients=np.linspace(-.1,.1,193)
    for rep in ('original','micro'):
        a[rep+'_processes']=np.zeros((n,65,5,8))
        a[rep+'_stage_corrections']=np.zeros((n,65,5,3))
        a[rep+'_latest_martingale']=np.zeros((n,65,5))
        a[rep+'_selected_martingale']=np.zeros((n,65,5))
        basis=np.zeros((n,5,193));basis[:,:,-1]=anchor
        shift=basis@coefficients
        raw=payoff[np.arange(n)[:,None],stops,np.arange(5)[None,:]]
        upper=payoff.max(axis=1)
        for key,value in dict(basis=basis,anchor=anchor,shift=shift,raw=raw,stable=raw,
                unchanged_lower=raw,unchanged_upper=upper,
                shared_lower=raw-shift,shared_upper=upper-shift).items():
            a[rep+'_'+key]=value.copy()
    prior={'bins':4,'weights':np.tile([.1,.2,.3],4).tolist()}
    grid={'bins':4,'weights':np.tile([.1,0.,.1,.1,0.,.1],4).tolist()}
    tail={'selected_control':'selected','controls':{'selected':{'bins':1,'weights':[.01]*7}}}
    target={'selected_control':'target 4 bins penalty0','controls':{
        'target 4 bins penalty0':{'bins':4,'weights':np.tile([.1,.2,.1,.2],4).tolist()}}}
    monkeypatch.setattr(audit,'_time_grid',lambda *args:physical)
    monkeypatch.setattr(audit,'_panel',lambda *args:[{'bounded':True}])
    monkeypatch.setattr(audit,'_rng_anchor',lambda *args:(payoff[:,-1]-anchor,[],np.zeros(4)))
    return a,bundle,prior,grid,tail,target,coefficients


def validate(sample,**kwargs):
    a,bundle,prior,grid,tail,target,coefficients=sample
    return audit.validate_shared_delta_samples(a,bundle,prior,grid,tail,target,
        paths=3,seed=123456,steps=128,representations=('original','micro'),
        coefficients=coefficients,models=(None,)*65,**kwargs)


def test_all_five_own_stops_and_two_representations_are_audited(sample):
    result=validate(sample)
    assert result['checks']['all193_bases_and_shifts_checked']
    assert result['checks']['all_own_stops_checked']
    assert set(result['stream']['representations'])=={'original','micro'}
    np.testing.assert_array_equal(result['stream']['representations']['original']['stops'],
        sample[0]['gh16_stops'])
    assert not np.array_equal(sample[0]['gh16_stops'][:,0],sample[0]['gh16_stops'][:,2])


@pytest.mark.parametrize('field,index',[
    ('original_processes',(0,10,1,0)),('original_stage_corrections',(1,20,2,0)),
    ('micro_stage_corrections',(0,10,0,1)),('original_stage_corrections',(1,25,2,2)),
    ('micro_latest_martingale',(2,30,4)),('original_selected_martingale',(0,20,3)),
    ('original_basis',(1,0,0)),('micro_anchor',(2,4)),('original_shift',(0,2)),
    ('original_raw',(0,0)),('original_stable',(1,3)),('micro_unchanged_lower',(0,4)),
    ('original_unchanged_upper',(1,2)),('micro_shared_lower',(1,4)),
    ('original_shared_upper',(2,3)),('payoff',(0,3,1)),('factors',(0,0,0))])
def test_used_primitives_endpoints_and_each_stock_basis_reject_tampering(sample,field,index):
    sample[0][field][index]+=.001
    with pytest.raises(ValueError):validate(sample)


@pytest.mark.parametrize('kind',[
    'mask','off_mask_continuation','nonfinite_continuation','stops','origin_mask',
    'omit_path','operator_grid','bump','coefficient','recenter','nonfinite_process',
    'missing_calendar','tail_weights','target_identity','changed_physical_components'])
def test_calendar_own_stopping_and_common_control_guards(sample,kind):
    a,bundle,prior,grid,tail,target,coef=sample
    if kind=='mask':a['gh16_computed_mask'][0,2,0]=True
    elif kind=='off_mask_continuation':a['gh16_continuation'][0,2,0]=1.
    elif kind=='nonfinite_continuation':a['gh16_continuation'][0,1,0]=np.inf
    elif kind=='stops':a['gh16_stops'][0,0]=2
    elif kind=='origin_mask':a['gh16_computed_mask'][0,0,0]=True
    elif kind=='omit_path':a['stocks']=a['stocks'][:-1]
    elif kind=='operator_grid':a['operator_time_grid']=a['operator_time_grid']+.001
    elif kind=='bump':a['initial_spots'][0]+=.001
    elif kind=='coefficient':coef[0]=2.01
    elif kind=='recenter':a['original_shift']-=a['original_shift'].mean()
    elif kind=='nonfinite_process':a['original_processes'][1,10,0,0]=np.inf
    elif kind=='missing_calendar':del a['calendar_european']
    elif kind=='tail_weights':tail['controls']['selected']['weights'][0]=1.1
    elif kind=='target_identity':target['selected_control']='unselected'
    else:a['micro_processes'][0,10,0,4]=.001
    with pytest.raises(ValueError):validate(sample)


@pytest.mark.parametrize('bad',[{'bins':0,'weights':[]},{'bins':True,'weights':[0.]},
    {'bins':4,'weights':[0.]*11},{'bins':1,'weights':[.5,.5,.1]},
    {'bins':1,'weights':[0.,np.nan,0.]},{'bins':1,'weights':[-.01,0.,0.]}])
def test_frozen_weights_have_finite_convex_bin_budgets(bad):
    with pytest.raises(ValueError):audit.weights_for(bad,3,0)


def test_calendar_bin_boundaries_keep_the_left_interval_convention():
    weights=np.array([[.1,.2,.3],[.2,.3,.1],[.3,.1,.2],[.1,.1,.1]])
    spec={'bins':4,'weights':weights.ravel().tolist()}
    for j in (0,15,16,31,32,47,48,63):
        np.testing.assert_array_equal(audit.weights_for(spec,3,j),weights[j//16])


def test_shared_translation_preserves_every_bump_gap_and_has_no_empirical_recentering(sample):
    result=validate(sample);a=sample[0]
    for rep in result['stream']['representations']:
        np.testing.assert_allclose(a[rep+'_shared_upper']-a[rep+'_shared_lower'],
            a[rep+'_unchanged_upper']-a[rep+'_unchanged_lower'],rtol=0.,atol=2e-13)
    assert result['checks']['empirical_valuation_mean_subtracted'] is False


def test_full_rng_paths_cannot_be_mislabelled_as_a_selected_subset(sample):
    a,bundle,*_=sample
    # This guard is checked before constructing a pricing surrogate.
    a.update(absolute_path_indices=np.array([0,2,4095]),physical_time_grid=a['actual_time_grid'])
    with pytest.raises(ValueError,match='Full absolute RNG paths'):
        ORIGINAL_RNG_ANCHOR(a,bundle,123456,3,128)


@pytest.fixture(scope='module')
def five_stock_replay():
    """Independent old scalar replay at five CRN spots, only eight test paths."""
    from rvlab.pricing import RoughHestonParams,approximate_fractional_kernel
    from rvlab.pricing.rough_estimators import RoughEuropeanControlConfig
    from rvlab.pricing.rough_stopping import RoughAmericanConfig
    from rvlab.pricing.rough_stopping_fourier import train_fourier_stopping_policy
    from rvlab.experiments._pricing_rough_foundation import _market
    from rvlab.experiments._pricing_rough_price_budget import conditional_european_replay
    params=RoughHestonParams(.04,.045,1.5,.3,-.6,.15)
    kernel=approximate_fractional_kernel(.15,1.,4,min_time_ratio=.003)
    control=RoughEuropeanControlConfig(8,40.,128)
    config=RoughAmericanConfig(training_paths=8,valuation_paths=8,steps=128,exercise_steps=2,
        training_seed=22201,valuation_seed=22202,batch_size=512,
        stock_knots=(.9,1.,1.1),premium_control=control)
    proxy=train_fourier_stopping_policy(params,_market(),kernel,strike=100.,maturity=60/365,
        config=config,control_config=control,is_call=False,exercise_style='european')
    spots=proxy.market.spot*(1+RELATIVE);runs=[]
    for spot in spots:
        bumped=replace(proxy,market=replace(proxy.market,spot=float(spot)))
        c,_=conditional_european_replay(bumped,seed=22202,paths=8)
        runs.append(c)
    a={key:np.stack([c[key] for c in runs],axis=-1) for key in (
        'conditional_log_means','conditional_expected_next','conditional_actual_next',
        'conditional_european_payoff','conditional_european_martingale','conditional_european_estimate')}
    a.update(initial_spots=spots,absolute_path_indices=np.arange(8),
        physical_time_grid=runs[0]['physical_time_grid'],actual_time_grid=runs[0]['physical_time_grid'],
        physical_factors=runs[0]['replay_physical_factors'],
        physical_stocks=np.stack([c['replay_physical_stocks'] for c in runs],axis=-1),
        conditional_variances=runs[0]['conditional_variances'],times=proxy.exercise_times)
    indices=np.searchsorted(a['physical_time_grid'],a['times'])
    a['stocks']=a['physical_stocks'][:,indices];a['factors']=a['physical_factors'][:,indices]
    return a,{'proxy':proxy}


def test_all_five_gaussian_clocks_and_physical_paths_match_independent_scalar_replays(five_stock_replay):
    a,bundle=five_stock_replay
    estimate,probes,errors=ORIGINAL_RNG_ANCHOR(a,bundle,22202,8,128)
    np.testing.assert_allclose(estimate,a['conditional_european_estimate'],rtol=0.,atol=2e-12)
    assert len(probes)==4
    assert max(errors)<2e-12
    assert all(row['all_bumps'] for row in probes)


@pytest.mark.parametrize('key,index',[
    ('physical_stocks',(1,30,4)),('physical_factors',(2,40,0)),
    ('conditional_log_means',(3,50,2)),('conditional_variances',(4,60)),
    ('conditional_expected_next',(0,32,1)),('conditional_actual_next',(1,64,3)),
    ('conditional_european_payoff',(5,0)),('conditional_european_martingale',(6,1)),
    ('conditional_european_estimate',(7,2))])
def test_changed_multistock_rng_or_conditional_anchor_primitive_is_rejected(five_stock_replay,key,index):
    original,bundle=five_stock_replay;a={**original,key:original[key].copy()}
    a[key][index]+=.001
    with pytest.raises(ValueError):ORIGINAL_RNG_ANCHOR(a,bundle,22202,8,128)


def test_physical_reference_keeps_distinct_continuation_and_premium_stock_grids(monkeypatch):
    """The left exercise interpolant and right premium surrogate have different knots."""
    import rvlab.pricing.rough_stopping_fourier as fourier
    proxy=SimpleNamespace(strike=100.,is_call=False,premium_cap=.7,
        config=SimpleNamespace(stock_knots=(.9,1.,1.1)),
        continuations=(SimpleNamespace(knots=(.8,1.2)),),
        market=SimpleNamespace(discount_curve=SimpleNamespace(discount=lambda t:1.)))
    monkeypatch.setattr(fourier,'_interpolate',lambda p,t,f:
        np.zeros(np.shape(f)[:-1]+(len(p.config.stock_knots)+1,3)))
    monkeypatch.setattr(fourier,'_continuation_coefficients',lambda p,t,f:
        np.zeros(np.shape(f)[:-1]+(len(p.continuations[0].knots)+1,3)))
    monkeypatch.setattr(audit,'_affine_values',lambda affine,t,f,m,v:
        np.zeros(np.broadcast_shapes(np.shape(m),np.shape(v))))
    n=2;factors=np.zeros((n,4));stocks=np.full((n,5),100.)
    law=dict(new_factors=np.zeros((n,3,4)),probabilities=np.full((n,3),1/3),
        log_stock_shifts=np.zeros((n,3,2)),orthogonal=np.zeros((n,3,2)))
    value=audit._physical_reference(proxy,None,law,stocks,stocks,factors,factors,0.,.1)
    np.testing.assert_allclose(value,0.,atol=2e-13,rtol=0.)
    assert value.shape==(n,5,4)
