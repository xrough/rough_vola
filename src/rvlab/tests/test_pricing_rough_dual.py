"""Independent train/valuation firewalls and convex dual weighting checks."""
from dataclasses import replace
from types import MappingProxyType
import numpy as np
import pytest
from rvlab.pricing import RoughHestonParams,SpotMarket,DiscountCurve,DividendSchedule,approximate_fractional_kernel
from rvlab.pricing.rough_stopping import RoughAmericanConfig
from rvlab.pricing.rough_stopping_fourier import train_fourier_stopping_policy,value_fourier_stopping_policy
from rvlab.pricing.rough_dual import (RoughDualConfig,RoughDualWeights,train_rough_dual_weights,
    apply_rough_dual_weights,value_rough_dual_policy,_basis,_fit_lp)


def policy():
    model=RoughHestonParams(.04,.04,1.5,0.,-.6,.15)
    market=SpotMarket(100.,DiscountCurve.flat(.04,1.),DividendSchedule((),1.))
    kernel=approximate_fractional_kernel(.15,1.,2,min_time_ratio=.003)
    settings=RoughAmericanConfig(training_paths=48,valuation_paths=64,steps=4,
        exercise_steps=2,batch_size=32,training_seed=42,valuation_seed=43,
        stock_knots=(.95,1.,1.05))
    return train_fourier_stopping_policy(model,market,kernel,strike=100.,maturity=.2,
        is_call=False,config=settings,premium_cap=.7)


def test_sparse_gap_lp_recovers_known_optimal_coefficient():
    # At maturity each path pays |Z|, with first martingale basis Z.
    # The stopped policy exercises at maturity, making gap=max(0,w*Z-|Z|).
    # Premium basis is chosen so unit weights have a strictly positive gap
    # and coefficient zero is optimal. Exact known fixture avoids mirroring LP.
    n=20;z=np.linspace(-1,1,n)
    payoff=np.column_stack((np.zeros(n),np.ones(n)))
    basis=np.zeros((n,2,2));basis[:,1,1]=3*np.abs(z)
    stops=np.ones(n,dtype=int)
    settings=RoughDualConfig(training_paths=n,candidate_bins=(1,),european_bounds=(1.,1.),premium_bounds=(0.,3.))
    weights,record=_fit_lp(payoff,basis,stops,np.array([True,True]),settings)
    assert record['optimizer_success']
    assert record['sampled_objective']<1e-10
    assert weights[1]<=1/3+1e-9
    assert record['maximum_constraint_violation']<1e-8


def test_dual_weights_use_third_stream_and_freeze_coefficients():
    p=policy()
    settings=RoughDualConfig(training_paths=64,training_seed=44,candidate_bins=(1,2))
    fit=train_rough_dual_weights(p,config=settings)
    assert fit.diagnostics['selection_source']=='held-out dual-training paths only'
    assert fit.diagnostics['independent_policy_dual_valuation_streams']
    with pytest.raises(ValueError):fit.weights[0,0]=1.
    with pytest.raises(TypeError):fit.diagnostics['selected_candidate']='changed'
    result=value_rough_dual_policy(p,fit,spots=[99.75,100.,100.25],trace=True)
    assert np.all(result['dual_samples']>=result['primal_samples']-1e-12)
    assert result['diagnostics']['same_martingale_primal_dual']
    for seed in (42,43):
        with pytest.raises(ValueError,match='seed'):
            train_rough_dual_weights(p,config=replace(settings,training_seed=seed))
    with pytest.raises(ValueError,match='valuation seed'):
        apply_rough_dual_weights(value_fourier_stopping_policy(p,
            config=replace(p.config,valuation_seed=44),trace=True)['trace'],fit)


def test_unit_weights_reproduce_unweighted_endpoints_and_all_stock_bumps():
    p=policy();data=value_fourier_stopping_policy(p,spots=[99.75,100.,100.25],trace=True)
    metadata={'selected_candidate':'unit fixture'}
    fit=RoughDualWeights(np.array([0.,.2]),np.ones((1,2)),data['trace']['policy_fingerprint'],42,44,43,
        RoughDualConfig(training_paths=64,training_seed=44,candidate_bins=(1,)),metadata)
    weighted=apply_rough_dual_weights(data['trace'],fit)
    for name in ('primal_samples','dual_samples','raw_samples'):
        np.testing.assert_allclose(weighted[name],data[name],rtol=0,atol=2e-13)
    basis=_basis(data['trace'],np.array([0.,.1,.2]))
    np.testing.assert_allclose(basis.sum(axis=-1),data['trace']['european_martingale']+
        data['trace']['premium_martingale'],rtol=0,atol=2e-13)


def test_training_trace_identity_and_shape_firewalls():
    p=policy();settings=RoughDualConfig(training_paths=64,training_seed=44,candidate_bins=(1,))
    data=value_fourier_stopping_policy(p,config=replace(p.config,valuation_seed=44),trace=True)['trace']
    bad=dict(data);bad['policy_fingerprint']='another-policy'
    with pytest.raises(ValueError,match='does not match'):train_rough_dual_weights(p,config=settings,training_trace=bad)
    bad=dict(data);bad['stop_indices']=np.full((64,1),100)
    with pytest.raises(ValueError,match='stopping indices'):train_rough_dual_weights(p,config=settings,training_trace=bad)
    bad=dict(data);bad['european_martingale']=np.full((64,3,1),np.nan)
    with pytest.raises(ValueError,match='malformed'):train_rough_dual_weights(p,config=settings,training_trace=bad)


@pytest.mark.parametrize('kwargs',[{'training_paths':3},{'training_seed':-1},{'candidate_bins':()},
    {'candidate_bins':(1,1)},{'candidate_bins':(True,)},{'validation_fraction':0},
    {'validation_fraction':1},{'european_bounds':(.8,.9)},{'premium_bounds':(2,3)},
    {'solver_tolerance':0},{'objective':'unsupported'}])
def test_invalid_dual_settings(kwargs):
    with pytest.raises((ValueError,TypeError)):
        RoughDualConfig(**kwargs)


def test_predictable_feature_uses_left_state_and_preserves_martingale_mean():
    # Exhaust all four equally likely paths of a two-step random walk.
    shock=np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
    m=np.column_stack((np.zeros(4),np.cumsum(shock,axis=1)))[:,:,None]
    stocks=100+10*m
    data={'times':np.array([0.,1.,2.]),'european_martingale':m,
        'premium_martingale':np.zeros_like(m),'stocks':stocks,'strike':100.,
        'discounts':np.ones(3),'continuation_proxy':np.zeros_like(m),
        'discounted_payoffs':np.zeros_like(m)}
    basis=_basis(data,np.array([0.,2.]),('moneyness',))
    expected=shock[:,0]*shock[:,1]
    np.testing.assert_allclose(basis[:,-1,0,2],expected,atol=1e-14)
    assert abs(basis[:,-1,0,2].mean())<1e-14


def test_predictable_and_intrinsic_coefficients_remain_frozen_and_zero_mean():
    p=policy();settings=RoughDualConfig(training_paths=64,training_seed=44,candidate_bins=(1,2),
        predictable_features=('moneyness','exercise_region'),
        components=('european','premium','intrinsic_minus_european'),european_bounds=(1.,1.))
    fit=train_rough_dual_weights(p,config=settings)
    assert fit.weights.shape[1]==9
    value=value_rough_dual_policy(p,fit,trace=True)
    assert np.all(value['dual_samples']>=value['primal_samples']-1e-12)
    assert np.all(fit.weights[:,0]==1.)
    bad=dict(value['trace']);bad['additional_control_training_seeds']=(43,)
    with pytest.raises(ValueError,match='additional control training'):apply_rough_dual_weights(bad,fit)


def test_error_budget_projection_penalizes_control_variance_without_valuation():
    from rvlab.pricing.rough_dual import _selection_record
    primal=np.array([1.,1.,0.,2.,0.,2.])[:,None]
    dual=primal+.01
    result=_selection_record(primal,dual,2,400)
    assert result['heldout_gap']==pytest.approx(.01)
    assert result['heldout_error_budget']>.01
    # Only held-out variation and the declared future path count enter.
    expected=.01+1.96*np.std([0.,2.,0.,2.],ddof=1)/20
    assert result['heldout_error_budget']==pytest.approx(expected)


def test_optimizer_exhaustion_remains_visible_and_unit_fallback_is_available(monkeypatch):
    from types import SimpleNamespace
    import rvlab.pricing.rough_dual as module
    p=policy()
    monkeypatch.setattr(module,'linprog',lambda *a,**kw:SimpleNamespace(
        success=False,status=1,message='iteration limit',nit=1,x=None))
    fit=module.train_rough_dual_weights(p,config=RoughDualConfig(
        training_paths=64,training_seed=44,candidate_bins=(1,2)))
    assert fit.diagnostics['selected_candidate']=='unit baseline'
    failed=[r for r in fit.diagnostics['candidates'] if r['status']=='optimizer_failure']
    assert len(failed)==2
    assert all(r['optimizer_status']==1 for r in failed)


def test_memory_guard_rejects_larger_lp_before_allocation_and_preserves_fallback():
    from rvlab.pricing.rough_dual import _dual_workspace_estimate
    from rvlab.pricing.rough_stopping_fourier import fourier_policy_fingerprint
    p=policy()
    data=dict(value_fourier_stopping_policy(p,config=replace(p.config,valuation_seed=44),trace=True)['trace'])
    settings=RoughDualConfig(training_paths=64,training_seed=44,candidate_bins=(1,2))
    small=_dual_workspace_estimate(data,48,2);large=_dual_workspace_estimate(data,48,4)
    assert small<large
    limited=replace(p,config=replace(p.config,memory_budget_mb=(small+large)/2/1024**2))
    # The cached decision/control functions are identical; only the declared
    # solver memory budget changes in this constructed fixture.
    data['policy_fingerprint']=fourier_policy_fingerprint(limited)
    fit=train_rough_dual_weights(limited,config=settings,training_trace=data)
    records=fit.diagnostics['candidates']
    assert next(r for r in records if r['bins']==1 and r['status']!='baseline')['status']=='completed'
    rejected=next(r for r in records if r['bins']==2)
    assert rejected['status']=='numerical_failure'
    assert 'memory_budget_mb' in rejected['error']
    assert rejected['projected_memory_bytes']>rejected['memory_budget_bytes']
    assert records[0]['status']=='baseline'


def test_dispersion_regularizer_finds_constant_stopped_cashflow_in_known_fixture():
    z=np.linspace(-1.,1.,20);payoff=np.column_stack((np.zeros(20),1+z))
    basis=np.zeros((20,2,2));basis[:,1,1]=z
    stops=np.ones(20,dtype=int)
    config=RoughDualConfig(training_paths=20,european_bounds=(1,1),premium_bounds=(0,3))
    weights,record=_fit_lp(payoff,basis,stops,np.ones(2,dtype=bool),config,.1)
    assert record['optimizer_success']
    assert weights[1]==pytest.approx(1.,abs=1e-9)
    assert record['fitted_stopped_cashflow_center']==pytest.approx(1.,abs=1e-9)
    assert record['fitted_stopped_cashflow_absolute_deviation']<1e-9
    assert record['sampled_objective']<1e-9
    assert record['constraint_count']==80
    assert record['variable_count']==43


@pytest.mark.parametrize('penalties',[(),(-.1,),(float('nan'),),(float('inf'),),(True,),(.1,.1)])
def test_invalid_dispersion_penalties(penalties):
    with pytest.raises(ValueError,match='dispersion_penalties'):
        RoughDualConfig(dispersion_penalties=penalties)


def test_regularizer_candidates_are_selected_only_on_training_holdout():
    p=policy();settings=RoughDualConfig(training_paths=64,training_seed=44,candidate_bins=(1,),
        dispersion_penalties=(0.,.1),predictable_features=('soft_exercise_region_narrow',
            'soft_exercise_region_medium','soft_exercise_region_broad','soft_exercise_region_wide'))
    fit=train_rough_dual_weights(p,config=settings)
    assert len(fit.diagnostics['candidates'])==3
    assert {r['dispersion_penalty'] for r in fit.diagnostics['candidates'][1:]}=={0.,.1}
    assert fit.diagnostics['selection_source']=='held-out dual-training paths only'
    assert fit.weights.shape[1]==10


def test_policy_stopped_feature_is_predictable_and_has_no_stopped_increment():
    # At t=1 the policy stops exactly when the first random-walk shock is +1.
    shock=np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
    m=np.column_stack((np.zeros(4),np.cumsum(shock,axis=1)))[:,:,None]
    stops=np.where(shock[:,0]>0,1,2)[:,None]
    data={'times':np.array([0.,1.,2.]),'european_martingale':m,
        'premium_martingale':np.zeros_like(m),'stocks':100+10*m,'strike':100.,
        'discounts':np.ones(3),'continuation_proxy':np.zeros_like(m),
        'discounted_payoffs':np.zeros_like(m),'stop_indices':stops}
    B=_basis(data,np.array([0.,2.]),('policy_stopped',))
    # Feature is zero until the policy's stopping event is known. It cannot
    # alter the martingale at its stopping time, hence preserves primal paths.
    np.testing.assert_array_equal(B[np.arange(4),stops[:,0],0,2],0.)
    np.testing.assert_array_equal(B[:,-1,0,2],[0.,0.,-1.,1.])
    assert B[:,-1,0,2].mean()==0.


def test_micro_boundary_component_is_used_directly_and_validated():
    p=policy();data=dict(value_fourier_stopping_policy(p,trace=True)['trace'])
    data['micro_boundary_martingale']=.2*data['european_martingale']
    B=_basis(data,np.array([0.,.2]),(),('european','premium','micro_boundary'))
    np.testing.assert_array_equal(B[:,:,:,2],data['micro_boundary_martingale'])
    data['micro_boundary_martingale']=np.full_like(data['european_martingale'],np.nan)
    with pytest.raises(ValueError,match='malformed micro_boundary'):
        _basis(data,np.array([0.,.2]),(),('european','premium','micro_boundary'))


@pytest.mark.parametrize('third',['micro_boundary','intrinsic_minus_european'])
def test_four_component_models_train_and_apply_with_separate_poststop_bounds(third):
    p=policy();config=RoughDualConfig(training_paths=64,training_seed=44,candidate_bins=(1,),
        components=('european','premium',third,'micro_poststop'),
        european_bounds=(1.,1.),premium_bounds=(1.,1.),intrinsic_bounds=(0.,.25),
        poststop_bounds=(0.,.75),predictable_features=('policy_stopped',),
        european_feature_bounds=(0,0),premium_feature_bounds=(0,0),
        intrinsic_feature_bounds=(0,0),poststop_feature_bounds=(0,0))
    fit=train_rough_dual_weights(p,config=config)
    assert fit.weights.shape[1]==8
    assert np.all((fit.weights[:,2]>=0)&(fit.weights[:,2]<=.25))
    assert np.all((fit.weights[:,3]>=0)&(fit.weights[:,3]<=.75))
    assert np.all(fit.weights[:,4:]==0.)
    out=value_rough_dual_policy(p,fit,spots=[99.75,100.,100.25],trace=True)
    assert np.all(out['dual_samples']>=out['primal_samples']-1e-12)


def test_poststop_control_cannot_change_controlled_lower_cashflows():
    p=policy();result=value_fourier_stopping_policy(p,spots=[99.,100.,101.],trace=True)
    data=result['trace'];settings=RoughDualConfig(training_paths=64,training_seed=44,candidate_bins=(1,),
        components=('european','premium','micro_poststop'))
    fit=RoughDualWeights(np.array([0.,.2]),np.array([[1.,1.,1.]]),
        data['policy_fingerprint'],42,44,43,settings,{'selected_candidate':'fixed post-stop fixture'})
    out=apply_rough_dual_weights(data,fit)
    # Summing a cumulative basis rather than direct increments may introduce
    # a few ulps. The independent post-stop component itself is exactly zero.
    stops=data['stop_indices'];a=np.arange(len(stops))[:,None];b=np.arange(3)[None,:]
    np.testing.assert_array_equal(data['micro_poststop_martingale'][a,stops,b],0.)
    np.testing.assert_allclose(out['primal_samples'],result['primal_samples'],atol=2e-13,rtol=0.)


@pytest.mark.parametrize('components',[('european','premium','micro_poststop'),
    ('european','premium','micro_boundary','micro_poststop')])
@pytest.mark.parametrize('axis',[None,'time','joint'])
def test_public_poststop_dual_preserves_five_bump_price_delta_samples_and_refinement(components,axis):
    from rvlab.pricing import evaluate_vanilla
    model=RoughHestonParams(.04,.045,1.5,.3,-.6,.15)
    market=SpotMarket(100.,DiscountCurve.flat(.04,1.),DividendSchedule((),1.))
    kernel=approximate_fractional_kernel(.15,1.,2,min_time_ratio=.003)
    dual=RoughDualConfig(training_paths=48,training_seed=44,candidate_bins=(1,2),
        components=components,european_bounds=(1,1),premium_bounds=(1,1),
        intrinsic_bounds=(0,0),poststop_bounds=(0,1.5))
    config=RoughAmericanConfig(training_paths=64,valuation_paths=128,steps=8,exercise_steps=4,
        batch_size=32,training_seed=42,valuation_seed=43,stock_knots=(.9,.95,1.,1.05,1.1),
        training_method='fourier_premium',dual_config=dual)
    if axis is not None:config=config.refined(axis)
    assert config.dual_config is dual
    options=dict(market=market,kernel=kernel,strike=100.,maturity=.2,is_call=False,
        exercise_style='american')
    baseline=evaluate_vanilla(model,config=replace(config,dual_config=None),**options)
    weighted=evaluate_vanilla(model,config=config,**options)
    for name in ('prices','standard_errors','deltas','delta_standard_errors',
            'bump_deltas','bump_delta_standard_errors','one_sided_deltas',
            'one_sided_delta_standard_errors','raw_prices','raw_standard_errors',
            'exercise_probabilities'):
        np.testing.assert_allclose(getattr(weighted,name),getattr(baseline,name),atol=3e-12,rtol=0.)
    record=weighted.diagnostics['contracts'][0]['dual_training']
    assert weighted.diagnostics['dual_training_evaluations']==1
    assert record['target_valuation_paths']==config.valuation_paths
    assert record['components']==list(components)
    assert len(record['candidates'])==3
    assert all(r['status'] in ('baseline','completed') for r in record['candidates'])
    assert record['independent_policy_dual_valuation_streams']
    assert np.isfinite(weighted.dual_prices)
    assert not weighted.diagnostics['american_pricing_ready']
    assert not weighted.diagnostics['stock_delta_ready']
