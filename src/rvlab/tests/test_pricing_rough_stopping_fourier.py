"""Exact centering and independent-stream checks for residual-premium controls."""
from dataclasses import replace
import numpy as np
import pytest
from rvlab.pricing import (RoughHestonParams, BlackScholesParams,
    approximate_fractional_kernel, price, CashDividend, DividendSchedule,
    SpotMarket, DiscountCurve)
from rvlab.pricing.rough_stopping import RoughAmericanConfig
from rvlab.pricing.rough_stopping_fourier import (
    train_fourier_stopping_policy, value_fourier_stopping_policy)
from rvlab.pricing.vanilla_types import UnsupportedModelError


def inputs():
    model=RoughHestonParams(.04,.04,1.5,0.,-.6,.15)
    market=SpotMarket(100.,DiscountCurve.flat(.04,1.),DividendSchedule((),1.))
    kernel=approximate_fractional_kernel(.15,1.,2,min_time_ratio=.003)
    config=RoughAmericanConfig(training_paths=48,valuation_paths=64,steps=4,
        exercise_steps=2,batch_size=32,training_seed=42,valuation_seed=43,
        stock_knots=(.95,1.,1.05))
    return model,market,kernel,config


@pytest.mark.parametrize('is_call',[True,False])
def test_fourier_control_constant_variance_european_is_analytic(is_call):
    model,market,kernel,config=inputs()
    policy=train_fourier_stopping_policy(model,market,kernel,strike=100.,
        maturity=.2,is_call=is_call,config=config,exercise_style='european')
    spots=np.array([99.75,100.,100.25])
    result=value_fourier_stopping_policy(policy,spots=spots)
    discount=market.discount_curve.discount(.2)
    exact=price(BlackScholesParams(.2),forward=spots/discount,discount=discount,
        strike=100.,maturity=.2,is_call=is_call)
    # In this exact-lognormal limit the European baseline has exact semigroup
    # conditional expectations: every controlled path is the same price.
    np.testing.assert_allclose(result['primal_samples'],
        np.broadcast_to(exact,(config.valuation_paths,3)),atol=2e-10,rtol=0)
    np.testing.assert_allclose(result['dual_samples'],result['primal_samples'],atol=2e-10,rtol=0)
    assert result['diagnostics']['independent_training_valuation']


def test_fourier_control_primal_dual_order_and_stream_reproducibility():
    model,market,kernel,config=inputs()
    model=replace(model,vol_of_vol=.3)
    policy=train_fourier_stopping_policy(model,market,kernel,strike=100.,maturity=.2,
        is_call=False,config=config)
    result=value_fourier_stopping_policy(policy)
    repeated=value_fourier_stopping_policy(policy)
    np.testing.assert_array_equal(result['primal_samples'],repeated['primal_samples'])
    assert np.all(result['dual_samples']>=result['primal_samples']-1e-12)
    with pytest.raises(ValueError,match='independent'):
        value_fourier_stopping_policy(policy,config=replace(config,valuation_seed=42,training_seed=44))
    changed=value_fourier_stopping_policy(policy,config=replace(config,valuation_seed=44))
    assert not np.array_equal(result['primal_samples'],changed['primal_samples'])


@pytest.mark.parametrize('spots',[[0.],[-1.],[np.nan],[],[[100.]],['100']])
def test_fourier_control_rejects_invalid_paired_stocks(spots):
    model,market,kernel,config=inputs()
    policy=train_fourier_stopping_policy(model,market,kernel,strike=100.,maturity=.2,config=config)
    with pytest.raises(ValueError,match='spots'):
        value_fourier_stopping_policy(policy,spots=spots)


def test_fourier_control_explicit_cash_exclusion_and_memory_guard():
    model,market,kernel,config=inputs()
    market=replace(market,dividends=DividendSchedule((CashDividend(.1,1.),),1.))
    with pytest.raises(UnsupportedModelError,match='cash'):
        train_fourier_stopping_policy(model,market,kernel,strike=100.,maturity=.2,config=config)
    with pytest.raises(MemoryError):
        train_fourier_stopping_policy(model,replace(market,dividends=DividendSchedule((),1.)),
            kernel,strike=100.,maturity=.2,config=replace(config,memory_budget_mb=.001))


def test_fourier_policies_expiration_and_immutability():
    from rvlab.pricing.rough_stopping_fourier import train_fourier_bellman_policy
    model,market,kernel,config=inputs()
    for train in (train_fourier_stopping_policy,train_fourier_bellman_policy):
        for style in ('american','european'):
            policy=train(model,market,kernel,strike=105.,maturity=0.,config=config,exercise_style=style)
            result=value_fourier_stopping_policy(policy,spots=[99.,100.,101.])
            np.testing.assert_array_equal(result['primal_samples'],np.tile([6.,5.,4.],(64,1)))
            np.testing.assert_array_equal(result['dual_samples'],result['primal_samples'])
            with pytest.raises(ValueError): policy.continuation_times[0]=1.
            with pytest.raises(TypeError): policy.diagnostics['changed']=True


def test_fourier_bellman_microsteps_do_not_enable_extra_exercise():
    from rvlab.pricing.rough_stopping_fourier import train_fourier_bellman_policy
    model,market,kernel,config=inputs()
    policy=train_fourier_bellman_policy(model,market,kernel,strike=100.,maturity=.2,
        config=config,premium_cap=.7)
    result=value_fourier_stopping_policy(policy)
    assert len(policy.continuation_times)>len(policy.exercise_times)
    assert set(np.unique(result['stop_times'])).issubset(set(policy.exercise_times))
    assert np.all(result['dual_samples']>=result['primal_samples']-1e-12)


@pytest.mark.parametrize('cap',[0.,-1.,np.inf,np.nan,True,'x',[]])
def test_fourier_invalid_premium_caps(cap):
    model,market,kernel,config=inputs()
    with pytest.raises(ValueError,match='premium_cap'):
        train_fourier_stopping_policy(model,market,kernel,strike=100.,maturity=.2,
            config=config,premium_cap=cap)


@pytest.mark.parametrize('method',['fourier_premium','fourier_bellman'])
def test_public_dispatch_prices_broadcast_expiration_and_stock_delta(method):
    from rvlab.pricing import evaluate_vanilla
    model,market,kernel,config=inputs()
    config=replace(config,training_method=method)
    result=evaluate_vanilla(model,market=market,kernel=kernel,strike=[95.,100.,105.],
        maturity=[0.,.2,.2],is_call=[False,True,False],exercise_style='european',config=config)
    discount=market.discount_curve.discount(np.array([0.,.2,.2]))
    exact=price(BlackScholesParams(.2),forward=100/discount,discount=discount,
        strike=[95.,100.,105.],maturity=[0.,.2,.2],is_call=[False,True,False])
    np.testing.assert_allclose(result.prices,exact,atol=3e-9,rtol=0)
    assert result.prices.shape==(3,)
    assert result.one_sided_deltas.shape==(3,2)
    assert np.max(result.standard_errors)<3e-10
    assert np.max(result.delta_standard_errors)<3e-9
    assert result.settings['training_method']==method
    assert not result.diagnostics['american_pricing_ready']
    # Parameters and the empty dividend declaration stay fixed under bumps.
    bump=100*config.bump_relative/2
    def value(stock):
        return price(BlackScholesParams(.2),forward=stock/discount,discount=discount,
            strike=[95.,100.,105.],maturity=[0.,.2,.2],is_call=[False,True,False])
    np.testing.assert_allclose(result.deltas,(value(100+bump)-value(100-bump))/(2*bump),atol=2e-8,rtol=0)


@pytest.mark.parametrize('kwargs',[
    {'premium_cap':0.}, {'premium_cap':True}, {'premium_cap':np.nan},
    {'premium_control':{}}, {'premium_control':None},
])
def test_public_config_rejects_invalid_premium_settings(kwargs):
    with pytest.raises((ValueError,TypeError)):
        RoughAmericanConfig(**kwargs)


def test_fourier_conditional_expectation_matches_independent_scalar_gaussian_integral():
    from scipy.integrate import quad
    from rvlab.pricing.rough_estimators import _AffineSurrogate, RoughEuropeanControlConfig
    from rvlab.pricing.rough_stopping_fourier import _affine_values
    model,market,kernel,_=inputs()
    model=replace(model,vol_of_vol=.3)
    grid=np.array([0.,.1,.2])
    proxy=_AffineSurrogate(model,market,kernel,np.array([100.]),np.array([.2]),
        np.array([False]),grid,RoughEuropeanControlConfig(24,100.,512))
    factors=np.array([[[.001,-.0002],[.0004,.0003]],[[.0002,.0001],[-.0001,.0005]]])
    mean=np.log(np.array([[[[98.,101.],[99.,102.]],[[97.,103.],[100.,104.]]],
        [[[96.,100.],[98.,101.]],[[99.,105.],[102.,106.]]]]))
    clock=np.full(mean.shape,.002)
    actual=_affine_values(proxy,.1,factors,mean,clock)
    # Independent scalar integration also catches branch/bump-axis reshaping
    # defects that identical vectorized actual/conditional code could share.
    for index in ((0,0,1,0),(0,1,0,1),(1,0,0,1),(1,1,1,0)):
        path,branch,order,bump=index
        state=factors[path,branch][None,None,:]
        def integrand(z):
            logspot=mean[index]+np.sqrt(clock[index])*z
            value=proxy.values(.1,state,np.full((1,1,1),logspot),
                np.zeros((1,1,1)),np.array([0])).item()
            return value*np.exp(-.5*z*z)/np.sqrt(2*np.pi)
        reference=quad(integrand,-10.,10.,epsabs=2e-10,limit=100)[0]
        assert actual[index]==pytest.approx(reference,abs=3e-10)


def test_public_dual_opt_in_preserves_exact_limit_and_stream_firewall():
    from rvlab.pricing import RoughDualConfig, evaluate_vanilla
    model,market,kernel,config=inputs()
    dual=RoughDualConfig(training_paths=32,training_seed=44,candidate_bins=(1,),
        european_bounds=(1.,1.),premium_bounds=(1.,1.))
    config=replace(config,training_method='fourier_premium',dual_config=dual,retrain_bumps=True)
    result=evaluate_vanilla(model,market=market,kernel=kernel,strike=100.,maturity=.2,
        is_call=False,exercise_style='european',config=config)
    discount=market.discount_curve.discount(.2)
    exact=price(BlackScholesParams(.2),forward=100/discount,discount=discount,
        strike=100.,maturity=.2,is_call=False)
    assert result.prices==pytest.approx(exact,abs=2e-9)
    assert result.standard_errors<2e-10
    assert result.delta_standard_errors<2e-9
    assert result.diagnostics['dual_training_evaluations']==3
    assert result.diagnostics['contracts'][0]['dual_training']['dual_training_seed']==44
    bumped=result.diagnostics['contracts'][0]['retrained_bump_dual_training']
    assert [item['spot'] for item in bumped]==[99.75,100.25]
    assert all(item['dual_training']['dual_training_seed']==44 for item in bumped)
    assert all(item['dual_training']['candidates'] for item in bumped)
    assert result.retrained_delta_standard_errors<2e-9
    assert not result.diagnostics['american_pricing_ready']
    for seed in (42,43):
        with pytest.raises(ValueError,match='seed'):
            replace(config,dual_config=replace(dual,training_seed=seed))
    with pytest.raises(ValueError,match='Fourier'):
        replace(config,training_method='conditional')
    with pytest.raises(TypeError,match='dual_config'):
        replace(config,dual_config={})


def test_frozen_policy_trace_reconstructs_exact_primal_dual_without_changing_paths():
    model,market,kernel,config=inputs()
    model=replace(model,vol_of_vol=.3)
    policy=train_fourier_stopping_policy(model,market,kernel,strike=100.,maturity=.2,
        is_call=False,config=config,premium_cap=.7)
    normal=value_fourier_stopping_policy(policy,spots=[99.75,100.,100.25])
    traced=value_fourier_stopping_policy(policy,spots=[99.75,100.,100.25],trace=True)
    for name in ('primal_samples','dual_samples','raw_samples','stop_times'):
        np.testing.assert_array_equal(normal[name],traced[name])
    data=traced['trace'];pay=data['discounted_payoffs']
    total=data['european_martingale']+data['premium_martingale']
    index=data['stop_indices'];paths=np.arange(len(index))[:,None]
    bumps=np.arange(index.shape[1])[None,:]
    raw=pay[paths,index,bumps]
    primal=raw-total[paths,index,bumps]
    dual=np.max(np.where(data['eligible'][None,:,None],pay-total,-np.inf),axis=1)
    np.testing.assert_allclose(raw,traced['raw_samples'],atol=0,rtol=0)
    np.testing.assert_allclose(primal,traced['primal_samples'],atol=1e-14,rtol=0)
    np.testing.assert_allclose(dual,traced['dual_samples'],atol=1e-14,rtol=0)
    assert data['factors'].shape==(64,len(data['times']),2)
    np.testing.assert_allclose(data['european_proxy'][:,-1],pay[:,-1],atol=1e-13)
    with pytest.raises(ValueError): data['stocks'][0,0,0]=101.
    with pytest.raises(TypeError): data['new']='bad'


def test_extra_control_training_seed_is_independent_from_valuation():
    model,market,kernel,config=inputs()
    policy=train_fourier_stopping_policy(model,market,kernel,strike=100.,maturity=.2,config=config)
    enriched=replace(policy,diagnostics={**dict(policy.diagnostics),'additional_control_training_seeds':(44,)})
    with pytest.raises(ValueError,match='additional control training'):
        value_fourier_stopping_policy(enriched,config=replace(config,valuation_seed=44))
    assert value_fourier_stopping_policy(enriched,trace=True)['trace']['additional_control_training_seeds']==(44,)


def test_one_step_intrinsic_martingale_uses_exact_log_gaussian_centering():
    model,market,kernel,config=inputs()
    config=replace(config,steps=1,exercise_steps=1)
    policy=train_fourier_stopping_policy(model,market,kernel,strike=100.,maturity=.2,
        is_call=False,config=config,exercise_style='european')
    result=value_fourier_stopping_policy(policy,trace=True)
    data=result['trace'];discount=market.discount_curve.discount(.2)
    exact=float(price(BlackScholesParams(.2),forward=market.spot/discount,
        discount=discount,strike=100.,maturity=.2,is_call=False))
    np.testing.assert_allclose(data['intrinsic_martingale'][:,-1,0],
        result['raw_samples'][:,0]-exact,atol=2e-13,rtol=0)


def test_micro_boundary_control_uses_the_left_state_and_preserves_cashflows():
    from scipy.special import expit
    model,market,kernel,config=inputs()
    model=replace(model,vol_of_vol=.3)
    config=replace(config,steps=1,exercise_steps=1)
    policy=train_fourier_stopping_policy(model,market,kernel,strike=100.,maturity=.2,
        is_call=False,config=config,exercise_style='european')
    spots=np.array([90.,100.,110.])
    plain=value_fourier_stopping_policy(policy,spots=spots)
    traced=value_fourier_stopping_policy(policy,spots=spots,trace=True)
    data=traced['trace']
    for name in ('primal_samples','dual_samples','raw_samples','stop_times'):
        np.testing.assert_array_equal(plain[name],traced[name])
    # At a single integration step, the entire multiplier is known at zero.
    # A right-state indicator would be random and violate this identity.
    initial_payoff=np.maximum(100.-spots,0.)
    multiplier=expit((initial_payoff-data['continuation_proxy'][:,0])/.05)
    expected=multiplier*(data['intrinsic_martingale'][:,-1]
        -data['european_martingale'][:,-1])
    np.testing.assert_allclose(data['micro_boundary_martingale'][:,-1],expected,
        atol=2e-13,rtol=0)
    assert data['micro_boundary_width']==.05
    # This European policy stops only at expiry: no earlier stopping event
    # may be inferred from its known FUTURE terminal stop index.
    np.testing.assert_array_equal(data['micro_poststop_martingale'],0.)
    with pytest.raises(ValueError):data['micro_boundary_martingale'][0,0,0]=1.


def test_micro_poststop_control_is_zero_at_every_frozen_policy_stop():
    model,market,kernel,config=inputs()
    model=replace(model,vol_of_vol=.3)
    config=replace(config,steps=16,exercise_steps=8,valuation_paths=256)
    policy=train_fourier_stopping_policy(model,market,kernel,strike=100.,maturity=.2,
        is_call=False,config=config)
    result=value_fourier_stopping_policy(policy,spots=[95.,100.,105.],trace=True)
    data=result['trace'];indices=data['stop_indices'];micro=data['micro_poststop_martingale']
    np.testing.assert_array_equal(micro[np.arange(len(indices))[:,None],indices,
        np.arange(indices.shape[1])[None,:]],0.)
    assert np.any(indices<len(data['times'])-1)
    assert np.any(micro[:,-1]!=0)


@pytest.mark.parametrize('is_call,rate',[(False,.04),(True,-.01)])
def test_stochastic_one_step_intrinsic_center_matches_scalar_six_case_integration(is_call,rate):
    """Cross-check the branch/order axes against independent Gaussian integrals.

    Other simulator tests validate the shared weak factor transition. Here
    scalar integration, with a separate loop over all three variance outcomes
    and both independent-stock orders, tests its use in the new control.
    """
    import math
    from scipy.integrate import quad
    from rvlab.pricing.rough_simulation import _drift_flow_with_integral,_weak_transition_branches
    model,market,kernel,config=inputs()
    model=replace(model,theta=.045,vol_of_vol=.3)
    market=replace(market,discount_curve=DiscountCurve.flat(rate,1.))
    config=replace(config,steps=1,exercise_steps=1)
    maturity=.08;strike=100.;spots=np.array([93.,100.,107.])
    policy=train_fourier_stopping_policy(model,market,kernel,strike=strike,maturity=maturity,
        is_call=is_call,config=config,exercise_style='european')
    result=value_fourier_stopping_policy(policy,spots=spots,trace=True)
    data=result['trace']
    observed_center=result['raw_samples']-data['intrinsic_martingale'][:,-1]
    nodes=np.asarray(kernel.nodes);weights=np.asarray(kernel.weights);count=len(nodes)
    matrix=-np.diag(nodes)-model.kappa*np.ones((count,1))*weights
    exact=_drift_flow_with_integral(matrix,np.full(count,model.kappa*(model.theta-model.v0)),
        weights,model.v0,maturity/2)
    discount=market.discount_curve.discount(maturity)
    law=_weak_transition_branches(np.zeros((1,count)),model,nodes=nodes,weights=weights,
        flow=exact[:count,:count],shift=exact[:count,-1],integrated_flow=exact,
        dt=maturity,theta=model.theta,growth=-math.log(discount),stock_integral='exact_drift')
    references=[]
    for spot in spots:
        total=0.
        for branch in range(3):
            for order in range(2):
                clock=float(law['orthogonal'][0,branch,order]);sigma=math.sqrt(clock)
                mean=math.log(spot)+float(law['correlated'][0,branch])-.5*clock
                threshold=(math.log(strike)-mean)/sigma
                def integrand(z):
                    stock=math.exp(mean+sigma*z)
                    payoff=max(stock-strike if is_call else strike-stock,0.)
                    return payoff*math.exp(-.5*z*z)/math.sqrt(2*math.pi)
                lower=max(-12.,threshold) if is_call else -12.
                upper=12. if is_call else min(12.,threshold)
                integral=0. if upper<=lower else quad(integrand,lower,upper,epsabs=2e-12,epsrel=2e-12)[0]
                total+=float(law['probabilities'][0,branch])*.5*integral
        references.append(discount*total)
    np.testing.assert_allclose(observed_center,np.broadcast_to(references,observed_center.shape),
        atol=3e-11,rtol=0)
