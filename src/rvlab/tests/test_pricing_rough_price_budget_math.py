"""Independent Gaussian, replay and shared-bracket control mathematics."""
from dataclasses import replace
import math
import numpy as np
import pytest
from scipy.integrate import quad

from rvlab.pricing import RoughHestonParams,approximate_fractional_kernel
from rvlab.pricing.rough_estimators import RoughEuropeanControlConfig
from rvlab.pricing.rough_stopping import RoughAmericanConfig
from rvlab.pricing.rough_stopping_fourier import train_fourier_stopping_policy,value_fourier_stopping_policy
from rvlab.experiments._pricing_rough_foundation import _market
from rvlab.experiments._pricing_rough_price_budget import conditional_european_replay,shared_european_control
from rvlab.experiments._pricing_rough_price_budget_samples import validate_conditional_european,validate_common_shift


@pytest.fixture(scope='module')
def replay_case():
    params=RoughHestonParams(.04,.045,1.5,.3,-.6,.15)
    kernel=approximate_fractional_kernel(.15,1.,2,min_time_ratio=.003)
    control=RoughEuropeanControlConfig(8,40.,128)
    config=RoughAmericanConfig(training_paths=8,valuation_paths=8,steps=128,exercise_steps=2,
        training_seed=22201,valuation_seed=22202,batch_size=4,stock_knots=(.9,1.,1.1),premium_control=control)
    proxy=train_fourier_stopping_policy(params,_market(),kernel,strike=100.,maturity=60/365,
        config=config,control_config=control,is_call=False,exercise_style='european')
    trace=value_fourier_stopping_policy(proxy,trace=True)['trace']
    conditional,diagnostics=conditional_european_replay(proxy,seed=config.valuation_seed,paths=8,reference=trace)
    arrays={**conditional,'stocks':trace['stocks'],'factors':trace['factors'],
        'actual_time_grid':trace['actual_time_grid'],'payoff':trace['discounted_payoffs'][:,:,0],
        'european_martingale':trace['european_martingale'][:,:,0]}
    return proxy,trace,conditional,diagnostics,arrays


def test_replay_and_independent_whole_stream_reconstruction(replay_case):
    proxy,trace,conditional,diagnostics,arrays=replay_case
    validated=validate_conditional_european(arrays,{'proxy':proxy},seed=22202,paths=8)
    assert diagnostics['max_full_stream_stock_error']<2e-12
    assert diagnostics['max_full_stream_factor_error']<2e-13
    assert len(validated['probes'])==4 and all(not r['exhaustive_affine_recomputation'] for r in validated['probes'])
    np.testing.assert_allclose(validated['shared_european_control'],shared_european_control(arrays,conditional),rtol=0.,atol=2e-12)


def test_subset_never_changes_rng_batch_or_absolute_path_identity(replay_case):
    proxy,trace,conditional,_,_=replay_case;indices=np.array([1,5,7])
    subset,diag=conditional_european_replay(proxy,seed=22202,paths=8,indices=indices,reference=trace)
    assert diag['all_random_draws_consumed'] and diag['full_stream_paths']==8
    for key in ('conditional_log_means','conditional_variances','conditional_european_payoff',
            'conditional_european_martingale','conditional_european_estimate'):
        np.testing.assert_allclose(subset[key],conditional[key][indices],rtol=0.,atol=2e-12)
    np.testing.assert_array_equal(subset['replay_physical_stocks'],conditional['replay_physical_stocks'][indices])
    np.testing.assert_array_equal(subset['replay_physical_factors'],conditional['replay_physical_factors'][indices])


def test_terminal_conditional_payoff_against_scalar_normal_integration(replay_case):
    proxy,_,conditional,_,_=replay_case;discount=proxy.market.discount_curve.discount(proxy.maturity)
    for path in (0,3,7):
        mean=conditional['conditional_log_means'][path,-1];clock=conditional['conditional_variances'][path,-1]
        kink=(math.log(proxy.strike)-mean)/math.sqrt(clock)
        points=sorted({-12.,12.,max(-12.,min(12.,kink))})
        def integrand(z):
            stock=math.exp(mean+math.sqrt(clock)*z)
            return max(proxy.strike-stock,0.)*math.exp(-.5*z*z)/math.sqrt(2*math.pi)
        expected=discount*sum(quad(integrand,a,b,epsabs=1e-12,epsrel=1e-12)[0] for a,b in zip(points[:-1],points[1:]))
        assert conditional['conditional_european_payoff'][path]==pytest.approx(expected,abs=2e-11)


def test_zero_orthogonal_noise_collapses_both_european_controls(replay_case):
    proxy,_,_,_,_=replay_case;proxy=replace(proxy,params=replace(proxy.params,rho=-1.))
    trace=value_fourier_stopping_policy(proxy,trace=True)['trace']
    conditional,_=conditional_european_replay(proxy,seed=22202,paths=8,reference=trace)
    arrays={'payoff':trace['discounted_payoffs'][:,:,0],
        'european_martingale':trace['european_martingale'][:,:,0]}
    np.testing.assert_array_equal(conditional['conditional_variances'],np.zeros_like(conditional['conditional_variances']))
    np.testing.assert_allclose(conditional['conditional_european_payoff'],arrays['payoff'][:,-1],rtol=0.,atol=2e-10)
    np.testing.assert_allclose(conditional['conditional_european_martingale'],arrays['european_martingale'][:,-1],rtol=0.,atol=2e-10)
    np.testing.assert_allclose(shared_european_control(arrays,conditional),0.,rtol=0.,atol=2e-10)


@pytest.mark.parametrize('field,index,change',[
    ('conditional_log_means',(3,10),.001),('conditional_variances',(1,45),.001),
    ('conditional_expected_next',(0,32),.001),('conditional_actual_next',(5,11),.001),
    ('conditional_european_payoff',(2,),.001),('conditional_european_martingale',(4,),.001),
    ('replay_physical_factors',(2,17,0),.001),('replay_physical_stocks',(6,90),.01)])
def test_changed_rng_clock_conditional_or_physical_primitive_fails(replay_case,field,index,change):
    proxy,_,_,_,original=replay_case;arrays={**original,field:original[field].copy()}
    arrays[field][index]+=change
    with pytest.raises(ValueError):validate_conditional_european(arrays,{'proxy':proxy},seed=22202,paths=8)


def test_common_signed_shift_preserves_every_gap_and_model_expectations():
    probability=np.array([.25,.5,.25]);feature=np.array([-1.,0.,1.])[:,None]
    lower=np.array([1.,2.,4.]);upper=lower+np.array([.1,.2,.3]);beta=np.array([-1.7])
    result=validate_common_shift(lower,upper,feature,beta)
    np.testing.assert_allclose(result['upper']-result['lower'],upper-lower,atol=1e-15,rtol=0.)
    assert probability@result['lower']==pytest.approx(probability@lower,abs=1e-15)
    assert probability@result['upper']==pytest.approx(probability@upper,abs=1e-15)


def test_final_sample_mean_is_never_used_to_recenter_a_control():
    feature=np.array([.2,.5,.8])[:,None];beta=np.array([.7]);shift=(feature@beta)
    lower=np.array([1.,2.,3.]);upper=lower+.1
    with pytest.raises(ValueError,match='recentered'):
        validate_common_shift(lower,upper,feature,beta,saved_shift=shift-shift.mean())
    with pytest.raises(ValueError,match='Upper endpoint'):
        validate_common_shift(lower,upper,feature,beta,shifted_lower=lower-shift,shifted_upper=upper)
