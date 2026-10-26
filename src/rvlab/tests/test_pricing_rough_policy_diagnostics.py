"""Fixed-law policy comparisons must change only the exercise decisions."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from rvlab.pricing import RoughHestonParams, approximate_fractional_kernel
from rvlab.pricing.rough_stopping import RoughAmericanConfig
from rvlab.pricing.rough_stopping_fourier import (
    train_fourier_stopping_policy, value_fourier_stopping_policy,
)
from rvlab.pricing.rough_policy_diagnostics import (
    ControlledExercisePolicy, compare_trace_policies, evaluate_trace_policy,
    fit_controlled_policy_iteration, fit_frozen_policy_value_proxy, select_trace_policy,
)
from rvlab.pricing.rough_dual import RoughDualConfig, RoughDualWeights
from rvlab.pricing.vanilla_types import DiscountCurve, DividendSchedule, SpotMarket


class ConstantContinuation:
    def __init__(self, value): self.constant = value
    def value(self, spots, factors, strike): return np.full(np.shape(spots), self.constant)


def _trace():
    times=np.array([0., .1, .2])
    stocks=np.array([[100.,90.,95.],[100.,95.,98.],[100.,105.,90.],[100.,80.,85.]])[...,None]
    payoff=np.maximum(100-stocks,0.)
    euro_m=np.array([[0.,1.,2.],[0.,-1.,0.],[0.,2.,1.],[0.,-2.,-1.]])[...,None]
    premium_m=.1*euro_m
    euro=np.full_like(stocks,5.)
    return dict(times=times,eligible=np.array([True,True,True]),discounted_payoffs=payoff,
        european_martingale=euro_m,premium_martingale=premium_m,
        stop_indices=np.full((4,1),2,dtype=int),stocks=stocks,
        factors=np.zeros((4,3,1)),european_proxy=euro,
        seed=1042,policy_training_seed=42,exercise_style='american')


def _policy():
    config=RoughAmericanConfig(training_paths=4,valuation_paths=4,steps=2,exercise_steps=2,
        training_seed=42,valuation_seed=1000042,stock_knots=())
    market=SpotMarket(100.,DiscountCurve.flat(0.,1.),DividendSchedule((),1.))
    return SimpleNamespace(config=config,market=market,strike=100.,maturity=.2,is_call=False,
        exercise_style='american',exercise_at_inception=False,premium_cap=.7,
        diagnostics={},
        continuation_times=np.array([0.,.1,.2]),
        continuations=tuple(ConstantContinuation(20.) for _ in range(3)))


def test_original_stop_trace_recovers_controlled_payoff_and_common_upper():
    trace=_trace()
    result=evaluate_trace_policy(trace,martingale_weights=(.5,2.))
    m=.5*trace['european_martingale']+2*trace['premium_martingale']
    np.testing.assert_allclose(result.controlled_samples,trace['discounted_payoffs'][:,-1]-m[:,-1])
    np.testing.assert_allclose(result.dual_samples,np.max(trace['discounted_payoffs']-m,axis=1))
    assert np.all(result.dual_samples>=result.controlled_samples)
    with pytest.raises(ValueError): result.raw_samples[0,0]=0.


def test_train_selected_bin_weights_keep_the_same_policy_upper_and_seed_firewalls():
    trace=_trace();trace['policy_fingerprint']='fixed-policy'
    config=RoughDualConfig(training_paths=4,training_seed=242,candidate_bins=(1,))
    weights=RoughDualWeights(np.array([0.,.2]),np.array([[.9,1.4]]),'fixed-policy',
        42,242,1000042,config,{'selected_candidate':'one fixed bin'})
    weighted=evaluate_trace_policy(trace,martingale_weights=weights)
    constants=evaluate_trace_policy(trace,martingale_weights=(.9,1.4))
    np.testing.assert_allclose(weighted.controlled_samples,constants.controlled_samples,rtol=0,atol=2e-15)
    fitted=fit_controlled_policy_iteration(_policy(),trace,martingale_weights=weights)
    assert fitted.training_seeds==(42,1042,242)
    reused=_trace();reused['seed']=242
    with pytest.raises(ValueError,match='overlaps'):
        evaluate_trace_policy(reused,policy=fitted)
    trace['seed']=242
    with pytest.raises(ValueError,match='dual-training'):
        evaluate_trace_policy(trace,martingale_weights=weights)


def test_trace_control_training_provenance_also_blocks_valuation_reuse():
    trace=_trace();trace['additional_control_training_seeds']=(1042,)
    with pytest.raises(ValueError,match='overlaps'):
        evaluate_trace_policy(trace)


def test_policy_comparison_changes_lower_samples_with_exactly_identical_upper():
    trace=_trace();base=_policy()
    candidate=ControlledExercisePolicy(base,trace['times'],tuple(ConstantContinuation(0.) for _ in range(3)),
        False,(42,242),0.,.7,{})
    comparison=compare_trace_policies(trace,candidate,martingale_weights=(1.,1.))
    a,b=comparison['baseline'],comparison['candidate']
    np.testing.assert_array_equal(b.stop_indices[:,0],[1,1,2,1])
    np.testing.assert_array_equal(a.dual_samples,b.dual_samples)
    expected=(b.controlled_samples-a.controlled_samples)[:,0]
    assert comparison['rows'][0]['paired_lower_gain']['mean']==pytest.approx(expected.mean())
    assert comparison['diagnostics']['valuation_outcomes_used_for_selection'] is False


def test_third_stream_controlled_training_labels_are_the_exact_common_martingale_labels():
    trace=_trace();base=_policy();labels=[]
    def fit(spots,factors,targets,strike,config):
        labels.append(targets.copy())
        return ConstantContinuation(0.)
    fitted=fit_controlled_policy_iteration(base,trace,fit_continuation=fit)
    m=(trace['european_martingale']+trace['premium_martingale'])[...,0]
    expected=trace['discounted_payoffs'][:,-1,0]-m[:,-1]+m[:,1]-trace['european_proxy'][:,1,0]
    np.testing.assert_allclose(labels[0],expected)
    assert fitted.training_seeds==(42,1042)
    assert fitted.diagnostics['valuation_used_in_training'] is False
    with pytest.raises(ValueError,match='overlaps'):
        evaluate_trace_policy(trace,policy=fitted)


@pytest.mark.parametrize('seed',[42,1000042])
def test_improvement_training_requires_a_third_stream(seed):
    trace=_trace();trace['seed']=seed
    with pytest.raises(ValueError,match='third'):
        fit_controlled_policy_iteration(_policy(),trace)


def test_new_policy_rejects_a_different_exercise_calendar():
    trace=_trace();base=_policy()
    fitted=ControlledExercisePolicy(base,np.array([0.,.15,.2]),base.continuations,False,(42,242),0.,.7,{})
    with pytest.raises(ValueError,match='same time'):
        evaluate_trace_policy(trace,policy=fitted)


def test_independent_policy_selection_keeps_fallback_upper_and_selection_seed_firewall():
    trace,base=_trace(),_policy()
    base.premium_cap=20.
    candidate=ControlledExercisePolicy(base,trace['times'],tuple(ConstantContinuation(0.) for _ in range(3)),
        False,(42,242),0.,.7,{})
    selection=select_trace_policy(trace,{'original':base,'candidate':candidate})
    assert selection['selected_name']=='candidate'
    assert all(row['same_upper_samples'] for row in selection['rows'])
    assert selection['diagnostics']['final_valuation_used_for_selection'] is False
    assert selection['diagnostics']['selection_confidence_multiplier']==1.96
    for policy in selection['candidates'].values():
        assert 1042 in policy.training_seeds
        with pytest.raises(ValueError,match='overlaps'):
            evaluate_trace_policy(trace,policy=policy)
    final=_trace();final['seed']=1043
    final_value=evaluate_trace_policy(final,policy=selection['selected_policy'])
    original_value=evaluate_trace_policy(final)
    np.testing.assert_array_equal(final_value.dual_samples,original_value.dual_samples)
    rebuilt=evaluate_trace_policy(final,policy=selection['candidates']['original'])
    np.testing.assert_array_equal(rebuilt.raw_samples,original_value.raw_samples)


def test_policy_selection_unresolved_and_tied_candidates_retain_baseline():
    trace,base=_trace(),_policy()
    result=select_trace_policy(trace,{'original':base,'identical':base})
    assert result['selected_name']=='original'
    assert all(row['selection_score']==0 for row in result['rows'])
    assert all(row['independent_final_acceptance_evidence'] is False for row in result['rows'])


def test_policy_selection_mean_rule_is_explicit_and_every_evaluated_policy_is_sealed():
    trace,base=_trace(),_policy()
    candidate=ControlledExercisePolicy(base,trace['times'],tuple(ConstantContinuation(0.) for _ in range(3)),
        False,(42,242),0.,.7,{})
    conservative=select_trace_policy(trace,{'original':base,'candidate':candidate})
    point=select_trace_policy(trace,{'original':base,'candidate':candidate},selection_rule='lower_mean')
    assert conservative['selected_name']=='original'
    assert point['selected_name']=='candidate'
    assert all(1042 in policy.training_seeds for policy in point['candidates'].values())
    with pytest.raises(ValueError,match='overlaps'):
        select_trace_policy(trace,point['candidates'])


@pytest.mark.parametrize('candidates,kwargs',[
    ({},{}),({'original':None},{}),({'other':_policy()},{}),
    ({'original':_policy()},{'selection_rule':'price_gate'}),
])
def test_policy_selection_rejects_malformed_candidates(candidates,kwargs):
    with pytest.raises(ValueError):
        select_trace_policy(_trace(),candidates,**kwargs)


def test_policy_fitting_preserves_decision_grid_independently_of_control_knots():
    trace,base=_trace(),_policy()
    original_knots=(.9,1.)
    for model in base.continuations:
        model.knots=original_knots
    base.config=replace(base.config,stock_knots=(.7,.8,.9,1.,1.1))
    observed=[]
    def fit(spots,factors,targets,strike,config):
        observed.append((config.stock_knots,config.ridge))
        return ConstantContinuation(0.)
    fitted=fit_controlled_policy_iteration(base,trace,fit_continuation=fit)
    assert observed==[(original_knots,base.config.ridge)]
    assert fitted.diagnostics['fit_stock_knots']==original_knots
    assert fitted.diagnostics['control_stock_knots']==base.config.stock_knots
    observed.clear()
    fit_controlled_policy_iteration(base,trace,fit_continuation=fit,stock_knots=(.8,1.1),ridge=1e-4)
    assert observed==[((.8,1.1),1e-4)]


@pytest.mark.parametrize('kwargs',[{'stock_knots':(.9,.8)},{'ridge':-1.},{'ridge':True}])
def test_policy_fitting_rejects_invalid_regression_overrides(kwargs):
    with pytest.raises(ValueError):
        fit_controlled_policy_iteration(_policy(),_trace(),**kwargs)


@pytest.mark.parametrize('weights',[(1.,),[np.nan,1.],['x','y'],[True,True]])
def test_invalid_martingale_weights_rejected(weights):
    with pytest.raises(ValueError,match='weights'):
        evaluate_trace_policy(_trace(),martingale_weights=weights)


@pytest.mark.parametrize('field',['times','european_martingale','stop_indices','european_proxy'])
def test_missing_or_malformed_trace_rejected(field):
    trace=_trace();trace.pop(field)
    with pytest.raises(ValueError,match='trace'):
        evaluate_trace_policy(trace,policy=_policy())


def test_real_fourier_trace_reconstruction_and_independent_policy_refit():
    params=RoughHestonParams(.04,.045,1.5,.3,-.6,.15)
    kernel=approximate_fractional_kernel(.15,1.,2,min_time_ratio=.003)
    market=SpotMarket(100.,DiscountCurve.flat(.04,1.),DividendSchedule((),1.))
    config=RoughAmericanConfig(training_paths=64,valuation_paths=64,steps=4,exercise_steps=4,
        batch_size=32,training_seed=42,valuation_seed=1000042,stock_knots=(.95,1.,1.05))
    policy=train_fourier_stopping_policy(params,market,kernel,strike=100.,maturity=.2,
        config=config,premium_cap=.7)
    value=value_fourier_stopping_policy(policy,trace=True)
    rebuilt=evaluate_trace_policy(value['trace'],policy=policy)
    np.testing.assert_allclose(rebuilt.raw_samples,value['raw_samples'],rtol=0.,atol=0.)
    np.testing.assert_allclose(rebuilt.controlled_samples,value['primal_samples'],rtol=0.,atol=2e-15)
    np.testing.assert_allclose(rebuilt.dual_samples,value['dual_samples'],rtol=0.,atol=2e-15)
    training=value_fourier_stopping_policy(policy,config=replace(config,valuation_seed=242),trace=True)['trace']
    improved=fit_controlled_policy_iteration(policy,training)
    comparison=compare_trace_policies(value['trace'],improved)
    assert comparison['diagnostics']['same_upper_samples']
    assert comparison['candidate'].diagnostics['independent_training_valuation']
    proxy=fit_frozen_policy_value_proxy(policy,training)
    assert proxy.continuations is policy.continuations
    assert proxy.exercise_at_inception==policy.exercise_at_inception
    assert proxy.diagnostics['exercise_policy_unchanged']
    assert proxy.diagnostics['additional_control_training_seeds']==(242,)
    refitted=value_fourier_stopping_policy(proxy)
    np.testing.assert_array_equal(refitted['raw_samples'],value['raw_samples'])
    np.testing.assert_array_equal(refitted['stop_times'],value['stop_times'])
    assert np.all(refitted['dual_samples']>=refitted['primal_samples'])
    with pytest.raises(ValueError,match='control.training'):
        value_fourier_stopping_policy(proxy,config=replace(config,valuation_seed=242))
    from rvlab.pricing.rough_obstacle_premium import build_rough_obstacle_proxy
    obstacle=build_rough_obstacle_proxy(policy,stock_knots=(.8,.9,1.,1.1,1.2))
    train_obstacle=value_fourier_stopping_policy(obstacle,
        config=replace(obstacle.config,valuation_seed=243),trace=True)['trace']
    exercise_refit=fit_controlled_policy_iteration(obstacle,train_obstacle)
    assert exercise_refit.diagnostics['fit_stock_knots']==config.stock_knots
    assert exercise_refit.diagnostics['control_stock_knots']==obstacle.config.stock_knots
    assert all(model.knots==config.stock_knots for model in exercise_refit.continuations)
