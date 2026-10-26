"""Independent conditional integration, including a planted nonzero drift."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

from rvlab.experiments._pricing_rough_centering_audit import audit_conditional_centers
from rvlab.experiments._pricing_rough_centering_precision_quadrature import (
    ConditionalCenteringQuadrature, _direct_premium, _tail_moment,
)
from rvlab.pricing import (DiscountCurve, DividendSchedule, RoughHestonParams,
    SpotMarket, approximate_fractional_kernel)
from rvlab.pricing.rough_dual import RoughDualConfig, RoughDualWeights
from rvlab.pricing.rough_obstacle_premium import build_rough_obstacle_proxy
from rvlab.pricing.rough_stopping import RoughAmericanConfig
from rvlab.pricing.rough_stopping_fourier import (
    fourier_policy_fingerprint, train_fourier_stopping_policy, value_fourier_stopping_policy,
)


@pytest.fixture(scope="module", params=[False, True])
def case(request):
    params=RoughHestonParams(.04,.045,1.5,.3,-.6,.15)
    market=SpotMarket(100.,DiscountCurve.flat(-.02 if request.param else .04,1.),DividendSchedule((),1.))
    kernel=approximate_fractional_kernel(.15,1.,4,min_time_ratio=.003)
    config=RoughAmericanConfig(training_paths=64,valuation_paths=64,training_seed=991_001,
        valuation_seed=991_002,steps=8,exercise_steps=4,batch_size=32,stock_knots=(.85,.95,1.,1.05,1.15))
    original=train_fourier_stopping_policy(params,market,kernel,strike=100.,maturity=60/365,
        is_call=request.param,config=config,premium_cap=.7)
    policy=build_rough_obstacle_proxy(original,stock_knots=(.75,.9,.96,1.,1.04,1.2))
    trace=value_fourier_stopping_policy(policy,trace=True)["trace"]
    weights=RoughDualWeights(np.array([0.,policy.exercise_times[2],policy.maturity]),
        np.array([[.8,.6,.2],[1.1,.9,-.15]]),fourier_policy_fingerprint(policy),
        config.training_seed,991_003,config.valuation_seed,
        RoughDualConfig(components=("european","premium","micro_boundary")),{})
    return policy,trace,weights


def test_batched_piecewise_gaussian_integration_agrees_with_independent_scalar_audit(case):
    policy,trace,weights=case
    q=ConditionalCenteringQuadrature(policy,martingale_weights=weights,time_grid=trace["actual_time_grid"])
    for date in (0,2,3):
        rows=[(path,date) for path in range(3)]
        scalar=audit_conditional_centers(policy,trace,state_indices=rows,martingale_weights=weights)
        actual=q.evaluate(left_time=float(trace["times"][date]),left_stocks=trace["stocks"][:3,date,0],
            left_factors=trace["factors"][:3,date],original_policy_stopped=date>=trace["stop_indices"][:3,0])
        expected=scalar["tables"]["components"].query("component=='selected_martingale'")
        np.testing.assert_allclose(actual["conditional_drift"],expected.conditional_increment_mean,atol=3e-10,rtol=0.)
        assert np.max(actual["refinement_and_tail_estimate"])<2e-9
        assert np.max(actual["absolute_tail_envelope"])<1e-24
        assert not actual["quadrature_error_is_rigorous_bound"]
        assert actual["primitive_drift_by_level"].shape==(3,3,3)
        np.testing.assert_allclose(actual["independent_primitive_expectation_by_level"][-1],
            actual["analytic_primitive_expectation"],atol=3e-10,rtol=0.)


def test_quadrature_does_not_assume_zero_and_detects_injected_center_error(case,monkeypatch):
    import rvlab.experiments._pricing_rough_centering_precision_quadrature as module
    policy,trace,weights=case
    weights=replace(weights,weights=np.array([[1.,0.],[1.,0.]]),
        config=RoughDualConfig(components=("european","premium")))
    original=module._affine_values
    monkeypatch.setattr(module,"_affine_values",lambda *a,**kw:original(*a,**kw)+.003)
    q=ConditionalCenteringQuadrature(policy,martingale_weights=weights,time_grid=trace["actual_time_grid"])
    result=q.evaluate(left_time=0.,left_stocks=trace["stocks"][:4,0,0],left_factors=trace["factors"][:4,0])
    np.testing.assert_allclose(result["conditional_drift"],-.003,atol=3e-11,rtol=0.)
    assert np.max(result["refinement_and_tail_estimate"])<3e-11


def test_actual_endpoint_functions_match_trace_values_and_direct_premium(case):
    policy,trace,weights=case
    q=ConditionalCenteringQuadrature(policy,martingale_weights=weights,time_grid=trace["actual_time_grid"])
    date=2;time=float(trace["times"][date])
    actual=q.actual_primitive_values(time=time,stocks=trace["stocks"][:,date,0],factors=trace["factors"][:,date])
    np.testing.assert_allclose(actual[:,0],trace["european_proxy"][:,date,0],atol=1e-13,rtol=1e-13)
    np.testing.assert_allclose(actual[:,2]*policy.market.discount_curve.discount(time),
        trace["discounted_payoffs"][:,date,0],atol=1e-13,rtol=0.)
    assert np.all((actual[:,1]>=0)&(actual[:,1]<=policy.premium_cap))


@pytest.mark.parametrize("sigma,power",[(.05,0),(.2,1),(.7,2)])
def test_omitted_normal_tail_moment_matches_independent_integration(sigma,power):
    limit=8.
    numerical=sum(quad(lambda z:np.exp(power*sigma*z-.5*z*z)/np.sqrt(2*np.pi),a,b,
        epsabs=1e-28,epsrel=1e-10)[0] for a,b in [(-np.inf,-limit),(limit,np.inf)])
    assert _tail_moment(power,sigma,limit)==pytest.approx(numerical,rel=2e-10,abs=1e-30)


def test_piecewise_direct_polynomial_honors_knots_and_clipping():
    coefficients=np.array([[1.,-2.,1.],[-1.,2.,0.]])
    stocks=np.array([50.,100.,150.])
    np.testing.assert_allclose(_direct_premium(stocks,coefficients,100.,(1.,),.7),[.25,.7,.7])


@pytest.mark.parametrize("levels",[(32,), (32,16), (16,16), (True,32), (3,32)])
def test_invalid_quadrature_levels_rejected(case,levels):
    policy,trace,weights=case
    q=ConditionalCenteringQuadrature(policy,martingale_weights=weights,time_grid=trace["actual_time_grid"])
    with pytest.raises(ValueError,match="quadrature_nodes"):
        q.evaluate(left_time=0.,left_stocks=np.array([100.]),left_factors=np.zeros((1,len(policy.kernel.nodes))),quadrature_nodes=levels)


def test_state_features_and_mismatched_policy_rejected(case):
    policy,trace,weights=case
    with pytest.raises(ValueError,match="does not match"):
        ConditionalCenteringQuadrature(policy,martingale_weights=replace(weights,policy_fingerprint="wrong"),time_grid=trace["actual_time_grid"])
    feature=replace(weights,weights=np.tile(weights.weights,(1,2)),
        config=replace(weights.config,predictable_features=("moneyness",)))
    with pytest.raises(ValueError,match="calendar-only"):
        ConditionalCenteringQuadrature(policy,martingale_weights=feature,time_grid=trace["actual_time_grid"])
