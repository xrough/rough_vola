"""Structural controls retain exact stock interpolation and frozen decisions."""
from dataclasses import replace
import math
import numpy as np
import pytest
from scipy.integrate import quad

from rvlab.pricing import RoughHestonParams,approximate_fractional_kernel
from rvlab.pricing.rough_bs_premium_proxy import (
    ConditionalBSPremium,build_bs_premium_proxy,audit_bs_premium_proxy,
)
from rvlab.pricing.rough_stopping import RoughAmericanConfig
from rvlab.pricing.rough_stopping_controls import stopping_value,expected_stopping_value
from rvlab.pricing.rough_stopping_fourier import (
    train_fourier_stopping_policy,value_fourier_stopping_policy,fourier_policy_fingerprint,
)
from rvlab.pricing.vanilla_types import SpotMarket,DiscountCurve,DividendSchedule


def _table():
    return ConditionalBSPremium(np.array([0.,.5,1.]),np.array([.5,1.5]),
        np.array([[2.,1.,0.],[4.,2.,0.]]),np.array([1.]),1.,1.)


def test_conditional_variance_interpolation_and_linear_stock_segments():
    model=_table();factors=np.array([[-1.],[0.],[1.]])
    coefficients=model.coefficients(factors)
    assert coefficients.shape==(3,3,3)
    spots=np.array([25.,50.,125.])
    values=stopping_value(spots,coefficients,strike=100.,is_call=False,
        exercise=False,stock_knots=(.5,1.))
    np.testing.assert_allclose(values,[1.5,1.5,0.],rtol=0,atol=1e-14)
    assert np.all(coefficients[...,2]==0.)
    with pytest.raises(ValueError):model.nodal_premiums[0,0]=5.


def test_exact_proxy_expectation_agrees_with_independent_normal_density_integral():
    model=_table();coefficients=model.coefficients(np.array([0.]))
    actual=expected_stopping_value(math.log(70.),.09,coefficients,strike=100.,
        is_call=False,exercise=False,stock_knots=(.5,1.))
    premium=.5*(model.nodal_premiums[0]+model.nodal_premiums[1])
    def f(z):
        value=np.interp(math.exp(math.log(70.)+.3*z)/100.,model.stock_nodes,premium)
        return value*math.exp(-.5*z*z)/math.sqrt(2*math.pi)
    cuts=[-12.,(math.log(50)-math.log(70))/.3,(math.log(100)-math.log(70))/.3,12.]
    reference=sum(quad(f,a,b,epsabs=1e-13,epsrel=1e-13)[0] for a,b in zip(cuts[:-1],cuts[1:]))
    assert actual==pytest.approx(reference,abs=2e-12)


def test_full_memory_can_distinguish_equal_aggregate_variance_states():
    model=ConditionalBSPremium(np.array([0.,1.]),np.array([0.,1.]),
        np.array([[1.,0.],[3.,0.]]),np.array([.2,.8]),0.,1.)
    factors=np.array([[.1,0.],[0.,.1]])
    # Both factor sums are .1, while their expected future integrals differ.
    np.testing.assert_allclose(model.average_variance(factors),[.02,.08])
    assert not np.array_equal(model.coefficients(factors)[0],model.coefficients(factors)[1])


def test_range_clipping_is_explicit_and_maturity_proxy_is_exactly_zero():
    model=_table()
    audit=model.range_diagnostics(np.array([[-1.],[0.],[1.]]))
    assert audit['below_variance_grid']==1 and audit['above_variance_grid']==1
    terminal=replace(model,remaining_time=0.)
    np.testing.assert_array_equal(terminal.coefficients(np.array([[0.],[1.]])),np.zeros((2,3,3)))
    assert terminal.range_diagnostics(np.array([[0.]]))['states']==0


def test_real_bs_premium_control_does_not_change_cashflows_or_stopping_dates():
    params=RoughHestonParams(.04,.045,1.5,.3,-.6,.15)
    kernel=approximate_fractional_kernel(.15,1.,2,min_time_ratio=.003)
    market=SpotMarket(100.,DiscountCurve.flat(.04,1.),DividendSchedule((),1.))
    config=RoughAmericanConfig(training_paths=64,valuation_paths=64,steps=4,exercise_steps=4,
        batch_size=32,training_seed=42,valuation_seed=43,stock_knots=(.95,1.,1.05))
    policy=train_fourier_stopping_policy(params,market,kernel,strike=100.,maturity=.2,
        config=config,premium_cap=.7)
    baseline=value_fourier_stopping_policy(policy,trace=True)
    structural=build_bs_premium_proxy(policy,volatilities=(.1,.2,.3),
        stock_knots=(.8,1.,1.2),spot_nodes=257)
    assert structural.continuations is policy.continuations
    assert structural.exercise_at_inception==policy.exercise_at_inception
    assert structural.config.stock_knots==(.8,1.,1.2)
    fingerprint=fourier_policy_fingerprint(structural)
    result=value_fourier_stopping_policy(structural,trace=True)
    assert fourier_policy_fingerprint(structural)==fingerprint
    np.testing.assert_array_equal(result['raw_samples'],baseline['raw_samples'])
    np.testing.assert_array_equal(result['stop_times'],baseline['stop_times'])
    assert np.all(result['dual_samples']>=result['primal_samples'])
    audit=audit_bs_premium_proxy(structural,result['trace'])
    assert audit['states']==64*4
    assert audit['clipping_changes_proxy_only']
