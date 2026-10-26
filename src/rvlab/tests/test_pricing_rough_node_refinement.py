"""Nested Snell controls refine interpolation, never the exercise contract."""
from dataclasses import replace
import math
import numpy as np
import pytest
from scipy.integrate import quad
from rvlab.tests.test_pricing_rough_local_control import local_case
from rvlab.experiments._pricing_rough_local_control import signed_linear_expectation
from rvlab.experiments._pricing_rough_node_refinement import (
    refined_stock_knots,interpolate_nodes,snell_refinement_coefficients,
    signed_shared_expectations,node_refinement_martingales)


def test_nested_grids_exactly_include_coarse_and_lower_tail_is_unchanged():
    coarse=np.array((.7,.85,.875,.9,.91,.92,.93,.94,.95,.96,.97,.98,.99,1.,1.01,1.02,1.05,1.1,1.2,1.3))
    a,b=(refined_stock_knots(coarse,m) for m in (4,8))
    assert len(a)==77 and len(b)==153
    np.testing.assert_array_equal(a[np.searchsorted(a,coarse)],coarse)
    np.testing.assert_array_equal(b[np.searchsorted(b,a)],a)
    assert a[0]==b[0]==coarse[0]


def test_shared_signed_moments_match_scalar_integrals_with_zero_clock():
    nodes=np.array([0.,.8,.95,1.,1.1,1.3]);knots=nodes[1:]
    from rvlab.experiments._pricing_rough_local_control import _linear_coefficients
    a=_linear_coefficients(nodes,np.array([-.2,.1,-.05,.12,-.08,.03]))
    b=_linear_coefficients(nodes,np.array([.1,-.1,.02,-.03,.07,-.02]))
    coefficient=np.stack((a,b))[None,None]
    mean=np.log(np.array([[[80.,96.,101.]]]))
    clocks=np.array([[[0.,.014,.003]]])
    actual=signed_shared_expectations(mean,clocks,coefficient,strike=100.,knots=knots)
    for i,c in enumerate((a,b)):
        expected=signed_linear_expectation(mean,clocks,c,strike=100.,knots=knots)
        np.testing.assert_allclose(actual[:,:,i],expected,rtol=0.,atol=2e-15)


def test_fine_minus_coarse_is_zero_at_all_coarse_nodes_and_scalar_quad_agrees(local_case):
    proxy,selected,trace=local_case
    nodes=np.r_[0.,refined_stock_knots(proxy.config.stock_knots,8)]
    factors=trace['factors'][:2,2]
    coefficient,_=snell_refinement_coefficients(proxy,selected,2,factors,fine_nodes=nodes)
    coarse=np.r_[0.,proxy.config.stock_knots]
    idx=np.searchsorted(nodes[1:],coarse,side='right')
    for row in range(2):
        for level in range(2):
            c=coefficient[row,level,idx]
            np.testing.assert_allclose(c[:,0]+c[:,1]*coarse,0.,atol=2e-12,rtol=0.)
    mean,clock=math.log(94.),.012
    expected=signed_shared_expectations(np.array([[[mean]]]),np.array([[[clock]]]),coefficient[:1,None],strike=100.,knots=nodes[1:])[0,0,:,0]
    points=[-12.]+[(math.log(100*k)-mean)/math.sqrt(clock) for k in nodes[1:] if -12<(math.log(100*k)-mean)/math.sqrt(clock)<12]+[12.]
    for level in range(2):
        c=coefficient[0,level]
        def integrand(z):
            stock=math.exp(mean+math.sqrt(clock)*z)/100.
            p=np.searchsorted(nodes[1:],stock,side='right')
            return (c[p,0]+c[p,1]*stock)*math.exp(-.5*z*z)/math.sqrt(2*math.pi)
        reference=sum(quad(integrand,a,b,epsabs=1e-12)[0] for a,b in zip(points[:-1],points[1:]))
        assert expected[level]==pytest.approx(reference,abs=2e-12)


def test_refinement_approaches_direct_selected_snell_value_on_held_states(local_case):
    proxy,selected,trace=local_case;j=2
    factors=trace['factors'][:3,j];direct_nodes=np.r_[0.,np.linspace(.8,1.2,2001)]
    def direct(nodes):
        euro=replace(proxy.value_premiums[j],stock_nodes=nodes).european_values(factors)
        stocks=100*nodes;q=np.clip(selected.continuations[j].value(np.broadcast_to(stocks,(3,len(stocks))),factors,100.),0.,.7)
        return np.clip(np.maximum(q,np.maximum(100-stocks,0.)-euro),0.,.7)
    truth=direct(direct_nodes);errors=[]
    for m in (1,4,8):
        nodes=np.r_[0.,proxy.config.stock_knots if m==1 else refined_stock_knots(proxy.config.stock_knots,m)]
        approximate=interpolate_nodes(nodes,direct(nodes),direct_nodes)
        errors.append(float(np.sqrt(np.mean((approximate-truth)**2))))
    assert errors[2]<errors[1]<errors[0]


def test_node_martingale_retains_source_states_and_calendar(local_case):
    proxy,selected,trace=local_case
    before={k:trace[k].copy() for k in ('stocks','factors','discounted_payoffs','actual_time_grid')}
    correction,diagnostics=node_refinement_martingales(proxy,selected,trace,batch_size=16)
    assert correction.shape==(32,5,6) and np.isfinite(correction).all()
    assert np.all(correction[:,0]==0) and diagnostics['coarse_nodes_exact_subsets']
    for key,value in before.items():np.testing.assert_array_equal(trace[key],value)


@pytest.mark.parametrize('multiple',[True,0,1,3,4.0])
def test_invalid_refinement_is_identified(multiple):
    with pytest.raises(ValueError):refined_stock_knots((.8,.9,1.),multiple)
