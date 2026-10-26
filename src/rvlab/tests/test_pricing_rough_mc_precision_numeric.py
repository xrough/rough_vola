"""Equivalence of the selected-only successor's numerical reductions."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr
from rvlab.experiments._pricing_rough_mc_precision import normal_interval,gaussian_moments,shared_expectations
from rvlab.experiments._pricing_rough_node_refinement_fast import signed_shared_expectations
from rvlab.experiments._pricing_rough_local_control import _linear_coefficients
from rvlab.pricing.rough_stopping_controls import expected_stopping_value


def test_selected_side_normal_cdf_is_bit_exact():
 left=np.array([-np.inf,-12.,-3.,-1.,0.,.1,1.,8.,20.]);right=left+1.;right[0]=-20.
 expected=np.where(left>0,ndtr(-left)-ndtr(-right),ndtr(right)-ndtr(left))
 assert np.array_equal(normal_interval(left,right),expected)


@pytest.mark.parametrize('variance',[0.,1e-12,.001,.25,4.])
def test_shared_moments_match_reference_at_knots_and_tails(variance):
 nodes=np.array([0.,.7,.95,1.,1.05,1.3]);knots=nodes[1:]
 values=np.array([[.7,.5,.2,.15,.1,.05],[0.,.001,.2,.3,.15,0.]])
 coefficient=_linear_coefficients(nodes,values)[None,None]
 mean=np.log(100*np.array([1e-8,.7,.95,1.,1.05,1.3,1e8]))[None,None]
 clock=np.full_like(mean,variance)
 a=shared_expectations(mean,clock,coefficient,strike=100.,knots=knots)
 b=signed_shared_expectations(mean,clock,coefficient,strike=100.,knots=knots)
 assert np.array_equal(a,b)
 moments=gaussian_moments(mean,clock,strike=100.,knots=knots)
 assert np.array_equal(a,shared_expectations(mean,clock,coefficient,strike=100.,knots=knots,moments=moments))


@pytest.mark.parametrize('variance',[0.,1e-10,.01,1.])
def test_bounded_linear_spline_agrees_with_original_clipped_integrator(variance):
 nodes=np.array([0.,.7,.95,1.,1.05,1.3]);knots=tuple(nodes[1:])
 values=np.array([[.7,.5,.2,.15,.1,.05],[0.,.001,.2,.3,.15,0.]])
 coefficient=_linear_coefficients(nodes,values)[None,None]
 mean=np.log(100*np.array([1e-8,.7,.95,1.,1.05,1.3,1e8]))[None,None]
 clock=np.full_like(mean,variance)
 new=shared_expectations(mean,clock,coefficient,strike=100.,knots=knots)
 for level in range(2):
  old=expected_stopping_value(mean,clock,coefficient[:,:,level,None],strike=100.,is_call=False,exercise=False,stock_knots=knots,value_cap=.7)
  assert np.allclose(new[:,:,level],old,rtol=0.,atol=2e-13)


def test_vectorized_spline_against_independent_scalar_normal_integral():
 nodes=np.array([0.,.7,1.,1.3]);values=np.array([.7,.1,.2,.05]);coef=_linear_coefficients(nodes,values)[None,None,None]
 mean=np.log(97.);variance=.08
 value=shared_expectations(np.array([[[mean]]]),np.array([[[variance]]]),coef,strike=100.,knots=nodes[1:])[0,0,0,0]
 def f(x):
  z=np.exp(mean+np.sqrt(variance)*x)/100
  return np.interp(z,nodes,values)*np.exp(-x*x/2)/np.sqrt(2*np.pi)
 points=[(np.log(100*k)-mean)/np.sqrt(variance) for k in nodes[1:]]
 bounds=[-12]+points+[12]
 scalar=sum(quad(f,a,b,epsabs=1e-13)[0] for a,b in zip(bounds[:-1],bounds[1:]))
 assert abs(value-scalar)<2e-13


def test_reject_negative_clock_and_non_linear_coefficients():
 with pytest.raises(ValueError):gaussian_moments(np.zeros((1,1,1)),-np.ones((1,1,1)),strike=100.,knots=[1.])
 with pytest.raises(ValueError):shared_expectations(np.zeros((1,1,1)),np.ones((1,1,1)),np.ones((1,1,1,2,3)),strike=100.,knots=[1.])


def test_zero_clock_sides_of_knots_cap_floor_and_constant_tails():
 nodes=np.array([0.,.7,.95,1.,1.05,1.3]);knots=tuple(nodes[1:])
 # Both cap and floor attainments; upper tail is exactly constant.
 values=np.array([[.7,.7,0.,0.,.7,.7],[0.,0.,.7,.7,0.,0.]])
 coefficient=_linear_coefficients(nodes,values)[None,None]
 z=np.r_[1e-14,np.sort(np.concatenate([nodes[1:]*(1+d) for d in (-1e-10,0.,1e-10)])),1e10]
 mean=np.log(100*z)[None,None];clock=np.zeros_like(mean)
 fast=shared_expectations(mean,clock,coefficient,strike=100.,knots=knots)
 for level in range(2):
  reference=expected_stopping_value(mean,clock,coefficient[:,:,level,None],strike=100.,is_call=False,exercise=False,stock_knots=knots,value_cap=.7)
  assert np.allclose(fast[:,:,level],reference,atol=2e-13,rtol=0.)
  assert np.allclose(fast[0,0,level],np.interp(z,nodes,values[level]),atol=2e-13,rtol=0.)


def test_mixed_degenerate_and_stochastic_gaussian_branches_match_reference():
 nodes=np.array([0.,.7,1.,1.3]);values=np.array([[.7,.2,.1,0.],[0.,.1,.2,.7]])
 coefficient=_linear_coefficients(nodes,values)[None,None]
 mean=np.log(100*np.array([.5,.7,1.,1.3,2.]))[None,None]
 clock=np.array([[[0.,1e-12,.02,0.,.4]]])
 fast=shared_expectations(mean,clock,coefficient,strike=100.,knots=nodes[1:])
 old=signed_shared_expectations(mean,clock,coefficient,strike=100.,knots=nodes[1:])
 assert np.array_equal(fast,old)


@pytest.mark.parametrize('chunk,expected',[(4096,[[0,128],[128,255]]),(64,[[0,64],[64,128],[128,192],[192,255]])])
def test_pure_worker_partition_preserves_order_and_no_random_work(monkeypatch,tmp_path,chunk,expected):
 import rvlab.experiments._pricing_rough_mc_precision as m
 for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):monkeypatch.setenv(k,'1')
 def physical(proxy,*,seed,paths,indices,progress,workspace):
  n=255;return {k:(np.arange(n)[:,None,None] if k!='factors' else np.arange(n)[:,None,None]) for k in ('stocks','factors','european_proxy','discounted_payoffs','european_martingale','premium_martingale','intrinsic_martingale','micro_boundary_martingale')}, {'conditional_european_estimate':np.arange(n)}, {'mock_only':True}
 def finish(*args,forbidden_seeds,progress):
  trace,q=args[-2:];v=trace['stocks'][:,0,0]
  return {'times':np.arange(65),'stocks':trace['stocks'],'value':v},v,{'basis_names':['mock']}
 monkeypatch.setattr(m,'selected_physical_trace',physical);monkeypatch.setattr(m,'_finish_selected',finish)
 bundle={'proxy':object(),'manifest':{'all_selection_training_seeds':[]}}
 arrays,q,d=m.selected_samples(bundle,None,None,None,None,None,None,seed=42,paths=255,workers=2,workspace=tmp_path,pure_chunk_paths=chunk)
 assert d['path_partitions']==expected
 assert isinstance(arrays['value'],np.memmap)
 assert d['file_backed_workspace'] is True
 assert np.array_equal(arrays['value'],np.arange(255))
 assert np.array_equal(q['shared_control'],np.arange(255))
 assert d['no_random_or_fitting_in_children'] is True
 assert all(c['cpu_seconds']>=0 for c in d['chunks'])


def test_pure_workers_require_explicit_single_thread_environment(monkeypatch):
 import rvlab.experiments._pricing_rough_mc_precision as m
 monkeypatch.delenv('OPENBLAS_NUM_THREADS',raising=False)
 with pytest.raises(ValueError,match='single BLAS'):
  m.selected_samples({'manifest':{'all_selection_training_seeds':[]}},None,None,None,None,None,None,seed=42,paths=4,workers=2)


def test_scratch_arrays_are_zero_origin_and_never_overwrite(tmp_path):
 from rvlab.experiments._pricing_rough_mc_precision import _zeros
 v=_zeros(tmp_path,'origin',(19,21,3))
 assert isinstance(v,np.memmap)
 assert np.count_nonzero(v)==0
 v[1,2,0]=4.;v.flush()
 with pytest.raises(FileExistsError):_zeros(tmp_path,'origin',(19,21,3))
 assert np.load(tmp_path/'origin.npy',mmap_mode='r')[1,2,0]==4.
