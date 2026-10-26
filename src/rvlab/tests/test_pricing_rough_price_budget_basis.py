"""Predictable gates preserve centering, including the stopping increment."""
from types import SimpleNamespace
import numpy as np
import pytest
from rvlab.experiments._pricing_rough_price_budget_basis import reconstruct_predictable_basis


def case():
    n=4;times=np.arange(65,dtype=float);factors=np.zeros((n,65,4));stocks=np.full((n,65,1),100.)
    first=np.array([-1.,-1.,1.,1.]);second=np.array([-1.,1.,-1.,1.])
    m=np.zeros((n,65));m[:,1]=first;m[:,2:]=(first+second)[:,None]
    factors[:,1:,0]=first[:,None];stocks[:,1:,0]+=first[:,None]
    models=tuple(SimpleNamespace(center=np.zeros(4),scale=np.ones(4)) for _ in times)
    bundle={'proxy':SimpleNamespace(strike=100.,kernel=SimpleNamespace(nodes=np.arange(4)),exercise_times=times),
        'selected_policy':SimpleNamespace(times=times,continuations=models)}
    arrays={'stocks':stocks,'factors':factors,'times':times,'latest_martingale':np.zeros_like(m),
        'target_martingales':np.zeros((n,65,4)),'gh16_stops':np.where(first>0,1,64)}
    for name in ('european_martingale','premium_martingale','intrinsic_martingale','micro_boundary_martingale'):
        arrays[name]=m.copy()
    return arrays,bundle,first,second


def test_exact_finite_tree_mean_and_left_state_gates():
    arrays,bundle,first,second=case();basis,metadata=reconstruct_predictable_basis(arrays,bundle)
    assert basis.shape==(4,192) and len(metadata['column_order'])==192
    structured=basis.reshape(4,4,8,6)
    expected=first+np.where(first<0,second,0.)
    np.testing.assert_array_equal(structured[:,0,4,0],expected)
    # State at time1 is the known first increment; the stopping decision at
    # time1 prevents further increments on positive branches.
    expected_factor=np.where(first<0,first*second,0.)
    np.testing.assert_array_equal(structured[:,0,4,2],expected_factor)
    np.testing.assert_allclose(basis.mean(axis=0),0.,rtol=0.,atol=1e-15)
    # Omitting the stopping increment creates a negative-mean contrast.
    wrong=np.where(first<0,first+second,0.)
    assert wrong.mean()==-.5


def test_date_bins_column_order_and_no_empirical_recenter():
    arrays,bundle,_,_=case();arrays['gh16_stops'][:]=64
    value=arrays['european_martingale'];value[:]=0.;value[:,17:]=np.arange(1.,5.)[:,None]
    basis,metadata=reconstruct_predictable_basis(arrays,bundle)
    s=basis.reshape(4,4,8,6)
    np.testing.assert_array_equal(s[:,1,4,0],np.arange(1.,5.))
    assert s[:,0,4,0].sum()==0 and basis[:,(1*8+4)*6].mean()==2.5
    assert metadata['column_order'][(1*8+4)*6]=='bin1/european/constant'
    assert not metadata['empirical_valuation_mean_subtracted']


@pytest.mark.parametrize('kind',['reset','origin','stop','normalization'])
def test_invalid_memory_or_martingale_identity_rejected(kind):
    arrays,bundle,_,_=case()
    if kind=='reset':arrays['factors']=arrays['factors'][:,:,:1]
    elif kind=='origin':arrays['european_martingale'][:,0]=1.
    elif kind=='stop':arrays['gh16_stops'][0]=65
    else:bundle['selected_policy'].continuations[1].scale[0]=0.
    with pytest.raises(ValueError):reconstruct_predictable_basis(arrays,bundle)
