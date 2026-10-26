"""Formal tail controls reject reused or unidentified policy/control streams."""
import numpy as np
import pytest
from rvlab.tests.test_pricing_rough_local_control import local_case
from rvlab.experiments._pricing_rough_tail_switch_guarded import tail_switch_corrections

@pytest.mark.parametrize('kind',['missing','policy fitting','explicit control fitting','boolean'])
def test_independent_seed_firewall(local_case,kind):
    proxy,selected,trace=local_case;changed=dict(trace);forbidden=()
    if kind=='missing':changed.pop('seed')
    elif kind=='policy fitting':changed['seed']=selected.training_seeds[0]
    elif kind=='explicit control fitting':forbidden=(trace['seed'],)
    else:changed['seed']=True
    m=trace['european_martingale'][:,:,0]
    with pytest.raises(ValueError,match='independent'):tail_switch_corrections(proxy,selected,changed,m,forbidden_seeds=forbidden)

def test_guard_does_not_change_valid_corrections(local_case):
    from rvlab.experiments._pricing_rough_tail_switch import tail_switch_corrections as preserved
    proxy,selected,trace=local_case;m=trace['european_martingale'][:,:,0]
    expected=preserved(proxy,selected,trace,m);actual=tail_switch_corrections(proxy,selected,trace,m)
    np.testing.assert_array_equal(actual['corrections'],expected['corrections'])
