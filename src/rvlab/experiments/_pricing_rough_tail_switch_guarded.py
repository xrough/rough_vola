"""Independent-stream firewall around the preserved spent-pilot control source."""
from numbers import Integral
from ._pricing_rough_tail_switch import FEATURES,FAMILIES
from ._pricing_rough_tail_switch import tail_switch_corrections as _corrections
from ._pricing_rough_tail_switch import current_control_workspace as _workspace

def _independent(selected,trace,forbidden_seeds):
    seed=trace.get('seed')
    if (isinstance(seed,bool) or not isinstance(seed,Integral) or seed<0
            or any(isinstance(s,bool) or not isinstance(s,Integral) or s<0 for s in forbidden_seeds)
            or seed in set(selected.training_seeds)|set(forbidden_seeds)):
        raise ValueError('Tail-switch evaluation requires a recorded stream independent of policy/control fitting and selection')

def tail_switch_corrections(proxy,selected,trace,current_martingale,*,forbidden_seeds=()):
    _independent(selected,trace,forbidden_seeds)
    return _corrections(proxy,selected,trace,current_martingale,forbidden_seeds=forbidden_seeds)

def current_control_workspace(proxy,selected,old_weights,prior_control,grid_control,trace,*,forbidden_seeds=()):
    _independent(selected,trace,forbidden_seeds)
    return _workspace(proxy,selected,old_weights,prior_control,grid_control,trace)

__all__=['FEATURES','FAMILIES','tail_switch_corrections','current_control_workspace']
