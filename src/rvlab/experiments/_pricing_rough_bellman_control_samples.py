"""Independent path identities for fixed-policy Bellman martingale controls."""
from __future__ import annotations
import numpy as np
from ._pricing_rough_tail_policy import margin_policy_specs
from ._pricing_rough_tail_samples import validate_tail_samples, _weighted_addition, _coefficient_rows

BASELINE = 'latest selected tail control'
NEW = {'latest_martingale', 'target_martingales', 'target_expected_european',
    'target_actual_premiums', 'target_expected_premiums', 'gh16_stops', 'gh16_raw',
    'gh16_continuation', 'gh16_computed_mask'}


def validate_specs(controls, completed):
    expected = [BASELINE]+[f'target {bins} bins penalty{penalty:g}'
        for bins in (1,4) for penalty in (0.,1e-4) if f'target {bins} bins penalty{penalty:g}' in completed]
    if list(controls) != expected:
        raise ValueError('The fallback or predeclared successful control family/order changed')
    for name, control in controls.items():
        _coefficient_rows(control,4)
        if name == BASELINE:
            if control != {'family':'fallback','bins':1,'penalty':0.,'weights':[0.]*4}:
                raise ValueError('The unchanged fallback control changed')
        elif (control.get('family') != 'target' or control.get('penalty') not in (0.,1e-4)
                or name != f"target {control['bins']} bins penalty{control['penalty']:g}"
                or set(control) != {'family','bins','penalty','weights'}):
            raise ValueError('An undeclared target-control or shrinkage candidate appeared')


def validate_control_samples(arrays, controls, bundle, prior, grid, tail_seal, *, paths):
    endpoint_names = {f'candidate{i}_{suffix}' for i in range(len(controls)) for suffix in ('lower','upper')}
    if not (NEW|endpoint_names).issubset(arrays):
        raise ValueError('Missing target primitive, policy mask or candidate endpoint')
    old_arrays = {k:v for k,v in arrays.items() if k not in NEW|endpoint_names}
    old = validate_tail_samples(old_arrays,tail_seal['controls'],bundle,prior,grid,margin_policy_specs(),paths=paths)
    proxy=bundle['proxy']; times=arrays['times']; discounts=arrays['discounts']; payoff=arrays['payoff']
    common=tail_seal['controls'][tail_seal['selected_control']]
    latest=arrays['current_martingale']+_weighted_addition(arrays['tail_corrections'],common)
    if arrays['latest_martingale'].shape != (paths,65) or not np.allclose(arrays['latest_martingale'],latest,atol=2e-13,rtol=0.):
        raise ValueError('Latest fixed martingale differs from prior sealed components')
    intrinsic=np.maximum((1 if proxy.is_call else -1)*(arrays['stocks'][:,:,0]-proxy.strike),0.)
    mask=arrays['gh16_computed_mask']; c=arrays['gh16_continuation']
    if (mask.shape != (paths,65) or mask.dtype.kind != 'b' or c.shape != mask.shape
            or not np.isfinite(c[mask]).all() or not np.isnan(c[~mask]).all() or np.any(c[mask]<0)):
        raise ValueError('Continuation must be finite on the exact alive-ITM mask and NaN elsewhere')
    stops=np.full(paths,64,dtype=np.int64); expected_mask=np.zeros((paths,65),bool)
    for j in range(1,64):
        active=(stops==64)&(intrinsic[:,j]>0); expected_mask[:,j]=active
        action=active&(intrinsic[:,j]>=c[:,j]); stops[action]=j
    if not np.array_equal(mask,expected_mask):
        raise ValueError('Frozen GH16 rule omitted an alive ITM state or used a future/stopped state')
    if arrays['gh16_stops'].dtype.kind not in 'iu' or not np.array_equal(arrays['gh16_stops'],stops):
        raise ValueError('Frozen GH16 stopping decisions changed')
    path=np.arange(paths); raw=payoff[path,stops]
    if not np.array_equal(arrays['gh16_raw'],raw):
        raise ValueError('Raw payoff does not match the policy own stopping date')
    shapes={'target_martingales':(paths,65,4),'target_expected_european':(paths,65),
        'target_actual_premiums':(paths,65,4),'target_expected_premiums':(paths,65,4)}
    if any(arrays[k].shape != shape or not np.isfinite(arrays[k]).all() for k,shape in shapes.items()):
        raise ValueError('Target conditional primitives changed dimensions or contain nonfinite paths')
    if (any(np.any(arrays[k][:,0]!=0) for k in shapes)
            or any(np.any(arrays[k][:,-1]!=0) for k in ('target_actual_premiums','target_expected_premiums'))
            or any(np.any(arrays[k]<-1e-12) for k in ('target_actual_premiums','target_expected_premiums'))):
        raise ValueError('Target origin, terminal premium or positive Snell premium changed')
    increments=((arrays['european_proxy'][:,1:,0]-arrays['target_expected_european'][:,1:])[:,:,None]
        +discounts[None,1:,None]*(arrays['target_actual_premiums'][:,1:]-arrays['target_expected_premiums'][:,1:]))
    targets=np.concatenate((np.zeros((paths,1,4)),np.cumsum(increments,axis=1)),axis=1)
    if not np.allclose(arrays['target_martingales'],targets,atol=2e-13,rtol=0.):
        raise ValueError('Target martingales differ from exact saved actual-minus-conditional increments')
    values={}; corrections=arrays['target_martingales']-arrays['latest_martingale'][:,:,None]
    for i,(name,control) in enumerate(controls.items()):
        m=arrays['latest_martingale']+_weighted_addition(corrections,control)
        lower=raw-m[path,stops]; upper=np.max(payoff-m,axis=1)
        for suffix, expected in [('lower',lower),('upper',upper)]:
            if arrays[f'candidate{i}_{suffix}'].shape != (paths,) or not np.allclose(
                    arrays[f'candidate{i}_{suffix}'],expected,atol=2e-13,rtol=0.):
                raise ValueError('A candidate changed its own lower/upper endpoint or omitted a path')
        if np.any(upper<lower-2e-11): raise ValueError('Pathwise dual upper/lower ordering failed')
        values[name]={'lower':lower,'upper':upper}
    pure={}
    for i in range(4):
        m=targets[:,:,i]
        pure[i]={'lower':raw-m[path,stops],'upper':np.max(payoff-m,axis=1)}
    return {'values':values,'pure_targets':pure,'raw':raw,'stops':stops,'latest_martingale':latest,
        'targets':targets,'computed_mask':mask,'payoff':payoff}
