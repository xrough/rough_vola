"""Adversarial checks of saved Bellman decisions and common-M endpoints."""
import numpy as np
import pytest

from rvlab.tests.test_pricing_rough_tail_samples import full_tail_arrays, weighted
from rvlab.experiments._pricing_rough_bellman_samples import validate_bellman_samples


@pytest.fixture
def bellman_arrays(full_tail_arrays):
    arrays, controls, bundle, prior, grid, _, _ = full_tail_arrays
    specs = {"baseline": {"order": None, "blend": 0.}}
    for order in (8,16,32):
        for blend in (.5,1.):
            specs[f"{order}/{blend}"] = {"order": order, "blend": blend}
    p = np.maximum(100.-arrays["stocks"][:,:,0],0.)
    mask = p > 0; mask[:,[0,64]] = False
    q = np.full((4,65),.006); q[:,-1] = 0.
    baseline = arrays["european_proxy"][:,:,0]/arrays["discounts"]+q
    c = np.full((4,65,3),np.nan)
    c[mask] = p[mask,None]*np.array([.9,.8,.7])
    c[:,63] = np.where(mask[:,63,None], .95*p[:,63,None], np.nan)
    stops = np.full((4,7),64,dtype=int)
    for col,spec in enumerate(specs.values()):
        for path in range(4):
            for j in range(1,64):
                value = baseline[path,j]
                if col:
                    value = (1-spec["blend"])*value+spec["blend"]*c[path,j,(8,16,32).index(spec["order"])]
                if p[path,j] > 0 and p[path,j] >= value:
                    stops[path,col] = j
                    break
    selected = "extended 4 bins penalty0.0001"
    m = arrays["current_martingale"]+weighted(arrays["tail_corrections"],controls[selected])
    raw = arrays["payoff"][np.arange(4)[:,None],stops]
    arrays.update(bellman_continuations=c, bellman_direct_continuations=c.copy(),
        bellman_computed_mask=mask,baseline_continuation=baseline,
        new_raw=raw,new_stops=stops,new_lower=raw-m[np.arange(4)[:,None],stops],
        common_upper=np.max(arrays["payoff"]-m,axis=1))
    return arrays,bundle,prior,grid,{"controls":controls,"selected_control":selected},specs


def test_full_bellman_mask_decisions_and_seven_common_m_endpoints(bellman_arrays):
    result = validate_bellman_samples(*bellman_arrays,paths=4)
    assert result["lower"].shape == (4,7)
    assert np.any(result["stops"][:,1:] != result["stops"][:,:1])
    assert np.isnan(result["continuation"][~result["computed_mask"]]).all()


@pytest.mark.parametrize("tamper", ["drop_itm", "add_otm", "nonfinite", "filled_nan", "negative",
    "baseline", "own_stop", "raw", "lower", "upper", "terminal", "spec", "missing", "unexpected"])
def test_changed_mask_policy_or_common_control_is_rejected(bellman_arrays,tamper):
    arrays,bundle,prior,grid,seal,specs = bellman_arrays
    if tamper == "drop_itm": arrays["bellman_computed_mask"][0,2] = False
    if tamper == "add_otm": arrays["bellman_computed_mask"][0,0] = True
    if tamper == "nonfinite": arrays["bellman_continuations"][0,2,0] = np.inf
    if tamper == "filled_nan": arrays["bellman_continuations"][0,0,0] = 0.
    if tamper == "negative": arrays["bellman_continuations"][0,2,0] = -.1
    if tamper == "baseline": arrays["baseline_continuation"][0,2] += .1
    if tamper == "own_stop": arrays["new_stops"][0,1] += 1
    if tamper == "raw": arrays["new_raw"][0,1] += .1
    if tamper == "lower": arrays["new_lower"][0,1] += .1
    if tamper == "upper": arrays["common_upper"][0] += .1
    if tamper == "terminal": arrays["bellman_direct_continuations"][0,63,0] += .1
    if tamper == "spec": specs["8/0.5"]["blend"] = .25
    if tamper == "missing": arrays.pop("bellman_direct_continuations")
    if tamper == "unexpected": arrays["future_cheat"] = np.zeros(4)
    with pytest.raises((ValueError,FloatingPointError)):
        validate_bellman_samples(arrays,bundle,prior,grid,seal,specs,paths=4)
