"""Independent all-array checks catch changed gates, rules and control seals."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import expit

from rvlab.experiments._pricing_rough_tail_policy import margin_policy_specs
from rvlab.experiments._pricing_rough_tail_samples import validate_control_specs, validate_tail_samples
from rvlab.pricing import DiscountCurve, DividendSchedule, RoughHestonParams, SpotMarket
from rvlab.pricing.rough_simulation import _time_grid


class ConstantPremium:
    def __init__(self, amount): self.amount = amount
    def value(self, stock, factors, strike): return np.full_like(stock, self.amount)


def controls_fixture():
    fallback = {"family": "fallback", "bins": 1, "penalty": 0., "shrink_scale": 1., "weights": [0.]*7}
    post = {"family": "poststop", "bins": 1, "penalty": 0., "shrink_scale": 1., "weights": [.2, .1, .1, 0., 0., 0., 0.]}
    extended = {"family": "extended", "bins": 4, "penalty": 1e-4, "shrink_scale": 1., "weights": ([.05]*7)*4}
    controls = {"current baseline": fallback, "poststop 1 bins penalty0": post,
        "extended 4 bins penalty0.0001": extended}
    parent = "extended 4 bins penalty0.0001"
    for scale in (.25, .5):
        controls[f"{parent} shrink{scale:g}"] = {**extended, "parent_control": parent, "shrink_scale": scale,
            "weights": (np.asarray(extended["weights"])*scale).tolist()}
    return controls, {"poststop 1 bins penalty0", parent}


def weighted(corrections, control):
    bins, features = control["bins"], corrections.shape[-1]
    cuts = np.linspace(0, 64, bins+1, dtype=int)
    rows = np.vstack([np.tile(w, (b-a, 1)) for w, a, b in
        zip(np.asarray(control["weights"]).reshape(bins, features), cuts[:-1], cuts[1:])])
    return np.column_stack((np.zeros(len(corrections)), np.cumsum(np.sum(np.diff(corrections, axis=1)*rows, axis=-1), axis=1)))


@pytest.fixture
def full_tail_arrays():
    # Cheap algebraic states, not simulated evidence of martingale centering.
    n = 4; times = np.linspace(0., .2, 65)
    market = SpotMarket(100., DiscountCurve.flat(.04, 1.), DividendSchedule((), 1.))
    params = RoughHestonParams(.04, .04, 1., .3, -.6, .15)
    proxy = SimpleNamespace(exercise_times=times, continuation_times=times, market=market,
        params=params, kernel=SimpleNamespace(nodes=np.arange(3)), is_call=False, strike=100., maturity=.2)
    models = tuple(ConstantPremium(.006 if j < 64 else 0.) for j in range(65))
    policy = SimpleNamespace(times=times, continuations=models, premium_cap=.7)
    fit = SimpleNamespace(config=SimpleNamespace(predictable_features=False,
        components=("european", "premium", "micro_boundary")), time_edges=np.array([0., .1, .2]),
        weights=np.array([[.4, .3, .2], [.2, .4, .1]]))
    bundle = {"proxy": proxy, "selected_policy": policy, "weights": fit}
    discount = market.discount_curve.discount(times)
    stocks = np.broadcast_to(np.linspace(100., 95., 65)[None, :, None], (n, 65, 1)).copy()
    stocks[1, 1:, 0] += 1.; stocks[2, 1:, 0] -= 1.; stocks[3, 1:, 0] += 3.
    factors = np.zeros((n, 65, 3)); factors[:, 1:] = .01
    intrinsic = np.maximum(100.-stocks[:, :, 0], 0.)
    payoff = intrinsic*discount
    euro = (payoff+.1*discount)[:, :, None]
    for p, j in ((0, 20), (1, 40), (2, 10)):
        euro[p, j, 0] = payoff[p, j]-.01*discount[j]
    euro[:, -1, 0] = payoff[:, -1]
    base = np.sin(times)[None, :]*np.arange(1., n+1)[:, None]*.01
    european, premium, micro, intrinsic_m = base, base**2, base*.5, base*.7
    components = np.stack((european, premium, micro), axis=-1)
    timebins = np.searchsorted(fit.time_edges[1:], times[1:], side="left")
    stable = np.column_stack((np.zeros(n), np.cumsum(np.sum(np.diff(components, axis=1)*fit.weights[timebins], axis=-1), axis=1)))
    prior_control = {"bins": 1, "weights": [.1, .05, .1]}
    grid_control = {"bins": 4, "weights": [.02]*24}
    prior_c = np.stack((base*.2, base*.3, base*.4), axis=-1)
    grid_c = np.stack([base*i*.1 for i in range(1, 7)], axis=-1)
    prior = stable+weighted(prior_c, prior_control)
    current = prior+weighted(grid_c, grid_control)
    q = np.full((n, 65), .006); q[:, -1] = 0.
    margin = (payoff-euro[:, :, 0])/discount-q
    specs = margin_policy_specs()
    stops = np.full((n, 7), 64, dtype=int)
    for p, spec in enumerate(specs.values()):
        for path in range(n):
            for j in range(1, 64):
                adjusted = np.clip(q[path, j]*spec["multiplier"]+spec["offset"]*(.2-times[j])/.2, 0., .7)
                if intrinsic[path, j] > 0 and intrinsic[path, j] >= euro[path, j, 0]/discount[j]+adjusted:
                    stops[path, p] = j
                    break
    post = (np.arange(64)[None, :] >= stops[:, :1]).astype(float)
    left = margin[:, :-1]
    gates = np.stack((post, post*expit(left/.1), post*expit(left/.3), expit(left/.03), expit(left/.1),
        expit(left/.3), (payoff[:, :-1] == 0).astype(float)), axis=-1)
    di, de = np.diff(intrinsic_m-current, axis=1), np.diff(european-current, axis=1)
    target = np.stack([di]*6+[de], axis=-1)*gates
    tail = np.concatenate((np.zeros((n, 1, 7)), np.cumsum(target, axis=1)), axis=1)
    raw = payoff[np.arange(n)[:, None], stops]
    arrays = {"times": times, "discounts": discount, "eligible": np.ones(65, bool),
        "actual_time_grid": _time_grid(params, market, times, 128), "stocks": stocks,
        "factors": factors, "european_proxy": euro, "payoff": payoff,
        "european_martingale": european, "intrinsic_martingale": intrinsic_m,
        "premium_martingale": premium, "micro_boundary_martingale": micro,
        "stable_old_martingale": stable, "prior_local_corrections": prior_c,
        "prior_martingale": prior, "grid_corrections": grid_c, "current_martingale": current,
        "tail_corrections": tail, "tail_gates": gates, "tail_margin": margin, "raw": raw, "stops": stops}
    controls, completed = controls_fixture()
    for i, control in enumerate(controls.values()):
        m = current+weighted(tail, control)
        arrays[f"control{i}_lower"] = raw-m[np.arange(n)[:, None], stops]
        arrays[f"control{i}_upper"] = np.max(payoff-m, axis=1)
    return arrays, controls, bundle, prior_control, grid_control, specs, completed


def test_full_algebra_fixture_retains_every_policy_path_and_poststop_identity(full_tail_arrays):
    arrays, controls, bundle, prior, grid, specs, completed = full_tail_arrays
    validate_control_specs(controls, completed)
    result = validate_tail_samples(arrays, controls, bundle, prior, grid, specs, paths=4)
    assert result["stops"].shape == (4, 7)
    assert not np.array_equal(result["stops"][:, 0], result["stops"][:, 4])
    np.testing.assert_array_equal(result["values"]["poststop 1 bins penalty0"]["lower"][:, 0],
        result["values"]["current baseline"]["lower"][:, 0])


@pytest.mark.parametrize("tamper", ["parent", "shrink_weight", "shrink_scale", "parent_bins", "missing_child",
    "poststop_feature", "perbin_sum", "negative_weight", "fallback", "unexpected", "missing_fit"])
def test_control_seals_reject_shrinkage_family_and_bin_budget_changes(tamper):
    controls, completed = controls_fixture()
    child = controls["extended 4 bins penalty0.0001 shrink0.25"]
    if tamper == "parent": child["parent_control"] = "poststop 1 bins penalty0"
    if tamper == "shrink_weight": child["weights"][0] += .001
    if tamper == "shrink_scale": child["shrink_scale"] = .75
    if tamper == "parent_bins": child["bins"] = 1; child["weights"] = child["weights"][:7]
    if tamper == "missing_child": controls.pop("extended 4 bins penalty0.0001 shrink0.25")
    if tamper == "poststop_feature": controls["poststop 1 bins penalty0"]["weights"][4] = .01
    if tamper == "perbin_sum": controls["extended 4 bins penalty0.0001"]["weights"][:7] = [.2]*7
    if tamper == "negative_weight": controls["poststop 1 bins penalty0"]["weights"][0] = -.01
    if tamper == "fallback": controls["current baseline"]["weights"][0] = .1
    if tamper == "unexpected": controls["undeclared"] = deepcopy(controls["current baseline"])
    if tamper == "missing_fit": controls.pop("poststop 1 bins penalty0")
    with pytest.raises(ValueError): validate_control_specs(controls, completed)


@pytest.mark.parametrize("tamper,match", [
    ("gate", "predictable"), ("future_gate", "predictable"), ("margin", "margin"),
    ("stop", "exercise decision"), ("raw", "own stopping"), ("upper", "endpoint"),
    ("lower", "endpoint"), ("baseline_component", "reconstructed"), ("prior_component", "reconstructed"),
    ("grid_component", "reconstructed"), ("tail_component", "increments"), ("payoff", "discounted vanilla"),
    ("physical_calendar", "calendars"), ("eligibility", "calendars"), ("missing_policy", "dimensions"),
    ("missing_array", "path arrays"), ("nonfinite", "path arrays")])
def test_full_arrays_detect_changed_rules_gates_and_endpoints(full_tail_arrays, tamper, match):
    arrays, controls, bundle, prior, grid, specs, _ = full_tail_arrays
    if tamper == "gate": arrays["tail_gates"][0, 0, 0] = 1.
    if tamper == "future_gate": arrays["tail_gates"] = np.roll(arrays["tail_gates"], -1, axis=1)
    if tamper == "margin": arrays["tail_margin"][0, 2] += .001
    if tamper == "stop": arrays["stops"][0, 0] = 19
    if tamper == "raw": arrays["raw"][0, 0] += .001
    if tamper == "upper": arrays["control0_upper"][0] += .001
    if tamper == "lower": arrays["control0_lower"][0, 0] += .001
    if tamper == "baseline_component": arrays["premium_martingale"][0, 2] += .001
    if tamper == "prior_component": arrays["prior_local_corrections"][0, 2, 0] += .001
    if tamper == "grid_component": arrays["grid_corrections"][0, 2, 0] += .001
    if tamper == "tail_component": arrays["tail_corrections"][0, 21, 0] += .001
    if tamper == "payoff": arrays["payoff"][0, 2] += .001
    if tamper == "physical_calendar": arrays["actual_time_grid"][1] += 1e-6
    if tamper == "eligibility": arrays["eligible"][1] = False
    if tamper == "missing_policy": arrays["stops"] = arrays["stops"][:, :-1]
    if tamper == "missing_array": arrays.pop("tail_gates")
    if tamper == "nonfinite": arrays["stocks"][0, 2, 0] = np.nan
    with pytest.raises(ValueError, match=match):
        validate_tail_samples(arrays, controls, bundle, prior, grid, specs, paths=4)
