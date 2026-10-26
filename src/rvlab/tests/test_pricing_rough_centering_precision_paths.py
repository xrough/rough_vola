"""Full microgrid saves states while preserving the frozen exercise calendar."""
from dataclasses import dataclass

import numpy as np
import pytest

from rvlab.experiments._pricing_rough_centering_precision_paths import (
    evaluate_precision_paths, full_microstate_paths, verify_microstate_replay)
from rvlab.pricing import (DiscountCurve, DividendSchedule, RoughHestonParams,
    SpotMarket, approximate_fractional_kernel)
from rvlab.pricing.rough_dual import RoughDualConfig, RoughDualWeights, apply_rough_dual_weights
from rvlab.pricing.rough_policy_diagnostics import ControlledExercisePolicy, evaluate_trace_policy
from rvlab.pricing.rough_stopping import RoughAmericanConfig
from rvlab.pricing.rough_stopping_fourier import (
    fourier_policy_fingerprint, train_fourier_stopping_policy)


@dataclass(frozen=True)
class _ConstantContinuation:
    value_at_state: float

    def value(self, stock, factors, strike):
        return np.full(np.asarray(stock).shape, self.value_at_state)


@pytest.fixture(scope="module")
def frozen():
    params=RoughHestonParams(.04,.045,1.5,.3,-.6,.15)
    market=SpotMarket(100.,DiscountCurve.flat(.04,1.),DividendSchedule((),1.))
    kernel=approximate_fractional_kernel(.15,1.,4,min_time_ratio=.003)
    config=RoughAmericanConfig(training_paths=64,valuation_paths=64,
        training_seed=992001,valuation_seed=992002,steps=8,exercise_steps=4,
        batch_size=32,stock_knots=(.85,.95,1.,1.05,1.15))
    proxy=train_fourier_stopping_policy(params,market,kernel,strike=100.,maturity=60/365,
        is_call=False,config=config,premium_cap=.7)
    original=ControlledExercisePolicy(proxy,proxy.exercise_times,
        tuple(_ConstantContinuation(100.) for _ in proxy.exercise_times),False,
        (992001,992005),None,None,{})
    selected=ControlledExercisePolicy(proxy,proxy.exercise_times,
        tuple(_ConstantContinuation(-100.) for _ in proxy.exercise_times),False,
        (992001,992006),None,None,{})
    weights=RoughDualWeights(np.array([0.,proxy.exercise_times[2],proxy.maturity]),
        np.array([[.8,.6,.2],[1.1,.9,-.15]]),fourier_policy_fingerprint(proxy),
        config.training_seed,992003,config.valuation_seed,
        RoughDualConfig(components=("european","premium","micro_boundary")),{"selected_candidate":0})
    return {"proxy":proxy,"original_policy":original,"selected_policy":selected,"weights":weights}


@pytest.fixture(scope="module")
def replay(frozen):
    return verify_microstate_replay(frozen,seed=992107,paths=32)


def test_full_microstate_replay_preserves_core_paths_stopping_and_raw_prices(replay):
    checks,data,trace=replay
    assert checks["original_stops_identical"] and checks["selected_stops_identical"]
    assert checks["no_new_exercise_dates"] and checks["excluded_from_primary_inference"]
    assert checks["stock_max_difference"]<1e-10
    assert checks["factor_max_difference"]<1e-12
    assert checks["european_proxy_max_difference"]<1e-10
    assert checks["raw_gain_max_difference"]<1e-10
    grid=data["simulation"].times
    exercise=trace["times"]
    assert len(grid)>len(exercise)
    np.testing.assert_array_equal(grid[data["exercise_indices"]],exercise)
    assert set(data["original_stop_times"]).issubset(set(exercise))
    assert set(data["selected_stop_times"]).issubset(set(exercise))


def test_signed_gates_reconstruct_arbitrary_stopped_microgrid_increments(replay):
    _,data,_=replay
    grid=data["simulation"].times
    n=data["paths"]
    left=grid[None,:-1]
    expected=(left<data["selected_stop_times"][:,None]).astype(int)-(left<data["original_stop_times"][:,None]).astype(int)
    np.testing.assert_array_equal(data["signed_left_gates"],expected)
    # This deterministic example tests gate orientation and exact event sides;
    # its increments are not claimed to be martingale differences.
    increments=np.sin(np.arange(n*len(left[0])).reshape(n,-1))
    cumulative=np.c_[np.zeros(n),increments.cumsum(axis=1)]
    original=np.searchsorted(grid,data["original_stop_times"])
    selected=np.searchsorted(grid,data["selected_stop_times"])
    endpoint=cumulative[np.arange(n),selected]-cumulative[np.arange(n),original]
    np.testing.assert_allclose((data["signed_left_gates"]*increments).sum(axis=1),endpoint,atol=1e-14,rtol=0.)
    assert np.any(data["signed_left_gates"])
    assert not np.any(data["signed_left_gates"][left>=np.maximum(data["original_stop_times"],data["selected_stop_times"])[:,None]])


def test_realized_gated_control_matches_complete_core_stopped_control(frozen,replay):
    _,data,trace=replay
    output=evaluate_precision_paths(frozen,data,quadrature_nodes=(16,32,64))
    original=evaluate_trace_policy(trace,policy=frozen["original_policy"],martingale_weights=frozen["weights"])
    selected=evaluate_trace_policy(trace,policy=frozen["selected_policy"],martingale_weights=frozen["weights"])
    control=apply_rough_dual_weights(trace,frozen["weights"])["martingale"][:,:,0]
    p=np.arange(data["paths"])
    expected=control[p,data["selected_stops"]]-control[p,data["original_stops"]]
    np.testing.assert_allclose(output["samples"]["raw_contrast"],expected,atol=2e-10,rtol=0.)
    np.testing.assert_allclose(output["controlled_gain"],
        selected.controlled_samples[:,0]-original.controlled_samples[:,0],atol=2e-10,rtol=0.)
    assert len(output["states"])==np.abs(data["signed_left_gates"]).sum()
    assert output["projected_by_level"].shape==(3,data["paths"])
    for name in ("quadrature_error","tail_error","arithmetic_error"):
        assert np.all(np.isfinite(output["samples"][name]))
        assert np.all(output["samples"][name]>=0)
    zero=np.all(data["signed_left_gates"]==0,axis=1)
    for values in output["samples"].values():
        np.testing.assert_array_equal(values[zero],0.)


@pytest.mark.parametrize("seed",[-1,True,np.bool_(True),1.5,"992107"])
def test_invalid_seed_rejected_before_simulation(frozen,seed):
    with pytest.raises(ValueError,match="nonnegative integer"):
        full_microstate_paths(frozen,seed=seed,paths=2)


@pytest.mark.parametrize("paths",[0,1,True,np.bool_(True),2.5,"32"])
def test_invalid_path_count_rejected_before_simulation(frozen,paths):
    with pytest.raises(ValueError,match="integer >=2"):
        full_microstate_paths(frozen,seed=992107,paths=paths)


@pytest.mark.parametrize("seed",[992001,992003,992005,992006])
def test_every_fitting_stream_is_excluded(frozen,seed):
    with pytest.raises(ValueError,match="independent of every fitting stream"):
        full_microstate_paths(frozen,seed=seed,paths=2)


@pytest.mark.parametrize("levels",[(32,), (32,16), (16,16), (True,32), (3,32)])
def test_invalid_quadrature_rejected_even_when_no_paths_disagree(frozen,replay,levels):
    _,data,_=replay
    data={**data,"signed_left_gates":np.zeros_like(data["signed_left_gates"])}
    with pytest.raises(ValueError,match="quadrature.nodes"):
        evaluate_precision_paths(frozen,data,quadrature_nodes=levels)


def test_identical_policies_return_all_paths_as_zero_without_quadrature(frozen,replay):
    _,data,_=replay
    data={**data,"signed_left_gates":np.zeros_like(data["signed_left_gates"]),"raw_gain":np.zeros(data["paths"])}
    output=evaluate_precision_paths(frozen,data)
    assert output["states"].empty
    assert not np.any(output["absolute_gate_counts"])
    for values in output["samples"].values():
        np.testing.assert_array_equal(values,0.)
