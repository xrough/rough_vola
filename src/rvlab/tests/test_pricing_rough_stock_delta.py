"""Frozen decisions are re-applied; exact terminal centers never use BS means."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from rvlab.experiments._pricing_rough_stock_delta import (
    frozen_policy_delta_samples, frozen_policy_martingale_delta_samples,
    terminal_conditional_control)
from rvlab.pricing import (BlackScholesParams, DiscountCurve, DividendSchedule,
    RoughHestonParams, SpotMarket, approximate_fractional_kernel, black_scholes_price)
from rvlab.pricing.rough_dual import RoughDualConfig, RoughDualWeights
from rvlab.pricing.rough_policy_diagnostics import ControlledExercisePolicy
from rvlab.pricing.rough_stopping import RoughAmericanConfig
from rvlab.pricing.rough_stopping_fourier import (
    fourier_policy_fingerprint, train_fourier_stopping_policy)


@pytest.fixture(scope="module")
def frozen():
    params=RoughHestonParams(.04,.04,1.5,.3,-.6,.15)
    market=SpotMarket(100.,DiscountCurve.flat(.04,1.),DividendSchedule((),1.))
    kernel=approximate_fractional_kernel(.15,1.,4,min_time_ratio=.003)
    config=RoughAmericanConfig(training_paths=128,valuation_paths=64,
        training_seed=994001,valuation_seed=994002,steps=8,exercise_steps=4,
        batch_size=32,stock_knots=(.85,.95,1.,1.05,1.15))
    proxy=train_fourier_stopping_policy(params,market,kernel,strike=100.,maturity=60/365,
        is_call=False,config=config,premium_cap=.7)
    original=ControlledExercisePolicy(proxy,proxy.exercise_times,proxy.continuations,
        proxy.exercise_at_inception,(994001,994005),0.,.7,{})
    # A distinct frozen decision policy makes ignoring bump-dependent exercise observable.
    selected=replace(original,premium_cap=.05,training_seeds=(994001,994006))
    weights=RoughDualWeights(np.array([0.,proxy.maturity]),np.array([[1.,1.,.2]]),
        fourier_policy_fingerprint(proxy),config.training_seed,994003,config.valuation_seed,
        RoughDualConfig(components=("european","premium","micro_boundary")),{"selected_candidate":0})
    return {"proxy":proxy,"original_policy":original,"selected_policy":selected,"weights":weights}


def test_raw_multistock_replay_matches_complete_control_stream(frozen):
    cheap=frozen_policy_delta_samples(frozen,seed=994107,paths=32)
    full=frozen_policy_martingale_delta_samples(frozen,seed=994107,paths=32)
    np.testing.assert_array_equal(cheap["relative_spots"],full["relative_spots"])
    for name in ("original_policy","selected_policy"):
        np.testing.assert_array_equal(cheap["stops"][name],full["stops"][name])
        np.testing.assert_allclose(cheap["samples"][name]["raw_prices"],
            full["samples"][name]["raw_prices"],atol=1e-10,rtol=0.)
    assert cheap["diagnostics"]["decisions_reapplied_at_each_bump"]
    assert not cheap["diagnostics"]["stopping_times_held_fixed"]
    assert full["diagnostics"]["same_common_martingale_for_policies"]


def test_shared_terminal_control_cancels_in_paired_policy_comparison(frozen):
    result=frozen_policy_delta_samples(frozen,seed=994108,paths=128)
    d=result["comparisons"]
    raw=d[(d.comparison=="selected minus original")&(d.estimator=="raw")]
    controlled=d[(d.comparison=="selected minus original")&(d.estimator=="terminal conditional control")]
    np.testing.assert_allclose(raw[["mean","standard_error"]],controlled[["mean","standard_error"]],atol=1e-14,rtol=0.)
    # This negative result is a structural limitation, not a sampling accident.
    assert not result["diagnostics"]["pathwise_price_bounds_preserved"]


def test_european_constant_variance_control_reproduces_bs_price_and_finite_bump_delta(frozen):
    proxy=replace(frozen["proxy"],params=replace(frozen["proxy"].params,vol_of_vol=0.),exercise_style="european")
    original=replace(frozen["original_policy"],base_policy=proxy)
    selected=replace(frozen["selected_policy"],base_policy=proxy)
    fixture={"proxy":proxy,"original_policy":original,"selected_policy":selected,
        "weights":SimpleNamespace(training_seed=994003)}
    result=frozen_policy_delta_samples(fixture,seed=994109,paths=64)
    spots=100*(1+result["relative_spots"])
    discount=proxy.market.discount_curve.discount(proxy.maturity)
    analytic=black_scholes_price(BlackScholesParams(.2),forward=spots/discount,
        discount=discount,strike=100.,maturity=proxy.maturity,is_call=False)
    for name in ("original_policy","selected_policy"):
        np.testing.assert_allclose(result["samples"][name]["controlled_prices"],
            np.broadcast_to(analytic,(64,len(spots))),atol=5e-13,rtol=0.)
    rows=result["estimates"]
    controlled=rows[(rows.policy=="selected_policy")&(rows.estimator=="terminal conditional control")]
    assert controlled.standard_error.max()<1e-12
    assert controlled.sampling_passed.all()
    assert not controlled.optimal_delta_ready.any()


@pytest.mark.parametrize("entrypoint",[frozen_policy_delta_samples,frozen_policy_martingale_delta_samples])
@pytest.mark.parametrize("seed",[994001,994002,994003,994005,994006])
def test_all_fitting_and_reserved_seed_firewalls(frozen,entrypoint,seed):
    with pytest.raises(ValueError,match="independent"):
        entrypoint(frozen,seed=seed,paths=2)


@pytest.mark.parametrize("levels",[(.005,),(.005,.01),(.005,.005),(True,.0025),(.25,.01),(.01,0)])
def test_invalid_bump_levels_rejected_before_simulation(frozen,levels):
    with pytest.raises(ValueError,match="bump_relatives"):
        frozen_policy_delta_samples(frozen,seed=994110,paths=2,bump_relatives=levels)


@pytest.mark.parametrize("kwargs",[{"seed":True,"paths":2},{"seed":2,"paths":1},
    {"seed":2,"paths":2,"control_coefficient":np.nan},{"seed":2,"paths":2,"steps":0}])
def test_invalid_settings_rejected_before_simulation(frozen,kwargs):
    with pytest.raises(ValueError):frozen_policy_delta_samples(frozen,**kwargs)


def test_missing_conditional_moments_rejected():
    simulation=SimpleNamespace(conditional_log_spots=None,conditional_variances=None)
    with pytest.raises(ValueError,match="no cash"):
        terminal_conditional_control(simulation,initial_spot=100.,spots=[99.,100.,101.],strike=100.,is_call=False)
