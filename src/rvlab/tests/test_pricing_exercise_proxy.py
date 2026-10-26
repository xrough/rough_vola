"""The exercise scenario is self-consistent, labelled and detached."""

import math

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from rvlab.experiments._pricing_exercise import (EXERCISE_PROXY_MODE,
    american_bs_premium, exercise_adjusted_surface)
from rvlab.pricing import BlackScholesParams, black_scholes_price


def _price(k, call, *, f=100., d=math.exp(-.04*.2), t=.2, vol=.2):
    return black_scholes_price(BlackScholesParams(vol), forward=f, discount=d,
        strike=k, maturity=t, is_call=call)


def _sample(steps=64):
    d, t, f, vol = math.exp(-.04*.2), .2, 100., .2
    anchors = []
    for k in (98., 100., 102.):
        c, p = _price(k, True), _price(k, False)
        premium = american_bs_premium(forward=f, discount=d, strike=k,
            maturity=t, volatility=vol, is_call=False, steps=steps)
        anchors.append({"snapshot_date": "2025-01-02", "expiry_date": "2025-03-15",
            "strike": k, "maturity": t, "discount": d,
            "market_price_call": c, "market_price_put": p+premium,
            "bid_call": c-.01, "ask_call": c+.01,
            "bid_put": p+premium-.01, "ask_put": p+premium+.01,
            "forward": k+(c-p-premium)/d})
    quotes = []
    raw_f = float(np.median([a["forward"] for a in anchors]))
    for i, (k, call) in enumerate(((95., False), (105., True))):
        premium = american_bs_premium(forward=f, discount=d, strike=k,
            maturity=t, volatility=vol, is_call=call, steps=steps)
        mid = _price(k, call)+premium
        quotes.append({"snapshot_date": "2025-01-02", "expiry_date": "2025-03-15",
            "contract_id": str(i), "strike": k, "is_call": call,
            "maturity": t, "discount": d, "forward": raw_f,
            "bid": mid-.01, "ask": mid+.01, "market_price": mid,
            "input_mode": "option_implied_forward_proxy", "eligible_for_hedging": False})
    return pd.DataFrame(quotes, index=[11, 7]), pd.DataFrame(anchors, index=[9, 4, 8])


def test_positive_rate_call_and_zero_rate_put_have_zero_premium():
    args = dict(forward=100., strike=100., maturity=.2, volatility=.2, steps=64)
    assert american_bs_premium(discount=.99, is_call=True, **args) == 0.
    assert american_bs_premium(discount=1., is_call=False, **args) == 0.
    assert american_bs_premium(discount=1.01, is_call=False, **args) == 0.


def test_real_american_labels_cannot_be_converted_into_european_calibration_targets():
    quotes, anchors = _sample()
    quotes["exercise_style"] = "american"
    with pytest.raises(ValueError, match="American observations.*row positions.*direct American"):
        exercise_adjusted_surface(quotes, anchors, steps=64)


def test_expiration_and_zero_variance_limits():
    assert american_bs_premium(forward=95., discount=.99, strike=100., maturity=0.,
        volatility=.2, is_call=False, steps=32) == 0.
    # Deterministic put: immediate versus discounted terminal exercise.
    assert american_bs_premium(forward=95., discount=.99, strike=100., maturity=.2,
        volatility=0., is_call=False, steps=32) == pytest.approx(.01*100.)


def test_premium_tree_refinement_is_small_for_representative_anchor():
    args = dict(forward=100., discount=math.exp(-.04*.2), strike=100.,
                maturity=.2, volatility=.2, is_call=False)
    coarse = american_bs_premium(**args, steps=512)
    fine = american_bs_premium(**args, steps=1024)
    assert coarse > 0.
    assert abs(fine-coarse) < .002


def test_synthetic_american_parity_and_implied_volatility_recover_inputs():
    quotes, anchors = _sample()
    result = exercise_adjusted_surface(quotes, anchors, steps=64)
    assert result.tree_steps == 64
    assert np.allclose(result.anchors.forward, 100., atol=1e-7, rtol=0)
    assert np.allclose(result.quotes.forward, 100., atol=1e-7, rtol=0)
    assert np.allclose(result.quotes.exercise_diagnostic_volatility, .2, atol=1e-7, rtol=0)
    for _, row in result.quotes.iterrows():
        assert row.market_price == pytest.approx(_price(row.strike, row.is_call), abs=1e-7)
    parity = (result.anchors.strike +
        (result.anchors.market_price_call-result.anchors.market_price_put)/result.anchors.discount)
    assert np.allclose(parity, result.anchors.forward, atol=1e-8, rtol=0)


def test_proxy_preserves_widths_order_raw_values_and_original_frames():
    quotes, anchors = _sample()
    before_q, before_a = quotes.copy(deep=True), anchors.copy(deep=True)
    result = exercise_adjusted_surface(quotes, anchors, steps=64)
    assert_frame_equal(quotes, before_q)
    assert_frame_equal(anchors, before_a)
    assert result.quotes.contract_id.tolist() == quotes.contract_id.tolist()
    assert result.anchors.strike.tolist() == anchors.strike.tolist()
    assert np.allclose(result.quotes.ask-result.quotes.bid, quotes.ask-quotes.bid)
    assert np.allclose(result.anchors.ask_put-result.anchors.bid_put, anchors.ask_put-anchors.bid_put)
    assert np.allclose(result.quotes.raw_target_price, quotes.market_price)
    assert np.allclose(result.quotes.raw_forward, quotes.forward)
    assert (result.quotes.input_mode == EXERCISE_PROXY_MODE).all()
    assert not result.quotes.eligible_for_hedging.any()
    assert not result.quotes.independent_stock_observation.any()
    assert result.provenance["strict_market_data_ready"] is False
    assert result.diagnostics.forward_change.iloc[0] > 0.


def test_an_exercise_proxy_cannot_be_adjusted_a_second_time():
    quotes, anchors = _sample()
    result = exercise_adjusted_surface(quotes, anchors, steps=64)
    with pytest.raises(ValueError, match="already contains exercise-adjusted"):
        exercise_adjusted_surface(result.quotes, result.anchors, steps=64)


def test_malformed_optional_anchor_quotes_have_row_level_errors():
    quotes, anchors = _sample()
    anchors.loc[9, "bid_put"] = np.nan
    with pytest.raises(ValueError, match="anchors row position 0: bid_put"):
        exercise_adjusted_surface(quotes, anchors, steps=64)


def test_malformed_optional_midpoint_has_a_row_level_error():
    quotes, anchors = _sample()
    quotes["market_price"] = quotes.market_price.astype(object)
    quotes.loc[11, "market_price"] = "bad"
    with pytest.raises(ValueError, match="quotes row position 0: market_price"):
        exercise_adjusted_surface(quotes, anchors, steps=64)


@pytest.mark.parametrize("change", [
    {"forward": -1.}, {"discount": 0.}, {"strike": np.nan},
    {"maturity": -.1}, {"volatility": -.1}, {"is_call": 1},
    {"steps": True}, {"steps": 15}, {"forward": True},
])
def test_premium_rejects_malformed_inputs(change):
    args = dict(forward=100., discount=.99, strike=100., maturity=.2,
                volatility=.2, is_call=False, steps=64)
    args.update(change)
    with pytest.raises(ValueError):
        american_bs_premium(**args)


@pytest.mark.parametrize("kind", ["empty", "missing", "bad_call", "crossed", "midpoint", "rate", "group", "maturity"])
def test_surface_errors_identify_malformed_rows_and_missing_inputs(kind):
    quotes, anchors = _sample()
    if kind == "empty":
        quotes = quotes.iloc[:0]
    elif kind == "missing":
        anchors = anchors.drop(columns="market_price_put")
    elif kind == "bad_call":
        quotes["is_call"] = [1, 0]
    elif kind == "crossed":
        quotes.loc[11, "bid"] = quotes.loc[11, "ask"]+.1
    elif kind == "midpoint":
        quotes.loc[11, "market_price"] += .1
    elif kind == "rate":
        quotes.loc[11, "discount"] = 1.01
    elif kind == "group":
        quotes.loc[11, "expiry_date"] = "2025-03-16"
    elif kind == "maturity":
        anchors.loc[9, "maturity"] += .001
    with pytest.raises(ValueError):
        exercise_adjusted_surface(quotes, anchors, steps=64)

