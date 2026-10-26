"""Local pricing-demo inputs remain deterministic, disjoint, and explicit proxies."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from rvlab.experiments._pricing_data import (
    DEFAULT_EXPIRIES, INPUT_MODE, implied_volatility, load_spy_demo,
    parity_forward, select_spy_snapshot,
)
from rvlab.market_data.snapshots import build_calibration_snapshot
from rvlab.pricing import BlackScholesParams, black_scholes_price


AS_OF = pd.Timestamp("2025-08-07 19:45Z")


def rates():
    return pd.DataFrame({"observation_date": ["2025-08-05"] * 3,
        "available_at": pd.to_datetime(["2025-08-06 21:00Z"] * 3),
        "tenor_days": [30, 91, 182], "rate": [.04] * 3,
        "series": ["DGS1MO", "DGS3MO", "DGS6MO"]})


def options():
    rows = []
    for expiry in DEFAULT_EXPIRIES:
        expiry_at = pd.Timestamp(f"{expiry} 16:00", tz="America/New_York").tz_convert("UTC")
        maturity = (expiry_at - AS_OF).total_seconds() / (365 * 86400)
        discount = np.exp(-.04 * maturity)
        for strike in np.arange(586., 676., 2.):
            for call in (True, False):
                mid = black_scholes_price(BlackScholesParams(.22), forward=630.,
                    discount=discount, strike=strike, maturity=maturity, is_call=call)
                half = min(.01, mid * .02)
                rows.append({"contract_id": f"SPY{expiry}{'C' if call else 'P'}{strike}",
                    "timestamp": AS_OF, "available_at": AS_OF, "expiry_at": expiry_at,
                    "strike": strike, "is_call": call, "bid": mid - half,
                    "ask": mid + half, "bid_size": 10, "ask_size": 10,
                    "quote_valid": True, "in_regular_session": True})
    return pd.DataFrame(rows)


def test_bs_implied_volatility_broadcast_roundtrip_and_scalar():
    strike = np.asarray([95., 100., 110.])
    maturity = np.asarray([[.1], [1.]])
    calls = np.asarray([True, False, True])
    prices = black_scholes_price(BlackScholesParams(.27), forward=102., discount=.97,
        strike=strike, maturity=maturity, is_call=calls)
    actual = implied_volatility(prices, forward=102., discount=.97,
        strike=strike, maturity=maturity, is_call=calls)
    assert actual.shape == (2, 3)
    np.testing.assert_allclose(actual, .27, atol=1e-11)
    assert isinstance(implied_volatility(float(prices[0, 0]), forward=102.,
        discount=.97, strike=95., maturity=.1), float)


def test_bs_inverse_intrinsic_expiry_and_invalid_prices():
    args = dict(forward=100., discount=.95, strike=105., maturity=.2, is_call=False)
    assert implied_volatility(4.75, **args) == 0.
    assert implied_volatility(4.75, **(args | {"maturity": 0})) == 0.
    with pytest.raises(ValueError, match="row position 0.*outside European bounds"):
        implied_volatility(4.74, **args)
    with pytest.raises(ValueError, match="no finite implied volatility"):
        implied_volatility(99.75, **args)
    with pytest.raises(ValueError, match="expiry price"):
        implied_volatility(5., **(args | {"maturity": 0}))
    with pytest.raises(ValueError, match="boolean"):
        implied_volatility(5., **(args | {"is_call": 1}))
    with pytest.raises(ValueError, match="finite and positive"):
        implied_volatility(5., **(args | {"discount": -1}))
    with pytest.raises(ValueError, match="real numeric"):
        implied_volatility(True, **args)


def test_discounted_parity_formula_and_validation():
    assert parity_forward(12., 3., 100., .9) == pytest.approx(110.)
    np.testing.assert_allclose(parity_forward([12., 3.], [3., 12.], 100., .9), [110., 90.])
    with pytest.raises(ValueError, match="discount"):
        parity_forward(2., 1., 100., 0.)
    with pytest.raises(ValueError, match="nonnegative"):
        parity_forward(-1., 2., 100., 1.)


def test_spy_selection_contract_splits_anchor_separation_and_discount():
    sample = select_spy_snapshot(options(), AS_OF, rates())
    assert len(sample.quotes) == 18
    assert sample.quotes.split.value_counts().to_dict() == {"train": 12, "heldout": 6}
    assert len(sample.anchors) == 21
    for expiry in DEFAULT_EXPIRIES:
        quotes = sample.quotes.loc[sample.quotes.expiry_date == expiry]
        anchors = sample.anchors.loc[sample.anchors.expiry_date == expiry]
        assert not set(quotes.strike) & set(anchors.strike)
        assert quotes.contract_id.is_unique
        assert set(quotes.loc[quotes.split == "heldout", "target_log_moneyness"]) == {-.02, .02}
    np.testing.assert_allclose(sample.quotes.forward, 630., atol=1e-12)
    np.testing.assert_allclose(sample.quotes.discount, np.exp(-.04 * sample.quotes.maturity))
    assert not sample.diagnostics.parity_intersection_empty.any()
    assert sample.quotes.input_mode.eq(INPUT_MODE).all()
    assert not sample.quotes.eligible_for_hedging.any()
    assert not sample.provenance["independent_stock_observation"]
    assert not sample.provenance["strict_market_data_ready"]
    assert "spot" not in sample.quotes


def test_sampling_does_not_depend_on_input_order_or_leak_future_rates():
    original = options()
    future_rates = rates().assign(rate=.9, available_at=AS_OF + pd.Timedelta(seconds=1))
    first = select_spy_snapshot(original, AS_OF, rates())
    second = select_spy_snapshot(original.sample(frac=1., random_state=42), AS_OF,
        pd.concat([rates(), future_rates], ignore_index=True))
    pd.testing.assert_frame_equal(first.quotes, second.quotes)
    pd.testing.assert_frame_equal(first.anchors, second.anchors)


def test_latest_invalid_quote_is_excluded_and_unavailable_quote_not_used():
    original = options()
    base = original.iloc[[0]].copy()
    original.loc[0, ["timestamp", "available_at"]] = AS_OF - pd.Timedelta(seconds=30)
    invalid = base.assign(quote_valid=False)
    future = original.iloc[[1]].assign(available_at=AS_OF + pd.Timedelta(seconds=1), bid=1000., ask=1001.)
    sample = select_spy_snapshot(pd.concat([original, invalid, future], ignore_index=True), AS_OF, rates())
    excluded = sample.exclusions.loc[sample.exclusions.contract_id == base.contract_id.iloc[0]]
    assert excluded.snapshot_exclusion_reason.str.contains("invalid_quote").any()
    assert not sample.quotes.contract_id.eq(base.contract_id.iloc[0]).any()
    assert not sample.anchors.contract_id_call.eq(base.contract_id.iloc[0]).any()
    assert sample.anchors.market_price_put.max() < 1000


def test_quality_screens_are_recorded_before_anchor_construction():
    original = options()
    original.loc[0, "bid_size"] = 0
    original.loc[1, "bid"] = 0
    original.loc[2, "ask"] *= 4
    sample = select_spy_snapshot(original, AS_OF, rates())
    reasons = ";".join(sample.exclusions.snapshot_exclusion_reason)
    assert "insufficient_quoted_size" in reasons
    assert "invalid_quote_values" in reasons
    assert "relative_spread_above_25pct" in reasons


def test_empty_european_parity_intersection_is_explicit_and_keeps_proxy():
    original = options()
    shift = original.strike.mod(8) / 20
    original.loc[original.is_call, ["bid", "ask"]] += shift.loc[original.is_call].to_numpy()[:, None]
    sample = select_spy_snapshot(original, AS_OF, rates())
    assert sample.diagnostics.parity_intersection_empty.all()
    assert sample.diagnostics.anchor_dispersion.min() > .1
    assert len(sample.quotes) == 18
    assert not sample.quotes.eligible_for_hedging.any()


def test_missing_pairs_and_missing_rates_raise_helpful_errors():
    original = options()
    with pytest.raises(ValueError, match="need 7 eligible call/put pairs"):
        select_spy_snapshot(original.loc[original.is_call], AS_OF, rates())
    with pytest.raises(ValueError, match="missing dated Treasury"):
        select_spy_snapshot(original, AS_OF, None)
    with pytest.raises(ValueError, match="timezone-aware"):
        select_spy_snapshot(original, "2025-08-07 15:45", rates())
    with pytest.raises(ValueError, match="bid_size"):
        select_spy_snapshot(original.drop(columns="bid_size"), AS_OF, rates())


def test_demo_does_not_change_strict_market_readiness_or_input_frames():
    original = options()
    original_copy = original.copy(deep=True)
    before = build_calibration_snapshot(original, AS_OF, rates=rates())
    select_spy_snapshot(original, AS_OF, rates())
    after = build_calibration_snapshot(original, AS_OF, rates=rates())
    assert not before.ready and not after.ready
    assert before.missing_inputs == after.missing_inputs
    assert any("independent stock" in value for value in after.missing_inputs)
    pd.testing.assert_frame_equal(original, original_copy)


def test_loader_missing_data_has_no_synthetic_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("RVLAB_FORCE_SYNTHETIC", "1")
    with pytest.raises(FileNotFoundError, match="Missing prepared Treasury"):
        load_spy_demo(tmp_path)
    (tmp_path / "references").mkdir()
    rates().to_parquet(tmp_path / "references" / "rates.parquet")
    with pytest.raises(FileNotFoundError, match="Missing prepared SPY"):
        load_spy_demo(tmp_path)


def test_loader_reads_one_minute_window_and_records_existing_source_hash(tmp_path, monkeypatch):
    from rvlab.market_data import pipeline
    monkeypatch.setenv("RVLAB_FORCE_SYNTHETIC", "1")
    (tmp_path / "references").mkdir()
    rates().to_parquet(tmp_path / "references" / "rates.parquet")
    directory = tmp_path / "options" / "session_date=2025-08-07"
    directory.mkdir(parents=True)
    options().to_parquet(directory / "quotes.parquet")
    (directory / "manifest.json").write_text(json.dumps({"source": {"sha256": "existing-dbn-hash"}}))
    calls = []
    original_read = pipeline.read_option_window
    def read(path, as_of):
        calls.append((path, as_of))
        return original_read(path, as_of)
    monkeypatch.setattr(pipeline, "read_option_window", read)
    result = load_spy_demo(tmp_path, dates=["2025-08-07"])
    assert len(calls) == 1 and calls[0][1] == AS_OF
    assert len(result.quotes) == 18
    assert result.provenance["source_days"][0]["source"]["sha256"] == "existing-dbn-hash"
    assert len(result.provenance["rates_file"]["sha256"]) == 64
    assert not (tmp_path / "readiness.json").exists()
