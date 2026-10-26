"""Availability alignment, explicit reference gaps, and model-input integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rvlab.market_data.snapshots import (
    MarketDataNotReady, build_calibration_snapshot, candidate_snapshot,
)
from rvlab.market_data import snapshots
from rvlab.pricing import BlackScholesParams, price


AS_OF = pd.Timestamp("2025-11-03 15:00:00Z")


def _options():
    expiry = pd.to_datetime(["2025-11-21 21:00Z", "2025-11-28 18:00Z",
                             "2025-12-05 21:00Z"], utc=True)
    maturity = (expiry - AS_OF).total_seconds().to_numpy() / snapshots.YEAR_SECONDS
    discount = np.exp(-0.04 * maturity)
    forward = 600 / discount
    strike = np.array([605.0, 610.0, 620.0])
    prices = price(BlackScholesParams(0.2), forward=forward, discount=discount,
                   strike=strike, maturity=maturity, is_call=True)
    return pd.DataFrame({
        "contract_id": ["SPY   251121C00605000", "SPY   251128C00610000", "SPY   251205C00620000"],
        "symbol": "SPY", "timestamp": AS_OF - pd.Timedelta(seconds=10),
        "available_at": AS_OF - pd.Timedelta(seconds=10), "expiry_at": expiry,
        "strike": strike, "is_call": True, "bid": prices - 0.05,
        "ask": prices + 0.05, "bid_size": 10, "ask_size": 12,
        "quote_valid": True, "in_regular_session": True,
    })


def _stock():
    return pd.DataFrame({"timestamp": [AS_OF - pd.Timedelta(minutes=1)],
        "available_at": [AS_OF], "price_time": [AS_OF], "spot": [600.0],
        "price_basis": ["raw"], "source": ["synthetic independent stock"]})


def _rates():
    return pd.DataFrame({"observation_date": [pd.Timestamp("2025-10-31").date()] * 3,
        "available_at": pd.to_datetime(["2025-10-31 22:00Z"] * 3),
        "tenor_days": [30, 90, 180], "rate": [0.04, 0.04, 0.04],
        "series": ["DGS1MO", "DGS3MO", "DGS6MO"], "source": "synthetic Treasury"})


def _dividends():
    # Explicit no-event expectation, with its coverage established by caller.
    table = pd.DataFrame({"ex_date": pd.Series(dtype=object),
        "amount": pd.Series(dtype=float),
        "available_at": pd.Series(dtype="datetime64[ns, UTC]")})
    table.attrs["coverage_through"] = "2025-12-31"
    return table


def _build(options=None, **kwargs):
    arguments = {"stock": _stock(), "rates": _rates(), "dividends": _dividends()}
    arguments.update(kwargs)
    return build_calibration_snapshot(_options() if options is None else options,
                                       AS_OF, **arguments)


def test_ready_surface_handoff_uses_american_spot_market_and_contract_identity():
    original = _options().iloc[[2, 0, 1]].copy()
    snapshot = _build(original)
    assert snapshot.ready
    assert len(snapshot.quotes) == 3
    np.testing.assert_allclose(snapshot.forward, snapshot.quotes.forward)
    np.testing.assert_allclose(snapshot.discount, snapshot.quotes.discount)
    assert snapshot.quotes.contract_id.is_unique
    arguments = snapshot.calibration_inputs()
    assert set(arguments) == {"quotes", "market", "exercise_style"}
    assert arguments["exercise_style"] == "american"
    assert arguments["market"].spot == 600
    assert arguments["market"].dividends.events == ()
    assert arguments["market"].dividends.coverage >= snapshot.quotes.maturity.max()
    np.testing.assert_allclose(arguments["market"].discount_curve.discount(
        snapshot.quotes.maturity.to_numpy()), snapshot.discount)
    assert arguments["quotes"].exercise_style.eq("american").all()
    assert "forward" not in original
    assert snapshot.provenance["stock"]["price_time"] == AS_OF
    arguments["quotes"].loc[0, "strike"] = -1
    assert snapshot.quotes.strike.iloc[0] > 0


def test_snapshot_handoff_reaches_american_calibrator_without_forward_arguments():
    from rvlab.pricing import FDConfig, calibrate_vanilla_surface

    snapshot = _build(_options().iloc[[2, 0, 1]])
    result = calibrate_vanilla_surface(initial_params=BlackScholesParams(.2),
        bounds={"volatility": (.2, .2)}, config=FDConfig(spot_steps=100, time_steps=100),
        **snapshot.calibration_inputs())
    assert result.exercise_style == "american"
    assert result.diagnostics["backend"] == "american_vanilla_fd"
    assert np.isfinite(result.fitted_quotes.model_price).all()
    assert np.isfinite(result.fitted_quotes.delta).all()
    assert result.fitted_quotes.exercise_style.eq("american").all()
    assert list(result.fitted_quotes.contract_id) == list(snapshot.quotes.contract_id)
    assert result.ready == (result.optimizer_success and result.price_stable)


def test_missing_references_preserve_candidates_without_ready_market_arrays():
    snapshot = build_calibration_snapshot(_options(), AS_OF)
    assert not snapshot.ready and len(snapshot.quotes) == 3
    assert len(snapshot.missing_inputs) == 3
    assert np.isnan(snapshot.forward).all()
    assert snapshot.quotes.exercise_style.eq("american").all()
    with pytest.raises(MarketDataNotReady, match="independent stock"):
        snapshot.require_ready()
    with pytest.raises(MarketDataNotReady):
        snapshot.calibration_inputs()


def test_no_future_quote_or_bar_leakage_and_bar_price_time_used_for_age():
    options = _options()
    future = options.iloc[[0]].copy()
    future["available_at"] = AS_OF + pd.Timedelta(seconds=1)
    future["bid"] = 999.0
    combined = pd.concat([options, future], ignore_index=True)
    stock = pd.concat([_stock(), _stock().assign(available_at=AS_OF + pd.Timedelta(seconds=1),
                                               spot=999.0)], ignore_index=True)
    snapshot = _build(combined, stock=stock, max_stock_age="1s")
    assert snapshot.ready
    assert snapshot.quotes.spot.eq(600).all()
    assert snapshot.quotes.bid.max() < 999


def test_latest_bad_quote_does_not_resurrect_older_good_quote():
    options = _options()
    invalid = options.iloc[[0]].copy()
    invalid["timestamp"] = AS_OF
    invalid["available_at"] = AS_OF
    invalid["quote_valid"] = False
    snapshot = _build(pd.concat([options, invalid], ignore_index=True))
    assert snapshot.ready
    assert len(snapshot.quotes) == 2
    assert "invalid_quote" in snapshot.exclusions.snapshot_exclusion_reason.iloc[0]


def test_stale_option_stock_and_outside_regular_session_are_explicit():
    options = _options()
    options.loc[0, "timestamp"] = AS_OF - pd.Timedelta(minutes=2)
    options.loc[1, "in_regular_session"] = False
    candidate = candidate_snapshot(options, AS_OF)
    assert len(candidate) == 1
    assert candidate.attrs["exclusion_counts"] == {"stale_option_quote": 1, "outside_regular_session": 1}
    snapshot = _build(options, stock=_stock().assign(price_time=AS_OF - pd.Timedelta(minutes=2)))
    assert not snapshot.ready
    assert "stale" in " ".join(snapshot.missing_inputs)


def test_actual_expiry_timestamp_sets_intraday_maturity_not_midnight():
    candidate = candidate_snapshot(_options(), AS_OF)
    expected = (pd.Timestamp("2025-11-28 18:00Z") - AS_OF).total_seconds() / snapshots.YEAR_SECONDS
    assert candidate.maturity.iloc[1] == pytest.approx(expected)
    boundary = _options().iloc[[0]].copy()
    boundary["expiry_at"] = AS_OF + pd.Timedelta(days=7) - pd.Timedelta(seconds=1)
    assert candidate_snapshot(boundary, AS_OF).empty


def test_rate_publication_availability_and_log_discount_interpolation():
    rates = _rates()
    rates["rate"] = [0.03, 0.05, 0.06]
    unavailable = rates.copy().assign(available_at=AS_OF + pd.Timedelta(seconds=1), rate=0.8)
    snapshot = _build(rates=pd.concat([rates, unavailable], ignore_index=True))
    times = snapshot.quotes.maturity.to_numpy()
    expected = np.exp(np.interp(times, [0, 30 / 365, 90 / 365, 180 / 365],
                                         [0, -0.03 * 30 / 365, -0.05 * 90 / 365, -0.06 * 180 / 365]))
    np.testing.assert_allclose(snapshot.discount, expected)
    assert snapshot.ready
    for row in snapshot.provenance["rates"]["observations"]:
        assert row["rate"] < 0.1


def test_sofr_does_not_substitute_for_the_term_curve():
    snapshot = _build(rates=_rates().assign(series="SOFR"))
    assert not snapshot.ready
    assert "SOFR is financing only" in " ".join(snapshot.missing_inputs)


def test_stale_and_noncovering_treasury_curves_are_not_extrapolated():
    stale = _rates().assign(observation_date=pd.Timestamp("2025-10-01").date())
    assert "stale" in " ".join(_build(rates=stale).missing_inputs)
    short = _rates().iloc[[0]].copy()
    assert "does not cover" in " ".join(_build(rates=short).missing_inputs)


def test_dividends_need_explicit_expectation_horizon_even_with_empty_calendar():
    dividends = _dividends()
    dividends.attrs.clear()
    snapshot = _build(dividends=dividends)
    assert not snapshot.ready
    assert "dividend expectation coverage" in " ".join(snapshot.missing_inputs)
    assert _build(dividends=dividends, dividend_coverage_through="2025-12-31").ready


def _history():
    return pd.DataFrame({"ex_date": ["2024-12-20", "2025-03-21", "2025-06-20", "2025-09-19"],
        "amount": [1.5] * 4,
        "available_at": pd.to_datetime(["2025-01-31 21:00Z", "2025-04-30 20:00Z",
                                       "2025-07-31 20:00Z", "2025-10-31 20:00Z"]),
        "is_forecast": False})


def test_trailing_dividend_proxy_never_substitutes_for_american_cash_schedule():
    history = _history()
    assert not _build(dividends=history).ready
    snapshot = _build(dividends=history, allow_dividend_proxy=True)
    assert not snapshot.ready
    assert "unsupported for American" in " ".join(snapshot.missing_inputs)
    assert snapshot.market is None
    history.loc[3, "available_at"] = AS_OF + pd.Timedelta(days=1)
    incomplete = _build(dividends=history, allow_dividend_proxy=True)
    assert not incomplete.ready
    assert "unsupported for American" in " ".join(incomplete.missing_inputs)


def test_unavailable_dividend_dates_never_filter_contracts_or_enter_schedule():
    dividends = pd.DataFrame({"ex_date": ["2025-11-25"], "amount": [2.0],
        "available_at": pd.to_datetime(["2025-12-01 21:00Z"])})
    dividends.attrs["coverage_through"] = "2025-12-31"
    snapshot = _build(dividends=dividends)
    assert snapshot.ready and len(snapshot.quotes) == 3
    assert snapshot.provenance["ex_post_dividend_exclusion_count"] == 0
    assert snapshot.exclusions.empty
    assert snapshot.market.dividends.events == ()
    # Caller explicitly asserted forecast coverage; an unavailable future
    # realized event cannot alter that historical schedule or selection.
    np.testing.assert_allclose(snapshot.forward, 600 / snapshot.discount)


def test_market_screens_keep_quote_and_market_array_alignment():
    options = _options()
    options.loc[0, "bid_size"] = 0
    options.loc[1, "ask"] += 100
    snapshot = _build(options)
    assert snapshot.ready and len(snapshot.quotes) == 1
    assert snapshot.quotes.contract_id.iloc[0] == options.contract_id.iloc[2]
    np.testing.assert_allclose(snapshot.forward, snapshot.quotes.forward)
    reasons = " ".join(snapshot.exclusions.snapshot_exclusion_reason)
    assert "insufficient_bid_size" in reasons and "wide_relative_spread" in reasons


def test_itm_and_above_european_bound_quotes_are_not_exercise_screened():
    options = _options()
    options.loc[0, "strike"] = 580
    options.loc[1, ["bid", "ask"]] = [700.0, 700.1]
    snapshot = _build(options)
    assert snapshot.ready and len(snapshot.quotes) == 3
    assert snapshot.exclusions.empty
    # Snapshot preparation checks quote quality, not a European model's bounds.
    # The American calibrator can diagnose impossible prices independently.


def test_declared_cash_event_discounted_only_after_its_availability():
    dividends = pd.DataFrame({"ex_date": ["2025-11-25"], "amount": [2.0],
        "available_at": pd.to_datetime(["2025-10-30 21:00Z"])})
    dividends.attrs["coverage_through"] = "2025-12-31"
    snapshot = _build(dividends=dividends)
    assert snapshot.ready and len(snapshot.quotes) == 3
    assert snapshot.provenance["ex_post_dividend_exclusion_count"] == 0
    crossed = snapshot.quotes.loc[snapshot.quotes.expiry_at >= pd.Timestamp("2025-11-25 14:30Z")]
    dividend_time = (pd.Timestamp("2025-11-25 14:30Z") - AS_OF).total_seconds() / snapshots.YEAR_SECONDS
    np.testing.assert_allclose(crossed.forward,
        (600 - 2 * np.exp(-0.04 * dividend_time)) / crossed.discount)
    assert snapshot.market.dividends.events[0].amount == 2
    assert snapshot.market.dividends.events[0].time == pytest.approx(dividend_time)


def test_crr_exercise_diagnostic_does_not_control_american_eligibility(monkeypatch):
    monkeypatch.setattr(snapshots, "_american_premium", lambda *args: 1.0)
    material = _build()
    assert material.ready and len(material.quotes) == 3


def test_crr_reference_failure_is_not_called_during_american_preparation(monkeypatch):
    def fail(*args):
        raise ValueError("invalid CRR probability")
    monkeypatch.setattr(snapshots, "_american_premium", fail)
    snapshot = _build()
    assert snapshot.ready
    assert "exercise_premium" not in snapshot.quotes


def test_american_call_without_dividends_matches_european_and_put_has_premium():
    call = snapshots._american_premium(100, 105, 0.3, 0.05, 0, 0.2, True, 200)
    put = snapshots._american_premium(100, 105, 0.3, 0.05, 0, 0.2, False, 200)
    assert call == pytest.approx(0, abs=1e-12)
    assert put > 0.1


@pytest.mark.parametrize("field,value", [("min_dte", -1), ("max_dte", 1),
    ("max_quote_age", "0s"), ("max_stock_age", "-1s"),
    ("max_stock_age", "61s"),
    ("max_relative_spread", -1), ("min_quote_size", -1)])
def test_invalid_configuration_is_rejected(field, value):
    with pytest.raises(ValueError):
        _build(**{field: value})


def test_naive_timestamps_are_rejected_instead_of_assumed_utc():
    with pytest.raises(ValueError, match="timezone-aware"):
        candidate_snapshot(_options(), "2025-11-03 15:00")
    options = _options()
    options["timestamp"] = pd.Timestamp("2025-11-03 15:00")
    with pytest.raises(ValueError, match="row position 0"):
        candidate_snapshot(options, AS_OF)


def test_unadjusted_stock_is_required_and_explicit_numeric_booleans_rejected():
    snapshot = _build(stock=_stock().assign(price_basis="split_and_dividend_adjusted"))
    assert not snapshot.ready
    assert "raw/unadjusted" in " ".join(snapshot.missing_inputs)
    options = _options().assign(is_call=1)
    with pytest.raises(ValueError, match="is_call must contain booleans"):
        candidate_snapshot(options, AS_OF)


def test_stock_requires_explicit_price_basis_instead_of_assuming_raw():
    snapshot = _build(stock=_stock().drop(columns="price_basis"))
    assert not snapshot.ready
    assert "explicit raw/unadjusted price_basis" in " ".join(snapshot.missing_inputs)


@pytest.mark.parametrize("as_of,timestamp,price_time", [
    # A premarket bar that ends exactly at the regular open contains no RTH
    # trading, despite satisfying a one-minute price-age check.
    ("2025-11-03 14:30:00Z", "2025-11-03 14:29:00Z", "2025-11-03 14:30:00Z"),
    ("2025-11-03 14:30:10Z", "2025-11-03 14:29:00Z", "2025-11-03 14:29:59Z"),
    # Early close: the Friday after Thanksgiving closes at 18:00 UTC.
    ("2025-11-28 18:00:30Z", "2025-11-28 17:59:00Z", "2025-11-28 18:00:00Z"),
    # Overnight and holiday observations cannot form a regular-session input.
    ("2025-11-04 00:00:00Z", "2025-11-03 23:59:00Z", "2025-11-04 00:00:00Z"),
    ("2025-11-27 15:00:00Z", "2025-11-27 14:59:00Z", "2025-11-27 15:00:00Z"),
])
def test_stock_regular_session_check_rejects_premarket_overnight_and_holidays(as_of, timestamp, price_time):
    as_of = pd.Timestamp(as_of)
    stock = _stock().assign(timestamp=pd.Timestamp(timestamp),
                            price_time=pd.Timestamp(price_time), available_at=as_of)
    row, reason = snapshots._stock_observation(stock, as_of, pd.Timedelta(minutes=1))
    assert row is None
    assert "session" in reason


@pytest.mark.parametrize("as_of", ["2025-08-07 13:31:00Z", "2025-11-03 14:31:00Z", "2025-11-28 18:00:00Z"])
def test_stock_regular_session_accepts_dst_and_early_close_bar_end(as_of):
    as_of = pd.Timestamp(as_of)
    stock = _stock().assign(timestamp=as_of - pd.Timedelta(minutes=1),
                            price_time=as_of, available_at=as_of)
    row, reason = snapshots._stock_observation(stock, as_of, pd.Timedelta(minutes=1))
    assert reason is None and row is not None


def test_future_dividend_schedule_uses_latest_available_revision_once_and_keeps_audit_rows():
    dividends = pd.DataFrame({
        "ex_date": ["2025-11-25"] * 4,
        "amount": [1.0, 2.0, 2.25, 99.0],
        "is_forecast": [True, True, False, True],
        "available_at": pd.to_datetime(["2025-10-01 20:00Z", "2025-10-15 20:00Z",
                                       "2025-10-31 20:00Z", "2025-11-04 20:00Z"]),
    })
    dividends.attrs["coverage_through"] = "2025-12-31"
    prepared, audit_rows, error = snapshots._dividend_inputs(
        dividends, AS_OF, pd.Timestamp("2025-12-05 21:00Z"), 600, False, None)
    assert error is None
    assert len(audit_rows) == 4
    assert len(prepared["events"]) == 1
    assert prepared["events"].amount.iloc[0] == 2.25
    snapshot = _build(dividends=dividends)
    assert snapshot.provenance["ex_post_dividend_exclusion_count"] == 0
    crossed = snapshot.quotes.loc[snapshot.quotes.expiry_at >= pd.Timestamp("2025-11-25 14:30Z")]
    assert snapshot.exclusions.empty and len(snapshot.quotes) == 3
    dividend_time = (pd.Timestamp("2025-11-25 14:30Z") - AS_OF).total_seconds() / snapshots.YEAR_SECONDS
    np.testing.assert_allclose(crossed.forward,
        (600 - 2.25 * np.exp(-0.04 * dividend_time)) / crossed.discount)


@pytest.mark.parametrize("reverse", [False, True])
def test_actual_dividend_replaces_forecast_at_identical_availability_independent_of_row_order(reverse):
    dividends = pd.DataFrame({
        "ex_date": ["2025-11-25", "2025-11-25"], "amount": [2.25, 2.0],
        "is_forecast": [False, True],
        "available_at": pd.to_datetime(["2025-10-31 20:00Z"] * 2),
    })
    if reverse:
        dividends = dividends.iloc[::-1]
    prepared, _, error = snapshots._dividend_inputs(
        dividends, AS_OF, pd.Timestamp("2025-12-05 21:00Z"), 600, False, "2025-12-31")
    assert error is None
    assert len(prepared["events"]) == 1
    assert prepared["events"].amount.iloc[0] == 2.25


def test_snapshot_cash_schedule_coverage_is_not_overstated_beyond_its_expiries():
    dividends = pd.DataFrame({"ex_date": ["2025-12-19"], "amount": [2.0],
        "available_at": pd.to_datetime(["2025-10-30 21:00Z"])})
    dividends.attrs["coverage_through"] = "2025-12-31"
    snapshot = _build(dividends=dividends)
    assert snapshot.ready and snapshot.market.dividends.events == ()
    assert snapshot.market.dividends.coverage == pytest.approx(snapshot.quotes.maturity.max())


def test_explicit_dividend_event_instant_is_respected_and_past_events_use_post_event_spot():
    dividends = pd.DataFrame({"ex_date": ["2025-11-03", "2025-11-25"],
        "ex_at": pd.to_datetime(["2025-11-03 14:30Z", "2025-11-25 14:31Z"]),
        "amount": [1.0, 2.0],
        "available_at": pd.to_datetime(["2025-10-30 21:00Z"] * 2)})
    dividends.attrs["coverage_through"] = "2025-12-31"
    snapshot = _build(dividends=dividends)
    assert snapshot.ready and snapshot.market.spot == 600
    assert len(snapshot.market.dividends.events) == 1
    expected = (pd.Timestamp("2025-11-25 14:31Z") - AS_OF).total_seconds() / snapshots.YEAR_SECONDS
    assert snapshot.market.dividends.events[0].time == pytest.approx(expected)


def test_missing_quote_size_is_a_row_level_exclusion_even_with_missing_references():
    snapshot = build_calibration_snapshot(_options().drop(columns="ask_size"), AS_OF)
    assert not snapshot.ready and snapshot.quotes.empty
    assert snapshot.exclusions.snapshot_exclusion_reason.eq("missing_ask_size").all()
