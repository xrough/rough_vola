"""Source provenance and strict fit-gate tests for the isolated SPY call pilot."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rvlab.market_data import spy_call_pilot as pilot
from rvlab.market_data.references import load_stock_prices, ReferenceValidationError


@pytest.fixture
def stock():
    times = pd.DatetimeIndex([])
    parts = [pd.date_range(f"{day} 09:30", periods=390, freq="min", tz=pilot.NY).tz_convert("UTC") for day in pilot.DATES]
    times = parts[0].append(parts[1:])
    count = len(times)
    prices = 100 + np.arange(count) / 100_000
    return load_stock_prices(pd.DataFrame({"timestamp": times, "spot": prices,
        "open": prices - .01, "high": prices + .02, "low": prices - .02, "close": prices,
        "volume": np.arange(count) + 100, "symbol": "SPY", "source": "unit fixture"}),
        timestamp_convention="minute_bar_start", price_basis="raw")


def _bars(frame):
    return [{"t": row.timestamp.isoformat(), "o": row.open, "h": row.high,
        "l": row.low, "c": row.close, "v": row.volume, "n": 5, "vw": row.close}
        for row in frame.itertuples()]


class Response:
    def __init__(self, payload=None, status=200):
        self.status_code = status
        self.content = json.dumps(payload).encode()
        self.headers = {"Date": "Mon, 26 Oct 2026 00:00:00 GMT", "Content-Type": "application/json",
                        "Set-Cookie": "private", "APCA-API-SECRET-KEY": "not-retained"}

    def json(self):
        return json.loads(self.content)


def _acquire(folder, stock):
    return pilot.acquire_repeat_stock(folder, api_key="unit-key-must-not-be-written",
        api_secret="unit-secret-must-not-be-written", request_get=lambda *a, **k:
        Response({"symbol": "SPY", "bars": _bars(stock), "next_page_token": None}))


def test_repeat_acquisition_preserves_pages_ohlcv_and_no_credentials(tmp_path, stock):
    calls = []
    def get(url, **kwargs):
        calls.append((url, kwargs["params"].copy()))
        chunk, token = (stock.iloc[:390], "page-two") if len(calls) == 1 else (stock.iloc[390:], None)
        return Response({"symbol": "SPY", "bars": _bars(chunk), "next_page_token": token})
    folder = tmp_path / "repeat"
    out = pilot.acquire_repeat_stock(folder, api_key="private-test-key", api_secret="private-test-secret", request_get=get)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["status"] == "complete" and len(manifest["pages"]) == 2
    assert calls[1][1]["page_token"] == "page-two"
    assert set(out) >= {"open", "high", "low", "close", "volume", "vwap", "trade_count"}
    assert pilot._verify_repeat_receipt(folder).equals(out)
    for file in folder.iterdir():
        assert b"private-test-key" not in file.read_bytes()
        assert b"private-test-secret" not in file.read_bytes()
        assert b"not-retained" not in file.read_bytes()
    assert "Set-Cookie" not in manifest["pages"][0]["response_headers"]
    with pytest.raises(FileExistsError):
        _acquire(folder, stock)


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_provider_errors_are_sanitized_and_never_complete(tmp_path, status):
    folder = tmp_path / str(status)
    with pytest.raises(ReferenceValidationError, match=f"HTTP {status}"):
        pilot.acquire_repeat_stock(folder, api_key="private-key", api_secret="private-secret",
            request_get=lambda *a, **k: Response({"error": "private-secret"}, status))
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "private-secret" not in (folder / "manifest.json").read_text()
    assert not (folder / "page_000.json").exists()
    assert not (folder / "stock_prices.parquet").exists()


def test_transport_error_cannot_write_request_credentials(tmp_path):
    def broken(*args, **kwargs):
        raise RuntimeError("private-secret in HTTP request")
    with pytest.raises(ReferenceValidationError, match="transport"):
        pilot.acquire_repeat_stock(tmp_path / "repeat", api_key="key", api_secret="private-secret", request_get=broken)
    assert "private-secret" not in (tmp_path / "repeat/manifest.json").read_text()


def test_pagination_loop_and_page_budget_are_identifiable(tmp_path, stock):
    for name, max_pages in [("loop", 5), ("budget", 1)]:
        with pytest.raises(ReferenceValidationError, match="repeated|exceeded"):
            pilot.acquire_repeat_stock(tmp_path / name, api_key="key", api_secret="secret", max_pages=max_pages,
                request_get=lambda *a, **k: Response({"bars": _bars(stock.iloc[:1]), "next_page_token": "again"}))


@pytest.mark.parametrize("change, failed", [
    (lambda x: x.drop(index=0), "complete_2025-08-07"),
    (lambda x: pd.concat([x, x.iloc[[-1]]], ignore_index=True), "unique"),
    (lambda x: x.iloc[::-1], "sorted"),
    (lambda x: x.assign(high=x.low), "ohlc_bounds"),
    (lambda x: x.assign(volume=-1), "valid_prices_volume"),
    (lambda x: x.assign(price_basis="adjusted"), "raw_spy"),
    (lambda x: x.assign(available_at=x.timestamp), "bar_end_availability"),
    (lambda x: x.assign(spot=x.spot + 1), "spot_is_close"),
    (lambda x: x.assign(timestamp=x.timestamp.dt.tz_localize(None)), "explicit_timezones"),
    (lambda x: x.assign(price_time=x.price_time.dt.tz_localize(None)), "explicit_timezones"),
    (lambda x: x.assign(available_at=x.available_at.dt.tz_localize(None)), "explicit_timezones"),
])
def test_structural_stock_failures_have_explicit_checks(stock, change, failed):
    result = pilot.validate_stock_bars(change(stock.copy()))
    assert not result["passed"] and failed in result["failed_checks"]


def test_raw_stock_passes_and_repeated_rth_values_must_match_exactly(stock):
    assert pilot.validate_stock_bars(stock)["passed"]
    assert pilot.compare_repeat_stock(stock, stock)["passed"]
    changed = stock.copy()
    changed.loc[0, "spot"] += .00001
    result = pilot.compare_repeat_stock(stock, changed)
    assert not result["passed"] and result["mismatch_rows"] == 1
    changed = stock.copy()
    changed.loc[1, "volume"] += 1
    assert not pilot.compare_repeat_stock(stock, changed)["passed"]


def test_crosscheck_targets_include_calibration_and_largest_within_day_moves(stock):
    targets = pilot.stock_crosscheck_targets(stock, source_status="unavailable_second_source_HTTP_429")
    assert len(targets) == 12
    assert targets.groupby("session_date").size().eq(4).all()
    calibration = targets.loc[targets.selection_criterion.eq("calibration_1545_bar_close")]
    assert len(calibration) == 3
    assert calibration.price_time.dt.tz_convert(pilot.NY).dt.strftime("%H:%M").eq("15:45").all()
    largest = targets.loc[targets.selection_criterion.str.startswith("largest_")]
    assert largest.previous_timestamp.notna().all()
    assert largest.timestamp.dt.tz_convert(pilot.NY).dt.date.equals(
        largest.previous_timestamp.dt.tz_convert(pilot.NY).dt.date)
    # All increments are equal in dollars; the relative moves are largest at
    # the lowest within-session prices, beginning at 09:31 rather than 09:30.
    first_rank = largest.loc[largest.selection_criterion.str.endswith("rank_1")]
    assert first_rank.timestamp.dt.tz_convert(pilot.NY).dt.strftime("%H:%M").eq("09:31").all()
    assert not targets.independent_intraday_prices_verified.any()
    assert targets.independent_source_status.eq("unavailable_second_source_HTTP_429").all()
    pd.testing.assert_frame_equal(targets, pilot.stock_crosscheck_targets(stock.sample(frac=1, random_state=42),
        source_status="unavailable_second_source_HTTP_429"))


def test_receipt_binds_raw_payload_to_processed_values(tmp_path, stock):
    folder = tmp_path / "repeat"
    _acquire(folder, stock)
    path = folder / "stock_prices.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "open"] -= .001
    frame.to_parquet(path, index=False)
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["stock"] = pilot.fingerprint(path)
    (folder / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReferenceValidationError, match="processed open disagree"):
        pilot._verify_repeat_receipt(folder)


def test_receipt_rejects_changed_feed_even_when_all_hashes_match(tmp_path, stock):
    folder = tmp_path / "repeat"
    _acquire(folder, stock)
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["base_request_params"]["feed"] = "iex"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ReferenceValidationError, match="raw SIP"):
        pilot._verify_repeat_receipt(folder)


def _quotes(day, as_of):
    rows = []
    for expiry in pilot.EXPIRIES:
        for strike in range(94, 107):
            rows.append({"contract_id": f"{expiry}-C-{strike}", "timestamp": as_of,
                "available_at": as_of, "expiry_at": pd.Timestamp(f"{expiry} 16:00", tz=pilot.NY).tz_convert("UTC"),
                "strike": float(strike), "is_call": True, "bid": 2., "ask": 2.1,
                "bid_size": 5, "ask_size": 5, "quote_valid": True, "in_regular_session": True})
    return pd.DataFrame(rows)


def test_selection_is_deterministic_disjoint_and_all_calls():
    quotes = _quotes(pilot.DATES[0], pd.Timestamp("2025-08-07 19:45", tz="UTC"))
    selected = pilot.select_call_contracts(quotes, 100.)
    shuffled = pilot.select_call_contracts(quotes.sample(frac=1, random_state=42), 100.)
    pd.testing.assert_frame_equal(selected, shuffled)
    assert len(selected) == 18 and selected.contract_id.nunique() == 18
    assert selected.is_call.all() and selected.exercise_style.eq("american").all()
    assert selected.split.value_counts().to_dict() == {"train": 12, "heldout": 6}
    assert set(selected.loc[selected.split.eq("train"), "contract_id"]).isdisjoint(
        selected.loc[selected.split.eq("heldout"), "contract_id"])
    assert selected.loc[selected.split.eq("heldout"), "target_log_moneyness"].abs().eq(.02).all()


def test_selection_uses_strike_then_contract_id_for_ties():
    quotes = _quotes(pilot.DATES[0], pd.Timestamp("2025-08-07 19:45", tz="UTC"))
    twin = quotes.iloc[[0]].copy()
    twin["contract_id"] = "000"
    quotes = pd.concat([quotes, twin], ignore_index=True)
    # This duplicate-strike identity wins a deterministic equality tie.
    result = pilot.select_call_contracts(quotes, 94 / np.exp(-.03))
    assert result.contract_id.iloc[0] == "000"


def _rate_table():
    rows = []
    for day, vintage in zip(pilot.DATES, ("2025-08-06", "2025-08-07", "2025-08-08")):
        for series, tenor in (("SOFR", 1), ("DGS1MO", 30), ("DGS3MO", 91), ("DGS6MO", 182)):
            rows.append({"pilot_as_of": f"{day}T19:45:00Z", "vintage": vintage,
                "available_at": f"{day}T04:00:00Z", "observation_date": vintage,
                "series": series, "tenor_days": tenor, "rate": .04, "source": "test ALFRED"})
    return pd.DataFrame(rows)


def test_rate_selection_cannot_use_current_day_or_future_vintage():
    rates = _rate_table()
    as_of = pd.Timestamp("2025-08-07T19:45:00Z")
    assert len(pilot.pilot_rate_rows(rates, "2025-08-07", as_of)) == 4
    bad = rates.copy()
    bad.loc[0, "vintage"] = "2025-08-07"
    with pytest.raises(ReferenceValidationError, match="incorrect historical"):
        pilot.pilot_rate_rows(bad, "2025-08-07", as_of)
    bad = rates.copy()
    bad.loc[0, "available_at"] = "2025-08-07T19:46:00Z"
    with pytest.raises(ReferenceValidationError, match="future rate"):
        pilot.pilot_rate_rows(bad, "2025-08-07", as_of)


@pytest.fixture
def project(tmp_path, monkeypatch, stock):
    root = tmp_path / "project"
    original_path = root / pilot.ORIGINAL
    original_path.parent.mkdir(parents=True)
    stock.drop(columns=["open", "high", "low", "close"]).to_parquet(original_path, index=False)
    evidence = root / pilot.EVIDENCE
    div = evidence / "dividend_evidence"
    div.mkdir(parents=True)
    calendar = div / "calendar.pdf"
    calendar.write_bytes(b"unit fixture of archived calendar")
    (div / "historical_calendar_manifest.json").write_text("{}")
    review = {"historical_calendar": {"file": "calendar.pdf", "sha256": pilot.fingerprint(calendar)["sha256"],
        "capture_timestamp_utc": "2025-07-31T05:14:12Z", "expected_events_inside_pilot_horizon": []}}
    (div / "evidence_review.json").write_text(json.dumps(review))
    (div / "evidence_inventory.json").write_text(json.dumps({"files": [{"path": "calendar.pdf", "sha256": pilot.fingerprint(calendar)["sha256"]}]}))
    rate = evidence / "rate_evidence"
    rate.mkdir()
    rates_path = rate / "pilot_prior_vintage_rates_candidate.csv"
    _rate_table().to_csv(rates_path, index=False)
    (rate / "evidence_inventory.json").write_text(json.dumps({"files": [{"path": rates_path.name, "sha256": pilot.fingerprint(rates_path)["sha256"]}]}))
    option_audit_rows = []
    for day in pilot.DATES:
        directory = root / "data/processed/spy/options" / f"session_date={day}"
        directory.mkdir(parents=True)
        quotes = _quotes(day, pd.Timestamp(f"{day}T19:45:00Z"))
        quotes_path = directory / "quotes.parquet"
        quotes.to_parquet(quotes_path, index=False)
        manifest_path = directory / "manifest.json"
        manifest_path.write_text(json.dumps({"rows_written": len(quotes)}))
        option_audit_rows.append({"session_date": day, "status": "pass", "checks": {"fixture_raw_fields_match": True},
            "counts": {"raw_rows": len(quotes)}, "manifest_sha256": pilot.fingerprint(manifest_path)["sha256"],
            "parquet": {**pilot.fingerprint(quotes_path), "footer_rows": len(quotes)}})
    option_evidence = evidence / "option_evidence"
    option_evidence.mkdir()
    receipt_path = option_evidence / "option_pilot_audit.json"
    receipt_path.write_text(json.dumps({"status": "pass", "days": option_audit_rows}))
    (option_evidence / "evidence_inventory.json").write_text(json.dumps({
        receipt_path.name: {"sha256": pilot.fingerprint(receipt_path)["sha256"], "bytes": receipt_path.stat().st_size}}))
    import rvlab.market_data.pipeline
    monkeypatch.setattr(rvlab.market_data.pipeline, "read_option_window", lambda path, as_of: _quotes("", as_of))
    readiness = root / "data/processed/spy/readiness.json"
    readiness.write_text('{"calibration_ready":false,"hedging_ready":false}')
    return root


def test_pending_inputs_cannot_be_promoted_and_resume_without_rewriting(tmp_path, project, stock):
    campaign = tmp_path / "campaign"
    canonical = (project / "data/processed/spy/readiness.json").read_bytes()
    pending = pilot.load_spy_call_pilot(campaign, root=project)
    assert not pending["ready"]
    assert not pending["audit"].calibration_allowed.any()
    sampling = pending["exclusions"].loc[pending["exclusions"].snapshot_exclusion_reason.eq(
        "eligible_call_outside_fixed_demonstration_selection")]
    assert len(sampling) == 21 * 6
    assert sampling.exclusion_kind.eq("sampling").all()
    assert pending["audit"].sampling_excluded_eligible_calls.eq(21).all()
    pending_receipt = campaign / "inputs/pending/input_gate_pending.json"
    before = pending_receipt.read_bytes()
    with pytest.raises(ReferenceValidationError, match="repeat acquisition"):
        pilot.load_spy_call_pilot(tmp_path / "other", root=project, allow_pending=False)
    _acquire(project / pilot.REPEAT, stock)
    accepted = pilot.load_spy_call_pilot(campaign, root=project)
    assert accepted["ready"] and accepted["audit"].calibration_allowed.all()
    assert (campaign / "inputs/input_gate_accepted.json").exists()
    assert pending_receipt.read_bytes() == before
    assert (project / "data/processed/spy/readiness.json").read_bytes() == canonical
    assert not accepted["audit"].hedging_ready.any()
    for day in pilot.DATES:
        assert list(accepted["panels"][day].contract_id) == list(accepted["previous_panels"][day].contract_id)
        assert not accepted["markets"][day].dividends.events
    sidecar = json.loads((campaign / "inputs/dividend_assumption.json").read_text())
    assert sidecar["events"] == [] and sidecar["assumption_accepted"]
    assert not sidecar["complete_notice_feed_verified"] and not sidecar["realized_future_dividends_used"]
    repeated = pilot.load_spy_call_pilot(campaign, root=project)
    assert repeated["diagnostics"] == accepted["diagnostics"]


def test_changed_repeat_blocks_market_fit_gate(tmp_path, project, stock):
    changed = stock.copy()
    changed.loc[0, ["spot", "close"]] += .000001
    _acquire(project / pilot.REPEAT, changed)
    bundle = pilot.load_spy_call_pilot(tmp_path / "campaign", root=project)
    assert not bundle["ready"] and bundle["diagnostics"]["repeat_comparison"]["mismatch_rows"] == 1


def test_options_bound_to_the_exact_independently_audited_bytes(project):
    receipt = pilot.verify_pilot_options(project)
    assert len(receipt["current_option_files"]) == 3
    path = project / "data/processed/spy/options/session_date=2025-08-07/quotes.parquet"
    altered = pd.read_parquet(path)
    altered.loc[0, "bid"] += .01
    altered.to_parquet(path, index=False)
    with pytest.raises(ReferenceValidationError, match="independently audited bytes"):
        pilot.verify_pilot_options(project)


def test_option_audit_inventory_and_counts_are_checked(project):
    folder = project / pilot.EVIDENCE / "option_evidence"
    path = folder / "option_pilot_audit.json"
    receipt = json.loads(path.read_text())
    receipt["days"][0]["counts"]["raw_rows"] += 1
    path.write_text(json.dumps(receipt))
    with pytest.raises(ReferenceValidationError, match="hash mismatch"):
        pilot.verify_pilot_options(project)
    (folder / "evidence_inventory.json").write_text(json.dumps({
        path.name: {"sha256": pilot.fingerprint(path)["sha256"]}}))
    with pytest.raises(ReferenceValidationError, match="row count"):
        pilot.verify_pilot_options(project)


def test_missing_original_data_has_no_synthetic_fallback(tmp_path):
    with pytest.raises(FileNotFoundError):
        pilot.load_spy_call_pilot(tmp_path / "campaign", root=tmp_path / "empty")
