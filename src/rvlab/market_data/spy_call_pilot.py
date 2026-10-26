"""Isolated, assumption-labelled SPY American-call research inputs.

The pilot never changes canonical readiness. A separate, provenance-preserving
repeat acquisition is required before market fits may be launched. Credentials
are accepted only in memory by the acquisition function and never serialized.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .references import load_stock_prices, ReferenceValidationError

ROOT = Path(__file__).resolve().parents[3]
DATES = ("2025-08-07", "2025-08-08", "2025-08-11")
EXPIRIES = ("2025-08-22", "2025-08-29", "2025-09-05")
NY = "America/New_York"
ORIGINAL = Path("data/raw/spy_reference/acquisition_pilot_2026-10-26/stock_prices.parquet")
REPEAT = Path("data/raw/spy_reference/acquisition_pilot_2026-10-26_repeat_v1")
EVIDENCE = Path("data/raw/spy_reference/acquisition_pilot_2026-10-23")
ENDPOINT = "https://data.alpaca.markets/v2/stocks/SPY/bars"


def fingerprint(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n").encode()


def _write_once(path, content):
    """Permit deterministic reruns, but never silently replace prior evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Refusing to replace pilot evidence: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(content)


def _write_json(path, value):
    _write_once(path, _json_bytes(value))


def acquire_repeat_stock(destination, *, api_key, api_secret, request_get=None, max_pages=100):
    """Acquire raw SIP OHLCV and all successful response pages into a new folder.

    The destination must not exist. HTTP error bodies are deliberately omitted:
    provider authorization failures are identifiable without recording secrets.
    Partial acquisitions retain a sanitized failure manifest and cannot pass QA.
    """
    if not api_key or not api_secret:
        raise ReferenceValidationError("Both Alpaca credentials are required")
    if isinstance(max_pages, bool) or not isinstance(max_pages, int) or max_pages < 1:
        raise ValueError("max_pages must be a positive integer")
    if request_get is None:
        import requests
        request_get = requests.get
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    params = {"timeframe": "1Min", "start": "2025-08-07T04:00:00+00:00",
              "end": "2025-08-12T04:00:00+00:00", "feed": "sip",
              "adjustment": "raw", "limit": 10000, "sort": "asc"}
    manifest = {"schema": "spy_stock_repeat_v1", "status": "incomplete",
                "provider": "Alpaca", "endpoint": ENDPOINT, "symbol": "SPY",
                "base_request_params": dict(params), "request_header_names":
                ["APCA-API-KEY-ID", "APCA-API-SECRET-KEY"],
                "retrieved_at": datetime.now(timezone.utc).isoformat(), "pages": [],
                "timestamp_convention": "minute_bar_start; available_at is bar end convention, not measured latency",
                "price_basis": "raw", "limitations": [
                    "Minute bars are observed price proxies, not executable quotes.",
                    "Repeated same-provider bars test consistency, not independent source agreement."]}
    rows, tokens = [], set()
    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret}
    try:
        for number in range(max_pages):
            request_params = dict(params)
            try:
                response = request_get(ENDPOINT, params=request_params, headers=headers, timeout=90)
            except Exception:
                # Request exceptions can retain a request object with headers.
                raise ReferenceValidationError("Alpaca transport request failed; no credentials logged") from None
            status = int(response.status_code)
            record = {"number": number, "request_params": request_params, "status_code": status,
                      "response_headers": {k: v for k, v in response.headers.items()
                      if k.lower() in {"date", "content-type", "x-request-id", "x-ratelimit-limit",
                                       "x-ratelimit-remaining", "x-ratelimit-reset"}}}
            manifest["pages"].append(record)
            if status != 200:
                raise ReferenceValidationError(f"Alpaca historical SIP request returned HTTP {status}; body omitted")
            payload = response.json()
            raw = response.content
            page_path = destination / f"page_{number:03d}.json"
            _write_once(page_path, raw)
            record["response"] = fingerprint(page_path)
            if not isinstance(payload, dict) or not isinstance(payload.get("bars"), list):
                raise ReferenceValidationError("Alpaca returned malformed historical bar payload")
            if payload.get("symbol", "SPY") != "SPY":
                raise ReferenceValidationError("Alpaca response symbol is not SPY")
            for bar in payload["bars"]:
                try:
                    rows.append({"timestamp": bar["t"], "spot": bar["c"],
                        "open": bar["o"], "high": bar["h"], "low": bar["l"], "close": bar["c"],
                        "volume": bar["v"], "trade_count": bar.get("n"), "vwap": bar.get("vw"),
                        "source": "Alpaca SIP 1-minute raw bars", "symbol": "SPY"})
                except (KeyError, TypeError):
                    raise ReferenceValidationError("Alpaca bar is missing required OHLCV fields") from None
            token = payload.get("next_page_token")
            record["next_page_token"] = token
            if not token:
                break
            if not isinstance(token, str) or token in tokens:
                raise ReferenceValidationError("Alpaca returned invalid or repeated pagination token")
            tokens.add(token)
            params["page_token"] = token
        else:
            raise ReferenceValidationError("Alpaca pagination exceeded max_pages")
        if not rows:
            raise ReferenceValidationError("Alpaca returned no bars")
        raw_frame = pd.DataFrame(rows)
        # Validate order before the canonical loader sorts it.
        raw_times = pd.to_datetime(raw_frame.timestamp, utc=True)
        if not raw_times.is_monotonic_increasing:
            raise ReferenceValidationError("Alpaca bars are not timestamp sorted")
        out = load_stock_prices(raw_frame, timestamp_convention="minute_bar_start", price_basis="raw")
        qa = validate_stock_bars(out, require_ohlc=True)
        if not qa["passed"]:
            raise ReferenceValidationError(f"Stock acquisition failed structural QA: {qa['failed_checks']}")
        path = destination / "stock_prices.parquet"
        out.to_parquet(path, index=False)
        manifest.update(status="complete", rows=len(out), stock=fingerprint(path), structural_qa=qa)
        _write_json(destination / "manifest.json", manifest)
        return out
    except BaseException as exc:
        # Only controlled exception messages are persisted; arbitrary provider
        # and HTTP-library text could contain request credentials.
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc) if isinstance(exc, ReferenceValidationError) else "Acquisition interrupted or local processing failed"
        _write_json(destination / "manifest.json", manifest)
        raise
    finally:
        headers.clear()


def _regular(frame):
    local = pd.to_datetime(frame.timestamp, utc=True).dt.tz_convert(NY)
    minutes = local.dt.hour * 60 + local.dt.minute
    return frame.loc[local.dt.strftime("%Y-%m-%d").isin(DATES) & minutes.ge(570) & minutes.lt(960)].copy()


def _explicit_timezones(values):
    """Never let ``utc=True`` manufacture a timezone for a naive observation."""
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        return bool(values.notna().all())
    try:
        return all(not pd.isna(value) and pd.Timestamp(value).tzinfo is not None for value in values)
    except (ValueError, TypeError):
        return False


def validate_stock_bars(frame, *, require_ohlc=True):
    """Check all raw rows and exact 390 start-labelled bars per pilot session."""
    required = {"timestamp", "spot", "volume", "symbol", "price_basis", "timestamp_convention", "price_time", "available_at"}
    if require_ohlc:
        required |= {"open", "high", "low", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        return {"passed": False, "failed_checks": ["missing_columns"], "missing_columns": missing}
    checks = {}
    try:
        checks["explicit_timezones"] = all(_explicit_timezones(frame[column])
            for column in ("timestamp", "price_time", "available_at"))
        times = pd.to_datetime(frame.timestamp, utc=True)
        local = times.dt.tz_convert(NY)
        price_time = pd.to_datetime(frame.price_time, utc=True)
        available = pd.to_datetime(frame.available_at, utc=True)
        checks["unique"] = not times.duplicated().any()
        checks["sorted"] = times.is_monotonic_increasing
        checks["all_times_valid"] = bool(times.notna().all())
        checks["pilot_dates_only"] = local.dt.strftime("%Y-%m-%d").isin(DATES).all()
        checks["minute_start_clock"] = (times.eq(times.dt.floor("min")).all() and
            frame.timestamp_convention.eq("minute_bar_start").all())
        checks["bar_end_availability"] = price_time.eq(times + pd.Timedelta("1min")).all() and available.eq(price_time).all()
        checks["raw_spy"] = frame.symbol.eq("SPY").all() and frame.price_basis.eq("raw").all()
        values = frame[["spot", "volume"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        checks["valid_prices_volume"] = (np.isfinite(values).all() and (values[:, 0] > 0).all()
            and (values[:, 1] >= 0).all() and np.equal(values[:, 1], np.floor(values[:, 1])).all())
        if require_ohlc:
            ohlc = frame[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            checks["finite_positive_ohlc"] = np.isfinite(ohlc).all() and (ohlc > 0).all()
            checks["ohlc_bounds"] = ((ohlc[:, 2] <= np.minimum(ohlc[:, 0], ohlc[:, 3])).all()
                and (ohlc[:, 1] >= np.maximum(ohlc[:, 0], ohlc[:, 3])).all()
                and (ohlc[:, 2] <= ohlc[:, 1]).all())
            checks["spot_is_close"] = np.array_equal(frame.spot.to_numpy(float), ohlc[:, 3])
        regular = _regular(frame)
        counts = {}
        missing_minutes = {}
        for day in DATES:
            expected = pd.date_range(f"{day} 09:30", periods=390, freq="min", tz=NY).tz_convert("UTC")
            actual = pd.DatetimeIndex(pd.to_datetime(regular.timestamp, utc=True))
            actual = actual[actual.tz_convert(NY).strftime("%Y-%m-%d") == day]
            counts[day] = len(actual)
            missing_minutes[day] = expected.difference(actual).astype(str).tolist()
            checks[f"complete_{day}"] = len(actual) == 390 and set(actual) == set(expected)
    except (ValueError, TypeError, AttributeError):
        return {"passed": False, "failed_checks": ["malformed_columns"]}
    checks = {key: bool(value) for key, value in checks.items()}
    return {"passed": all(checks.values()), "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "rows": len(frame), "regular_rows": len(regular), "regular_counts": counts,
        "missing_regular_minutes": missing_minutes}


def compare_repeat_stock(original, repeated):
    """Exact regular-session timestamp/close/volume comparison, without repair."""
    first, second = _regular(original), _regular(repeated)
    def selected(frame, label):
        return pd.DataFrame({"timestamp": pd.to_datetime(frame.timestamp, utc=True),
            f"close_{label}": frame.spot.to_numpy(), f"volume_{label}": frame.volume.to_numpy()})
    compared = selected(first, "original").merge(selected(second, "repeat"), on="timestamp", how="outer", indicator=True)
    compared["exact_match"] = (compared["_merge"].eq("both") & compared.close_original.eq(compared.close_repeat)
        & compared.volume_original.eq(compared.volume_repeat))
    compared = compared.sort_values("timestamp", kind="stable").reset_index(drop=True)
    passed = len(compared) == 1170 and bool(compared.exact_match.all())
    return {"passed": passed, "compared_rows": len(compared),
        "mismatch_rows": int((~compared.exact_match).sum()), "comparison": compared}


def stock_crosscheck_targets(original, *, source_status="unavailable_second_source"):
    """Three calibration closes and three largest within-session moves per day.

    Return calculations restart at each regular-session opening, so overnight
    moves cannot enter the selection. Equal moves are ordered by timestamp.
    All values remain the original independent stock observations; the status
    explicitly distinguishes a target list from completed source agreement.
    """
    regular = _regular(original).sort_values("timestamp", kind="stable").copy()
    regular["session_date"] = pd.to_datetime(regular.timestamp, utc=True).dt.tz_convert(NY).dt.strftime("%Y-%m-%d")
    grouped = regular.groupby("session_date", sort=False)
    regular["previous_close"] = grouped.spot.shift()
    regular["previous_timestamp"] = grouped.timestamp.shift()
    regular["simple_close_return"] = regular.spot / regular.previous_close - 1
    regular["absolute_simple_close_return"] = regular.simple_close_return.abs()
    selected = []
    for day in DATES:
        part = regular.loc[regular.session_date.eq(day)]
        observation_time = pd.Timestamp(f"{day} 15:45", tz=NY).tz_convert("UTC")
        calibration = part.loc[pd.to_datetime(part.price_time, utc=True).eq(observation_time)].copy()
        if len(calibration) != 1:
            raise ReferenceValidationError(f"{day}: missing unique calibration cross-check stock observation")
        calibration["selection_criterion"] = "calibration_1545_bar_close"
        selected.append(calibration)
        largest = part.loc[part.simple_close_return.notna()].sort_values(
            ["absolute_simple_close_return", "timestamp"], ascending=[False, True], kind="stable").head(3).copy()
        largest["selection_criterion"] = [f"largest_absolute_intraday_simple_return_rank_{rank}" for rank in range(1, len(largest) + 1)]
        selected.append(largest)
    result = pd.concat(selected, ignore_index=True).rename(columns={"spot": "original_close"})
    result["independent_source_status"] = source_status
    result["independent_intraday_prices_verified"] = False
    return result[["session_date", "selection_criterion", "timestamp", "price_time", "available_at",
        "original_close", "volume", "previous_timestamp", "previous_close", "simple_close_return",
        "absolute_simple_close_return", "source", "independent_source_status", "independent_intraday_prices_verified"]]


def _verify_inventory(folder):
    """Existing source evidence inventories are integrity receipts, not trust roots."""
    inventory = json.loads((folder / "evidence_inventory.json").read_text())
    records = inventory.get("files", inventory.get("inventory", inventory)) if isinstance(inventory, dict) else inventory
    if isinstance(records, dict):
        records = [{"path": key, **value} for key, value in records.items()]
    verified = 0
    for record in records:
        filename = record.get("path", record.get("filename", record.get("file")))
        if filename and record.get("sha256"):
            candidate = Path(filename)
            path = candidate if candidate.is_absolute() else folder / candidate
            if not path.is_file():
                # Inventories may record project-relative paths.
                path = ROOT / candidate
            if not path.is_file() or fingerprint(path)["sha256"] != record["sha256"]:
                raise ReferenceValidationError(f"Historical evidence hash mismatch: {filename}")
            verified += 1
    if not verified:
        raise ReferenceValidationError(f"No verifiable evidence inventory entries: {folder}")
    return verified


def verify_pilot_options(root):
    """Bind current whole-day Parquet bytes to the prior independent DBN audit.

    The old audit scanned each full day and compared the selected interval
    fields independently with the original vendor archive. Here we verify
    that its exact audited files and row counts are still the files in use.
    """
    import pyarrow.parquet as pq

    root = Path(root)
    directory = root / EVIDENCE / "option_evidence"
    verified = _verify_inventory(directory)
    receipt_path = directory / "option_pilot_audit.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "pass":
        raise ReferenceValidationError("Independent option audit did not pass")
    days = receipt.get("days", [])
    if len(days) != len(DATES) or {row.get("session_date") for row in days} != set(DATES):
        raise ReferenceValidationError("Independent option audit does not cover the three pilot sessions")
    rows = []
    for day in DATES:
        record = next(row for row in days if row["session_date"] == day)
        path = root / "data/processed/spy/options" / f"session_date={day}" / "quotes.parquet"
        manifest_path = path.with_name("manifest.json")
        current = fingerprint(path)
        expected = record["parquet"]
        if current["sha256"] != expected["sha256"] or current["bytes"] != expected["bytes"]:
            raise ReferenceValidationError(f"{day}: current option Parquet differs from independently audited bytes")
        if fingerprint(manifest_path)["sha256"] != record["manifest_sha256"]:
            raise ReferenceValidationError(f"{day}: current option manifest differs from independent audit")
        count = pq.ParquetFile(path).metadata.num_rows
        manifest = json.loads(manifest_path.read_text())
        if (count != expected["footer_rows"] or count != record["counts"]["raw_rows"]
                or count != manifest["rows_written"]):
            raise ReferenceValidationError(f"{day}: current option row count differs from independent audit")
        if record.get("status") != "pass" or not record.get("checks") or not all(record["checks"].values()):
            raise ReferenceValidationError(f"{day}: independent option audit contains failed checks")
        rows.append({"session_date": day, **current, "rows": count,
            "manifest": fingerprint(manifest_path)})
    return {"audit_receipt": fingerprint(receipt_path),
        "audit_inventory": fingerprint(directory / "evidence_inventory.json"),
        "verified_inventory_files": verified, "current_option_files": rows,
        "source_to_normalized_verification": "Exact whole-day bytes match the Oct23 independent DBN/Parquet audit; no new source scan claimed."}


def _verify_repeat_receipt(folder):
    """Bind decoded bar values to the preserved raw HTTP response chain."""
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ReferenceValidationError("Repeat acquisition is incomplete")
    expected = {"timeframe": "1Min", "start": "2025-08-07T04:00:00+00:00",
        "end": "2025-08-12T04:00:00+00:00", "feed": "sip", "adjustment": "raw", "limit": 10000, "sort": "asc"}
    if manifest.get("endpoint") != ENDPOINT or manifest.get("base_request_params") != expected:
        raise ReferenceValidationError("Repeat acquisition request is not the declared raw SIP pilot")
    pages = manifest.get("pages", [])
    if not 1 <= len(pages) <= 100:
        raise ReferenceValidationError("Repeat acquisition has invalid page count")
    token, all_rows = None, []
    for number, page in enumerate(pages):
        request = dict(expected)
        if token:
            request["page_token"] = token
        if page.get("number") != number or page.get("request_params") != request or page.get("status_code") != 200:
            raise ReferenceValidationError("Repeat acquisition pagination request chain mismatch")
        path = folder / f"page_{number:03d}.json"
        if fingerprint(path)["sha256"] != page.get("response", {}).get("sha256"):
            raise ReferenceValidationError("Repeat raw HTTP page fingerprint differs from manifest")
        payload = json.loads(path.read_bytes())
        if payload.get("symbol", "SPY") != "SPY" or not isinstance(payload.get("bars"), list):
            raise ReferenceValidationError("Repeat raw HTTP response is malformed")
        all_rows.extend(payload["bars"])
        token = payload.get("next_page_token")
        if token != page.get("next_page_token") or (not token and number != len(pages) - 1):
            raise ReferenceValidationError("Repeat acquisition pagination response chain mismatch")
    if token:
        raise ReferenceValidationError("Repeat acquisition ended before all pages were acquired")
    stock_file = folder / "stock_prices.parquet"
    if fingerprint(stock_file)["sha256"] != manifest.get("stock", {}).get("sha256"):
        raise ReferenceValidationError("Repeat stock fingerprint differs from acquisition manifest")
    frame = pd.read_parquet(stock_file)
    raw = pd.DataFrame(all_rows)
    if len(raw) != len(frame) or manifest.get("rows") != len(frame):
        raise ReferenceValidationError("Repeat raw HTTP and processed bar counts disagree")
    if not np.array_equal(pd.to_datetime(raw.t, utc=True).to_numpy(), pd.to_datetime(frame.timestamp, utc=True).to_numpy()):
        raise ReferenceValidationError("Repeat raw HTTP and processed timestamps disagree")
    for name, alias in (("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"), ("v", "volume")):
        if name not in raw or alias not in frame or not np.array_equal(raw[name].to_numpy(float), frame[alias].to_numpy(float)):
            raise ReferenceValidationError(f"Repeat raw HTTP and processed {alias} disagree")
    return frame


def pilot_rate_rows(rates, day, as_of):
    """Use exactly the predeclared prior ALFRED vintage for one pilot date."""
    expected = {"2025-08-07": "2025-08-06", "2025-08-08": "2025-08-07", "2025-08-11": "2025-08-08"}
    if day not in expected:
        raise ReferenceValidationError("No approved pilot vintage exists for this date")
    rows = rates.loc[pd.to_datetime(rates.pilot_as_of, utc=True).dt.tz_convert(NY).dt.strftime("%Y-%m-%d").eq(day)].copy()
    if (len(rows) != 4 or not rows.vintage.eq(expected[day]).all()
        or set(rows.series) != {"DGS1MO", "DGS3MO", "DGS6MO", "SOFR"}):
        raise ReferenceValidationError(f"{day}: incorrect historical rate vintage selection")
    if (pd.to_datetime(rows.available_at, utc=True) > as_of).any():
        raise ReferenceValidationError(f"{day}: future rate vintage cannot enter snapshot")
    return rows


def select_call_contracts(quotes, spot):
    """Six distinct calls per expiry; ±2% held out; ties by strike/identity."""
    quotes = quotes.loc[quotes.is_call.eq(True)].copy()
    quotes["expiry_date"] = pd.to_datetime(quotes.expiry_at, utc=True).dt.tz_convert(NY).dt.strftime("%Y-%m-%d")
    selected = []
    for expiry in EXPIRIES:
        group = quotes.loc[quotes.expiry_date.eq(expiry)]
        used = set()
        for target in (-.03, -.02, -.01, .01, .02, .03):
            candidates = group.loc[~group.contract_id.isin(used)].copy()
            candidates["distance"] = abs(np.log(candidates.strike.astype(float) / spot) - target)
            candidates = candidates.sort_values(["distance", "strike", "contract_id"], kind="stable")
            if candidates.empty:
                raise ReferenceValidationError(f"{expiry}: insufficient distinct calls at target {target:+.2f}")
            row = candidates.iloc[[0]].drop(columns="distance").copy()
            row["target_log_moneyness"] = target
            row["split"] = "heldout" if abs(target) == .02 else "train"
            row["option_type"] = "call"
            row["exercise_style"] = "american"
            selected.append(row)
            used.add(row.contract_id.iloc[0])
    return pd.concat(selected, ignore_index=True)


def load_spy_call_pilot(campaign_root, *, root=ROOT, repeat_root=None, allow_pending=True):
    """Materialize isolated snapshots and return panels/markets with a hard fit gate.

    Tentative panels may be audited before the repeat file exists; ``ready`` is
    false and callers must not calibrate them. Files below ``inputs`` are
    immutable by content. A resumed campaign may add newly available repeat QA.
    """
    from .pipeline import read_option_window
    from .snapshots import build_calibration_snapshot

    root, output = Path(root), Path(campaign_root) / "inputs"
    output.mkdir(parents=True, exist_ok=True)
    option_integrity = verify_pilot_options(root)
    original_path = root / ORIGINAL
    original = pd.read_parquet(original_path)
    original_qa = validate_stock_bars(original, require_ohlc=False)
    if not original_qa["passed"]:
        raise ReferenceValidationError(f"Original stock failed QA: {original_qa['failed_checks']}")
    repeat_path = Path(repeat_root) if repeat_root is not None else root / REPEAT
    stock, repeat_qa, repeat_compare = original, None, None
    repeat_ready = False
    if (repeat_path / "manifest.json").is_file():
        manifest = json.loads((repeat_path / "manifest.json").read_text())
        if manifest.get("status") == "complete":
            stock = _verify_repeat_receipt(repeat_path)
            repeat_qa = validate_stock_bars(stock, require_ohlc=True)
            repeat_compare = compare_repeat_stock(original, stock)
            repeat_ready = repeat_qa["passed"] and repeat_compare["passed"]
    # Pending evidence is separate so that later accepted snapshots can bind
    # the richer repeated OHLCV row without rewriting a previous audit receipt.
    if not repeat_ready:
        output = output / "pending"
    if repeat_compare is not None:
        _write_once(output / "stock_repeat_comparison.csv", repeat_compare["comparison"].to_csv(index=False).encode())
        _write_json(output / "stock_repeat_qa.json", repeat_qa)
    if not repeat_ready and not allow_pending:
        raise ReferenceValidationError("SPY pilot blocked: separate raw OHLCV repeat acquisition and exact regular-session agreement required")
    _write_json(output / "original_stock_qa.json", original_qa)
    _write_json(output / "option_integrity.json", option_integrity)
    independent_path = root / "data/raw/spy_reference/spy_rough_calls_independent_v1/attempt_manifest.json"
    second_source_status = "unavailable_second_source_not_attempted_in_this_bundle"
    if independent_path.is_file():
        attempt = json.loads(independent_path.read_text())
        statuses = sorted({str(row.get("status")) for row in attempt.get("attempts", [])})
        second_source_status = "unavailable_second_source_HTTP_" + "_".join(statuses)
    crosscheck_targets = stock_crosscheck_targets(original, source_status=second_source_status)
    _write_once(output / "stock_crosscheck_targets.csv", crosscheck_targets.to_csv(index=False).encode())

    evidence = root / EVIDENCE
    dividend_dir = evidence / "dividend_evidence"
    rate_dir = evidence / "rate_evidence"
    _verify_inventory(dividend_dir)
    _verify_inventory(rate_dir)
    review = json.loads((dividend_dir / "evidence_review.json").read_text())
    historical = review["historical_calendar"]
    calendar_path = dividend_dir / historical["file"]
    if fingerprint(calendar_path)["sha256"] != historical["sha256"]:
        raise ReferenceValidationError("Issuer calendar fingerprint mismatch")
    if historical["capture_timestamp_utc"] != "2025-07-31T05:14:12Z" or historical["expected_events_inside_pilot_horizon"]:
        raise ReferenceValidationError("Pilot calendar evidence does not support expected empty schedule")
    assumption = {"schema": "spy_conditional_empty_expected_dividends_v1", "symbol": "SPY",
        "events": [], "valuation_dates": DATES, "coverage_through": "2025-09-05",
        "known_by": historical["capture_timestamp_utc"], "calendar": fingerprint(calendar_path),
        "calendar_manifest": fingerprint(dividend_dir / "historical_calendar_manifest.json"),
        "review": fingerprint(dividend_dir / "evidence_review.json"),
        "assumption_accepted": True,
        "assumption": "Archived issuer calendar applies without subsequent special-distribution or schedule-change notices.",
        "complete_notice_feed_verified": False, "canonical_readiness_changed": False,
        "hedging_ready": False, "realized_future_dividends_used": False}
    _write_json(output / "dividend_assumption.json", assumption)
    dividends = pd.DataFrame({"ex_date": pd.Series([], dtype=object), "amount": pd.Series([], dtype=float),
                              "available_at": pd.Series([], dtype="datetime64[ns, UTC]")})
    rates_path = rate_dir / "pilot_prior_vintage_rates_candidate.csv"
    rates = pd.read_csv(rates_path)
    panels, markets, previous_panels, previous_markets, full_snapshots = {}, {}, {}, {}, {}
    audit, exclusions, references, snapshots = [], [], [], []
    for day in DATES:
        for clock in ("15:45", "15:44"):
            as_of = pd.Timestamp(f"{day} {clock}", tz=NY).tz_convert("UTC")
            day_rates = pilot_rate_rows(rates, day, as_of)
            options_path = root / "data/processed/spy/options" / f"session_date={day}" / "quotes.parquet"
            options = read_option_window(options_path, as_of)
            expiry_dates = pd.to_datetime(options.expiry_at, utc=True).dt.tz_convert(NY).dt.strftime("%Y-%m-%d")
            outside = options.loc[~expiry_dates.isin(EXPIRIES)].copy()
            outside["snapshot_exclusion_reason"] = "outside_declared_pilot_expiries"
            options = options.loc[expiry_dates.isin(EXPIRIES)].copy()
            snapshot = build_calibration_snapshot(options, as_of, stock=stock, rates=day_rates,
                dividends=dividends, min_dte=0, max_dte=365, max_relative_spread=.25,
                max_quote_age="1min", min_quote_size=1, dividend_coverage_through="2025-09-05")
            snapshot.require_ready()
            observed = snapshot.quotes.copy()
            observed["snapshot_date"], observed["snapshot_clock"] = day, clock
            excluded = pd.concat([snapshot.exclusions, outside], ignore_index=True)
            excluded["snapshot_date"], excluded["snapshot_clock"] = day, clock
            puts = observed.loc[~observed.is_call].copy()
            puts["snapshot_exclusion_reason"] = "put_outside_american_call_reduction"
            excluded = pd.concat([excluded, puts], ignore_index=True)
            exclusions.append(excluded)
            if clock == "15:45":
                panels[day] = select_call_contracts(snapshot.quotes, snapshot.market.spot)
                markets[day] = snapshot.market
                full_snapshots[day] = snapshot
            else:
                # Timing sensitivity uses the exact baseline contract set.
                required_ids = panels[day].contract_id.tolist()
                available = snapshot.quotes.set_index("contract_id", drop=False)
                missing_ids = sorted(set(required_ids) - set(available.index))
                if missing_ids:
                    raise ReferenceValidationError(f"{day} 15:44: baseline contracts unavailable: {missing_ids}")
                previous = available.loc[required_ids].reset_index(drop=True)
                for column in ("target_log_moneyness", "split", "option_type"):
                    previous[column] = panels[day][column].to_numpy()
                previous_panels[day], previous_markets[day] = previous, snapshot.market
            unselected = observed.loc[observed.is_call & ~observed.contract_id.isin(panels[day].contract_id)].copy()
            unselected["snapshot_exclusion_reason"] = "eligible_call_outside_fixed_demonstration_selection"
            unselected["exclusion_kind"] = "sampling"
            exclusions.append(unselected)
            snapshots.append(observed)
            references.append({"date": day, "clock": clock, "market": {"spot": snapshot.market.spot,
                "discount_times": snapshot.market.discount_curve.times,
                "discounts": snapshot.market.discount_curve.discounts,
                "dividend_events": [], "dividend_coverage": snapshot.market.dividends.coverage},
                "snapshot_provenance": dict(snapshot.provenance),
                "option_manifest": fingerprint(options_path.with_name("manifest.json")),
                "observable_option_window_sha256": hashlib.sha256(pd.util.hash_pandas_object(options, index=False).values.tobytes()).hexdigest()})
            audit.append({"date": day, "clock": clock, "spot": snapshot.market.spot,
                "eligible_all_options": len(snapshot.quotes), "eligible_calls": int(snapshot.quotes.is_call.sum()),
                "sampling_excluded_eligible_calls": len(unselected),
                "conditional_input_ready": True, "repeat_stock_gate_passed": repeat_ready,
                "calibration_allowed": repeat_ready, "hedging_ready": False,
                "snapshot_convention": "reconstructed_interval_end; publication_latency_not_measured"})
    all_exclusions = pd.concat(exclusions, ignore_index=True)
    audit_frame = pd.DataFrame(audit)
    for day in DATES:
        _write_once(output / f"{day}_selected_1545.csv", panels[day].to_csv(index=False).encode())
        _write_once(output / f"{day}_selected_1544.csv", previous_panels[day].to_csv(index=False).encode())
    _write_once(output / "observed_snapshots.csv", pd.concat(snapshots, ignore_index=True).to_csv(index=False).encode())
    _write_once(output / "exclusions.csv", all_exclusions.to_csv(index=False).encode())
    _write_json(output / "market_inputs.json", references)
    # Gate receipt names distinguish a pending observation from accepted repeat QA.
    gate_name = "input_gate_accepted.json" if repeat_ready else "input_gate_pending.json"
    diagnostics = {"schema": "spy_call_pilot_inputs_v1", "ready": repeat_ready,
        "status": "ready_conditional_dividend_assumption" if repeat_ready else "blocked_repeat_stock_acquisition",
        "original_stock": fingerprint(original_path), "rates": fingerprint(rates_path),
        "option_integrity": option_integrity,
        "repeat_stock": fingerprint(repeat_path / "stock_prices.parquet") if repeat_ready else None,
        "repeat_manifest": fingerprint(repeat_path / "manifest.json") if (repeat_path / "manifest.json").is_file() else None,
        "independent_price_source_attempt": fingerprint(independent_path) if independent_path.is_file() else None,
        "independent_intraday_prices_verified": False,
        "stock_crosscheck_targets": {key: value for key, value in
            fingerprint(output / "stock_crosscheck_targets.csv").items() if key != "path"},
        "original_structural_qa": original_qa, "repeat_structural_qa": repeat_qa,
        "repeat_comparison": {key: value for key, value in (repeat_compare or {}).items() if key != "comparison"},
        "assumption": assumption, "snapshot_audit": audit,
        "limitations": ["Stock is an interval close, not an executable quote or measured point-in-time latency.",
            "The dividend assumption is accepted only for this isolated research pilot.",
            "Historical rough factor states are not estimated; surfaces are independent date fits.",
            "Treasury yields are term discount proxies, not a bootstrapped OIS curve.",
            "Independent intraminute stock-source agreement is not established by a repeated Alpaca acquisition."],
        "canonical_readiness_changed": False, "hedging_ready": False}
    _write_json(output / gate_name, diagnostics)
    return {"status": diagnostics["status"], "ready": repeat_ready, "panels": panels, "markets": markets,
        "previous_panels": previous_panels, "previous_markets": previous_markets,
        "full_snapshots": full_snapshots, "audit": audit_frame, "exclusions": all_exclusions,
        "diagnostics": diagnostics}


build_pilot_bundle = load_spy_call_pilot

