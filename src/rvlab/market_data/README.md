# SPY market inputs

This package normalizes the local Databento SPY option archive, validates
independent reference inputs, and builds auditable daily calibration candidates.
The current source is `demo/data/OPRA-20260208-3T68RYYKF9.zip`, covering
2025-08-07 through 2026-02-06. See
[`lab/reports/spy-data-readiness.md`](../../../lab/reports/spy-data-readiness.md)
and its adjacent JSON for the completed audit, exact coverage, and remaining gaps.

The canonical full-archive stock reference remains unavailable. A separate
three-session raw SIP stock pilot was acquired on October 26, 2026; its local
validation covers all 1,170 regular-session minutes on August 7, 8 and 11, 2025.
It has not been promoted to a full-period or executable bid/ask reference.
Historical cash distributions and rate proxies alone cannot supply missing
stock observations or executable quotes. The pipeline preserves candidates
and reports blockers instead of inventing inputs.

## Run

From the `MVP` directory, install the optional data dependencies:

```bash
python -m pip install -e '.[market-data,dev]'
python -m rvlab.market_data prepare --workers 2
```

`prepare` decodes all daily CBBO files and then audits references and snapshots.
`prepare-options` runs only normalization; `audit` reuses normalized options and
refreshes supporting inputs without decoding DBN again. The thin source-tree
entry point also works without an editable package install:

```bash
.venv/bin/python scripts/prepare_spy_dataset.py audit
```

Useful explicit commands:

```bash
# Download public rates and realised distributions, including trailing history.
python -m rvlab.market_data fetch-public --start 2024-01-01 --end 2026-04-30

# Reaudit after a raw stock file becomes available.
python -m rvlab.market_data audit --stock /absolute/path/spy_minutes.csv \
  --stock-timestamp-convention minute_bar_start \
  --stock-price-basis raw --stock-source 'Provider/export identifier'

# Assert coverage of a supplied historically dated cash-dividend forecast.
python -m rvlab.market_data audit --dividends /absolute/path/spy_forecasts.parquet \
  --dividend-coverage-through 2026-04-07

# Exit with status 2 when the full dataset fails calibration readiness.
python -m rvlab.market_data audit --require-ready
```

An ordinary audit exits successfully when it writes its report, even if inputs
are incomplete. `--require-ready` checks calibration readiness; consult
`hedging_ready` separately. `--max-days` creates a pilot, which cannot establish
full-archive readiness. Workers are bounded to 1–4; the default is 2.

## Preserved data and clocks

Outputs live under `data/processed/spy/` and are excluded from Git:

| Output | Purpose |
|---|---|
| `options/session_date=YYYY-MM-DD/quotes.parquet` | Full normalized daily option history, including rejected and extended-hours rows |
| `options/session_date=YYYY-MM-DD/manifest.json` | Source fingerprint, row counts, exclusions, and minute coverage |
| `manifest.json` | Archive inventory and completed normalization summary |
| `references/*.parquet` | Validated stock, rate, dividend, and exchange-session tables |
| `stock_coverage.parquet` | Required, observed, and missing stock minutes per session |
| `snapshots/session_date=YYYY-MM-DD/` | Candidate quotes, exclusions, and snapshot provenance |
| `readiness.json`, `readiness.md` | Readiness flags and missing-input explanations |

Daily writes are atomic. Resume checks source/configuration fingerprints,
Parquet schema, file size, and row counts; it does not silently accept partial
outputs. Duplicate contract/timestamp keys and missing exchange minutes are
reported. Normalization retains all strikes, expiries, and source observations;
calibration eligibility is a separate screen.

CBBO-1m `ts_recv` labels the **end** of an interval. Normalized `timestamp` and
`available_at` use that instant in UTC. The source `ts_event` is a last-trade time,
stored as `last_trade_ts`; it cannot establish quote-change freshness. Minute
coverage uses the actual XNYS session, including daylight saving and early
closes. Pricing maturity uses ACT/365 from the observation to the exchange close
on expiry; this is a valuation convention, not an exercise deadline.

## Independent stock acquisition and import

If an Alpaca account becomes available, configure `APCA_API_KEY_ID` and
`APCA_API_SECRET_KEY` in the local process environment through your usual secret
manager. Do not put keys in notebooks, source files, command-line arguments, or
chat. The helper requests historical **SIP, unadjusted one-minute bars** and
handles pagination:

```bash
python -m rvlab.market_data fetch-alpaca --start 2025-08-07 --end 2026-02-06
python -m rvlab.market_data audit
```

Date-only acquisition endpoints are inclusive. Provider access errors remain
explicit failures. Bar closes can support an observed spot proxy; this command
does not acquire executable bid/ask quotes. Stock bid/ask history remains a
separate requirement for the strict hedging flag.

Local CSV and Parquet files are supported. The raw-reference directory defaults
to `data/raw/spy_reference/`; the auditor discovers `stock_prices`, `rates`, and
`dividends` with either extension. Explicit CLI paths override discovery. It can
also parse downloaded `SOFR.csv`, `DGS1MO.csv`, `DGS3MO.csv`, `DGS6MO.csv`, and
`ssga_historical_distributions.xlsx` directly.

| Table | Required columns and units |
|---|---|
| Stock | `timestamp`, positive `spot`, `source`, `price_basis="raw"`; `timestamp_convention` or explicit `available_at` |
| Rates | `observation_date`, timezone-aware `available_at`, positive integer `tenor_days`, annual decimal `rate`, `series`, `source` |
| Dividends | `ex_date`, `pay_date`, nonnegative cash `amount` per share, `source`; `available_at` is required for a usable historical expectation |

Stock conventions are `observation`, `minute_bar_start`, and `minute_bar_end`.
A bar-start close becomes available one minute after its timestamp; `price_time`
records that end, while `available_at` may be later. A canonical stock CSV could
contain:

```csv
timestamp,spot,source,price_basis,timestamp_convention
2025-08-07T13:30:00Z,635.12,example-provider,raw,minute_bar_start
```

The displayed price is illustrative. Optional `bid` and `ask` must both be
positive and uncrossed. Adjusted stock prices and duplicate observations are
rejected. The CLI expects timezone-aware timestamps; API imports may explicitly
pass a timezone for naive source timestamps. Alignment uses only available
observations, within one minute and the same regular session. Daily bars do not
supply intraday coverage.

## Rates, dividends, and American exercise

### Isolated American-call research pilot

`scripts/acquire_spy_call_pilot.py` prompts in the user's local Terminal for an
Alpaca key and secret. It writes a new, non-overwriting repeat-acquisition
folder containing raw successful HTTP pages, full OHLCV, sanitized pagination
and request metadata, and SHA256 fingerprints. Credentials are not persisted.
Structural checks and an exact regular-session timestamp/close/volume
comparison with the original pilot must pass before real calibration.

`load_spy_call_pilot` in `spy_call_pilot.py` builds a separate campaign input
bundle, including 15:45 and synchronized 15:44 snapshots, selection/exclusion
tables and historical rate evidence. Six calls per expiry target log-moneyness
`log(strike / observed_spot)` of ±1%, ±2% and ±3%; ±2% contracts are held out.
Deterministic strike/contract ordering resolves ties. It never rewrites canonical
references, snapshots, readiness flags or notebook outputs.

The user-authorized dividend assumption is explicit: the issuer schedule
captured on July 31, 2025 applies without subsequent special-distribution or
schedule-change notices. It supports an empty expected schedule only for the
three pilot dates through September 5, 2025. Evidence hashes, capture
availability and incomplete notice coverage are retained. A repeated download
does not establish independent vendor agreement, and bar-end availability
does not establish measured publication latency. Strict hedging readiness
remains blocked.

### Full-period reference conventions

Public rates come from [FRED](https://fred.stlouisfed.org/): DGS1MO, DGS3MO, and
DGS6MO become labelled short-end Treasury yield proxies; SOFR supports financing
coverage only. Discounts use interpolation of log discounts from these yields,
not an OIS bootstrap. Rates are converted from percentages to decimals. The
conservative availability proxy is 17:00 New York on the next federal business
day. Downloads contain the latest published values, **not historical vintages**;
the lag does not recover revisions or exact original publication times.

The [State Street historical distribution workbook](https://www.ssga.com/library-content/products/fund-data/etfs/us/spdr-etf-historical-distributions.xlsx)
supplies realised SPY cash flows. Its availability proxy is payment-date 16:00
New York, labelled explicitly rather than presented as a declaration timestamp.
Realised future amounts never enter past forwards. Default strict preparation
requires a historically available expected-dividend schedule with an explicit
coverage horizon; an absent forecast does not mean zero dividends.
`--dividend-coverage-through` is the caller's assertion that the supplied
schedule is complete through that date, including a justified absence of any
events. Every expectation still needs its historical `available_at`; the flag
does not make future realised distributions admissible as past expectations.
The superseded `--allow-dividend-proxy` convention used the last four known paid
distributions divided by current spot. That approximation remains recorded in
historical provenance. It cannot supply the dated cash
events required for American exercise and no longer unlocks readiness. Existing
invocations using that flag produce an explicit missing-input explanation.

SPY options are standard unadjusted, physically settled American contracts with
a 100-share multiplier; prices remain per share. Calibration candidates default
to 7–60 days to expiry, valid recent positive bid/ask quotes, positive quoted
sizes and spread checks. ITM contracts and expiries crossing known cash-dividend
dates remain eligible. European price bounds, CRR exercise-premium rejection,
and hindsight dividend-date exclusions are removed. American BS and Heston
calibrate directly to the observed quotes. American rough-Heston synthetic
stopping prices and stock deltas are available separately; rough calibration
and historical conditional-factor estimation remain unsupported.

Dividend amounts are per underlying share. Only revisions whose `available_at`
is no later than the observation enter the schedule, and the latest available
revision for each ex-date is used once. A dated event jumps at the XNYS opening
on its ex-date unless an explicit timezone-aware `ex_at` is provided. Unavailable
future realized events neither enter valuation nor exclude contracts. A supplied
coverage horizon is the caller's historical completeness assertion; the code
cannot verify that assertion from a retrospective distribution workbook.

## Calibration handoff and readiness

The audit materializes a snapshot 15 minutes before each session close. The full
minute history stays available for later hedge policies. The latest observable
quote is selected before quality filtering, so a bad current quote cannot cause
an older good quote to be silently revived.

For a chosen as-of time, use the API to set a historically justified dividend
coverage horizon and pass only a ready snapshot to calibration:

```python
from rvlab.market_data.pipeline import read_option_window
from rvlab.market_data.references import load_stock_prices, load_rates, load_dividends
from rvlab.market_data.snapshots import build_calibration_snapshot
from rvlab.pricing import HestonParams, calibrate_vanilla_surface

options = read_option_window(option_partition, as_of)
snapshot = build_calibration_snapshot(
    options, as_of,
    stock=load_stock_prices(stock_file),
    rates=load_rates(rates_file),
    dividends=load_dividends(historical_expected_dividends_file),
    dividend_coverage_through=verified_forecast_horizon,
)
fit = calibrate_vanilla_surface(
    initial_params=HestonParams(
        v0=0.04, theta=0.04, kappa=2.0, vol_of_vol=0.4, rho=-0.7
    ), **snapshot.calibration_inputs(),
)
```

The file paths, as-of instant, and forecast horizon above are caller-supplied.
`calibration_inputs()` raises `MarketDataNotReady` when dependencies or screens
fail. Its keyword arguments are `quotes`, an independent-spot `market`, and
`exercise_style="american"`; descriptive forward/discount columns are never
pricing inputs. Audit candidates remain labelled American when blocked.

Snapshot schema **2**, backend `american_vanilla_fd`, supersedes the old European
approximation snapshots. `audit` rebuilds saved snapshots; readers of cached
manifests must call `snapshot_manifest_current()` before trusting readiness.
The full normalized option archive is unchanged.

`options_complete` requires the entire archive and its coverage/quality checks.
`calibration_ready` additionally requires every daily snapshot to pass configured
input preparation. `hedging_ready` also requires regular-session stock prices
and bid/ask coverage, financing observations, verified historical rate vintages,
and historically available cash-dividend expectations. Latest FRED downloads alone cannot satisfy
this historical replay requirement.
These flags describe data preparation. Numerical calibration stability,
transaction-cost modelling, exercise handling, and trading validation remain
separate research tasks.

Run the data tests with:

```bash
python -m pytest src/rvlab/tests/test_market_*.py
```
