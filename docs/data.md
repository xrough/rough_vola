# Market data and reproducibility

The public source and synthetic notebooks require no market-data account.
Market studies require separately acquired data and provider entitlements.
Never treat synthetic demonstrations or option-implied inputs as independently
observed stock prices.

The historical SPY pilot used:

- Vendor option observations with timestamps, American exercise labels, bid/ask
  prices and positive quoted sizes.
- Independently acquired, unadjusted SPY minute OHLCV observations. Minute labels
  and reconstructed interval-end stock/option alignment remain explicit.
- Dated ALFRED-vintage Treasury yields as discount proxies, with conservative
  prior-day availability rules. These are not an OIS curve.
- A hash-verified July 31, 2025 issuer dividend-calendar archive and an explicitly
  scoped no-later-change assumption through September 5.

The three observation dates are August 7, 8 and 11, 2025. The selected expiries
are August 22, August 29 and September 5. The pilot uses a one-minute option
window, positive valid quotes/sizes and a 25% maximum relative spread. Fixed
contract selection and input sensitivities are part of the original campaign.

No raw or processed quote tables, credentials, historical payloads or full
acquisition responses are included. The archived campaign runners require those
separately preserved inputs; copying the runner alone does not reproduce their
results. Published aggregate findings are historical exports, not live refits.

For new acquisitions, use the market-data CLI's explicit settings and inspect
its row-level rejection/readiness reports:

```bash
python -m pip install -e '.[market-data]'
python -m rvlab.market_data --help
python scripts/acquire_spy_call_pilot.py --help
```

The acquisition script prompts for Alpaca credentials locally. Keep credentials
out of notebooks, shell history, committed configuration and reports. Credential
files and local data directories are excluded by the root `.gitignore`.

For calibration, American-labelled market contracts route through American
pricing or the explicitly guarded dividend-free call equivalence. The European
calibrator rejects American-labelled input. An empty dividend schedule declares
complete absence of expected cash events over its coverage; missing dividend
information is distinct. A trailing dividend yield is not a dated cash schedule.

Calibration inputs and causal historical hedging inputs have different readiness
requirements. The scoped pilot does not promote full-archive or hedge readiness.
