# Testing the public source edition

The default suite checks numerical pricing, calibration, finite differences,
rough simulation, stopping estimators, data validation and synthetic research
workflows. It does not require the private SPY archive or a sibling checkout.

```bash
python -m pip install -e '.[dev,notebooks,market-data,research]'
python -m pytest -q
python scripts/run_notebooks.py --all --output-dir /tmp/rough-vola-notebooks
```

The lighter numerical job needs only `.[dev]`; its exact files are listed in
`.github/workflows/tests.yml`. This tests that the core has no plotting, market
provider or machine-learning dependency.

Tests marked `archive` inspect earlier report editions that are intentionally
not distributed with this source repository. They remain visible as explicit
skips, and can be selected after separately restoring those historical inputs:

```bash
python -m pytest -q -m archive --run-archive-tests
```

This is a data-availability distinction, not a numerical acceptance waiver.
The numerical engines and synthetic failure/accuracy tests remain enabled.
The old simulator's byte-identical source is retained as a small test fixture
for the fixed-seed numerical compatibility regression; that test runs normally.
Original archived-notebook structure checks belong to the old report edition,
while the public notebooks have their own validation and fresh-kernel smoke run.

For local validation with an existing interpreter, use `PYTHONPATH=$PWD/src` and
check `rvlab.__file__`/`roughvol.__file__`. This prevents an older editable install
from accidentally supplying the code under test.

The compatibility fixture is
`src/rvlab/tests/fixtures/historical_rough_simulation_2026_10_18.py`, copied
unchanged from the project's archived
`rough_simulation_precision_2026-10-18/source/src/rvlab/pricing/rough_simulation.py`.
Its SHA256 is
`09b71db02df881ddbcca57cfad135d9249f0b49ffc787a3670637e13a28763fb`,
and the regression verifies this identity before comparing fixed-seed paths.

Two small, byte-identical historical protocol JSONs are also retained under
`src/rvlab/tests/fixtures/`: `bellman_control_protocol_2026_10_19.json` and
`price_budget_protocol_2026_10_19.json`. These exercise the original design,
tolerance and seed-firewall guards without loading the omitted Monte Carlo
samples. They are protocol fixtures, not a claim that the complete historical
campaign has been restored or rerun. Synthetic inventory fixtures independently
test source/input tampering and preservation checks.
