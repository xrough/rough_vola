# rough_vola

Python research on stochastic and rough volatility: European and American
option pricing, numerical validation, surface calibration and the foundations
for comparing hedge policies on identical option positions.

This repository consolidates the Python research developed in `effective_engine`
and selected simulation and empirical-analysis components from `roughVol`.
The Python imports remain `rvlab` and `roughvol`. One installation supplies both;
no sibling checkout or C++ build is required.

## Start with three notebooks

1. [Pricing and exercise](notebooks/01_pricing_and_exercise.ipynb):
   European prices, American exercise, stock deltas and numerical accuracy.
2. [Synthetic calibration](notebooks/02_synthetic_calibration.ipynb):
   recovered prices, parameter identification, optimizer diagnostics and refinement.
3. [SPY pilot](notebooks/03_spy_pilot.ipynb):
   American-call assumptions, aggregate findings and the inputs required to
   reproduce real-data work. Missing market data remain an explicit blocker.

The public notebooks run without the large historical experiment archive.
Historical findings are identified separately from newly executed demonstrations.
See the [notebook guide](notebooks/README.md) and [research status](docs/research-status.md).

## Installation and a first experiment

Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[notebooks]'
python examples/pricing_calibration.py --model heston
python scripts/run_notebooks.py --all
```

The core numerical dependencies are NumPy, SciPy and pandas. Plotting,
notebooks, market acquisition and broader statistical tools use optional extras.
To run the full public test suite:

```bash
python -m pip install -e '.[dev,notebooks,market-data,research]'
python -m pytest
```

Private historical-archive checks have explicit skip reasons; they are distinct
from the numerical source tests. See [testing and publication](docs/publication.md).

## Numerical scope

| Component | Implementation and scope |
| --- | --- |
| European Black–Scholes | Analytic forward/discount prices |
| European Heston and rough Heston | Shared Fourier quadrature, Black–Scholes control variate, fractional Riccati solver |
| American Black–Scholes and Heston | Finite differences, cash-dividend jumps, exercise constraints, stock deltas and independent refinement diagnostics |
| Maturity-dependent variance specifications | Heston and rough-Heston parameter curves with explicit calendar-time conventions |
| General American rough Heston | Experimental finite-factor simulation, independent policy training/valuation, controls and dual stopping diagnostics |
| Dividend-free American calls | Guarded exact no-early-exercise reduction for BS, Heston and rough Heston |
| Rough Bergomi | Selected Gaussian/Volterra simulation methods with numerical tests and explicit limitations |
| Empirical roughness and synthetic hedge accounting | Session-aware analysis and common-path European accounting foundations |

The [pricing guide](src/rvlab/pricing/README.md) specifies conventions and
supported ranges. Prices are per underlying unit. Contract multipliers belong
in portfolio accounting. Numerical convergence, optimizer termination, market
fit and parameter identification are reported separately.

## What the first research stage establishes

European pricing and American BS/Heston have numerical and limit tests.
General American rough-Heston experiments retain unresolved stopping and
refinement errors; a focused successful estimate does not establish accuracy
across the general engine's parameter and contract range.

The three-date SPY pilot uses independently observed stock and directly observed
American call quotes. Its deterministic rough-Heston route requires complete
empty dividend coverage, nonnegative forward rates and a true-martingale model.
The archived dividend-calendar assumption and Treasury discount proxies remain
explicit. The initial Fourier cutoff was insufficient for the stochastic fits;
later fixed-parameter refinement checks do not replace a newly calibrated,
numerically accepted fit. See [SPY findings](reports/spy_pilot.md).

Historical rough-state estimation, general American rough-Heston calibration
and portfolio hedge replay remain research milestones.

## Data, results and provenance

Raw market data, credentials, processed quote tables and large simulation
payloads are not distributed. The code license applies to this software;
market-data access and redistribution follow their providers' terms.
The public report contains aggregate findings and no contract-level SPY quotes.
See [data requirements](docs/data.md) and [consolidation provenance](docs/provenance.md).

The project is published with a clean research history. Earlier working trees,
Git histories and sealed numerical campaigns are preserved locally. The former
`effective_engine` GitHub address redirects here; the separate `roughVol`
repository is private.
