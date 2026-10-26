# Archived SPY American-call pilot — aggregate public report

**Evidence role:** historical aggregate derivative; no market fit is recomputed
by the public notebooks. The files contain model/date errors, bounds and
numerical statuses. They omit observed option contracts, strikes, quotes,
stock prices, vendor records and absolute local paths. Original sealed summary,
selection and inventory content hashes are recorded in the
[manifest](aggregate_findings/manifest.json); full provenance stays local.

## Economic and data assumptions

The pilot considered American calls at 15:45 New York time on August 7, 8 and
11, 2025, across three short expiries. The call exercise label was retained.
The exact European-value reduction required complete explicit zero-dividend
expectations over the horizon, nonnegative deterministic forward rates,
frictionless trading without additional carry benefit and true-martingale
discounted stock dynamics. Spot was independently sourced; the forward was
spot divided by discount, not inferred from European parity.

The archived dividend calendar was accepted conditionally on **no subsequent
notice**. This is a documented assumption, not proof of an unbiased historical
forecast. Treasury yields supplied discount proxies. Interval-end observations
were reconstructed snapshots, not executable bid/ask fills. Repeat stock
acquisition passed its local gate, but independent intraday cross-source
agreement was unavailable. These qualifications remain attached to every result.

## Fitting and numerical findings

The [31-attempt table](aggregate_findings/pilot_fit_attempts.csv) includes eight
synthetic controls, 18 dated market fits and five Hurst-profile fits. All original optimizers
converged. Synthetic clean/noisy recovered-price gates passed, but optimizer
convergence did not make market fits accurate or numerically stable. The
original market-fit acceptance flag is false throughout. Bound activity and
held-out errors remain visible rather than selecting a favorable start.

Training objectives alone selected illustrative full-rough fits. At the
original resolution, all 20 market stochastic/profile attempts failed prescribed-grid checks;
BS was numerically stable but had large market errors. Relative-to-spread
errors remained material. These are neither an accepted model ranking nor
identified six-parameter rough dynamics.

The later [nine fixed-vector refinement outcomes](aggregate_findings/pilot_refinement_outcomes.csv)
retain five fully completed/stable outcomes, one **partial timeout** and
three **unstarted at the campaign deadline**. The August 8 full-rough extension
completed nine of ten pricing evaluations; its terminal 4,096-step call was
interrupted and no timeout price array was accepted. Original machine status
is retained separately from the normalized display status. Repricing at finer settings did not refit or
overwrite the original attempts. The
[aggregated refinement checks](aggregate_findings/pilot_refinement_summary.csv)
retain coarse-grid drift, intermediate failures and terminal passes where
available. A terminal numerical pass is not a market-fit acceptance decision.

## Repeating the study needs new inputs

The public repository does not distribute the prepared SPY archive or independent
reference observations. Set up licensed independent stock and option inputs,
dated rate proxies and dividend expectations with explicit availability rules.
Re-run acquisition/preparation gates before any campaign. File presence alone
never establishes calibration readiness. Do not put credentials or vendor data
in the repository.

Notebook 03 therefore displays **live calibration not recomputed** and links to
these aggregates. It performs no parity substitution, de-Americanization,
synthetic fallback, new parameter fitting or data download. Rough historical
state, stock deltas, American puts and self-financing hedging replay remain
unvalidated by this call pilot.
