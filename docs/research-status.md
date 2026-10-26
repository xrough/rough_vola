# Research status

This is a first-stage numerical research report. Its components have different
acceptance levels. Optimizer convergence alone is never a pricing-accuracy test.

## Numerical engines

European BS, Heston and rough Heston share forward/discount conventions.
The rough-Heston solver includes the initial fractional forcing term and handles
the Hurst 0.5 and deterministic-variance limits explicitly. Failures raise
identifiable exceptions. Finite-difference American BS/Heston engines include
exercise, dated cash dividends, stock deltas and separate grid/domain checks.

Acceptance thresholds remain `0.001 + 1e-4 * abs(price)` for price changes and
`0.002 + 0.005 * abs(delta)` for stock-delta changes. Spread-weighted fits also
require every refinement change below 10% of the effective half-spread.
These are empirical acceptance checks, not rigorous universal error bounds.

## General American rough-Heston research

The target is approximated by a finite-factor Volterra model, a simulated time
grid and an exercise calendar. Continuation policies are trained independently
of valuation paths. Policy payoffs give lower estimates; martingale controls
reduce variance and support dual upper estimates. Their gap combines policy
error and upper-estimator slack. It does not isolate either source by itself.

The historical study explored controls, Bellman lookahead, independent-path
checks, stock bumps and physical/kernel/exercise refinements. In the final
policy-comparison study all 36 primary general price budgets failed. A separate
focused, high-path-count case passed its discrete-law bracket-and-sampling
budget. That success does not certify the general optimal stopping price or an
optimal-policy hedge delta. Refinement sensitivity remains material.

Historical result bundles are preserved outside the public source tree.
The public notebook distinguishes those findings from its small fresh examples.

## Calibration and identification

Synthetic studies validate recovered prices across strikes and maturities,
including multiple starts, observation noise and spread-normalized residuals.
They do not demand unique recovered parameters. Long maturities can add
information while weakly identified variance levels and mean reversion remain
coupled. Bound activity, Jacobian singular values, budget exhaustion and
numerical stability must accompany the fitted surface.

## SPY pilot

The historical pilot uses August 7, 8 and 11, 2025, at reconstructed 15:45 New
York interval ends, with expiries August 22, August 29 and September 5.
It uses American calls, observed stock and prior-day ALFRED-vintage Treasury
discount proxies. The July 31 issuer-calendar capture supports an empty expected
schedule through September 5 subject to the accepted assumption of no later
special distribution or schedule-change notice.

All 31 pilot optimizations terminated successfully, including synthetic,
market and Hurst-profile attempts. The eight synthetic fits passed the declared
refinement checks. The prescribed market grid's Fourier cutoff of 160 was
insufficient for all 20 stochastic/profile market fits, despite optimizer
convergence. BS was numerically stable with poor market fit.

A bounded follow-up repriced fixed parameter vectors at larger cutoffs. Five
candidates passed its full checks; one was incomplete and three were not started
before the deadline. These findings diagnose truncation error and retain the
need to refit at validated settings. They establish no accepted model ranking,
unique Hurst estimate, historical rough trajectory or hedging readiness.

The stock repeat download agreed exactly on all 2,541 timestamps, closes and
volumes, including 1,170 regular-session minutes. A second-vendor stock check
was unavailable. This validates repeatability within the acquisition route,
without establishing independent vendor agreement, measured publication latency
or executable prices.

## Next research milestones

1. Recalibrate the conditional American-call pilot at validated Fourier settings.
2. Separate input sensitivity and parameter identification from price-fit quality.
3. Resolve historical cash expectations, financing and rough-state requirements
   before a causal hedge replay.
4. Evaluate policies on identical option positions using common accounting and
   explicit exercise/dividend events.
