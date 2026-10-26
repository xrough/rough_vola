# Vanilla pricing, calibration and stock deltas

`rvlab.pricing` is the numerical source layer for the Python volatility research
project. It prices American BS/Heston vanillas with finite differences,
American rough-Heston vanillas with independent-policy simulation, and
European BS/Heston/rough-Heston vanillas with Fourier methods. It fits
exercise-matched surfaces with bounded least squares. Dependencies
are NumPy, SciPy and pandas. It does not import C++, `roughvol`, PyTorch or a
data vendor. Existing `rvlab.models` APIs remain intact.

## American and spot-based European contracts

```python
from rvlab.pricing import (
    BlackScholesParams, DiscountCurve, CashDividend, DividendSchedule,
    SpotMarket, FDConfig, evaluate_vanilla, calibrate_vanilla_surface,
)

market = SpotMarket(
    spot=100., discount_curve=DiscountCurve.flat(.04, 1.),
    dividends=DividendSchedule((CashDividend(.25, 1.2),), coverage=1.),
)
valuation = evaluate_vanilla(
    BlackScholesParams(.2), market=market, strike=[95., 100., 105.],
    maturity=.5, is_call=False, exercise_style="american",
    config=FDConfig(spot_steps=400, time_steps=400), check_refinement=True,
)
print(valuation.prices, valuation.deltas)
print(valuation.diagnostics["checks"])
# fit = calibrate_vanilla_surface(quotes, BlackScholesParams(.25), market=market)
```

`exercise_style` is explicitly `american` or `european`. The volatility model,
call/put payoff and exercise choice are independent. The spot API supports
`BlackScholesParams`, `HestonParams` and `HestonCurveParams`, plus inception-only
`RoughHestonParams`/`RoughHestonCurveParams` with an explicitly supplied finite
kernel and `RoughAmericanConfig`. The rough backend returns statistical
stopping diagnostics described below. General stopping-based rough calibration
remains unsupported; the guarded American-call reduction below supports a
separate deterministic calibration study.
European rough Heston also retains the original forward-based Fourier API.
Bermudan and exotic contracts follow separately.

Numerical times are years from valuation (the market adapter uses ACT/365).
Discount factors must be positive, start at `(0,1)`, and cover every maturity.
Interpolation is linear in log discounts, producing piecewise constant forward
rates, including negative rates. No extrapolation is inferred.
`DividendSchedule((), coverage)` declares no expected dividends through that
horizon; `None` is missing information and is rejected. Cash events occur at
strictly positive times. An observation on an ex-date uses the post-event spot
when the event has already occurred. Schedules are held fixed during stock
bumps. A trailing dividend yield cannot stand in for expected dated cash events.

Cash events use limited liability: `S_after=max(S_before-cash,0)` and the actual
payout is `min(cash,S_before)`. The holder may exercise immediately before the
jump, including a jump at expiry. Fixed-cash European parity need not hold
under this capped-payout convention. The result contains local one-sided stock
derivatives and event diagnostics at the stated reference spot. Event
sensitivities are pre/post-event value-function slopes, not a forecast stock
path. Exercise boundaries are grid estimates at the initial variance; they
are not a state-estimation or hedge-replay engine.

BS uses a nonuniform spot grid, Crank–Nicolson, policy iteration for the
obstacle and two implicit-Euler half steps at terminal payoff and each cash
jump. Heston uses nonuniform spot/variance grids, modified Craig–Sneyd ADI with
Ikonen–Toivanen splitting and coefficient `1/3`, following
[Haentjens and in ’t Hout](https://arxiv.org/abs/1309.0110).
Contracts of one maturity share operators and Heston directional
factorizations. The zero-variance boundary solves the degenerate equation;
the upper spot boundary is linear and upper variance has zero derivative.
The cash jump resets the splitting multiplier and damping. Every clock
includes exact rate, cash and calendar-time mean-reversion knots. Zero
vol-of-vol uses a 1D PDE with the exact time-dependent deterministic variance
path, rather than substituting its average into an American BS price.

Starting budgets are BS 400 spot intervals/400 time steps, Heston
240 spot/80 variance/300 time, `Smax=8*max(spot,strikes)` and `vmax=5`.
Strike insertion can add nodes. Nearby redundant base nodes are removed before
inserting exact spot/strike nodes; this prevents tiny payoff cells from disabling
terminal kink averaging. `FDConfig.spot_concentration` explicitly controls the
sinh-mesh width as a fraction of observed spot (default `.1`). A smaller width
places more nodes near spot and fewer in the wings, so its accuracy must be
checked over all contracts and exercise boundaries. `FDConfig` also exposes
budgets, domains, policy tolerance and iteration cap; `FrozenFDGrid` supports repeatable spot
bumps. They are not accuracy guarantees. Feller violation is supported.
Tests cover moderate variances around `.035–.25`, short and intermediate
maturities, positive/negative rates, discrete cash, and published American
Heston cases with vol-of-vol up to `.9`. Extreme variance, correlation,
moneyness, large cash events and long maturities require independent checks.
Published Heston values are finite-grid references; the Feller-violating test
explicitly allows `.006` per-unit disagreement.

`check_refinement=True` separately doubles spot, time, variance (if active),
domains and the joint spatial/temporal grid. Domain expansion preserves the
central mesh. Price checks use `.001+1e-4*abs(reference price)` and delta checks
use `.002+.005*abs(reference delta)`. Every probe must pass. Failure records
remain visible and block acceptance; tolerances are never relaxed
automatically. These empirical checks do not certify an error bound.
Unchecked prices have stability fields `None`.

`calibrate_vanilla_surface` shares the European calibrator's bounds,
fixed-parameter, coordinate and Jacobian machinery. Initial parameters are
required. One grid/domain stays frozen through every optimizer attempt;
post-fit repricing independently refines it. Midpoint targets use
`max(half_spread, price_scale_floor)` (default `.005`) as residual scales;
price-only synthetic observations use dollars. Spread fits additionally
require **every** probe's price change below 10% of the effective half-spread.
Output preserves rows and separates optimizer convergence, active bounds,
price/delta stability, physical optimality, actual surface pricing evaluations
and runtime. `ready` means optimizer convergence plus price stability;
`hedge_delta_ready` additionally requires delta stability. Neither certifies
historical data readiness, fit quality or parameter identification.
Optimization pricing failures raise `PricingError`; failed post-fit checks
retain the fitted prices and failure records. Exhaustion remains unsuccessful.

Real SPY preparation labels contracts American and requires independent spot,
discount curves and explicitly covered cash expectations. European
`calibrate_surface` rejects explicitly American rows. The public SPY notebook
reports input blockers and historical aggregate findings; it performs no new
market fit without data. The separately preserved three-date American-call pilot
uses the guarded reduction below, with an explicit archived-calendar assumption.
Its optimizer convergence, Fourier-tail failures and subsequent fixed-parameter
checks are documented in [the public findings](../../../reports/spy_pilot.md).
Previous European SPY results remain superseded provenance. Numerical delta
readiness and historical hedging readiness remain separate.

## American calls with no early-exercise value

`price_american_calls` and `calibrate_american_call_surface` support constant
Black--Scholes, Heston and rough Heston through an explicit mathematical
reduction. Under a complete empty expected-dividend schedule, nonnegative
deterministic forward rates, frictionless trading without additional carry
benefits, and a true-martingale discounted stock, an American call equals the
corresponding European call. The route preserves American contract labels;
it does not de-Americanize observed targets or estimate spot from parity.
Forwards are derived from independent spot as `S / D(T)`.

```python
from rvlab.pricing import (
    RoughHestonParams, SpotMarket, DiscountCurve, DividendSchedule, PricingConfig,
    price_american_calls, calibrate_american_call_surface,
)

params = RoughHestonParams(.04, .045, 1.5, .3, -.6, .15)
market = SpotMarket(100., DiscountCurve.flat(.04, .25), DividendSchedule((), .25))
prices = price_american_calls(params, market=market, strike=[95., 100., 105.],
    maturity=.1, config=PricingConfig(192, 160, 512))
# quotes must retain exercise_style='american' and is_call=True on every row.
# fit = calibrate_american_call_surface(quotes, params, market=market,
#     bounds=explicit_bounds, config=PricingConfig(192, 160, 512))
```

Puts, nonempty dividends, incomplete coverage, negative forward-rate intervals,
curve parameter objects and historical conditional states are rejected. Every
relevant curve interval is checked, even when the total discount implies a
positive average rate. No stopping-engine fallback occurs.

Prices use the existing analytic/Fourier/fractional-Riccati implementations.
Calibration shares the bounded least-squares, coordinate and spread-weighting
workflow. Its default price checks remain `.001+1e-4*abs(price)`; every separate
quadrature, cutoff and Riccati probe must also change prices by less than 10%
of effective half-spread. Optimizer convergence, numerical stability, economic
fit and identification remain separate. Results retain backend/assumption
metadata, physical-residual Jacobians and singular values; they validate no
stock Greeks or historical hedging state. The European calibrator continues
to reject American-labelled inputs.

The martingale argument belongs to the model, not the numerical identity
`CF(-i)=1`: see Abi Jaber, Larsson and Pulido, [Affine Volterra processes,
Theorem 7.1(iii) and Lemma 7.3](https://arxiv.org/abs/1708.08796).

The bounded three-date SPY study has a separate executable entry point:

```bash
.venv/bin/python scripts/acquire_spy_call_pilot.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
  .venv/bin/python examples/spy_rough_calls.py
```

The first command prompts locally for credentials and preserves a new raw
OHLCV acquisition. The second writes only the new
`lab/reports/artifacts/spy_rough_calls_v1` campaign; real fits require exact
regular-session agreement with the first stock download and a passed synthetic
recovered-price gate. Its calendar assumption applies only to the declared
pilot through September 5, 2025. It changes no canonical readiness or notebooks.
The numerical clock allows at most one hour, stops parameter optimization at
40 minutes, and retains failures, exhausted attempts and input sensitivities.

## European forward-based conventions

- Supply maturity-matched **forward** `forward` and **discount factor**
  `discount` explicitly. With flat rates and continuous dividends these are
  `S * exp((r-q)*T)` and `exp(-r*T)`. Curve construction, discrete dividends
  and calendar-to-year conversion are outside this forward-based interface.
- Prices are per underlying unit. Apply contract multipliers in portfolio
  accounting. Maturities are years, volatility is a decimal, and `v0` and
  `theta` are annualised **variances**.
- Heston uses `dV = kappa*(theta-V)*dt + vol_of_vol*sqrt(V)*dB`, with stock/variance
  correlation `rho`. The Feller condition is not required.
- Rough Heston places these drift and diffusion terms inside convolution with
  `K(t) = t**(hurst-.5) / Gamma(hurst+.5)`. Its coefficient time scaling differs
  from Heston when `hurst != .5`; `hurst=.5` delegates to Heston exactly.
- Quote inputs broadcast using NumPy rules. Scalar inputs return a float;
  arrays retain their broadcast shape. `is_call` contains booleans.
- These are initial-state European models for synthetic research. They do not
  infer historical rough memory or produce conditional hedge ratios. Real SPY
  uses the exercise-aware spot API above.

## European Fourier pricing

```python
from rvlab.pricing import HestonParams, PricingConfig, price

params = HestonParams(v0=.04, theta=.045, kappa=1.5,
                      vol_of_vol=.3, rho=-.6)
config = PricingConfig(quadrature_nodes=128, fourier_cutoff=80., riccati_steps=256)
values = price(params, forward=100., discount=.99,
               strike=[95., 100., 105.], maturity=.1, config=config)
```

The stochastic models use the characteristic function of `log(S_T/F_T)` and a
Lewis Fourier contour with a Black–Scholes control variate. Contracts sharing
a maturity reuse a characteristic-function grid. Rough Heston uses a
fractional Adams product integration, including the nonzero forcing at the
initial node. The corrector solves its scalar complex quadratic implicitly,
using the finite physical root and cancellation-safe evaluation. This avoids
the explicit predictor's high-frequency stiffness restriction; it does not
remove time-discretization error.

`PricingConfig` exposes quadrature-node count, Fourier cutoff and Riccati step
count. `config.refined()` retains its original inexpensive joint-grid utility
(double resolutions, 25% longer cutoff). It is insufficient as the sole
acceptance check: the omitted tail can remain material and different errors
can cancel.

`pricing_diagnostics(params, ..., config=config)` separately checks quadrature,
extends the cutoff to 1.5 and 2 times its original value at fixed quadrature
and time resolution, checks the cumulative tail change, then doubles the
Riccati resolution for rough Heston. A joint comparison also checks accumulated
error. All comparisons must pass. Probe failures remain explicit table rows
and block stability. These are empirical checks, not a certified error bound.

Small Hurst values, long maturities, strong vol-of-vol, limiting correlations
and extreme moneyness can require substantially finer settings. Increasing
cutoff without enough Riccati steps can give inaccurate prices. Nonfinite values
raise `PricingError`; they are never replaced by zero Fourier contributions.
Validate across resolutions for the actual parameter and quote range. Default
settings are not a universal accuracy guarantee.

## American rough-Heston prices and stock deltas

```python
from rvlab.pricing import (
    RoughHestonParams, RoughAmericanConfig, approximate_fractional_kernel,
    DiscountCurve, DividendSchedule, SpotMarket, evaluate_vanilla,
)

params = RoughHestonParams(.04, .045, 1.5, .3, -.6, .15)
market = SpotMarket(100., DiscountCurve.flat(.04, 1.), DividendSchedule((), 1.))
kernel = approximate_fractional_kernel(.15, 1., 4, min_time_ratio=.003)
estimate = evaluate_vanilla(params, market=market, kernel=kernel,
    strike=100., maturity=60/365, is_call=False, exercise_style="american",
    config=RoughAmericanConfig(training_paths=2048, valuation_paths=2048,
        steps=128, exercise_steps=32, training_seed=42, valuation_seed=43,
        training_method="fourier_premium", premium_cap=.7))
print(estimate.prices, estimate.standard_errors, estimate.dual_prices)
print(estimate.dual_gaps, estimate.dual_gap_standard_errors)
print(estimate.deltas, estimate.delta_standard_errors)
```

These deliberately bounded example budgets are **not an accuracy guarantee**.
The immutable policy fits quadratic stock splines and all finite-kernel
factor coordinates, including factor cross products. Normalization and
regression use training paths only. Valuation uses a distinct recorded random
stream; matching training and valuation seeds are rejected. This includes
separate policies for flat and calendar-time mean-reversion curves. Dividend
dates enter the exercise calendar exactly, with exercise before or after the
limited-liability cash jump. European exercise is terminal/post-event only.

`training_method='least_squares'` regresses future realized stopping cashflows.
The optional `'conditional'` method instead projects exact one-step Bellman
conditional expectations backward on the simulator time grid. Training dates
and eligible exercise dates are distinct; intermediate numerical dates do not
introduce early exercise. Conditional labels remove cashflow sampling noise,
but repeated polynomial extrapolation can still amplify projection error.
Notebook 01 preserves failed uncapped trials rather than treating the method
as automatically more accurate.

For an explicitly empty cash-dividend schedule, `training_method='fourier_premium'`
learns only the stopping premium relative to an affine European-value proxy.
The regression cashflows are first controlled by exact-discrete European
martingale increments. Independent valuation combines the European proxy
with the fitted premium, centering each function under the actual simulation
law. A Fourier approximation is never substituted for the conditional mean
of a discrete transition. Its numerical error affects control quality, not
the primal/dual target. `premium_control=RoughEuropeanControlConfig(24,100.,512)`
exposes its Fourier/Riccati settings. `premium_cap=.7` is an explicit regression
and surrogate stabilization choice, not a price bound or acceptance tolerance.
Exercise-policy mistakes remain visible in the independent dual gap. Full
stock-delta and policy-refinement diagnostics are unchanged.

The alternative `'fourier_bellman'` projects controlled one-step expectations
at every integration date. Exercise is still restricted to the declared
exercise dates. Both Fourier-premium methods currently reject positive cash
dividends: a no-dividend European baseline needs signed residuals around cash
jumps, which this version does not fit. The cash-capable whole-value methods
remain available and keep their failed accuracy flags visible.

For whole-value methods, `surrogate_interpolation='value_bridge'` bridges the
left continuation function to the right exercise value, including the terminal
payoff. The historical `'coefficients'` interpolator is retained for reproducible
ablations; it loses this terminal anchor between exercise dates. The repaired
bridge changes only the centered control, leaving the frozen stopping policy
and raw cashflows identical. It reduces variance in the recorded trials but
does not by itself meet the accuracy targets.

`bounded_surrogate=True` stabilizes the continuation proxy with a stock cap
for calls and `K*max_future_discount/D(t)` for puts, including negative rates.
The conditional integral partitions at payoff/cap intersections and integrates
the selected polynomial directly, avoiding subtraction of huge tail moments.
These caps constrain the approximate function, not the reported Monte Carlo
price. They neither replace numerical failures by zero nor establish a price
upper bound for the weak stock law. The independent primal/dual gap remains
the policy-quality check; nonfinite coefficients and inconsistent integrals
raise identifiable failures. The uncapped switch remains available to
reproduce the documented failed method.

Rare cancellation in the three truncated lognormal moments is checked before
accepting a conditional integral. Flagged rows recompute the same analytic
formula with 60–100 decimal digits, an analytic Gaussian-CDF series and an
explicit error certificate. This uses Python's standard library and retains
the original local precision guard. It is not a nested simulation or Gaussian
quadrature. Unresolvable cap crossings still raise a numerical failure.

At every simulator microstep the engine constructs a martingale increment by
enumerating the three variance branches and two stock-order branches, then
integrating the remaining Gaussian exactly. Piecewise quadratic maxima with
the payoff use truncated lognormal moments. A stopped martingale reduces the
policy-value variance; its running value produces a dual upper estimate,
following [Andersen and Broadie](https://pubsonline.informs.org/doi/10.1287/mnsc.1040.0258).
The primal and dual **expectations** bound only the discrete simulation law
and declared exercise dates for this finite kernel. Sample means are not
clipped to theoretical option bounds; their standard errors and the paired
gap error are returned. A finite sample is not a deterministic bound on the
continuous American fractional model. Conditional valuation standard errors
also exclude uncertainty from training the exercise policy.

A failed paired-gap gate does not establish that the lower policy value is
biased by the whole gap. The upper martingale proxy can be loose. Notebook 01
reports independently refined BS/Heston reference errors separately from the
paired-gap certificate; a correct limit price can pass its reference check
while the available primal/dual certificate remains insufficient.

Stock deltas use paired common-random-number bumps at `h` and `h/2`, keeping
parameters, the initial factor memory, rates and cash schedule fixed. The
returned delta uses `h/2`; left/right estimates are inception stock-bump
sensitivities, not pre/post-dividend state forecasts. Setting
`retrain_bumps=True` adds a separately trained bumped-policy comparison.
A small price dual gap does not certify an optimal-value delta. At expiration
the one-sided sensitivities retain the payoff kink.
Persistent one-sided slope differences under bump refinement block smooth
delta readiness, even if symmetric differences appear stable.

`check_refinement=True` runs independent time, exercise-date, sampling,
training, bump, kernel and joint checks. Supply a distinct
`refinement_kernel=...`; a missing kernel check blocks readiness.
Only changing kernel metadata does not count as refinement; the exact constant
H=.5 kernel needs no approximation probe. New training
streams and bumped retraining probe policy error. Refinement changes include
sampling uncertainty and retain the original price allowance
`0.001 + 1e-4*abs(price)` and delta allowance
`0.002 + 0.005*abs(delta)`. The stopping-gap and price sampling intervals must
share the price budget. Failures remain visible in `diagnostics['checks']`.
Unchecked calls have `price_stable=delta_stable=None` and readiness false.
Kernel/exercise refinements provide empirical evidence, not a global error
proof. Explicit kernels prevent an unreported approximation choice.

The demonstrated range is spot 100, strikes 95–105, maturities 14–60 days,
H=.15 (plus the H=.5 Heston control), v0=.04, theta=.045, kappa=1.5,
vol-of-vol=.3, rho=−.6, and the declared mean-reversion/dividend variants in
notebook 01. These are tested scenarios rather than a universal supported
accuracy region. Cone violations, overflow and memory exhaustion raise
identifiable errors. Historical conditional rough states remain unsupported;
resetting their factor memory to zero is rejected. Calibration through this
general stopping backend, SPY state estimation and portfolio replay remain
later milestones. The separate no-dividend American-call reduction above
supports deterministic calibration under its explicitly checked conditions.

## Rough-Heston dynamics foundation

`rough_lift` and `rough_simulation` establish inception-only dynamics used by
the stopping layer. Their historical European controls remain archived with
their original source and results. The foundation itself does not implement
stopping, rough-model calibration or conditional memory estimation.

The immutable `ExponentialKernel` represents
`K_N(t)=sum(w_i*exp(-x_i*t))`, with positive weights and ordered nonnegative
nodes. `approximate_fractional_kernel` fits exact fractional kernel cell
masses, including the singular first cell. It exposes the factor count,
construction grid, smallest cell and optimizer budget. A failed or exhausted
construction raises `PricingError`; it is never a successful approximation.
The fit is a deterministic local construction, rather than the optimal
quadratures in [Bayer–Breneis](https://arxiv.org/abs/2309.07023).

For flat and calendar-time mean-reversion levels, `lifted_heston_price` uses
an independent finite-factor affine transform with the shared Fourier
quadrature. `lifted_forward_variance` and `lifted_integrated_variance` solve
the mean equations exactly. These references separate kernel error from
simulation error. Transform prices assume no discrete cash dividends; they
are not obtained by subtracting the present value of cash from spot.

```python
from rvlab.pricing import (
    RoughHestonParams, approximate_fractional_kernel, lifted_heston_price,
    RoughSimulationConfig, simulate_rough_heston, european_payoff_samples,
    DiscountCurve, DividendSchedule, SpotMarket, PricingConfig,
)

params = RoughHestonParams(.04, .045, 1.5, .3, -.6, .15)
# Explicit research construction; validate for each horizon/parameter range.
kernel = approximate_fractional_kernel(.15, 1., 4, min_time_ratio=.003,
                                      max_nfev=4000)
market = SpotMarket(100., DiscountCurve.flat(.04, 1.),
                    DividendSchedule((), coverage=1.))
paths = simulate_rough_heston(params, market, kernel,
    observation_times=[.25],
    config=RoughSimulationConfig(paths=8192, steps=1024, seed=42))
samples = european_payoff_samples(paths, strike=[95., 100., 105.],
                                  maturity=.25)
reference = lifted_heston_price(params, kernel=kernel,
    forward=100./market.discount_curve.discount(.25),
    discount=market.discount_curve.discount(.25),
    strike=[95., 100., 105.], maturity=.25,
    config=PricingConfig(384, 320., 1024))
print(samples.mean(axis=0), reference)  # Inspect uncertainty and refinement too.
```

The simulator uses exact affine drift steps and nonnegative three-point
variance transitions, following the weak splitting construction in
[Bayer–Breneis](https://arxiv.org/abs/2310.04146) and its
[admissible-state formulation](https://arxiv.org/abs/2412.17526).
Individual memory factors can be negative; they are not clipped. Material
admissible-state violations raise `RoughSimulationError`. The implemented
three-point rule matches the first four square-root-transition moments
exactly; the fifth has an explicitly tested order-three local discrepancy.
No universal convergence order or accuracy is inferred from that fact.
At positive horizons, supported vol-of-vol is exact zero or at least `1e-8`.
Smaller positive values raise `RoughSimulationError`: the correlated-stock
representation divides factor increments by vol-of-vol and loses precision
there. No near-zero input is silently replaced by the deterministic model.

The default `RoughSimulationConfig.stock_integral="exact_drift"` also integrates
variance exactly along each deterministic half-flow in the stock generator.
The common-noise stock increment is reconstructed directly from the diffusion
factor jump. This removes the subtraction of stiff drift integrals that caused
large errors in the original stock step. `stock_integral="trapezoidal"` retains
that original method for reproducible comparisons. The factor transition law
is identical in both modes, so factor stiffness still needs independent time
refinement. All three diffusion branches are checked for admissibility.

Every rate, dividend, observation and mean-reversion knot is included exactly.
Cash payouts are capped by the pre-event stock and discounted gains include
the actual payout. Factors remain continuous through cash events. No-cash
European payoffs can integrate the independent stock Brownian driver
analytically; positive cash events disable that calculation. Zero
vol-of-vol uses exact deterministic variance paths and integrals. The
`hurst=.5` limit is forced through the simulator with one constant factor,
providing an independent check rather than a dispatch identity.

Results retain the whole factor state at requested observations, raw stocks,
variance, discounted gains and diagnostics. A memory budget is checked before
allocation. The seed reproduces a fixed experiment across path-batch sizes;
changing path count or time grid changes the random stream. A historical
conditional state is currently unsupported and cannot be replaced by zero
memory at a later valuation date.

Notebook 01 preserves the naive and revised constructions. On its nine
contracts (14/60/365 days, strikes 95/100/105, spot 100, H=.15), the default
eight-factor kernel matches the fractional Fourier reference but contains
rates near 1.8 million per year. The splitting scheme has substantial bias
at the initial step budgets. A four-factor construction with
`min_time_ratio=.003` reduces the maximum rate to about 542 and passes the
same European price checks for both flat and curved mean reversion. This is
a tested candidate, not a generally recommended kernel. Raising factor count
alone can worsen simulation accuracy.

The study keeps the price allowance `.001+1e-4*abs(reference)` unchanged.
Monte Carlo acceptance additionally requires
`abs(estimate-reference)+1.96*standard_error` below that allowance and stable
independent Fourier references. The stochastic simulation budgets in the
**preserved original study** do not satisfy that strict gate. Deterministic limits pass; statistically
consistent moment or martingale checks do not certify option-price accuracy.
The subsequent precision study compares the two stock methods under stronger
European controls and repeats the final flat, curve and Heston-limit checks
with independent seeds. American rough pricing and delta readiness remain false.

### European simulation estimates with controlled uncertainty

```python
from rvlab.pricing import (
    estimate_rough_european, RoughEuropeanControlConfig,
)

estimate = estimate_rough_european(params, market, kernel,
    strike=[95., 100., 105.], maturity=.25,
    config=RoughSimulationConfig(paths=2048, steps=2048, seed=42),
    control_config=RoughEuropeanControlConfig(24, 100., 512))
print(estimate.prices, estimate.standard_errors)
print(estimate.conditional_prices, estimate.conditional_standard_errors)
```

The fixed-coefficient martingale control uses an approximate conditional
affine-Fourier value `g`. At each step it subtracts
`g(next)-E_discrete[g(next)|current]`. The expectation enumerates the shared
three variance branches and two independent-stock orders exactly. Its
expectation is zero under the actual simulated law, so the estimator retains
the discretization bias. Surrogate quadrature and Riccati budgets affect
variance reduction rather than the discrete expectation. The estimator takes
no reference price as an input. Even a poor surrogate must reproduce a
directly enumerated one-step discrete price; tests explicitly preserve the
gap from the continuous model in that case.

All contracts share IID paths and retain their broadcast order. Reported
standard errors are path sample standard deviations divided by the square
root of the path count. The result retains raw conditional-payoff and control
means, standard errors and their covariance matrices, allowing independent
replications to be pooled with between-run shifts included. The seed and path
count must identify independent runs; reusing paths cannot increase sample
size. A controlled estimate can have tiny sampling error while its time grid
or kernel remains inaccurate.

This estimator supports inception European payoffs without positive cash
dividends. Cash-event simulation remains available through
`simulate_rough_heston`; full factor paths and stock draws are unchanged by
the European control. It supplies no exercise policy, stock delta or
conditional historical state. Notebook 01 validates the actual tested
parameter and contract range; refinement remains necessary outside that range.

The recorded precision campaign uses the same nine contracts at 14/60/365 days,
spot 100 and strikes 95/100/105. For rough Heston it tests H=.15, v0=.04,
kappa=1.5, vol-of-vol=.3 and rho=-.6, with flat theta=.045 or levels
.035/.065/.045 at the 20/45-day knots; the H=.5 control uses the same flat
coefficients and one constant kernel. At 2,048 steps and three independent
2,048-path replications, all 27 pooled contract/specification checks pass both
the finite-lift and fractional-reference gates. Maximum fractional-reference
errors are approximately `.000203` for flat, `.000199` for curve and `9e-7`
for the Heston limit; the largest 95% sampling half-width is below `.00008`.
The corrected 256-step grid and the old stock rule still fail, and those
trials are preserved. These results validate the declared European sample,
rather than arbitrary parameters, cash-event option values or stopping rules.

## European surface calibration

```python
from rvlab.pricing import calibrate_surface

result = calibrate_surface(quotes, params, forward=100., discount=.99,
                           config=config, max_nfev=200)
print(result.params)
print(result.success, result.numerical_stable, result.ready)
print(result.fitted_quotes)
```

`quotes` contains `strike`, positive `maturity`, boolean `is_call`, and either
`market_price` or both `bid` and `ask`. Bid/ask targets their midpoint, with
residual `(model-mid)/max(half_spread, price_scale_floor)`; the default per-unit
floor is `.005`. A supplied `market_price` is retained but ignored in that
case. Price-only inputs use dollar residuals with scale `1`. Forward and
discount may be scalars or positionally aligned arrays. Invalid rows are
rejected explicitly; output preserves row order and index.

Initial parameters are required. Optional `bounds` maps names to `(lower,
upper)` pairs. Equal endpoints fix a parameter, useful for controlled recovery
and partial identification studies. Default search boxes are documented in
`calibrate_surface`; they are configurable numerical limits, not market priors.

Curve levels and kappa can have a strongly curved product ridge. Optional
`OptimizerConfig` exposes monotone coordinates that keep exactly the same
physical bounds, including attainable zero levels:

```python
from rvlab.pricing import OptimizerConfig, calibrate_surface

result = calibrate_surface(quotes, initial_curve_params,
    forward=quotes.forward, discount=quotes.discount, bounds=bounds,
    config=config, max_nfev=80,
    optimizer=OptimizerConfig(parameterization="asinh"))
print(result.optimizer_settings)
print(result.optimality, result.physical_optimality)
```

`physical` preserves the original optimization coordinates. `asinh` and
`log1p` act on nonnegative parameters, with scales `.0025` for variance levels
and `.01` for kappa, vol-of-vol, BS volatility and Hurst; rho stays linear.
The scales change optimizer geometry rather than the search box. These are
explicit numerical choices, not new model assumptions or regularization.

Transformed fits estimate a bound-aware Jacobian in **physical** parameters
and then apply the exact coordinate chain rule. This prevents the transform
from shrinking finite-difference steps below useful price precision near zero
curve levels. Two-point and three-point schemes use absolute step floors;
raw `parameter*relative_step` is unsafe near zero. Jacobian choice and all
resolved settings are retained. The default stopping controls remain
`ftol=None`, `xtol=gtol=1e-9`; tolerances can be declared in `OptimizerConfig`.
The underlying optimizer is [SciPy's bounded trust-region least squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).

SciPy's `optimality` uses the optimizer coordinates and should not be compared
numerically across nonlinear transforms. `physical_optimality` converts the
final Jacobian back to physical coordinates and reports
`max(abs(gradient)*distance_to_bound_in_descent_direction)`, for the gradient
of half the **sum** of squared residuals. Compare this measure only for the
same data, residual weights and physical box. It assesses local stationarity,
not identification or a global optimum. Optimizer success, numerical
refinement, physical stationarity and fit error are reported separately.

The same optimizer argument passes through `calibrate_surface_refined` and
the demonstration `fit_attempt`. The [archived convergence investigation](../../../notebooks/legacy/pricing/07_spy_curve_convergence.ipynb)
compares ordinary and transformed coordinates with paired coarse/fine budgets,
and separately profiles Heston's kappa and volatility bounds.

`CalibrationResult` retains optimizer status, optimality, parameters, bound
activity, settings, and aligned fitted prices/residuals. `nfev` is SciPy's
optimizer count; `pricing_evaluations` also includes finite-difference Jacobian
calls and post-fit probes. `objective` is mean squared scaled residual. The
`diagnostics` table separates the numerical comparisons. `refined_config` and
refined prices refer to its final reference grid (twice the fitted cutoff),
retaining fitted and reference objectives separately. `max_price_change` is
the largest available change across all checks; the per-row
`refinement_difference` column describes the joint comparison only.

`ready` requires optimizer convergence and every numerical comparison to be
within `stability_atol + stability_rtol * abs(comparison_price)`. Defaults are
`.005` per unit and `1e-4`. This does not establish fit quality, parameter
identifiability, economic adequacy or trading value. Optimizer exhaustion
returns an unsuccessful result. Numerical pricing failures during optimization
raise `PricingError` with calibration-stage context. Failed post-fit probes
preserve the valid fitted parameters, optimizer status and failure diagnostics;
unavailable reference prices are NaN, and the result is never ready.

An inaccurate-grid optimum must be **refitted**, not just repriced. Use a finite,
explicit continuation when the first grid fails:

```python
from rvlab.pricing import calibrate_surface_refined

study = calibrate_surface_refined(
    quotes, params, forward=forwards, discount=discounts,
    configs=[PricingConfig(192, 160., 512), PricingConfig(192, 400., 1024)],
    bounds=bounds, max_nfev=40,
)
print(study.ready, study.total_nfev, study.total_pricing_evaluations)
for stage in study.stages:
    print(stage.config, stage.error, stage.result.ready if stage.result else False)
```

Each optimizer runs on one fixed grid and warm-starts the next declared grid.
The evaluation budget applies per stage. Exhausted, unstable and failed stages
remain in the history; invalid inputs still raise. These grids originated in a
superseded European approximation to SPY and remain historical examples, not
a supported real-data workflow or universal settings. A numerically stable fit
can still hit bounds or fit synthetic observations poorly.

## Synthetic example and tests

From the repository root, in an environment with the package dependencies:

```bash
python examples/pricing_calibration.py
python examples/pricing_calibration.py --model rough_heston
python -m pytest src/rvlab/tests/test_pricing.py src/rvlab/tests/test_pricing_calibration.py -q
```

The example fits Black–Scholes and Heston, then demonstrates rough-Heston `v0`
recovery with its other parameters fixed at known synthetic values. All three
report numerical refinement. No market data is downloaded or silently
synthesised by the pricing or calibration APIs.

The [pricing demonstration notebooks](../../../notebooks/pricing/README.md)
extend this with parameter effects, numerical convergence, a small explicitly
approximate SPY case study, and bounded all-six-parameter rough-Heston fits on
clean and noisy synthetic surfaces. Their saved results separate parameter
recovery, price recovery, and numerical readiness.

The fit-attribution notebook further separates restrictive demonstration bounds,
American-exercise/forward sensitivity, and joint maturity/smile restrictions.
Numerically stable prices do not by themselves establish a good fit to quoted
spreads or identify the underlying parameters.

The model equations follow El Euch and Rosenbaum,
[*The characteristic function of rough Heston models*](https://arxiv.org/abs/1609.02108).
The sibling `roughvol` implementation was audited as a development reference;
this package uses local implementations and does not require its checkout.

## Maturity-dependent forward variance

`HestonCurveParams` and `RoughHestonCurveParams` extend the existing models
without changing their interfaces. They use a nonnegative piecewise-constant
mean-reversion level `theta(t)` to generate the initial expected variance
curve `xi0(t) = E[V_t]` under the pricing measure. The model is

\[
V_t=v_0+\int_0^t K(t-s)\kappa[\theta(s)-V_s]ds
    +\eta\int_0^t K(t-s)\sqrt{V_s}\,dB_s.
\]

For Heston, `K=1`; rough Heston retains
`K(t)=t**(H-.5)/Gamma(H+.5)`. Nonnegative immigration `kappa*theta(t)` is a
conservative admissible curve family. Merely requiring an arbitrary proposed
`xi0` to be positive is insufficient. The relation is
`xi0 + kappa*K*xi0 = v0 + K*(kappa*theta)`; this construction follows the
admissible inputs in [Abi Jaber and El Euch, Example 2.2 and Theorem 2.3](https://arxiv.org/html/1803.00477).

```python
from rvlab.pricing import (
    MeanReversionCurve, RoughHestonCurveParams, PricingConfig,
    forward_variance, integrated_forward_variance, price,
)

# Knot times are years FROM VALUATION. Values are theta levels, not xi0 knots.
curve = MeanReversionCurve((15/365, 22/365), (.03, .08, .02))
p = RoughHestonCurveParams(v0=.02, kappa=1., vol_of_vol=.3, rho=-.65,
                          curve=curve, hurst=.15)
xi = forward_variance(p, [0., 15/365, 22/365, 29/365])
expected_integrated_variance = integrated_forward_variance(p, 29/365)
value = price(p, forward=633., discount=.997, strike=640., maturity=29/365,
              config=PricingConfig(192, 400., 1024))
```

The final theta level extends indefinitely. This extrapolation is explicit;
three short expiries cannot identify a long-term variance curve. A constant
schedule recovers the legacy stochastic model, and `hurst=.5` delegates to time-dependent
Heston. Forward-based European pricing with zero vol-of-vol uses the integrated
expected variance and an analytic lognormal terminal law; the rough deterministic curve uses the resolvent mean
directly instead of the legacy Riccati discretization. With `kappa=0`, the
schedule has no influence. As in the original rough model, coefficients inside
the kernel have H-dependent time scaling. These are time-zero models, without estimated
historical factors or conditional rough-model state.

The Riccati equation retains its `-kappa*h` term. For `alpha=H+.5`, the curve
enters the log characteristic function as

\[
v_0 I^{1-\alpha}h(T)
  +\int_0^T \kappa\theta(s)h(T-s)ds.
\]

Rough integration splits at the exact calendar knots, integrating the
piecewise-linear Riccati solution rather than snapping the knots to a grid.
Heston uses analytic exponent increments. Expected forward variances use the
fractional relaxation resolvent; their numerical evaluation is checked against
independent Mittag-Leffler series in the tests.

`calibrate_surface` accepts these parameter types in the same workflow.
`parameter_values(p)` exposes scalar coordinates: `v0`, `kappa`, `vol_of_vol`,
`rho`, optional `hurst`, and `theta_0`, `theta_1`, ... . Curve levels default to
[0,2]; supplied equal bounds fix a coordinate. Knot times are fixed structural
inputs and are not calibration variables. `replace_parameters` changes these
coordinates while preserving knots. Diagnostics include fractional-time
refinement for active rough curves. No parameter range or curve family carries
a universal Fourier-accuracy guarantee; inspect the recorded refinement checks.

The [SPY calibration notebook](../../../notebooks/pricing/03_spy_calibration.ipynb)
uses American Heston curves and direct raw-price calibration when independent
spot and dated cash expectations are available. Until then it displays the input
blockers. Previous European fits and failed trials remain superseded provenance.
Additional curve coefficients do not establish identification or hedging value.

Curve tests include independent calendar-time ODE integration, the equivalent
forward-variance transform, fractional series and spectral-mixture references,
and synthetic multi-maturity price recovery:

```bash
python -m pytest src/rvlab/tests/test_pricing_variance_curve.py \
  src/rvlab/tests/test_pricing_curve_reference.py \
  src/rvlab/tests/test_pricing_curve_calibration.py -q
```
