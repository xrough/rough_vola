# Selected roughVol components

This distribution includes a small `roughvol` namespace so the existing `rvlab`
synthetic and feature interfaces work without an adjacent checkout. The canonical
BS/Heston/rough-Heston pricing, American exercise, calibration and market-data
validation remain in `rvlab`; none of the old CF pricers, IV calibration machinery,
rough-Heston simulation variants, downloader or empirical market outputs was copied.

The imported source came from the local working tree of `xrough/roughVol`
(`https://github.com/xrough/roughVol`), based on commit
`b4fbaf901fb8e412659d782c03cfece0dfb6c0ef`. That working tree contained uncommitted
changes; in particular the hedge ledger and its tests were untracked. The hashes
below identify the actual files read, rather than claiming they all match that
commit. Original repositories and old results were left untouched. The sibling's
license file was an incomplete MIT heading; this distribution supplies its own
complete project license. These are project source imports, not an assertion
that cited research papers or vendor datasets have the same license.

## Included methods and local corrections

- Rough Bergomi: midpoint and hybrid Volterra drivers, plus joint
  Riemann–Liouville/Brownian Gaussian covariance simulation as a small-grid
  reference. All use one NumPy backend; optional JAX dispatch was removed.
- The hybrid near-field residuals are now sampled jointly across lag kernels.
  Independent residuals gave incorrect temporal covariance when the near field
  contained multiple lags. The hybrid grid is explicitly uniform; nonuniform
  input is rejected. Antithetic near-field residuals and the exact simulator's
  orthogonal Brownian innovations now share the declared negative pairing.
- `var` and `variance` expose the same variance path, resolving the existing
  `rvlab.data.rbergomi_panel` interface mismatch. Nonfinite model inputs and
  variance curves are rejected.
- Session/group roughness and realized-variance helpers preserve local dates
  and clocks. The `rvlab` wrapper applies its requested log transformation and
  retains session groups. Variance blocks have a stable schema, including
  explicit annualized/raw fields and compatibility aliases.
- Full-sample intraday deseasonalization remains an offline diagnostic. It is
  disabled by default in the `rvlab` feature wrapper, so adding future prices
  cannot revise already completed default variance blocks.
- The synthetic European hedge ledger accepts externally supplied causal delta
  arrays, with flat rates, continuous dividend yield and stock trading costs.
  It is not an American assignment engine or historical SPY execution replay.

## Limits

The Gaussian driver reference computes the *Riemann–Liouville* covariance,
not stationary fractional Brownian covariance. Its Cholesky construction uses
numerical covariance integrals and a small diagonal regularizer. Midpoint and
far-field approximations retain time-discretization error. In particular the
variance exponential compensator uses the continuum `t**(2H)` while a midpoint
driver has discretized variance; this scheme is a biased comparator, not a
validated high-accuracy pricing backend. No American rough-Bergomi reduction,
calibration, delta or historical factor-state readiness is established.

Regular sampling, an appropriate local exchange timezone and session filtering
remain caller responsibilities for empirical roughness. Short-sample H estimates
are descriptive and sensitive to proxy noise and lag choices. No previous
benchmark, cross-sectional finding or old notebook output was relabelled as a
new result by the import.

## Provenance: original working-tree SHA256

| Original relative path | Bytes | SHA256 |
|---|---:|---|
| `src/roughvol/types.py` | 6518 | `79fb9d02bf5bfcbe429a1f8ff290f24b212e29463742bdd89494d6372dd1272f` |
| `src/roughvol/models/rough_bergomi_model.py` | 12115 | `e0c6d3d8cc7e08d1d4b7fcbeebffd3cc3d90d0bb141489c01dcfbfab9750cdf2` |
| `src/roughvol/sim/brownian.py` | 3586 | `4efd2b5ad9b26ddc1889b7c7df392d53cebb9acbc08a2e8dcdaf98013c1b91d9` |
| `src/roughvol/sim/volterra.py` | 9359 | `8ddd40a58c4c6ea2b38891c61f7a51b5bd9bd2836a9ff01df0fe01e6a7e7de59` |
| `src/roughvol/kernels/rough_bergomi.py` | 970 | `629ebc16bc5564317f6554ef29d344994aee591b839e057cdc788fe9201c2192` |
| `src/roughvol/kernels/rough_bergomi_exact.py` | 3714 | `f748daf46993fb6a302ad01eb26a69d6b80dbfe800b8f53b92b4776413d891e4` |
| `src/roughvol/analytics/roughness.py` | 14124 | `d4ea15160aaaf5f9f7553a719cf3b14123e044834b9857969cc0aa9a08753ada` |
| `src/roughvol/experiments/model_comparison/hedge_backtest.py` | 18068 | `42f693a19c7c338f0953827dc7cdadcc1cdb3d807629402cbe0aa3d452b65e40` |
| `tests/test_roughness.py` | 5519 | `d5d82c09e8dcffe4466572e0e8e747d1498ffe7b2aa0d2574f2394946be33e42` |
| `tests/test_hedge_backtest.py` | 8225 | `bb0a35ddd1fbc54410b3dbbc5dc02d6b8879e53c2664d33447040716e4e5fa3c` |


Validation command from the repository root:

```bash
python -m pytest tests/roughvol -q
```

The tests cover analytic zero-vol-of-vol BS limits using independent antithetic
pair means, near-field conditional covariance, Gaussian-driver variance and
cross-covariance, pairing, invalid grids, variance aliases, causal/default
variance blocks, transformations/session boundaries and European ledger
accounting. They do not certify a generic derivatives or trading engine.
