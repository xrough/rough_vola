"""
rvlab.data.synthetic
====================
The fallback that keeps every notebook runnable.

Two generators, in increasing order of realism:

  `synthetic_smile_panel()` — a latent rough-style smile manifold where
      rr25 = alpha_t * T^(H-0.5) * atm_iv_t
      bf25 = gamma_t * T^(2H-1) * atm_total_var_t
  is *known by construction*. Because ground truth is known, this is also the
  right dataset for asking "would our estimator recover H if H were real?".

  `rbergomi_panel()` — actual rough Bergomi Monte-Carlo paths via `roughvol`,
  used where a real stochastic model matters more than a known smile manifold.

Both return the same columns as the real SPY caches, so a notebook never
branches on which one it got.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import compat, config


def synthetic_smile_panel(
    n_bars: int = 400,
    bar_minutes: int = 5,
    hurst: float = config.H_PRIOR,
    expiries_dte: tuple[int, ...] = (7, 14, 21, 35, 49),
    seed: int = config.DEFAULT_SEED,
) -> pd.DataFrame:
    """A smile panel with a rough structure planted in it.

    Prefers the project's own generator in demo/python/research/shared, which
    round-trips through the same IV solver used on the real OPRA data. Falls
    back to a self-contained construction when that tree is unavailable.

    Returns the standard columns: ts, expiry, T, atm_iv, atm_total_var,
    rr25, bf25, alpha, gamma.
    """
    if compat.ensure_research_shared():
        try:
            return _panel_via_research_shared(n_bars, bar_minutes, hurst, expiries_dte, seed)
        except Exception:                       # generator is optional, never fatal
            pass
    return _panel_native(n_bars, bar_minutes, hurst, expiries_dte, seed)


def _quiet(fn, *args, **kwargs):
    """Run a third-party generator without its DeprecationWarning noise."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return fn(*args, **kwargs)


def _panel_via_research_shared(n_bars, bar_minutes, hurst, expiries_dte, seed) -> pd.DataFrame:
    """Use shared/synthetic_smile.py so synthetic data traverses the real solver."""
    ss = compat.research_module("synthetic_smile")
    cfg = ss.SyntheticRoughConfig(
        seed=seed, H=hurst, n_bars=n_bars, bar_minutes=bar_minutes,
        expiries_dte=tuple(expiries_dte),
    )
    # returns (panel_df, state_df, records) — we want the extracted records
    _panel, _state, records = _quiet(ss.generate_rough_synthetic_records, cfg)
    df = pd.DataFrame(records)
    df.attrs["generator"] = "research_shared.synthetic_smile"
    df.attrs["true_hurst"] = hurst
    return df


def _panel_native(n_bars, bar_minutes, hurst, expiries_dte, seed) -> pd.DataFrame:
    """Self-contained generator — no dependency beyond numpy/pandas.

    Latent state (atm_iv, alpha, gamma) follows mean-reverting dynamics with
    return-driven vol-of-vol; the smile observables are then read off the rough
    scaling laws and given observation noise.
    """
    rng = np.random.default_rng(seed)
    dt_min = bar_minutes

    # ── latent state paths ────────────────────────────────────────────────────
    atm_iv = np.empty(n_bars)
    alpha = np.empty(n_bars)
    gamma = np.empty(n_bars)
    spot = np.empty(n_bars)

    atm_iv[0], alpha[0], gamma[0], spot[0] = 0.18, -0.17, 0.55, 580.0
    rets = rng.normal(0.0, 0.0012, n_bars)
    shocks = rng.random(n_bars) < 0.05
    rets[shocks] += rng.normal(0.0, 0.006, shocks.sum())

    for i in range(1, n_bars):
        spot[i] = spot[i - 1] * np.exp(rets[i])
        atm_iv[i] = np.clip(
            atm_iv[i - 1] + 0.12 * (0.18 - atm_iv[i - 1]) + 0.70 * abs(rets[i])
            + rng.normal(0.0, 0.0015), 0.05, 1.2)
        alpha[i] = alpha[i - 1] + 0.008 * (-0.17 - alpha[i - 1]) + rng.normal(0.0, 0.0010)
        gamma[i] = gamma[i - 1] + 0.008 * (0.55 - gamma[i - 1]) + rng.normal(0.0, 0.0040)

    # ── observation grid ──────────────────────────────────────────────────────
    start = pd.Timestamp("2026-01-05 13:35", tz="UTC")
    ts = pd.date_range(start, periods=n_bars, freq=f"{dt_min}min")

    rows = []
    for i in range(n_bars):
        for dte in expiries_dte:
            T = dte / 365.0
            aiv = atm_iv[i]
            atv = aiv ** 2 * T
            rows.append({
                "ts": ts[i],
                "expiry": (ts[i] + pd.Timedelta(days=int(dte))).date(),
                "T": T,
                "forward": spot[i],
                "atm_iv": aiv,
                "atm_total_var": atv,
                # the rough scaling laws, plus observation noise
                "rr25": alpha[i] * T ** (hurst - 0.5) * aiv + rng.normal(0.0, 0.0008),
                "bf25": gamma[i] * T ** (2 * hurst - 1) * atv + rng.normal(0.0, 0.0003),
            })

    df = pd.DataFrame(rows)
    # Re-derive the structural coefficients the same way the real pipeline does,
    # so alpha/gamma here are estimates, not the latent truth.
    df["alpha"] = df["rr25"] / (df["T"] ** (hurst - 0.5) * df["atm_iv"])
    df["gamma"] = df["bf25"] / (df["T"] ** (2 * hurst - 1) * df["atm_total_var"])
    df.attrs["generator"] = "rvlab.native"
    df.attrs["true_hurst"] = hurst
    return df


def rbergomi_panel(
    n_paths: int = 2_000,
    maturity: float = 0.25,
    n_steps: int = 100,
    hurst: float = 0.10,
    eta: float = 1.5,
    rho: float = -0.7,
    xi0: float = 0.0625,
    spot: float = 100.0,
    seed: int = config.DEFAULT_SEED,
    scheme: str = "blp-hybrid",
) -> pd.DataFrame:
    """Simulate rough Bergomi paths with `roughvol`; return a tidy long frame.

    Columns: path, step, t, spot, variance. Raises ImportError when roughvol is
    unavailable — callers that need a fallback should use `fbm_variance_paths`.
    """
    rvt = compat.roughvol_module("roughvol.types")
    rbm = compat.roughvol_module("roughvol.models.rough_bergomi_model")

    model = rbm.RoughBergomiModel(hurst=hurst, eta=eta, rho=rho, xi0=xi0, scheme=scheme)
    market = rvt.MarketData(spot=spot, rate=0.0, div_yield=0.0)
    sim = rvt.SimConfig(n_paths=n_paths, maturity=maturity, n_steps=n_steps,
                        seed=seed, store_paths=True)
    bundle = model.simulate_paths(market=market, sim=sim, rng=np.random.default_rng(seed))

    t = np.asarray(bundle.t)
    s = np.asarray(bundle.spot)
    v = np.asarray(bundle.state.get("variance", np.full_like(s, np.nan)))
    n_p, n_t = s.shape

    df = pd.DataFrame({
        "path": np.repeat(np.arange(n_p), n_t),
        "step": np.tile(np.arange(n_t), n_p),
        "t": np.tile(t, n_p),
        "spot": s.ravel(),
        "variance": v.ravel(),
    })
    df.attrs["generator"] = f"roughvol.RoughBergomiModel[{scheme}]"
    df.attrs["true_hurst"] = hurst
    return df


def fractional_brownian_motion(
    n_paths: int = 200,
    n_steps: int = 512,
    horizon: float = 1.0,
    hurst: float = 0.10,
    seed: int = config.DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Riemann-Liouville fractional Brownian motion. Returns (t, W_H).

    W_H(t) = sqrt(2H) * integral_0^t (t - s)^(H - 1/2) dW(s)

    Discretised so that each cell contributes its **exact** variance:

        G[i, j]^2 * dt = integral over cell j of 2H (t_i - s)^(2H - 1) ds
                       = (t_i - t_{j-1})^(2H) - (t_i - t_j)^(2H)

    which telescopes to Var(W_H(t_i)) = t_i^(2H) exactly, for every H and every
    step size. A midpoint or left-endpoint rule does not: the kernel is singular
    at u = 0 whenever H < 1/2 — precisely the case rough volatility is about —
    and those rules under-integrate the near-diagonal cell badly enough that a
    structure-function fit on the output returns roughly 0.5 no matter what H
    went in.

    Implemented as one lower-triangular weight matrix applied to all paths at
    once: the kernel does not depend on the path, so it is built once.

    Note that Riemann-Liouville fBM has *non-stationary* increments, so a
    structure-function estimate of H on the output is mildly biased downward
    even here. That is a property of the process and the estimator, not a bug —
    see [04/R4.5] where the size of the bias is measured.
    """
    dt = horizon / n_steps
    t = np.linspace(0.0, horizon, n_steps + 1)

    i = np.arange(1, n_steps + 1)[:, None]
    j = np.arange(1, n_steps + 1)[None, :]
    upper = np.maximum((i - j + 1) * dt, 0.0)        # t_i - t_{j-1}
    lower = np.maximum((i - j) * dt, 0.0)            # t_i - t_j
    G2 = np.where(i >= j, (upper ** (2 * hurst) - lower ** (2 * hurst)) / dt, 0.0)
    G = np.sqrt(np.clip(G2, 0.0, None))

    rng = np.random.default_rng(seed)
    dW = rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
    W = np.hstack([np.zeros((n_paths, 1)), dW @ G.T])
    return t, W


def fbm_variance_paths(
    n_paths: int = 200,
    n_steps: int = 512,
    horizon: float = 1.0,
    hurst: float = 0.10,
    initial_vol: float = 0.20,
    vol_of_vol: float = 1.5,
    seed: int = config.DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Log-normal rough volatility paths. Returns (t, vol) shaped (n_paths, n_steps+1).

    The rough Bergomi variance form,

        sigma(t) = sigma_0 * exp(nu * W_H(t) - 0.5 * nu^2 * t^(2H))

    where W_H is Riemann-Liouville fBM and the drift term keeps E[sigma(t)^2]
    level rather than letting it grow with the exponential's own convexity.

    Because log sigma is nu * W_H plus a deterministic drift, applying
    `rvlab.features.hurst_structure_function` to a path recovers `hurst` — which
    is what makes this the right dataset for validating that estimator before
    pointing it at a market.
    """
    t, W = fractional_brownian_motion(n_paths, n_steps, horizon, hurst, seed)
    vol = initial_vol * np.exp(vol_of_vol * W - 0.5 * vol_of_vol**2 * t ** (2 * hurst))
    return t, vol


__all__ = ["synthetic_smile_panel", "rbergomi_panel", "fbm_variance_paths",
           "fractional_brownian_motion"]
