from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roughvol.sim.brownian import correlated_brownian_increments
from roughvol.sim.volterra import simulate_blp, simulate_exact, simulate_midpoint
from roughvol.types import MarketData, PathBundle, SimConfig

_VALID_SCHEMES = ("volterra-midpoint", "exact-gaussian", "blp-hybrid")

@dataclass(frozen=True)
class RoughBergomiModel:
    """Rough Bergomi model with selectable simulation scheme.

    Parameters
    ----------
    hurst  : roughness parameter H in (0, 0.5)
    eta    : volatility-of-volatility
    rho    : correlation between the variance driver and spot driver
    xi0    : flat initial forward variance level, used when market.forward_variance_curve is absent
    scheme : simulation scheme; one of:
        - ``"volterra-midpoint"``  (default) — O(n²) midpoint Volterra quadrature (§2.5)
        - ``"exact-gaussian"``    — O(n³) precomp Cholesky of joint fBM covariance (§2.1 benchmark)
        - ``"blp-hybrid"``        — O(n log n) Bennedsen-Lunde-Pakkanen hybrid scheme (§2.3)
    blp_kappa : near-field cutoff for the BLP scheme (default 10)
    """

    hurst: float
    eta: float
    rho: float
    xi0: float
    scheme: str = "volterra-midpoint"
    blp_kappa: int = 10

    def simulate_paths(
        self,
        *,
        market: MarketData,
        sim: SimConfig,
        rng: np.random.Generator,
    ) -> PathBundle:
        scheme = str(self.scheme)
        if scheme not in _VALID_SCHEMES:
            raise ValueError(f"Unknown scheme {scheme!r}. Valid: {_VALID_SCHEMES}")

        H = float(self.hurst)
        eta = float(self.eta)
        rho = float(self.rho)
        xi0 = float(self.xi0)

        if not np.isfinite([H, eta, rho, xi0, market.spot, market.rate, market.div_yield]).all():
            raise ValueError("Model and market inputs must be finite")

        if not (0.0 < H < 0.5):
            raise ValueError("hurst must lie in (0, 0.5).")
        if eta < 0.0:
            raise ValueError("eta must be non-negative.")
        if xi0 < 0.0:
            raise ValueError("xi0 must be non-negative.")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError("rho must be in [-1, 1].")

        S0 = float(market.spot)
        r = float(market.rate)
        q = float(market.div_yield)
        if S0 <= 0.0:
            raise ValueError("market.spot must be positive.")

        t = np.asarray(sim.grid(), dtype=float)
        if t.ndim != 1 or not np.isfinite(t).all() or not len(t) or t[0] != 0.:
            raise ValueError("Time grid must be finite, one-dimensional and start at zero")
        if not np.isfinite(sim.maturity) or sim.maturity < 0 or abs(t[-1]-sim.maturity) > 1e-12:
            raise ValueError("Time grid must end at the nonnegative stated maturity")
        n_times = t.size
        if n_times < 1:
            raise ValueError("Empty time grid.")

        n_paths = int(sim.n_paths)
        if n_paths <= 0:
            raise ValueError("sim.n_paths must be positive.")

        antithetic = bool(sim.antithetic)
        if antithetic and (n_paths % 2 != 0):
            raise ValueError("n_paths must be even when antithetic=True.")

        if n_times == 1:
            S = np.full((n_paths, 1), S0, dtype=float)
            xi_curve = _forward_variance_curve(t, market, xi0)
            return PathBundle(
                t=t,
                state={
                    "spot": S,
                    "var": np.broadcast_to(xi_curve, (n_paths, 1)).copy(),
                    "variance": np.broadcast_to(xi_curve, (n_paths, 1)).copy(),
                    "Y": np.zeros((n_paths, 1), dtype=float),
                },
                metadata={"model": "RoughBergomi", "scheme": scheme},
            )

        dt = np.diff(t)
        if np.any(dt <= 0.0):
            raise ValueError("Time grid must be strictly increasing.")

        n_steps = n_times - 1
        xi_curve = _forward_variance_curve(t, market, xi0)

        # ------------------------------------------------------------------
        # Dispatch to selected scheme
        # ------------------------------------------------------------------
        if scheme == "volterra-midpoint":
            return self._simulate_volterra_midpoint(
                t=t, dt=dt, n_steps=n_steps, n_paths=n_paths,
                antithetic=antithetic, H=H, eta=eta, rho=rho, xi0=xi0,
                S0=S0, r=r, q=q, xi_curve=xi_curve, rng=rng,
            )
        elif scheme == "exact-gaussian":
            return self._simulate_exact_gaussian(
                t=t, dt=dt, n_steps=n_steps, n_paths=n_paths,
                antithetic=antithetic, H=H, eta=eta, rho=rho, xi0=xi0,
                S0=S0, r=r, q=q, xi_curve=xi_curve, rng=rng,
            )
        else:  # blp-hybrid
            return self._simulate_blp_hybrid(
                t=t, dt=dt, n_steps=n_steps, n_paths=n_paths,
                antithetic=antithetic, H=H, eta=eta, rho=rho, xi0=xi0,
                S0=S0, r=r, q=q, xi_curve=xi_curve, rng=rng,
            )

    # ------------------------------------------------------------------
    # Private scheme implementations
    # ------------------------------------------------------------------

    def _simulate_volterra_midpoint(
        self, *, t, dt, n_steps, n_paths, antithetic, H, eta, rho, xi0,
        S0, r, q, xi_curve, rng,
    ) -> PathBundle:
        n_times = n_steps + 1
        dW_y, dW_s = correlated_brownian_increments(
            n_paths=n_paths, n_steps=n_steps, dt=1.0,
            rho=rho, rng=rng, antithetic=antithetic,
        )
        dW_y *= np.sqrt(dt)[None, :]
        dW_s *= np.sqrt(dt)[None, :]

        Y = simulate_midpoint(dW_y, t, H)

        var, S = _var_and_spot(Y, xi_curve, t, dt, dW_s, H, eta, S0, r, q, n_steps, n_paths)

        return PathBundle(
            t=t,
            state={"spot": S, "var": var, "variance": var, "Y": Y},
            extras={
                "dW_y": dW_y,
                "dW_s": dW_s,
                "forward_variance_curve": np.broadcast_to(xi_curve, (n_paths, n_times)),
            },
            metadata={
                "model": "RoughBergomi", "scheme": "volterra-midpoint",
                "hurst": H, "eta": eta, "rho": rho, "xi0": xi0, "antithetic": antithetic,
            },
        )

    def _simulate_exact_gaussian(
        self, *, t, dt, n_steps, n_paths, antithetic, H, eta, rho, xi0,
        S0, r, q, xi_curve, rng,
    ) -> PathBundle:
        """Exact Gaussian simulation via Cholesky of joint (Ytilde, W) covariance (§2.1)."""
        n_times = n_steps + 1

        Y, dW_y = simulate_exact(t, H, n_paths, antithetic=antithetic, rng=rng)

        # Orthogonal BM increments for spot; combine with variance-driver BM for correlation
        base = rng.standard_normal((n_paths // 2 if antithetic else n_paths, n_steps))
        dWperp = (np.concatenate([base, -base], axis=0) if antithetic else base) * np.sqrt(dt)[None, :]
        dW_s = rho * dW_y + np.sqrt(1.0 - rho ** 2) * dWperp

        var, S = _var_and_spot(Y, xi_curve, t, dt, dW_s, H, eta, S0, r, q, n_steps, n_paths)

        return PathBundle(
            t=t,
            state={"spot": S, "var": var, "variance": var, "Y": Y},
            extras={
                "dW_y": dW_y,
                "dW_s": dW_s,
                "forward_variance_curve": np.broadcast_to(xi_curve, (n_paths, n_times)),
            },
            metadata={
                "model": "RoughBergomi", "scheme": "exact-gaussian",
                "hurst": H, "eta": eta, "rho": rho, "xi0": xi0, "antithetic": antithetic,
            },
        )

    def _simulate_blp_hybrid(
        self, *, t, dt, n_steps, n_paths, antithetic, H, eta, rho, xi0,
        S0, r, q, xi_curve, rng,
    ) -> PathBundle:
        """BLP hybrid scheme (§2.3): near-field exact + far-field FFT convolution."""
        n_times = n_steps + 1
        kappa = int(self.blp_kappa)

        # Generate correlated raw BM increments (variance driver dW_y, spot driver dW_s)
        dW_y, dW_s = correlated_brownian_increments(
            n_paths=n_paths, n_steps=n_steps, dt=1.0,
            rho=rho, rng=rng, antithetic=antithetic,
        )
        dW_y *= np.sqrt(dt)[None, :]
        dW_s *= np.sqrt(dt)[None, :]

        # BLP scheme returns Y of shape (n_paths, n_times)
        Y = simulate_blp(dW_y, t, H, kappa=kappa, rng=rng, antithetic=antithetic)

        var, S = _var_and_spot(Y, xi_curve, t, dt, dW_s, H, eta, S0, r, q, n_steps, n_paths)

        return PathBundle(
            t=t,
            state={"spot": S, "var": var, "variance": var, "Y": Y},
            extras={
                "dW_y": dW_y,
                "dW_s": dW_s,
                "forward_variance_curve": np.broadcast_to(xi_curve, (n_paths, n_times)),
            },
            metadata={
                "model": "RoughBergomi", "scheme": "blp-hybrid",
                "hurst": H, "eta": eta, "rho": rho, "xi0": xi0,
                "antithetic": antithetic, "blp_kappa": kappa,
            },
        )


def _var_and_spot(
    Y: np.ndarray,
    xi_curve: np.ndarray,
    t: np.ndarray,
    dt: np.ndarray,
    dW_s: np.ndarray,
    H: float,
    eta: float,
    S0: float,
    r: float,
    q: float,
    n_steps: int,
    n_paths: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute variance and spot paths given the Volterra driver Y.

    The published implementation uses one explicit NumPy backend.
    """
    # NumPy fallback
    n_times = n_steps + 1
    var = np.empty((n_paths, n_times), dtype=float)
    var[:, 0] = xi_curve[0]
    variance_correction = np.power(t[1:], 2.0 * H)
    var[:, 1:] = xi_curve[1:][None, :] * np.exp(
        eta * Y[:, 1:] - 0.5 * (eta ** 2) * variance_correction[None, :]
    )

    S = np.empty((n_paths, n_times), dtype=float)
    S[:, 0] = S0
    for j in range(n_steps):
        v = np.maximum(var[:, j], 0.0)
        drift = (r - q - 0.5 * v) * dt[j]
        diffusion = np.sqrt(v) * dW_s[:, j]
        S[:, j + 1] = S[:, j] * np.exp(drift + diffusion)

    return var, S


def _forward_variance_curve(t: np.ndarray, market: MarketData, xi0: float) -> np.ndarray:
    if market.forward_variance_curve is None:
        return np.full_like(t, xi0, dtype=float)

    xi_curve = np.asarray(market.forward_variance_curve(t), dtype=float)
    if xi_curve.shape != t.shape:
        raise ValueError("forward_variance_curve must return an array aligned with t.")
    if not np.isfinite(xi_curve).all() or np.any(xi_curve < 0.0):
        raise ValueError("forward_variance_curve must be finite and non-negative.")
    return xi_curve
