"""Fractional Brownian motion (Volterra driver) simulation for rough volatility models.

Mirrors sim/brownian.py for standard Brownian motion: this module is concerned
only with simulating the Volterra process Y (the fBM driver), independent of
model parameters (eta, rho, xi0).

Three schemes are provided:

  simulate_midpoint  — O(n²) midpoint quadrature (fast, biased for small n)
  simulate_exact     — O(n³) exact Cholesky of the joint fBM covariance
  simulate_blp       — O(n log n) Bennedsen-Lunde-Pakkanen hybrid scheme

All three return Y of shape (n_paths, n_times) with Y[:, 0] = 0.
"""
from __future__ import annotations

from functools import lru_cache
from numbers import Integral

import numpy as np
from scipy.integrate import quad

from roughvol.kernels.rough_bergomi import rough_bergomi_midpoint_weights
from roughvol.kernels.rough_bergomi_exact import rough_bergomi_exact_cholesky
from roughvol.types import ArrayF


# ---------------------------------------------------------------------------
# Scheme 1: volterra-midpoint  O(n²)
# ---------------------------------------------------------------------------

def simulate_midpoint(
    dW_y: ArrayF,
    t: ArrayF,
    hurst: float,
) -> ArrayF:
    """Simulate the Volterra driver Y using midpoint quadrature.

    Y_{t_i} = sqrt(2H) * sum_{j=0}^{i-1} (t_i - mid_j)^{H-0.5} * dW_y_j

    where mid_j is the midpoint of [t_j, t_{j+1}].

    Parameters
    ----------
    dW_y  : (n_paths, n_steps) Brownian increments for the variance driver
    t     : 1D time grid of shape (n_steps + 1,), t[0] = 0
    hurst : Hurst exponent H in (0, 0.5)

    Returns
    -------
    Y : (n_paths, n_steps + 1) Volterra driver paths, Y[:, 0] = 0
    """
    dW_y = np.asarray(dW_y, dtype=float)
    t = np.asarray(t, dtype=float)
    n_paths, n_steps = dW_y.shape

    weights = rough_bergomi_midpoint_weights(t, hurst)  # (n_steps, n_steps) upper-triangular
    Y = np.zeros((n_paths, n_steps + 1), dtype=float)
    Y[:, 1:] = dW_y @ weights.T
    return Y


# ---------------------------------------------------------------------------
# Scheme 2: exact-gaussian  O(n³ precompute, O(n²) per sample)
# ---------------------------------------------------------------------------

def simulate_exact(
    t: ArrayF,
    hurst: float,
    n_paths: int,
    *,
    antithetic: bool = False,
    rng: np.random.Generator,
) -> tuple[ArrayF, ArrayF]:
    """Simulate the Volterra driver Y exactly via Cholesky of the joint fBM covariance.

    Samples (Ytilde, W) jointly from their true Gaussian distribution.  W is
    the standard Brownian motion that also drives the Volterra integral, so its
    increments dW_y can be recovered directly from the simulation without any
    additional approximation.

    Parameters
    ----------
    t         : 1D time grid of shape (n_steps + 1,), t[0] = 0
    hurst     : Hurst exponent H in (0, 0.5)
    n_paths   : number of simulation paths
    antithetic: if True, produce n_paths//2 base paths and mirror them
    rng       : numpy Generator for reproducibility

    Returns
    -------
    Y     : (n_paths, n_steps + 1) Volterra driver, Y[:, 0] = 0
    dW_BM : (n_paths, n_steps) increments of the underlying BM W
    """
    t = np.asarray(t, dtype=float)
    n_steps = t.size - 1
    if isinstance(n_paths, bool) or not isinstance(n_paths, Integral) or n_paths < 1:
        raise ValueError("n_paths must be a positive integer")
    if antithetic and n_paths % 2:
        raise ValueError("antithetic n_paths must be even")

    # O(n³) precomputation: build Cholesky factor of the joint 2n × 2n covariance
    # Block layout: [Ytilde (rows 0:n), W (rows n:2n)]
    L = rough_bergomi_exact_cholesky(t, hurst)  # (2n, 2n)
    n = n_steps

    if antithetic:
        half = n_paths // 2
        Z_half = rng.standard_normal((2 * n, half))
        X_half = L @ Z_half                    # (2n, half)
        Wtilde_pos = X_half[:n, :].T           # (half, n)
        W_pos = X_half[n:, :].T               # (half, n)
        # Antithetic negation: both Ytilde and W are linear in Z
        Ytilde = np.concatenate([Wtilde_pos, -Wtilde_pos], axis=0)  # (n_paths, n)
        W_full = np.concatenate([W_pos, -W_pos], axis=0)            # (n_paths, n)
    else:
        Z = rng.standard_normal((2 * n, n_paths))
        X = L @ Z                    # (2n, n_paths)
        Ytilde = X[:n, :].T          # (n_paths, n)
        W_full = X[n:, :].T          # (n_paths, n)

    # Prepend zero to W_full to get level paths, then difference to get increments
    W_levels = np.concatenate([np.zeros((n_paths, 1)), W_full], axis=1)  # (n_paths, n+1)
    dW_BM = np.diff(W_levels, axis=1)  # (n_paths, n_steps)

    # Prepend zero column to get Y on the full time grid
    Y = np.zeros((n_paths, n_steps + 1), dtype=float)
    Y[:, 1:] = Ytilde

    return Y, dW_BM


# ---------------------------------------------------------------------------
# Scheme 3: blp-hybrid  O(n log n)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _near_conditional_factor(hurst: float, dt: float, kappa: int):
    """Joint kernel integrals on ONE Brownian interval, conditioned on dW.

    For kappa>1 the residuals for different lag kernels are correlated. Sampling
    them independently gives the wrong temporal covariance of the driver.
    """
    H, alpha = hurst, hurst-.5
    lags = np.arange(1, kappa+1, dtype=float)
    covariance = np.empty((kappa, kappa))
    variance = dt**(2*H)*(lags**(2*H)-(lags-1)**(2*H))/(2*H)
    cov_dw = dt**(H+.5)*(lags**(H+.5)-(lags-1)**(H+.5))/(H+.5)
    for i in range(kappa):
        covariance[i, i] = variance[i]
        for j in range(i):
            value = dt**(2*H)*quad(lambda u: (lags[i]-u)**alpha*(lags[j]-u)**alpha,
                0., 1., epsabs=1e-12, epsrel=1e-11)[0]
            covariance[i, j] = covariance[j, i] = value
    conditional = covariance-np.outer(cov_dw, cov_dw)/dt
    conditional = (conditional+conditional.T)/2
    eig, vectors = np.linalg.eigh(conditional)
    tolerance = 1e-10*max(float(np.max(variance)), np.finfo(float).tiny)
    if eig.min() < -tolerance:
        raise FloatingPointError("BLP conditional covariance is not positive semidefinite")
    factor = vectors*np.sqrt(np.maximum(eig, 0.))[None, :]
    factor.setflags(write=False)
    beta = cov_dw/dt
    beta.setflags(write=False)
    return beta, factor


def simulate_blp(dW_y: ArrayF, t: ArrayF, hurst: float, kappa: int = 10,
                 rng: np.random.Generator | None = None, *, antithetic=False) -> ArrayF:
    """Uniform-grid hybrid Volterra driver with jointly sampled near fields.

    Near-field Brownian integrals share the correct conditional covariance for
    all kappa lag kernels. Far-field midpoint weights use FFT convolution.
    Antithetic input must consist of a first half followed by its exact negative;
    the extra Gaussian residuals are mirrored with that same pairing.
    """
    H = float(hurst)
    if not 0 < H < .5:
        raise ValueError("hurst must lie in (0, 0.5)")
    if isinstance(kappa, bool) or not isinstance(kappa, Integral) or kappa < 0:
        raise ValueError("kappa must be a nonnegative integer")
    dW_y, t = np.asarray(dW_y, float), np.asarray(t, float)
    if dW_y.ndim != 2 or not np.isfinite(dW_y).all():
        raise ValueError("dW_y must be a finite path-by-step matrix")
    n_paths, n_steps = dW_y.shape
    if (not n_paths or not n_steps or t.ndim != 1 or t.size != n_steps+1
            or not np.isfinite(t).all() or t[0] != 0.):
        raise ValueError("time grid must be finite, start at zero and align with nonempty increments")
    dt = np.diff(t)
    if np.any(dt <= 0.) or not np.allclose(dt, dt[0], rtol=1e-12, atol=1e-15):
        raise ValueError("BLP hybrid simulation requires a strictly increasing uniform grid")
    if antithetic and (n_paths % 2 or not np.array_equal(dW_y[n_paths//2:], -dW_y[:n_paths//2])):
        raise ValueError("antithetic increments must be paired exact negatives")
    rng = np.random.default_rng() if rng is None else rng
    dt_val, k_near, scale = float(dt[0]), min(int(kappa), n_steps), np.sqrt(2*H)
    Y = np.zeros((n_paths, n_steps+1))
    if k_near < n_steps:
        lags = np.arange(k_near+1, n_steps+1, dtype=float)
        weights = scale*((lags-.5)*dt_val)**(H-.5)
        n_fft = 1 << int(np.ceil(np.log2(2*n_steps)))
        padded = np.zeros(n_fft)
        padded[:len(weights)] = weights
        convolution = np.fft.irfft(np.fft.rfft(dW_y, n=n_fft, axis=1)*np.fft.rfft(padded)[None, :],
                                  n=n_fft, axis=1)
        Y[:, k_near+1:] = convolution[:, :n_steps-k_near]
    if k_near:
        beta, factor = _near_conditional_factor(H, dt_val, k_near)
        bases = n_paths//2 if antithetic else n_paths
        # Small time blocks bound memory independently of total path length.
        for start in range(0, n_steps, 16):
            stop = min(start+16, n_steps)
            residuals = rng.standard_normal((bases, stop-start, k_near)) @ factor.T
            if antithetic:
                residuals = np.concatenate([residuals, -residuals], axis=0)
            for index in range(k_near):
                lag = index+1
                width = min(stop, n_steps-lag+1)-start
                if width > 0:
                    Y[:, start+lag:start+lag+width] += scale*(
                        beta[index]*dW_y[:, start:start+width]+residuals[:, :width, index])
    return Y
