from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from roughvol.types import ArrayF


def _rl_cov_entry(ti: float, tj: float, H: float) -> float:
    """Compute Cov(Wtilde_ti, Wtilde_tj) for the Riemann-Liouville fBM.

    Wtilde_t = sqrt(2H) * integral_0^t (t-s)^(H-1/2) dW_s

    Covariance: 2H * integral_0^{min(ti,tj)} (ti-u)^(H-0.5) * (tj-u)^(H-0.5) du

    Uses the substitution u = s*v to map to [0,1], making it numerically stable.
    """
    if ti == 0.0 or tj == 0.0:
        return 0.0
    s = min(ti, tj)  # upper integration limit
    t = max(ti, tj)
    r = t / s  # >= 1

    # After substitution u = s*v:
    # 2H * s^{2H} * integral_0^1 (r - v)^(H-0.5) * (1 - v)^(H-0.5) dv
    # Singularity at v=1 when H < 0.5 (integrable: exponent H-0.5 in (-0.5, 0))
    def integrand(v: float) -> float:
        a = r - v
        b = 1.0 - v
        if a <= 0.0 or b <= 0.0:
            return 0.0
        return a ** (H - 0.5) * b ** (H - 0.5)

    val, _ = quad(integrand, 0.0, 1.0, limit=200, points=[0.5, 0.9, 0.99])
    return 2.0 * H * (s ** (2.0 * H)) * val


def rough_bergomi_exact_cholesky(t: ArrayF, hurst: float) -> np.ndarray:
    """Build Cholesky factor for exact Gaussian simulation of rough Bergomi (§2.1).

    Constructs the (2n × 2n) Cholesky factor of the joint covariance of
    (Wtilde_{t_1}, ..., Wtilde_{t_n}, W_{t_1}, ..., W_{t_n}), where Wtilde is
    the causal Volterra driver and W is the underlying Brownian motion.

    Covariance blocks:

    Wtilde-Wtilde (RL fBM covariance, computed numerically):
        C_ff[i,j] = 2H * integral_0^{min(ti,tj)} (ti-u)^(H-0.5) * (tj-u)^(H-0.5) du

    W-W (standard BM):
        C_bm[i,j] = min(ti, tj)

    Cross Cov(Wtilde_ti, W_tj) — closed form:
        C_cross[i,j] = sqrt(2H)/(H+0.5) * (ti^(H+0.5) - max(ti-tj, 0)^(H+0.5))

    Note: the Wtilde-Wtilde block uses the Riemann-Liouville covariance (not the
    Mandelbrot-Van Ness fBM formula) to be consistent with the rBergomi model.

    This is a benchmark-quality method. Precomputation is O(n^2) numerical integrals
    plus an O(n^3) Cholesky, so it is not suitable for large n.

    Parameters
    ----------
    t     : 1D time grid including t_0 = 0, shape (n_times,)
    hurst : Hurst parameter H in (0, 0.5)

    Returns
    -------
    L : (2n, 2n) lower-triangular Cholesky factor, where n = n_times - 1
    """
    H = float(hurst)
    if not (0.0 < H < 0.5):
        raise ValueError("hurst must lie in (0, 0.5).")

    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size < 2 or not np.isfinite(t).all():
        raise ValueError("t must be a 1D time grid with at least two points.")
    if t[0] != 0.0 or np.any(np.diff(t) <= 0):
        raise ValueError("Time grid must start at 0 and increase strictly.")

    t_inner = t[1:]  # (n,) — exclude t_0 = 0
    n = len(t_inner)
    alpha = H + 0.5  # H + 1/2

    # --- Wtilde-Wtilde block: RL fBM covariance (numerical) ---
    C_ff = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1):
            c = _rl_cov_entry(t_inner[i], t_inner[j], H)
            C_ff[i, j] = c
            C_ff[j, i] = c

    # --- W-W block: standard BM covariance ---
    ti = t_inner[:, None]  # (n, 1)
    tj = t_inner[None, :]  # (1, n)
    C_bm = np.minimum(ti, tj)

    # --- Cross block: Cov(Wtilde_ti, W_tj) closed form ---
    # = sqrt(2H)/(H+0.5) * (ti^(H+0.5) - max(ti-tj, 0)^(H+0.5))
    scale_cross = np.sqrt(2.0 * H) / alpha
    C_cross = scale_cross * (ti ** alpha - np.maximum(ti - tj, 0.0) ** alpha)

    # --- Assemble 2n × 2n covariance ---
    C = np.block([[C_ff, C_cross], [C_cross.T, C_bm]])

    # Regularise for numerical stability
    C += 1e-10 * np.eye(2 * n)

    L = np.linalg.cholesky(C)
    return L
