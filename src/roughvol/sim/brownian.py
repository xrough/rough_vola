from __future__ import annotations

import numpy as np
from roughvol.types import ArrayF

def time_grid(*, maturity: float, n_steps: int) -> ArrayF:
    '''
    Uniform grid from 0 to maturity (inclusive). Python中*之后的所有输入必须通过名称而非位置. 
    '''
    if maturity < 0:
        raise ValueError("maturity must be non-negative.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")

    # Allow maturity == 0: grid is just [0.0]
    if maturity == 0.0:
        return np.array([0.0], dtype=float)

    return np.linspace(0.0, float(maturity), int(n_steps) + 1)


def brownian_increments(
    *,
    n_paths: int,
    n_steps: int,
    dt: float,
    rng: np.random.Generator,
    antithetic: bool = False,
) -> ArrayF:
    '''
    Generate Brownian increments dW ~ Normal(0, dt).

    If antithetic=True, uses half paths and mirrors with -dW.
    Requires n_paths even in that case.

    Returns
    -------
    np.ndarray
        Shape (n_paths, n_steps). Column j is the increment over step j.
    '''
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if dt < 0:
        raise ValueError("dt must be non-negative.")
    if antithetic and (n_paths % 2 != 0):
        raise ValueError("n_paths must be even when antithetic=True.")

    # dt == 0 => all increments are 0, avoid RNG calls (nice for determinism)
    if dt == 0.0:
        return np.zeros((int(n_paths), int(n_steps)), dtype=float)

    if not antithetic:
        z = rng.standard_normal(size=(int(n_paths), int(n_steps)))
        return np.sqrt(float(dt)) * z

    half = int(n_paths) // 2
    z_half = rng.standard_normal(size=(half, int(n_steps)))
    dW_half = np.sqrt(float(dt)) * z_half
    return np.vstack([dW_half, -dW_half])


def correlated_brownian_increments(
    *,
    n_paths: int,
    n_steps: int,
    dt: float,
    rho: float,
    rng: np.random.Generator,
    antithetic: bool = False,
) -> tuple[ArrayF, ArrayF]:
    '''Generate correlated Brownian increments (dW1, dW2) with corr = rho.

    Returns
    -------
    (dW1, dW2): each shape (n_paths, n_steps), each ~ N(0, dt)
    '''
    if dt < 0:
        raise ValueError("dt must be non-negative.")
    z1, z2 = correlated_standard_normals(
        n_paths=n_paths,
        n_steps=n_steps,
        rho=rho,
        rng=rng,
        antithetic=antithetic,
    )
    sdt = np.sqrt(float(dt))
    return sdt * z1, sdt * z2


def correlated_standard_normals(
    *,
    n_paths: int,
    n_steps: int,
    rho: float,
    rng: np.random.Generator,
    antithetic: bool = False,
) -> tuple[ArrayF, ArrayF]:
    '''
    Generate (Z1, Z2) with corr(Z1, Z2)=rho, each ~ N(0,1).

    Returns
    -------
    (Z1, Z2): each shape (n_paths, n_steps)
    '''
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("rho must be in [-1, 1].")
    if antithetic and (n_paths % 2 != 0):
        raise ValueError("n_paths must be even when antithetic=True.")

    if antithetic:
        half = n_paths // 2
        z1_half = rng.standard_normal(size=(half, n_steps))
        z2_half = rng.standard_normal(size=(half, n_steps))
        z1 = np.vstack([z1_half, -z1_half])
        z2 = np.vstack([z2_half, -z2_half])
    else:
        z1 = rng.standard_normal(size=(n_paths, n_steps))
        z2 = rng.standard_normal(size=(n_paths, n_steps))

    # correlate: Z2_corr = rho*Z1 + sqrt(1-rho^2)*Z2
    s = np.sqrt(max(0.0, 1.0 - float(rho) ** 2))
    z2_corr = float(rho) * z1 + s * z2
    return z1, z2_corr
