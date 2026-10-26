from __future__ import annotations

from dataclasses import dataclass, field 

from typing import Protocol, runtime_checkable, Mapping, Any, Callable
import numpy as np

ArrayF = np.ndarray # 简写ndarray

# ============================================================================
# Input containers
# ============================================================================

@dataclass(frozen=True)
class SimConfig:
    '''
    Numerical/simulation configuration. 
    Input includes n_paths, maturity, n_steps/time_grid, seed, antithetic, scheme, store_paths and metadata.'''
    n_paths: int
    maturity: float

    n_steps: int | None = None
    time_grid: ArrayF | None = None

    seed: int | None = None
    
    # Extensions
    
    antithetic: bool = False
    scheme: str = "euler"  # e.g. "euler", "qe", "hybrid"
    store_paths: bool = True  # allow terminal-only mode

    metadata: Mapping[str, Any] = field(default_factory=dict) # dict储存额外信息。

    def grid(self) -> ArrayF:
        if self.time_grid is not None:
            t = np.asarray(self.time_grid, dtype=float)
            if t.ndim != 1:
                raise ValueError("time_grid must be 1D")
            return t

        if self.n_steps is None:
            raise ValueError("Provide either time_grid or n_steps")
        if self.maturity == 0.0:
            return np.array([0.0], dtype=float)

        return np.linspace(0.0, self.maturity, self.n_steps + 1)

@dataclass(frozen=True)
class MarketData:
    '''Economic environment.'''
    spot: float
    rate: float = 0.0
    div_yield: float = 0.0

    # general models
    # callable是确定对象是否是一个函数，而Callable是static typing，在此处定义一些method为Callable。
    discount_curve: Any | None = None
    forward_variance_curve: Callable[[ArrayF], ArrayF] | None = None # default to be None. 
    metadata: Mapping[str, Any] = field(default_factory=dict) # 我们允许通过dict记录一些额外的信息。


# ============================================================================
# Model output container
# ============================================================================

@dataclass(frozen=True)
class PathBundle:
    '''
    Standardized model output container.
    - spot shape: (n_paths, n_times), economic state variable along paths (spot S_t, volatility v_t etc.).
      - therefore should align with time grid t.
    - t: time shape: (n_times,)
    - variance shape (if present): (n_paths, n_times)
    '''
    t: ArrayF
    state: Mapping[str, ArrayF]
    
    '''
    This object may optionally contain a dictionary of extra arrays, indexed by string names.
    '''
    
    extras: Mapping[str, ArrayF] = field(default_factory=dict) # numerical arrays produced by the simulation (path-level data).
    metadata: Mapping[str, Any] = field(default_factory=dict) # descriptive context about how/why the simulation was produced.
    
    def __post_init__(self) -> None: # __post_init__ runs after dataclass is initialized.
        # Validate time grid and state arrays alignment
        if "spot" not in self.state:
            raise ValueError("PathBundle.state must contain 'spot'")

        spot = np.asarray(self.state["spot"])
        t = np.asarray(self.t)

        if spot.ndim != 2:
            raise ValueError("state['spot'] must be 2D (n_paths, n_times)")
        if spot.shape[1] != t.shape[0]:
            raise ValueError("spot arrays must align with time grid")

        for k, v in self.state.items(): # items() returns key-value pairs.
            arr = np.asarray(v)
            if arr.shape != spot.shape:
                raise ValueError(f"state['{k}'] shape mismatch")

    # properties are snapshots of the object state, read-only attributes computed on-the-fly.
    @property
    def spot(self) -> ArrayF:
        return self.state["spot"]

    @property
    def spot_T(self) -> ArrayF:
        return self.spot[:, -1]

    @property
    def n_paths(self) -> int:
        return int(self.spot.shape[0])

    @property
    def n_times(self) -> int:
        return int(self.spot.shape[1])
    
    # for extension: get state by name. 
    def get(self, name: str) -> ArrayF:
        return self.state[name]
    

# ============================================================================
# Pricing output
# ============================================================================

@dataclass(frozen=True)
class PriceResult:
    '''Standard output container for pricing engines.'''
    price: float
    stderr: float # MC standard error
    ci95: tuple # MC 95% confidence interval
    n_paths: int
    n_steps: int
    seed: int | None = None
    
    metadata: Mapping[str, Any] = field(default_factory=dict)

# ============================================================================
# Protocols (capability boundaries)
# ============================================================================

@runtime_checkable
class Instrument(Protocol): # Protocol规定class至少要满足的条件。
    '''Contract: an instrument defines payoff, may depend on the full path.'''

    maturity: float  #结算时间

    def payoff(self, paths: PathBundle) -> ArrayF:
        ...

@runtime_checkable
class PathModel(Protocol):
    '''
    Model that can simulate paths.
    '''
    def simulate_paths(
        self,
        *,
        market: MarketData,
        sim: SimConfig,
        rng: np.random.Generator,
    ) -> PathBundle:
        ...

# ============================================================================
# Aux functions
# ============================================================================ 

def compute_payoff(instrument: Instrument, paths: PathBundle) -> ArrayF:
    payoff_paths = getattr(instrument, "payoff", None)
    if not callable(payoff_paths):
        raise TypeError("Instrument must implement payoff(paths: PathBundle).")

    out = np.asarray(payoff_paths(paths))
    if out.ndim != 1 or out.shape[0] != paths.n_paths:
        raise ValueError(
            f"Instrument.payoff must return shape (n_paths,), got {out.shape} "
            f"for n_paths={paths.n_paths}."
        )
    return out


def make_rng(seed: int | None) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    # Use SeedSequence for robust, reproducible streams
    return np.random.default_rng(np.random.SeedSequence(seed))


def flat_discount_factor(rate: float, t: float) -> float:
    return float(np.exp(-rate * t))