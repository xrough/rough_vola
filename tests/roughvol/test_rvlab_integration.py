"""The installed distribution must not depend on an adjacent checkout."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from rvlab import compat
from rvlab.data.synthetic import rbergomi_panel
from rvlab.features.realized import realized_variance_blocks
from rvlab.features.roughness import hurst_structure_function
from roughvol.analytics.roughness import log_returns_from_close


def test_bundled_namespace_never_adds_sibling_paths():
    before = list(sys.path)
    assert compat.ensure_roughvol()
    assert sys.path == before
    assert Path(compat.paths()["roughvol"]).resolve() == Path(__file__).resolve().parents[2]/"src"


def test_root_and_data_configuration_are_explicit(tmp_path):
    env = dict(os.environ, RVLAB_REPO_ROOT=str(tmp_path), RVLAB_DATA_DIR=str(tmp_path/"private_data"))
    code = "from rvlab.config import REPO_ROOT,DATA_DIR,ROUGHVOL_CANDIDATES; print(REPO_ROOT); print(DATA_DIR); assert ROUGHVOL_CANDIDATES == ()"
    output = subprocess.check_output([sys.executable, "-c", code], env=env, text=True).splitlines()
    assert output == [str(tmp_path), str(tmp_path/"private_data")]


def test_rbergomi_panel_has_real_variance_values_from_bundled_source():
    panel = rbergomi_panel(n_paths=4, n_steps=5, maturity=.1, eta=0., seed=42)
    assert panel.variance.notna().all()
    np.testing.assert_array_equal(panel.variance, np.full(len(panel), .0625))


def test_hurst_wrapper_preserves_log_transform_and_session_groups():
    first = pd.date_range("2025-08-07 09:30", periods=20, freq="min", tz="America/New_York")
    second = pd.date_range("2025-08-08 09:30", periods=20, freq="min", tz="America/New_York")
    x = np.tile(np.arange(20)*.01, 2)+np.repeat([0., 4.], 20)
    series = pd.Series(np.exp(x), index=first.append(second))
    logged = hurst_structure_function(series, max_lag=4)
    np.testing.assert_allclose(logged.structure_function, np.arange(1, 5)*.01, rtol=1e-12)
    assert abs(logged.hurst-1.) < 1e-12
    raw = hurst_structure_function(series, max_lag=4, log_transform=False)
    assert not np.allclose(raw.structure_function, logged.structure_function)


def test_local_sessions_not_utc_dates_and_stable_variance_block_schema():
    # A UTC midnight inside the same local session must not reset the returns.
    index = pd.date_range("2025-08-07 18:30", periods=9, freq="h", tz="America/New_York")
    price = pd.Series(100.*np.exp(np.arange(9)*.01), index=index)
    returns = log_returns_from_close(price)
    assert len(returns) == 7  # one local midnight reset, not the UTC midnight
    short = realized_variance_blocks(price.iloc[:5], block_size=2, annualization=1.)
    longer = realized_variance_blocks(price, block_size=2, annualization=1.)
    pd.testing.assert_frame_equal(short, longer.loc[short.index])
    assert {"block_end", "realized_variance", "raw_realized_variance", "annualized_variance", "annualized_volatility"} <= set(short)
    np.testing.assert_array_equal(short.realized_variance, short.annualized_variance)
    np.testing.assert_allclose(short.annualized_variance, .0001, atol=1e-15)
