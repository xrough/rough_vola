"""State-only diagnostics cannot silently replace a sealed Monte Carlo law."""
from dataclasses import replace

import numpy as np
import pytest

from rvlab.tests.test_pricing_rough_local_control import local_case
from rvlab.experiments._pricing_rough_state_replay import replay_inception_states


def test_replay_recovers_complete_fourier_state_and_payoff_stream_bit_exact(local_case):
    proxy, _, trace = local_case
    replay = replay_inception_states(proxy, seed=trace["seed"], paths=len(trace["stocks"]),
        expected_payoffs=trace["discounted_payoffs"])
    for name in ("times", "actual_time_grid", "stocks", "factors", "discounted_payoffs"):
        np.testing.assert_array_equal(replay[name], trace[name])
    assert replay["diagnostics"]["sealed_payoffs_bit_exact"]


def test_different_stream_and_missing_paths_are_not_silently_accepted(local_case):
    proxy, _, trace = local_case
    for seed, paths in ((trace["seed"]+1, len(trace["stocks"])), (trace["seed"], len(trace["stocks"])-1)):
        with pytest.raises(ValueError, match="sealed stream"):
            replay_inception_states(proxy, seed=seed, paths=paths, expected_payoffs=trace["discounted_payoffs"])


@pytest.mark.parametrize("seed,paths", [(True, 32), (-1, 32), (4, 0), (4, True)])
def test_invalid_stream_declarations_rejected(local_case, seed, paths):
    with pytest.raises(ValueError): replay_inception_states(local_case[0], seed=seed, paths=paths)


def test_memory_budget_is_checked_before_allocating_replay(local_case):
    proxy = local_case[0]
    proxy = replace(proxy, config=replace(proxy.config, memory_budget_mb=.0001))
    with pytest.raises(MemoryError): replay_inception_states(proxy, seed=44, paths=32)
