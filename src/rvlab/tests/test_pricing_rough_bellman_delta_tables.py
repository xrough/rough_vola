"""Independent-resolution uncertainty and finite-bump decision statistics."""
import numpy as np
import pytest

from rvlab.experiments._pricing_rough_bellman_delta_tables import summarize_delta_streams


def stream(seed, steps, deltas):
    relative = np.array([-.005, -.0025, 0., .0025, .005])
    x = 100 * relative
    price = 3. + np.asarray(deltas)[:, None] * x
    return {"seed": seed, "steps": steps, "relative_spots": relative,
        "baseline": {"raw": price, "controlled": price,
            "stops": np.full(price.shape, 64)},
        "bellman": {"raw": price-.01*x, "controlled": price-.01*x,
            "stops": np.full(price.shape, 64)}}


def test_time_check_uses_independent_variance_even_with_identical_path_values():
    # Deliberately identical values cannot establish cross-resolution pairing.
    result = summarize_delta_streams([stream(1, 128, [-.3, -.4, -.5]),
        stream(2, 256, [-.3, -.4, -.5])])
    checks = result["time_refinement"]
    assert len(checks) == 8
    assert np.allclose(checks.delta_change, 0)
    assert np.allclose(checks.standard_error, np.sqrt(2*.01/3))
    assert not checks.time_check_pass.any()
    assert checks.same_frozen_rule.all()
    assert not checks.optimal_delta_ready.any()


def test_linear_frozen_rule_passes_bump_checks_and_preserves_policy_difference():
    result = summarize_delta_streams([stream(1, 128, [-.4]*8), stream(2, 128, [-.4]*8)])
    assert len(result["delta_sampling"]) == 8
    assert result["bump_refinement"].bump_check_pass.all()
    assert np.allclose(result["policy_delta_change"].estimate, -.01)
    assert result["raw_control"].tolerance_bias_resolved.all()


def test_strata_are_weighted_by_their_declared_path_counts():
    result = summarize_delta_streams([stream(1, 128, [-.4]*8), stream(2, 128, [-.2]*4)])
    row = result["delta_sampling"].query("policy == 'baseline'").iloc[0]
    assert row.estimate == pytest.approx(-1/3)
    assert row.paths == 12
    assert row.streams == 2


def test_repeated_seed_cannot_be_reported_as_an_independent_stream():
    with pytest.raises(ValueError, match="reuse"):
        summarize_delta_streams([stream(1, 128, [-.4]*8), stream(1, 256, [-.4]*8)])
