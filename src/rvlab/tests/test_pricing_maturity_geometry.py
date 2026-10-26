"""Identification comparisons use physical noise and declared fit weights."""

import numpy as np
import pytest

from rvlab.experiments._pricing_maturity_geometry import (
    coupled_linearized_noise, linearized_geometry,
)


def test_sandwich_noise_map_matches_direct_weighted_least_squares():
    jacobian = np.array([[1., 0.], [0., 2.], [1., 1.], [2., -1.]])
    spreads = np.array([.01, .02, .05, .03])
    noise_std = .005
    result = linearized_geometry(jacobian, {"a": .04, "b": 1.5}, spreads,
                                 noise_std=noise_std)
    weights = np.diag(1/spreads**2)
    direct = np.linalg.solve(jacobian.T@weights@jacobian, jacobian.T@weights)
    np.testing.assert_allclose(result["physical_target_to_parameter_map"], direct)
    np.testing.assert_allclose(result["minimum_norm_noise_covariance"],
                               noise_std**2*direct@direct.T)
    assert result["full_column_rank"]
    assert result["rank"] == 2


def test_repeated_short_quotes_reduce_noise_without_improving_geometry():
    jacobian = np.array([[1., .99], [1., 1.], [1., 1.01]])
    spreads = np.array([.01, .02, .03])
    short = linearized_geometry(jacobian, {"a": .04, "b": 1.5}, spreads)
    repeated = linearized_geometry(np.tile(jacobian, (2, 1)),
        {"a": .04, "b": 1.5}, np.tile(spreads, 2))
    np.testing.assert_allclose(list(repeated["local_parameter_noise_sd"].values()),
        np.asarray(list(short["local_parameter_noise_sd"].values()))/np.sqrt(2))
    assert repeated["column_normalized_condition_number"] == pytest.approx(
        short["column_normalized_condition_number"])
    np.testing.assert_allclose([row["nuisance_projected_fraction"] for row in repeated["sensitivities"]],
        [row["nuisance_projected_fraction"] for row in short["sensitivities"]])


def test_new_horizon_direction_improves_weak_coordinate_beyond_count_control():
    short_jacobian = np.array([[1., .99], [1., 1.], [1., 1.01]])
    long_jacobian = np.vstack([short_jacobian, [[1., -1.], [2., -1.], [1., -2.]]])
    short = linearized_geometry(short_jacobian, {"a": .04, "b": .045}, np.ones(3)*.01)
    augmented = linearized_geometry(long_jacobian, {"a": .04, "b": .045}, np.ones(6)*.01)
    assert augmented["local_parameter_noise_sd"]["b"] < short["local_parameter_noise_sd"]["b"]/10
    assert augmented["column_normalized_condition_number"] < 2


def test_rank_deficiency_does_not_report_artificial_precision():
    result = linearized_geometry([[1., 0.], [2., 0.]], {"a": 1., "b": 2.}, [.01, .02])
    assert result["rank"] == 1
    assert not result["full_column_rank"]
    assert np.isinf(result["column_normalized_condition_number"])
    assert np.isinf(result["local_parameter_noise_sd"]["b"])
    assert np.isfinite(result["local_parameter_noise_sd"]["a"])
    table = coupled_linearized_noise({"short": result}, draws=100, seed=4)
    assert np.isinf(table.loc[table.parameter.eq("b"), "simulated_sd"]).all()
    assert table.loc[table.parameter.eq("b"), "q025"].isna().all()


def test_coupled_noise_reproducible_with_different_row_counts():
    params = {"a": .04, "b": .045}
    short = linearized_geometry([[1., 0.], [0., 1.]], params, [.01, .02],
                                bounds={"a": (.0025, .25), "b": (.0025, .25)})
    augmented = linearized_geometry([[1., 0.], [0., 1.], [1., 1.]], params, [.01, .02, .03],
                                    bounds={"a": (.0025, .25), "b": (.0025, .25)})
    first = coupled_linearized_noise({"short": short, "augmented": augmented}, draws=10000)
    second = coupled_linearized_noise({"short": short, "augmented": augmented}, draws=10000)
    assert first.equals(second)
    np.testing.assert_allclose(first.simulated_sd, first.analytic_sd, rtol=.025)
    assert first.draws.eq(10000).all()


@pytest.mark.parametrize("jacobian,parameters,spreads,kwargs", [
    ([[1.]], {}, [.01], {}),
    ([[1.]], {"a": 1., "b": 2.}, [.01], {}),
    ([[np.nan]], {"a": 1.}, [.01], {}),
    ([[1.]], {"a": np.inf}, [.01], {}),
    ([[1.]], {"a": 1.}, [0.], {}),
    ([[1.]], {"a": 1.}, [.01, .02], {}),
    ([[1.]], {"a": 1.}, [.01], {"noise_std": -1.}),
    ([[1.]], {"a": 1.}, [.01], {"rcond": 0.}),
    ([[1.]], {"a": 1.}, [.01], {"bounds": {"b": (0., 1.)}}),
    ([[1.]], {"a": 1.}, [.01], {"bounds": {"a": (1., 0.)}}),
])
def test_invalid_inputs_rejected(jacobian, parameters, spreads, kwargs):
    with pytest.raises(ValueError):
        linearized_geometry(jacobian, parameters, spreads, **kwargs)
