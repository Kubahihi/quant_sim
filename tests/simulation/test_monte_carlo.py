from __future__ import annotations

import numpy as np
import pytest

from src.simulation.monte_carlo import (
    calculate_percentile_paths,
    run_advanced_monte_carlo_simulation,
    run_monte_carlo_simulation,
)


def test_seeded_gbm_is_reproducible_and_converges_to_analytic_mean():
    paths_a, stats_a = run_monte_carlo_simulation(
        current_value=100_000.0,
        expected_return=0.08,
        volatility=0.20,
        time_horizon=252,
        n_simulations=20_000,
        random_seed=42,
    )
    paths_b, stats_b = run_monte_carlo_simulation(
        current_value=100_000.0,
        expected_return=0.08,
        volatility=0.20,
        time_horizon=252,
        n_simulations=20_000,
        random_seed=42,
    )

    assert np.array_equal(paths_a, paths_b)
    assert stats_a == stats_b
    assert stats_a["model"] == "geometric_brownian_motion"
    assert stats_a["mean"] == pytest.approx(stats_a["analytic_mean"], rel=0.01)
    assert abs(stats_a["mean"] - stats_a["analytic_mean"]) <= 3 * stats_a["standard_error_mean"]
    assert stats_a["relative_standard_error_mean"] < 0.01


@pytest.mark.parametrize(
    "kwargs",
    [
        {"current_value": 0.0},
        {"volatility": -0.1},
        {"time_horizon": 0},
        {"n_simulations": 1},
    ],
)
def test_gbm_rejects_invalid_inputs(kwargs):
    base = {
        "current_value": 100.0,
        "expected_return": 0.05,
        "volatility": 0.2,
        "time_horizon": 20,
        "n_simulations": 100,
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        run_monte_carlo_simulation(**base)


def test_percentile_paths_validates_shape_and_bounds():
    with pytest.raises(ValueError):
        calculate_percentile_paths(np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        calculate_percentile_paths(np.ones((2, 3)), percentiles=[101])


def test_seeded_advanced_monte_carlo_is_reproducible_without_global_rng_side_effects():
    np.random.seed(777)
    expected_global_draw = np.random.random(3)
    np.random.seed(777)

    paths_a, stats_a = run_advanced_monte_carlo_simulation(
        current_value=50_000.0,
        expected_return=0.07,
        volatility=0.18,
        time_horizon=126,
        n_simulations=5_000,
        jump_intensity=1.2,
        jump_mean=-0.04,
        jump_volatility=0.07,
        random_seed=99,
    )
    global_draw_after_call = np.random.random(3)
    paths_b, stats_b = run_advanced_monte_carlo_simulation(
        current_value=50_000.0,
        expected_return=0.07,
        volatility=0.18,
        time_horizon=126,
        n_simulations=5_000,
        jump_intensity=1.2,
        jump_mean=-0.04,
        jump_volatility=0.07,
        random_seed=99,
    )

    assert np.array_equal(paths_a, paths_b)
    assert stats_a == stats_b
    assert np.array_equal(global_draw_after_call, expected_global_draw)
    assert stats_a["model"] == "merton_jump_diffusion"
    assert stats_a["random_seed"] == 99
    assert "relative_standard_error_mean" in stats_a
    assert "expected_shortfall_95_loss" in stats_a
    assert stats_a["realized_average_jumps_per_path"] >= 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"jump_intensity": -0.1},
        {"jump_volatility": -0.1},
        {"current_value": -1.0},
        {"n_simulations": 1},
    ],
)
def test_advanced_monte_carlo_rejects_invalid_inputs(kwargs):
    base = {
        "current_value": 100.0,
        "expected_return": 0.05,
        "volatility": 0.2,
        "time_horizon": 20,
        "n_simulations": 100,
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        run_advanced_monte_carlo_simulation(**base)
