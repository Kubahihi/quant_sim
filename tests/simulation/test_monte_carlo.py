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


def test_batched_percentile_paths_match_individual_percentiles_exactly():
    paths = np.random.default_rng(123).lognormal(size=(31, 257))
    percentiles = [5, 25, 50, 75, 95]

    result = calculate_percentile_paths(paths, percentiles=percentiles)

    for percentile in percentiles:
        assert np.array_equal(
            result[f"p{percentile}"].to_numpy(),
            np.percentile(paths, percentile, axis=1),
        )


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


def test_optimized_advanced_paths_match_direct_merton_reference_exactly():
    current_value = 25_000.0
    expected_return = 0.06
    volatility = 0.21
    time_horizon = 40
    n_simulations = 300
    jump_intensity = 2.5
    jump_mean = -0.03
    jump_volatility = 0.09
    random_seed = 818

    rng = np.random.default_rng(random_seed)
    dt = 1.0 / 252.0
    jump_compensator = jump_intensity * (
        np.exp(jump_mean + 0.5 * jump_volatility**2) - 1.0
    )
    drift = (expected_return - 0.5 * volatility**2 - jump_compensator) * dt
    diffusion = volatility * np.sqrt(dt)
    diffusion_returns = drift + diffusion * rng.standard_normal(
        (time_horizon, n_simulations)
    )
    n_jumps = rng.poisson(
        jump_intensity * dt,
        (time_horizon, n_simulations),
    )
    jump_returns = np.zeros_like(diffusion_returns)
    jump_mask = n_jumps > 0
    jump_returns[jump_mask] = rng.normal(
        n_jumps[jump_mask] * jump_mean,
        np.sqrt(n_jumps[jump_mask]) * jump_volatility,
    )
    reference_paths = current_value * np.exp(
        np.vstack(
            [
                np.zeros(n_simulations),
                np.cumsum(diffusion_returns + jump_returns, axis=0),
            ]
        )
    )

    optimized_paths, statistics = run_advanced_monte_carlo_simulation(
        current_value=current_value,
        expected_return=expected_return,
        volatility=volatility,
        time_horizon=time_horizon,
        n_simulations=n_simulations,
        jump_intensity=jump_intensity,
        jump_mean=jump_mean,
        jump_volatility=jump_volatility,
        random_seed=random_seed,
    )

    assert np.array_equal(optimized_paths, reference_paths)
    assert statistics["realized_average_jumps_per_path"] == pytest.approx(
        np.mean(np.sum(n_jumps, axis=0))
    )


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
