from __future__ import annotations

import numpy as np
import pytest

from src.simulation.goal_funding import (
    GoalFundingConfig,
    compare_goal_funding_scenarios,
    compare_portfolios_from_asset_returns,
    evaluate_goal_funding,
    project_goal_funding,
    simulate_goal_funding,
)


def _config(**overrides) -> GoalFundingConfig:
    inputs = {
        "target_wealth": 120.0,
        "horizon_years": 2,
        "initial_capital": 100.0,
    }
    inputs.update(overrides)
    return GoalFundingConfig(**inputs)


def test_terminal_scenarios_report_goal_probability_and_conditional_deficit():
    metrics = evaluate_goal_funding(
        _config(),
        terminal_wealth=np.array([80.0, 100.0, 120.0, 150.0]),
    )

    assert metrics.probability_goal_achieved == 0.5
    assert metrics.shortfall_probability == 0.5
    assert metrics.expected_terminal_wealth == 112.5
    assert metrics.median_terminal_wealth == 110.0
    assert metrics.percentile_10_terminal_wealth == pytest.approx(86.0)
    # Conditional on failure: mean of the 40 and 20 deficits.
    assert metrics.expected_shortfall_vs_goal == 30.0
    assert metrics.ruin_probability is None
    assert metrics.ruin_probability_confidence_interval is None
    lower, upper = metrics.goal_achievement_confidence_interval
    assert 0.0 < lower < metrics.probability_goal_achieved < upper < 1.0


def test_no_failed_scenario_has_zero_expected_shortfall():
    metrics = evaluate_goal_funding(
        _config(target_wealth=100.0),
        terminal_wealth=[100.0, 110.0, 120.0],
    )

    assert metrics.probability_goal_achieved == 1.0
    assert metrics.shortfall_probability == 0.0
    assert metrics.expected_shortfall_vs_goal == 0.0


def test_complete_paths_enable_ruin_probability_without_counting_initial_wealth():
    metrics = evaluate_goal_funding(
        _config(),
        wealth_paths=np.array(
            [
                [100.0, 0.0, 0.0],
                [100.0, 80.0, 130.0],
            ]
        ),
    )

    assert metrics.probability_goal_achieved == 0.5
    assert metrics.ruin_probability == 0.5
    assert metrics.ruin_probability_confidence_interval is not None


def test_projection_applies_end_of_year_cashflow_and_floors_wealth_at_zero():
    projection = project_goal_funding(
        _config(annual_net_cashflow=-60.0),
        np.array(
            [
                [0.0, 0.0],
                [-0.5, 0.5],
            ]
        ),
    )

    np.testing.assert_allclose(
        projection.wealth_paths,
        np.array(
            [
                [100.0, 40.0, 0.0],
                [100.0, 0.0, 0.0],
            ]
        ),
    )
    assert projection.metrics.ruin_probability == 1.0
    assert not projection.wealth_paths.flags.writeable
    assert not projection.portfolio_return_scenarios.flags.writeable


def test_real_basis_deflates_nominal_returns_and_keeps_real_cashflow_constant():
    projection = project_goal_funding(
        _config(
            target_wealth=110.0,
            horizon_years=1,
            annual_net_cashflow=10.0,
            inflation_rate=0.10,
            wealth_basis="real",
        ),
        np.array([[0.10], [0.10]]),
        returns_basis="nominal",
    )

    # A 10% nominal return at 10% inflation is a 0% real return.
    np.testing.assert_allclose(projection.portfolio_return_scenarios, 0.0, atol=1e-15)
    np.testing.assert_allclose(projection.wealth_paths[:, -1], 110.0)
    assert projection.metrics.wealth_basis == "real"
    assert projection.metrics.probability_goal_achieved == 1.0


def test_start_of_year_contribution_participates_in_that_years_return():
    projection = project_goal_funding(
        _config(
            target_wealth=121.0,
            horizon_years=1,
            annual_net_cashflow=10.0,
            cashflow_timing="start",
        ),
        [[0.10], [0.10]],
    )

    np.testing.assert_allclose(projection.wealth_paths[:, -1], 121.0)


def test_seeded_bootstrap_is_reproducible_and_does_not_touch_global_rng():
    annual_asset_returns = np.array(
        [
            [0.10, 0.02],
            [-0.05, 0.04],
            [0.20, -0.01],
            [0.03, 0.06],
        ]
    )
    weights = np.array([0.6, 0.4])

    np.random.seed(917)
    expected_global_draw = np.random.random(4)
    np.random.seed(917)
    first = simulate_goal_funding(
        _config(horizon_years=3),
        annual_asset_returns,
        weights,
        n_scenarios=50,
        random_seed=42,
    )
    global_draw_after_call = np.random.random(4)
    second = simulate_goal_funding(
        _config(horizon_years=3),
        annual_asset_returns,
        weights,
        n_scenarios=50,
        random_seed=42,
    )

    assert np.array_equal(global_draw_after_call, expected_global_draw)
    assert np.array_equal(first.sampled_observation_indices, second.sampled_observation_indices)
    assert np.array_equal(first.portfolio_return_scenarios, second.portfolio_return_scenarios)
    assert np.array_equal(first.wealth_paths, second.wealth_paths)
    assert first.metrics == second.metrics


def test_named_scenario_comparison_is_paired_and_exposes_decision_metrics():
    scenarios = {
        "Current": np.array([[0.0, 0.0], [0.0, 0.0], [0.2, 0.2]]),
        "Growth": np.array([[0.1, 0.1], [0.1, 0.1], [0.2, 0.2]]),
    }

    results = compare_goal_funding_scenarios(_config(), scenarios)

    assert list(results) == ["Current", "Growth"]
    assert results["Current"].probability_goal_achieved == pytest.approx(1 / 3)
    assert results["Growth"].probability_goal_achieved == 1.0
    assert results["Growth"].expected_shortfall_vs_goal == 0.0


def test_weight_comparison_uses_identical_bootstrap_shocks_for_every_candidate():
    annual_asset_returns = np.array(
        [
            [0.10, 0.01],
            [-0.08, 0.03],
            [0.15, -0.02],
        ]
    )
    results = compare_portfolios_from_asset_returns(
        _config(horizon_years=4),
        annual_asset_returns,
        {
            "A": [0.6, 0.4],
            "A copy": [0.6, 0.4],
            "B": [0.2, 0.8],
        },
        n_scenarios=40,
        random_seed=11,
    )

    assert np.array_equal(
        results["A"].sampled_observation_indices,
        results["B"].sampled_observation_indices,
    )
    assert np.array_equal(results["A"].wealth_paths, results["A copy"].wealth_paths)
    assert results["A"].metrics == results["A copy"].metrics


@pytest.mark.parametrize(
    "overrides, error_type",
    [
        ({"target_wealth": 0.0}, ValueError),
        ({"horizon_years": 0}, ValueError),
        ({"horizon_years": 1.5}, TypeError),
        ({"initial_capital": -1.0}, ValueError),
        ({"inflation_rate": -1.0}, ValueError),
        ({"wealth_basis": "today"}, ValueError),
        ({"cashflow_timing": "middle"}, ValueError),
        ({"confidence_level": 1.0}, ValueError),
    ],
)
def test_config_rejects_invalid_inputs(overrides, error_type):
    with pytest.raises(error_type):
        _config(**overrides)


def test_evaluation_requires_exactly_one_well_formed_scenario_input():
    config = _config()
    with pytest.raises(ValueError, match="exactly one"):
        evaluate_goal_funding(config)
    with pytest.raises(ValueError, match="exactly one"):
        evaluate_goal_funding(
            config,
            terminal_wealth=[100.0, 120.0],
            wealth_paths=[[100.0, 110.0, 120.0], [100.0, 100.0, 100.0]],
        )
    with pytest.raises(ValueError, match="first wealth_paths column"):
        evaluate_goal_funding(
            config,
            wealth_paths=[[99.0, 110.0, 120.0], [99.0, 100.0, 100.0]],
        )


@pytest.mark.parametrize(
    "returns, weights, expected_message",
    [
        ([[0.1, 0.2]], [0.5, 0.5], "at least two observations"),
        ([[0.1, np.nan], [0.2, 0.3]], [0.5, 0.5], "finite"),
        ([[0.1, 0.2], [0.2, 0.3]], [0.4, 0.4], "sum to 1"),
        ([[0.1, 0.2], [0.2, 0.3]], [1.0], "one value per asset"),
    ],
)
def test_bootstrap_rejects_invalid_return_data_and_weights(
    returns,
    weights,
    expected_message,
):
    with pytest.raises(ValueError, match=expected_message):
        simulate_goal_funding(
            _config(),
            returns,
            weights,
            n_scenarios=10,
            random_seed=1,
        )


def test_projection_rejects_impossible_simple_returns_and_wrong_horizon():
    with pytest.raises(ValueError, match="below -100%"):
        project_goal_funding(_config(), [[-1.01, 0.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="2 columns"):
        project_goal_funding(_config(), [[0.0], [0.0]])


def test_comparison_rejects_unpaired_scenario_counts():
    with pytest.raises(ValueError, match="share one shape"):
        compare_goal_funding_scenarios(
            _config(),
            {
                "A": np.zeros((2, 2)),
                "B": np.zeros((3, 2)),
            },
        )
