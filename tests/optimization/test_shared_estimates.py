from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimization import (
    calculate_efficient_frontier,
    estimate_portfolio_inputs,
    optimize_cost_aware_rebalance,
    optimize_maximum_sharpe,
    optimize_minimum_variance,
    sample_portfolio_cloud,
)


def _sample_returns(periods: int = 220, assets: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(777)
    common = rng.normal(0.00035, 0.006, size=(periods, 1))
    residual = rng.normal(0.00015, 0.009, size=(periods, assets))
    return pd.DataFrame(
        common + residual,
        columns=[f"A{index}" for index in range(assets)],
    )


def test_shared_estimates_are_finite_psd_and_auditable():
    returns = _sample_returns()
    returns.iloc[0, 0] = np.inf
    returns.iloc[1, 1] = np.nan

    estimates = estimate_portfolio_inputs(
        returns,
        covariance_shrinkage=0.35,
        return_shrinkage=0.60,
    )

    assert estimates.observations == len(returns) - 2
    assert np.all(np.isfinite(estimates.mean_returns))
    assert np.min(np.linalg.eigvalsh(estimates.covariance)) > 0
    assert estimates.metadata()["covariance_shrinkage"] == pytest.approx(0.35)
    assert estimates.metadata()["return_shrinkage"] == pytest.approx(0.60)


def test_optimizers_use_the_same_default_estimation_contract():
    returns = _sample_returns()
    current = np.full(returns.shape[1], 1.0 / returns.shape[1])

    minimum_variance = optimize_minimum_variance(returns)
    maximum_sharpe = optimize_maximum_sharpe(returns)
    cost_aware = optimize_cost_aware_rebalance(
        returns,
        current,
        max_weight=0.50,
        turnover_limit=1.0,
    )
    frontier = calculate_efficient_frontier(returns, n_points=12)
    cloud = sample_portfolio_cloud(returns, n_samples=50)

    metadata = minimum_variance["estimation"]
    assert maximum_sharpe["estimation"] == metadata
    assert cost_aware["estimation"] == metadata
    assert frontier[0]["estimation"] == metadata
    assert cloud.attrs["estimation"] == metadata


def test_precomputed_estimates_preserve_results_and_reject_misalignment():
    returns = _sample_returns()
    estimates = estimate_portfolio_inputs(returns)

    direct = optimize_maximum_sharpe(returns, max_weight=0.40)
    reused = optimize_maximum_sharpe(
        returns,
        max_weight=0.40,
        portfolio_estimates=estimates,
    )

    assert reused["success"] is True
    assert np.array_equal(reused["weights"], direct["weights"])
    assert reused["estimation"] == direct["estimation"]
    with pytest.raises(ValueError, match="symbols must match"):
        optimize_minimum_variance(
            returns[list(reversed(returns.columns))],
            portfolio_estimates=estimates,
        )


def test_frontier_starts_at_global_minimum_variance_and_uses_supplied_rate():
    returns = _sample_returns()
    risk_free_rate = 0.0175
    minimum_variance = optimize_minimum_variance(
        returns,
        risk_free_rate=risk_free_rate,
    )
    frontier = calculate_efficient_frontier(
        returns,
        n_points=18,
        risk_free_rate=risk_free_rate,
    )

    first = frontier[0]
    assert first["volatility"] == pytest.approx(
        minimum_variance["volatility"], abs=1e-8
    )
    assert np.allclose(first["weights"], minimum_variance["weights"], atol=1e-7)
    assert all(
        float(point["volatility"]) >= float(first["volatility"]) - 1e-8
        for point in frontier
    )
    assert np.all(np.diff([float(point["return"]) for point in frontier]) >= -1e-8)
    assert first["sharpe_ratio"] == pytest.approx(
        (float(first["return"]) - risk_free_rate) / float(first["volatility"])
    )


def test_portfolio_cloud_respects_position_cap():
    returns = _sample_returns(assets=5)
    cloud = sample_portfolio_cloud(
        returns,
        n_samples=250,
        max_weight=0.30,
    )

    assert float(cloud["max_weight"].max()) <= 0.30 + 1e-10
