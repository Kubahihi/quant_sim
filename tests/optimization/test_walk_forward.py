from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.optimization.estimators as estimators_module
import src.optimization.walk_forward as walk_forward_module
from src.optimization import run_optimization_walk_forward


def _sample_returns(periods: int = 420, assets: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(998)
    dates = pd.date_range("2023-01-02", periods=periods, freq="B")
    factors = rng.normal(0.00025, 0.006, size=(periods, 1))
    residuals = rng.normal(0.00015, 0.010, size=(periods, assets))
    return pd.DataFrame(
        factors + residuals,
        index=dates,
        columns=[f"A{index}" for index in range(assets)],
    )


def test_walk_forward_is_causal_and_charges_costs():
    returns = _sample_returns()
    changed_future = returns.copy()
    mutation_start = returns.index[-84]
    changed_future.loc[mutation_start:, "A0"] += 0.03

    original = run_optimization_walk_forward(
        returns,
        train_periods=126,
        rebalance_periods=21,
        max_weight=0.60,
        transaction_cost_bps=15.0,
    )
    changed = run_optimization_walk_forward(
        changed_future,
        train_periods=126,
        rebalance_periods=21,
        max_weight=0.60,
        transaction_cost_bps=15.0,
    )

    assert original["causal"] is True
    assert original["validation_type"] == "rolling_reoptimization_out_of_sample"
    pd.testing.assert_frame_equal(
        original["weights_history"].loc[:mutation_start],
        changed["weights_history"].loc[:mutation_start],
    )
    pd.testing.assert_series_equal(
        original["net_returns"].loc[: mutation_start - pd.Timedelta(days=1)],
        changed["net_returns"].loc[: mutation_start - pd.Timedelta(days=1)],
    )
    assert original["metrics"]["transaction_cost_drag"] >= 0
    assert np.all(
        original["net_returns"].loc[original["transaction_costs"].index]
        <= original["gross_returns"].loc[original["transaction_costs"].index] + 1e-12
    )


def test_walk_forward_reports_equal_weight_baseline_and_valid_windows():
    returns = _sample_returns(periods=300, assets=3)
    result = run_optimization_walk_forward(
        returns,
        optimizer="minimum_variance",
        train_periods=100,
        rebalance_periods=20,
        max_weight=0.70,
    )

    assert len(result["net_returns"]) == 200
    assert len(result["equal_weight_returns"]) == 200
    assert len(result["windows"]) == 10
    assert result["weights_history"].shape == (10, 3)
    assert all(
        window["train_end"] < window["test_start"]
        for window in result["windows"]
    )
    assert set(result["metrics"]) == set(result["equal_weight_metrics"])


def test_walk_forward_rejects_invalid_configuration():
    returns = _sample_returns(periods=80)
    with pytest.raises(ValueError, match="optimizer"):
        run_optimization_walk_forward(returns, optimizer="clairvoyant", train_periods=40)
    with pytest.raises(ValueError, match="training window"):
        run_optimization_walk_forward(returns, train_periods=80)
    with pytest.raises(ValueError, match="maximum_utility"):
        run_optimization_walk_forward(
            returns,
            optimizer="maximum_sharpe",
            train_periods=40,
            strategy={"long_only": True},
        )


@pytest.mark.parametrize("objective", ["maximum_utility", "minimum_cvar"])
def test_walk_forward_validates_new_convex_objectives(objective):
    returns = _sample_returns(periods=260, assets=4)
    result = run_optimization_walk_forward(
        returns,
        optimizer=objective,
        train_periods=100,
        rebalance_periods=20,
        max_weight=0.50,
        turnover_limit=1.0,
        transaction_cost_bps=12.0,
    )

    assert result["success"] is True
    assert result["optimizer"] == objective
    assert len(result["windows"]) == 8
    assert all(window["optimizer_success"] for window in result["windows"])


def test_walk_forward_minimum_variance_keeps_mandate_constraints():
    returns = _sample_returns(periods=220, assets=4)
    metadata = {
        "A0": {"sector": "Technology", "asset_type": "Stock"},
        "A1": {"sector": "Technology", "asset_type": "Stock"},
        "A2": {"sector": "Health Care", "asset_type": "Stock"},
        "A3": {"sector": "Industrials", "asset_type": "Stock"},
    }
    result = run_optimization_walk_forward(
        returns,
        optimizer="minimum_variance",
        train_periods=100,
        rebalance_periods=20,
        strategy={"long_only": True, "max_sector_weight": 0.50},
        asset_metadata=metadata,
        transaction_cost_bps=10.0,
    )

    assert result["success"] is True
    technology_weights = (
        result["weights_history"]["A0"] + result["weights_history"]["A1"]
    )
    assert np.all(technology_weights <= 0.50 + 1e-5)


@pytest.mark.parametrize("optimizer", ["minimum_variance", "maximum_utility"])
def test_walk_forward_estimates_and_cleans_each_window_once(
    monkeypatch: pytest.MonkeyPatch,
    optimizer: str,
):
    returns = _sample_returns(periods=180, assets=4)
    calls = {"clean": 0, "estimate": 0}
    original_clean = estimators_module.clean_returns
    original_estimate = walk_forward_module.estimate_portfolio_inputs

    def counted_clean(frame: pd.DataFrame) -> pd.DataFrame:
        calls["clean"] += 1
        return original_clean(frame)

    def counted_estimate(frame: pd.DataFrame, **kwargs):
        calls["estimate"] += 1
        return original_estimate(frame, **kwargs)

    monkeypatch.setattr(estimators_module, "clean_returns", counted_clean)
    monkeypatch.setattr(walk_forward_module, "clean_returns", counted_clean)
    monkeypatch.setattr(
        walk_forward_module,
        "estimate_portfolio_inputs",
        counted_estimate,
    )

    result = walk_forward_module.run_optimization_walk_forward(
        returns,
        optimizer=optimizer,
        train_periods=100,
        rebalance_periods=20,
        max_weight=0.50,
    )

    window_count = len(result["windows"])
    assert calls["estimate"] == window_count
    assert calls["clean"] == window_count + 1


def test_walk_forward_black_litterman_estimates_each_window_once(
    monkeypatch: pytest.MonkeyPatch,
):
    returns = _sample_returns(periods=180, assets=4)
    calls = {"clean": 0, "black_litterman": 0}
    original_clean = estimators_module.clean_returns
    original_black_litterman = (
        walk_forward_module.estimate_black_litterman_inputs
    )

    def counted_clean(frame: pd.DataFrame) -> pd.DataFrame:
        calls["clean"] += 1
        return original_clean(frame)

    def counted_black_litterman(frame: pd.DataFrame, **kwargs):
        calls["black_litterman"] += 1
        return original_black_litterman(frame, **kwargs)

    monkeypatch.setattr(estimators_module, "clean_returns", counted_clean)
    monkeypatch.setattr(walk_forward_module, "clean_returns", counted_clean)
    monkeypatch.setattr(
        walk_forward_module,
        "estimate_black_litterman_inputs",
        counted_black_litterman,
    )

    result = walk_forward_module.run_optimization_walk_forward(
        returns,
        optimizer="maximum_utility",
        train_periods=100,
        rebalance_periods=20,
        max_weight=0.50,
        expected_return_model="black_litterman",
        black_litterman_views={"A0": 0.08},
    )

    window_count = len(result["windows"])
    assert calls["black_litterman"] == window_count
    assert calls["clean"] == window_count + 1
