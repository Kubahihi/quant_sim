from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.risk_metrics import (
    calculate_drawdown_series,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)


def test_drawdown_uses_running_peak_and_preserves_index():
    index = pd.date_range("2026-01-01", periods=5)
    returns = pd.Series([0.10, -0.20, 0.05, 0.25, -0.10], index=index)

    result = calculate_drawdown_series(returns)
    cumulative = (1.0 + returns).cumprod()
    running_peak = cumulative.expanding().max().clip(lower=1.0)
    expected = (cumulative - running_peak) / running_peak

    pd.testing.assert_series_equal(result, expected)
    assert calculate_max_drawdown(returns) == expected.min()


def test_drawdown_matches_expanding_reference_with_missing_observation():
    returns = pd.Series([0.02, np.nan, -0.03, 0.04, -0.01])
    cumulative = (1.0 + returns).cumprod()
    running_peak = cumulative.expanding().max().clip(lower=1.0)
    expected = (cumulative - running_peak) / running_peak

    pd.testing.assert_series_equal(calculate_drawdown_series(returns), expected)
    assert calculate_max_drawdown(returns) == expected.min()


def test_drawdown_includes_loss_before_first_new_high():
    returns = pd.Series([-0.20, 0.10])

    expected = pd.Series([-0.20, -0.12])
    pd.testing.assert_series_equal(calculate_drawdown_series(returns), expected)
    assert calculate_max_drawdown(returns) == pytest.approx(-0.20)


def test_sharpe_uses_effective_periodic_risk_free_rate():
    returns = pd.Series([0.001, 0.002, -0.001, 0.003])
    annual_rf = 0.10
    periodic_rf = np.expm1(np.log1p(annual_rf) / 252)
    excess = returns - periodic_rf
    expected = excess.mean() / excess.std() * np.sqrt(252)

    assert calculate_sharpe_ratio(returns, annual_rf) == expected


def test_sortino_uses_all_observations_for_target_downside_deviation():
    returns = pd.Series([0.02, -0.01, 0.03, -0.02])
    downside = np.minimum(returns, 0.0)
    expected = returns.mean() / np.sqrt(np.mean(np.square(downside))) * np.sqrt(252)

    assert calculate_sortino_ratio(returns, risk_free_rate=0.0) == expected
