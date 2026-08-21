from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.risk_metrics import (
    calculate_drawdown_series,
    calculate_max_drawdown,
)


def test_drawdown_uses_running_peak_and_preserves_index():
    index = pd.date_range("2026-01-01", periods=5)
    returns = pd.Series([0.10, -0.20, 0.05, 0.25, -0.10], index=index)

    result = calculate_drawdown_series(returns)
    cumulative = (1.0 + returns).cumprod()
    expected = (cumulative - cumulative.expanding().max()) / cumulative.expanding().max()

    pd.testing.assert_series_equal(result, expected)
    assert calculate_max_drawdown(returns) == expected.min()


def test_drawdown_matches_expanding_reference_with_missing_observation():
    returns = pd.Series([0.02, np.nan, -0.03, 0.04, -0.01])
    cumulative = (1.0 + returns).cumprod()
    expected = (cumulative - cumulative.expanding().max()) / cumulative.expanding().max()

    pd.testing.assert_series_equal(calculate_drawdown_series(returns), expected)
    assert calculate_max_drawdown(returns) == expected.min()
