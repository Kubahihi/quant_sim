from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimization import optimize_maximum_sharpe


def _sample_returns(periods: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(321)
    dates = pd.date_range("2025-01-02", periods=periods, freq="B")
    return pd.DataFrame(
        {
            "AAA": rng.normal(0.0008, 0.013, periods),
            "BBB": rng.normal(0.0005, 0.010, periods),
            "CCC": rng.normal(0.0003, 0.009, periods),
        },
        index=dates,
    )


def test_maximum_sharpe_uses_shrunk_inputs_and_reports_diagnostics():
    returns = _sample_returns()

    result = optimize_maximum_sharpe(
        returns,
        covariance_shrinkage=0.40,
        return_shrinkage=0.60,
    )

    weights = np.asarray(result["weights"], dtype=float)
    assert result["success"] is True
    assert np.isclose(float(weights.sum()), 1.0, atol=1e-8)
    assert np.all(weights >= -1e-10)
    assert result["estimation"]["method"] == "shrunk_mean_shrunk_covariance"
    assert result["estimation"]["covariance_shrinkage"] == 0.40
    assert result["estimation"]["return_shrinkage"] == 0.60
    assert set(result["estimation"]["sample_expected_returns"]) == {"AAA", "BBB", "CCC"}
    assert set(result["estimation"]["shrunk_expected_returns"]) == {"AAA", "BBB", "CCC"}


def test_maximum_sharpe_relaxes_infeasible_long_only_max_weight():
    returns = _sample_returns()

    result = optimize_maximum_sharpe(returns, max_weight=0.10)

    weights = np.asarray(result["weights"], dtype=float)
    assert result["success"] is True
    assert np.isclose(float(weights.sum()), 1.0, atol=1e-8)
    assert np.max(weights) <= (1.0 / returns.shape[1]) + 1e-8


def test_maximum_sharpe_rejects_empty_returns():
    with pytest.raises(ValueError, match="returns are empty"):
        optimize_maximum_sharpe(pd.DataFrame())
