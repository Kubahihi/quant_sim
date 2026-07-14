from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.model_validation import (
    build_model_validation_report,
    distribution_diagnostics,
    moving_block_bootstrap_intervals,
)
from src.analytics.modular.backtest import walk_forward_baseline_backtest
from src.simulation.monte_carlo import run_monte_carlo_simulation


def _returns(n: int = 756) -> pd.Series:
    rng = np.random.default_rng(123)
    values = rng.standard_t(df=5, size=n) * 0.009 + 0.0003
    return pd.Series(values, index=pd.date_range("2022-01-03", periods=n, freq="B"))


def test_block_bootstrap_intervals_are_seeded_and_ordered():
    series = _returns(504)
    first = moving_block_bootstrap_intervals(series, n_bootstrap=200, random_seed=7)
    second = moving_block_bootstrap_intervals(series, n_bootstrap=200, random_seed=7)

    assert first == second
    assert set(first) == {"annualized_return", "volatility", "sharpe_ratio", "var_95", "cvar_95"}
    for interval in first.values():
        assert interval["ci_low"] <= interval["ci_high"]


def test_distribution_diagnostics_detect_fat_tails_without_scipy_dependency():
    diagnostics = distribution_diagnostics(_returns(2000))
    assert diagnostics["excess_kurtosis"] > 0
    assert 0.0 <= diagnostics["normality_p_value"] <= 1.0


def test_validation_report_does_not_claim_predictive_accuracy():
    series = _returns()
    _, simulation = run_monte_carlo_simulation(
        current_value=100_000,
        expected_return=float(series.mean() * 252),
        volatility=float(series.std() * np.sqrt(252)),
        time_horizon=252,
        n_simulations=5_000,
        random_seed=42,
    )
    backtest = walk_forward_baseline_backtest(series)
    report = build_model_validation_report(
        series,
        simulation_stats=simulation,
        backtest=backtest,
        n_bootstrap=200,
    )

    assert 0.0 <= report["methodology_score"] <= 100.0
    assert report["predictive_accuracy_measured"] is False
    assert report["generated_with"]["backtest_validation_type"] == "walk_forward_causal_baseline"
    assert any("full model ensemble" in item.lower() for item in report["limitations"])
