import numpy as np
import pandas as pd

from src.analytics.competition_analytics import (
    build_reproducibility_manifest,
    calculate_brinson_attribution,
    calculate_policy_benchmark,
    run_walk_forward_rank_backtest,
)


def test_policy_benchmark_normalizes_weights():
    returns = pd.DataFrame({"ACWI": [0.01, -0.02], "AGG": [0.002, 0.003]})
    result = calculate_policy_benchmark(returns, {"ACWI": 60, "AGG": 40})
    np.testing.assert_allclose(result, [0.0068, -0.0108])


def test_brinson_effects_reconcile_to_active_return():
    portfolio = pd.DataFrame({
        "sector": ["Tech", "Health"], "weight": [0.6, 0.4], "return": [0.12, 0.03],
    })
    benchmark = pd.DataFrame({
        "sector": ["Tech", "Health"], "weight": [0.5, 0.5], "return": [0.10, 0.04],
    })
    result = calculate_brinson_attribution(portfolio, benchmark)
    active = 0.6 * 0.12 + 0.4 * 0.03 - (0.5 * 0.10 + 0.5 * 0.04)
    assert np.isclose(result["total_effect"].sum(), active)


def test_walk_forward_lags_scores_and_charges_turnover():
    dates = pd.date_range("2026-01-01", periods=4)
    returns = pd.DataFrame({"A": [0.0, 0.1, 0.0, 0.0], "B": [0.0, 0.0, 0.2, 0.0]}, index=dates)
    scores = pd.DataFrame({"A": [2, 0, 0, 0], "B": [1, 3, 3, 3]}, index=dates)
    result = run_walk_forward_rank_backtest(returns, scores, top_n=1, rebalance_every=1, transaction_cost_bps=10)
    assert result.loc[dates[0], "gross_return"] == 0
    assert result.loc[dates[1], "gross_return"] == 0.1
    assert result.loc[dates[2], "gross_return"] == 0.2
    assert result.loc[dates[1], "transaction_cost"] > 0


def test_manifest_hashes_are_deterministic_and_sensitive():
    frame = pd.DataFrame({"B": [2.0], "A": [1.0]}, index=["x"])
    first = build_reproducibility_manifest(frame, {"window": 60}, source="unit", as_of="2026-07-31")
    second = build_reproducibility_manifest(frame, {"window": 60}, source="unit", as_of="2026-07-31")
    changed = build_reproducibility_manifest(frame, {"window": 61}, source="unit", as_of="2026-07-31")
    assert first["data_sha256"] == second["data_sha256"]
    assert first["config_sha256"] != changed["config_sha256"]
