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


def test_walk_forward_broadcasts_weights_between_rebalances_without_lookahead():
    dates = pd.date_range("2026-01-01", periods=7)
    returns = pd.DataFrame({
        "A": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        "B": [0.0, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06],
    }, index=dates)
    scores = pd.DataFrame({
        "A": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "B": [1.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
    }, index=dates)

    result = run_walk_forward_rank_backtest(
        returns,
        scores,
        top_n=1,
        rebalance_every=3,
        transaction_cost_bps=0.0,
    )

    np.testing.assert_allclose(
        result["gross_return"],
        [0.0, 0.01, 0.02, 0.03, -0.04, -0.05, -0.06],
    )
    np.testing.assert_allclose(
        result["turnover"],
        [0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0],
    )


def test_walk_forward_block_implementation_matches_row_reference():
    rng = np.random.default_rng(818)
    dates = pd.date_range("2025-01-01", periods=37)
    columns = ["A", "B", "C", "D", "E"]
    returns = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(len(dates), len(columns))),
        index=dates,
        columns=columns,
    )
    scores = pd.DataFrame(
        rng.normal(size=(len(dates), len(columns))),
        index=dates,
        columns=columns,
    )
    scores.iloc[::5, 2] = np.nan

    reference_weights = pd.DataFrame(0.0, index=dates, columns=columns)
    reference_turnover = pd.Series(0.0, index=dates)
    current = pd.Series(0.0, index=columns)
    for position, timestamp in enumerate(dates[:-1]):
        if position % 7 == 0:
            available = scores.loc[timestamp].dropna().sort_values(ascending=False)
            selected = available.head(3).index
            target = pd.Series(0.0, index=columns)
            target.loc[selected] = 1.0 / len(selected)
            reference_turnover.iloc[position + 1] = float((target - current).abs().sum())
            current = target
        reference_weights.iloc[position + 1] = current
    reference_gross = (reference_weights * returns).sum(axis=1)
    reference_costs = reference_turnover * 12.0 / 10_000.0
    reference = pd.DataFrame({
        "gross_return": reference_gross,
        "turnover": reference_turnover,
        "transaction_cost": reference_costs,
        "net_return": reference_gross - reference_costs,
    })

    optimized = run_walk_forward_rank_backtest(
        returns,
        scores,
        top_n=3,
        rebalance_every=7,
        transaction_cost_bps=12.0,
    )

    pd.testing.assert_frame_equal(optimized, reference)


def test_manifest_hashes_are_deterministic_and_sensitive():
    frame = pd.DataFrame({"B": [2.0], "A": [1.0]}, index=["x"])
    first = build_reproducibility_manifest(frame, {"window": 60}, source="unit", as_of="2026-07-31")
    second = build_reproducibility_manifest(frame, {"window": 60}, source="unit", as_of="2026-07-31")
    changed = build_reproducibility_manifest(frame, {"window": 61}, source="unit", as_of="2026-07-31")
    assert first["data_sha256"] == second["data_sha256"]
    assert first["config_sha256"] != changed["config_sha256"]
