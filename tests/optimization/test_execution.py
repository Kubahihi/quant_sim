from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.optimization import (
    build_execution_plan,
    estimate_trade_costs,
    optimize_portfolio,
    parse_tax_lots,
)


def test_square_root_impact_grows_faster_than_trade_size() -> None:
    small = estimate_trade_costs(
        [0.01],
        ["A"],
        portfolio_value=1_000_000.0,
        market_impact_bps=20.0,
        average_daily_dollar_volume={"A": 1_000_000.0},
    )
    large = estimate_trade_costs(
        [0.04],
        ["A"],
        portfolio_value=1_000_000.0,
        market_impact_bps=20.0,
        average_daily_dollar_volume={"A": 1_000_000.0},
    )

    assert large["market_impact_drag"] > 4.0 * small["market_impact_drag"]


def test_execution_plan_respects_lots_cash_minimum_trade_and_holding_count() -> None:
    result = build_execution_plan(
        ["A", "B", "C"],
        [0.55, 0.30, 0.15],
        prices={"A": 101.0, "B": 49.0, "C": 24.0},
        portfolio_value=10_000.0,
        current_shares={"A": 20.0, "B": 60.0, "C": 50.0},
        maximum_holdings=2,
        minimum_trade_value=100.0,
        transaction_cost_bps=5.0,
        half_spread_bps=4.0,
        market_impact_bps=10.0,
        average_daily_dollar_volume={"A": 5_000_000.0, "B": 4_000_000.0, "C": 2_000_000.0},
        maximum_adv_participation=0.05,
    )

    assert result["success"] is True
    assert result["cash"] >= 0.0
    assert result["holding_count"] <= 2
    assert all(float(shares).is_integer() for shares in result["final_shares"])
    assert all(trade["notional"] >= 100.0 for trade in result["trades"])
    assert all(trade["adv_participation"] <= 0.05 for trade in result["trades"])


def test_execution_plan_allocates_sales_to_tax_minimizing_lots_first() -> None:
    result = build_execution_plan(
        ["A", "B"],
        [0.25, 0.75],
        prices={"A": 100.0, "B": 100.0},
        portfolio_value=2_000.0,
        current_shares={"A": 10.0, "B": 10.0},
        tax_lots={
            "A": [
                {"shares": 5.0, "cost_basis_per_share": 120.0, "acquired_at": "2025-01-01"},
                {"shares": 5.0, "cost_basis_per_share": 80.0, "acquired_at": "2020-01-01"},
            ]
        },
        short_term_tax_rate=0.35,
        long_term_tax_rate=0.20,
        as_of=date(2026, 1, 10),
    )

    sale = next(trade for trade in result["trades"] if trade["symbol"] == "A")
    first_lot = sale["tax_lot_allocations"][0]
    assert first_lot["cost_basis_per_share"] == pytest.approx(120.0)
    assert sale["realized_gain"] <= 0.0
    assert sale["estimated_tax"] <= 0.0


def test_execution_plan_fails_closed_when_liquidity_blocks_holding_cap() -> None:
    result = build_execution_plan(
        ["A", "B"],
        [1.0, 0.0],
        prices={"A": 100.0, "B": 100.0},
        portfolio_value=1_000.0,
        current_shares={"A": 9.0, "B": 1.0},
        maximum_holdings=1,
        average_daily_dollar_volume={"A": 1_000_000.0, "B": 50.0},
        maximum_adv_participation=0.10,
    )

    assert result["success"] is False
    assert result["holding_constraints_satisfied"] is False
    assert result["holding_count"] == 2
    assert "maximum holding count" in result["message"]


def test_tax_lot_parser_accepts_long_form_csv_shape() -> None:
    lots = parse_tax_lots(pd.DataFrame({
        "Ticker": ["a", "A"],
        "Shares": [2, 3],
        "Cost Basis Per Share": [90.0, 110.0],
        "Acquired At": ["2024-01-01", "2025-02-02"],
    }))

    assert list(lots) == ["A"]
    assert sum(item["shares"] for item in lots["A"]) == pytest.approx(5.0)


def test_optimizer_enforces_adv_limit_and_reports_nonlinear_costs() -> None:
    rng = np.random.default_rng(20260804)
    returns = pd.DataFrame(
        rng.normal([0.0008, 0.0002, 0.0001], [0.012, 0.008, 0.006], size=(320, 3)),
        columns=["A", "B", "C"],
    )
    result = optimize_portfolio(
        returns,
        objective="maximum_utility",
        current_weights=[1 / 3, 1 / 3, 1 / 3],
        max_weight=0.60,
        portfolio_value=1_000_000.0,
        average_daily_dollar_volume={"A": 1_000_000.0, "B": 1_000_000.0, "C": 1_000_000.0},
        max_adv_participation=0.05,
        transaction_cost_bps=5.0,
        half_spread_bps=4.0,
        market_impact_bps=20.0,
        risk_aversion=2.0,
    )

    assert result["success"] is True, result.get("message")
    assert result["solver"] != "OSQP"
    assert result["transaction_cost_breakdown"]["market_impact_drag"] >= 0.0
    assert result["liquidity_report"]
    assert all(row["passed"] for row in result["liquidity_report"])
