from __future__ import annotations

from copy import deepcopy

import pytest

from src.portfolio_tracker.pretrade_analysis import (
    TURNOVER_DEFINITION,
    analyze_pretrade_impact,
    build_competition_strategy_snapshot,
    simulate_trade_plan,
)


def _mandate() -> dict:
    return {
        "client_name": "Competition client",
        "goals": [
            {"name": "Growth", "target_weight": 0.5},
            {"name": "Income", "target_weight": 0.5},
        ],
    }


def _strategy() -> dict:
    return {
        "name": "Balanced strategy",
        "max_position_weight": 0.60,
        "min_cash_weight": 0.0,
        "max_cash_weight": 1.0,
        "max_goal_drift": 0.10,
        "max_sector_drift": 0.10,
        "sector_targets": [
            {"sector": "Technology", "target_weight": 0.5},
            {"sector": "Health Care", "target_weight": 0.5},
        ],
    }


def _theses() -> list[dict]:
    return [
        {
            "ticker": "AAA",
            "status": "active",
            "payload": {
                "sector": "Technology",
                "primary_goal": "Growth",
                "beta": 1.1,
                "tags": ["Quality"],
            },
        },
        {
            "ticker": "BBB",
            "status": "active",
            "payload": {
                "sector": "Health Care",
                "primary_goal": "Income",
                "beta": 0.8,
                "tags": ["Defensive"],
            },
        },
    ]


def test_snapshot_aggregates_lots_and_derives_true_cash_with_realized_pnl():
    positions = [
        {
            "id": 1,
            "ticker": "aaa",
            "status": "open",
            "quantity": 10,
            "entry_price": 100,
            "last_price": 110,
            "security_type": "Stock",
        },
        {
            "id": 2,
            "ticker": "AAA",
            "status": "open",
            "quantity": 5,
            "entry_price": 120,
            "last_price": 130,
            "security_type": "Stock",
        },
        {
            "id": 3,
            "ticker": "CCC",
            "status": "closed",
            "quantity": 2,
            "entry_price": 50,
            "exit_price": 70,
        },
    ]
    original = deepcopy(positions)

    snapshot = build_competition_strategy_snapshot(
        positions,
        {"AAA": 140},
        theses=_theses(),
        approved_securities={"AAA": True},
    )

    assert positions == original
    assert len(snapshot["holdings"]) == 1
    holding = snapshot["holdings"][0]
    assert holding["ticker"] == "AAA"
    assert holding["quantity"] == pytest.approx(15)
    assert holding["lot_count"] == 2
    assert holding["market_value"] == pytest.approx(2_100)
    assert holding["current_price"] == pytest.approx(140)
    assert holding["sector"] == "Technology"
    assert holding["approved"] is True
    assert holding["tags"] == ["quality"]
    assert snapshot["open_quantities"] == {"AAA": pytest.approx(15)}
    assert snapshot["cash_value"] == pytest.approx(500_000 - 1_600 + 40)
    assert snapshot["portfolio_value"] == pytest.approx(500_540)


def test_snapshot_marks_configured_but_rejected_approved_universe():
    snapshot = build_competition_strategy_snapshot(
        [{"ticker": "AAA", "quantity": 1, "entry_price": 10, "status": "open"}],
        approved_securities={"AAA": False},
    )

    assert snapshot["approved_universe_configured"] is True
    assert snapshot["holdings"][0]["approved"] is False


@pytest.mark.parametrize(
    ("live_prices", "last_price", "entry_price", "explicit_price", "expected", "source"),
    [
        ({"AAA": 15}, 12, 10, None, 15, "live"),
        ({}, 12, 10, None, 12, "stored"),
        ({}, None, 10, None, 10, "entry"),
        ({"AAA": 15}, 12, 10, 14, 14, "explicit"),
    ],
)
def test_trade_price_fallback_order(
    live_prices, last_price, entry_price, explicit_price, expected, source
):
    position = {
        "ticker": "AAA",
        "quantity": 2,
        "entry_price": entry_price,
        "last_price": last_price,
        "status": "open",
    }
    trade = {"ticker": "AAA", "action": "sell", "quantity": 1}
    if explicit_price is not None:
        trade["price"] = explicit_price

    result = simulate_trade_plan([position], [trade], live_prices, initial_capital=100)

    assert result["plan_valid"] is True
    assert result["trades"][0]["price"] == pytest.approx(expected)
    assert result["trades"][0]["price_source"] == source


def test_ordered_sell_can_fund_later_buy_but_reverse_order_is_blocked():
    positions = [
        {
            "id": 1,
            "ticker": "AAA",
            "quantity": 1,
            "entry_price": 100,
            "last_price": 100,
            "entry_date": "2025-01-01",
            "status": "open",
        }
    ]
    funded = simulate_trade_plan(
        positions,
        [
            {"ticker": "AAA", "action": "sell", "quantity": 1, "price": 100},
            {"ticker": "BBB", "action": "buy", "quantity": 1, "price": 100},
        ],
        initial_capital=100,
    )
    reversed_plan = simulate_trade_plan(
        positions,
        [
            {"ticker": "BBB", "action": "buy", "quantity": 1, "price": 100},
            {"ticker": "AAA", "action": "sell", "quantity": 1, "price": 100},
        ],
        initial_capital=100,
    )

    assert funded["plan_valid"] is True
    funded_snapshot = build_competition_strategy_snapshot(
        funded["after_positions"], initial_capital=100
    )
    assert funded_snapshot["open_quantities"] == {"BBB": pytest.approx(1)}
    assert funded_snapshot["cash_value"] == pytest.approx(0)
    assert reversed_plan["plan_valid"] is False
    assert {item["code"] for item in reversed_plan["blockers"]} == {"insufficient_cash"}
    assert reversed_plan["after_positions"] == positions


def test_fifo_partial_sell_splits_lots_and_preserves_equity():
    positions = [
        {
            "id": 1,
            "ticker": "AAA",
            "quantity": 10,
            "entry_price": 10,
            "entry_date": "2025-01-01",
            "status": "open",
        },
        {
            "id": 2,
            "ticker": "AAA",
            "quantity": 10,
            "entry_price": 20,
            "entry_date": "2025-02-01",
            "status": "open",
        },
    ]
    original = deepcopy(positions)

    result = simulate_trade_plan(
        positions,
        [{"ticker": "AAA", "action": "sell", "quantity": 15, "price": 30}],
        {"AAA": 30},
        initial_capital=1_000,
    )

    assert positions == original
    assert result["plan_valid"] is True
    snapshot = build_competition_strategy_snapshot(
        result["after_positions"], {"AAA": 30}, initial_capital=1_000
    )
    assert snapshot["open_quantities"] == {"AAA": pytest.approx(5)}
    assert snapshot["cash_value"] == pytest.approx(1_150)
    assert snapshot["portfolio_value"] == pytest.approx(1_300)
    closed = [row for row in result["after_positions"] if row.get("status") == "closed"]
    assert [(row["entry_price"], row["quantity"]) for row in closed] == [
        (10, 10),
        (20, 5),
    ]


def test_realized_proceeds_are_available_to_a_new_buy():
    positions = [
        {
            "ticker": "AAA",
            "quantity": 1,
            "entry_price": 100,
            "exit_price": 150,
            "status": "closed",
        }
    ]

    result = simulate_trade_plan(
        positions,
        [{"ticker": "BBB", "action": "buy", "quantity": 1, "price": 140}],
        initial_capital=100,
    )

    assert result["before_cash_value"] == pytest.approx(150)
    assert result["plan_valid"] is True
    assert result["after_cash_value"] == pytest.approx(10)


@pytest.mark.parametrize(
    ("positions", "trade", "expected_code"),
    [
        (
            [{"ticker": "AAA", "quantity": 1, "entry_price": 10, "status": "open"}],
            {"ticker": "AAA", "action": "sell", "quantity": 2, "price": 10},
            "oversell",
        ),
        (
            [],
            {"ticker": "AAA", "action": "buy", "quantity": 2, "price": 60},
            "insufficient_cash",
        ),
        (
            [],
            {"ticker": "AAA", "action": "buy", "quantity": 1},
            "missing_price",
        ),
        (
            [],
            {"ticker": "AAA", "action": "buy", "quantity": 1, "price": -5},
            "invalid_price",
        ),
    ],
)
def test_trade_blockers_are_atomic(positions, trade, expected_code):
    original_positions = deepcopy(positions)
    original_trade = deepcopy(trade)

    result = simulate_trade_plan(
        positions,
        [trade],
        initial_capital=100,
    )

    assert positions == original_positions
    assert trade == original_trade
    assert result["plan_valid"] is False
    assert expected_code in {item["code"] for item in result["blockers"]}
    assert result["after_positions"] == original_positions
    assert result["after_performance"] == result["before_performance"]


def test_turnover_is_gross_notional_over_pretrade_equity():
    result = simulate_trade_plan(
        [],
        [
            {"ticker": "AAA", "action": "buy", "quantity": 2, "price": 100},
            {"ticker": "BBB", "action": "buy", "quantity": 1, "price": 50},
        ],
        initial_capital=1_000,
    )

    assert result["gross_proposed_notional"] == pytest.approx(250)
    assert result["incremental_turnover"] == pytest.approx(0.25)
    assert result["turnover_definition"] == TURNOVER_DEFINITION


def test_pretrade_analysis_reports_new_position_limit_violation_and_weight_delta():
    positions = [
        {
            "ticker": "AAA",
            "quantity": 4,
            "entry_price": 100,
            "last_price": 100,
            "status": "open",
        }
    ]
    strategy = {
        "max_position_weight": 0.50,
        "min_cash_weight": 0.0,
        "max_cash_weight": 1.0,
    }

    result = analyze_pretrade_impact(
        positions,
        [{"ticker": "AAA", "action": "buy", "quantity": 2, "price": 100}],
        {"goals": [{"name": "Growth", "target_weight": 1.0}]},
        strategy,
        theses=_theses(),
        approved_securities={"AAA": True},
        initial_capital=1_000,
    )

    assert result["status"] == "review"
    assert result["plan_valid"] is True
    assert result["gross_proposed_notional"] == pytest.approx(200)
    assert result["incremental_turnover"] == pytest.approx(0.20)
    assert "position_limit_exceeded" in {
        item["code"] for item in result["violation_changes"]["new"]
    }
    change = next(item for item in result["holding_changes"] if item["ticker"] == "AAA")
    assert change["before_weight"] == pytest.approx(0.4)
    assert change["after_weight"] == pytest.approx(0.6)
    assert change["weight_delta"] == pytest.approx(0.2)
    assert result["deltas"]["cash_value"] == pytest.approx(-200)


def test_pretrade_analysis_marks_resolved_violation_stably():
    positions = [
        {
            "ticker": "AAA",
            "quantity": 6,
            "entry_price": 100,
            "last_price": 100,
            "status": "open",
        }
    ]
    result = analyze_pretrade_impact(
        positions,
        [{"ticker": "AAA", "action": "sell", "quantity": 2, "price": 100}],
        {"goals": [{"name": "Growth", "target_weight": 1.0}]},
        {"max_position_weight": 0.50, "min_cash_weight": 0.0, "max_cash_weight": 1.0},
        theses=_theses(),
        initial_capital=1_000,
    )

    resolved = result["violation_changes"]["resolved"]
    item = next(item for item in resolved if item["code"] == "position_limit_exceeded")
    assert item["identity"] == "position_limit_exceeded|holding|aaa"
    assert result["status"] == "pass"


def test_goal_and_sector_drift_comparison_uses_before_after_allocations():
    positions = [
        {
            "ticker": "AAA",
            "quantity": 5,
            "entry_price": 100,
            "last_price": 100,
            "status": "open",
        }
    ]

    result = analyze_pretrade_impact(
        positions,
        [{"ticker": "BBB", "action": "buy", "quantity": 5, "price": 100}],
        _mandate(),
        _strategy(),
        theses=_theses(),
        initial_capital=1_000,
    )

    growth = next(item for item in result["goal_changes"] if item["goal_name"] == "Growth")
    income = next(item for item in result["goal_changes"] if item["goal_name"] == "Income")
    technology = next(
        item for item in result["sector_changes"] if item["sector_name"] == "Technology"
    )
    assert growth["before_actual_weight"] == pytest.approx(1.0)
    assert growth["after_actual_weight"] == pytest.approx(0.5)
    assert growth["abs_drift_delta"] == pytest.approx(-0.5)
    assert income["before_actual_weight"] == pytest.approx(0.0)
    assert income["after_actual_weight"] == pytest.approx(0.5)
    assert technology["before_actual_weight"] == pytest.approx(1.0)
    assert technology["after_actual_weight"] == pytest.approx(0.5)
    assert result["deltas"]["goal_absolute_drift"] == pytest.approx(-1.0)
    assert result["deltas"]["sector_absolute_drift"] == pytest.approx(-1.0)
    assert result["status"] == "pass"


def test_incremental_turnover_is_added_only_when_baseline_is_explicit():
    result = analyze_pretrade_impact(
        [],
        [{"ticker": "AAA", "action": "buy", "quantity": 1, "price": 100}],
        {},
        {
            "max_turnover": 0.10,
            "min_cash_weight": 0.0,
            "max_cash_weight": 1.0,
        },
        theses=_theses(),
        initial_capital=1_000,
        current_turnover=0.05,
    )

    assert result["incremental_turnover"] == pytest.approx(0.10)
    assert "turnover_limit_exceeded" in {
        item["code"] for item in result["violation_changes"]["new"]
    }


def test_blocked_pretrade_analysis_keeps_before_and_after_identical():
    result = analyze_pretrade_impact(
        [],
        [{"ticker": "AAA", "action": "buy", "quantity": 2, "price": 100}],
        {},
        {},
        initial_capital=100,
    )

    assert result["status"] == "blocked"
    assert result["before"] == result["after"]
    assert result["violation_changes"] == {"new": [], "resolved": [], "persistent": []}
    assert result["deltas"]["alignment_score"] == pytest.approx(0)


def test_analyze_pretrade_impact_does_not_mutate_any_input():
    positions = [{"ticker": "AAA", "quantity": 2, "entry_price": 100, "status": "open"}]
    trades = [{"ticker": "AAA", "action": "sell", "quantity": 1}]
    mandate = _mandate()
    strategy = _strategy()
    theses = _theses()
    approved = [{"ticker": "AAA", "approved": True}]
    originals = deepcopy((positions, trades, mandate, strategy, theses, approved))

    analyze_pretrade_impact(
        positions,
        trades,
        mandate,
        strategy,
        theses=theses,
        approved_securities=approved,
    )

    assert (positions, trades, mandate, strategy, theses, approved) == originals
