from __future__ import annotations

from copy import deepcopy

import pytest

from src.portfolio_tracker.strategy_alignment import (
    analyze_portfolio_alignment,
    analyze_strategy_alignment,
    normalize_client_mandate,
    normalize_strategy_rulebook,
)


def _mandate() -> dict:
    return {
        "client_name": "Future Foundation",
        "case_status": "active",
        "risk_tolerance": "moderate",
        "horizon_years": 12,
        "liquidity_need_pct": 5,
        "base_currency": "usd",
        "values_constraints": {
            "excluded_sectors": ["Tobacco"],
            "excluded_tickers": ["BAD"],
        },
        "goals": [
            {
                "name": "Long-term growth",
                "target_weight": 70,
                "priority": 5,
                "horizon": "10+ years",
                "description": "Compound capital",
            },
            {"name": "Social impact", "target_weight": 30, "priority": 4},
        ],
    }


def _strategy() -> dict:
    return {
        "name": "Quality compounders",
        "thesis": "Durable growth with controlled concentration.",
        "max_position_weight": 0.50,
        "max_sector_weight": 0.70,
        "min_cash_weight": 0.05,
        "max_cash_weight": 0.15,
        "target_holdings": 3,
        "max_turnover": 0.30,
        "sector_targets": [
            {"sector": "Technology", "target_weight": 0.60, "min_weight": 0.50, "max_weight": 0.70},
            {"sector": "Health Care", "target_weight": 0.40, "min_weight": 0.30, "max_weight": 0.50},
        ],
    }


def test_normalizers_accept_editable_payloads_and_apply_client_floor():
    mandate = normalize_client_mandate(_mandate())
    strategy = normalize_strategy_rulebook(_strategy(), mandate)

    assert mandate["base_currency"] == "USD"
    assert mandate["liquidity_need_pct"] == pytest.approx(0.05)
    assert [goal["target_weight"] for goal in mandate["goals"]] == pytest.approx([0.7, 0.3])
    assert mandate["values_constraints"]["excluded_tickers"] == ["BAD"]
    assert strategy["min_cash_weight"] == pytest.approx(0.05)
    assert strategy["max_position_weight"] == pytest.approx(0.50)
    assert strategy["target_holdings"] == 3
    assert [row["target_weight"] for row in strategy["sector_targets"]] == pytest.approx([0.6, 0.4])
    assert "client_values_constraints" in strategy["configured_rules"]


def test_alignment_calculates_goal_sector_cash_and_concentration_metrics():
    holdings = [
        {
            "ticker": "AAA",
            "market_value": 500,
            "sector": "Technology",
            "primary_goal": "Long-term growth",
            "beta": 1.1,
            "approved": True,
            "thesis_status": "active",
        },
        {
            "ticker": "BBB",
            "market_value": 300,
            "sector": "Health Care",
            "primary_goal": "Social impact",
            "beta": 0.9,
            "approved": True,
            "thesis_status": "active",
        },
        {"ticker": "CASH", "market_value": 200, "asset_type": "cash"},
    ]

    result = analyze_strategy_alignment(holdings, _mandate(), _strategy())

    summary = result["portfolio_summary"]
    assert summary["total_value"] == pytest.approx(1_000)
    assert summary["invested_value"] == pytest.approx(800)
    assert summary["cash_weight"] == pytest.approx(0.20)
    assert summary["concentration_hhi"] == pytest.approx(0.625**2 + 0.375**2)
    assert summary["effective_holdings"] == pytest.approx(1 / (0.625**2 + 0.375**2))
    assert summary["weighted_beta"] == pytest.approx(1.025)

    goals = {row["goal_id"]: row for row in result["goal_allocation"]}
    assert goals["long-term-growth"]["actual_weight"] == pytest.approx(0.625)
    assert goals["social-impact"]["actual_weight"] == pytest.approx(0.375)
    sectors = {row["sector"]: row for row in result["sector_allocation"]}
    assert sectors["Technology"]["actual_weight"] == pytest.approx(0.625)
    assert sectors["Health Care"]["actual_weight"] == pytest.approx(0.375)
    assert any(item["code"] == "cash_outside_range" for item in result["violations"])
    assert 0 <= result["alignment_score"] <= 100
    assert result["components"]["cash"]["applicable"] is True


def test_restricted_unapproved_invalidated_and_unassigned_holding_is_flagged():
    holdings = [
        {
            "ticker": "BAD",
            "market_value": 900,
            "sector": "Tobacco",
            "approved": False,
            "thesis_status": "invalidated",
        }
    ]
    strategy = {**_strategy(), "max_position_weight": 0.40, "require_approved": True}

    result = analyze_strategy_alignment(
        holdings,
        _mandate(),
        strategy,
        cash_value=100,
    )

    codes = {item["code"] for item in result["violations"]}
    assert {
        "holding_goal_missing",
        "holding_not_approved",
        "holding_prohibited",
        "holding_sector_restricted",
        "holding_thesis_invalidated",
        "position_limit_exceeded",
        "unassigned_goal_exposure",
    } <= codes
    holding = result["holdings"][0]
    assert holding["status"] == "misaligned"
    assert holding["alignment_score"] < 50


def test_mapping_holdings_and_multi_goal_splits_are_deterministic_without_mutation():
    holdings = {
        "BBB": {
            "market_value": 400,
            "sector": "Health Care",
            "goals": {"Long-term growth": 1, "Social impact": 1},
        },
        "AAA": {
            "market_value": 600,
            "sector": "Technology",
            "goals": ["Long-term growth"],
        },
    }
    original = deepcopy(holdings)

    first = analyze_strategy_alignment(holdings, _mandate(), _strategy(), cash_value=0)
    second = analyze_portfolio_alignment(holdings, _mandate(), _strategy(), cash_value=0)

    assert first == second
    assert holdings == original
    assert [row["ticker"] for row in first["holdings"]] == ["AAA", "BBB"]
    goals = {row["goal_id"]: row for row in first["goal_allocation"]}
    assert goals["long-term-growth"]["market_value"] == pytest.approx(800)
    assert goals["social-impact"]["market_value"] == pytest.approx(200)


def test_supplied_portfolio_value_infers_cash_and_weight_based_values():
    holdings = [
        {
            "ticker": "AAA",
            "weight": 0.60,
            "sector": "Technology",
            "primary_goal": "Long-term growth",
        },
        {
            "ticker": "BBB",
            "weight": 0.30,
            "sector": "Health Care",
            "primary_goal": "Social impact",
        },
    ]

    result = analyze_strategy_alignment(
        holdings,
        _mandate(),
        _strategy(),
        portfolio_value=1_000,
    )

    assert result["portfolio_summary"]["invested_value"] == pytest.approx(900)
    assert result["portfolio_summary"]["cash_value"] == pytest.approx(100)
    assert result["portfolio_summary"]["cash_weight"] == pytest.approx(0.10)
    assert result["holdings"][0]["market_value"] == pytest.approx(600)


def test_empty_inputs_have_a_stable_neutral_result():
    result = analyze_strategy_alignment([], {}, {})

    assert result["alignment_score"] == pytest.approx(100)
    assert result["portfolio_summary"]["effective_holdings"] == 0
    assert result["goal_allocation"] == []
    assert result["sector_allocation"] == []
    assert result["holdings"] == []


def test_duplicate_ticker_lots_are_one_position_for_concentration_and_limits():
    holdings = [
        {
            "id": "lot-1",
            "ticker": "AAA",
            "market_value": 300,
            "sector": "Technology",
            "beta": 1.0,
            "tags": ["quality"],
        },
        {
            "id": "lot-2",
            "ticker": "aaa",
            "market_value": 200,
            "sector": "Technology",
            "beta": 1.3,
            "tags": ["compounder"],
        },
    ]

    result = analyze_strategy_alignment(
        holdings,
        {},
        {"max_position_weight": 0.40},
        portfolio_value=1_000,
    )

    summary = result["portfolio_summary"]
    assert summary["position_count"] == 1
    assert summary["concentration_hhi"] == pytest.approx(1.0)
    assert summary["effective_holdings"] == pytest.approx(1.0)
    assert summary["largest_position_weight"] == pytest.approx(0.50)
    assert len(result["holdings"]) == 1
    holding = result["holdings"][0]
    assert holding["market_value"] == pytest.approx(500)
    assert holding["lot_count"] == 2
    assert holding["beta"] == pytest.approx((300 * 1.0 + 200 * 1.3) / 500)
    assert any(item["code"] == "position_limit_exceeded" for item in result["violations"])


def test_missing_thesis_status_is_not_active_coverage():
    result = analyze_strategy_alignment(
        [{"ticker": "AAA", "market_value": 100, "thesis_status": "missing"}],
        {},
        {},
    )

    assert result["portfolio_summary"]["active_thesis_coverage"] == pytest.approx(0.0)
    assert "thesis_active" in result["holdings"][0]["failed_rules"]
    assert any(item["code"] == "holding_thesis_missing" for item in result["violations"])


@pytest.mark.parametrize("approved", [None, False])
def test_approval_is_not_a_rule_when_require_approved_is_false(approved):
    holding = {"ticker": "AAA", "market_value": 100}
    if approved is not None:
        holding["approved"] = approved

    result = analyze_strategy_alignment(
        [holding],
        {},
        {"require_approved": False},
    )

    assert "approved" not in {check["code"] for check in result["holdings"][0]["checks"]}
    assert not any(item["code"] == "holding_not_approved" for item in result["violations"])


def test_unknown_approval_fails_when_require_approved_is_true():
    result = analyze_strategy_alignment(
        [{"ticker": "AAA", "market_value": 100}],
        {},
        {"require_approved": True},
    )

    approval_check = next(
        check for check in result["holdings"][0]["checks"] if check["code"] == "approved"
    )
    assert approval_check["passed"] is False
    assert any(item["code"] == "holding_not_approved" for item in result["violations"])
