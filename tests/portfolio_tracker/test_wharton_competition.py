from __future__ import annotations

import pytest

from src.portfolio_tracker.wharton_competition import (
    INITIAL_CAPITAL_USD,
    calculate_portfolio_performance,
    evaluate_compliance,
)


def _compliant_settings() -> dict:
    return {
        "team_size": 5,
        "leader_age": 17,
        "advisor_team_count": 2,
        "same_school": 1,
        "eligible_students": 1,
        "leader_designated": 1,
        "advisor_is_teacher": 1,
        "one_wins_account": 1,
        "members_single_team": 1,
        "no_client_contact": 1,
        "no_paid_advisor": 1,
        "student_owned_work": 1,
        "ai_cited": 1,
        "sources_cited": 1,
        "school_permission": 1,
    }


def test_compliance_reports_exact_team_and_capital_failures():
    settings = _compliant_settings()
    settings["team_size"] = 3
    positions = [{"status": "open", "ticker": "AAA", "quantity": 6_000, "entry_price": 100, "opened_by": "Jakub"}]

    checks = evaluate_compliance(settings, positions)
    failures = [item for item in checks if item["status"] == "fail"]

    assert any("Current value is 3" in item["detail"] for item in failures)
    assert any("exceeding starting capital by $100,000.00" in item["detail"] for item in failures)


def test_compliance_keeps_unpublished_rules_pending():
    checks = evaluate_compliance(_compliant_settings(), [])

    assert not [item for item in checks if item["status"] == "fail"]
    assert len([item for item in checks if item["status"] == "pending"]) == 2


def test_portfolio_performance_tracks_open_closed_and_authors():
    positions = [
        {
            "id": 1,
            "ticker": "AAA",
            "status": "open",
            "quantity": 100,
            "entry_price": 50,
            "last_price": 51,
            "opened_by": "Jakub",
        },
        {
            "id": 2,
            "ticker": "BBB",
            "status": "closed",
            "quantity": 20,
            "entry_price": 100,
            "exit_price": 90,
            "opened_by": "Matěj",
        },
    ]

    result = calculate_portfolio_performance(positions, {"AAA": 55})

    assert result["unrealized_pnl"] == pytest.approx(500)
    assert result["realized_pnl"] == pytest.approx(-200)
    assert result["total_pnl"] == pytest.approx(300)
    assert result["equity"] == pytest.approx(INITIAL_CAPITAL_USD + 300)
    assert result["total_return_pct"] == pytest.approx(0.06)
    assert result["positions"][0]["opened_by"] == "Jakub"
    assert result["positions"][0]["return_pct"] == pytest.approx(10.0)
