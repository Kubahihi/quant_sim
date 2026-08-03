from datetime import date

from src.portfolio_tracker.competition_readiness import (
    assess_security_dossier,
    assess_strategy_constitution,
    build_competition_readiness,
    build_pitch_question_bank,
    generate_competition_brief,
)


MANDATE = {
    "client_name": "Case Client", "case_status": "Official case entered",
    "goals": [{"name": "Legacy", "target_weight": 1.0}],
    "risk_tolerance": "Growth", "risk_capacity": "Moderate",
    "max_tolerated_drawdown": 0.18, "drawdown_response": "Review, do not panic sell",
    "total_financial_picture": "Operating assets plus liquid portfolio",
    "horizon_years": 10, "liquidity_need_pct": 0.05,
    "values_constraints": {"excluded_sectors": ["Tobacco"]},
    "policy_benchmark": "60% ACWI / 40% AGG",
    "policy_benchmark_rationale": "Matches growth and liquidity buckets.",
    "behavioral_profile": {
        "answers": {f"q{index}": 3 for index in range(12)},
        "drawdown_actions": {
            "-10%": "Hold the strategic allocation",
            "-20%": "Seek advice before acting",
            "-30%": "Seek advice before acting",
        },
        "decision_protocol": "Review the mandate, wait 48 hours, and obtain a second review.",
    },
}
STRATEGY = {
    "strategy_thesis": "Own durable compounders with identifiable catalysts.",
    "selection_factors": ["ROIC", "FCF"], "max_position_weight": 0.1,
    "max_sector_weight": 0.25, "sell_discipline": "Sell on thesis break.",
    "drift_limit": 0.03,
}
THESIS = {
    "ticker": "ABC", "payload": {
        "portfolio_role": "Quality compounder", "primary_goal": "Legacy",
        "why_now": "Temporary demand slowdown", "investment_thesis": "Moat supports reinvestment.",
        "value_drivers": ["Volume", "margin"], "monitoring_kpis": ["ROIC", "retention"],
        "bear_case": "Flat", "base_case": "Recover", "bull_case": "Accelerate",
        "fair_value_scenarios": {"bear": 80, "base": 110, "bull": 145},
        "margin_of_safety": 0.15, "counter_thesis": "Moat is eroding",
        "risks": ["Competition"], "invalidation": "Retention below 80%",
        "review_date": "2026-09-01",
    },
}


def test_complete_constitution_and_dossier_score_full_marks():
    source = {"ticker": "ABC", "title": "10-K", "primary_source": True}
    catalyst = {"ticker": "ABC", "title": "Investor day"}
    assert assess_strategy_constitution(MANDATE, STRATEGY)["score"] == 100
    assert assess_security_dossier("ABC", THESIS, [source], [catalyst])["score"] == 100


def test_readiness_exposes_gaps_and_generates_defensible_brief():
    readiness = build_competition_readiness(
        mandate=MANDATE,
        strategy=STRATEGY,
        theses=[THESIS],
        sources=[], catalysts=[], decisions=[], thesis_reviews=[],
        decision_reviews=[], red_team_reviews=[], ai_usage=[],
    )
    assert readiness["overall_score"] < 100
    questions = build_pitch_question_bank(readiness)
    assert any("evidence" in question.lower() for question in questions)
    brief = generate_competition_brief(
        readiness, mandate=MANDATE, strategy=STRATEGY,
        generated_on=date(2026, 7, 31),
    )
    assert "Competition Readiness Brief" in brief
    assert "60% ACWI / 40% AGG" in brief
    assert "team must verify every claim" in brief
