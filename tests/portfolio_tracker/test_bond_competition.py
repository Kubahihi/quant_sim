from __future__ import annotations

import pandas as pd

from src.portfolio_tracker.bond_competition import (
    ELIGIBILITY_PENDING,
    ELIGIBILITY_VERIFIED,
    assess_bond_competition_case,
    build_bond_pitch_questions,
    build_bond_relative_value_table,
    generate_bond_competition_memo,
)


def _position(**overrides):
    values = {
        "id": 1,
        "ticker": "US0001",
        "security_type": "Bond",
        "bond_instrument_type": "individual",
        "bond_category": "Corporate",
        "isin": "US0000000001",
        "issuer": "Example Corp",
        "currency": "USD",
        "quantity": 10,
        "face_value": 1_000,
        "entry_price": 99,
        "last_price": 100,
        "coupon_rate": 0.05,
        "coupon_frequency": 2,
        "maturity_date": "2031-08-03",
        "next_coupon_date": "2027-02-03",
        "fx_rate_to_usd": 1,
        "yield_to_maturity": 0.05,
        "benchmark_name": "5Y Treasury",
        "benchmark_yield": 0.035,
        "default_probability": 0.01,
        "recovery_rate": 0.4,
        "credit_rating": "A",
        "valuation_source": "WInS statement",
        "source_url": "official-statement-reference",
        "price_observed_at": "2026-08-03",
        "competition_eligibility_status": ELIGIBILITY_VERIFIED,
        "eligibility_source": "Official WInS security list",
        "eligibility_checked_at": "2026-08-03",
        "status": "open",
    }
    values.update(overrides)
    return values


def _metrics():
    return {
        "clean_price": 100.0,
        "dirty_price": 100.0,
        "yield_to_maturity": 0.05,
        "yield_to_call": None,
        "yield_to_worst": 0.05,
        "spread_to_benchmark": 0.015,
        "modified_duration": 4.3,
        "dv01_usd": 43.0,
        "annual_income_usd": 500.0,
        "expected_loss_usd": 60.0,
        "default_probability": 0.01,
        "recovery_rate": 0.4,
    }


def _case(**overrides):
    values = {
        "client_goal": "Preserve capital and generate income",
        "portfolio_role": "Income",
        "proposed_weight": 0.02,
        "max_position_weight": 0.05,
        "eligibility_status": ELIGIBILITY_VERIFIED,
        "eligibility_source": "Official WInS security list",
        "eligibility_checked_at": "2026-08-03",
        "thesis": "Contractual income compensates for measured rate and credit risk.",
        "why_now": "Spread exceeds the team's required threshold.",
        "risks": "Rates, spread widening, default, and liquidity.",
        "invalidation": "Credit rating falls below the portfolio floor.",
        "sell_discipline": "Review monthly and sell on thesis breach.",
        "counter_thesis": "The spread does not compensate for recession risk.",
    }
    values.update(overrides)
    return values


def test_complete_case_is_ready_and_generates_auditable_memo_and_questions():
    scenario = pd.DataFrame({"ExpectedReturn": [-0.08, 0.04]})
    readiness = assess_bond_competition_case(
        _position(), _metrics(), {"score": 100, "issues": []}, _case(), scenario_grid=scenario,
    )

    assert readiness["score"] == 100
    assert readiness["status"] == "Ready for team approval"
    assert readiness["blockers"] == []
    assert readiness["worst_scenario_return"] == -0.08

    questions = build_bond_pitch_questions(_position(), _metrics(), readiness, _case())
    assert any("100 bp" in question for question in questions)
    assert any("return decomposition" in question for question in questions)

    memo = generate_bond_competition_memo(
        _position(), _metrics(), readiness, _case(), questions=questions,
    )
    assert "# Bond Competition Case: US0001" in memo
    assert "student team must write" in memo
    assert "Official WInS security list" in memo


def test_pending_eligibility_and_oversized_position_are_hard_blockers():
    case = _case(
        eligibility_status=ELIGIBILITY_PENDING,
        eligibility_source="",
        eligibility_checked_at=None,
        proposed_weight=0.10,
        max_position_weight=0.05,
    )
    readiness = assess_bond_competition_case(
        _position(competition_eligibility_status=ELIGIBILITY_PENDING),
        _metrics(),
        {"score": 100, "issues": []},
        case,
        scenario_grid=pd.DataFrame({"ExpectedReturn": [0.0]}),
    )

    assert readiness["status"] == "Do not trade"
    assert any("eligibility" in item.lower() for item in readiness["blockers"])
    assert any("position weight" in item.lower() for item in readiness["blockers"])


def test_relative_value_table_keeps_evidence_and_risk_components_separate():
    table = build_bond_relative_value_table(
        [_position()],
        [{"id": 1, "current_price": 100.0, "current_value": 10_000.0}],
        as_of="2026-08-03",
    )

    assert list(table["Identifier"]) == ["US0001"]
    assert table.loc[0, "Eligibility"] == ELIGIBILITY_VERIFIED
    assert bool(table.loc[0, "EvidenceReady"]) is True
    assert table.loc[0, "CarryAfterExpectedLoss"] < table.loc[0, "CurrentYield"]
    assert "Score" not in table.columns
