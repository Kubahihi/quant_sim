from copy import deepcopy

import pandas as pd
import pytest

from src.portfolio_tracker.client_behavior import (
    BEHAVIORAL_QUESTIONS,
    DEFAULT_DRAWDOWN_ACTIONS,
    assess_behavioral_profile,
    parse_likert_answer,
)


def _answers(value: int) -> dict[str, int]:
    return {question["id"]: value for question in BEHAVIORAL_QUESTIONS}


def test_likert_parser_accepts_labels_and_rejects_out_of_range_values():
    assert parse_likert_answer("4 - Agree") == 4
    assert parse_likert_answer(2) == 2
    with pytest.raises(ValueError, match="between 1 and 5"):
        parse_likert_answer(6)


def test_disciplined_answers_produce_low_vulnerability_and_strong_resilience():
    result = assess_behavioral_profile(
        _answers(1),
        drawdown_actions={
            "-10%": "Rebalance back to policy targets",
            "-20%": "Rebalance back to policy targets",
            "-30%": "Rebalance back to policy targets",
        },
        risk_tolerance="Aggressive",
    )

    assert result["OverallVulnerabilityScore"] == pytest.approx(0.0)
    assert result["VulnerabilityBand"] == "Low"
    assert result["DecisionStyle"] == "Process-oriented"
    assert result["DrawdownResilienceScore"] == pytest.approx(100.0)
    assert result["RiskToleranceConsistency"] == "Broadly consistent"
    assert isinstance(result["Guardrails"], pd.DataFrame)


def test_high_answers_surface_biases_guardrails_and_tolerance_mismatch():
    result = assess_behavioral_profile(
        _answers(5),
        drawdown_actions={key: "Sell all risk assets" for key in DEFAULT_DRAWDOWN_ACTIONS},
        risk_tolerance="Growth",
    )

    assert result["OverallVulnerabilityScore"] == pytest.approx(100.0)
    assert result["VulnerabilityBand"] == "High"
    assert len(result["TopBiases"]) == 3
    assert set(result["BiasScores"]["Band"]) == {"High"}
    assert len(result["Guardrails"]) >= 8
    assert result["RiskToleranceConsistency"] == "Behaviour below declared tolerance"
    assert any("no trade" in item.lower() for item in result["CommunicationPlan"])


def test_partial_profile_reports_coverage_and_does_not_mutate_inputs():
    answers = {"loss_1": "5 - Strongly agree", "loss_2": 3}
    original = deepcopy(answers)

    result = assess_behavioral_profile(answers)

    assert answers == original
    assert result["CoveragePct"] == pytest.approx(2 / len(BEHAVIORAL_QUESTIONS))
    loss = result["BiasScores"].set_index("Category").loc["loss_aversion"]
    assert loss["Score"] == pytest.approx(75.0)
    assert result["DrawdownResilienceScore"] is None


def test_unknown_questions_and_drawdown_actions_are_rejected():
    with pytest.raises(ValueError, match="Unknown behavioural question"):
        assess_behavioral_profile({"unknown": 3})
    with pytest.raises(ValueError, match="Unknown drawdown action"):
        assess_behavioral_profile({"loss_1": 3}, drawdown_actions={"-10%": "Panic"})
