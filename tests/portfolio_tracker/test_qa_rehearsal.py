from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json

import pytest

from src.portfolio_tracker.qa_rehearsal import (
    add_qa_question,
    complete_mock_round,
    create_mock_round,
    create_qa_rehearsal_workspace,
    killer_question_status,
    member_qa_history,
    record_qa_response,
)


NOW = datetime(2026, 8, 15, 10, 0, tzinfo=timezone.utc)


def _question_bank():
    workspace = create_qa_rehearsal_workspace(
        "finale-qa",
        team_members=["Anna", "Boris", "Cyril"],
        created_by="Anna",
        passing_score=3.5,
        now=NOW,
    )
    workspace = add_qa_question(
        workspace,
        "client-drawdown",
        prompt="Why is this drawdown acceptable for the client?",
        model_answer="It is inside the documented loss capacity and preserves goal-one liquidity.",
        evidence_ids=["mandate-loss-capacity", "scenario-8"],
        primary_responder="Anna",
        backup_responder="Boris",
        time_limit_seconds=60,
        category="client_fit",
        follow_ups=["What would make you de-risk?"],
        killer_question=True,
        created_by="Cyril",
        now=NOW,
    )
    workspace = add_qa_question(
        workspace,
        "sizing-rule",
        prompt="Which rule determined the position size?",
        model_answer="The active rulebook cap and sector risk budget set the maximum.",
        evidence_ids=["rulebook-v4", "decision-aapl"],
        primary_responder="Boris",
        backup_responder="Anna",
        time_limit_seconds=45,
        category="governance",
        created_by="Cyril",
        now=NOW,
    )
    return workspace


def test_timed_round_scores_answers_history_and_killer_questions():
    bank = _question_bank()
    original = deepcopy(bank)
    round_state = create_mock_round(
        bank,
        "mock-1",
        started_by="Cyril",
        question_ids=["client-drawdown", "sizing-rule"],
        random_seed=7,
        now=NOW,
    )
    for slot in round_state["rounds"]["mock-1"]["slots"]:
        question_id = slot["question_id"]
        is_killer = question_id == "client-drawdown"
        round_state = record_qa_response(
            round_state,
            "mock-1",
            question_id,
            responder=slot["assigned_responder"],
            answer="A concise answer tied to the cited evidence.",
            duration_seconds=50 if is_killer else 50,
            scores={"clarity": 4, "evidence": 4, "client_fit": 4},
            evaluator="Cyril",
            follow_up_answers={"What would make you de-risk?": "A threshold breach."}
            if is_killer
            else None,
            now=NOW,
        )
    completed = complete_mock_round(round_state, "mock-1", completed_by="Cyril", now=NOW)

    assert bank == original
    assert completed["rounds"]["mock-1"]["summary"]["question_count"] == 2
    assert completed["rounds"]["mock-1"]["summary"]["passed_count"] == 1
    assert member_qa_history(completed, "Anna")["attempt_count"] == 1
    assert member_qa_history(completed, "Boris")["within_time_pct"] == 0
    assert killer_question_status(completed)["unresolved_count"] == 0
    json.dumps(completed, allow_nan=False)


def test_question_requires_independent_backup_evidence_and_valid_time_limit():
    workspace = create_qa_rehearsal_workspace(
        "qa",
        team_members=["Anna", "Boris"],
        created_by="Anna",
        now=NOW,
    )
    with pytest.raises(ValueError, match="different"):
        add_qa_question(
            workspace,
            "q1",
            prompt="Question?",
            model_answer="Answer.",
            evidence_ids=["e1"],
            primary_responder="Anna",
            backup_responder="Anna",
            time_limit_seconds=60,
            created_by="Boris",
            now=NOW,
        )
    with pytest.raises(ValueError, match="evidence"):
        add_qa_question(
            workspace,
            "q1",
            prompt="Question?",
            model_answer="Answer.",
            evidence_ids=[],
            primary_responder="Anna",
            backup_responder="Boris",
            time_limit_seconds=60,
            created_by="Boris",
            now=NOW,
        )


def test_round_rejects_self_scoring_duplicate_answers_and_early_completion():
    workspace = create_mock_round(
        _question_bank(),
        "mock-2",
        started_by="Cyril",
        question_ids=["client-drawdown"],
        now=NOW,
    )
    slot = workspace["rounds"]["mock-2"]["slots"][0]

    with pytest.raises(ValueError, match="evaluate"):
        record_qa_response(
            workspace,
            "mock-2",
            slot["question_id"],
            responder=slot["assigned_responder"],
            answer="Answer",
            duration_seconds=30,
            scores={"clarity": 4, "evidence": 4, "client_fit": 4},
            evaluator=slot["assigned_responder"],
            now=NOW,
        )
    with pytest.raises(ValueError, match="unanswered"):
        complete_mock_round(workspace, "mock-2", completed_by="Cyril", now=NOW)

