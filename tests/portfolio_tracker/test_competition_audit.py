from datetime import datetime, timezone
import sqlite3

import pytest

from src.portfolio_tracker.competition_audit import (
    append_ai_usage,
    append_red_team_review,
    init_competition_audit_tables,
    list_ai_usage,
    list_red_team_reviews,
)


def test_competition_audit_is_append_only_and_filterable():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    init_competition_audit_tables(connection)
    now = datetime(2026, 7, 31, tzinfo=timezone.utc)

    ai = append_ai_usage(
        connection,
        "Brainstorm downside questions",
        "ChatGPT",
        prompt_summary="Challenge the thesis without writing submission prose.",
        output_used="Two questions added to the research checklist.",
        verification_notes="Checked against primary filings.",
        citation="Team AI usage log, 2026-07-31",
        recorded_by="analyst",
        now=now,
    )
    review = append_red_team_review(
        connection,
        ticker="abc",
        strongest_counterargument="Margins are cyclical, not structural.",
        disconfirming_evidence="Peer margins normalized faster.",
        rejected_alternative="Broad sector ETF",
        verdict="revise",
        reviewed_by="reviewer",
        now=now,
    )

    assert ai["id"] == 1
    assert review["ticker"] == "ABC"
    assert list_ai_usage(connection)[0]["tool_name"] == "ChatGPT"
    assert list_red_team_reviews(connection, ticker="ABC")[0]["verdict"] == "revise"


def test_red_team_requires_a_target_and_valid_verdict():
    connection = sqlite3.connect(":memory:")
    with pytest.raises(ValueError, match="ticker or decision"):
        append_red_team_review(
            connection,
            strongest_counterargument="Counter",
            reviewed_by="reviewer",
        )
    with pytest.raises(ValueError, match="Verdict"):
        append_red_team_review(
            connection,
            ticker="ABC",
            strongest_counterargument="Counter",
            verdict="maybe",
            reviewed_by="reviewer",
        )
