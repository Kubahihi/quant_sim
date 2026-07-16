from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import sqlite3

import pytest

from src.portfolio_tracker.governance_store import (
    append_decision_review,
    append_thesis_review,
    delete_catalyst_event,
    delete_research_source,
    get_catalyst_event,
    get_decision_review,
    get_research_source,
    get_thesis_review,
    init_governance_tables,
    list_catalyst_events,
    list_decision_reviews,
    list_research_sources,
    list_thesis_reviews,
    upsert_catalyst_event,
    upsert_research_source,
)


NOW = datetime(2026, 7, 16, 10, 30, tzinfo=timezone.utc)
LATER = NOW + timedelta(hours=3)


def test_schema_initialization_creates_all_governance_tables_and_indices():
    connection = sqlite3.connect(":memory:")

    init_governance_tables(connection)

    objects = {
        row[0]: row[1]
        for row in connection.execute(
            """
            SELECT name, type FROM sqlite_master
            WHERE name LIKE 'analytical_%' OR name LIKE 'idx_analytical_%'
            """
        ).fetchall()
    }
    assert objects["analytical_thesis_reviews"] == "table"
    assert objects["analytical_decision_reviews"] == "table"
    assert objects["analytical_catalyst_events"] == "table"
    assert objects["analytical_research_sources"] == "table"
    assert objects["idx_analytical_thesis_reviews_ticker_time"] == "index"
    assert objects["idx_analytical_decision_reviews_decision_time"] == "index"
    assert objects["idx_analytical_catalysts_status_date"] == "index"
    assert objects["idx_analytical_research_sources_type"] == "index"


def test_thesis_reviews_are_appended_with_complete_before_and_after_snapshots():
    connection = sqlite3.connect(":memory:")
    first = append_thesis_review(
        connection,
        " msft ",
        {"decision": "retain", "evidence_source_ids": [1, 2]},
        prior_status="watch",
        new_status="under review",
        prior_conviction=3,
        new_conviction=2.5,
        prior_snapshot={"thesis": "Cloud scale", "risks": ["Competition"]},
        new_snapshot={"thesis": "Cloud and AI scale", "risks": ["Competition"]},
        reviewed_by="Anna",
        now=NOW,
    )
    second = append_thesis_review(
        connection,
        "MSFT",
        {"decision": "reactivate"},
        prior_status="under_review",
        new_status="active",
        prior_conviction=2.5,
        new_conviction=4,
        prior_snapshot=first["new_snapshot"],
        new_snapshot={"thesis": "Evidence confirmed"},
        reviewed_by="Matej",
        now=LATER,
    )

    assert first["ticker"] == "MSFT"
    assert first["new_status"] == "under_review"
    assert first["prior_conviction"] == pytest.approx(3)
    assert first["new_conviction"] == pytest.approx(2.5)
    assert first["payload"]["evidence_source_ids"] == [1, 2]
    assert first["prior_snapshot"]["thesis"] == "Cloud scale"
    assert first["new_snapshot"]["thesis"] == "Cloud and AI scale"
    assert first["reviewed_by"] == "Anna"
    assert first["reviewed_at"] == NOW.isoformat()
    assert get_thesis_review(connection, first["id"]) == first
    assert [item["id"] for item in list_thesis_reviews(connection, ticker="msft")] == [
        second["id"],
        first["id"],
    ]
    assert list_thesis_reviews(connection, new_status="active") == [second]
    assert connection.execute("SELECT COUNT(*) FROM analytical_thesis_reviews").fetchone()[0] == 2


def test_decision_reviews_keep_process_and_market_outcomes_separate():
    connection = sqlite3.connect(":memory:")
    process_only = append_decision_review(
        connection,
        12,
        "nvda",
        {"rules_followed": True, "lesson": "Sizing was disciplined"},
        process_outcome="confirmed",
        reviewed_by="Anna",
        now=NOW,
    )
    completed = append_decision_review(
        connection,
        12,
        "NVDA",
        {
            "ticker_return": -0.04,
            "benchmark_return": -0.08,
            "active_return": 0.04,
            "calculation_as_of": "2026-07-16",
        },
        process_outcome="mixed",
        market_outcome="win",
        reviewed_by="Matej",
        now=LATER,
    )

    assert process_only["market_outcome"] is None
    assert completed["process_outcome"] == "mixed"
    assert completed["market_outcome"] == "win"
    assert completed["payload"]["active_return"] == pytest.approx(0.04)
    assert get_decision_review(connection, completed["id"]) == completed
    assert list_decision_reviews(connection, decision_id=12) == [completed, process_only]
    assert list_decision_reviews(connection, ticker="nvda", process_outcome="mixed") == [completed]


def test_research_source_and_catalyst_crud_preserve_creation_audit_and_links():
    connection = sqlite3.connect(":memory:")
    source = upsert_research_source(
        connection,
        "Form 10-K for fiscal 2025",
        ticker=" aapl ",
        publisher="Apple Inc.",
        url="https://www.sec.gov/example/10-k",
        source_type="regulatory filing",
        primary_source=True,
        published_at="2026-01-30",
        period_end=date(2025, 9, 27),
        accessed_at="2026-07-16",
        verified_by="Anna",
        notes="Primary filing used for segment evidence.",
        payload={"claims": ["services margin"]},
        updated_by="Anna",
        now=NOW,
    )
    event = upsert_catalyst_event(
        connection,
        "aapl",
        "September product launch",
        {"expected_effect": "Revenue mix improves"},
        window_start="2026-09-08",
        date_confidence="exact",
        probability=4,
        impact=3,
        source_id=source["id"],
        updated_by="Anna",
        now=NOW,
    )
    updated = upsert_catalyst_event(
        connection,
        "AAPL",
        "September product launch window",
        {"expected_effect": "Revenue mix improves", "actual_result": "Pending"},
        event_id=event["id"],
        window_start="2026-09-08",
        window_end="2026-09-10",
        date_confidence="window",
        probability=5,
        impact=4,
        status="delayed",
        source_id=source["id"],
        updated_by="Matej",
        now=LATER,
    )

    assert source["ticker"] == "AAPL"
    assert source["source_type"] == "regulatory_filing"
    assert source["primary_source"] is True
    assert source["period_end"] == "2025-09-27"
    assert source["verified_at"] == NOW.isoformat()
    assert event["window_end"] == event["window_start"] == "2026-09-08"
    assert updated["id"] == event["id"]
    assert updated["created_by"] == "Anna"
    assert updated["created_at"] == NOW.isoformat()
    assert updated["updated_by"] == "Matej"
    assert updated["updated_at"] == LATER.isoformat()
    assert updated["source_id"] == source["id"]
    assert get_catalyst_event(connection, event["id"]) == updated
    assert list_catalyst_events(connection, ticker="aapl", status="delayed") == [updated]

    assert delete_research_source(
        connection,
        source["id"],
        updated_by="Petra",
        now=LATER + timedelta(hours=1),
    )
    detached = get_catalyst_event(connection, event["id"])
    assert detached["source_id"] is None
    assert detached["updated_by"] == "Petra"
    assert get_research_source(connection, source["id"]) is None
    assert not delete_research_source(connection, source["id"])
    assert delete_catalyst_event(connection, event["id"])
    assert not delete_catalyst_event(connection, event["id"])


def test_catalyst_calendar_filters_overlapping_windows_and_sorts_unknown_dates_last():
    connection = sqlite3.connect(":memory:")
    late = upsert_catalyst_event(
        connection,
        "AAA",
        "Late event",
        window_start="2026-10-10",
        window_end="2026-10-20",
        date_confidence="window",
        probability=2,
        impact=-2,
        now=NOW,
    )
    early = upsert_catalyst_event(
        connection,
        "BBB",
        "Early event",
        window_start="2026-08-01",
        date_confidence="estimated",
        probability=3,
        impact=1,
        now=NOW,
    )
    unknown = upsert_catalyst_event(
        connection,
        "CCC",
        "Undated regulatory outcome",
        date_confidence="unknown",
        probability=1,
        impact=-5,
        now=NOW,
    )

    assert [row["id"] for row in list_catalyst_events(connection)] == [
        early["id"],
        late["id"],
        unknown["id"],
    ]
    assert list_catalyst_events(
        connection,
        window_from="2026-10-15",
        window_to="2026-10-16",
    ) == [late]
    assert list_catalyst_events(connection, window_to="2026-08-31") == [early]


def test_research_source_filters_support_ticker_specific_and_global_evidence():
    connection = sqlite3.connect(":memory:")
    global_source = upsert_research_source(
        connection,
        "World Bank methodology",
        publisher="World Bank",
        url="https://data.worldbank.org/about/get-started",
        source_type="official data",
        primary_source=True,
        accessed_at="2026-07-15",
        updated_by="Anna",
        now=NOW,
    )
    verified = upsert_research_source(
        connection,
        "AAPL annual filing",
        ticker="AAPL",
        source_type="annual_report",
        primary_source=True,
        accessed_at="2026-07-16",
        verified_by="Matej",
        verified_at="2026-07-16T09:00:00+00:00",
        now=NOW,
    )
    unverified = upsert_research_source(
        connection,
        "AAPL press coverage",
        ticker="AAPL",
        source_type="news",
        accessed_at="2026-07-14",
        now=NOW,
    )
    upsert_research_source(
        connection,
        "MSFT source",
        ticker="MSFT",
        source_type="website",
        now=NOW,
    )

    assert list_research_sources(connection, ticker="aapl") == [verified, unverified]
    assert list_research_sources(connection, ticker="AAPL", include_global=True) == [
        verified,
        global_source,
        unverified,
    ]
    assert list_research_sources(connection, ticker="AAPL", verified=True) == [verified]
    assert list_research_sources(connection, ticker="AAPL", verified=False) == [unverified]
    assert list_research_sources(connection, source_type="official_data") == [global_source]
    assert list_research_sources(connection, primary_source=True) == [verified, global_source]


def test_upsert_with_explicit_new_ids_is_supported_for_replica_imports():
    connection = sqlite3.connect(":memory:")
    source = upsert_research_source(
        connection,
        "Imported source",
        source_id=40,
        source_type="other",
        now=NOW,
    )
    event = upsert_catalyst_event(
        connection,
        "TSM",
        "Imported event",
        event_id=70,
        date_confidence="unknown",
        probability=3,
        impact=2,
        source_id=source["id"],
        now=NOW,
    )

    assert source["id"] == 40
    assert event["id"] == 70


@pytest.mark.parametrize(
    "operation",
    [
        lambda connection: append_thesis_review(
            connection, "AAPL", {}, prior_status="unknown", new_status="active", reviewed_by="Anna"
        ),
        lambda connection: append_thesis_review(
            connection, "AAPL", {}, prior_status="active", new_status="active", reviewed_by=""
        ),
        lambda connection: append_thesis_review(
            connection,
            "AAPL",
            {},
            prior_status="active",
            new_status="active",
            new_conviction=float("inf"),
            reviewed_by="Anna",
        ),
        lambda connection: append_decision_review(
            connection, 0, "AAPL", {}, process_outcome="confirmed", reviewed_by="Anna"
        ),
        lambda connection: append_decision_review(
            connection, 1, "AAPL", {}, process_outcome="lucky", reviewed_by="Anna"
        ),
        lambda connection: append_decision_review(
            connection,
            1,
            "AAPL",
            {},
            process_outcome="mixed",
            market_outcome="outperformed",
            reviewed_by="Anna",
        ),
        lambda connection: upsert_catalyst_event(
            connection, "AAPL", "Event", probability=0, impact=1
        ),
        lambda connection: upsert_catalyst_event(
            connection, "AAPL", "Event", probability=2.5, impact=1
        ),
        lambda connection: upsert_catalyst_event(
            connection, "AAPL", "Event", probability=3, impact=6
        ),
        lambda connection: upsert_catalyst_event(
            connection, "AAPL", "Event", date_confidence="certain"
        ),
        lambda connection: upsert_catalyst_event(
            connection,
            "AAPL",
            "Event",
            window_start="2026-08-02",
            window_end="2026-08-01",
            date_confidence="window",
        ),
        lambda connection: upsert_catalyst_event(
            connection,
            "AAPL",
            "Event",
            window_start="2026-08-01",
            window_end="2026-08-02",
            date_confidence="exact",
        ),
        lambda connection: upsert_research_source(
            connection, "Source", source_type="social_media"
        ),
        lambda connection: upsert_research_source(
            connection, "Source", url="ftp://example.com/file"
        ),
        lambda connection: upsert_research_source(
            connection, "Source", primary_source=1
        ),
        lambda connection: upsert_research_source(
            connection, "Source", verified_at="2026-07-16", verified_by=""
        ),
        lambda connection: upsert_research_source(
            connection, "Source", published_at="16/07/2026"
        ),
    ],
)
def test_invalid_enums_ranges_dates_and_audit_fields_are_rejected(operation):
    connection = sqlite3.connect(":memory:")
    with pytest.raises(ValueError):
        operation(connection)


@pytest.mark.parametrize(
    "operation",
    [
        lambda connection: append_thesis_review(
            connection,
            "AAPL",
            {"bad": float("nan")},
            prior_status="active",
            new_status="watch",
            reviewed_by="Anna",
        ),
        lambda connection: append_thesis_review(
            connection,
            "AAPL",
            {},
            prior_status="active",
            new_status="watch",
            prior_snapshot=["not", "an", "object"],
            reviewed_by="Anna",
        ),
        lambda connection: append_decision_review(
            connection,
            1,
            "AAPL",
            {"bad": object()},
            process_outcome="confirmed",
            reviewed_by="Anna",
        ),
        lambda connection: upsert_catalyst_event(
            connection, "AAPL", "Event", ["not", "an", "object"]
        ),
        lambda connection: upsert_research_source(
            connection, "Source", payload={"bad": float("-inf")}
        ),
    ],
)
def test_every_json_payload_and_snapshot_rejects_non_objects_or_non_finite_values(operation):
    connection = sqlite3.connect(":memory:")
    with pytest.raises(ValueError):
        operation(connection)


def test_corrupt_json_rows_are_not_returned_or_listed():
    connection = sqlite3.connect(":memory:")
    review = append_decision_review(
        connection,
        1,
        "AAPL",
        {"lesson": "test"},
        process_outcome="confirmed",
        reviewed_by="Anna",
        now=NOW,
    )
    source = upsert_research_source(connection, "Source", now=NOW)
    connection.execute(
        "UPDATE analytical_decision_reviews SET payload_json = '{broken' WHERE id = ?",
        (review["id"],),
    )
    connection.execute(
        "UPDATE analytical_research_sources SET payload_json = '[]' WHERE id = ?",
        (source["id"],),
    )
    connection.commit()

    assert get_decision_review(connection, review["id"]) is None
    assert list_decision_reviews(connection) == []
    assert get_research_source(connection, source["id"]) is None
    assert list_research_sources(connection) == []


def test_plain_tuples_and_sqlite_rows_are_both_supported():
    tuple_connection = sqlite3.connect(":memory:")
    tuple_source = upsert_research_source(
        tuple_connection,
        "Tuple source",
        ticker="TSM",
        now=NOW,
    )
    assert get_research_source(tuple_connection, tuple_source["id"])["ticker"] == "TSM"

    row_connection = sqlite3.connect(":memory:")
    row_connection.row_factory = sqlite3.Row
    row_review = append_thesis_review(
        row_connection,
        "TSM",
        {},
        prior_status="watch",
        new_status="active",
        reviewed_by="Anna",
        now=NOW,
    )
    assert get_thesis_review(row_connection, row_review["id"])["ticker"] == "TSM"


class _SyncingConnection:
    def __init__(self) -> None:
        self.connection = sqlite3.connect(":memory:")
        self.commit_calls = 0
        self.sync_calls = 0

    def execute(self, *args, **kwargs):
        return self.connection.execute(*args, **kwargs)

    def commit(self) -> None:
        self.commit_calls += 1
        self.connection.commit()

    def sync(self) -> None:
        self.sync_calls += 1


def test_schema_and_every_mutation_commit_and_sync_exactly_once():
    connection = _SyncingConnection()

    init_governance_tables(connection)
    thesis_review = append_thesis_review(
        connection,
        "AAPL",
        {},
        prior_status="watch",
        new_status="active",
        reviewed_by="Anna",
        now=NOW,
    )
    decision_review = append_decision_review(
        connection,
        1,
        "AAPL",
        {},
        process_outcome="confirmed",
        reviewed_by="Anna",
        now=NOW,
    )
    source = upsert_research_source(connection, "Source", updated_by="Anna", now=NOW)
    event = upsert_catalyst_event(
        connection,
        "AAPL",
        "Event",
        source_id=source["id"],
        updated_by="Anna",
        now=NOW,
    )
    upsert_research_source(
        connection,
        "Updated source",
        source_id=source["id"],
        updated_by="Matej",
        now=LATER,
    )
    upsert_catalyst_event(
        connection,
        "AAPL",
        "Updated event",
        event_id=event["id"],
        updated_by="Matej",
        now=LATER,
    )
    delete_research_source(connection, source["id"], updated_by="Matej", now=LATER)
    delete_catalyst_event(connection, event["id"])

    assert thesis_review["id"] > 0
    assert decision_review["id"] > 0
    assert connection.commit_calls == 9
    assert connection.sync_calls == 9
