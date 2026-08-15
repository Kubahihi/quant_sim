from __future__ import annotations

from datetime import datetime, timezone
import sqlite3

import pytest

from src.portfolio_tracker.operating_system_store import (
    append_event,
    get_current_record,
    init_operating_system_tables,
    list_current_records,
    list_events,
    list_record_versions,
    save_record,
)


def _connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    init_operating_system_tables(connection)
    return connection


def test_records_are_versioned_and_support_optimistic_concurrency() -> None:
    connection = _connection()
    now = datetime(2026, 8, 15, 10, 0, tzinfo=timezone.utc)
    first = save_record(
        connection,
        "dossier",
        "MSFT",
        {"thesis": "Durable cash flows"},
        actor="Jakub",
        status="draft",
        expected_version=0,
        now=now,
    )
    second = save_record(
        connection,
        "dossier",
        "MSFT",
        {"thesis": "Durable cash flows", "frozen": True},
        actor="Martin",
        status="frozen",
        expected_version=1,
        now=now,
    )

    assert first["version"] == 1
    assert second["version"] == 2
    assert get_current_record(connection, "dossier", "MSFT") == second
    versions = list_record_versions(connection, "dossier", "MSFT")
    assert [item["version"] for item in versions] == [2, 1]
    assert [item["is_current"] for item in versions] == [True, False]
    assert list_current_records(connection, "dossier", status="frozen") == [second]

    with pytest.raises(RuntimeError, match="changed since it was loaded"):
        save_record(
            connection,
            "dossier",
            "MSFT",
            {"stale": True},
            actor="Jakub",
            expected_version=1,
        )


def test_events_are_append_only_and_hash_the_canonical_payload() -> None:
    connection = _connection()
    first = append_event(
        connection,
        "investment_case",
        "CASE-1",
        "pre_vote_submitted",
        {"member": "Martin", "vote": "watch"},
        actor="Martin",
    )
    second = append_event(
        connection,
        "investment_case",
        "CASE-1",
        "dissent_recorded",
        {"objection": "Valuation"},
        actor="Martin",
    )

    assert len(first["payload_hash"]) == 64
    assert second["id"] > first["id"]
    assert [event["event_type"] for event in list_events(
        connection, "investment_case", "CASE-1"
    )] == ["pre_vote_submitted", "dissent_recorded"]
    assert list_events(
        connection,
        "investment_case",
        "CASE-1",
        event_type="dissent_recorded",
    ) == [second]


def test_store_rejects_non_json_values() -> None:
    connection = _connection()
    with pytest.raises(ValueError, match="valid finite JSON"):
        save_record(
            connection,
            "dossier",
            "BAD",
            {"score": float("nan")},
            actor="Jakub",
        )

