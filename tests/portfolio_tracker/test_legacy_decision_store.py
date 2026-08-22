from __future__ import annotations

import sqlite3

import pytest

from src.portfolio_tracker.legacy_decision_store import delete_legacy_decision


def _connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE canonical_investment_lifecycles "
        "(id INTEGER PRIMARY KEY, ticker TEXT)"
    )
    connection.execute("CREATE TABLE decision_log (id INTEGER PRIMARY KEY, thesis TEXT)")
    connection.execute(
        "CREATE TABLE decision_edit_log "
        "(id INTEGER PRIMARY KEY, decision_id INTEGER NOT NULL, edited_by TEXT)"
    )
    connection.execute(
        "CREATE TABLE analytical_decision_reviews "
        "(id INTEGER PRIMARY KEY, decision_id INTEGER NOT NULL, reviewed_by TEXT)"
    )
    return connection


def test_delete_legacy_decision_removes_its_dependent_history_only():
    connection = _connection()
    connection.execute(
        "INSERT INTO canonical_investment_lifecycles (id, ticker) VALUES (1, 'AAPL')"
    )
    connection.executemany(
        "INSERT INTO decision_log (id, thesis) VALUES (?, ?)",
        [(1, "Old"), (2, "Keep")],
    )
    connection.executemany(
        "INSERT INTO decision_edit_log (id, decision_id, edited_by) VALUES (?, ?, ?)",
        [(1, 1, "Anna"), (2, 2, "Matej")],
    )
    connection.executemany(
        "INSERT INTO analytical_decision_reviews (id, decision_id, reviewed_by) VALUES (?, ?, ?)",
        [(1, 1, "Anna"), (2, 2, "Matej")],
    )

    assert delete_legacy_decision(connection, 1)
    assert connection.execute("SELECT id FROM decision_log ORDER BY id").fetchall() == [(2,)]
    assert connection.execute("SELECT decision_id FROM decision_edit_log").fetchall() == [(2,)]
    assert connection.execute("SELECT decision_id FROM analytical_decision_reviews").fetchall() == [(2,)]
    assert connection.execute(
        "SELECT id, ticker FROM canonical_investment_lifecycles"
    ).fetchall() == [(1, "AAPL")]


def test_delete_legacy_decision_rejects_invalid_id_and_missing_record_is_safe():
    connection = _connection()

    with pytest.raises(ValueError, match="positive"):
        delete_legacy_decision(connection, 0)

    assert not delete_legacy_decision(connection, 999)
