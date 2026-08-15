from __future__ import annotations

import sqlite3

import pytest

import src.portfolio_tracker.investment_lifecycle_store as lifecycle_store
import src.portfolio_tracker.security_dossier_store as dossier_store
from ui.investment_os import (
    _activate_reconciled_tracker_position,
    _committee_roster,
)


TEAM = [
    {"username": "Jakub", "role": "Co-Captain"},
    {"username": "Lukas", "role": "Geopolitics"},
    {"username": "Martin", "role": "Risk"},
    {"username": "Matej", "role": "Co-Captain"},
]


def test_committee_member_ids_are_stable_when_roster_order_changes():
    first, first_ids, first_captains = _committee_roster(TEAM)
    second, second_ids, second_captains = _committee_roster(list(reversed(TEAM)))

    assert first_ids == second_ids
    assert set(first_captains) == set(second_captains)
    assert {item["vote_scope"] for item in first} == {"investment", "advisory"}
    assert {item["role"] for item in second if item["vote_scope"] == "advisory"} == {
        "clarity_reviewer",
        "client_fit_reviewer",
    }
    assert {
        item["name"]: item["role"] for item in first if item["vote_scope"] == "advisory"
    } == {
        item["name"]: item["role"] for item in second if item["vote_scope"] == "advisory"
    }


def _tracker_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE competition_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT, security_type TEXT, quantity REAL, entry_price REAL,
            entry_date TEXT, opened_by TEXT, opened_at TEXT, last_price REAL,
            notes TEXT, status TEXT, currency TEXT, lifecycle_id INTEGER,
            competition_eligibility_status TEXT, eligibility_source TEXT,
            eligibility_checked_at TEXT
        )
        """
    )
    return connection


def test_clean_lifecycle_reconciliation_creates_one_linked_tracker_position(monkeypatch):
    connection = _tracker_connection()
    lifecycle = {
        "id": 42,
        "ticker": "AAPL",
        "dossier_id": 3,
        "dossier_version": 2,
        "universe_snapshot_id": 7,
        "proposal": {"security_type": "Stock"},
        "wins_execution": {
            "execution": {
                "wins_transaction_id": "WINS-9",
                "quantity": 4,
                "average_price": 210.5,
                "executed_at": "2026-08-15T14:00:00Z",
                "currency": "USD",
            }
        },
    }
    captured = {}
    monkeypatch.setattr(
        dossier_store,
        "get_dossier_version",
        lambda *args, **kwargs: {"payload": {"asset_type": "Stock"}},
    )

    def fake_record(connection, lifecycle_id, reconciliation, *, recorded_by):
        captured.update(reconciliation)
        connection.commit()
        return {"state": "active"}

    monkeypatch.setattr(lifecycle_store, "record_wins_reconciliation", fake_record)

    result = _activate_reconciled_tracker_position(
        connection,
        lifecycle,
        {"status": "clean", "wins_snapshot_id": "wins-snapshot-1", "exceptions": []},
        actor="Jakub",
    )

    row = connection.execute("SELECT * FROM competition_positions").fetchone()
    assert result["state"] == "active"
    assert row["lifecycle_id"] == 42
    assert row["ticker"] == "AAPL"
    assert row["currency"] == "USD"
    assert captured["position_id"] == str(row["id"])

    _activate_reconciled_tracker_position(
        connection,
        lifecycle,
        {"status": "clean", "wins_snapshot_id": "wins-snapshot-2", "exceptions": []},
        actor="Jakub",
    )
    count = connection.execute("SELECT COUNT(*) FROM competition_positions").fetchone()[0]
    assert count == 1


def test_tracker_activation_fails_closed_when_execution_currency_is_missing():
    connection = _tracker_connection()
    lifecycle = {
        "id": 1,
        "ticker": "AAPL",
        "dossier_id": 1,
        "dossier_version": 1,
        "universe_snapshot_id": 1,
        "proposal": {},
        "wins_execution": {
            "execution": {
                "wins_transaction_id": "WINS-1",
                "quantity": 1,
                "average_price": 10,
                "executed_at": "2026-08-15T14:00:00Z",
            }
        },
    }

    with pytest.raises(ValueError, match="currency"):
        _activate_reconciled_tracker_position(
            connection,
            lifecycle,
            {"status": "clean", "wins_snapshot_id": "wins-1", "exceptions": []},
            actor="Jakub",
        )
