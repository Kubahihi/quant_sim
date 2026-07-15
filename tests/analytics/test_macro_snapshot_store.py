from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sqlite3

from src.analytics.macro_snapshot_store import (
    delete_macro_snapshot,
    init_macro_snapshot_table,
    load_macro_snapshot,
    upsert_macro_snapshot,
)


NOW = datetime(2026, 7, 15, 10, 0, tzinfo=timezone.utc)


def _snapshot(value: float = 2.5) -> dict:
    return {
        "available": True,
        "economy_code": "JPN",
        "economy_name": "Japan",
        "reference_year": 2024,
        "fetched_at": "2026-07-15T09:59:00+00:00",
        "indicators": {
            "FP.CPI.TOTL.ZG": {
                "label": "Inflation",
                "latest_value": value,
                "latest_year": 2024,
            }
        },
    }


def test_macro_snapshot_round_trip_adds_database_cache_metadata():
    connection = sqlite3.connect(":memory:")

    assert upsert_macro_snapshot(connection, _snapshot(), 5, ttl_seconds=3600, now=NOW)
    loaded = load_macro_snapshot(connection, "jpn", 2024, 5, now=NOW + timedelta(minutes=1))

    assert loaded is not None
    assert loaded["indicators"]["FP.CPI.TOTL.ZG"]["latest_value"] == 2.5
    assert loaded["cache_info"] == {
        "origin": "database",
        "schema_version": 5,
        "stored_at": NOW.isoformat(),
        "expires_at": (NOW + timedelta(hours=1)).isoformat(),
    }


def test_expired_or_different_schema_snapshot_is_not_reused():
    connection = sqlite3.connect(":memory:")
    upsert_macro_snapshot(connection, _snapshot(), 5, ttl_seconds=60, now=NOW)

    assert load_macro_snapshot(connection, "JPN", 2024, 5, now=NOW + timedelta(seconds=61)) is None
    assert load_macro_snapshot(connection, "JPN", 2024, 6, now=NOW) is None
    assert load_macro_snapshot(connection, "JPN", 2023, 5, now=NOW) is None


def test_upsert_replaces_payload_for_same_versioned_key():
    connection = sqlite3.connect(":memory:")
    upsert_macro_snapshot(connection, _snapshot(2.5), 5, now=NOW)
    upsert_macro_snapshot(connection, _snapshot(2.8), 5, now=NOW + timedelta(minutes=5))

    loaded = load_macro_snapshot(connection, "JPN", 2024, 5, now=NOW + timedelta(minutes=6))

    assert loaded is not None
    assert loaded["indicators"]["FP.CPI.TOTL.ZG"]["latest_value"] == 2.8
    count = connection.execute("SELECT COUNT(*) FROM macro_snapshots").fetchone()[0]
    assert count == 1


def test_unavailable_or_corrupt_snapshot_is_never_returned():
    connection = sqlite3.connect(":memory:")
    init_macro_snapshot_table(connection)
    unavailable = {**_snapshot(), "available": False}

    assert not upsert_macro_snapshot(connection, unavailable, 5, now=NOW)
    assert connection.execute("SELECT COUNT(*) FROM macro_snapshots").fetchone()[0] == 0

    upsert_macro_snapshot(connection, _snapshot(), 5, now=NOW)
    connection.execute("UPDATE macro_snapshots SET payload_json = '{broken json'")
    connection.commit()
    assert load_macro_snapshot(connection, "JPN", 2024, 5, now=NOW) is None


class _SyncingConnection:
    def __init__(self) -> None:
        self.connection = sqlite3.connect(":memory:")
        self.sync_calls = 0

    def execute(self, *args, **kwargs):
        return self.connection.execute(*args, **kwargs)

    def commit(self) -> None:
        self.connection.commit()

    def sync(self) -> None:
        self.sync_calls += 1


def test_online_connection_is_synced_after_upsert_and_invalidation():
    connection = _SyncingConnection()

    upsert_macro_snapshot(connection, _snapshot(), 5, now=NOW)
    delete_macro_snapshot(connection, "JPN", 2024, 5)

    assert connection.sync_calls == 2
    assert load_macro_snapshot(connection, "JPN", 2024, 5, now=NOW) is None
