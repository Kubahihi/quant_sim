from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

from src.auth import database


def test_remote_sync_is_throttled_per_database(monkeypatch):
    monkeypatch.setenv("TURSO_SYNC_INTERVAL_SECONDS", "30")
    database._LAST_REMOTE_SYNC_BY_DATABASE.clear()
    connection = Mock()

    with patch.object(database.time, "monotonic", side_effect=[100.0, 100.0, 110.0, 131.0, 131.0]):
        database._sync_remote_if_due(connection, "portfolio.db")
        database._sync_remote_if_due(connection, "portfolio.db")
        database._sync_remote_if_due(connection, "portfolio.db")

    assert connection.sync.call_count == 2


def test_remote_sync_throttle_is_independent_per_database(monkeypatch):
    monkeypatch.setenv("TURSO_SYNC_INTERVAL_SECONDS", "30")
    database._LAST_REMOTE_SYNC_BY_DATABASE.clear()
    connection = Mock()

    with patch.object(database.time, "monotonic", side_effect=[100.0, 100.0, 101.0, 101.0]):
        database._sync_remote_if_due(connection, "auth.db")
        database._sync_remote_if_due(connection, "portfolio.db")

    assert connection.sync.call_count == 2


def test_auth_schema_initialization_is_memoized(monkeypatch, tmp_path):
    database._INITIALIZED_AUTH_DATABASES.clear()
    monkeypatch.setattr(database, "AUTH_DB_PATH", tmp_path / "auth.db")
    connection = Mock()
    get_connection = Mock(return_value=connection)
    monkeypatch.setattr(database, "_get_connection", get_connection)

    database.init_auth_database()
    database.init_auth_database()

    assert get_connection.call_count == 1
    assert connection.executescript.call_count == 1


def test_recent_session_lookup_does_not_write_heartbeat(monkeypatch):
    now = datetime.now(timezone.utc)
    row = {
        "id": 1,
        "username": "user",
        "email": "user@example.com",
        "created_at": now.isoformat(),
        "_last_accessed": now.isoformat(),
    }
    cursor = Mock()
    cursor.fetchone.return_value = row
    connection = Mock()
    connection.execute.return_value = cursor
    monkeypatch.setattr(database, "_get_connection", lambda: connection)

    user = database.get_user_by_session_token("token")

    assert user is not None
    assert "_last_accessed" not in user
    assert connection.execute.call_count == 1
    connection.commit.assert_not_called()


def test_stale_session_lookup_persists_one_heartbeat(monkeypatch):
    now = datetime.now(timezone.utc)
    row = {
        "id": 1,
        "username": "user",
        "email": "user@example.com",
        "created_at": now.isoformat(),
        "_last_accessed": (now - timedelta(minutes=10)).isoformat(),
    }
    cursor = Mock()
    cursor.fetchone.return_value = row
    connection = Mock()
    connection.execute.side_effect = [cursor, Mock()]
    monkeypatch.setattr(database, "_get_connection", lambda: connection)

    database.get_user_by_session_token("token")

    assert connection.execute.call_count == 2
    connection.commit.assert_called_once()
    connection.sync.assert_called_once()
