from __future__ import annotations

from unittest.mock import Mock

from src.auth import database, manager


def _prepare_database(monkeypatch, tmp_path) -> None:
    database._INITIALIZED_AUTH_DATABASES.clear()
    database._LAST_REMOTE_SYNC_BY_DATABASE.clear()
    monkeypatch.setattr(database, "AUTH_DB_PATH", tmp_path / "auth.db")
    database.init_auth_database()


def test_registration_uses_one_database_connection(monkeypatch, tmp_path) -> None:
    _prepare_database(monkeypatch, tmp_path)
    original_get_connection = database._get_connection
    tracked_connection = Mock(side_effect=original_get_connection)
    monkeypatch.setattr(database, "_get_connection", tracked_connection)

    user, errors = manager.register_user(
        "batched_register",
        "batched-register@example.com",
        "SecurePass123",
        "SecurePass123",
    )

    assert errors == []
    assert user is not None
    assert tracked_connection.call_count == 1


def test_login_rate_limit_lookup_audit_and_session_share_one_connection(
    monkeypatch,
    tmp_path,
) -> None:
    _prepare_database(monkeypatch, tmp_path)
    user, errors = manager.register_user(
        "batched_login",
        "batched-login@example.com",
        "SecurePass123",
        "SecurePass123",
    )
    assert errors == [] and user is not None

    original_get_connection = database._get_connection
    tracked_connection = Mock(side_effect=original_get_connection)
    monkeypatch.setattr(database, "_get_connection", tracked_connection)

    token, logged_in_user, login_errors = manager.login_user(
        "batched_login",
        "SecurePass123",
    )

    assert login_errors == []
    assert token
    assert logged_in_user is not None
    assert tracked_connection.call_count == 1
