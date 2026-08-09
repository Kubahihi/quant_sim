from __future__ import annotations

import sqlite3

import pytest

from src.auth import database, manager


RATE_LIMIT_ERROR = "Too many failed attempts. Please try again in 10 minutes."
INVALID_CREDENTIALS_ERROR = "Invalid username or password"


@pytest.fixture
def isolated_auth_database(monkeypatch, tmp_path):
    database._INITIALIZED_AUTH_DATABASES.clear()
    database._LAST_REMOTE_SYNC_BY_DATABASE.clear()
    database_path = tmp_path / "auth.db"
    monkeypatch.setattr(database, "AUTH_DB_PATH", database_path)
    database.init_auth_database()
    yield database_path
    database._INITIALIZED_AUTH_DATABASES.clear()
    database._LAST_REMOTE_SYNC_BY_DATABASE.clear()


def _register_user(username: str = "login_target") -> None:
    user, errors = manager.register_user(
        username,
        f"{username}@example.com",
        "CorrectPass123",
        "CorrectPass123",
    )
    assert errors == []
    assert user is not None


def test_attacker_address_cannot_lock_account_for_another_address(
    isolated_auth_database,
) -> None:
    _register_user()
    attacker_address = "198.51.100.10"
    legitimate_address = "203.0.113.20"

    for _ in range(5):
        token, user, errors = manager.login_user(
            "login_target",
            "WrongPass456",
            ip_address=attacker_address,
        )
        assert token is None
        assert user is None
        assert errors == [INVALID_CREDENTIALS_ERROR]

    blocked_token, blocked_user, blocked_errors = manager.login_user(
        "login_target",
        "CorrectPass123",
        ip_address=attacker_address,
    )
    assert blocked_token is None
    assert blocked_user is None
    assert blocked_errors == [RATE_LIMIT_ERROR]

    token, user, errors = manager.login_user(
        "login_target",
        "CorrectPass123",
        ip_address=legitimate_address,
    )
    assert errors == []
    assert token
    assert user is not None
    assert user["username"] == "login_target"

    # A success elsewhere must not clear the attacker's throttle state.
    _, _, attacker_errors_after_success = manager.login_user(
        "login_target",
        "CorrectPass123",
        ip_address=attacker_address,
    )
    assert attacker_errors_after_success == [RATE_LIMIT_ERROR]


def test_success_resets_only_failures_for_the_same_account_and_address(
    isolated_auth_database,
) -> None:
    _register_user()
    client_address = "192.0.2.40"

    for _ in range(4):
        manager.login_user(
            "login_target",
            "WrongPass456",
            ip_address=client_address,
        )

    token, _, errors = manager.login_user(
        "login_target",
        "CorrectPass123",
        ip_address=client_address,
    )
    assert token
    assert errors == []

    _, _, errors = manager.login_user(
        "login_target",
        "WrongPass456",
        ip_address=client_address,
    )
    assert errors == [INVALID_CREDENTIALS_ERROR]

    with sqlite3.connect(isolated_auth_database) as connection:
        stored_addresses = connection.execute(
            "SELECT DISTINCT ip_address FROM login_attempts"
        ).fetchall()
    assert stored_addresses == [(client_address,)]
    assert database.get_recent_failed_attempts(
        "login_target",
        ip_address=client_address,
    ) == 1


def test_one_address_is_limited_across_many_account_names(
    isolated_auth_database,
) -> None:
    client_address = "192.0.2.50"

    for attempt in range(database.DEFAULT_MAXIMUM_IP_FAILED_ATTEMPTS):
        _, _, errors = manager.login_user(
            f"unknown_user_{attempt}",
            "WrongPass456",
            ip_address=client_address,
        )
        assert errors == [INVALID_CREDENTIALS_ERROR]

    _, _, errors = manager.login_user(
        "another_unknown_user",
        "WrongPass456",
        ip_address=client_address,
    )
    assert errors == [RATE_LIMIT_ERROR]


def test_legacy_call_without_address_cannot_lock_out_a_valid_password(
    isolated_auth_database,
) -> None:
    _register_user()

    for _ in range(5):
        _, _, errors = manager.login_user("login_target", "WrongPass456")
        assert errors == [INVALID_CREDENTIALS_ERROR]

    token, user, errors = manager.login_user("login_target", "CorrectPass123")
    assert errors == []
    assert token
    assert user is not None

    # The successful authentication resets only the relevant no-address state.
    _, _, errors = manager.login_user("login_target", "WrongPass456")
    assert errors == [INVALID_CREDENTIALS_ERROR]
    assert database.get_recent_failed_attempts("login_target") == 1
