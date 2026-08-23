from __future__ import annotations

import sqlite3

import pytest

from src.auth import database
from src.utils import environment


def test_production_database_refuses_local_sqlite(monkeypatch, tmp_path):
    monkeypatch.setenv("QUANT_SIM_ENV", "production")
    monkeypatch.setattr(database, "_resolve_turso_credentials", lambda: (None, None))

    with pytest.raises(database.ProductionDatabaseConfigError, match="Turso is required"):
        database.get_db_connection(tmp_path / "auth.db")

    assert not (tmp_path / "auth.db").exists()


def test_streamlit_production_secret_also_refuses_local_sqlite(monkeypatch, tmp_path):
    monkeypatch.delenv("QUANT_SIM_ENV", raising=False)
    monkeypatch.delenv("STREAMLIT_SERVER_PORT", raising=False)
    monkeypatch.setattr(environment, "_streamlit_environment", lambda: "production")
    monkeypatch.setattr(database, "_resolve_turso_credentials", lambda: (None, None))

    with pytest.raises(database.ProductionDatabaseConfigError, match="Turso is required"):
        database.get_db_connection(tmp_path / "auth.db")

    assert not (tmp_path / "auth.db").exists()


def test_partial_turso_configuration_is_rejected_in_every_environment(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("QUANT_SIM_ENV", "development")
    monkeypatch.setattr(
        database,
        "_resolve_turso_credentials",
        lambda: ("libsql://database.example", None),
    )

    with pytest.raises(database.ProductionDatabaseConfigError, match="configured together"):
        database.get_db_connection(tmp_path / "auth.db")


def test_development_database_keeps_local_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("QUANT_SIM_ENV", "development")
    monkeypatch.setattr(database, "_resolve_turso_credentials", lambda: (None, None))

    connection = database.get_db_connection(tmp_path / "auth.db")
    try:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        connection.close()


def test_local_database_configures_persistent_wal_only_once(monkeypatch, tmp_path):
    monkeypatch.setenv("QUANT_SIM_ENV", "development")
    monkeypatch.setattr(database, "_resolve_turso_credentials", lambda: (None, None))
    database._SQLITE_WAL_IDENTITIES.clear()
    real_connect = sqlite3.connect
    wal_calls = 0

    class TrackedConnection:
        def __init__(self, connection):
            self._connection = connection

        def __getattr__(self, name):
            return getattr(self._connection, name)

        def execute(self, sql, *args, **kwargs):
            nonlocal wal_calls
            if str(sql).strip().upper() == "PRAGMA JOURNAL_MODE = WAL":
                wal_calls += 1
            return self._connection.execute(sql, *args, **kwargs)

    monkeypatch.setattr(
        database.sqlite3,
        "connect",
        lambda *args, **kwargs: TrackedConnection(real_connect(*args, **kwargs)),
    )
    path = tmp_path / "auth.db"

    first = database.get_db_connection(path)
    first.close()
    second = database.get_db_connection(path)
    try:
        assert second.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        second.close()

    assert wal_calls == 1


def test_streamlit_database_settings_are_cached_but_environment_remains_live(
    monkeypatch,
):
    database._streamlit_turso_credentials.cache_clear()
    monkeypatch.setattr(
        database,
        "_streamlit_turso_credentials",
        lambda: (None, None),
    )
    monkeypatch.setenv("TURSO_DATABASE_URL", "libsql://first.example")
    monkeypatch.setenv("TURSO_AUTH_TOKEN", "first-token")

    assert database._resolve_turso_credentials() == (
        "libsql://first.example",
        "first-token",
    )

    monkeypatch.setenv("TURSO_DATABASE_URL", "libsql://second.example")
    monkeypatch.setenv("TURSO_AUTH_TOKEN", "second-token")
    assert database._resolve_turso_credentials() == (
        "libsql://second.example",
        "second-token",
    )
