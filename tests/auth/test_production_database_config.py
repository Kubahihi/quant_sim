from __future__ import annotations

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
