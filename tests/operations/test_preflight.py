"""Production preflight and restore-drill tests."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

from src.operations import preflight


def _create_backup(path: Path, tables: set[str]) -> None:
    connection = sqlite3.connect(path)
    try:
        for table in sorted(tables):
            connection.execute(f'CREATE TABLE "{table}" (id INTEGER PRIMARY KEY)')
        connection.commit()
    finally:
        connection.close()


def test_restore_drill_verifies_copy_without_modifying_source(tmp_path):
    backup = tmp_path / "backup.db"
    _create_backup(backup, set(preflight.REQUIRED_DATABASE_TABLES))
    original = backup.read_bytes()

    result = preflight.run_restore_drill(backup)

    assert result == {"status": "healthy"}
    assert backup.read_bytes() == original


def test_restore_drill_rejects_incomplete_schema(tmp_path):
    backup = tmp_path / "backup.db"
    tables = set(preflight.REQUIRED_DATABASE_TABLES) - {"decision_log"}
    _create_backup(backup, tables)

    assert preflight.run_restore_drill(backup) == {
        "status": "unhealthy",
        "reason": "database_schema_incomplete",
    }


def test_restore_drill_rejects_corrupt_file_without_exposing_details(tmp_path):
    backup = tmp_path / "corrupt.db"
    backup.write_bytes(b"not-a-sqlite-database")

    result = preflight.run_restore_drill(backup)

    assert result == {"status": "unhealthy", "reason": "restore_check_failed"}
    assert str(backup) not in json.dumps(result)


def test_wharton_preflight_accepts_production_shared_password(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "streamlit",
        SimpleNamespace(secrets={"WHARTON_SHARED_PASSWORD": "wharton123"}),
    )

    assert preflight._check_wharton_credentials() is None


def test_config_preflight_can_pass_without_live_mutating_checks(monkeypatch):
    monkeypatch.setattr(preflight, "_check_environment", lambda: None)
    monkeypatch.setattr(preflight, "_check_api_configuration", lambda: None)
    monkeypatch.setattr(preflight, "_check_database_configuration", lambda: None)
    monkeypatch.setattr(preflight, "_check_storage_configuration", lambda: None)
    monkeypatch.setattr(preflight, "_check_wharton_credentials", lambda: None)

    result = preflight.run_production_preflight(live=False)

    assert result["ready"] is True
    assert set(result["checks"]) == {
        "environment",
        "api_configuration",
        "database_configuration",
        "storage_configuration",
        "wharton_credentials",
    }


def test_preflight_hides_exception_details_and_skips_live_checks(monkeypatch):
    secret_detail = "token=never-return-this"

    def fail_environment() -> None:
        raise RuntimeError(secret_detail)

    monkeypatch.setattr(preflight, "_check_environment", fail_environment)
    monkeypatch.setattr(preflight, "_check_api_configuration", lambda: None)
    monkeypatch.setattr(preflight, "_check_database_configuration", lambda: None)
    monkeypatch.setattr(preflight, "_check_storage_configuration", lambda: None)
    monkeypatch.setattr(preflight, "_check_wharton_credentials", lambda: None)
    monkeypatch.setattr(
        preflight,
        "_check_live_database",
        lambda: (_ for _ in ()).throw(AssertionError("must not run")),
    )

    result = preflight.run_production_preflight(live=True)

    assert result["ready"] is False
    assert result["checks"]["environment"] == {
        "status": "unhealthy",
        "reason": "check_failed",
    }
    assert result["checks"]["database"]["reason"] == "configuration_invalid"
    assert secret_detail not in json.dumps(result)
