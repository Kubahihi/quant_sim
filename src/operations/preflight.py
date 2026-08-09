"""Privacy-safe production preflight and SQLite restore verification."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Any

from src.utils.environment import resolve_environment


REQUIRED_DATABASE_TABLES = frozenset({
    "users",
    "sessions",
    "login_attempts",
    "user_data",
    "wharton_users",
    "tasks",
    "decision_log",
    "competition_positions",
    "files",
})


def _safe_check(check: Callable[[], str | None]) -> dict[str, str]:
    """Convert a check into a stable result without exposing exception text."""
    try:
        reason = check()
    except Exception:
        reason = "check_failed"
    if reason is None:
        return {"status": "healthy"}
    return {"status": "unhealthy", "reason": str(reason)}


def _check_environment() -> str | None:
    return None if resolve_environment(default=None) == "production" else "not_production"


def _check_api_configuration() -> str | None:
    from src.api.config import APIConfig

    config = APIConfig.from_yaml()
    config.validate()
    return None if config.environment == "production" else "not_production"


def _check_database_configuration() -> str | None:
    from src.auth.database import _resolve_turso_credentials

    database_url, auth_token = _resolve_turso_credentials()
    return None if database_url and auth_token else "turso_not_configured"


def _check_storage_configuration() -> str | None:
    from src.storage.backend import storage_config

    if not storage_config.load_from_secrets():
        return "r2_not_configured"
    config = storage_config.config or {}
    if config.get("backend") != "r2":
        return "r2_not_selected"
    if storage_config.validate_r2_config():
        return "r2_configuration_incomplete"
    return None


def _check_wharton_credentials() -> str | None:
    import streamlit as st

    from src.auth.wharton_credentials import validate_wharton_credentials

    try:
        raw_credentials = st.secrets["wharton_users"]
    except Exception:
        return "wharton_credentials_not_configured"
    try:
        validate_wharton_credentials(raw_credentials)
    except Exception:
        return "wharton_credentials_invalid"
    return None


def _database_tables(connection: Any) -> set[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    return {str(row[0]) for row in rows}


def _check_live_database() -> str | None:
    from src.auth.database import AUTH_DB_PATH, get_db_connection

    connection = get_db_connection(AUTH_DB_PATH)
    try:
        row = connection.execute("SELECT 1").fetchone()
        if row is None or int(row[0]) != 1:
            return "database_query_failed"
        if not REQUIRED_DATABASE_TABLES.issubset(_database_tables(connection)):
            return "database_schema_incomplete"
        return None
    finally:
        connection.close()


def _check_live_storage() -> str | None:
    from src.storage.backend import get_storage_backend

    result = get_storage_backend().health_check()
    if not isinstance(result, dict) or result.get("status") != "healthy":
        return "storage_health_failed"
    return None


def run_restore_drill(backup_path: str | Path) -> dict[str, str]:
    """Copy a SQLite backup and verify integrity/schema in read-only isolation."""
    try:
        source = Path(backup_path).expanduser().resolve(strict=True)
        if not source.is_file() or source.stat().st_size == 0:
            return {"status": "unhealthy", "reason": "backup_invalid"}

        with tempfile.TemporaryDirectory(prefix="quant-sim-restore-") as temp_dir:
            restored_path = Path(temp_dir) / "restored.db"
            shutil.copy2(source, restored_path)
            uri = f"file:{restored_path.as_posix()}?mode=ro"
            connection = sqlite3.connect(uri, uri=True)
            try:
                connection.execute("PRAGMA query_only = ON")
                integrity_rows = connection.execute("PRAGMA integrity_check").fetchall()
                if not integrity_rows or any(str(row[0]).lower() != "ok" for row in integrity_rows):
                    return {"status": "unhealthy", "reason": "integrity_check_failed"}
                if not REQUIRED_DATABASE_TABLES.issubset(_database_tables(connection)):
                    return {"status": "unhealthy", "reason": "database_schema_incomplete"}
            finally:
                connection.close()
    except Exception:
        return {"status": "unhealthy", "reason": "restore_check_failed"}

    return {"status": "healthy"}


def run_production_preflight(
    *,
    live: bool = True,
    restore_backup: str | Path | None = None,
) -> dict[str, Any]:
    """Run configuration, dependency, and optional restore checks."""
    checks = {
        "environment": _safe_check(_check_environment),
        "api_configuration": _safe_check(_check_api_configuration),
        "database_configuration": _safe_check(_check_database_configuration),
        "storage_configuration": _safe_check(_check_storage_configuration),
        "wharton_credentials": _safe_check(_check_wharton_credentials),
    }
    configuration_ready = all(
        check["status"] == "healthy" for check in checks.values()
    )

    if live:
        if configuration_ready:
            checks["database"] = _safe_check(_check_live_database)
            checks["storage"] = _safe_check(_check_live_storage)
        else:
            checks["database"] = {
                "status": "unhealthy",
                "reason": "configuration_invalid",
            }
            checks["storage"] = {
                "status": "unhealthy",
                "reason": "configuration_invalid",
            }

    if restore_backup is not None:
        checks["restore_drill"] = run_restore_drill(restore_backup)

    return {
        "ready": all(check["status"] == "healthy" for check in checks.values()),
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
    }
