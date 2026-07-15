"""Database-backed cache for versioned macro-economic snapshots.

The store is deliberately connection-agnostic so it works with both the local
SQLite fallback and the Turso/libSQL connection used by the Wharton dashboard.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from typing import Any, Mapping


MACRO_SNAPSHOT_TTL_SECONDS = 21_600


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime | None) -> datetime:
    current = value or _utc_now()
    if current.tzinfo is None:
        return current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc)


def _row_value(row: Any, name: str, index: int) -> Any:
    try:
        return row[name]
    except (KeyError, TypeError, IndexError):
        return row[index]


def _parse_timestamp(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return _as_utc(parsed)


def init_macro_snapshot_table(connection: Any) -> None:
    """Create the shared snapshot table and its expiry index if needed."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS macro_snapshots (
            economy_code TEXT NOT NULL,
            reference_year INTEGER NOT NULL,
            schema_version INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            stored_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            PRIMARY KEY (economy_code, reference_year, schema_version)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_macro_snapshots_expires_at
        ON macro_snapshots(expires_at)
        """
    )


def load_macro_snapshot(
    connection: Any,
    economy_code: str,
    reference_year: int,
    schema_version: int,
    *,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Load a valid, unexpired snapshot for the requested data contract."""
    code = str(economy_code or "").upper().strip()
    year = int(reference_year)
    version = int(schema_version)
    init_macro_snapshot_table(connection)
    row = connection.execute(
        """
        SELECT payload_json, stored_at, expires_at
        FROM macro_snapshots
        WHERE economy_code = ? AND reference_year = ? AND schema_version = ?
        """,
        (code, year, version),
    ).fetchone()
    if row is None:
        return None

    expires_at = _parse_timestamp(_row_value(row, "expires_at", 2))
    if expires_at is None or expires_at <= _as_utc(now):
        return None
    try:
        payload = json.loads(str(_row_value(row, "payload_json", 0)))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not payload.get("available"):
        return None
    if str(payload.get("economy_code") or "").upper() != code:
        return None
    if int(payload.get("reference_year") or 0) != year:
        return None

    result = deepcopy(payload)
    result["cache_info"] = {
        "origin": "database",
        "schema_version": version,
        "stored_at": str(_row_value(row, "stored_at", 1)),
        "expires_at": str(_row_value(row, "expires_at", 2)),
    }
    return result


def upsert_macro_snapshot(
    connection: Any,
    snapshot: Mapping[str, Any],
    schema_version: int,
    *,
    ttl_seconds: int = MACRO_SNAPSHOT_TTL_SECONDS,
    now: datetime | None = None,
) -> bool:
    """Persist one successful snapshot and synchronize it when using libSQL."""
    if not snapshot.get("available"):
        return False
    code = str(snapshot.get("economy_code") or "").upper().strip()
    reference_year = int(snapshot.get("reference_year") or 0)
    version = int(schema_version)
    if not code or reference_year <= 0 or version <= 0 or int(ttl_seconds) <= 0:
        return False

    stored_at = _as_utc(now)
    expires_at = stored_at + timedelta(seconds=int(ttl_seconds))
    clean_payload = deepcopy(dict(snapshot))
    clean_payload.pop("cache_info", None)
    try:
        payload_json = json.dumps(clean_payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return False

    init_macro_snapshot_table(connection)
    connection.execute(
        """
        INSERT INTO macro_snapshots (
            economy_code, reference_year, schema_version, payload_json,
            fetched_at, stored_at, expires_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(economy_code, reference_year, schema_version) DO UPDATE SET
            payload_json = excluded.payload_json,
            fetched_at = excluded.fetched_at,
            stored_at = excluded.stored_at,
            expires_at = excluded.expires_at
        """,
        (
            code,
            reference_year,
            version,
            payload_json,
            str(snapshot.get("fetched_at") or stored_at.isoformat()),
            stored_at.isoformat(),
            expires_at.isoformat(),
        ),
    )
    connection.commit()
    if hasattr(connection, "sync"):
        connection.sync()
    return True


def delete_macro_snapshot(
    connection: Any,
    economy_code: str,
    reference_year: int,
    schema_version: int,
) -> None:
    """Invalidate one cached snapshot, including its Turso replica row."""
    init_macro_snapshot_table(connection)
    connection.execute(
        """
        DELETE FROM macro_snapshots
        WHERE economy_code = ? AND reference_year = ? AND schema_version = ?
        """,
        (str(economy_code or "").upper().strip(), int(reference_year), int(schema_version)),
    )
    connection.commit()
    if hasattr(connection, "sync"):
        connection.sync()
