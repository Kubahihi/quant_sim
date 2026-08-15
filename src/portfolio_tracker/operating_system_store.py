"""Versioned persistence primitives for the investment operating system.

The product workflows (committee, reconciliation, reports, rules and Q&A) all
need the same two guarantees: a single current document and an immutable audit
trail.  This module provides those guarantees without coupling domain logic to
Streamlit or to a specific database driver.  The SQL intentionally stays in the
SQLite subset supported by the local database and Turso/libSQL.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from typing import Any, Mapping


_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS investment_os_records (
        record_type TEXT NOT NULL,
        record_id TEXT NOT NULL,
        version INTEGER NOT NULL,
        status TEXT NOT NULL DEFAULT '',
        payload_json TEXT NOT NULL,
        payload_hash TEXT NOT NULL,
        created_by TEXT NOT NULL,
        created_at TEXT NOT NULL,
        is_current INTEGER NOT NULL DEFAULT 1,
        PRIMARY KEY (record_type, record_id, version),
        CHECK (version > 0),
        CHECK (is_current IN (0, 1))
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_investment_os_one_current
    ON investment_os_records(record_type, record_id)
    WHERE is_current = 1
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_investment_os_records_type_status
    ON investment_os_records(record_type, status, created_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS investment_os_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        aggregate_type TEXT NOT NULL,
        aggregate_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        payload_hash TEXT NOT NULL,
        actor TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_investment_os_events_aggregate
    ON investment_os_events(aggregate_type, aggregate_id, id)
    """,
)


def _utc_iso(now: datetime | None = None) -> str:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _text(value: Any, label: str, *, required: bool = True, maximum: int = 160) -> str:
    result = str(value or "").strip()
    if required and not result:
        raise ValueError(f"{label} must not be empty.")
    if len(result) > maximum:
        raise ValueError(f"{label} must be at most {maximum} characters.")
    return result


def _json(payload: Mapping[str, Any]) -> tuple[str, str]:
    if not isinstance(payload, Mapping):
        raise ValueError("Payload must be a JSON object.")
    try:
        encoded = json.dumps(
            deepcopy(dict(payload)),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Payload must contain only valid finite JSON values.") from exc
    import hashlib

    return encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _decode(value: Any) -> dict[str, Any] | None:
    try:
        result = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return result if isinstance(result, dict) else None


def _value(row: Any, key: str, index: int) -> Any:
    try:
        keys = row.keys()
    except (AttributeError, TypeError):
        keys = ()
    if keys:
        actual = {str(item).lower(): item for item in keys}.get(key.lower())
        if actual is not None:
            return row[actual]
    try:
        return row[key]
    except (KeyError, TypeError, IndexError):
        return row[index]


def _commit(connection: Any) -> None:
    connection.commit()
    sync = getattr(connection, "sync", None)
    if callable(sync):
        sync()


def init_operating_system_tables(connection: Any) -> None:
    """Create the shared version and event tables."""
    for statement in _SCHEMA:
        connection.execute(statement)
    _commit(connection)


def _ensure(connection: Any) -> None:
    for statement in _SCHEMA:
        connection.execute(statement)


_RECORD_SELECT = """
    SELECT record_type, record_id, version, status, payload_json, payload_hash,
           created_by, created_at, is_current
    FROM investment_os_records
"""


def _record(row: Any) -> dict[str, Any] | None:
    payload = _decode(_value(row, "payload_json", 4))
    if payload is None:
        return None
    return {
        "record_type": str(_value(row, "record_type", 0)),
        "record_id": str(_value(row, "record_id", 1)),
        "version": int(_value(row, "version", 2)),
        "status": str(_value(row, "status", 3) or ""),
        "payload": payload,
        "payload_hash": str(_value(row, "payload_hash", 5)),
        "created_by": str(_value(row, "created_by", 6)),
        "created_at": str(_value(row, "created_at", 7)),
        "is_current": bool(int(_value(row, "is_current", 8))),
    }


def get_current_record(connection: Any, record_type: str, record_id: str) -> dict[str, Any] | None:
    _ensure(connection)
    kind = _text(record_type, "Record type")
    identifier = _text(record_id, "Record id")
    row = connection.execute(
        _RECORD_SELECT + " WHERE record_type = ? AND record_id = ? AND is_current = 1",
        (kind, identifier),
    ).fetchone()
    return None if row is None else _record(row)


def list_current_records(
    connection: Any,
    record_type: str,
    *,
    status: str | None = None,
) -> list[dict[str, Any]]:
    _ensure(connection)
    kind = _text(record_type, "Record type")
    params: list[Any] = [kind]
    query = _RECORD_SELECT + " WHERE record_type = ? AND is_current = 1"
    if status is not None:
        query += " AND status = ?"
        params.append(_text(status, "Status", required=False))
    query += " ORDER BY created_at DESC, record_id"
    rows = connection.execute(query, tuple(params)).fetchall()
    return [item for row in rows if (item := _record(row)) is not None]


def list_record_versions(connection: Any, record_type: str, record_id: str) -> list[dict[str, Any]]:
    _ensure(connection)
    kind = _text(record_type, "Record type")
    identifier = _text(record_id, "Record id")
    rows = connection.execute(
        _RECORD_SELECT
        + " WHERE record_type = ? AND record_id = ? ORDER BY version DESC",
        (kind, identifier),
    ).fetchall()
    return [item for row in rows if (item := _record(row)) is not None]


def save_record(
    connection: Any,
    record_type: str,
    record_id: str,
    payload: Mapping[str, Any],
    *,
    actor: str,
    status: str = "",
    expected_version: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append a new immutable version and atomically make it current.

    ``expected_version`` enables optimistic concurrency.  Supplying ``0`` means
    the caller expects to create a new record; any stale browser tab therefore
    fails instead of silently overwriting a teammate's work.
    """
    _ensure(connection)
    kind = _text(record_type, "Record type")
    identifier = _text(record_id, "Record id")
    author = _text(actor, "Actor", maximum=200)
    state = _text(status, "Status", required=False, maximum=80)
    payload_json, digest = _json(payload)
    row = connection.execute(
        "SELECT version FROM investment_os_records "
        "WHERE record_type = ? AND record_id = ? AND is_current = 1",
        (kind, identifier),
    ).fetchone()
    current_version = int(_value(row, "version", 0)) if row is not None else 0
    if expected_version is not None and int(expected_version) != current_version:
        raise RuntimeError(
            f"Record changed since it was loaded (expected version {expected_version}, "
            f"current version {current_version}). Reload before saving."
        )
    next_version = current_version + 1
    timestamp = _utc_iso(now)
    if current_version:
        connection.execute(
            "UPDATE investment_os_records SET is_current = 0 "
            "WHERE record_type = ? AND record_id = ? AND is_current = 1",
            (kind, identifier),
        )
    connection.execute(
        """
        INSERT INTO investment_os_records (
            record_type, record_id, version, status, payload_json, payload_hash,
            created_by, created_at, is_current
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
        """,
        (kind, identifier, next_version, state, payload_json, digest, author, timestamp),
    )
    _commit(connection)
    result = get_current_record(connection, kind, identifier)
    if result is None:
        raise RuntimeError("The saved record could not be reloaded.")
    return result


_EVENT_SELECT = """
    SELECT id, aggregate_type, aggregate_id, event_type, payload_json,
           payload_hash, actor, created_at
    FROM investment_os_events
"""


def _event(row: Any) -> dict[str, Any] | None:
    payload = _decode(_value(row, "payload_json", 4))
    if payload is None:
        return None
    return {
        "id": int(_value(row, "id", 0)),
        "aggregate_type": str(_value(row, "aggregate_type", 1)),
        "aggregate_id": str(_value(row, "aggregate_id", 2)),
        "event_type": str(_value(row, "event_type", 3)),
        "payload": payload,
        "payload_hash": str(_value(row, "payload_hash", 5)),
        "actor": str(_value(row, "actor", 6)),
        "created_at": str(_value(row, "created_at", 7)),
    }


def append_event(
    connection: Any,
    aggregate_type: str,
    aggregate_id: str,
    event_type: str,
    payload: Mapping[str, Any],
    *,
    actor: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append an immutable workflow event."""
    _ensure(connection)
    aggregate = _text(aggregate_type, "Aggregate type")
    identifier = _text(aggregate_id, "Aggregate id")
    kind = _text(event_type, "Event type")
    author = _text(actor, "Actor", maximum=200)
    payload_json, digest = _json(payload)
    cursor = connection.execute(
        """
        INSERT INTO investment_os_events (
            aggregate_type, aggregate_id, event_type, payload_json,
            payload_hash, actor, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (aggregate, identifier, kind, payload_json, digest, author, _utc_iso(now)),
    )
    event_id = int(getattr(cursor, "lastrowid", 0) or 0)
    if event_id <= 0:
        row = connection.execute("SELECT MAX(id) FROM investment_os_events").fetchone()
        event_id = int(_value(row, "MAX(id)", 0) or 0)
    _commit(connection)
    row = connection.execute(_EVENT_SELECT + " WHERE id = ?", (event_id,)).fetchone()
    result = None if row is None else _event(row)
    if result is None:
        raise RuntimeError("The appended event could not be reloaded.")
    return result


def list_events(
    connection: Any,
    aggregate_type: str,
    aggregate_id: str,
    *,
    event_type: str | None = None,
) -> list[dict[str, Any]]:
    _ensure(connection)
    params: list[Any] = [
        _text(aggregate_type, "Aggregate type"),
        _text(aggregate_id, "Aggregate id"),
    ]
    query = _EVENT_SELECT + " WHERE aggregate_type = ? AND aggregate_id = ?"
    if event_type is not None:
        query += " AND event_type = ?"
        params.append(_text(event_type, "Event type"))
    query += " ORDER BY id"
    rows = connection.execute(query, tuple(params)).fetchall()
    return [item for row in rows if (item := _event(row)) is not None]

