"""Persistent analytical inputs for the Wharton investment workflow.

The store accepts any DB-API-like connection exposing ``execute`` and
``commit``.  It deliberately uses only SQLite-compatible SQL so the same code
works with the local SQLite database and the application's Turso/libSQL
replica.  Callers own the connection lifecycle.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from typing import Any, Iterable, Mapping


_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS analytical_client_mandate (
        singleton_id INTEGER PRIMARY KEY,
        version INTEGER NOT NULL,
        payload_json TEXT NOT NULL,
        updated_by TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        CHECK (singleton_id = 1),
        CHECK (version > 0)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS analytical_strategy_versions (
        version INTEGER PRIMARY KEY AUTOINCREMENT,
        payload_json TEXT NOT NULL,
        created_by TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        is_active INTEGER NOT NULL DEFAULT 0,
        CHECK (is_active IN (0, 1))
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_analytical_strategy_one_active
    ON analytical_strategy_versions(is_active)
    WHERE is_active = 1
    """,
    """
    CREATE TABLE IF NOT EXISTS analytical_holding_theses (
        ticker TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        conviction REAL,
        strategy_version INTEGER,
        next_review_at TEXT,
        payload_json TEXT NOT NULL,
        updated_by TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        CHECK (strategy_version IS NULL OR strategy_version > 0)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_analytical_theses_status
    ON analytical_holding_theses(status)
    """,
    """
    CREATE TABLE IF NOT EXISTS analytical_approved_securities (
        ticker TEXT PRIMARY KEY,
        approved INTEGER NOT NULL DEFAULT 1,
        payload_json TEXT NOT NULL,
        updated_by TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        CHECK (approved IN (0, 1))
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_analytical_securities_approved
    ON analytical_approved_securities(approved)
    """,
    """
    CREATE TABLE IF NOT EXISTS analytical_company_research (
        ticker TEXT PRIMARY KEY,
        payload_json TEXT NOT NULL,
        updated_by TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_timestamp(value: datetime | None = None) -> str:
    current = value or _utc_now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc).isoformat()


def _normalise_ticker(ticker: str) -> str:
    value = str(ticker or "").strip().upper()
    if not value:
        raise ValueError("Ticker must not be empty.")
    if len(value) > 32:
        raise ValueError("Ticker must be at most 32 characters.")
    return value


def _encode_payload(payload: Mapping[str, Any]) -> str:
    if not isinstance(payload, Mapping):
        raise ValueError("Payload must be a JSON object.")
    try:
        value = deepcopy(dict(payload))
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Payload must contain only valid JSON values.") from exc
    return encoded


def _decode_payload(value: Any) -> dict[str, Any] | None:
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None
    return decoded


def _row_value(row: Any, name: str, index: int) -> Any:
    """Read sqlite3.Row, mapping, libSQL wrapper, or a plain tuple."""
    try:
        keys = row.keys()
    except (AttributeError, TypeError):
        keys = ()
    if keys:
        matching = {str(key).lower(): key for key in keys}
        actual = matching.get(name.lower())
        if actual is not None:
            try:
                return row[actual]
            except (KeyError, TypeError, IndexError):
                pass
    try:
        return row[name]
    except (KeyError, TypeError, IndexError):
        return row[index]


def _ensure_schema(connection: Any) -> None:
    for statement in _SCHEMA_STATEMENTS:
        connection.execute(statement)


def _commit_and_sync(connection: Any) -> None:
    connection.commit()
    sync = getattr(connection, "sync", None)
    if callable(sync):
        sync()


def init_strategy_tables(connection: Any) -> None:
    """Create all analytical persistence tables and sync an online replica."""
    _ensure_schema(connection)
    _commit_and_sync(connection)


def _client_mandate_record(row: Any) -> dict[str, Any] | None:
    payload = _decode_payload(_row_value(row, "payload_json", 1))
    if payload is None:
        return None
    return {
        "version": int(_row_value(row, "version", 0)),
        "payload": payload,
        "updated_by": str(_row_value(row, "updated_by", 2) or ""),
        "created_at": str(_row_value(row, "created_at", 3)),
        "updated_at": str(_row_value(row, "updated_at", 4)),
    }


def load_client_mandate(connection: Any) -> dict[str, Any] | None:
    """Return the single current mandate, or ``None`` for absent/corrupt data."""
    _ensure_schema(connection)
    row = connection.execute(
        """
        SELECT version, payload_json, updated_by, created_at, updated_at
        FROM analytical_client_mandate
        WHERE singleton_id = 1
        """
    ).fetchone()
    return None if row is None else _client_mandate_record(row)


def save_client_mandate(
    connection: Any,
    payload: Mapping[str, Any],
    *,
    updated_by: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Save the current mandate and atomically increment its edit version."""
    payload_json = _encode_payload(payload)
    timestamp = _iso_timestamp(now)
    _ensure_schema(connection)
    connection.execute(
        """
        INSERT INTO analytical_client_mandate (
            singleton_id, version, payload_json, updated_by, created_at, updated_at
        ) VALUES (1, 1, ?, ?, ?, ?)
        ON CONFLICT(singleton_id) DO UPDATE SET
            version = analytical_client_mandate.version + 1,
            payload_json = excluded.payload_json,
            updated_by = excluded.updated_by,
            updated_at = excluded.updated_at
        """,
        (payload_json, str(updated_by or "").strip(), timestamp, timestamp),
    )
    _commit_and_sync(connection)
    record = load_client_mandate(connection)
    if record is None:  # Defensive guard for non-conforming DB drivers.
        raise RuntimeError("The client mandate could not be read after saving.")
    return record


def _strategy_record(row: Any) -> dict[str, Any] | None:
    payload = _decode_payload(_row_value(row, "payload_json", 1))
    if payload is None:
        return None
    return {
        "version": int(_row_value(row, "version", 0)),
        "payload": payload,
        "created_by": str(_row_value(row, "created_by", 2) or ""),
        "created_at": str(_row_value(row, "created_at", 3)),
        "is_active": bool(int(_row_value(row, "is_active", 4) or 0)),
    }


def get_strategy_version(connection: Any, version: int) -> dict[str, Any] | None:
    _ensure_schema(connection)
    row = connection.execute(
        """
        SELECT version, payload_json, created_by, created_at, is_active
        FROM analytical_strategy_versions
        WHERE version = ?
        """,
        (int(version),),
    ).fetchone()
    return None if row is None else _strategy_record(row)


def append_strategy_version(
    connection: Any,
    payload: Mapping[str, Any],
    *,
    created_by: str = "",
    activate: bool = True,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append an immutable strategy payload and optionally make it active."""
    payload_json = _encode_payload(payload)
    timestamp = _iso_timestamp(now)
    _ensure_schema(connection)
    if activate:
        connection.execute(
            "UPDATE analytical_strategy_versions SET is_active = 0 WHERE is_active = 1"
        )
    cursor = connection.execute(
        """
        INSERT INTO analytical_strategy_versions (
            payload_json, created_by, created_at, is_active
        ) VALUES (?, ?, ?, ?)
        """,
        (
            payload_json,
            str(created_by or "").strip(),
            timestamp,
            1 if activate else 0,
        ),
    )
    version = getattr(cursor, "lastrowid", None)
    try:
        version_number = int(version)
    except (TypeError, ValueError):
        version_number = 0
    if version_number <= 0:
        row = connection.execute(
            "SELECT MAX(version) FROM analytical_strategy_versions"
        ).fetchone()
        version_number = int(_row_value(row, "MAX(version)", 0))
    _commit_and_sync(connection)
    record = get_strategy_version(connection, version_number)
    if record is None:
        raise RuntimeError("The strategy version could not be read after saving.")
    return record


def get_active_strategy_version(connection: Any) -> dict[str, Any] | None:
    """Return the active strategy version without falling back silently."""
    _ensure_schema(connection)
    row = connection.execute(
        """
        SELECT version, payload_json, created_by, created_at, is_active
        FROM analytical_strategy_versions
        WHERE is_active = 1
        ORDER BY version DESC
        LIMIT 1
        """
    ).fetchone()
    return None if row is None else _strategy_record(row)


def list_strategy_versions(connection: Any) -> list[dict[str, Any]]:
    _ensure_schema(connection)
    rows = connection.execute(
        """
        SELECT version, payload_json, created_by, created_at, is_active
        FROM analytical_strategy_versions
        ORDER BY version DESC
        """
    ).fetchall()
    return [record for row in rows if (record := _strategy_record(row)) is not None]


def set_active_strategy_version(connection: Any, version: int) -> bool:
    """Activate an existing strategy version while preserving its payload."""
    target = get_strategy_version(connection, int(version))
    if target is None:
        return False
    connection.execute(
        "UPDATE analytical_strategy_versions SET is_active = 0 WHERE is_active = 1"
    )
    connection.execute(
        "UPDATE analytical_strategy_versions SET is_active = 1 WHERE version = ?",
        (int(version),),
    )
    _commit_and_sync(connection)
    return True


def _optional_finite_number(value: float | int | None, name: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric or None.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _thesis_record(row: Any) -> dict[str, Any] | None:
    payload = _decode_payload(_row_value(row, "payload_json", 5))
    if payload is None:
        return None
    conviction_value = _row_value(row, "conviction", 2)
    strategy_value = _row_value(row, "strategy_version", 3)
    review_value = _row_value(row, "next_review_at", 4)
    return {
        "ticker": str(_row_value(row, "ticker", 0)),
        "status": str(_row_value(row, "status", 1)),
        "conviction": None if conviction_value is None else float(conviction_value),
        "strategy_version": None if strategy_value is None else int(strategy_value),
        "next_review_at": None if review_value in (None, "") else str(review_value),
        "payload": payload,
        "updated_by": str(_row_value(row, "updated_by", 6) or ""),
        "created_at": str(_row_value(row, "created_at", 7)),
        "updated_at": str(_row_value(row, "updated_at", 8)),
    }


_THESIS_SELECT = """
    SELECT ticker, status, conviction, strategy_version, next_review_at,
           payload_json, updated_by, created_at, updated_at
    FROM analytical_holding_theses
"""


def get_holding_thesis(connection: Any, ticker: str) -> dict[str, Any] | None:
    _ensure_schema(connection)
    row = connection.execute(
        _THESIS_SELECT + " WHERE ticker = ?",
        (_normalise_ticker(ticker),),
    ).fetchone()
    return None if row is None else _thesis_record(row)


def upsert_holding_thesis(
    connection: Any,
    ticker: str,
    payload: Mapping[str, Any],
    *,
    status: str = "watchlist",
    conviction: float | int | None = None,
    strategy_version: int | None = None,
    next_review_at: str | None = None,
    updated_by: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    code = _normalise_ticker(ticker)
    state = str(status or "").strip().lower()
    if not state:
        raise ValueError("Thesis status must not be empty.")
    score = _optional_finite_number(conviction, "Conviction")
    linked_version = None if strategy_version is None else int(strategy_version)
    if linked_version is not None and linked_version <= 0:
        raise ValueError("Strategy version must be positive or None.")
    payload_json = _encode_payload(payload)
    timestamp = _iso_timestamp(now)
    _ensure_schema(connection)
    connection.execute(
        """
        INSERT INTO analytical_holding_theses (
            ticker, status, conviction, strategy_version, next_review_at,
            payload_json, updated_by, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            status = excluded.status,
            conviction = excluded.conviction,
            strategy_version = excluded.strategy_version,
            next_review_at = excluded.next_review_at,
            payload_json = excluded.payload_json,
            updated_by = excluded.updated_by,
            updated_at = excluded.updated_at
        """,
        (
            code,
            state,
            score,
            linked_version,
            None if next_review_at in (None, "") else str(next_review_at),
            payload_json,
            str(updated_by or "").strip(),
            timestamp,
            timestamp,
        ),
    )
    _commit_and_sync(connection)
    record = get_holding_thesis(connection, code)
    if record is None:
        raise RuntimeError("The holding thesis could not be read after saving.")
    return record


def list_holding_theses(
    connection: Any,
    *,
    status: str | None = None,
) -> list[dict[str, Any]]:
    _ensure_schema(connection)
    if status is None:
        cursor = connection.execute(_THESIS_SELECT + " ORDER BY ticker")
    else:
        cursor = connection.execute(
            _THESIS_SELECT + " WHERE status = ? ORDER BY ticker",
            (str(status).strip().lower(),),
        )
    return [record for row in cursor.fetchall() if (record := _thesis_record(row)) is not None]


def delete_holding_thesis(connection: Any, ticker: str) -> bool:
    _ensure_schema(connection)
    cursor = connection.execute(
        "DELETE FROM analytical_holding_theses WHERE ticker = ?",
        (_normalise_ticker(ticker),),
    )
    changed = int(getattr(cursor, "rowcount", 0) or 0) > 0
    _commit_and_sync(connection)
    return changed


def _approved_security_record(row: Any) -> dict[str, Any] | None:
    payload = _decode_payload(_row_value(row, "payload_json", 2))
    if payload is None:
        return None
    return {
        "ticker": str(_row_value(row, "ticker", 0)),
        "approved": bool(int(_row_value(row, "approved", 1) or 0)),
        "payload": payload,
        "updated_by": str(_row_value(row, "updated_by", 3) or ""),
        "created_at": str(_row_value(row, "created_at", 4)),
        "updated_at": str(_row_value(row, "updated_at", 5)),
    }


_APPROVED_SELECT = """
    SELECT ticker, approved, payload_json, updated_by, created_at, updated_at
    FROM analytical_approved_securities
"""


def get_approved_security(connection: Any, ticker: str) -> dict[str, Any] | None:
    _ensure_schema(connection)
    row = connection.execute(
        _APPROVED_SELECT + " WHERE ticker = ?",
        (_normalise_ticker(ticker),),
    ).fetchone()
    return None if row is None else _approved_security_record(row)


def upsert_approved_security(
    connection: Any,
    ticker: str,
    payload: Mapping[str, Any],
    *,
    approved: bool = True,
    updated_by: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    code = _normalise_ticker(ticker)
    payload_json = _encode_payload(payload)
    timestamp = _iso_timestamp(now)
    _ensure_schema(connection)
    connection.execute(
        """
        INSERT INTO analytical_approved_securities (
            ticker, approved, payload_json, updated_by, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            approved = excluded.approved,
            payload_json = excluded.payload_json,
            updated_by = excluded.updated_by,
            updated_at = excluded.updated_at
        """,
        (
            code,
            1 if approved else 0,
            payload_json,
            str(updated_by or "").strip(),
            timestamp,
            timestamp,
        ),
    )
    _commit_and_sync(connection)
    record = get_approved_security(connection, code)
    if record is None:
        raise RuntimeError("The approved-security record could not be read after saving.")
    return record


def replace_approved_securities(
    connection: Any,
    securities: Iterable[Mapping[str, Any]],
    *,
    updated_by: str = "",
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Replace the complete approved-security universe in one transaction.

    Every item must contain ``ticker`` and ``payload`` keys and may contain an
    ``approved`` boolean (defaulting to ``True``).  The complete iterable is
    normalised and JSON-validated before any SQL is executed, so a malformed
    item cannot leave the existing universe partially replaced.  Duplicate
    tickers are rejected rather than silently applying last-write-wins logic.
    """
    try:
        candidates = list(securities)
    except TypeError as exc:
        raise ValueError("Securities must be an iterable of records.") from exc

    prepared: list[tuple[str, int, str]] = []
    seen: set[str] = set()
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, Mapping):
            raise ValueError(f"Security record {index} must be an object.")
        if "ticker" not in candidate:
            raise ValueError(f"Security record {index} must contain a ticker.")
        if "payload" not in candidate:
            raise ValueError(f"Security record {index} must contain a payload.")

        try:
            code = _normalise_ticker(candidate["ticker"])
        except ValueError as exc:
            raise ValueError(f"Security record {index}: {exc}") from exc
        if code in seen:
            raise ValueError(f"Duplicate ticker in approved universe: {code}.")

        approved = candidate.get("approved", True)
        if not isinstance(approved, bool):
            raise ValueError(
                f"Security record {index} approved flag must be a boolean."
            )
        try:
            payload_json = _encode_payload(candidate["payload"])
        except ValueError as exc:
            raise ValueError(f"Security record {index}: {exc}") from exc

        seen.add(code)
        prepared.append((code, 1 if approved else 0, payload_json))

    timestamp = _iso_timestamp(now)
    editor = str(updated_by or "").strip()
    _ensure_schema(connection)
    connection.execute("DELETE FROM analytical_approved_securities")
    for code, approved_value, payload_json in prepared:
        connection.execute(
            """
            INSERT INTO analytical_approved_securities (
                ticker, approved, payload_json, updated_by, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (code, approved_value, payload_json, editor, timestamp, timestamp),
        )
    _commit_and_sync(connection)
    return list_approved_securities(connection, approved_only=None)


def list_approved_securities(
    connection: Any,
    *,
    approved_only: bool | None = True,
) -> list[dict[str, Any]]:
    """List approved names by default; pass ``None`` to include all decisions."""
    _ensure_schema(connection)
    if approved_only is None:
        cursor = connection.execute(_APPROVED_SELECT + " ORDER BY ticker")
    else:
        cursor = connection.execute(
            _APPROVED_SELECT + " WHERE approved = ? ORDER BY ticker",
            (1 if approved_only else 0,),
        )
    return [
        record
        for row in cursor.fetchall()
        if (record := _approved_security_record(row)) is not None
    ]


def delete_approved_security(connection: Any, ticker: str) -> bool:
    _ensure_schema(connection)
    cursor = connection.execute(
        "DELETE FROM analytical_approved_securities WHERE ticker = ?",
        (_normalise_ticker(ticker),),
    )
    changed = int(getattr(cursor, "rowcount", 0) or 0) > 0
    _commit_and_sync(connection)
    return changed


def _company_research_record(row: Any) -> dict[str, Any] | None:
    payload = _decode_payload(_row_value(row, "payload_json", 1))
    if payload is None:
        return None
    return {
        "ticker": str(_row_value(row, "ticker", 0)),
        "payload": payload,
        "updated_by": str(_row_value(row, "updated_by", 2) or ""),
        "created_at": str(_row_value(row, "created_at", 3)),
        "updated_at": str(_row_value(row, "updated_at", 4)),
    }


_COMPANY_RESEARCH_SELECT = """
    SELECT ticker, payload_json, updated_by, created_at, updated_at
    FROM analytical_company_research
"""


def load_company_research(connection: Any, ticker: str) -> dict[str, Any] | None:
    _ensure_schema(connection)
    row = connection.execute(
        _COMPANY_RESEARCH_SELECT + " WHERE ticker = ?",
        (_normalise_ticker(ticker),),
    ).fetchone()
    return None if row is None else _company_research_record(row)


def save_company_research(
    connection: Any,
    ticker: str,
    payload: Mapping[str, Any],
    *,
    updated_by: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Upsert manual Porter, SWOT, peers, and other company research inputs."""
    code = _normalise_ticker(ticker)
    payload_json = _encode_payload(payload)
    timestamp = _iso_timestamp(now)
    _ensure_schema(connection)
    connection.execute(
        """
        INSERT INTO analytical_company_research (
            ticker, payload_json, updated_by, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            payload_json = excluded.payload_json,
            updated_by = excluded.updated_by,
            updated_at = excluded.updated_at
        """,
        (code, payload_json, str(updated_by or "").strip(), timestamp, timestamp),
    )
    _commit_and_sync(connection)
    record = load_company_research(connection, code)
    if record is None:
        raise RuntimeError("The company research could not be read after saving.")
    return record


def list_company_research(connection: Any) -> list[dict[str, Any]]:
    _ensure_schema(connection)
    rows = connection.execute(_COMPANY_RESEARCH_SELECT + " ORDER BY ticker").fetchall()
    return [
        record
        for row in rows
        if (record := _company_research_record(row)) is not None
    ]


def delete_company_research(connection: Any, ticker: str) -> bool:
    _ensure_schema(connection)
    cursor = connection.execute(
        "DELETE FROM analytical_company_research WHERE ticker = ?",
        (_normalise_ticker(ticker),),
    )
    changed = int(getattr(cursor, "rowcount", 0) or 0) > 0
    _commit_and_sync(connection)
    return changed


__all__ = [
    "append_strategy_version",
    "delete_approved_security",
    "delete_company_research",
    "delete_holding_thesis",
    "get_active_strategy_version",
    "get_approved_security",
    "get_holding_thesis",
    "get_strategy_version",
    "init_strategy_tables",
    "list_approved_securities",
    "list_company_research",
    "list_holding_theses",
    "list_strategy_versions",
    "load_client_mandate",
    "load_company_research",
    "replace_approved_securities",
    "save_client_mandate",
    "save_company_research",
    "set_active_strategy_version",
    "upsert_approved_security",
    "upsert_holding_thesis",
]
