"""Persistent governance records for the Wharton analytical workflow.

The module deliberately uses SQLite-compatible SQL and a small DB-API surface
(``execute``, ``commit`` and optional ``sync``), so the same functions work
with the local SQLite database and the application's libSQL/Turso connection.
Review records are append-only through this public API. Catalyst events and
research sources are mutable latest-state records with preserved creation
metadata and explicit update audit fields.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timezone
import json
import math
from typing import Any, Mapping
from urllib.parse import urlparse


THESIS_REVIEW_STATUSES = frozenset(
    {
        "active",
        "watch",
        "watchlist",
        "holding",
        "under_review",
        "invalidated",
        "exited",
    }
)
PROCESS_OUTCOMES = frozenset({"confirmed", "mixed", "invalidated", "not_assessed"})
MARKET_OUTCOMES = frozenset({"win", "loss", "flat", "not_assessed"})
CATALYST_DATE_CONFIDENCE = frozenset({"exact", "estimated", "window", "unknown"})
CATALYST_STATUSES = frozenset({"expected", "occurred", "delayed", "cancelled"})
RESEARCH_SOURCE_TYPES = frozenset(
    {
        "annual_report",
        "quarterly_report",
        "regulatory_filing",
        "company_release",
        "earnings_call",
        "official_data",
        "news",
        "analyst_research",
        "academic",
        "website",
        "other",
    }
)


_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS analytical_research_sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        title TEXT NOT NULL,
        publisher TEXT NOT NULL DEFAULT '',
        url TEXT NOT NULL DEFAULT '',
        source_type TEXT NOT NULL,
        primary_source INTEGER NOT NULL DEFAULT 0,
        published_at TEXT,
        period_end TEXT,
        accessed_at TEXT,
        verified_by TEXT NOT NULL DEFAULT '',
        verified_at TEXT,
        notes TEXT NOT NULL DEFAULT '',
        payload_json TEXT NOT NULL,
        created_by TEXT NOT NULL DEFAULT '',
        updated_by TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        CHECK (length(trim(title)) > 0),
        CHECK (primary_source IN (0, 1)),
        CHECK (source_type IN (
            'annual_report', 'quarterly_report', 'regulatory_filing',
            'company_release', 'earnings_call', 'official_data', 'news',
            'analyst_research', 'academic', 'website', 'other'
        ))
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_analytical_research_sources_ticker
    ON analytical_research_sources(ticker)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_analytical_research_sources_type
    ON analytical_research_sources(source_type)
    """,
    """
    CREATE TABLE IF NOT EXISTS analytical_catalyst_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        title TEXT NOT NULL,
        window_start TEXT,
        window_end TEXT,
        date_confidence TEXT NOT NULL,
        probability INTEGER NOT NULL,
        impact INTEGER NOT NULL,
        status TEXT NOT NULL,
        source_id INTEGER,
        payload_json TEXT NOT NULL,
        created_by TEXT NOT NULL DEFAULT '',
        updated_by TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (source_id) REFERENCES analytical_research_sources(id)
            ON DELETE SET NULL,
        CHECK (length(trim(ticker)) > 0),
        CHECK (length(trim(title)) > 0),
        CHECK (date_confidence IN ('exact', 'estimated', 'window', 'unknown')),
        CHECK (probability BETWEEN 1 AND 5),
        CHECK (probability = CAST(probability AS INTEGER)),
        CHECK (impact BETWEEN -5 AND 5),
        CHECK (impact = CAST(impact AS INTEGER)),
        CHECK (status IN ('expected', 'occurred', 'delayed', 'cancelled')),
        CHECK (source_id IS NULL OR source_id > 0),
        CHECK (window_start IS NULL OR window_end IS NULL OR window_end >= window_start)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_analytical_catalysts_ticker_date
    ON analytical_catalyst_events(ticker, window_start)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_analytical_catalysts_status_date
    ON analytical_catalyst_events(status, window_start)
    """,
    """
    CREATE TABLE IF NOT EXISTS analytical_thesis_reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        prior_status TEXT NOT NULL,
        new_status TEXT NOT NULL,
        prior_conviction REAL,
        new_conviction REAL,
        payload_json TEXT NOT NULL,
        prior_snapshot_json TEXT NOT NULL,
        new_snapshot_json TEXT NOT NULL,
        reviewed_by TEXT NOT NULL,
        reviewed_at TEXT NOT NULL,
        CHECK (length(trim(ticker)) > 0),
        CHECK (length(trim(reviewed_by)) > 0),
        CHECK (prior_status IN (
            'active', 'watch', 'watchlist', 'holding', 'under_review',
            'invalidated', 'exited'
        )),
        CHECK (new_status IN (
            'active', 'watch', 'watchlist', 'holding', 'under_review',
            'invalidated', 'exited'
        ))
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_analytical_thesis_reviews_ticker_time
    ON analytical_thesis_reviews(ticker, reviewed_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS analytical_decision_reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        decision_id INTEGER NOT NULL,
        ticker TEXT NOT NULL,
        process_outcome TEXT NOT NULL,
        market_outcome TEXT,
        payload_json TEXT NOT NULL,
        reviewed_by TEXT NOT NULL,
        reviewed_at TEXT NOT NULL,
        CHECK (decision_id > 0),
        CHECK (length(trim(ticker)) > 0),
        CHECK (length(trim(reviewed_by)) > 0),
        CHECK (process_outcome IN ('confirmed', 'mixed', 'invalidated', 'not_assessed')),
        CHECK (market_outcome IS NULL OR market_outcome IN ('win', 'loss', 'flat', 'not_assessed'))
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_analytical_decision_reviews_decision_time
    ON analytical_decision_reviews(decision_id, reviewed_at)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_analytical_decision_reviews_ticker_time
    ON analytical_decision_reviews(ticker, reviewed_at)
    """,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_timestamp(value: datetime | None = None) -> str:
    current = value or _utc_now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc).isoformat()


def _normalise_ticker(ticker: Any, *, optional: bool = False) -> str | None:
    value = str(ticker or "").strip().upper()
    if not value:
        if optional:
            return None
        raise ValueError("Ticker must not be empty.")
    if len(value) > 32:
        raise ValueError("Ticker must be at most 32 characters.")
    return value


def _normalise_text(
    value: Any,
    name: str,
    *,
    required: bool = False,
    max_length: int | None = None,
) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise ValueError(f"{name} must not be empty.")
    if max_length is not None and len(text) > max_length:
        raise ValueError(f"{name} must be at most {max_length} characters.")
    return text


def _normalise_enum(value: Any, name: str, allowed: frozenset[str]) -> str:
    normalised = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalised not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of: {choices}.")
    return normalised


def _normalise_positive_id(value: Any, name: str, *, optional: bool = False) -> int | None:
    if optional and value in (None, ""):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    try:
        original = float(value)
    except (TypeError, ValueError):
        original = float(number)
    if number <= 0 or not math.isfinite(original) or original != float(number):
        raise ValueError(f"{name} must be a positive integer.")
    return number


def _normalise_integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer from {minimum} through {maximum}.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer from {minimum} through {maximum}.") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{name} must be an integer from {minimum} through {maximum}.")
    integer = int(number)
    if integer < minimum or integer > maximum:
        raise ValueError(f"{name} must be an integer from {minimum} through {maximum}.")
    return integer


def _optional_finite_number(value: Any, name: str) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric or None.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric or None.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _optional_iso_date(value: Any, name: str) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO date (YYYY-MM-DD) or None.") from exc


def _optional_iso_date_or_timestamp(value: Any, name: str) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return _iso_timestamp(value)
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    try:
        if "T" in text or " " in text:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return _iso_timestamp(parsed)
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid ISO date or timestamp.") from exc


def _normalise_url(value: Any) -> str:
    url = _normalise_text(value, "URL", max_length=2048)
    if not url:
        return ""
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError("URL must be an absolute HTTP or HTTPS URL.")
    return url


def _encode_payload(payload: Mapping[str, Any] | None) -> str:
    if not isinstance(payload, Mapping):
        raise ValueError("Payload must be a JSON object.")
    try:
        return json.dumps(
            deepcopy(dict(payload)),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Payload must contain only valid finite JSON values.") from exc


def _decode_payload(value: Any) -> dict[str, Any] | None:
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return decoded if isinstance(decoded, dict) else None


def _row_value(row: Any, name: str, index: int) -> Any:
    """Read sqlite3.Row, a mapping/libSQL wrapper, or a plain tuple."""
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


def _inserted_id(connection: Any, cursor: Any, table: str) -> int:
    candidate = getattr(cursor, "lastrowid", None)
    try:
        identifier = int(candidate)
    except (TypeError, ValueError):
        identifier = 0
    if identifier <= 0:
        row = connection.execute(f"SELECT MAX(id) FROM {table}").fetchone()
        identifier = int(_row_value(row, "MAX(id)", 0) or 0)
    if identifier <= 0:
        raise RuntimeError(f"Could not determine inserted id for {table}.")
    return identifier


def init_governance_tables(connection: Any) -> None:
    """Create all governance tables and synchronize an online replica once."""
    _ensure_schema(connection)
    _commit_and_sync(connection)


_THESIS_REVIEW_SELECT = """
    SELECT id, ticker, prior_status, new_status, prior_conviction,
           new_conviction, payload_json, prior_snapshot_json,
           new_snapshot_json, reviewed_by, reviewed_at
    FROM analytical_thesis_reviews
"""


def _thesis_review_record(row: Any) -> dict[str, Any] | None:
    payload = _decode_payload(_row_value(row, "payload_json", 6))
    prior_snapshot = _decode_payload(_row_value(row, "prior_snapshot_json", 7))
    new_snapshot = _decode_payload(_row_value(row, "new_snapshot_json", 8))
    if payload is None or prior_snapshot is None or new_snapshot is None:
        return None
    prior_conviction = _row_value(row, "prior_conviction", 4)
    new_conviction = _row_value(row, "new_conviction", 5)
    return {
        "id": int(_row_value(row, "id", 0)),
        "ticker": str(_row_value(row, "ticker", 1)),
        "prior_status": str(_row_value(row, "prior_status", 2)),
        "new_status": str(_row_value(row, "new_status", 3)),
        "prior_conviction": None if prior_conviction is None else float(prior_conviction),
        "new_conviction": None if new_conviction is None else float(new_conviction),
        "payload": payload,
        "prior_snapshot": prior_snapshot,
        "new_snapshot": new_snapshot,
        "reviewed_by": str(_row_value(row, "reviewed_by", 9)),
        "reviewed_at": str(_row_value(row, "reviewed_at", 10)),
    }


def get_thesis_review(connection: Any, review_id: int) -> dict[str, Any] | None:
    _ensure_schema(connection)
    identifier = _normalise_positive_id(review_id, "Review id")
    row = connection.execute(
        _THESIS_REVIEW_SELECT + " WHERE id = ?",
        (identifier,),
    ).fetchone()
    return None if row is None else _thesis_review_record(row)


def append_thesis_review(
    connection: Any,
    ticker: str,
    payload: Mapping[str, Any],
    *,
    prior_status: str,
    new_status: str,
    prior_conviction: float | int | None = None,
    new_conviction: float | int | None = None,
    prior_snapshot: Mapping[str, Any] | None = None,
    new_snapshot: Mapping[str, Any] | None = None,
    reviewed_by: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append an immutable thesis-review event and return its stored record."""
    code = _normalise_ticker(ticker)
    before_status = _normalise_enum(prior_status, "Prior thesis status", THESIS_REVIEW_STATUSES)
    after_status = _normalise_enum(new_status, "New thesis status", THESIS_REVIEW_STATUSES)
    before_score = _optional_finite_number(prior_conviction, "Prior conviction")
    after_score = _optional_finite_number(new_conviction, "New conviction")
    reviewer = _normalise_text(reviewed_by, "Reviewer", required=True, max_length=200)
    payload_json = _encode_payload(payload)
    prior_json = _encode_payload({} if prior_snapshot is None else prior_snapshot)
    new_json = _encode_payload({} if new_snapshot is None else new_snapshot)
    reviewed_at = _iso_timestamp(now)
    _ensure_schema(connection)
    cursor = connection.execute(
        """
        INSERT INTO analytical_thesis_reviews (
            ticker, prior_status, new_status, prior_conviction, new_conviction,
            payload_json, prior_snapshot_json, new_snapshot_json,
            reviewed_by, reviewed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            code,
            before_status,
            after_status,
            before_score,
            after_score,
            payload_json,
            prior_json,
            new_json,
            reviewer,
            reviewed_at,
        ),
    )
    identifier = _inserted_id(connection, cursor, "analytical_thesis_reviews")
    _commit_and_sync(connection)
    record = get_thesis_review(connection, identifier)
    if record is None:
        raise RuntimeError("The thesis review could not be read after saving.")
    return record


def list_thesis_reviews(
    connection: Any,
    *,
    ticker: str | None = None,
    new_status: str | None = None,
) -> list[dict[str, Any]]:
    _ensure_schema(connection)
    clauses: list[str] = []
    parameters: list[Any] = []
    if ticker is not None:
        clauses.append("ticker = ?")
        parameters.append(_normalise_ticker(ticker))
    if new_status is not None:
        clauses.append("new_status = ?")
        parameters.append(_normalise_enum(new_status, "New thesis status", THESIS_REVIEW_STATUSES))
    query = _THESIS_REVIEW_SELECT
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY reviewed_at DESC, id DESC"
    rows = connection.execute(query, tuple(parameters)).fetchall()
    return [record for row in rows if (record := _thesis_review_record(row)) is not None]


_DECISION_REVIEW_SELECT = """
    SELECT id, decision_id, ticker, process_outcome, market_outcome,
           payload_json, reviewed_by, reviewed_at
    FROM analytical_decision_reviews
"""


def _decision_review_record(row: Any) -> dict[str, Any] | None:
    payload = _decode_payload(_row_value(row, "payload_json", 5))
    if payload is None:
        return None
    market = _row_value(row, "market_outcome", 4)
    return {
        "id": int(_row_value(row, "id", 0)),
        "decision_id": int(_row_value(row, "decision_id", 1)),
        "ticker": str(_row_value(row, "ticker", 2)),
        "process_outcome": str(_row_value(row, "process_outcome", 3)),
        "market_outcome": None if market in (None, "") else str(market),
        "payload": payload,
        "reviewed_by": str(_row_value(row, "reviewed_by", 6)),
        "reviewed_at": str(_row_value(row, "reviewed_at", 7)),
    }


def get_decision_review(connection: Any, review_id: int) -> dict[str, Any] | None:
    _ensure_schema(connection)
    identifier = _normalise_positive_id(review_id, "Review id")
    row = connection.execute(
        _DECISION_REVIEW_SELECT + " WHERE id = ?",
        (identifier,),
    ).fetchone()
    return None if row is None else _decision_review_record(row)


def append_decision_review(
    connection: Any,
    decision_id: int,
    ticker: str,
    payload: Mapping[str, Any],
    *,
    process_outcome: str,
    market_outcome: str | None = None,
    reviewed_by: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append a process/outcome review without mutating the original decision."""
    linked_decision = _normalise_positive_id(decision_id, "Decision id")
    code = _normalise_ticker(ticker)
    process = _normalise_enum(process_outcome, "Process outcome", PROCESS_OUTCOMES)
    market = (
        None
        if market_outcome in (None, "")
        else _normalise_enum(market_outcome, "Market outcome", MARKET_OUTCOMES)
    )
    reviewer = _normalise_text(reviewed_by, "Reviewer", required=True, max_length=200)
    payload_json = _encode_payload(payload)
    reviewed_at = _iso_timestamp(now)
    _ensure_schema(connection)
    cursor = connection.execute(
        """
        INSERT INTO analytical_decision_reviews (
            decision_id, ticker, process_outcome, market_outcome,
            payload_json, reviewed_by, reviewed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (linked_decision, code, process, market, payload_json, reviewer, reviewed_at),
    )
    identifier = _inserted_id(connection, cursor, "analytical_decision_reviews")
    _commit_and_sync(connection)
    record = get_decision_review(connection, identifier)
    if record is None:
        raise RuntimeError("The decision review could not be read after saving.")
    return record


def list_decision_reviews(
    connection: Any,
    *,
    decision_id: int | None = None,
    ticker: str | None = None,
    process_outcome: str | None = None,
) -> list[dict[str, Any]]:
    _ensure_schema(connection)
    clauses: list[str] = []
    parameters: list[Any] = []
    if decision_id is not None:
        clauses.append("decision_id = ?")
        parameters.append(_normalise_positive_id(decision_id, "Decision id"))
    if ticker is not None:
        clauses.append("ticker = ?")
        parameters.append(_normalise_ticker(ticker))
    if process_outcome is not None:
        clauses.append("process_outcome = ?")
        parameters.append(_normalise_enum(process_outcome, "Process outcome", PROCESS_OUTCOMES))
    query = _DECISION_REVIEW_SELECT
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY reviewed_at DESC, id DESC"
    rows = connection.execute(query, tuple(parameters)).fetchall()
    return [record for row in rows if (record := _decision_review_record(row)) is not None]


_CATALYST_SELECT = """
    SELECT id, ticker, title, window_start, window_end, date_confidence,
           probability, impact, status, source_id, payload_json,
           created_by, updated_by, created_at, updated_at
    FROM analytical_catalyst_events
"""


def _catalyst_record(row: Any) -> dict[str, Any] | None:
    payload = _decode_payload(_row_value(row, "payload_json", 10))
    if payload is None:
        return None
    source = _row_value(row, "source_id", 9)
    return {
        "id": int(_row_value(row, "id", 0)),
        "ticker": str(_row_value(row, "ticker", 1)),
        "title": str(_row_value(row, "title", 2)),
        "window_start": _row_value(row, "window_start", 3),
        "window_end": _row_value(row, "window_end", 4),
        "date_confidence": str(_row_value(row, "date_confidence", 5)),
        "probability": int(_row_value(row, "probability", 6)),
        "impact": int(_row_value(row, "impact", 7)),
        "status": str(_row_value(row, "status", 8)),
        "source_id": None if source is None else int(source),
        "payload": payload,
        "created_by": str(_row_value(row, "created_by", 11) or ""),
        "updated_by": str(_row_value(row, "updated_by", 12) or ""),
        "created_at": str(_row_value(row, "created_at", 13)),
        "updated_at": str(_row_value(row, "updated_at", 14)),
    }


def get_catalyst_event(connection: Any, event_id: int) -> dict[str, Any] | None:
    _ensure_schema(connection)
    identifier = _normalise_positive_id(event_id, "Catalyst event id")
    row = connection.execute(_CATALYST_SELECT + " WHERE id = ?", (identifier,)).fetchone()
    return None if row is None else _catalyst_record(row)


def upsert_catalyst_event(
    connection: Any,
    ticker: str,
    title: str,
    payload: Mapping[str, Any] | None = None,
    *,
    event_id: int | None = None,
    window_start: date | str | None = None,
    window_end: date | str | None = None,
    date_confidence: str = "unknown",
    probability: int = 3,
    impact: int = 0,
    status: str = "expected",
    source_id: int | None = None,
    updated_by: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Create or update a structured catalyst while preserving creation audit."""
    identifier = _normalise_positive_id(event_id, "Catalyst event id", optional=True)
    code = _normalise_ticker(ticker)
    event_title = _normalise_text(title, "Catalyst title", required=True, max_length=500)
    confidence = _normalise_enum(date_confidence, "Date confidence", CATALYST_DATE_CONFIDENCE)
    start = _optional_iso_date(window_start, "Window start")
    end = _optional_iso_date(window_end, "Window end")
    if end is not None and start is None:
        raise ValueError("Window start is required when window end is provided.")
    if start is not None and end is not None and end < start:
        raise ValueError("Window end must not be before window start.")
    if confidence in {"exact", "estimated"}:
        if start is None:
            raise ValueError(f"Window start is required when date confidence is {confidence}.")
        if confidence == "exact" and end is not None and end != start:
            raise ValueError("An exact catalyst date cannot span multiple days.")
        end = end or start
    elif confidence == "window" and (start is None or end is None):
        raise ValueError("Both window start and window end are required for window confidence.")
    chance = _normalise_integer(probability, "Probability", 1, 5)
    effect = _normalise_integer(impact, "Impact", -5, 5)
    state = _normalise_enum(status, "Catalyst status", CATALYST_STATUSES)
    linked_source = _normalise_positive_id(source_id, "Source id", optional=True)
    payload_json = _encode_payload({} if payload is None else payload)
    editor = _normalise_text(updated_by, "Updated by", max_length=200)
    timestamp = _iso_timestamp(now)
    _ensure_schema(connection)
    values = (
        code,
        event_title,
        start,
        end,
        confidence,
        chance,
        effect,
        state,
        linked_source,
        payload_json,
        editor,
        editor,
        timestamp,
        timestamp,
    )
    if identifier is None:
        cursor = connection.execute(
            """
            INSERT INTO analytical_catalyst_events (
                ticker, title, window_start, window_end, date_confidence,
                probability, impact, status, source_id, payload_json,
                created_by, updated_by, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )
        identifier = _inserted_id(connection, cursor, "analytical_catalyst_events")
    else:
        connection.execute(
            """
            INSERT INTO analytical_catalyst_events (
                id, ticker, title, window_start, window_end, date_confidence,
                probability, impact, status, source_id, payload_json,
                created_by, updated_by, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                ticker = excluded.ticker,
                title = excluded.title,
                window_start = excluded.window_start,
                window_end = excluded.window_end,
                date_confidence = excluded.date_confidence,
                probability = excluded.probability,
                impact = excluded.impact,
                status = excluded.status,
                source_id = excluded.source_id,
                payload_json = excluded.payload_json,
                updated_by = excluded.updated_by,
                updated_at = excluded.updated_at
            """,
            (identifier, *values),
        )
    _commit_and_sync(connection)
    record = get_catalyst_event(connection, identifier)
    if record is None:
        raise RuntimeError("The catalyst event could not be read after saving.")
    return record


def list_catalyst_events(
    connection: Any,
    *,
    ticker: str | None = None,
    status: str | None = None,
    window_from: date | str | None = None,
    window_to: date | str | None = None,
) -> list[dict[str, Any]]:
    _ensure_schema(connection)
    clauses: list[str] = []
    parameters: list[Any] = []
    if ticker is not None:
        clauses.append("ticker = ?")
        parameters.append(_normalise_ticker(ticker))
    if status is not None:
        clauses.append("status = ?")
        parameters.append(_normalise_enum(status, "Catalyst status", CATALYST_STATUSES))
    start_bound = _optional_iso_date(window_from, "Window from")
    end_bound = _optional_iso_date(window_to, "Window to")
    if start_bound is not None and end_bound is not None and end_bound < start_bound:
        raise ValueError("Window to must not be before window from.")
    if start_bound is not None:
        clauses.append("COALESCE(window_end, window_start) >= ?")
        parameters.append(start_bound)
    if end_bound is not None:
        clauses.append("window_start <= ?")
        parameters.append(end_bound)
    query = _CATALYST_SELECT
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY CASE WHEN window_start IS NULL THEN 1 ELSE 0 END, window_start, id"
    rows = connection.execute(query, tuple(parameters)).fetchall()
    return [record for row in rows if (record := _catalyst_record(row)) is not None]


def delete_catalyst_event(connection: Any, event_id: int) -> bool:
    _ensure_schema(connection)
    identifier = _normalise_positive_id(event_id, "Catalyst event id")
    existed = connection.execute(
        "SELECT 1 FROM analytical_catalyst_events WHERE id = ?",
        (identifier,),
    ).fetchone() is not None
    connection.execute("DELETE FROM analytical_catalyst_events WHERE id = ?", (identifier,))
    _commit_and_sync(connection)
    return existed


_SOURCE_SELECT = """
    SELECT id, ticker, title, publisher, url, source_type, primary_source,
           published_at, period_end, accessed_at, verified_by, verified_at,
           notes, payload_json, created_by, updated_by, created_at, updated_at
    FROM analytical_research_sources
"""


def _source_record(row: Any) -> dict[str, Any] | None:
    payload = _decode_payload(_row_value(row, "payload_json", 13))
    if payload is None:
        return None
    ticker = _row_value(row, "ticker", 1)
    return {
        "id": int(_row_value(row, "id", 0)),
        "ticker": None if ticker in (None, "") else str(ticker),
        "title": str(_row_value(row, "title", 2)),
        "publisher": str(_row_value(row, "publisher", 3) or ""),
        "url": str(_row_value(row, "url", 4) or ""),
        "source_type": str(_row_value(row, "source_type", 5)),
        "primary_source": bool(int(_row_value(row, "primary_source", 6) or 0)),
        "published_at": _row_value(row, "published_at", 7),
        "period_end": _row_value(row, "period_end", 8),
        "accessed_at": _row_value(row, "accessed_at", 9),
        "verified_by": str(_row_value(row, "verified_by", 10) or ""),
        "verified_at": _row_value(row, "verified_at", 11),
        "notes": str(_row_value(row, "notes", 12) or ""),
        "payload": payload,
        "created_by": str(_row_value(row, "created_by", 14) or ""),
        "updated_by": str(_row_value(row, "updated_by", 15) or ""),
        "created_at": str(_row_value(row, "created_at", 16)),
        "updated_at": str(_row_value(row, "updated_at", 17)),
    }


def get_research_source(connection: Any, source_id: int) -> dict[str, Any] | None:
    _ensure_schema(connection)
    identifier = _normalise_positive_id(source_id, "Source id")
    row = connection.execute(_SOURCE_SELECT + " WHERE id = ?", (identifier,)).fetchone()
    return None if row is None else _source_record(row)


def upsert_research_source(
    connection: Any,
    title: str,
    *,
    source_id: int | None = None,
    ticker: str | None = None,
    publisher: str = "",
    url: str = "",
    source_type: str = "other",
    primary_source: bool = False,
    published_at: date | str | None = None,
    period_end: date | str | None = None,
    accessed_at: date | str | None = None,
    verified_by: str = "",
    verified_at: date | datetime | str | None = None,
    notes: str = "",
    payload: Mapping[str, Any] | None = None,
    updated_by: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Create or update a structured evidence source with verification audit."""
    identifier = _normalise_positive_id(source_id, "Source id", optional=True)
    code = _normalise_ticker(ticker, optional=True)
    source_title = _normalise_text(title, "Source title", required=True, max_length=500)
    source_publisher = _normalise_text(publisher, "Publisher", max_length=300)
    source_url = _normalise_url(url)
    kind = _normalise_enum(source_type, "Source type", RESEARCH_SOURCE_TYPES)
    if not isinstance(primary_source, bool):
        raise ValueError("Primary source must be a boolean.")
    publication_date = _optional_iso_date(published_at, "Published at")
    reporting_period = _optional_iso_date(period_end, "Period end")
    access_date = _optional_iso_date(accessed_at, "Accessed at")
    verifier = _normalise_text(verified_by, "Verified by", max_length=200)
    verification_time = _optional_iso_date_or_timestamp(verified_at, "Verified at")
    timestamp = _iso_timestamp(now)
    if verifier and verification_time is None:
        verification_time = timestamp
    if verification_time is not None and not verifier:
        raise ValueError("Verified by is required when verified at is provided.")
    source_notes = _normalise_text(notes, "Notes", max_length=20_000)
    payload_json = _encode_payload({} if payload is None else payload)
    editor = _normalise_text(updated_by, "Updated by", max_length=200)
    _ensure_schema(connection)
    values = (
        code,
        source_title,
        source_publisher,
        source_url,
        kind,
        1 if primary_source else 0,
        publication_date,
        reporting_period,
        access_date,
        verifier,
        verification_time,
        source_notes,
        payload_json,
        editor,
        editor,
        timestamp,
        timestamp,
    )
    if identifier is None:
        cursor = connection.execute(
            """
            INSERT INTO analytical_research_sources (
                ticker, title, publisher, url, source_type, primary_source,
                published_at, period_end, accessed_at, verified_by, verified_at,
                notes, payload_json, created_by, updated_by, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )
        identifier = _inserted_id(connection, cursor, "analytical_research_sources")
    else:
        connection.execute(
            """
            INSERT INTO analytical_research_sources (
                id, ticker, title, publisher, url, source_type, primary_source,
                published_at, period_end, accessed_at, verified_by, verified_at,
                notes, payload_json, created_by, updated_by, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                ticker = excluded.ticker,
                title = excluded.title,
                publisher = excluded.publisher,
                url = excluded.url,
                source_type = excluded.source_type,
                primary_source = excluded.primary_source,
                published_at = excluded.published_at,
                period_end = excluded.period_end,
                accessed_at = excluded.accessed_at,
                verified_by = excluded.verified_by,
                verified_at = excluded.verified_at,
                notes = excluded.notes,
                payload_json = excluded.payload_json,
                updated_by = excluded.updated_by,
                updated_at = excluded.updated_at
            """,
            (identifier, *values),
        )
    _commit_and_sync(connection)
    record = get_research_source(connection, identifier)
    if record is None:
        raise RuntimeError("The research source could not be read after saving.")
    return record


def list_research_sources(
    connection: Any,
    *,
    ticker: str | None = None,
    include_global: bool = False,
    source_type: str | None = None,
    primary_source: bool | None = None,
    verified: bool | None = None,
) -> list[dict[str, Any]]:
    _ensure_schema(connection)
    if not isinstance(include_global, bool):
        raise ValueError("Include global must be a boolean.")
    clauses: list[str] = []
    parameters: list[Any] = []
    if ticker is not None:
        code = _normalise_ticker(ticker)
        if include_global:
            clauses.append("(ticker = ? OR ticker IS NULL)")
        else:
            clauses.append("ticker = ?")
        parameters.append(code)
    if source_type is not None:
        clauses.append("source_type = ?")
        parameters.append(_normalise_enum(source_type, "Source type", RESEARCH_SOURCE_TYPES))
    if primary_source is not None:
        if not isinstance(primary_source, bool):
            raise ValueError("Primary source filter must be a boolean or None.")
        clauses.append("primary_source = ?")
        parameters.append(1 if primary_source else 0)
    if verified is not None:
        if not isinstance(verified, bool):
            raise ValueError("Verified filter must be a boolean or None.")
        clauses.append(
            "(verified_by <> '' AND verified_at IS NOT NULL)"
            if verified
            else "(verified_by = '' OR verified_at IS NULL)"
        )
    query = _SOURCE_SELECT
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY CASE WHEN accessed_at IS NULL THEN 1 ELSE 0 END, accessed_at DESC, id DESC"
    rows = connection.execute(query, tuple(parameters)).fetchall()
    return [record for row in rows if (record := _source_record(row)) is not None]


def delete_research_source(
    connection: Any,
    source_id: int,
    *,
    updated_by: str = "",
    now: datetime | None = None,
) -> bool:
    """Delete a source and detach it from catalysts in the same transaction."""
    _ensure_schema(connection)
    identifier = _normalise_positive_id(source_id, "Source id")
    existed = connection.execute(
        "SELECT 1 FROM analytical_research_sources WHERE id = ?",
        (identifier,),
    ).fetchone() is not None
    editor = _normalise_text(updated_by, "Updated by", max_length=200)
    timestamp = _iso_timestamp(now)
    connection.execute(
        """
        UPDATE analytical_catalyst_events
        SET source_id = NULL, updated_by = ?, updated_at = ?
        WHERE source_id = ?
        """,
        (editor, timestamp, identifier),
    )
    connection.execute("DELETE FROM analytical_research_sources WHERE id = ?", (identifier,))
    _commit_and_sync(connection)
    return existed


__all__ = [
    "CATALYST_DATE_CONFIDENCE",
    "CATALYST_STATUSES",
    "MARKET_OUTCOMES",
    "PROCESS_OUTCOMES",
    "RESEARCH_SOURCE_TYPES",
    "THESIS_REVIEW_STATUSES",
    "append_decision_review",
    "append_thesis_review",
    "delete_catalyst_event",
    "delete_research_source",
    "get_catalyst_event",
    "get_decision_review",
    "get_research_source",
    "get_thesis_review",
    "init_governance_tables",
    "list_catalyst_events",
    "list_decision_reviews",
    "list_research_sources",
    "list_thesis_reviews",
    "upsert_catalyst_event",
    "upsert_research_source",
]
