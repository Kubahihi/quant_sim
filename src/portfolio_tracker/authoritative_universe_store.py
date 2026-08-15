"""Immutable, provenance-aware security-universe snapshots.

Every publication creates a complete snapshot and atomically moves a singleton
active pointer.  Historical snapshots and their entries are never updated by
the public API, so a proposal can retain the exact eligibility evidence that
was available when it entered committee review.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Mapping

from ._governance_utils import (
    boolean,
    canonical_hash,
    commit_and_sync,
    decode_object,
    ensure_schema,
    enum,
    inserted_id,
    iso_date,
    json_object,
    positive_int,
    row_value,
    text,
    ticker,
    utc_timestamp,
)


PROVENANCE_STATUSES = frozenset(
    {"official", "analyst_assumption", "outdated", "not_checked"}
)
ELIGIBILITY_STATUSES = frozenset({"eligible", "ineligible", "unknown"})

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS authoritative_universe_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        version INTEGER NOT NULL UNIQUE,
        source_name TEXT NOT NULL,
        source_url TEXT NOT NULL DEFAULT '',
        provenance_status TEXT NOT NULL,
        as_of_date TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        supersedes_snapshot_id INTEGER,
        published_by TEXT NOT NULL,
        published_at TEXT NOT NULL,
        CHECK (version > 0),
        CHECK (length(trim(source_name)) > 0),
        CHECK (provenance_status IN (
            'official', 'analyst_assumption', 'outdated', 'not_checked'
        )),
        CHECK (supersedes_snapshot_id IS NULL OR supersedes_snapshot_id > 0),
        CHECK (length(trim(published_by)) > 0)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS authoritative_universe_entries (
        snapshot_id INTEGER NOT NULL,
        ticker TEXT NOT NULL,
        eligibility TEXT NOT NULL,
        provenance_status TEXT NOT NULL,
        security_type TEXT NOT NULL DEFAULT '',
        payload_json TEXT NOT NULL,
        PRIMARY KEY (snapshot_id, ticker),
        FOREIGN KEY (snapshot_id) REFERENCES authoritative_universe_snapshots(id),
        CHECK (length(trim(ticker)) > 0),
        CHECK (eligibility IN ('eligible', 'ineligible', 'unknown')),
        CHECK (provenance_status IN (
            'official', 'analyst_assumption', 'outdated', 'not_checked'
        ))
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_authoritative_universe_entry_ticker
    ON authoritative_universe_entries(ticker, snapshot_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS authoritative_universe_active (
        singleton_id INTEGER PRIMARY KEY,
        snapshot_id INTEGER NOT NULL UNIQUE,
        activated_at TEXT NOT NULL,
        CHECK (singleton_id = 1),
        CHECK (snapshot_id > 0),
        FOREIGN KEY (snapshot_id) REFERENCES authoritative_universe_snapshots(id)
    )
    """,
)


def _ensure(connection: Any) -> None:
    ensure_schema(connection, _SCHEMA)


def init_authoritative_universe_tables(connection: Any) -> None:
    _ensure(connection)
    commit_and_sync(connection)


def _active_id(connection: Any) -> int | None:
    row = connection.execute(
        "SELECT snapshot_id FROM authoritative_universe_active WHERE singleton_id = 1"
    ).fetchone()
    return None if row is None else int(row_value(row, "snapshot_id", 0))


def _normalise_entry(
    raw: Mapping[str, Any],
    *,
    default_provenance: str,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("Each universe entry must be a JSON object.")
    code = ticker(raw.get("ticker"))
    if "eligibility" in raw:
        eligibility = enum(raw.get("eligibility"), "Eligibility", ELIGIBILITY_STATUSES)
    elif "eligible" in raw:
        eligibility = "eligible" if boolean(raw.get("eligible"), "Eligible") else "ineligible"
    else:
        eligibility = "unknown"
    provenance = enum(
        raw.get("provenance_status", default_provenance),
        "Entry provenance status",
        PROVENANCE_STATUSES,
    )
    security_type = text(raw.get("security_type"), "Security type", limit=100)
    payload, _ = json_object(raw.get("payload", {}), "Entry payload")
    return {
        "ticker": code,
        "eligibility": eligibility,
        "provenance_status": provenance,
        "security_type": security_type,
        "payload": payload,
    }


def publish_authoritative_universe(
    connection: Any,
    entries: Iterable[Mapping[str, Any]],
    *,
    source_name: str,
    source_url: str = "",
    provenance_status: str = "official",
    as_of_date: Any,
    payload: Mapping[str, Any] | None = None,
    published_by: str,
    expected_active_snapshot_id: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Publish and activate a complete immutable universe snapshot.

    ``expected_active_snapshot_id`` provides optional optimistic concurrency:
    the publication is rejected if another user activated a newer snapshot
    after the caller loaded the page.
    """

    source = text(source_name, "Source name", required=True, limit=500)
    url = text(source_url, "Source URL", limit=2_048)
    provenance = enum(provenance_status, "Provenance status", PROVENANCE_STATUSES)
    as_of = iso_date(as_of_date, "As-of date")
    publisher = text(published_by, "Published by", required=True, limit=200)
    snapshot_payload, payload_json = json_object(payload or {}, "Snapshot payload")
    try:
        raw_entries = list(entries)
    except TypeError as exc:
        raise ValueError("Entries must be an iterable of JSON objects.") from exc
    if not raw_entries:
        raise ValueError("An authoritative universe must contain at least one entry.")
    normalised = [_normalise_entry(item, default_provenance=provenance) for item in raw_entries]
    tickers = [item["ticker"] for item in normalised]
    if len(tickers) != len(set(tickers)):
        duplicate = next(code for code in tickers if tickers.count(code) > 1)
        raise ValueError(f"Duplicate ticker in authoritative universe: {duplicate}.")
    normalised.sort(key=lambda item: item["ticker"])
    timestamp = utc_timestamp(now)

    _ensure(connection)
    current_active_id = _active_id(connection)
    if expected_active_snapshot_id is not None:
        expected = positive_int(expected_active_snapshot_id, "Expected active snapshot id")
        if current_active_id != expected:
            raise ValueError(
                "The active universe changed after it was loaded; refresh before publishing."
            )

    version_row = connection.execute(
        "SELECT COALESCE(MAX(version), 0) + 1 FROM authoritative_universe_snapshots"
    ).fetchone()
    version = int(row_value(version_row, "COALESCE(MAX(version), 0) + 1", 0))
    hash_input = {
        "source_name": source,
        "source_url": url,
        "provenance_status": provenance,
        "as_of_date": as_of,
        "payload": snapshot_payload,
        "entries": normalised,
    }
    content_hash = canonical_hash(hash_input)
    cursor = connection.execute(
        """
        INSERT INTO authoritative_universe_snapshots (
            version, source_name, source_url, provenance_status, as_of_date,
            payload_json, content_hash, supersedes_snapshot_id,
            published_by, published_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            version,
            source,
            url,
            provenance,
            as_of,
            payload_json,
            content_hash,
            current_active_id,
            publisher,
            timestamp,
        ),
    )
    snapshot_id = inserted_id(connection, cursor, "authoritative_universe_snapshots")
    for item in normalised:
        _, entry_payload_json = json_object(item["payload"], "Entry payload")
        connection.execute(
            """
            INSERT INTO authoritative_universe_entries (
                snapshot_id, ticker, eligibility, provenance_status,
                security_type, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_id,
                item["ticker"],
                item["eligibility"],
                item["provenance_status"],
                item["security_type"],
                entry_payload_json,
            ),
        )
    connection.execute(
        """
        INSERT INTO authoritative_universe_active (singleton_id, snapshot_id, activated_at)
        VALUES (1, ?, ?)
        ON CONFLICT(singleton_id) DO UPDATE SET
            snapshot_id = excluded.snapshot_id,
            activated_at = excluded.activated_at
        """,
        (snapshot_id, timestamp),
    )
    commit_and_sync(connection)
    record = get_authoritative_universe_snapshot(connection, snapshot_id)
    if record is None:
        raise RuntimeError("The universe snapshot could not be read after publication.")
    return record


_SNAPSHOT_SELECT = """
    SELECT s.id, s.version, s.source_name, s.source_url, s.provenance_status,
           s.as_of_date, s.payload_json, s.content_hash,
           s.supersedes_snapshot_id, s.published_by, s.published_at,
           CASE WHEN a.snapshot_id IS NULL THEN 0 ELSE 1 END AS is_active
    FROM authoritative_universe_snapshots s
    LEFT JOIN authoritative_universe_active a ON a.snapshot_id = s.id
"""


def _entry_record(row: Any) -> dict[str, Any] | None:
    payload = decode_object(row_value(row, "payload_json", 4))
    if payload is None:
        return None
    return {
        "ticker": str(row_value(row, "ticker", 0)),
        "eligibility": str(row_value(row, "eligibility", 1)),
        "provenance_status": str(row_value(row, "provenance_status", 2)),
        "security_type": str(row_value(row, "security_type", 3) or ""),
        "payload": payload,
    }


def _snapshot_record(connection: Any, row: Any, *, include_entries: bool) -> dict[str, Any] | None:
    payload = decode_object(row_value(row, "payload_json", 6))
    if payload is None:
        return None
    snapshot_id = int(row_value(row, "id", 0))
    record: dict[str, Any] = {
        "id": snapshot_id,
        "version": int(row_value(row, "version", 1)),
        "source_name": str(row_value(row, "source_name", 2)),
        "source_url": str(row_value(row, "source_url", 3) or ""),
        "provenance_status": str(row_value(row, "provenance_status", 4)),
        "as_of_date": str(row_value(row, "as_of_date", 5)),
        "payload": payload,
        "content_hash": str(row_value(row, "content_hash", 7)),
        "supersedes_snapshot_id": (
            None
            if row_value(row, "supersedes_snapshot_id", 8) is None
            else int(row_value(row, "supersedes_snapshot_id", 8))
        ),
        "published_by": str(row_value(row, "published_by", 9)),
        "published_at": str(row_value(row, "published_at", 10)),
        "is_active": bool(row_value(row, "is_active", 11)),
    }
    if include_entries:
        rows = connection.execute(
            """
            SELECT ticker, eligibility, provenance_status, security_type, payload_json
            FROM authoritative_universe_entries
            WHERE snapshot_id = ? ORDER BY ticker
            """,
            (snapshot_id,),
        ).fetchall()
        entries = [entry for item in rows if (entry := _entry_record(item)) is not None]
        record["entries"] = entries
        record["entry_count"] = len(entries)
    return record


def get_authoritative_universe_snapshot(
    connection: Any,
    snapshot_id: int,
    *,
    include_entries: bool = True,
) -> dict[str, Any] | None:
    _ensure(connection)
    identifier = positive_int(snapshot_id, "Snapshot id")
    row = connection.execute(_SNAPSHOT_SELECT + " WHERE s.id = ?", (identifier,)).fetchone()
    return None if row is None else _snapshot_record(connection, row, include_entries=include_entries)


def get_active_authoritative_universe(
    connection: Any,
    *,
    include_entries: bool = True,
) -> dict[str, Any] | None:
    _ensure(connection)
    row = connection.execute(_SNAPSHOT_SELECT + " WHERE a.singleton_id = 1").fetchone()
    return None if row is None else _snapshot_record(connection, row, include_entries=include_entries)


def list_authoritative_universe_snapshots(connection: Any) -> list[dict[str, Any]]:
    _ensure(connection)
    rows = connection.execute(_SNAPSHOT_SELECT + " ORDER BY s.version DESC").fetchall()
    return [
        record
        for row in rows
        if (record := _snapshot_record(connection, row, include_entries=False)) is not None
    ]


def verify_authoritative_universe_snapshot(record: Mapping[str, Any]) -> bool:
    """Verify a complete snapshot record without accessing the database."""

    entries = record.get("entries")
    if not isinstance(entries, list):
        return False
    try:
        expected = canonical_hash(
            {
                "source_name": record["source_name"],
                "source_url": record.get("source_url", ""),
                "provenance_status": record["provenance_status"],
                "as_of_date": record["as_of_date"],
                "payload": record["payload"],
                "entries": entries,
            }
        )
    except (KeyError, TypeError, ValueError):
        return False
    return expected == record.get("content_hash")


def check_security_eligibility(
    connection: Any,
    security_ticker: str,
    *,
    snapshot_id: int | None = None,
) -> dict[str, Any]:
    """Return a fail-closed eligibility decision with provenance reasons."""

    _ensure(connection)
    code = ticker(security_ticker)
    identifier = _active_id(connection) if snapshot_id is None else positive_int(snapshot_id, "Snapshot id")
    if identifier is None:
        return {
            "ticker": code,
            "snapshot_id": None,
            "eligibility": "unknown",
            "provenance_status": "not_checked",
            "snapshot_provenance_status": "not_checked",
            "is_active_snapshot": False,
            "content_hash_valid": False,
            "is_authoritative": False,
            "can_trade": False,
            "reasons": ["No authoritative universe has been published."],
        }
    snapshot = get_authoritative_universe_snapshot(connection, identifier, include_entries=True)
    if snapshot is None:
        raise ValueError("Universe snapshot does not exist.")
    content_hash_valid = verify_authoritative_universe_snapshot(snapshot)
    entry = next((item for item in snapshot["entries"] if item["ticker"] == code), None)
    reasons: list[str] = []
    if entry is None:
        eligibility = "unknown"
        entry_provenance = "not_checked"
        reasons.append("Ticker is absent from the selected universe snapshot.")
    else:
        eligibility = entry["eligibility"]
        entry_provenance = entry["provenance_status"]
    is_active = bool(snapshot["is_active"])
    if not is_active:
        reasons.append("Selected universe snapshot is no longer active.")
    if not content_hash_valid:
        reasons.append("Universe snapshot content hash is invalid.")
    if snapshot["provenance_status"] != "official":
        reasons.append("Universe snapshot is not official.")
    if entry_provenance != "official":
        reasons.append("Ticker eligibility is not backed by official provenance.")
    if eligibility != "eligible":
        reasons.append(f"Ticker eligibility is {eligibility}.")
    is_authoritative = (
        is_active
        and content_hash_valid
        and snapshot["provenance_status"] == "official"
        and entry_provenance == "official"
    )
    can_trade = is_authoritative and eligibility == "eligible"
    return {
        "ticker": code,
        "snapshot_id": identifier,
        "snapshot_version": snapshot["version"],
        "eligibility": eligibility,
        "provenance_status": entry_provenance,
        "snapshot_provenance_status": snapshot["provenance_status"],
        "is_active_snapshot": is_active,
        "content_hash_valid": content_hash_valid,
        "is_authoritative": is_authoritative,
        "can_trade": can_trade,
        "reasons": reasons,
        "entry": entry,
    }


def require_security_eligible(
    connection: Any,
    security_ticker: str,
    *,
    snapshot_id: int | None = None,
) -> dict[str, Any]:
    decision = check_security_eligibility(
        connection,
        security_ticker,
        snapshot_id=snapshot_id,
    )
    if not decision["can_trade"]:
        reason = " ".join(decision["reasons"]) or "Eligibility could not be established."
        raise ValueError(f"{decision['ticker']} is not eligible for trading. {reason}")
    return decision


__all__ = [
    "ELIGIBILITY_STATUSES",
    "PROVENANCE_STATUSES",
    "check_security_eligibility",
    "get_active_authoritative_universe",
    "get_authoritative_universe_snapshot",
    "init_authoritative_universe_tables",
    "list_authoritative_universe_snapshots",
    "publish_authoritative_universe",
    "require_security_eligible",
    "verify_authoritative_universe_snapshot",
]
