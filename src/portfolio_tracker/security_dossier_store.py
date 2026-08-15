"""Canonical security dossiers and an append-only KPI thesis monitor.

The dossier replaces duplicate thesis, catalyst, position-role and sell-rule
inputs with one versioned entity.  A frozen version embeds the then-current KPI
definitions and a SHA-256 content hash; later KPI or thesis revisions cannot
rewrite evidence used by an Investment Committee proposal.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Mapping

from ._governance_utils import (
    canonical_hash,
    commit_and_sync,
    decode_array,
    decode_object,
    ensure_schema,
    enum,
    finite_number,
    inserted_id,
    json_array,
    json_object,
    positive_int,
    row_value,
    text,
    ticker,
    utc_timestamp,
)


DOSSIER_STATUSES = frozenset({"draft", "frozen"})
KPI_FREQUENCIES = frozenset(
    {"daily", "weekly", "monthly", "quarterly", "annual", "event_driven"}
)
KPI_HEALTH_STATUSES = frozenset({"on_track", "watch", "breach"})

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS canonical_security_dossiers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL UNIQUE,
        candidate_payload_json TEXT NOT NULL,
        created_by TEXT NOT NULL,
        created_at TEXT NOT NULL,
        CHECK (length(trim(ticker)) > 0),
        CHECK (length(trim(created_by)) > 0)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS canonical_security_dossier_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dossier_id INTEGER NOT NULL,
        version INTEGER NOT NULL,
        status TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        kpi_snapshot_json TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        created_by TEXT NOT NULL,
        created_at TEXT NOT NULL,
        frozen_by TEXT,
        frozen_at TEXT,
        UNIQUE (dossier_id, version),
        FOREIGN KEY (dossier_id) REFERENCES canonical_security_dossiers(id),
        CHECK (version > 0),
        CHECK (status IN ('draft', 'frozen')),
        CHECK (length(trim(created_by)) > 0),
        CHECK (
            (status = 'draft' AND frozen_by IS NULL AND frozen_at IS NULL)
            OR
            (status = 'frozen' AND length(trim(frozen_by)) > 0 AND frozen_at IS NOT NULL)
        )
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_canonical_dossier_versions
    ON canonical_security_dossier_versions(dossier_id, version DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS canonical_security_kpis (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dossier_id INTEGER NOT NULL,
        kpi_key TEXT NOT NULL,
        created_by TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (dossier_id, kpi_key),
        FOREIGN KEY (dossier_id) REFERENCES canonical_security_dossiers(id),
        CHECK (length(trim(kpi_key)) > 0),
        CHECK (length(trim(created_by)) > 0)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS canonical_security_kpi_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        kpi_id INTEGER NOT NULL,
        revision INTEGER NOT NULL,
        definition_json TEXT NOT NULL,
        definition_hash TEXT NOT NULL,
        created_by TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (kpi_id, revision),
        FOREIGN KEY (kpi_id) REFERENCES canonical_security_kpis(id),
        CHECK (revision > 0),
        CHECK (length(trim(created_by)) > 0)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_canonical_kpi_versions
    ON canonical_security_kpi_versions(kpi_id, revision DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS canonical_security_kpi_observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        kpi_id INTEGER NOT NULL,
        definition_version_id INTEGER NOT NULL,
        observed_value REAL NOT NULL,
        observed_at TEXT NOT NULL,
        health_status TEXT NOT NULL,
        source_ref TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        recorded_by TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        FOREIGN KEY (kpi_id) REFERENCES canonical_security_kpis(id),
        FOREIGN KEY (definition_version_id) REFERENCES canonical_security_kpi_versions(id),
        CHECK (health_status IN ('on_track', 'watch', 'breach')),
        CHECK (length(trim(source_ref)) > 0),
        CHECK (length(trim(recorded_by)) > 0)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_canonical_kpi_observations
    ON canonical_security_kpi_observations(kpi_id, observed_at DESC, id DESC)
    """,
)


def _ensure(connection: Any) -> None:
    ensure_schema(connection, _SCHEMA)


def init_security_dossier_tables(connection: Any) -> None:
    _ensure(connection)
    commit_and_sync(connection)


def _normalise_kpi_key(value: Any) -> str:
    key = text(value, "KPI key", required=True, limit=100).lower().replace(" ", "_")
    if not all(character.isalnum() or character in {"_", "-", "."} for character in key):
        raise ValueError("KPI key may contain only letters, numbers, underscores, hyphens and dots.")
    return key


def _normalise_observed_at(value: Any) -> str:
    if isinstance(value, datetime):
        return utc_timestamp(value)
    if isinstance(value, date):
        return value.isoformat()
    raw = text(value, "Observed at", required=True, limit=100)
    try:
        if "T" in raw or " " in raw:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return utc_timestamp(parsed)
        return date.fromisoformat(raw).isoformat()
    except ValueError as exc:
        raise ValueError("Observed at must be a valid ISO date or timestamp.") from exc


def _dossier_identity(connection: Any, dossier_id: int) -> dict[str, Any] | None:
    identifier = positive_int(dossier_id, "Dossier id")
    row = connection.execute(
        """
        SELECT id, ticker, candidate_payload_json, created_by, created_at
        FROM canonical_security_dossiers WHERE id = ?
        """,
        (identifier,),
    ).fetchone()
    if row is None:
        return None
    candidate = decode_object(row_value(row, "candidate_payload_json", 2))
    if candidate is None:
        return None
    return {
        "id": int(row_value(row, "id", 0)),
        "ticker": str(row_value(row, "ticker", 1)),
        "candidate": candidate,
        "created_by": str(row_value(row, "created_by", 3)),
        "created_at": str(row_value(row, "created_at", 4)),
    }


def create_security_dossier(
    connection: Any,
    security_ticker: str,
    payload: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any] | None = None,
    created_by: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    code = ticker(security_ticker)
    author = text(created_by, "Created by", required=True, limit=200)
    candidate_copy, candidate_json = json_object(candidate or {}, "Candidate payload")
    thesis_copy, thesis_json = json_object(payload, "Dossier payload")
    timestamp = utc_timestamp(now)
    _ensure(connection)
    if connection.execute(
        "SELECT 1 FROM canonical_security_dossiers WHERE ticker = ?", (code,)
    ).fetchone() is not None:
        raise ValueError(f"A canonical dossier already exists for {code}.")
    cursor = connection.execute(
        """
        INSERT INTO canonical_security_dossiers (
            ticker, candidate_payload_json, created_by, created_at
        ) VALUES (?, ?, ?, ?)
        """,
        (code, candidate_json, author, timestamp),
    )
    dossier_id = inserted_id(connection, cursor, "canonical_security_dossiers")
    draft_hash = canonical_hash(
        {
            "dossier_id": dossier_id,
            "ticker": code,
            "version": 1,
            "payload": thesis_copy,
            "kpis": [],
        }
    )
    connection.execute(
        """
        INSERT INTO canonical_security_dossier_versions (
            dossier_id, version, status, payload_json, kpi_snapshot_json,
            content_hash, created_by, created_at, frozen_by, frozen_at
        ) VALUES (?, 1, 'draft', ?, '[]', ?, ?, ?, NULL, NULL)
        """,
        (dossier_id, thesis_json, draft_hash, author, timestamp),
    )
    commit_and_sync(connection)
    record = get_security_dossier(connection, dossier_id)
    if record is None:
        raise RuntimeError("The security dossier could not be read after creation.")
    record["candidate"] = candidate_copy
    return record


_VERSION_SELECT = """
    SELECT v.id, v.dossier_id, d.ticker, v.version, v.status, v.payload_json,
           v.kpi_snapshot_json, v.content_hash, v.created_by, v.created_at,
           v.frozen_by, v.frozen_at
    FROM canonical_security_dossier_versions v
    JOIN canonical_security_dossiers d ON d.id = v.dossier_id
"""


def _version_record(row: Any) -> dict[str, Any] | None:
    payload = decode_object(row_value(row, "payload_json", 5))
    kpis = decode_array(row_value(row, "kpi_snapshot_json", 6))
    if payload is None or kpis is None:
        return None
    return {
        "id": int(row_value(row, "id", 0)),
        "dossier_id": int(row_value(row, "dossier_id", 1)),
        "ticker": str(row_value(row, "ticker", 2)),
        "version": int(row_value(row, "version", 3)),
        "status": str(row_value(row, "status", 4)),
        "payload": payload,
        "kpi_snapshot": kpis,
        "content_hash": str(row_value(row, "content_hash", 7)),
        "created_by": str(row_value(row, "created_by", 8)),
        "created_at": str(row_value(row, "created_at", 9)),
        "frozen_by": (
            None if row_value(row, "frozen_by", 10) is None else str(row_value(row, "frozen_by", 10))
        ),
        "frozen_at": (
            None if row_value(row, "frozen_at", 11) is None else str(row_value(row, "frozen_at", 11))
        ),
    }


def get_dossier_version(
    connection: Any,
    dossier_id: int,
    version: int,
) -> dict[str, Any] | None:
    _ensure(connection)
    identifier = positive_int(dossier_id, "Dossier id")
    revision = positive_int(version, "Dossier version")
    row = connection.execute(
        _VERSION_SELECT + " WHERE v.dossier_id = ? AND v.version = ?",
        (identifier, revision),
    ).fetchone()
    return None if row is None else _version_record(row)


def list_dossier_versions(connection: Any, dossier_id: int) -> list[dict[str, Any]]:
    _ensure(connection)
    identifier = positive_int(dossier_id, "Dossier id")
    rows = connection.execute(
        _VERSION_SELECT + " WHERE v.dossier_id = ? ORDER BY v.version DESC",
        (identifier,),
    ).fetchall()
    return [record for row in rows if (record := _version_record(row)) is not None]


def get_security_dossier(connection: Any, dossier_id: int) -> dict[str, Any] | None:
    _ensure(connection)
    identity = _dossier_identity(connection, dossier_id)
    if identity is None:
        return None
    versions = list_dossier_versions(connection, dossier_id)
    if not versions:
        return None
    return {
        **identity,
        "current_version": versions[0],
        "version_count": len(versions),
        "latest_frozen_version": next(
            (item for item in versions if item["status"] == "frozen"),
            None,
        ),
    }


def get_security_dossier_by_ticker(
    connection: Any,
    security_ticker: str,
) -> dict[str, Any] | None:
    _ensure(connection)
    code = ticker(security_ticker)
    row = connection.execute(
        "SELECT id FROM canonical_security_dossiers WHERE ticker = ?", (code,)
    ).fetchone()
    return None if row is None else get_security_dossier(connection, int(row_value(row, "id", 0)))


def list_security_dossiers(connection: Any) -> list[dict[str, Any]]:
    _ensure(connection)
    rows = connection.execute("SELECT id FROM canonical_security_dossiers ORDER BY ticker").fetchall()
    return [
        record
        for row in rows
        if (record := get_security_dossier(connection, int(row_value(row, "id", 0)))) is not None
    ]


def append_dossier_version(
    connection: Any,
    dossier_id: int,
    payload: Mapping[str, Any],
    *,
    created_by: str,
    expected_current_version: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    identifier = positive_int(dossier_id, "Dossier id")
    author = text(created_by, "Created by", required=True, limit=200)
    payload_copy, payload_json = json_object(payload, "Dossier payload")
    timestamp = utc_timestamp(now)
    _ensure(connection)
    identity = _dossier_identity(connection, identifier)
    if identity is None:
        raise ValueError("Security dossier does not exist.")
    row = connection.execute(
        """
        SELECT COALESCE(MAX(version), 0)
        FROM canonical_security_dossier_versions WHERE dossier_id = ?
        """,
        (identifier,),
    ).fetchone()
    current_version = int(row_value(row, "COALESCE(MAX(version), 0)", 0))
    if expected_current_version is not None:
        expected = positive_int(expected_current_version, "Expected current version")
        if expected != current_version:
            raise ValueError("The dossier changed after it was loaded; refresh before revising it.")
    next_version = current_version + 1
    draft_hash = canonical_hash(
        {
            "dossier_id": identifier,
            "ticker": identity["ticker"],
            "version": next_version,
            "payload": payload_copy,
            "kpis": [],
        }
    )
    connection.execute(
        """
        INSERT INTO canonical_security_dossier_versions (
            dossier_id, version, status, payload_json, kpi_snapshot_json,
            content_hash, created_by, created_at, frozen_by, frozen_at
        ) VALUES (?, ?, 'draft', ?, '[]', ?, ?, ?, NULL, NULL)
        """,
        (identifier, next_version, payload_json, draft_hash, author, timestamp),
    )
    commit_and_sync(connection)
    record = get_dossier_version(connection, identifier, next_version)
    if record is None:
        raise RuntimeError("The dossier version could not be read after creation.")
    return record


def _normalise_definition(
    *,
    name: Any,
    baseline: Any,
    expected_min: Any,
    expected_max: Any,
    breach_below: Any,
    breach_above: Any,
    unit: Any,
    source: Any,
    frequency: Any,
    owner: Any,
    payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    lower_expected = finite_number(expected_min, "Expected minimum", optional=True)
    upper_expected = finite_number(expected_max, "Expected maximum", optional=True)
    lower_breach = finite_number(breach_below, "Lower breach threshold", optional=True)
    upper_breach = finite_number(breach_above, "Upper breach threshold", optional=True)
    if lower_expected is None and upper_expected is None:
        raise ValueError("Provide an expected minimum or expected maximum.")
    if lower_breach is None and upper_breach is None:
        raise ValueError("Provide a lower or upper breach threshold.")
    if lower_expected is not None and upper_expected is not None and lower_expected > upper_expected:
        raise ValueError("Expected minimum must not exceed expected maximum.")
    if lower_breach is not None and lower_expected is not None and lower_breach > lower_expected:
        raise ValueError("Lower breach threshold must not exceed expected minimum.")
    if upper_breach is not None and upper_expected is not None and upper_breach < upper_expected:
        raise ValueError("Upper breach threshold must not be below expected maximum.")
    extra, _ = json_object(payload or {}, "KPI payload")
    return {
        "name": text(name, "KPI name", required=True, limit=300),
        "baseline": finite_number(baseline, "Baseline"),
        "expected_min": lower_expected,
        "expected_max": upper_expected,
        "breach_below": lower_breach,
        "breach_above": upper_breach,
        "unit": text(unit, "Unit", required=True, limit=100),
        "source": text(source, "KPI source", required=True, limit=2_000),
        "frequency": enum(frequency, "KPI frequency", KPI_FREQUENCIES),
        "owner": text(owner, "KPI owner", required=True, limit=200),
        "payload": extra,
    }


def upsert_kpi_definition(
    connection: Any,
    dossier_id: int,
    kpi_key: str,
    *,
    name: str,
    baseline: float,
    expected_min: float | None = None,
    expected_max: float | None = None,
    breach_below: float | None = None,
    breach_above: float | None = None,
    unit: str,
    source: str,
    frequency: str,
    owner: str,
    payload: Mapping[str, Any] | None = None,
    updated_by: str,
    expected_current_revision: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Create a KPI identity or append an immutable definition revision."""

    identifier = positive_int(dossier_id, "Dossier id")
    key = _normalise_kpi_key(kpi_key)
    author = text(updated_by, "Updated by", required=True, limit=200)
    definition = _normalise_definition(
        name=name,
        baseline=baseline,
        expected_min=expected_min,
        expected_max=expected_max,
        breach_below=breach_below,
        breach_above=breach_above,
        unit=unit,
        source=source,
        frequency=frequency,
        owner=owner,
        payload=payload,
    )
    _, definition_json = json_object(definition, "KPI definition")
    definition_hash = canonical_hash(definition)
    timestamp = utc_timestamp(now)
    _ensure(connection)
    if _dossier_identity(connection, identifier) is None:
        raise ValueError("Security dossier does not exist.")
    row = connection.execute(
        "SELECT id FROM canonical_security_kpis WHERE dossier_id = ? AND kpi_key = ?",
        (identifier, key),
    ).fetchone()
    if row is None:
        if expected_current_revision not in (None, 0):
            raise ValueError("KPI does not yet have the expected revision.")
        cursor = connection.execute(
            """
            INSERT INTO canonical_security_kpis (
                dossier_id, kpi_key, created_by, created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (identifier, key, author, timestamp),
        )
        kpi_id = inserted_id(connection, cursor, "canonical_security_kpis")
        revision = 1
    else:
        kpi_id = int(row_value(row, "id", 0))
        revision_row = connection.execute(
            "SELECT COALESCE(MAX(revision), 0) FROM canonical_security_kpi_versions WHERE kpi_id = ?",
            (kpi_id,),
        ).fetchone()
        current_revision = int(row_value(revision_row, "COALESCE(MAX(revision), 0)", 0))
        if expected_current_revision is not None:
            expected = positive_int(expected_current_revision, "Expected current revision")
            if expected != current_revision:
                raise ValueError("The KPI definition changed after it was loaded.")
        revision = current_revision + 1
    cursor = connection.execute(
        """
        INSERT INTO canonical_security_kpi_versions (
            kpi_id, revision, definition_json, definition_hash, created_by, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (kpi_id, revision, definition_json, definition_hash, author, timestamp),
    )
    version_id = inserted_id(connection, cursor, "canonical_security_kpi_versions")
    commit_and_sync(connection)
    record = get_kpi_definition(connection, identifier, key)
    if record is None or record["definition_version_id"] != version_id:
        raise RuntimeError("The KPI definition could not be read after saving.")
    return record


_KPI_SELECT = """
    SELECT k.id AS kpi_id, k.dossier_id, k.kpi_key, v.id AS definition_version_id,
           v.revision, v.definition_json, v.definition_hash,
           k.created_by, k.created_at, v.created_by AS updated_by,
           v.created_at AS updated_at
    FROM canonical_security_kpis k
    JOIN canonical_security_kpi_versions v ON v.kpi_id = k.id
    WHERE v.revision = (
        SELECT MAX(latest.revision) FROM canonical_security_kpi_versions latest
        WHERE latest.kpi_id = k.id
    )
"""


def _kpi_record(row: Any) -> dict[str, Any] | None:
    definition = decode_object(row_value(row, "definition_json", 5))
    if definition is None:
        return None
    return {
        "kpi_id": int(row_value(row, "kpi_id", 0)),
        "dossier_id": int(row_value(row, "dossier_id", 1)),
        "kpi_key": str(row_value(row, "kpi_key", 2)),
        "definition_version_id": int(row_value(row, "definition_version_id", 3)),
        "revision": int(row_value(row, "revision", 4)),
        "definition": definition,
        "definition_hash": str(row_value(row, "definition_hash", 6)),
        "created_by": str(row_value(row, "created_by", 7)),
        "created_at": str(row_value(row, "created_at", 8)),
        "updated_by": str(row_value(row, "updated_by", 9)),
        "updated_at": str(row_value(row, "updated_at", 10)),
    }


def get_kpi_definition(
    connection: Any,
    dossier_id: int,
    kpi_key: str,
) -> dict[str, Any] | None:
    _ensure(connection)
    identifier = positive_int(dossier_id, "Dossier id")
    key = _normalise_kpi_key(kpi_key)
    row = connection.execute(
        _KPI_SELECT + " AND k.dossier_id = ? AND k.kpi_key = ?",
        (identifier, key),
    ).fetchone()
    return None if row is None else _kpi_record(row)


def list_kpi_definitions(connection: Any, dossier_id: int) -> list[dict[str, Any]]:
    _ensure(connection)
    identifier = positive_int(dossier_id, "Dossier id")
    rows = connection.execute(
        _KPI_SELECT + " AND k.dossier_id = ? ORDER BY k.kpi_key",
        (identifier,),
    ).fetchall()
    return [record for row in rows if (record := _kpi_record(row)) is not None]


def _health_for_value(definition: Mapping[str, Any], value: float) -> str:
    lower_breach = definition.get("breach_below")
    upper_breach = definition.get("breach_above")
    if (lower_breach is not None and value <= float(lower_breach)) or (
        upper_breach is not None and value >= float(upper_breach)
    ):
        return "breach"
    lower_expected = definition.get("expected_min")
    upper_expected = definition.get("expected_max")
    if (lower_expected is not None and value < float(lower_expected)) or (
        upper_expected is not None and value > float(upper_expected)
    ):
        return "watch"
    return "on_track"


def append_kpi_observation(
    connection: Any,
    dossier_id: int,
    kpi_key: str,
    observed_value: float,
    *,
    observed_at: Any,
    source_ref: str | None = None,
    payload: Mapping[str, Any] | None = None,
    recorded_by: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    identifier = positive_int(dossier_id, "Dossier id")
    key = _normalise_kpi_key(kpi_key)
    value = finite_number(observed_value, "Observed value")
    observation_time = _normalise_observed_at(observed_at)
    actor = text(recorded_by, "Recorded by", required=True, limit=200)
    payload_copy, payload_json = json_object(payload or {}, "Observation payload")
    recorded_at = utc_timestamp(now)
    _ensure(connection)
    definition = get_kpi_definition(connection, identifier, key)
    if definition is None:
        raise ValueError("KPI definition does not exist.")
    source = text(
        source_ref if source_ref is not None else definition["definition"]["source"],
        "Observation source",
        required=True,
        limit=2_000,
    )
    health = _health_for_value(definition["definition"], float(value))
    cursor = connection.execute(
        """
        INSERT INTO canonical_security_kpi_observations (
            kpi_id, definition_version_id, observed_value, observed_at,
            health_status, source_ref, payload_json, recorded_by, recorded_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            definition["kpi_id"],
            definition["definition_version_id"],
            value,
            observation_time,
            health,
            source,
            payload_json,
            actor,
            recorded_at,
        ),
    )
    observation_id = inserted_id(connection, cursor, "canonical_security_kpi_observations")
    commit_and_sync(connection)
    record = get_kpi_observation(connection, observation_id)
    if record is None:
        raise RuntimeError("The KPI observation could not be read after saving.")
    record["payload"] = payload_copy
    return record


_OBSERVATION_SELECT = """
    SELECT o.id, o.kpi_id, k.dossier_id, k.kpi_key, o.definition_version_id,
           v.revision AS definition_revision, o.observed_value, o.observed_at,
           o.health_status, o.source_ref, o.payload_json,
           o.recorded_by, o.recorded_at
    FROM canonical_security_kpi_observations o
    JOIN canonical_security_kpis k ON k.id = o.kpi_id
    JOIN canonical_security_kpi_versions v ON v.id = o.definition_version_id
"""


def _observation_record(row: Any) -> dict[str, Any] | None:
    payload = decode_object(row_value(row, "payload_json", 10))
    if payload is None:
        return None
    return {
        "id": int(row_value(row, "id", 0)),
        "kpi_id": int(row_value(row, "kpi_id", 1)),
        "dossier_id": int(row_value(row, "dossier_id", 2)),
        "kpi_key": str(row_value(row, "kpi_key", 3)),
        "definition_version_id": int(row_value(row, "definition_version_id", 4)),
        "definition_revision": int(row_value(row, "definition_revision", 5)),
        "observed_value": float(row_value(row, "observed_value", 6)),
        "observed_at": str(row_value(row, "observed_at", 7)),
        "health_status": str(row_value(row, "health_status", 8)),
        "source_ref": str(row_value(row, "source_ref", 9)),
        "payload": payload,
        "recorded_by": str(row_value(row, "recorded_by", 11)),
        "recorded_at": str(row_value(row, "recorded_at", 12)),
    }


def get_kpi_observation(connection: Any, observation_id: int) -> dict[str, Any] | None:
    _ensure(connection)
    identifier = positive_int(observation_id, "Observation id")
    row = connection.execute(_OBSERVATION_SELECT + " WHERE o.id = ?", (identifier,)).fetchone()
    return None if row is None else _observation_record(row)


def list_kpi_observations(
    connection: Any,
    dossier_id: int,
    *,
    kpi_key: str | None = None,
) -> list[dict[str, Any]]:
    _ensure(connection)
    identifier = positive_int(dossier_id, "Dossier id")
    parameters: list[Any] = [identifier]
    query = _OBSERVATION_SELECT + " WHERE k.dossier_id = ?"
    if kpi_key is not None:
        query += " AND k.kpi_key = ?"
        parameters.append(_normalise_kpi_key(kpi_key))
    query += " ORDER BY o.observed_at DESC, o.id DESC"
    rows = connection.execute(query, tuple(parameters)).fetchall()
    return [record for row in rows if (record := _observation_record(row)) is not None]


def _validate_required_dossier_fields(payload: Mapping[str, Any]) -> None:
    checks = {
        "thesis": payload.get("thesis"),
        "catalysts": payload.get("catalysts", payload.get("catalyst")),
        "invalidation condition": payload.get("invalidation_condition", payload.get("invalidation")),
        "portfolio role": payload.get("portfolio_role"),
        "sell discipline": payload.get("sell_discipline", payload.get("sell_condition")),
    }
    missing: list[str] = []
    for name, value in checks.items():
        if value in (None, "", [], {}):
            missing.append(name)
    if missing:
        raise ValueError("Dossier cannot be frozen; missing: " + ", ".join(missing) + ".")


def freeze_dossier(
    connection: Any,
    dossier_id: int,
    *,
    version: int | None = None,
    frozen_by: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    identifier = positive_int(dossier_id, "Dossier id")
    actor = text(frozen_by, "Frozen by", required=True, limit=200)
    timestamp = utc_timestamp(now)
    _ensure(connection)
    versions = list_dossier_versions(connection, identifier)
    if not versions:
        raise ValueError("Security dossier does not exist.")
    target_version = versions[0]["version"] if version is None else positive_int(version, "Dossier version")
    if target_version != versions[0]["version"]:
        raise ValueError("Only the current dossier version can be frozen.")
    target = get_dossier_version(connection, identifier, target_version)
    if target is None:
        raise ValueError("Dossier version does not exist.")
    if target["status"] == "frozen":
        return target
    _validate_required_dossier_fields(target["payload"])
    kpis = list_kpi_definitions(connection, identifier)
    if not kpis:
        raise ValueError("Dossier cannot be frozen without at least one KPI definition.")
    kpi_snapshot = [
        {
            "kpi_id": item["kpi_id"],
            "kpi_key": item["kpi_key"],
            "definition_version_id": item["definition_version_id"],
            "revision": item["revision"],
            "definition": item["definition"],
            "definition_hash": item["definition_hash"],
        }
        for item in kpis
    ]
    _, snapshot_json = json_array(kpi_snapshot, "KPI snapshot")
    content_hash = canonical_hash(
        {
            "dossier_id": identifier,
            "ticker": target["ticker"],
            "version": target_version,
            "payload": target["payload"],
            "kpis": kpi_snapshot,
        }
    )
    connection.execute(
        """
        UPDATE canonical_security_dossier_versions
        SET status = 'frozen', kpi_snapshot_json = ?, content_hash = ?,
            frozen_by = ?, frozen_at = ?
        WHERE dossier_id = ? AND version = ? AND status = 'draft'
        """,
        (snapshot_json, content_hash, actor, timestamp, identifier, target_version),
    )
    commit_and_sync(connection)
    record = get_dossier_version(connection, identifier, target_version)
    if record is None:
        raise RuntimeError("The frozen dossier could not be read after saving.")
    return record


def verify_frozen_dossier(record: Mapping[str, Any]) -> bool:
    if record.get("status") != "frozen":
        return False
    try:
        expected = canonical_hash(
            {
                "dossier_id": int(record["dossier_id"]),
                "ticker": str(record["ticker"]),
                "version": int(record["version"]),
                "payload": record["payload"],
                "kpis": record["kpi_snapshot"],
            }
        )
    except (KeyError, TypeError, ValueError):
        return False
    return expected == record.get("content_hash")


def _parse_observation_time(value: str) -> datetime:
    if "T" in value:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return datetime.combine(date.fromisoformat(value), datetime.min.time(), tzinfo=timezone.utc)


def get_kpi_monitor(
    connection: Any,
    dossier_id: int,
    *,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    """Return current KPI definitions, latest observations and staleness."""

    identifier = positive_int(dossier_id, "Dossier id")
    current = as_of or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)
    stale_after_days = {
        "daily": 2,
        "weekly": 8,
        "monthly": 35,
        "quarterly": 100,
        "annual": 370,
        "event_driven": None,
    }
    definitions = list_kpi_definitions(connection, identifier)
    items: list[dict[str, Any]] = []
    counts = {"on_track": 0, "watch": 0, "breach": 0, "missing": 0, "stale": 0}
    for definition in definitions:
        observations = list_kpi_observations(
            connection,
            identifier,
            kpi_key=definition["kpi_key"],
        )
        latest = observations[0] if observations else None
        frequency = definition["definition"]["frequency"]
        days = stale_after_days[frequency]
        is_stale = False
        next_due_at: str | None = None
        if latest is not None and days is not None:
            observed = _parse_observation_time(latest["observed_at"])
            next_due = observed + timedelta(days=days)
            next_due_at = next_due.isoformat()
            is_stale = current > next_due
        health = "missing" if latest is None else latest["health_status"]
        counts[health] += 1
        if is_stale:
            counts["stale"] += 1
        items.append(
            {
                **definition,
                "latest_observation": latest,
                "health_status": health,
                "is_stale": is_stale,
                "next_due_at": next_due_at,
                "last_updated_at": None if latest is None else latest["observed_at"],
            }
        )
    return {
        "dossier_id": identifier,
        "as_of": current.isoformat(),
        "items": items,
        "summary": counts,
        # ``counts`` is kept as a UI-friendly alias; both values are detached
        # JSON objects so callers can safely annotate one without the other.
        "counts": dict(counts),
    }


__all__ = [
    "DOSSIER_STATUSES",
    "KPI_FREQUENCIES",
    "KPI_HEALTH_STATUSES",
    "append_dossier_version",
    "append_kpi_observation",
    "create_security_dossier",
    "freeze_dossier",
    "get_dossier_version",
    "get_kpi_definition",
    "get_kpi_monitor",
    "get_kpi_observation",
    "get_security_dossier",
    "get_security_dossier_by_ticker",
    "init_security_dossier_tables",
    "list_dossier_versions",
    "list_kpi_definitions",
    "list_kpi_observations",
    "list_security_dossiers",
    "upsert_kpi_definition",
    "verify_frozen_dossier",
]
