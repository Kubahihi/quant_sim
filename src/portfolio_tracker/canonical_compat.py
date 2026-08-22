"""Read-only projections from canonical governance stores to legacy UI shapes.

The application historically read holding theses and approved-security rows
from ``analytical_*`` tables.  Canonical security dossiers and the active
authoritative-universe snapshot now own those facts.  This module lets older
analytics consume the canonical records without copying data back into the
legacy tables.

Both public functions are deliberately SELECT-only.  They do not initialise
schemas, commit, sync, migrate, or otherwise mutate the supplied connection.
Canonical records win per ticker; a legacy record is returned only when that
ticker has no canonical counterpart.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from typing import Any, Mapping


_DOSSIER_TABLE = "canonical_security_dossiers"
_DOSSIER_VERSION_TABLE = "canonical_security_dossier_versions"
_KPI_TABLE = "canonical_security_kpis"
_KPI_VERSION_TABLE = "canonical_security_kpi_versions"
_UNIVERSE_ACTIVE_TABLE = "authoritative_universe_active"
_UNIVERSE_SNAPSHOT_TABLE = "authoritative_universe_snapshots"
_UNIVERSE_ENTRY_TABLE = "authoritative_universe_entries"
_LEGACY_THESIS_TABLE = "analytical_holding_theses"
_LEGACY_APPROVED_TABLE = "analytical_approved_securities"


def _row_value(row: Any, key: str, index: int) -> Any:
    if isinstance(row, Mapping):
        return row.get(key)
    keys = getattr(row, "keys", None)
    if callable(keys):
        try:
            return row[key]
        except (KeyError, IndexError, TypeError):
            pass
    try:
        return row[index]
    except (IndexError, KeyError, TypeError):
        return None


def _table_exists(connection: Any, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _all_tables_exist(connection: Any, table_names: tuple[str, ...]) -> bool:
    return all(_table_exists(connection, name) for name in table_names)


def _ticker(value: Any) -> str:
    return str(value or "").strip().upper()


def _decode_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return decoded if isinstance(decoded, dict) else None


def _optional_number(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _optional_positive_int(value: Any) -> int | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _first_text(payload: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _legacy_theses(connection: Any) -> dict[str, dict[str, Any]]:
    if not _table_exists(connection, _LEGACY_THESIS_TABLE):
        return {}
    rows = connection.execute(
        """
        SELECT ticker, status, conviction, strategy_version, next_review_at,
               payload_json, updated_by, created_at, updated_at
        FROM analytical_holding_theses
        ORDER BY ticker
        """
    ).fetchall()
    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        code = _ticker(_row_value(row, "ticker", 0))
        payload = _decode_object(_row_value(row, "payload_json", 5))
        if not code or payload is None:
            continue
        conviction = _optional_number(_row_value(row, "conviction", 2))
        records[code] = {
            "ticker": code,
            "status": str(_row_value(row, "status", 1) or ""),
            "conviction": conviction,
            "strategy_version": _optional_positive_int(
                _row_value(row, "strategy_version", 3)
            ),
            "next_review_at": (
                None
                if _row_value(row, "next_review_at", 4) in (None, "")
                else str(_row_value(row, "next_review_at", 4))
            ),
            "payload": payload,
            "updated_by": str(_row_value(row, "updated_by", 6) or ""),
            "created_at": str(_row_value(row, "created_at", 7) or ""),
            "updated_at": str(_row_value(row, "updated_at", 8) or ""),
            "source_system": "legacy_analytical_holding_thesis",
        }
    return records


def _canonical_monitoring_kpis(connection: Any, dossier_id: int) -> list[dict[str, Any]]:
    if not _all_tables_exist(connection, (_KPI_TABLE, _KPI_VERSION_TABLE)):
        return []
    rows = connection.execute(
        """
        SELECT k.kpi_key, v.definition_json
        FROM canonical_security_kpis k
        JOIN canonical_security_kpi_versions v
          ON v.kpi_id = k.id
         AND v.revision = (
             SELECT MAX(latest.revision)
             FROM canonical_security_kpi_versions latest
             WHERE latest.kpi_id = k.id
         )
        WHERE k.dossier_id = ?
        ORDER BY k.kpi_key
        """,
        (dossier_id,),
    ).fetchall()
    definitions: list[dict[str, Any]] = []
    for row in rows:
        definition = _decode_object(_row_value(row, "definition_json", 1)) or {}
        definitions.append({"kpi_key": str(_row_value(row, "kpi_key", 0) or ""), **definition})
    return definitions


def _canonical_theses(connection: Any) -> dict[str, dict[str, Any]]:
    if not _all_tables_exist(connection, (_DOSSIER_TABLE, _DOSSIER_VERSION_TABLE)):
        return {}
    rows = connection.execute(
        """
        SELECT d.ticker, d.id AS dossier_id, d.created_at AS dossier_created_at,
               v.version, v.status, v.payload_json, v.content_hash,
               v.created_by, v.created_at AS version_created_at
        FROM canonical_security_dossiers d
        JOIN canonical_security_dossier_versions v
          ON v.dossier_id = d.id
         AND v.version = (
             SELECT MAX(latest.version)
             FROM canonical_security_dossier_versions latest
             WHERE latest.dossier_id = d.id
         )
        ORDER BY d.ticker
        """
    ).fetchall()
    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        code = _ticker(_row_value(row, "ticker", 0))
        if not code:
            continue
        # A malformed canonical payload must not resurrect stale legacy facts.
        canonical_payload = _decode_object(_row_value(row, "payload_json", 5)) or {}
        payload = deepcopy(canonical_payload)

        thesis = _first_text(payload, "investment_thesis", "thesis")
        if thesis is not None:
            payload.setdefault("investment_thesis", thesis)
        invalidation = _first_text(payload, "invalidation", "invalidation_condition")
        if invalidation is not None:
            payload.setdefault("invalidation", invalidation)
        primary_goal = _first_text(payload, "primary_goal", "client_goal")
        if primary_goal is not None:
            payload.setdefault("primary_goal", primary_goal)
            payload.setdefault("goals", [primary_goal])

        next_review_at = _first_text(payload, "next_review_at", "review_date")
        if next_review_at is not None:
            payload.setdefault("review_date", next_review_at)
        conviction = _optional_number(payload.get("conviction"))
        governance_status = str(_row_value(row, "status", 4) or "draft")
        thesis_status = _first_text(payload, "thesis_status", "status") or governance_status
        dossier_id = int(_row_value(row, "dossier_id", 1))
        dossier_version = int(_row_value(row, "version", 3))
        fair_value_scenarios = payload.get("fair_value_scenarios")
        if not isinstance(fair_value_scenarios, Mapping):
            fair_value_scenarios = {
                "bear": payload.get("fair_value_bear"),
                "base": payload.get("fair_value_base"),
                "bull": payload.get("fair_value_bull"),
            }
            if any(value not in (None, "") for value in fair_value_scenarios.values()):
                payload["fair_value_scenarios"] = fair_value_scenarios
        if "margin_of_safety" not in payload and payload.get("margin_of_safety_pct") is not None:
            payload["margin_of_safety"] = payload["margin_of_safety_pct"]
        monitoring_kpis = _canonical_monitoring_kpis(connection, dossier_id)
        if monitoring_kpis:
            payload.setdefault("monitoring_kpis", monitoring_kpis)

        records[code] = {
            "ticker": code,
            "status": thesis_status,
            "conviction": conviction,
            "strategy_version": _optional_positive_int(payload.get("strategy_version")),
            "next_review_at": next_review_at,
            "payload": payload,
            "updated_by": str(_row_value(row, "created_by", 7) or ""),
            "created_at": str(_row_value(row, "dossier_created_at", 2) or ""),
            "updated_at": str(_row_value(row, "version_created_at", 8) or ""),
            "source_system": "canonical_security_dossier",
            "canonical_dossier_id": dossier_id,
            "canonical_dossier_version": dossier_version,
            "canonical_dossier_status": governance_status,
            "content_hash": str(_row_value(row, "content_hash", 6) or ""),
        }
    return records


def list_canonical_first_holding_theses(
    connection: Any,
    *,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """Return legacy-compatible theses with canonical dossiers taking priority.

    Legacy-only tickers remain available during the transition.  ``status`` is
    applied after the two sources are merged, so an obsolete legacy row can
    never reappear because the canonical row has a different status.
    """

    merged = _legacy_theses(connection)
    merged.update(_canonical_theses(connection))
    records = [merged[code] for code in sorted(merged)]
    if status is None:
        return records
    expected = str(status).strip().casefold()
    return [item for item in records if str(item["status"]).casefold() == expected]


def _legacy_approved_securities(connection: Any) -> dict[str, dict[str, Any]]:
    if not _table_exists(connection, _LEGACY_APPROVED_TABLE):
        return {}
    rows = connection.execute(
        """
        SELECT ticker, approved, payload_json, updated_by, created_at, updated_at
        FROM analytical_approved_securities
        ORDER BY ticker
        """
    ).fetchall()
    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        code = _ticker(_row_value(row, "ticker", 0))
        payload = _decode_object(_row_value(row, "payload_json", 2))
        if not code or payload is None:
            continue
        approved = bool(int(_row_value(row, "approved", 1) or 0))
        eligibility = "eligible" if approved else "ineligible"
        security_type = _first_text(payload, "security_type", "asset_type") or ""
        provenance = (
            _first_text(payload, "provenance_status", "authority_status")
            or "not_checked"
        ).strip().lower().replace(" ", "_")
        records[code] = {
            "ticker": code,
            "approved": approved,
            "status": eligibility,
            "eligibility": eligibility,
            "security_type": security_type,
            "provenance_status": provenance,
            "payload": payload,
            "updated_by": str(_row_value(row, "updated_by", 3) or ""),
            "created_at": str(_row_value(row, "created_at", 4) or ""),
            "updated_at": str(_row_value(row, "updated_at", 5) or ""),
            "source_system": "legacy_analytical_approved_security",
        }
    return records


def _canonical_approved_securities(connection: Any) -> dict[str, dict[str, Any]]:
    tables = (
        _UNIVERSE_ACTIVE_TABLE,
        _UNIVERSE_SNAPSHOT_TABLE,
        _UNIVERSE_ENTRY_TABLE,
    )
    if not _all_tables_exist(connection, tables):
        return {}
    rows = connection.execute(
        """
        SELECT e.ticker, e.eligibility, e.provenance_status, e.security_type,
               e.payload_json, s.id AS snapshot_id, s.version AS snapshot_version,
               s.source_name, s.source_url, s.as_of_date,
               s.published_by, s.published_at
        FROM authoritative_universe_active a
        JOIN authoritative_universe_snapshots s ON s.id = a.snapshot_id
        JOIN authoritative_universe_entries e ON e.snapshot_id = s.id
        WHERE a.singleton_id = 1
        ORDER BY e.ticker
        """
    ).fetchall()
    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        code = _ticker(_row_value(row, "ticker", 0))
        if not code:
            continue
        eligibility = str(_row_value(row, "eligibility", 1) or "unknown")
        provenance = str(_row_value(row, "provenance_status", 2) or "not_checked")
        security_type = str(_row_value(row, "security_type", 3) or "")
        source_name = str(_row_value(row, "source_name", 7) or "")
        source_url = str(_row_value(row, "source_url", 8) or "")
        as_of_date = str(_row_value(row, "as_of_date", 9) or "")
        payload = _decode_object(_row_value(row, "payload_json", 4)) or {}
        # Snapshot authority fields override conflicting free-form entry data.
        payload.update(
            {
                "eligibility": eligibility,
                "security_type": security_type,
                "provenance_status": provenance,
                "authority_status": provenance.replace("_", " "),
                "source_name": source_name,
                "source_url": source_url,
                "source_as_of": as_of_date,
            }
        )
        snapshot_id = int(_row_value(row, "snapshot_id", 5))
        snapshot_version = int(_row_value(row, "snapshot_version", 6))
        published_at = str(_row_value(row, "published_at", 11) or "")
        records[code] = {
            "ticker": code,
            "approved": eligibility == "eligible",
            "status": eligibility,
            "eligibility": eligibility,
            "security_type": security_type,
            "provenance_status": provenance,
            "payload": payload,
            "updated_by": str(_row_value(row, "published_by", 10) or ""),
            "created_at": published_at,
            "updated_at": published_at,
            "source_system": "active_authoritative_universe",
            "authoritative_snapshot_id": snapshot_id,
            "authoritative_snapshot_version": snapshot_version,
        }
    return records


def list_canonical_first_approved_securities(
    connection: Any,
    *,
    approved_only: bool | None = True,
) -> list[dict[str, Any]]:
    """Return the active universe in the legacy approved-security shape.

    The active snapshot wins per ticker.  Legacy-only tickers remain as a
    compatibility fallback.  Passing ``None`` includes eligible, ineligible,
    and unknown decisions; the default returns only eligible/approved rows.
    """

    merged = _legacy_approved_securities(connection)
    merged.update(_canonical_approved_securities(connection))
    records = [merged[code] for code in sorted(merged)]
    if approved_only is None:
        return records
    expected = bool(approved_only)
    return [item for item in records if bool(item["approved"]) is expected]


__all__ = [
    "list_canonical_first_approved_securities",
    "list_canonical_first_holding_theses",
]
