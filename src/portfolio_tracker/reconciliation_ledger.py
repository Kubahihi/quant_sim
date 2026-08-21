"""Append-only WInS reconciliation history and readiness governance.

The ledger stores immutable reconciliation records and immutable workflow
events.  A current view is materialised by replaying events, so ownership,
resolutions, dissent, and sign-offs remain auditable instead of overwriting
earlier state.  All public APIs are pure and return JSON-serialisable values.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from typing import Any

from src.data.reliability import verify_snapshot_integrity
from src.portfolio_tracker.wins_reconciliation import reconcile_wins_positions


SCHEMA_VERSION = 1
_UNVERSIONED_SCHEMA = 0
_MISSING = object()


def _json_copy(value: Any, *, field: str) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain only finite JSON values.") from exc


def _utc_datetime(value: datetime | str | None, *, field: str) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{field} must be an ISO-8601 timestamp.") from exc
    else:
        raise TypeError(f"{field} must be a datetime, ISO-8601 string, or None.")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _timestamp(value: datetime | str | None, *, field: str) -> str:
    return _utc_datetime(value, field=field).isoformat()


def _required_text(value: Any, *, field: str) -> str:
    text = " ".join(str(value or "").strip().split())
    if not text:
        raise ValueError(f"{field} is required.")
    return text


def _digest(prefix: str, value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return f"{prefix}_{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:20]}"


def new_reconciliation_ledger() -> dict[str, Any]:
    """Create an empty append-only ledger envelope."""
    return {"schema_version": SCHEMA_VERSION, "reconciliations": [], "events": []}


def _schema_version(value: Any) -> int:
    """Parse an on-disk schema marker without accepting lossy coercions."""
    if value is _MISSING or value is None or value == "":
        return _UNVERSIONED_SCHEMA
    if isinstance(value, bool):
        raise ValueError("ledger.schema_version must be an integer.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ValueError("ledger.schema_version must be an integer.")
    if isinstance(value, str):
        text = value.strip()
        if text and text.lstrip("+-").isdigit():
            return int(text)
    raise ValueError("ledger.schema_version must be an integer.")


def _migrate_unversioned_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    """Upgrade the pre-envelope ledger without rewriting audit records.

    The first deployed pipeline stored ``reconciliations`` and ``events`` but
    some workspaces were saved before the schema marker was added.  Both lists
    are already in the v1 format, so the lossless migration only supplies
    absent envelope fields and leaves every record, event, ID, and unknown
    extension field untouched.
    """
    migrated = dict(ledger)
    if "reconciliations" not in migrated:
        migrated["reconciliations"] = []
    if "events" not in migrated:
        migrated["events"] = []
    migrated["schema_version"] = 1
    return migrated


def migrate_reconciliation_ledger(
    ledger: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a current, JSON-safe ledger while preserving legacy audit data.

    Missing schema markers are treated as the unversioned v0 envelope.  Known
    older versions are upgraded one step at a time.  Future versions fail
    closed so a newer writer can never be silently downgraded.
    """
    if ledger is None:
        return new_reconciliation_ledger()
    copied = _json_copy(ledger, field="ledger")
    if not isinstance(copied, dict):
        raise TypeError("ledger must be a mapping.")

    version = _schema_version(copied.get("schema_version", _MISSING))
    if version < _UNVERSIONED_SCHEMA:
        raise ValueError("ledger.schema_version must not be negative.")
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"ledger.schema_version {version} is newer than supported version "
            f"{SCHEMA_VERSION}."
        )

    while version < SCHEMA_VERSION:
        if version == _UNVERSIONED_SCHEMA:
            copied = _migrate_unversioned_ledger(copied)
        else:  # pragma: no cover - guards future migration-table omissions.
            raise ValueError(f"No ledger migration is registered for schema version {version}.")
        version = _schema_version(copied.get("schema_version", _MISSING))

    # Normalise integer-like legacy markers (for example ``"1"``) without
    # altering any append-only reconciliation or event payload.
    copied["schema_version"] = SCHEMA_VERSION
    if not isinstance(copied.get("reconciliations"), list):
        raise ValueError("ledger.reconciliations must be a list.")
    if not isinstance(copied.get("events"), list):
        raise ValueError("ledger.events must be a list.")
    return copied


def _validated_ledger(ledger: Mapping[str, Any] | None) -> dict[str, Any]:
    return migrate_reconciliation_ledger(ledger)


def _snapshot_positions(snapshot: Mapping[str, Any]) -> Any:
    payload = snapshot.get("payload")
    if isinstance(payload, Mapping):
        for field in ("positions", "holdings", "items"):
            if isinstance(payload.get(field), (list, dict)):
                return payload[field]
    for field in ("positions", "holdings", "items"):
        if isinstance(snapshot.get(field), (list, dict)):
            return snapshot[field]
    return []


def _cash_number(value: Any) -> float | None:
    """Parse a finite cash balance without treating booleans as numbers."""
    if value is None or value == "" or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    text = str(value).strip()
    if not text or text.casefold() in {"n/a", "na", "none", "null", "-", "--"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    text = re.sub(r"[$€£,\s]", "", text)
    try:
        number = float(text)
    except ValueError:
        return None
    if negative:
        number = -number
    return number if math.isfinite(number) else None


def _snapshot_envelope(snapshot: Mapping[str, Any]) -> bool:
    """Distinguish an auditable snapshot from a legacy position-row mapping."""
    return any(
        field in snapshot
        for field in ("snapshot_id", "payload", "dataset", "integrity", "metadata", "cash_value")
    )


def _snapshot_cash(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Extract cash and denomination from current or legacy snapshot envelopes."""
    containers: list[tuple[str, Mapping[str, Any]]] = []
    payload = snapshot.get("payload")
    metadata = snapshot.get("metadata")
    if isinstance(payload, Mapping):
        containers.append(("payload", payload))
    if isinstance(metadata, Mapping):
        containers.append(("metadata", metadata))
    containers.append(("snapshot", snapshot))

    raw_value: Any = _MISSING
    value_source: str | None = None
    for container_name, container in containers:
        for field in ("cash_value", "cash_balance", "cash"):
            if field in container:
                raw_value = container[field]
                value_source = f"{container_name}.{field}"
                break
        if value_source:
            break

    currency: str | None = None
    currency_source: str | None = None
    for container_name, container in containers:
        for field in ("cash_currency", "base_currency", "currency"):
            if field not in container:
                continue
            candidate = " ".join(str(container.get(field) or "").strip().split()).upper()
            if candidate:
                currency = candidate
                currency_source = f"{container_name}.{field}"
                break
        if currency_source:
            break

    return {
        "value_present": raw_value is not _MISSING,
        "value": _cash_number(raw_value) if raw_value is not _MISSING else None,
        "currency": currency,
        "value_source": value_source,
        "currency_source": currency_source,
    }


def _compare_snapshot_cash(
    wins_snapshot: Mapping[str, Any],
    tracked_snapshot: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Compare cash only when the tracker input is an auditable snapshot.

    Historical callers may still pass a bare list (or a single row mapping) of
    positions. Those inputs never carried a cash balance, so they remain
    explicitly ``not_compared`` rather than being reinterpreted as zero cash.
    Canonical snapshot envelopes fail closed when either cash value or its
    denomination is incomplete.
    """
    wins = _snapshot_cash(wins_snapshot)
    tracker_is_snapshot = isinstance(tracked_snapshot, Mapping) and _snapshot_envelope(
        tracked_snapshot
    )
    tracked = (
        _snapshot_cash(tracked_snapshot)
        if tracker_is_snapshot
        else {
            "value_present": False,
            "value": None,
            "currency": None,
            "value_source": None,
            "currency_source": None,
        }
    )
    comparison: dict[str, Any] = {
        "status": "not_compared",
        "is_match": None,
        "amount_match": None,
        "currency_match": None,
        "difference": None,
        "tolerance": tolerance,
        "wins": wins,
        "tracked": tracked,
        "reason": None,
    }
    if not tracker_is_snapshot:
        comparison["reason"] = "legacy_positions_only_input"
        return comparison

    if not wins["value_present"] or not tracked["value_present"]:
        comparison["status"] = "incomplete"
        comparison["is_match"] = False
        comparison["reason"] = "cash_value_missing"
        return comparison
    if wins["value"] is None or tracked["value"] is None:
        comparison["status"] = "incomplete"
        comparison["is_match"] = False
        comparison["reason"] = "cash_value_invalid"
        return comparison
    if not wins["currency"] or not tracked["currency"]:
        comparison["status"] = "incomplete"
        comparison["is_match"] = False
        comparison["reason"] = "cash_currency_missing"
        return comparison

    currency_match = wins["currency"] == tracked["currency"]
    difference = (
        float(wins["value"]) - float(tracked["value"]) if currency_match else None
    )
    amount_match = abs(difference) <= tolerance if difference is not None else None
    is_match = currency_match and amount_match is True
    comparison.update(
        {
            "status": "matched" if is_match else "difference",
            "is_match": is_match,
            "amount_match": amount_match,
            "currency_match": currency_match,
            "difference": difference,
            "reason": None if is_match else "cash_balance_mismatch",
        }
    )
    return comparison


def _snapshot_identity(snapshot: Mapping[str, Any], *, prefix: str) -> str:
    supplied = " ".join(str(snapshot.get("snapshot_id") or "").strip().split())
    if supplied:
        return supplied
    return _digest(prefix, snapshot)


def _exception(
    reconciliation_id: str,
    *,
    ticker: str,
    category: str,
    details: Mapping[str, Any],
    owner: str,
    opened_at: str,
) -> dict[str, Any]:
    basis = {
        "reconciliation_id": reconciliation_id,
        "ticker": ticker,
        "category": category,
        "details": details,
    }
    return {
        "exception_id": _digest("exception", basis),
        "ticker": ticker,
        "category": category,
        "opened_at": opened_at,
        "initial_owner": owner,
        "details": _json_copy(details, field="exception details"),
    }


def _derive_exceptions(
    result: Mapping[str, Any],
    *,
    reconciliation_id: str,
    owner: str,
    opened_at: str,
) -> list[dict[str, Any]]:
    exceptions: list[dict[str, Any]] = []
    for row in result.get("matched", []):
        status = row.get("status")
        if status == "matched":
            continue
        if status == "difference":
            failed_fields = sorted(
                field for field, matched in row.get("field_matches", {}).items() if matched is False
            )
            if row.get("security_type_match") is False:
                failed_fields.append("security_type")
            category = "position_mismatch"
            details = {
                "failed_fields": failed_fields,
                "differences": row.get("differences", {}),
                "wins": row.get("wins", {}),
                "tracked": row.get("tracked", {}),
            }
        else:
            incomplete_fields = sorted(
                field for field, matched in row.get("field_matches", {}).items() if matched is None
            )
            if row.get("security_type_match") is None:
                incomplete_fields.append("security_type")
            category = "incomplete_position_data"
            details = {
                "incomplete_fields": incomplete_fields,
                "wins": row.get("wins", {}),
                "tracked": row.get("tracked", {}),
            }
        exceptions.append(
            _exception(
                reconciliation_id,
                ticker=str(row.get("ticker") or ""),
                category=category,
                details=details,
                owner=owner,
                opened_at=opened_at,
            )
        )
    for category, rows in (
        ("missing_in_wins", result.get("missing", [])),
        ("extra_in_wins", result.get("extra", [])),
    ):
        for row in rows:
            exceptions.append(
                _exception(
                    reconciliation_id,
                    ticker=str(row.get("ticker") or ""),
                    category=category,
                    details={"position": row},
                    owner=owner,
                    opened_at=opened_at,
                )
            )
    cash = result.get("cash_comparison")
    if isinstance(cash, Mapping) and cash.get("status") in {"difference", "incomplete"}:
        status = str(cash.get("status"))
        exceptions.append(
            _exception(
                reconciliation_id,
                ticker="CASH",
                category=("cash_mismatch" if status == "difference" else "incomplete_cash_data"),
                details={"cash_comparison": cash},
                owner=owner,
                opened_at=opened_at,
            )
        )
    return exceptions


def create_reconciliation_record(
    wins_snapshot: Mapping[str, Any],
    tracked_snapshot: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    owner: str,
    performed_at: datetime | str | None = None,
    quantity_tolerance: float = 1e-8,
    currency_tolerance: float = 0.01,
    cash_tolerance: float | None = None,
    supersedes_reconciliation_id: str | None = None,
) -> dict[str, Any]:
    """Reconcile one immutable WInS snapshot against one tracker snapshot."""
    if not isinstance(wins_snapshot, Mapping):
        raise TypeError("wins_snapshot must be a mapping.")
    if "integrity" in wins_snapshot and not verify_snapshot_integrity(wins_snapshot):
        raise ValueError("wins_snapshot failed its integrity check.")
    owner_name = _required_text(owner, field="owner")
    performed = _timestamp(performed_at, field="performed_at")
    wins_id = _snapshot_identity(wins_snapshot, prefix="wins_snapshot")
    if isinstance(tracked_snapshot, Mapping):
        if "integrity" in tracked_snapshot and not verify_snapshot_integrity(tracked_snapshot):
            raise ValueError("tracked_snapshot failed its integrity check.")
        tracked_positions = _snapshot_positions(tracked_snapshot)
        if not tracked_positions and "ticker" in tracked_snapshot:
            tracked_positions = [tracked_snapshot]
        tracked_id = _snapshot_identity(tracked_snapshot, prefix="tracker_snapshot")
    elif isinstance(tracked_snapshot, Sequence) and not isinstance(
        tracked_snapshot, (str, bytes, bytearray)
    ):
        tracked_positions = list(tracked_snapshot)
        tracked_id = _digest("tracker_snapshot", tracked_positions)
    else:
        raise TypeError("tracked_snapshot must be a mapping or sequence of positions.")

    wins_positions = _snapshot_positions(wins_snapshot)
    position_result = reconcile_wins_positions(
        wins_positions,
        tracked_positions,
        quantity_tolerance=quantity_tolerance,
        currency_tolerance=currency_tolerance,
    )
    cash_limit = max(
        0.0,
        float(currency_tolerance if cash_tolerance is None else cash_tolerance),
    )
    cash_comparison = _compare_snapshot_cash(
        wins_snapshot,
        tracked_snapshot,
        tolerance=cash_limit,
    )
    result = dict(position_result)
    result["position_status"] = position_result["status"]
    result["cash_comparison"] = cash_comparison
    if cash_comparison["status"] in {"difference", "incomplete"}:
        result["status"] = "differences"
        result["is_reconciled"] = False
    result["summary"] = {
        **position_result["summary"],
        "position_status": position_result["status"],
        "cash_status": cash_comparison["status"],
    }
    observed_at = wins_snapshot.get("observed_at") or performed
    observed = _timestamp(observed_at, field="wins_snapshot.observed_at")
    basis = {
        "wins_snapshot_id": wins_id,
        "tracked_snapshot_id": tracked_id,
        "performed_at": performed,
        "result": result,
    }
    reconciliation_id = _digest("reconciliation", basis)
    exceptions = _derive_exceptions(
        result,
        reconciliation_id=reconciliation_id,
        owner=owner_name,
        opened_at=performed,
    )
    return {
        "reconciliation_id": reconciliation_id,
        "wins_snapshot_id": wins_id,
        "wins_observed_at": observed,
        "tracked_snapshot_id": tracked_id,
        "performed_at": performed,
        "owner": owner_name,
        "supersedes_reconciliation_id": supersedes_reconciliation_id,
        "tolerances": {
            "quantity": max(0.0, float(quantity_tolerance)),
            "currency": max(0.0, float(currency_tolerance)),
            "cash": cash_limit,
        },
        "base_status": result["status"],
        "base_is_clean": bool(result["is_reconciled"]),
        "result": result,
        "exceptions": exceptions,
    }


def append_reconciliation(
    ledger: Mapping[str, Any] | None,
    wins_snapshot: Mapping[str, Any],
    tracked_snapshot: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    owner: str,
    performed_at: datetime | str | None = None,
    quantity_tolerance: float = 1e-8,
    currency_tolerance: float = 0.01,
    cash_tolerance: float | None = None,
) -> dict[str, Any]:
    """Append a reconciliation without modifying prior records or events."""
    updated = _validated_ledger(ledger)
    previous = updated["reconciliations"][-1] if updated["reconciliations"] else None
    record = create_reconciliation_record(
        wins_snapshot,
        tracked_snapshot,
        owner=owner,
        performed_at=performed_at,
        quantity_tolerance=quantity_tolerance,
        currency_tolerance=currency_tolerance,
        cash_tolerance=cash_tolerance,
        supersedes_reconciliation_id=(
            previous.get("reconciliation_id") if isinstance(previous, Mapping) else None
        ),
    )
    existing_ids = {item.get("reconciliation_id") for item in updated["reconciliations"]}
    if record["reconciliation_id"] in existing_ids:
        raise ValueError("This reconciliation record already exists in the ledger.")
    if previous and _utc_datetime(record["performed_at"], field="performed_at") < _utc_datetime(
        previous.get("performed_at"), field="previous.performed_at"
    ):
        raise ValueError("A reconciliation cannot be appended before the prior record.")
    updated["reconciliations"].append(record)
    return updated


def _record_by_id(ledger: Mapping[str, Any], reconciliation_id: str) -> dict[str, Any]:
    for record in ledger["reconciliations"]:
        if record.get("reconciliation_id") == reconciliation_id:
            return record
    raise KeyError(f"Unknown reconciliation_id: {reconciliation_id}")


def _base_exception(record: Mapping[str, Any], exception_id: str) -> dict[str, Any]:
    for exception in record.get("exceptions", []):
        if exception.get("exception_id") == exception_id:
            return exception
    raise KeyError(f"Unknown exception_id: {exception_id}")


def _append_event(
    ledger: Mapping[str, Any],
    *,
    reconciliation_id: str,
    action: str,
    actor: str,
    occurred_at: datetime | str | None,
    exception_id: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    updated = _validated_ledger(ledger)
    _record_by_id(updated, reconciliation_id)
    event_time = _timestamp(occurred_at, field="occurred_at")
    record = _record_by_id(updated, reconciliation_id)
    if _utc_datetime(event_time, field="occurred_at") < _utc_datetime(
        record.get("performed_at"), field="reconciliation.performed_at"
    ):
        raise ValueError("A reconciliation event cannot predate its reconciliation.")
    prior_event_times = [
        _utc_datetime(item.get("occurred_at"), field="prior_event.occurred_at")
        for item in updated["events"]
        if item.get("reconciliation_id") == reconciliation_id
    ]
    if prior_event_times and _utc_datetime(event_time, field="occurred_at") < max(
        prior_event_times
    ):
        raise ValueError("Reconciliation events must be appended in chronological order.")
    event_basis = {
        "reconciliation_id": reconciliation_id,
        "exception_id": exception_id,
        "action": action,
        "actor": _required_text(actor, field="actor"),
        "occurred_at": event_time,
        "payload": payload or {},
    }
    event = {"event_id": _digest("reconciliation_event", event_basis), **event_basis}
    if any(item.get("event_id") == event["event_id"] for item in updated["events"]):
        raise ValueError("This reconciliation event already exists in the ledger.")
    updated["events"].append(_json_copy(event, field="event"))
    return updated


def materialize_reconciliation(
    ledger: Mapping[str, Any],
    reconciliation_id: str,
) -> dict[str, Any]:
    """Replay append-only events into the current view for one record."""
    validated = _validated_ledger(ledger)
    record = _json_copy(_record_by_id(validated, reconciliation_id), field="reconciliation record")
    views: dict[str, dict[str, Any]] = {}
    for base in record.get("exceptions", []):
        view = {
            **base,
            "owner": base.get("initial_owner"),
            "status": "open",
            "assignment_history": [],
            "resolution": None,
            "sign_off": None,
        }
        views[base["exception_id"]] = view

    reconciliation_sign_off = None
    relevant_events = [
        event
        for event in validated["events"]
        if event.get("reconciliation_id") == reconciliation_id
    ]
    for event in relevant_events:
        action = event.get("action")
        exception_id = event.get("exception_id")
        payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
        if action == "assign_exception" and exception_id in views:
            views[exception_id]["owner"] = payload.get("owner")
            views[exception_id]["assignment_history"].append(
                {
                    "owner": payload.get("owner"),
                    "assigned_by": event.get("actor"),
                    "assigned_at": event.get("occurred_at"),
                }
            )
        elif action == "resolve_exception" and exception_id in views:
            views[exception_id]["status"] = "pending_sign_off"
            views[exception_id]["resolution"] = {
                **payload,
                "resolved_by": event.get("actor"),
                "resolved_at": event.get("occurred_at"),
            }
            views[exception_id]["sign_off"] = None
        elif action == "sign_off_exception" and exception_id in views:
            approved = payload.get("decision") == "approved"
            views[exception_id]["status"] = "closed" if approved else "open"
            views[exception_id]["sign_off"] = {
                **payload,
                "signed_off_by": event.get("actor"),
                "signed_off_at": event.get("occurred_at"),
            }
        elif action == "sign_off_reconciliation":
            reconciliation_sign_off = {
                **payload,
                "signed_off_by": event.get("actor"),
                "signed_off_at": event.get("occurred_at"),
            }

    current_exceptions = [views[base["exception_id"]] for base in record.get("exceptions", [])]
    open_count = sum(item["status"] != "closed" for item in current_exceptions)
    record["exceptions"] = current_exceptions
    record["open_exception_count"] = open_count
    record["all_exceptions_closed"] = open_count == 0
    record["sign_off"] = reconciliation_sign_off
    record["workflow_status"] = (
        "approved"
        if reconciliation_sign_off and reconciliation_sign_off.get("decision") == "approved"
        else "rejected"
        if reconciliation_sign_off and reconciliation_sign_off.get("decision") == "rejected"
        else "exceptions_open"
        if open_count
        else "awaiting_sign_off"
    )
    return record


def assign_exception(
    ledger: Mapping[str, Any],
    reconciliation_id: str,
    exception_id: str,
    *,
    owner: str,
    assigned_by: str,
    assigned_at: datetime | str | None = None,
) -> dict[str, Any]:
    validated = _validated_ledger(ledger)
    record = _record_by_id(validated, reconciliation_id)
    _base_exception(record, exception_id)
    current = materialize_reconciliation(validated, reconciliation_id)
    if _base_exception(current, exception_id).get("status") == "closed":
        raise ValueError("A closed exception cannot be reassigned.")
    return _append_event(
        validated,
        reconciliation_id=reconciliation_id,
        exception_id=exception_id,
        action="assign_exception",
        actor=assigned_by,
        occurred_at=assigned_at,
        payload={"owner": _required_text(owner, field="owner")},
    )


def resolve_exception(
    ledger: Mapping[str, Any],
    reconciliation_id: str,
    exception_id: str,
    *,
    resolution_type: str,
    summary: str,
    resolved_by: str,
    evidence_refs: Sequence[str] = (),
    resolved_at: datetime | str | None = None,
) -> dict[str, Any]:
    current = materialize_reconciliation(ledger, reconciliation_id)
    exception = _base_exception(current, exception_id)
    if exception.get("status") == "closed":
        raise ValueError("A closed exception cannot be resolved again.")
    references = [
        " ".join(str(item).strip().split()) for item in evidence_refs if str(item).strip()
    ]
    return _append_event(
        ledger,
        reconciliation_id=reconciliation_id,
        exception_id=exception_id,
        action="resolve_exception",
        actor=resolved_by,
        occurred_at=resolved_at,
        payload={
            "resolution_type": _required_text(resolution_type, field="resolution_type"),
            "summary": _required_text(summary, field="summary"),
            "evidence_refs": references,
        },
    )


def sign_off_exception(
    ledger: Mapping[str, Any],
    reconciliation_id: str,
    exception_id: str,
    *,
    decision: str,
    signed_off_by: str,
    note: str = "",
    signed_off_at: datetime | str | None = None,
    require_independent: bool = True,
) -> dict[str, Any]:
    current = materialize_reconciliation(ledger, reconciliation_id)
    exception = _base_exception(current, exception_id)
    if exception.get("status") != "pending_sign_off" or not exception.get("resolution"):
        raise ValueError("Exception must have a pending resolution before sign-off.")
    choice = str(decision or "").strip().lower()
    if choice not in {"approved", "rejected"}:
        raise ValueError("decision must be 'approved' or 'rejected'.")
    signer = _required_text(signed_off_by, field="signed_off_by")
    if (
        require_independent
        and signer.casefold() == str(exception["resolution"].get("resolved_by") or "").casefold()
    ):
        raise ValueError("Exception sign-off must be performed by a different person.")
    return _append_event(
        ledger,
        reconciliation_id=reconciliation_id,
        exception_id=exception_id,
        action="sign_off_exception",
        actor=signer,
        occurred_at=signed_off_at,
        payload={
            "decision": choice,
            "note": " ".join(str(note or "").strip().split()),
        },
    )


def sign_off_reconciliation(
    ledger: Mapping[str, Any],
    reconciliation_id: str,
    *,
    decision: str,
    signed_off_by: str,
    note: str = "",
    signed_off_at: datetime | str | None = None,
    require_independent: bool = True,
) -> dict[str, Any]:
    """Approve only a truly clean snapshot; resolved mismatches require a rerun."""
    current = materialize_reconciliation(ledger, reconciliation_id)
    if current.get("sign_off"):
        raise ValueError("This reconciliation already has a final sign-off.")
    choice = str(decision or "").strip().lower()
    if choice not in {"approved", "rejected"}:
        raise ValueError("decision must be 'approved' or 'rejected'.")
    if choice == "approved" and not current.get("base_is_clean"):
        raise ValueError("A reconciliation with snapshot differences cannot be approved; rerun it.")
    if choice == "approved" and not current.get("all_exceptions_closed"):
        raise ValueError("All reconciliation exceptions must be closed before approval.")
    signer = _required_text(signed_off_by, field="signed_off_by")
    if require_independent and signer.casefold() == str(current.get("owner") or "").casefold():
        raise ValueError("Reconciliation sign-off must be performed by a different person.")
    return _append_event(
        ledger,
        reconciliation_id=reconciliation_id,
        action="sign_off_reconciliation",
        actor=signer,
        occurred_at=signed_off_at,
        payload={
            "decision": choice,
            "note": " ".join(str(note or "").strip().split()),
        },
    )


def latest_reconciliation(ledger: Mapping[str, Any]) -> dict[str, Any] | None:
    validated = _validated_ledger(ledger)
    if not validated["reconciliations"]:
        return None
    latest = max(
        validated["reconciliations"],
        key=lambda item: _utc_datetime(item.get("performed_at"), field="performed_at"),
    )
    return materialize_reconciliation(validated, latest["reconciliation_id"])


def reconciliation_readiness_gate(
    ledger: Mapping[str, Any],
    *,
    now: datetime | str | None = None,
    max_age_seconds: float = 86_400,
    expected_wins_snapshot_id: str | None = None,
) -> dict[str, Any]:
    """Block report readiness unless the latest WInS snapshot is clean and signed."""
    current_time = _utc_datetime(now, field="now")
    max_age = max(0.0, float(max_age_seconds))
    latest = latest_reconciliation(ledger)
    if latest is None:
        return {
            "ready": False,
            "status": "blocked",
            "blockers": ["no_reconciliation"],
            "latest_reconciliation_id": None,
            "wins_snapshot_id": None,
            "age_seconds": None,
            "open_exception_count": 0,
        }

    blockers: list[str] = []
    age_seconds = (
        current_time - _utc_datetime(latest.get("wins_observed_at"), field="wins_observed_at")
    ).total_seconds()
    if age_seconds < -300:
        blockers.append("snapshot_timestamp_in_future")
    elif age_seconds > max_age:
        blockers.append("reconciliation_stale")
    if expected_wins_snapshot_id and latest.get("wins_snapshot_id") != expected_wins_snapshot_id:
        blockers.append("newer_snapshot_not_reconciled")
    if not latest.get("base_is_clean"):
        blockers.append("snapshot_has_differences")
    if latest.get("open_exception_count"):
        blockers.append("open_exceptions")
    sign_off = latest.get("sign_off")
    if not sign_off:
        blockers.append("missing_sign_off")
    elif sign_off.get("decision") != "approved":
        blockers.append("sign_off_rejected")
    return {
        "ready": not blockers,
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "latest_reconciliation_id": latest.get("reconciliation_id"),
        "wins_snapshot_id": latest.get("wins_snapshot_id"),
        "age_seconds": age_seconds,
        "open_exception_count": latest.get("open_exception_count", 0),
        "signed_off_by": sign_off.get("signed_off_by") if sign_off else None,
        "signed_off_at": sign_off.get("signed_off_at") if sign_off else None,
    }


def reconciliation_history(ledger: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return a concise newest-first audit history for UI/reporting surfaces."""
    validated = _validated_ledger(ledger)
    history = [
        materialize_reconciliation(validated, record["reconciliation_id"])
        for record in validated["reconciliations"]
    ]
    history.sort(
        key=lambda item: _utc_datetime(item.get("performed_at"), field="performed_at"),
        reverse=True,
    )
    return [
        {
            "reconciliation_id": item["reconciliation_id"],
            "wins_snapshot_id": item["wins_snapshot_id"],
            "tracked_snapshot_id": item["tracked_snapshot_id"],
            "performed_at": item["performed_at"],
            "owner": item["owner"],
            "base_status": item["base_status"],
            "workflow_status": item["workflow_status"],
            "exception_count": len(item["exceptions"]),
            "open_exception_count": item["open_exception_count"],
            "signed_off_by": (
                item["sign_off"].get("signed_off_by") if item.get("sign_off") else None
            ),
        }
        for item in history
    ]


__all__ = [
    "append_reconciliation",
    "assign_exception",
    "create_reconciliation_record",
    "latest_reconciliation",
    "materialize_reconciliation",
    "migrate_reconciliation_ledger",
    "new_reconciliation_ledger",
    "reconciliation_history",
    "reconciliation_readiness_gate",
    "resolve_exception",
    "sign_off_exception",
    "sign_off_reconciliation",
]
