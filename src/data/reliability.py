"""Pure data-reliability primitives for auditable application snapshots.

This module deliberately performs no network, database, or filesystem I/O.
Callers persist the returned dictionaries using their existing storage layer.
Every public result is JSON serialisable and timestamps are normalised to UTC.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from typing import Any


_VALID_SOURCE_METHODS = {"live", "fallback", "cache", "manual_import"}


def _utc_datetime(value: datetime | str | None, *, field: str) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{field} must not be empty.")
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


def _json_copy(value: Any, *, field: str) -> Any:
    """Return a defensive JSON copy while rejecting NaN and exotic values."""
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain only finite JSON values.") from exc


def _normalise_text(value: Any, *, field: str) -> str:
    text = " ".join(str(value or "").strip().split())
    if not text:
        raise ValueError(f"{field} is required.")
    return text


def _records(payload: Any, *, records_path: str | None) -> list[Mapping[str, Any]]:
    value = payload
    if records_path:
        for segment in records_path.split("."):
            if not isinstance(value, Mapping) or segment not in value:
                return []
            value = value[segment]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _present(value: Any) -> bool:
    if value is None or isinstance(value, bool) and value is False:
        return value is not None
    if isinstance(value, float) and not math.isfinite(value):
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def measure_completeness(
    payload: Any,
    *,
    records_path: str | None = None,
    required_fields: Sequence[str] = (),
    expected_keys: Sequence[str] | None = None,
    key_field: str = "ticker",
) -> dict[str, Any]:
    """Measure record/cell coverage without treating an empty portfolio as corrupt.

    ``expected_keys`` is optional.  When supplied, missing and unexpected record
    identifiers affect completeness.  Required-field coverage is calculated for
    records that are actually present.  Duplicate identifiers are reported and
    prevent the snapshot from being considered complete.
    """
    clean_payload = _json_copy(payload, field="payload")
    fields = tuple(
        dict.fromkeys(str(item).strip() for item in required_fields if str(item).strip())
    )
    rows = _records(clean_payload, records_path=records_path)

    expected = {str(item).strip().upper() for item in (expected_keys or ()) if str(item).strip()}
    identifiers: list[str] = []
    missing_cells: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        identifier = str(row.get(key_field) or "").strip().upper()
        if identifier:
            identifiers.append(identifier)
        row_label = identifier or f"row:{index + 1}"
        for field_name in fields:
            if field_name not in row or not _present(row.get(field_name)):
                missing_cells.append({"record": row_label, "field": field_name})

    present = set(identifiers)
    missing_keys = sorted(expected - present)
    unexpected_keys = sorted(present - expected) if expected_keys is not None else []
    duplicate_keys = sorted({item for item in identifiers if identifiers.count(item) > 1})

    expected_record_units = len(expected) if expected_keys is not None else len(rows)
    present_record_units = len(expected & present) if expected_keys is not None else len(rows)
    cell_units = len(rows) * len(fields)
    present_cell_units = cell_units - len(missing_cells)
    # Unexpected and duplicate identifiers are quality violations, not extra
    # evidence that can accidentally push coverage back to 100 percent.
    violation_units = len(unexpected_keys) + len(duplicate_keys)
    total_units = expected_record_units + cell_units + violation_units
    covered_units = present_record_units + present_cell_units
    completeness_pct = 100.0 if total_units == 0 else 100.0 * covered_units / total_units

    return {
        "completeness_pct": completeness_pct,
        "is_complete": (
            not missing_keys and not unexpected_keys and not missing_cells and not duplicate_keys
        ),
        "record_count": len(rows),
        "expected_record_count": len(expected) if expected_keys is not None else None,
        "required_fields": list(fields),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "duplicate_keys": duplicate_keys,
        "missing_cells": missing_cells,
    }


def create_data_snapshot(
    payload: Any,
    *,
    dataset: str,
    provider: str,
    observed_at: datetime | str,
    received_at: datetime | str | None = None,
    method: str = "live",
    source_reference: str = "",
    records_path: str | None = None,
    required_fields: Sequence[str] = (),
    expected_keys: Sequence[str] | None = None,
    key_field: str = "ticker",
    imported_by: str = "",
    notes: str = "",
) -> dict[str, Any]:
    """Create an immutable, integrity-hashed snapshot envelope."""
    if observed_at is None:
        raise ValueError("observed_at is required.")
    dataset_name = _normalise_text(dataset, field="dataset")
    provider_name = _normalise_text(provider, field="provider")
    source_method = str(method or "").strip().lower()
    if source_method not in _VALID_SOURCE_METHODS:
        raise ValueError(f"method must be one of {sorted(_VALID_SOURCE_METHODS)}.")
    importer = " ".join(str(imported_by or "").strip().split())
    if source_method == "manual_import" and not importer:
        raise ValueError("imported_by is required for manual imports.")

    clean_payload = _json_copy(payload, field="payload")
    observed = _timestamp(observed_at, field="observed_at")
    received = _timestamp(received_at, field="received_at")
    quality = measure_completeness(
        clean_payload,
        records_path=records_path,
        required_fields=required_fields,
        expected_keys=expected_keys,
        key_field=key_field,
    )
    source = {
        "provider": provider_name,
        "method": source_method,
        "reference": " ".join(str(source_reference or "").strip().split()),
        "imported_by": importer or None,
    }
    clean_notes = " ".join(str(notes or "").strip().split())
    digest_input = {
        "dataset": dataset_name,
        "observed_at": observed,
        "received_at": received,
        "source": source,
        "quality": quality,
        "notes": clean_notes,
        "payload": clean_payload,
    }
    canonical = json.dumps(digest_input, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "snapshot_id": f"snapshot_{digest[:20]}",
        "dataset": dataset_name,
        "observed_at": observed,
        "received_at": received,
        "source": source,
        "quality": quality,
        "integrity": {"algorithm": "sha256", "digest": digest},
        "notes": clean_notes,
        "payload": clean_payload,
    }


def import_manual_snapshot(
    payload: Any,
    *,
    dataset: str,
    imported_by: str,
    observed_at: datetime | str,
    received_at: datetime | str | None = None,
    source_reference: str = "",
    records_path: str | None = None,
    required_fields: Sequence[str] = (),
    expected_keys: Sequence[str] | None = None,
    key_field: str = "ticker",
    notes: str = "",
) -> dict[str, Any]:
    """Create a provenance-complete manual snapshot for provider outages."""
    return create_data_snapshot(
        payload,
        dataset=dataset,
        provider="manual",
        observed_at=observed_at,
        received_at=received_at,
        method="manual_import",
        source_reference=source_reference,
        records_path=records_path,
        required_fields=required_fields,
        expected_keys=expected_keys,
        key_field=key_field,
        imported_by=imported_by,
        notes=notes,
    )


def verify_snapshot_integrity(snapshot: Mapping[str, Any]) -> bool:
    """Verify that core payload/provenance fields still match the stored hash."""
    try:
        integrity = snapshot.get("integrity")
        if not isinstance(integrity, Mapping):
            return False
        digest_input = {
            "dataset": snapshot["dataset"],
            "observed_at": snapshot["observed_at"],
            "received_at": snapshot["received_at"],
            "source": snapshot["source"],
            "quality": snapshot["quality"],
            "notes": snapshot["notes"],
            "payload": snapshot["payload"],
        }
        canonical = json.dumps(
            digest_input,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return (
            integrity.get("algorithm") == "sha256"
            and actual == integrity.get("digest")
            and snapshot.get("snapshot_id") == f"snapshot_{actual[:20]}"
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def assess_snapshot(
    snapshot: Mapping[str, Any],
    *,
    now: datetime | str | None = None,
    max_age_seconds: float = 86_400,
    min_completeness_pct: float = 100.0,
    future_tolerance_seconds: float = 300,
) -> dict[str, Any]:
    """Return freshness, completeness, provenance, and integrity badges."""
    current = _utc_datetime(now, field="now")
    max_age = max(0.0, float(max_age_seconds))
    minimum = min(100.0, max(0.0, float(min_completeness_pct)))
    future_tolerance = max(0.0, float(future_tolerance_seconds))
    reasons: list[str] = []

    try:
        raw_observed = snapshot.get("observed_at")
        if raw_observed in (None, ""):
            raise ValueError("observed_at is required.")
        observed = _utc_datetime(raw_observed, field="observed_at")
        age_seconds = (current - observed).total_seconds()
    except (TypeError, ValueError):
        observed = None
        age_seconds = None
        reasons.append("invalid_observed_at")

    if age_seconds is None:
        freshness = "invalid"
    elif age_seconds < -future_tolerance:
        freshness = "future"
        reasons.append("observed_at_in_future")
    elif age_seconds > max_age:
        freshness = "stale"
        reasons.append("stale")
    else:
        freshness = "fresh"

    quality = snapshot.get("quality") if isinstance(snapshot.get("quality"), Mapping) else {}
    try:
        completeness = float(quality.get("completeness_pct", 0.0))
    except (TypeError, ValueError):
        completeness = 0.0
    if not math.isfinite(completeness):
        completeness = 0.0
    complete_enough = completeness >= minimum
    if not complete_enough:
        reasons.append("incomplete")

    integrity_valid = verify_snapshot_integrity(snapshot)
    if not integrity_valid:
        reasons.append("integrity_failed")
    source = snapshot.get("source") if isinstance(snapshot.get("source"), Mapping) else {}
    provider = str(source.get("provider") or "unknown")
    method = str(source.get("method") or "unknown")
    usable = integrity_valid and complete_enough and freshness not in {"invalid", "future"}
    return {
        "snapshot_id": snapshot.get("snapshot_id"),
        "usable": usable,
        "is_fresh": freshness == "fresh",
        "freshness": freshness,
        "age_seconds": age_seconds,
        "max_age_seconds": max_age,
        "completeness_pct": completeness,
        "min_completeness_pct": minimum,
        "complete_enough": complete_enough,
        "integrity_valid": integrity_valid,
        "source_badge": f"{provider}:{method}",
        "reason_codes": reasons,
    }


def initial_circuit_state(provider: str) -> dict[str, Any]:
    """Return a closed, JSON-only circuit-breaker state."""
    return {
        "provider": _normalise_text(provider, field="provider"),
        "state": "closed",
        "consecutive_failures": 0,
        "opened_at": None,
        "retry_at": None,
        "last_attempt_at": None,
        "last_success_at": None,
        "last_error": None,
    }


def circuit_request_decision(
    state: Mapping[str, Any],
    *,
    now: datetime | str | None = None,
) -> dict[str, Any]:
    """Decide whether a provider request may run, including half-open probes."""
    current = _utc_datetime(now, field="now")
    current_state = str(state.get("state") or "closed").lower()
    if current_state not in {"closed", "open", "half_open"}:
        current_state = "closed"
    retry_at = state.get("retry_at")
    if current_state == "open" and retry_at:
        try:
            retry = _utc_datetime(retry_at, field="retry_at")
        except (TypeError, ValueError):
            retry = current
        if current < retry:
            return {
                "allowed": False,
                "effective_state": "open",
                "retry_at": retry.isoformat(),
                "reason": "cooldown_active",
            }
        return {
            "allowed": True,
            "effective_state": "half_open",
            "retry_at": retry.isoformat(),
            "reason": "probe_after_cooldown",
        }
    return {
        "allowed": current_state != "half_open" or bool(state.get("probe_available", True)),
        "effective_state": current_state,
        "retry_at": retry_at,
        "reason": "request_allowed",
    }


def record_circuit_result(
    state: Mapping[str, Any] | None,
    *,
    succeeded: bool,
    now: datetime | str | None = None,
    error: str = "",
    failure_threshold: int = 3,
    cooldown_seconds: float = 300,
) -> dict[str, Any]:
    """Return the next circuit state after one provider attempt."""
    threshold = int(failure_threshold)
    if threshold < 1:
        raise ValueError("failure_threshold must be at least 1.")
    cooldown = max(0.0, float(cooldown_seconds))
    if state is None:
        raise ValueError("state is required; initialise it with initial_circuit_state().")
    updated = deepcopy(dict(state))
    provider = _normalise_text(updated.get("provider"), field="state.provider")
    attempted_at = _utc_datetime(now, field="now")
    updated["provider"] = provider
    updated["last_attempt_at"] = attempted_at.isoformat()
    if succeeded:
        updated.update(
            {
                "state": "closed",
                "consecutive_failures": 0,
                "opened_at": None,
                "retry_at": None,
                "last_success_at": attempted_at.isoformat(),
                "last_error": None,
            }
        )
        return _json_copy(updated, field="circuit state")

    failures = int(updated.get("consecutive_failures") or 0) + 1
    updated["consecutive_failures"] = failures
    updated["last_error"] = " ".join(str(error or "provider request failed").strip().split())
    if failures >= threshold or str(updated.get("state")) in {"open", "half_open"}:
        updated["state"] = "open"
        updated["opened_at"] = attempted_at.isoformat()
        updated["retry_at"] = (attempted_at + timedelta(seconds=cooldown)).isoformat()
    else:
        updated["state"] = "closed"
    return _json_copy(updated, field="circuit state")


def plan_provider_attempts(
    providers: Sequence[str],
    circuit_states: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    now: datetime | str | None = None,
) -> dict[str, Any]:
    """Build a primary-then-fallback fetch plan while skipping open circuits."""
    states = circuit_states or {}
    attempts: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for index, raw_provider in enumerate(providers):
        provider = _normalise_text(raw_provider, field="provider")
        state = states.get(provider) or initial_circuit_state(provider)
        decision = circuit_request_decision(state, now=now)
        item = {
            "provider": provider,
            "role": "primary" if index == 0 else "fallback",
            **decision,
        }
        (attempts if decision["allowed"] else skipped).append(item)
    return {
        "attempts": attempts,
        "skipped": skipped,
        "has_available_provider": bool(attempts),
    }


def select_reliable_snapshot(
    snapshots: Sequence[Mapping[str, Any]],
    *,
    now: datetime | str | None = None,
    max_age_seconds: float = 86_400,
    min_completeness_pct: float = 100.0,
    provider_priority: Sequence[str] = (),
    allow_last_known_good: bool = True,
) -> dict[str, Any]:
    """Select a fresh snapshot or explicitly degrade to the last known good.

    Freshness wins before provider priority: a current fallback-provider value
    is safer than stale primary-provider data.  Among equally fresh candidates,
    provider priority is respected and the newest observation wins.
    """
    priority = {str(provider): index for index, provider in enumerate(provider_priority)}
    assessed: list[tuple[Mapping[str, Any], dict[str, Any], datetime]] = []
    rejected: list[dict[str, Any]] = []
    for snapshot in snapshots:
        assessment = assess_snapshot(
            snapshot,
            now=now,
            max_age_seconds=max_age_seconds,
            min_completeness_pct=min_completeness_pct,
        )
        try:
            observed = _utc_datetime(snapshot.get("observed_at"), field="observed_at")
        except (TypeError, ValueError):
            rejected.append(assessment)
            continue
        if assessment["usable"]:
            assessed.append((snapshot, assessment, observed))
        else:
            rejected.append(assessment)

    def sort_key(
        item: tuple[Mapping[str, Any], dict[str, Any], datetime],
    ) -> tuple[int, float, float]:
        snapshot, assessment, observed = item
        provider = str(snapshot.get("source", {}).get("provider") or "")
        provider_rank = float(priority.get(provider, len(priority)))
        if assessment["is_fresh"]:
            return (0, provider_rank, -observed.timestamp())
        # "Last known good" means the newest usable historical observation;
        # provider priority only breaks an equal-timestamp tie.
        return (1, -observed.timestamp(), provider_rank)

    assessed.sort(key=sort_key)
    fresh = [item for item in assessed if item[1]["is_fresh"]]
    chosen = fresh[0] if fresh else assessed[0] if assessed and allow_last_known_good else None
    if chosen is None:
        return {
            "status": "unavailable",
            "snapshot": None,
            "selection": None,
            "rejected": rejected + [item[1] for item in assessed],
        }

    snapshot, assessment, _ = chosen
    used_lkg = not assessment["is_fresh"]
    selection = {
        **assessment,
        "used_last_known_good": used_lkg,
        "selection_reason": "last_known_good" if used_lkg else "fresh_best_available",
    }
    unselected = [item[1] for item in assessed if item[0] is not snapshot]
    return {
        "status": "degraded" if used_lkg else "ready",
        "snapshot": _json_copy(snapshot, field="snapshot"),
        "selection": selection,
        "rejected": rejected + unselected,
    }


__all__ = [
    "assess_snapshot",
    "circuit_request_decision",
    "create_data_snapshot",
    "import_manual_snapshot",
    "initial_circuit_state",
    "measure_completeness",
    "plan_provider_attempts",
    "record_circuit_result",
    "select_reliable_snapshot",
    "verify_snapshot_integrity",
]
