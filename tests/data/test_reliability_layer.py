from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json

import pytest

from src.data.reliability import (
    assess_snapshot,
    circuit_request_decision,
    create_data_snapshot,
    import_manual_snapshot,
    initial_circuit_state,
    measure_completeness,
    plan_provider_attempts,
    record_circuit_result,
    select_reliable_snapshot,
    verify_snapshot_integrity,
)


NOW = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)


def _snapshot(*, provider: str, observed_at: str, price: float = 100.0):
    return create_data_snapshot(
        {"items": [{"ticker": "AAA", "price": price}]},
        dataset="prices",
        provider=provider,
        observed_at=observed_at,
        received_at=NOW,
        records_path="items",
        required_fields=("ticker", "price"),
        expected_keys=("AAA",),
    )


def test_completeness_reports_missing_expected_records_fields_and_duplicates():
    quality = measure_completeness(
        {
            "items": [
                {"ticker": "AAA", "price": 10},
                {"ticker": "AAA", "price": None},
                {"ticker": "EXTRA", "price": 5},
            ]
        },
        records_path="items",
        required_fields=("ticker", "price"),
        expected_keys=("AAA", "BBB"),
    )

    assert quality["is_complete"] is False
    assert quality["missing_keys"] == ["BBB"]
    assert quality["unexpected_keys"] == ["EXTRA"]
    assert quality["duplicate_keys"] == ["AAA"]
    assert quality["missing_cells"] == [{"record": "AAA", "field": "price"}]
    assert 0 < quality["completeness_pct"] < 100


def test_manual_snapshot_has_provenance_hash_and_rejects_tampering():
    snapshot = import_manual_snapshot(
        {"positions": [{"ticker": "AAA", "quantity": 2}]},
        dataset="wins_portfolio",
        imported_by="Martin",
        observed_at="2026-08-15T10:00:00Z",
        received_at=NOW,
        source_reference="WInS export 2026-08-15.csv",
        records_path="positions",
        required_fields=("ticker", "quantity"),
        expected_keys=("AAA",),
        notes="Yahoo unavailable",
    )

    assert snapshot["source"] == {
        "provider": "manual",
        "method": "manual_import",
        "reference": "WInS export 2026-08-15.csv",
        "imported_by": "Martin",
    }
    assert verify_snapshot_integrity(snapshot) is True
    json.dumps(snapshot, allow_nan=False)

    tampered = deepcopy(snapshot)
    tampered["payload"]["positions"][0]["quantity"] = 999
    assert verify_snapshot_integrity(tampered) is False
    tampered_provenance = deepcopy(snapshot)
    tampered_provenance["source"]["imported_by"] = "Someone else"
    assert verify_snapshot_integrity(tampered_provenance) is False


def test_manual_snapshot_requires_a_named_importer():
    with pytest.raises(ValueError, match="imported_by"):
        import_manual_snapshot(
            [],
            dataset="portfolio",
            imported_by="",
            observed_at=NOW,
        )


def test_assessment_exposes_freshness_completeness_source_and_integrity():
    fresh = _snapshot(provider="primary", observed_at="2026-08-15T11:30:00Z")
    assessment = assess_snapshot(fresh, now=NOW, max_age_seconds=3600)

    assert assessment["usable"] is True
    assert assessment["freshness"] == "fresh"
    assert assessment["age_seconds"] == 1800
    assert assessment["completeness_pct"] == 100
    assert assessment["source_badge"] == "primary:live"

    stale = assess_snapshot(fresh, now="2026-08-16T12:00:00Z", max_age_seconds=3600)
    assert stale["usable"] is True
    assert stale["is_fresh"] is False
    assert stale["reason_codes"] == ["stale"]


def test_incomplete_and_future_snapshots_are_not_usable():
    incomplete = create_data_snapshot(
        {"items": [{"ticker": "AAA"}]},
        dataset="prices",
        provider="primary",
        observed_at=NOW,
        records_path="items",
        required_fields=("ticker", "price"),
    )
    future = _snapshot(provider="primary", observed_at="2026-08-15T13:00:01Z")

    incomplete_result = assess_snapshot(incomplete, now=NOW)
    future_result = assess_snapshot(future, now=NOW, future_tolerance_seconds=300)
    assert incomplete_result["usable"] is False
    assert "incomplete" in incomplete_result["reason_codes"]
    assert future_result["usable"] is False
    assert future_result["freshness"] == "future"


def test_selector_prefers_fresh_fallback_over_stale_primary_then_uses_lkg():
    stale_primary = _snapshot(provider="primary", observed_at="2026-08-14T08:00:00Z")
    fresh_fallback = _snapshot(provider="fallback", observed_at="2026-08-15T11:55:00Z", price=101)

    selected = select_reliable_snapshot(
        [stale_primary, fresh_fallback],
        now=NOW,
        max_age_seconds=3600,
        provider_priority=("primary", "fallback"),
    )
    assert selected["status"] == "ready"
    assert selected["snapshot"]["source"]["provider"] == "fallback"
    assert selected["selection"]["used_last_known_good"] is False

    lkg = select_reliable_snapshot(
        [stale_primary],
        now=NOW,
        max_age_seconds=3600,
        provider_priority=("primary",),
    )
    assert lkg["status"] == "degraded"
    assert lkg["selection"]["selection_reason"] == "last_known_good"

    unavailable = select_reliable_snapshot(
        [stale_primary],
        now=NOW,
        max_age_seconds=3600,
        allow_last_known_good=False,
    )
    assert unavailable["status"] == "unavailable"
    assert unavailable["snapshot"] is None


def test_circuit_breaker_opens_skips_primary_allows_probe_and_recovers():
    state = initial_circuit_state("Yahoo")
    state = record_circuit_result(
        state,
        succeeded=False,
        now="2026-08-15T12:00:00Z",
        error="429 Too Many Requests",
        failure_threshold=2,
        cooldown_seconds=60,
    )
    assert state["state"] == "closed"
    state = record_circuit_result(
        state,
        succeeded=False,
        now="2026-08-15T12:00:10Z",
        error="429 Too Many Requests",
        failure_threshold=2,
        cooldown_seconds=60,
    )
    assert state["state"] == "open"
    assert circuit_request_decision(state, now="2026-08-15T12:00:30Z")["allowed"] is False
    probe = circuit_request_decision(state, now="2026-08-15T12:01:10Z")
    assert probe["allowed"] is True
    assert probe["effective_state"] == "half_open"

    plan = plan_provider_attempts(
        ["Yahoo", "Stooq"],
        {"Yahoo": state},
        now="2026-08-15T12:00:30Z",
    )
    assert [item["provider"] for item in plan["attempts"]] == ["Stooq"]
    assert [item["provider"] for item in plan["skipped"]] == ["Yahoo"]

    recovered = record_circuit_result(
        state,
        succeeded=True,
        now="2026-08-15T12:01:10Z",
    )
    assert recovered["state"] == "closed"
    assert recovered["consecutive_failures"] == 0
    assert recovered["last_error"] is None


def test_snapshot_creation_rejects_nan_so_every_result_is_strict_json():
    with pytest.raises(ValueError, match="finite JSON"):
        create_data_snapshot(
            {"price": float("nan")},
            dataset="prices",
            provider="primary",
            observed_at=NOW,
        )


def test_malformed_snapshot_assessment_is_blocking_not_an_exception():
    malformed = {
        "snapshot_id": "fake",
        "observed_at": None,
        "quality": {"completeness_pct": float("nan")},
        "integrity": "not-an-object",
    }

    result = assess_snapshot(malformed, now=NOW)

    assert result["usable"] is False
    assert result["freshness"] == "invalid"
    assert result["completeness_pct"] == 0
    assert result["integrity_valid"] is False
