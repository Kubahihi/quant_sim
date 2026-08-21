from __future__ import annotations

from copy import deepcopy

import pytest

from src.portfolio_tracker.portfolio_pipeline import (
    build_live_portfolio_pipeline,
    create_portfolio_snapshot,
)
from src.portfolio_tracker.reconciliation_ledger import (
    append_reconciliation,
    latest_reconciliation,
    migrate_reconciliation_ledger,
    new_reconciliation_ledger,
    sign_off_reconciliation,
)


NOW = "2026-08-15T12:00:00Z"
_MISSING = object()


def _positions() -> list[dict[str, object]]:
    return [
        {
            "ticker": "AAA",
            "quantity": 10,
            "entry_price": 100,
            "last_price": 110,
            "security_type": "Stock",
            "currency": "USD",
        }
    ]


def _wins_snapshot() -> dict[str, object]:
    return create_portfolio_snapshot(
        _positions(),
        provider="WInS",
        observed_at="2026-08-15T11:30:00Z",
        received_at=NOW,
        source_reference="legacy WInS export",
        expected_tickers=("AAA",),
        cash_value=100,
    )


def _approved_ledger() -> tuple[dict[str, object], dict[str, object]]:
    wins = _wins_snapshot()
    ledger = append_reconciliation(
        new_reconciliation_ledger(),
        wins,
        _positions(),
        owner="Lukas",
        performed_at=NOW,
    )
    reconciliation_id = latest_reconciliation(ledger)["reconciliation_id"]
    ledger = sign_off_reconciliation(
        ledger,
        reconciliation_id,
        decision="approved",
        signed_off_by="Jakub",
        signed_off_at=NOW,
    )
    return ledger, wins


@pytest.mark.parametrize("legacy_marker", [_MISSING, None, "", 0, "0", 0.0])
def test_all_unversioned_markers_migrate_without_rewriting_audit_data(legacy_marker):
    current, _ = _approved_ledger()
    legacy = deepcopy(current)
    if legacy_marker is _MISSING:
        legacy.pop("schema_version")
    else:
        legacy["schema_version"] = legacy_marker
    legacy["legacy_extension"] = {"source": "original workspace", "sequence": [3, 1, 2]}
    before = deepcopy(legacy)

    migrated = migrate_reconciliation_ledger(legacy)

    assert legacy == before
    assert migrated["schema_version"] == 1
    assert migrated["reconciliations"] == current["reconciliations"]
    assert migrated["events"] == current["events"]
    assert migrated["legacy_extension"] == before["legacy_extension"]


def test_empty_pre_schema_workspace_no_longer_breaks_canonical_pipeline():
    legacy_ledger = {"reconciliations": [], "events": []}

    pipeline = build_live_portfolio_pipeline(
        [],
        legacy_ledger,
        mandate={},
        rulebook={},
        expected_return_assumptions={},
        now=NOW,
    )

    assert pipeline["status"] == "blocked"
    assert pipeline["reconciliation_gate"]["blockers"] == ["no_reconciliation"]
    assert legacy_ledger == {"reconciliations": [], "events": []}


def test_nonempty_legacy_ledger_still_authorises_the_same_wins_snapshot():
    current, wins = _approved_ledger()
    legacy = deepcopy(current)
    legacy.pop("schema_version")

    pipeline = build_live_portfolio_pipeline(
        [wins],
        legacy,
        mandate={"status": "active", "mandate_id": "m1"},
        rulebook={"status": "active", "rulebook_id": "r1"},
        expected_return_assumptions={"status": "active", "values": {"AAA": 0.08}},
        now=NOW,
    )

    assert pipeline["status"] == "ready"
    assert pipeline["authority"] == "wins_reconciled"
    assert pipeline["canonical_snapshot"]["snapshot_id"] == wins["snapshot_id"]
    assert pipeline["reconciliation_gate"]["ready"] is True


def test_first_write_upgrades_legacy_ledger_and_preserves_existing_history():
    current, _ = _approved_ledger()
    legacy = deepcopy(current)
    legacy["schema_version"] = 0
    previous_reconciliations = deepcopy(legacy["reconciliations"])
    previous_events = deepcopy(legacy["events"])
    newer_wins = create_portfolio_snapshot(
        _positions(),
        provider="WInS",
        observed_at="2026-08-15T12:30:00Z",
        received_at="2026-08-15T12:31:00Z",
        expected_tickers=("AAA",),
    )

    updated = append_reconciliation(
        legacy,
        newer_wins,
        _positions(),
        owner="Martin",
        performed_at="2026-08-15T12:31:00Z",
    )

    assert updated["schema_version"] == 1
    assert updated["reconciliations"][:-1] == previous_reconciliations
    assert updated["events"] == previous_events
    assert legacy["schema_version"] == 0


def test_current_integer_like_marker_is_normalised_but_future_schema_fails_closed():
    assert migrate_reconciliation_ledger(
        {"schema_version": "1", "reconciliations": [], "events": []}
    )["schema_version"] == 1

    with pytest.raises(ValueError, match="newer than supported"):
        migrate_reconciliation_ledger(
            {"schema_version": 2, "reconciliations": [], "events": []}
        )


@pytest.mark.parametrize(
    "ledger, message",
    [
        ({"schema_version": True, "reconciliations": [], "events": []}, "integer"),
        ({"schema_version": -1, "reconciliations": [], "events": []}, "negative"),
        ({"schema_version": 1, "reconciliations": {}, "events": []}, "reconciliations"),
        ({"schema_version": 1, "reconciliations": [], "events": {}}, "events"),
    ],
)
def test_migration_does_not_mask_corrupt_or_unsupported_ledgers(ledger, message):
    with pytest.raises(ValueError, match=message):
        migrate_reconciliation_ledger(ledger)
