from __future__ import annotations

from copy import deepcopy
import json

import pytest

from src.portfolio_tracker.portfolio_pipeline import create_portfolio_snapshot
from src.portfolio_tracker.reconciliation_ledger import (
    append_reconciliation,
    assign_exception,
    latest_reconciliation,
    materialize_reconciliation,
    new_reconciliation_ledger,
    reconciliation_history,
    reconciliation_readiness_gate,
    resolve_exception,
    sign_off_exception,
    sign_off_reconciliation,
)


NOW = "2026-08-15T12:00:00Z"


def _wins_snapshot(
    *,
    observed_at: str = "2026-08-15T11:30:00Z",
    quantity: float = 10,
    cash_value: float = 0.0,
):
    return create_portfolio_snapshot(
        [
            {
                "ticker": "AAA",
                "quantity": quantity,
                "total_cost": quantity * 100,
                "current_value": quantity * 110,
                "security_type": "Stock",
            }
        ],
        provider="WInS",
        observed_at=observed_at,
        received_at=NOW,
        source_reference="WInS account export",
        expected_tickers=("AAA",),
        cash_value=cash_value,
    )


def _tracked(*, quantity: float = 10):
    return [
        {
            "ticker": "AAA",
            "quantity": quantity,
            "entry_price": 100,
            "last_price": 110,
            "security_type": "Equity",
        }
    ]


def _tracked_snapshot(*, cash_value: float) -> dict:
    return create_portfolio_snapshot(
        _tracked(),
        provider="Portfolio Tracker",
        observed_at=NOW,
        received_at=NOW,
        source_reference="competition_positions",
        expected_tickers=("AAA",),
        cash_value=cash_value,
    )


def test_clean_reconciliation_needs_independent_signoff_before_readiness():
    wins = _wins_snapshot()
    empty = new_reconciliation_ledger()
    original = deepcopy(empty)
    ledger = append_reconciliation(
        empty,
        wins,
        _tracked(),
        owner="Lukas",
        performed_at=NOW,
    )

    assert empty == original
    assert len(ledger["reconciliations"]) == 1
    record = latest_reconciliation(ledger)
    assert record["base_is_clean"] is True
    assert record["exceptions"] == []
    assert record["workflow_status"] == "awaiting_sign_off"

    gate = reconciliation_readiness_gate(ledger, now=NOW)
    assert gate["ready"] is False
    assert gate["blockers"] == ["missing_sign_off"]

    with pytest.raises(ValueError, match="different person"):
        sign_off_reconciliation(
            ledger,
            record["reconciliation_id"],
            decision="approved",
            signed_off_by="Lukas",
            signed_off_at=NOW,
        )

    ledger = sign_off_reconciliation(
        ledger,
        record["reconciliation_id"],
        decision="approved",
        signed_off_by="Jakub",
        note="WInS and tracker agree",
        signed_off_at=NOW,
    )
    gate = reconciliation_readiness_gate(
        ledger,
        now=NOW,
        expected_wins_snapshot_id=wins["snapshot_id"],
    )
    assert gate["ready"] is True
    assert gate["signed_off_by"] == "Jakub"
    json.dumps(ledger, allow_nan=False)


def test_cash_mismatch_blocks_clean_reconciliation_even_when_positions_match():
    wins = _wins_snapshot(cash_value=100.0)
    tracked = _tracked_snapshot(cash_value=80.0)
    ledger = append_reconciliation(
        new_reconciliation_ledger(),
        wins,
        tracked,
        owner="Lukas",
        performed_at=NOW,
        cash_tolerance=0.01,
    )

    record = latest_reconciliation(ledger)
    comparison = record["result"]["cash_comparison"]
    assert record["result"]["position_status"] == "reconciled"
    assert comparison == {
        "status": "difference",
        "is_match": False,
        "amount_match": False,
        "currency_match": True,
        "difference": pytest.approx(20.0),
        "tolerance": pytest.approx(0.01),
        "wins": {
            "value_present": True,
            "value": pytest.approx(100.0),
            "currency": "USD",
            "value_source": "payload.cash_value",
            "currency_source": "payload.base_currency",
        },
        "tracked": {
            "value_present": True,
            "value": pytest.approx(80.0),
            "currency": "USD",
            "value_source": "payload.cash_value",
            "currency_source": "payload.base_currency",
        },
        "reason": "cash_balance_mismatch",
    }
    assert record["base_status"] == "differences"
    assert record["base_is_clean"] is False
    assert [item["category"] for item in record["exceptions"]] == ["cash_mismatch"]

    gate = reconciliation_readiness_gate(ledger, now=NOW)
    assert gate["ready"] is False
    assert "snapshot_has_differences" in gate["blockers"]
    assert "open_exceptions" in gate["blockers"]


def test_cash_within_tolerance_can_be_signed_and_report_ready():
    wins = _wins_snapshot(cash_value=100.0)
    ledger = append_reconciliation(
        new_reconciliation_ledger(),
        wins,
        _tracked_snapshot(cash_value=99.995),
        owner="Lukas",
        performed_at=NOW,
        cash_tolerance=0.01,
    )
    record = latest_reconciliation(ledger)

    assert record["result"]["cash_comparison"]["status"] == "matched"
    assert record["result"]["cash_comparison"]["difference"] == pytest.approx(0.005)
    assert record["base_is_clean"] is True
    assert record["exceptions"] == []

    ledger = sign_off_reconciliation(
        ledger,
        record["reconciliation_id"],
        decision="approved",
        signed_off_by="Jakub",
        signed_off_at=NOW,
    )
    assert reconciliation_readiness_gate(
        ledger,
        now=NOW,
        expected_wins_snapshot_id=wins["snapshot_id"],
    )["ready"] is True


def test_materialization_does_not_backfill_cash_into_legacy_records():
    ledger = append_reconciliation(
        new_reconciliation_ledger(),
        _wins_snapshot(),
        _tracked(),
        owner="Lukas",
        performed_at=NOW,
    )
    legacy = deepcopy(ledger)
    stored = legacy["reconciliations"][0]
    stored["result"].pop("cash_comparison")
    stored["result"].pop("position_status")
    stored["result"]["summary"].pop("cash_status")
    stored["result"]["summary"].pop("position_status")
    stored["tolerances"].pop("cash")
    before = deepcopy(legacy)

    materialized = materialize_reconciliation(legacy, stored["reconciliation_id"])

    assert legacy == before
    assert "cash_comparison" not in materialized["result"]
    assert "cash" not in materialized["tolerances"]
    assert materialized["base_is_clean"] is True


def test_dirty_snapshot_creates_owned_exception_resolution_and_signoff_audit():
    ledger = append_reconciliation(
        new_reconciliation_ledger(),
        _wins_snapshot(quantity=11),
        _tracked(quantity=10),
        owner="Lukas",
        performed_at=NOW,
    )
    record = latest_reconciliation(ledger)
    assert record["base_status"] == "differences"
    assert len(record["exceptions"]) == 1
    exception = record["exceptions"][0]
    assert exception["category"] == "position_mismatch"
    assert exception["owner"] == "Lukas"
    assert exception["status"] == "open"

    before_events = deepcopy(ledger["events"])
    ledger = assign_exception(
        ledger,
        record["reconciliation_id"],
        exception["exception_id"],
        owner="Martin",
        assigned_by="Lukas",
        assigned_at="2026-08-15T12:01:00Z",
    )
    assert before_events == []
    ledger = resolve_exception(
        ledger,
        record["reconciliation_id"],
        exception["exception_id"],
        resolution_type="tracker_correction_required",
        summary="Tracker is missing one executed share.",
        resolved_by="Martin",
        evidence_refs=("wins://trade/123",),
        resolved_at="2026-08-15T12:02:00Z",
    )
    pending = materialize_reconciliation(ledger, record["reconciliation_id"])
    assert pending["exceptions"][0]["owner"] == "Martin"
    assert pending["exceptions"][0]["status"] == "pending_sign_off"

    with pytest.raises(ValueError, match="different person"):
        sign_off_exception(
            ledger,
            record["reconciliation_id"],
            exception["exception_id"],
            decision="approved",
            signed_off_by="Martin",
            signed_off_at="2026-08-15T12:03:00Z",
        )

    ledger = sign_off_exception(
        ledger,
        record["reconciliation_id"],
        exception["exception_id"],
        decision="approved",
        signed_off_by="Jakub",
        signed_off_at="2026-08-15T12:03:00Z",
    )
    current = materialize_reconciliation(ledger, record["reconciliation_id"])
    assert current["exceptions"][0]["status"] == "closed"
    assert current["open_exception_count"] == 0

    # Documenting a mismatch does not make an inconsistent snapshot clean.
    with pytest.raises(ValueError, match="rerun"):
        sign_off_reconciliation(
            ledger,
            record["reconciliation_id"],
            decision="approved",
            signed_off_by="Jakub",
            signed_off_at="2026-08-15T12:04:00Z",
        )
    gate = reconciliation_readiness_gate(ledger, now=NOW)
    assert gate["ready"] is False
    assert "snapshot_has_differences" in gate["blockers"]


def test_new_clean_reconciliation_supersedes_dirty_history_without_overwriting_it():
    dirty_wins = _wins_snapshot(quantity=11)
    ledger = append_reconciliation(
        new_reconciliation_ledger(),
        dirty_wins,
        _tracked(quantity=10),
        owner="Lukas",
        performed_at="2026-08-15T10:00:00Z",
    )
    dirty_record = latest_reconciliation(ledger)
    clean_wins = _wins_snapshot(observed_at="2026-08-15T11:45:00Z", quantity=10)
    ledger = append_reconciliation(
        ledger,
        clean_wins,
        _tracked(quantity=10),
        owner="Lukas",
        performed_at="2026-08-15T12:00:00Z",
    )
    clean_record = latest_reconciliation(ledger)

    assert clean_record["supersedes_reconciliation_id"] == dirty_record["reconciliation_id"]
    assert len(ledger["reconciliations"]) == 2
    ledger = sign_off_reconciliation(
        ledger,
        clean_record["reconciliation_id"],
        decision="approved",
        signed_off_by="Jakub",
        signed_off_at="2026-08-15T12:01:00Z",
    )
    history = reconciliation_history(ledger)
    assert [item["base_status"] for item in history] == ["reconciled", "differences"]
    assert history[0]["workflow_status"] == "approved"
    assert history[1]["exception_count"] == 1


def test_gate_blocks_stale_or_not_latest_wins_snapshot():
    wins = _wins_snapshot(observed_at="2026-08-14T08:00:00Z")
    ledger = append_reconciliation(
        new_reconciliation_ledger(),
        wins,
        _tracked(),
        owner="Lukas",
        performed_at="2026-08-14T08:05:00Z",
    )
    record = latest_reconciliation(ledger)
    ledger = sign_off_reconciliation(
        ledger,
        record["reconciliation_id"],
        decision="approved",
        signed_off_by="Jakub",
        signed_off_at="2026-08-14T08:06:00Z",
    )

    gate = reconciliation_readiness_gate(
        ledger,
        now=NOW,
        max_age_seconds=3600,
        expected_wins_snapshot_id="snapshot_newer",
    )
    assert gate["ready"] is False
    assert gate["blockers"] == ["reconciliation_stale", "newer_snapshot_not_reconciled"]


def test_missing_and_extra_positions_create_distinct_exceptions():
    wins = create_portfolio_snapshot(
        [
            {"ticker": "AAA", "quantity": 1, "current_value": 10, "total_cost": 9},
            {"ticker": "EXTRA", "quantity": 1, "current_value": 20, "total_cost": 18},
        ],
        provider="WInS",
        observed_at=NOW,
        expected_tickers=("AAA", "EXTRA"),
    )
    tracked = [
        {"ticker": "AAA", "quantity": 1, "last_price": 10, "entry_price": 9},
        {"ticker": "MISSING", "quantity": 2, "last_price": 5, "entry_price": 4},
    ]
    ledger = append_reconciliation(
        new_reconciliation_ledger(), wins, tracked, owner="Lukas", performed_at=NOW
    )
    categories = {item["category"] for item in latest_reconciliation(ledger)["exceptions"]}
    assert categories == {"missing_in_wins", "extra_in_wins", "incomplete_position_data"}
