from __future__ import annotations

import json

import pytest

from src.portfolio_tracker.portfolio_pipeline import (
    ANALYSIS_CONSUMERS,
    build_live_portfolio_pipeline,
    create_portfolio_snapshot,
    materialize_consumer_input,
    normalize_portfolio_positions,
    snapshot_badges,
)
from src.portfolio_tracker.reconciliation_ledger import (
    append_reconciliation,
    latest_reconciliation,
    new_reconciliation_ledger,
    sign_off_reconciliation,
)


NOW = "2026-08-15T12:00:00Z"
MANDATE = {"mandate_id": "m1", "status": "active", "max_drawdown": 0.15}
RULEBOOK = {"rulebook_id": "r1", "status": "active", "max_position": 0.10}
RETURNS = {"assumption_set_id": "er1", "status": "active", "values": {"AAA": 0.08}}


def _positions(quantity: float = 10):
    return [
        {
            "ticker": "AAA",
            "quantity": quantity,
            "entry_price": 100,
            "last_price": 110,
            "security_type": "Stock",
            "currency": "USD",
        }
    ]


def _wins(*, observed_at: str = "2026-08-15T11:30:00Z", quantity: float = 10):
    return create_portfolio_snapshot(
        _positions(quantity),
        provider="WInS",
        observed_at=observed_at,
        received_at=NOW,
        source_reference="WInS export",
        expected_tickers=("AAA",),
        cash_value=100,
    )


def _approved_ledger(wins):
    ledger = append_reconciliation(
        new_reconciliation_ledger(),
        wins,
        _positions(),
        owner="Lukas",
        performed_at=NOW,
    )
    reconciliation_id = latest_reconciliation(ledger)["reconciliation_id"]
    return sign_off_reconciliation(
        ledger,
        reconciliation_id,
        decision="approved",
        signed_off_by="Jakub",
        signed_off_at=NOW,
    )


def test_position_normalisation_aggregates_lots_and_derives_values():
    result = normalize_portfolio_positions(
        [
            {
                "symbol": " aaa ",
                "shares": 2,
                "cost_basis": 90,
                "price": 100,
                "type": "Stock",
                "currency": "usd",
            },
            {
                "ticker": "AAA",
                "quantity": 3,
                "entry_price": 110,
                "current_value": 360,
                "security_type": "Stock",
                "currency": "USD",
            },
        ]
    )

    assert result == [
        {
            "ticker": "AAA",
            "quantity": 5.0,
            "unit_cost": pytest.approx(102.0),
            "total_cost": 510.0,
            "current_price": pytest.approx(112.0),
            "market_value": 560.0,
            "asset_type": "Stock",
            "currency": "USD",
            "source_lot_count": 2,
        }
    ]


def test_one_reconciled_snapshot_feeds_every_consumer_with_strict_contexts():
    wins = _wins()
    ledger = _approved_ledger(wins)
    pipeline = build_live_portfolio_pipeline(
        [wins],
        ledger,
        mandate=MANDATE,
        rulebook=RULEBOOK,
        expected_return_assumptions=RETURNS,
        now=NOW,
    )

    assert pipeline["status"] == "ready"
    assert pipeline["authority"] == "wins_reconciled"
    assert pipeline["reconciliation_gate"]["ready"] is True
    assert pipeline["last_known_good"] is False
    assert {binding["snapshot_id"] for binding in pipeline["consumer_bindings"].values()} == {
        wins["snapshot_id"]
    }
    assert all(binding["allowed"] for binding in pipeline["consumer_bindings"].values())

    inputs = [materialize_consumer_input(pipeline, name) for name in ANALYSIS_CONSUMERS]
    assert {item["portfolio_snapshot_id"] for item in inputs} == {wins["snapshot_id"]}
    assert all(item["portfolio"] == wins["payload"] for item in inputs)
    assert inputs[1]["rulebook"] == RULEBOOK
    assert wins["payload"]["fx_exposures"]["USD"] == pytest.approx(1_200)
    json.dumps(pipeline, allow_nan=False)


def test_new_dirty_wins_snapshot_keeps_analysis_on_lkg_but_blocks_reporting():
    clean = _wins(observed_at="2026-08-15T10:00:00Z")
    ledger = _approved_ledger(clean)
    dirty = _wins(observed_at="2026-08-15T11:45:00Z", quantity=11)
    ledger = append_reconciliation(
        ledger,
        dirty,
        _positions(quantity=10),
        owner="Lukas",
        performed_at="2026-08-15T12:01:00Z",
    )

    pipeline = build_live_portfolio_pipeline(
        [clean, dirty],
        ledger,
        mandate=MANDATE,
        rulebook=RULEBOOK,
        expected_return_assumptions=RETURNS,
        now="2026-08-15T12:05:00Z",
    )

    assert pipeline["status"] == "degraded"
    assert pipeline["canonical_snapshot"]["snapshot_id"] == clean["snapshot_id"]
    assert pipeline["latest_wins_snapshot_id"] == dirty["snapshot_id"]
    assert pipeline["last_known_good"] is True
    assert pipeline["consumer_bindings"]["quant"]["allowed"] is True
    assert pipeline["consumer_bindings"]["risk"]["allowed"] is True
    assert pipeline["consumer_bindings"]["reporting"] == {
        "snapshot_id": clean["snapshot_id"],
        "allowed": False,
        "blockers": ["latest_wins_reconciliation_not_clean"],
    }
    assert "snapshot_has_differences" in pipeline["reconciliation_gate"]["blockers"]


def test_tracker_snapshot_is_provisional_and_cannot_make_report_ready():
    tracker = create_portfolio_snapshot(
        _positions(),
        provider="tracker",
        observed_at="2026-08-15T11:55:00Z",
        received_at=NOW,
        expected_tickers=("AAA",),
    )
    pipeline = build_live_portfolio_pipeline(
        [tracker],
        new_reconciliation_ledger(),
        mandate=MANDATE,
        rulebook=RULEBOOK,
        expected_return_assumptions=RETURNS,
        now=NOW,
    )

    assert pipeline["authority"] == "provisional"
    assert pipeline["status"] == "degraded"
    assert pipeline["consumer_bindings"]["tracker"]["allowed"] is True
    assert pipeline["consumer_bindings"]["risk"]["allowed"] is True
    assert pipeline["consumer_bindings"]["reporting"]["allowed"] is False
    assert pipeline["reconciliation_gate"]["blockers"] == ["no_reconciliation"]


def test_rulebook_and_expected_returns_are_mandatory_not_optional_toggles():
    tracker = create_portfolio_snapshot(
        _positions(),
        provider="tracker",
        observed_at=NOW,
        expected_tickers=("AAA",),
    )
    pipeline = build_live_portfolio_pipeline(
        [tracker],
        new_reconciliation_ledger(),
        mandate=MANDATE,
        rulebook=None,
        expected_return_assumptions=None,
        now=NOW,
    )

    assert pipeline["consumer_bindings"]["tracker"]["allowed"] is True
    assert pipeline["consumer_bindings"]["risk"]["blockers"] == ["active_rulebook_missing"]
    assert pipeline["consumer_bindings"]["quant"]["blockers"] == [
        "active_rulebook_missing",
        "expected_return_assumptions_missing",
    ]


def test_stale_signed_wins_snapshot_is_lkg_and_report_gate_is_blocked():
    wins = _wins(observed_at="2026-08-14T08:00:00Z")
    ledger = _approved_ledger(wins)
    pipeline = build_live_portfolio_pipeline(
        [wins],
        ledger,
        mandate=MANDATE,
        rulebook=RULEBOOK,
        expected_return_assumptions=RETURNS,
        now=NOW,
        max_age_seconds=3600,
    )

    assert pipeline["status"] == "degraded"
    assert pipeline["last_known_good"] is True
    assert pipeline["selection"]["selection_reason"] == "last_known_good"
    assert pipeline["consumer_bindings"]["reporting"]["blockers"] == [
        "snapshot_stale",
        "latest_wins_reconciliation_not_clean",
    ]
    badges = snapshot_badges(wins, now=NOW, max_age_seconds=3600)
    assert badges["source"] == "WInS:live"
    assert badges["freshness"] == "stale"
    assert badges["status"] == "degraded"


def test_manual_wins_export_can_complete_pipeline_during_provider_outage():
    wins = create_portfolio_snapshot(
        _positions(),
        provider="WInS",
        method="manual_import",
        imported_by="Martin",
        observed_at="2026-08-15T11:30:00Z",
        received_at=NOW,
        source_reference="WInS manual CSV export",
        expected_tickers=("AAA",),
    )
    ledger = _approved_ledger(wins)
    pipeline = build_live_portfolio_pipeline(
        [wins],
        ledger,
        mandate=MANDATE,
        rulebook=RULEBOOK,
        expected_return_assumptions=RETURNS,
        now=NOW,
    )

    assert pipeline["status"] == "ready"
    consumer = materialize_consumer_input(pipeline, "risk")
    assert consumer["portfolio_source"]["method"] == "manual_import"
    assert consumer["portfolio_source"]["imported_by"] == "Martin"


def test_pipeline_with_no_usable_snapshot_blocks_every_consumer():
    pipeline = build_live_portfolio_pipeline(
        [],
        new_reconciliation_ledger(),
        mandate=MANDATE,
        rulebook=RULEBOOK,
        expected_return_assumptions=RETURNS,
        now=NOW,
    )
    assert pipeline["status"] == "blocked"
    assert pipeline["canonical_snapshot"] is None
    assert all(not binding["allowed"] for binding in pipeline["consumer_bindings"].values())
    with pytest.raises(ValueError, match="consumer must be"):
        materialize_consumer_input(pipeline, "unknown")


def test_missing_currency_is_not_silently_assumed_to_be_usd():
    snapshot = create_portfolio_snapshot(
        [{"ticker": "AAA", "quantity": 1, "last_price": 10}],
        provider="tracker",
        observed_at=NOW,
        expected_tickers=("AAA",),
    )

    assert snapshot["payload"]["positions"][0]["currency"] is None
    assert snapshot["payload"]["fx_exposures"] == {"UNKNOWN": 10.0, "USD": 0.0}
    assert snapshot["quality"]["is_complete"] is False
    assert snapshot["quality"]["missing_cells"] == [{"record": "AAA", "field": "currency"}]
