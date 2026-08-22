from __future__ import annotations

from inspect import getsource
import sqlite3

import pytest

import src.portfolio_tracker.investment_lifecycle_store as lifecycle_store
import src.portfolio_tracker.security_dossier_store as dossier_store
from ui.investment_os import (
    _activate_reconciled_tracker_position,
    _committee_roster,
    _payload_number,
    _record_wins_execution_and_stage_tracker_position,
    _render_security_dossier_fields,
    _security_dossier_payload,
    _stage_pending_tracker_position,
    _tracker_cash_value,
    _tracker_snapshot_rows,
    render_investment_committee,
    render_security_dossiers,
)


TEAM = [
    {"username": "Jakub", "role": "Co-Captain"},
    {"username": "Lukas", "role": "Geopolitics"},
    {"username": "Martin", "role": "Risk"},
    {"username": "Matej", "role": "Co-Captain"},
]


def test_canonical_dossier_payload_covers_research_monitoring_and_valuation():
    payload = _security_dossier_payload(
        {
            "sector": "Technology",
            "asset_type": "Stock",
            "owner": "Jakub",
            "client_goal": "Long-term growth",
            "portfolio_role": "Core compounder",
            "conviction": 4,
            "tags": "quality, AI, Quality",
            "thesis": "Durable cash-flow compounding is underappreciated.",
            "why_now": "The investment cycle is becoming visible in revenue.",
            "value_drivers": "Recurring revenue\nOperating leverage",
            "base_case": "Margins expand with disciplined reinvestment.",
            "bull_case": "Faster adoption",
            "bear_case": "Returns on capex disappoint",
            "counter_thesis": "Competition captures the economics.",
            "risks": "Regulation\nExecution",
            "reference_price": 100,
            "fair_value_bear": 80,
            "fair_value_base": 125,
            "fair_value_bull": 160,
            "fair_value_currency": "usd",
            "valuation_as_of": "2026-08-20",
            "catalysts": "Quarterly KPI\nProduct launch",
            "invalidation_condition": "Two quarters below the KPI floor",
            "sell_discipline": "Review on invalidation or valuation",
            "next_review_at": "2026-09-30",
            "evidence_refs": "10-K p.42\nindustry-source-2",
        },
        existing={"legacy_note": "preserve me", "primary_goal": "old duplicate"},
    )

    assert payload["sector"] == "Technology"
    assert payload["client_goal"] == "Long-term growth"
    assert payload["conviction"] == 4
    assert payload["tags"] == ["quality", "AI"]
    assert payload["why_now"].startswith("The investment cycle")
    assert payload["value_drivers"] == ["Recurring revenue", "Operating leverage"]
    assert payload["base_case"].startswith("Margins expand")
    assert payload["counter_thesis"].startswith("Competition")
    assert payload["fair_value_bear"] == 80.0
    assert payload["fair_value_base"] == 125.0
    assert payload["fair_value_bull"] == 160.0
    assert payload["margin_of_safety_pct"] == 25.0
    assert payload["next_review_at"] == "2026-09-30"
    assert payload["legacy_note"] == "preserve me"
    assert "primary_goal" not in payload


def test_canonical_dossier_rejects_an_incoherent_valuation_range():
    with pytest.raises(ValueError, match="bear ≤ base ≤ bull"):
        _security_dossier_payload(
            {
                "conviction": 3,
                "fair_value_bear": 120,
                "fair_value_base": 100,
                "fair_value_bull": 150,
            }
        )


def test_dossier_reads_early_nested_valuation_without_migrating_old_versions():
    payload = {"valuation": {"market_price": 90, "bear": 70, "base": 110, "bull": 140}}

    assert _payload_number(payload, "reference_price") == 90.0
    assert _payload_number(payload, "fair_value_bear") == 70.0
    assert _payload_number(payload, "fair_value_base") == 110.0
    assert _payload_number(payload, "fair_value_bull") == 140.0


def test_security_dossier_is_evidence_only_and_has_no_duplicate_vote_controls():
    surface_source = getsource(render_security_dossiers)
    form_source = getsource(_render_security_dossier_fields)

    assert "Evidence only" in surface_source
    assert "Voting happens only in Investment Committee" in surface_source
    assert 'st.form_submit_button("Submit vote' not in surface_source
    assert 'st.radio("Vote' not in surface_source
    assert 'st.selectbox("Approval' not in surface_source
    assert all(
        section in form_source
        for section in (
            "Identity & ownership",
            "Investment case",
            "Valuation & challenge",
            "Monitoring & exit",
        )
    )


def test_committee_member_ids_are_stable_when_roster_order_changes():
    first, first_ids, first_captains = _committee_roster(TEAM)
    second, second_ids, second_captains = _committee_roster(list(reversed(TEAM)))

    assert first_ids == second_ids
    assert set(first_captains) == set(second_captains)
    assert {item["vote_scope"] for item in first} == {"investment", "advisory"}
    assert {item["role"] for item in second if item["vote_scope"] == "advisory"} == {
        "clarity_reviewer",
        "client_fit_reviewer",
    }
    assert {
        item["name"]: item["role"] for item in first if item["vote_scope"] == "advisory"
    } == {
        item["name"]: item["role"] for item in second if item["vote_scope"] == "advisory"
    }


def _tracker_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE competition_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT, security_type TEXT, quantity REAL, entry_price REAL,
            entry_date TEXT, opened_by TEXT, opened_at TEXT, last_price REAL,
            notes TEXT, status TEXT, currency TEXT, lifecycle_id INTEGER,
            competition_eligibility_status TEXT, eligibility_source TEXT,
            eligibility_checked_at TEXT
        )
        """
    )
    return connection


def _lifecycle(*, quantity: float = 4.0, currency: str | None = "USD") -> dict:
    execution = {
        "wins_transaction_id": "WINS-9",
        "side": "buy",
        "quantity": quantity,
        "average_price": 210.5,
        "executed_at": "2026-08-16T10:00:00+00:00",
    }
    if currency is not None:
        execution["currency"] = currency
    return {
        "id": 42,
        "ticker": "AAPL",
        "dossier_id": 3,
        "dossier_version": 2,
        "universe_snapshot_id": 7,
        "proposal": {"security_type": "Stock"},
        "wins_execution": {"execution": execution},
    }


def _position(*, quantity: float = 4.0, currency: str = "USD") -> dict:
    return {
        "ticker": "AAPL",
        "quantity": quantity,
        "current_price": 210.5,
        "market_value": quantity * 210.5,
        "total_cost": quantity * 210.5,
        "asset_type": "Stock",
        "currency": currency,
    }


def _patch_dossier(monkeypatch) -> None:
    monkeypatch.setattr(
        dossier_store,
        "get_dossier_version",
        lambda *args, **kwargs: {"payload": {"asset_type": "Stock"}},
    )


def test_wins_execution_immediately_stages_idempotent_pending_projection(monkeypatch):
    connection = _tracker_connection()
    lifecycle = _lifecycle()
    _patch_dossier(monkeypatch)
    recorded = {}

    def fake_execution(connection, lifecycle_id, execution, *, recorded_by, commit):
        recorded.update(
            lifecycle_id=lifecycle_id,
            execution=dict(execution),
            recorded_by=recorded_by,
            commit=commit,
        )
        return lifecycle

    monkeypatch.setattr(lifecycle_store, "record_wins_execution", fake_execution)

    result = _record_wins_execution_and_stage_tracker_position(
        connection,
        42,
        lifecycle["wins_execution"]["execution"],
        actor="Jakub",
    )
    staged_again = _stage_pending_tracker_position(connection, lifecycle, actor="Jakub")

    rows = connection.execute("SELECT * FROM competition_positions").fetchall()
    assert result is lifecycle
    assert recorded["lifecycle_id"] == 42
    assert recorded["recorded_by"] == "Jakub"
    assert recorded["commit"] is False
    assert len(rows) == 1
    assert rows[0]["status"] == "pending_reconciliation"
    assert rows[0]["lifecycle_id"] == 42
    assert staged_again == {
        "id": rows[0]["id"],
        "status": "pending_reconciliation",
        "created": False,
    }


def test_execution_is_rolled_back_when_pending_staging_fails(monkeypatch):
    connection = _tracker_connection()
    connection.execute("CREATE TABLE execution_marker (lifecycle_id INTEGER)")
    connection.commit()
    lifecycle = _lifecycle()

    def fake_execution(connection, lifecycle_id, execution, *, recorded_by, commit):
        assert commit is False
        connection.execute(
            "INSERT INTO execution_marker (lifecycle_id) VALUES (?)",
            (lifecycle_id,),
        )
        return lifecycle

    def missing_dossier(*args, **kwargs):
        raise ValueError("missing dossier")

    monkeypatch.setattr(lifecycle_store, "record_wins_execution", fake_execution)
    monkeypatch.setattr(dossier_store, "get_dossier_version", missing_dossier)

    with pytest.raises(ValueError, match="missing dossier"):
        _record_wins_execution_and_stage_tracker_position(
            connection,
            42,
            lifecycle["wins_execution"]["execution"],
            actor="Jakub",
        )

    assert connection.execute("SELECT COUNT(*) FROM execution_marker").fetchone()[0] == 0


def test_ic_execution_side_is_locked_and_sell_fails_before_lifecycle_write(monkeypatch):
    connection = _tracker_connection()
    source = getsource(render_investment_committee)
    called = False

    def fake_execution(*args, **kwargs):
        nonlocal called
        called = True
        return _lifecycle()

    monkeypatch.setattr(lifecycle_store, "record_wins_execution", fake_execution)
    execution = {**_lifecycle()["wins_execution"]["execution"], "side": "sell"}

    with pytest.raises(ValueError, match="only buy"):
        _record_wins_execution_and_stage_tracker_position(
            connection,
            42,
            execution,
            actor="Jakub",
        )

    assert called is False
    assert 'st.selectbox("Side"' not in source
    assert 'side = "buy"' in source


def test_pending_projection_is_included_in_canonical_tracker_snapshot():
    rows = _tracker_snapshot_rows(
        [
            {**_position(), "status": "pending_reconciliation", "entry_price": 210.5},
            {
                "ticker": "MSFT",
                "quantity": 2,
                "entry_price": 100,
                "last_price": 110,
                "security_type": "Stock",
                "currency": "USD",
                "status": "open",
            },
            {**_position(), "ticker": "CLOSED", "status": "closed", "entry_price": 210.5},
        ]
    )

    assert {item["ticker"] for item in rows} == {"AAPL", "MSFT"}
    pending = next(item for item in rows if item["ticker"] == "AAPL")
    assert pending["quantity"] == 4
    assert pending["currency"] == "USD"


def test_tracker_cash_includes_execution_cost_and_realised_pnl():
    positions = [
        {
            "ticker": "OPEN",
            "security_type": "Stock",
            "quantity": 10,
            "entry_price": 100,
            "status": "open",
            "currency": "USD",
        },
        {
            "ticker": "PENDING",
            "security_type": "Stock",
            "quantity": 5,
            "entry_price": 20,
            "status": "pending_reconciliation",
            "currency": "USD",
        },
        {
            "ticker": "CLOSED",
            "security_type": "Stock",
            "quantity": 2,
            "entry_price": 50,
            "exit_price": 60,
            "status": "closed",
            "currency": "USD",
        },
    ]

    # 500,000 - 1,000 open cost - 100 pending cost + 20 realised P/L.
    assert _tracker_cash_value(positions) == pytest.approx(498_920.0)


def test_activation_adapter_delegates_without_caller_supplied_reconciliation(monkeypatch):
    connection = _tracker_connection()
    lifecycle = _lifecycle()
    captured = {}

    def fake_record(connection, lifecycle_id, reconciliation=None, *, recorded_by, now=None):
        captured.update(
            lifecycle_id=lifecycle_id,
            reconciliation=reconciliation,
            recorded_by=recorded_by,
            now=now,
        )
        return {"state": "active"}

    monkeypatch.setattr(lifecycle_store, "record_wins_reconciliation", fake_record)

    result = _activate_reconciled_tracker_position(
        connection,
        lifecycle,
        actor="Jakub",
    )

    assert result == {"state": "active"}
    assert captured == {
        "lifecycle_id": 42,
        "reconciliation": None,
        "recorded_by": "Jakub",
        "now": None,
    }
