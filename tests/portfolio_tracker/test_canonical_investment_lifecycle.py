from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import sqlite3

import pytest

from src.portfolio_tracker.authoritative_universe_store import (
    check_security_eligibility,
    get_active_authoritative_universe,
    get_authoritative_universe_snapshot,
    list_authoritative_universe_snapshots,
    publish_authoritative_universe,
    require_security_eligible,
    verify_authoritative_universe_snapshot,
)
from src.portfolio_tracker.investment_lifecycle_store import (
    append_position_review,
    close_vote_round,
    create_investment_proposal,
    get_investment_lifecycle,
    get_vote_round,
    has_member_submitted_vote,
    list_lifecycle_audit_events,
    lock_proposal_dossier,
    open_pre_vote,
    record_committee_discussion,
    record_position_exit,
    record_position_sizing,
    record_rule_check,
    record_wins_execution,
    record_wins_reconciliation,
    submit_committee_vote,
    submit_final_approval,
    update_committee_member_status,
    verify_lifecycle_audit_chain,
)
from src.portfolio_tracker.operating_system_store import save_record
from src.portfolio_tracker.portfolio_pipeline import create_portfolio_snapshot
from src.portfolio_tracker.reconciliation_ledger import (
    append_reconciliation,
    latest_reconciliation,
    new_reconciliation_ledger,
    sign_off_reconciliation,
)
from src.portfolio_tracker.security_dossier_store import (
    append_dossier_version,
    append_kpi_observation,
    create_security_dossier,
    freeze_dossier,
    get_dossier_version,
    get_kpi_monitor,
    list_kpi_definitions,
    upsert_kpi_definition,
    verify_frozen_dossier,
)


NOW = datetime(2026, 8, 15, 9, 0, tzinfo=timezone.utc)


def _connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    return connection


class _SyncingTupleConnection:
    def __init__(self) -> None:
        self.connection = sqlite3.connect(":memory:")
        self.commit_calls = 0
        self.sync_calls = 0

    def execute(self, *args, **kwargs):
        return self.connection.execute(*args, **kwargs)

    def commit(self) -> None:
        self.commit_calls += 1
        self.connection.commit()

    def sync(self) -> None:
        self.sync_calls += 1


def _publish_universe(connection: sqlite3.Connection, *, now: datetime = NOW):
    return publish_authoritative_universe(
        connection,
        [
            {
                "ticker": "AAPL",
                "eligibility": "eligible",
                "provenance_status": "official",
                "security_type": "common_stock",
                "payload": {"exchange": "NASDAQ"},
            },
            {
                "ticker": "XYZ",
                "eligibility": "unknown",
                "provenance_status": "not_checked",
            },
        ],
        source_name="Wharton WInS eligible-security export",
        source_url="https://example.test/official-universe.csv",
        provenance_status="official",
        as_of_date="2026-08-15",
        payload={"source_sha256": "source-hash"},
        published_by="Anna",
        now=now,
    )


def _create_frozen_dossier(connection: sqlite3.Connection):
    dossier = create_security_dossier(
        connection,
        "aapl",
        {
            "thesis": "Services mix and installed-base monetisation support durable FCF growth.",
            "catalysts": ["Services acceleration", "Capital return"],
            "invalidation_condition": "Two consecutive quarters below the services KPI breach.",
            "portfolio_role": "Quality compounder in the long-horizon goal bucket.",
            "sell_discipline": "Exit on invalidation or valuation/risk-budget breach.",
        },
        candidate={"screen_id": "quality-2026-08", "security_type": "equity"},
        created_by="Anna",
        now=NOW,
    )
    kpi = upsert_kpi_definition(
        connection,
        dossier["id"],
        "services_growth",
        name="Services revenue growth",
        baseline=0.12,
        expected_min=0.10,
        expected_max=0.18,
        breach_below=0.07,
        unit="fraction_yoy",
        source="Apple quarterly filing",
        frequency="quarterly",
        owner="Research owner",
        payload={"evidence_source_id": 17},
        updated_by="Anna",
        now=NOW,
    )
    frozen = freeze_dossier(connection, dossier["id"], frozen_by="Anna", now=NOW)
    assert frozen["kpi_snapshot"][0]["definition_version_id"] == kpi["definition_version_id"]
    return dossier, frozen


def _committee():
    return [
        {"member_id": "anna", "name": "Anna", "vote_scope": "investment"},
        {"member_id": "jakub", "name": "Jakub", "vote_scope": "investment"},
        {"member_id": "challenger", "name": "Challenger", "vote_scope": "investment"},
        {
            "member_id": "lukas",
            "name": "Lukáš",
            "role": "clarity_reviewer",
            "vote_scope": "advisory",
        },
        {
            "member_id": "recused",
            "name": "Recused member",
            "vote_scope": "investment",
            "conflicted": True,
            "conflict_reason": "Family account owns the security.",
        },
    ]


def _create_proposal(connection: sqlite3.Connection):
    universe = _publish_universe(connection)
    dossier, frozen = _create_frozen_dossier(connection)
    lifecycle = create_investment_proposal(
        connection,
        security_ticker="AAPL",
        dossier_id=dossier["id"],
        dossier_version=frozen["version"],
        universe_snapshot_id=universe["id"],
        proposal={"action": "buy", "rationale": "Client-aligned quality exposure."},
        committee_members=_committee(),
        owner_id="anna",
        challenger_id="challenger",
        required_approvers=["anna", "jakub"],
        quorum=3,
        created_by="anna",
        now=NOW,
    )
    return universe, dossier, frozen, lifecycle


def _submit_round(connection: sqlite3.Connection, lifecycle_id: int, vote_round: str):
    ballots = [
        ("anna", "buy", 8.0, 5, {"client_fit": 5}),
        ("jakub", "buy", 7.0, 4, {"client_fit": 4}),
        ("challenger", "watch" if vote_round == "pre" else "buy", None if vote_round == "pre" else 5.0, 3, {}),
        ("lukas", "watch", None, 4, {"clarity": 4, "client_fit": 5}),
    ]
    last = None
    for member, decision, weight, confidence, dimensions in ballots:
        last = submit_committee_vote(
            connection,
            lifecycle_id,
            vote_round,
            member,
            decision=decision,
            proposed_weight_pct=weight,
            confidence=confidence,
            rationale=f"Independent {vote_round} rationale from {member}.",
            strongest_objection="Valuation could compress despite sound fundamentals.",
            dimensions=dimensions,
            now=NOW + timedelta(minutes=len(dimensions) + confidence),
        )
    return last


def _advance_to_rule_check(connection: sqlite3.Connection):
    _, _, _, lifecycle = _create_proposal(connection)
    lifecycle_id = lifecycle["id"]
    lock_proposal_dossier(connection, lifecycle_id, locked_by="anna", now=NOW)
    open_pre_vote(connection, lifecycle_id, opened_by="anna", now=NOW)
    _submit_round(connection, lifecycle_id, "pre")
    close_vote_round(connection, lifecycle_id, "pre", closed_by="anna", now=NOW)
    record_committee_discussion(
        connection,
        lifecycle_id,
        bull_case="The owner presented client fit, evidence and upside conditions.",
        bear_case="The challenger presented valuation and concentration failure modes.",
        q_and_a=[
            {
                "question": "Why this rather than the benchmark?",
                "answer": "The dossier links differentiated FCF durability to the client goal.",
                "primary_responder": "anna",
                "evidence_source_ids": [17],
            }
        ],
        recorded_by="secretary",
        now=NOW,
    )
    _submit_round(connection, lifecycle_id, "post")
    result = close_vote_round(connection, lifecycle_id, "post", closed_by="anna", now=NOW)
    assert result["state"] == "rule_check"
    return lifecycle_id


def _init_competition_positions(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE competition_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            security_type TEXT NOT NULL,
            quantity REAL NOT NULL,
            entry_price REAL NOT NULL,
            entry_date TEXT NOT NULL,
            last_price REAL,
            status TEXT NOT NULL,
            currency TEXT NOT NULL,
            lifecycle_id INTEGER
        )
        """
    )


def _insert_tracker_position(
    connection: sqlite3.Connection,
    *,
    lifecycle_id: int | None,
    quantity: float,
    status: str,
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO competition_positions (
            ticker, security_type, quantity, entry_price, entry_date,
            last_price, status, currency, lifecycle_id
        ) VALUES ('AAPL', 'Stock', ?, 225.5, '2026-08-15', 225.5, ?, 'USD', ?)
        """,
        (quantity, status, lifecycle_id),
    )
    return int(cursor.lastrowid)


def _tracker_reconciliation_rows(*quantities: float) -> list[dict[str, object]]:
    return [
        {
            "ticker": "AAPL",
            "quantity": quantity,
            "current_price": 225.5,
            "market_value": quantity * 225.5,
            "total_cost": quantity * 225.5,
            "asset_type": "Stock",
            "currency": "USD",
        }
        for quantity in quantities
    ]


def _persist_clean_pipeline(
    connection: sqlite3.Connection,
    tracker_rows: list[dict[str, object]],
    *,
    wins_quantity: float,
) -> tuple[dict[str, object], dict[str, object]]:
    tracker_cash = 500_000.0 - sum(
        float(row.get("total_cost") or 0.0) for row in tracker_rows
    )
    wins = create_portfolio_snapshot(
        [
            {
                "ticker": "AAPL",
                "quantity": wins_quantity,
                "entry_price": 225.5,
                "last_price": 225.5,
                "security_type": "Stock",
                "currency": "USD",
            }
        ],
        provider="WInS",
        observed_at=NOW + timedelta(minutes=5),
        received_at=NOW + timedelta(minutes=6),
        source_reference="WInS account export",
        expected_tickers=("AAPL",),
        cash_value=tracker_cash,
    )
    tracker_snapshot = create_portfolio_snapshot(
        tracker_rows,
        provider="Portfolio Tracker",
        observed_at=NOW + timedelta(minutes=5),
        received_at=NOW + timedelta(minutes=6),
        method="cache",
        source_reference="competition_positions",
        expected_tickers=("AAPL",),
        cash_value=tracker_cash,
    )
    ledger = append_reconciliation(
        new_reconciliation_ledger(),
        wins,
        tracker_snapshot,
        owner="operations",
        performed_at=NOW + timedelta(minutes=7),
    )
    reconciliation_id = str(latest_reconciliation(ledger)["reconciliation_id"])
    ledger = sign_off_reconciliation(
        ledger,
        reconciliation_id,
        decision="approved",
        signed_off_by="jakub",
        signed_off_at=NOW + timedelta(minutes=8),
    )
    save_record(
        connection,
        "portfolio_pipeline",
        "competition",
        {"snapshots": [tracker_snapshot, wins], "ledger": ledger},
        actor="operations",
        status="active",
        now=NOW + timedelta(minutes=9),
    )
    return wins, ledger


def _approved_lifecycle_ready_for_execution(connection: sqlite3.Connection) -> int:
    lifecycle_id = _advance_to_rule_check(connection)
    record_rule_check(
        connection,
        lifecycle_id,
        rulebook_version=1,
        mandate_version=1,
        checks=[{"rule_id": "max_position", "passed": True, "limit_pct": 10}],
        evaluated_by="risk-owner",
        now=NOW,
    )
    submit_final_approval(
        connection,
        lifecycle_id,
        "anna",
        decision="approve",
        comment="Approved.",
        now=NOW,
    )
    submit_final_approval(
        connection,
        lifecycle_id,
        "jakub",
        decision="approve",
        comment="Independently approved.",
        now=NOW,
    )
    record_position_sizing(
        connection,
        lifecycle_id,
        {
            "target_weight_pct": 7,
            "rationale": "Within the approved rulebook limit.",
            "starter_position": False,
        },
        sized_by="portfolio-owner",
        now=NOW,
    )
    return lifecycle_id


def _record_standard_execution(
    connection: sqlite3.Connection,
    lifecycle_id: int,
    *,
    commit: bool = True,
) -> dict[str, object]:
    return record_wins_execution(
        connection,
        lifecycle_id,
        {
            "wins_transaction_id": f"WINS-{lifecycle_id}",
            "side": "buy",
            "quantity": 10,
            "average_price": 225.5,
            "executed_at": "2026-08-15T14:30:00Z",
            "currency": "USD",
        },
        recorded_by="team-leader",
        now=NOW,
        commit=commit,
    )


def test_authoritative_universe_snapshots_are_immutable_versioned_and_fail_closed():
    connection = _connection()
    first = _publish_universe(connection)

    assert first["is_active"] is True
    assert first["version"] == 1
    assert verify_authoritative_universe_snapshot(first)
    assert require_security_eligible(connection, "aapl")["can_trade"] is True
    unknown = check_security_eligibility(connection, "XYZ")
    assert unknown["can_trade"] is False
    assert unknown["provenance_status"] == "not_checked"

    second = publish_authoritative_universe(
        connection,
        [{"ticker": "MSFT", "eligibility": "eligible"}],
        source_name="Analyst working assumption",
        provenance_status="analyst_assumption",
        as_of_date="2026-08-16",
        published_by="Analyst",
        expected_active_snapshot_id=first["id"],
        now=NOW + timedelta(days=1),
    )

    assert second["version"] == 2
    assert get_active_authoritative_universe(connection)["id"] == second["id"]
    historical = get_authoritative_universe_snapshot(connection, first["id"])
    assert historical["is_active"] is False
    assert historical["entries"][0]["ticker"] == "AAPL"
    assert [item["version"] for item in list_authoritative_universe_snapshots(connection)] == [2, 1]
    assert check_security_eligibility(connection, "AAPL", snapshot_id=first["id"])["can_trade"] is False
    with pytest.raises(ValueError, match="not eligible"):
        require_security_eligible(connection, "MSFT")
    with pytest.raises(ValueError, match="changed after it was loaded"):
        publish_authoritative_universe(
            connection,
            [{"ticker": "TSM", "eligibility": "eligible"}],
            source_name="Stale editor",
            provenance_status="official",
            as_of_date="2026-08-16",
            published_by="Anna",
            expected_active_snapshot_id=first["id"],
        )


def test_authoritative_universe_hash_tampering_fails_closed():
    connection = _connection()
    snapshot = _publish_universe(connection)
    connection.execute(
        """
        UPDATE authoritative_universe_entries SET eligibility = 'ineligible'
        WHERE snapshot_id = ? AND ticker = 'AAPL'
        """,
        (snapshot["id"],),
    )
    connection.commit()

    decision = check_security_eligibility(connection, "AAPL")
    assert decision["content_hash_valid"] is False
    assert decision["can_trade"] is False
    assert "content hash" in " ".join(decision["reasons"])


def test_frozen_dossier_keeps_exact_thesis_and_kpi_definition_snapshot():
    connection = _connection()
    dossier, first_frozen = _create_frozen_dossier(connection)
    assert verify_frozen_dossier(first_frozen)

    revised_kpi = upsert_kpi_definition(
        connection,
        dossier["id"],
        "services_growth",
        name="Services revenue growth",
        baseline=0.12,
        expected_min=0.11,
        expected_max=0.18,
        breach_below=0.08,
        unit="fraction_yoy",
        source="Apple quarterly filing",
        frequency="quarterly",
        owner="New KPI owner",
        updated_by="Jakub",
        expected_current_revision=1,
        now=NOW + timedelta(hours=1),
    )
    assert revised_kpi["revision"] == 2
    preserved = get_dossier_version(connection, dossier["id"], 1)
    assert preserved["kpi_snapshot"][0]["revision"] == 1
    assert verify_frozen_dossier(preserved)

    draft = append_dossier_version(
        connection,
        dossier["id"],
        {**first_frozen["payload"], "thesis": "Revised thesis backed by new evidence."},
        created_by="Jakub",
        expected_current_version=1,
        now=NOW + timedelta(hours=2),
    )
    second_frozen = freeze_dossier(
        connection,
        dossier["id"],
        version=draft["version"],
        frozen_by="Jakub",
        now=NOW + timedelta(hours=3),
    )
    assert second_frozen["kpi_snapshot"][0]["revision"] == 2
    assert second_frozen["content_hash"] != first_frozen["content_hash"]


def test_kpi_monitor_classifies_watch_and_breach_and_preserves_observation_history():
    connection = _connection()
    dossier, _ = _create_frozen_dossier(connection)
    watch = append_kpi_observation(
        connection,
        dossier["id"],
        "services_growth",
        0.09,
        observed_at="2026-05-01",
        source_ref="Apple Q2 filing, p. 12",
        recorded_by="Anna",
        now=NOW,
    )
    breach = append_kpi_observation(
        connection,
        dossier["id"],
        "services_growth",
        0.07,
        observed_at="2026-08-01",
        source_ref="Apple Q3 filing, p. 10",
        payload={"evidence_source_id": 22},
        recorded_by="Jakub",
        now=NOW,
    )

    assert watch["health_status"] == "watch"
    assert breach["health_status"] == "breach"
    monitor = get_kpi_monitor(connection, dossier["id"], as_of=NOW)
    assert monitor["summary"]["breach"] == 1
    assert monitor["items"][0]["latest_observation"]["id"] == breach["id"]
    assert monitor["items"][0]["last_updated_at"] == "2026-08-01"


def test_blind_two_round_votes_hide_partial_ballots_exclude_conflict_and_capture_dissent():
    connection = _connection()
    _, _, _, lifecycle = _create_proposal(connection)
    lifecycle_id = lifecycle["id"]
    assert lifecycle["state"] == "proposal"
    assert lifecycle["committee_status"]["conflicts"] == [
        {"member_id": "recused", "reason": "Family account owns the security."}
    ]

    lock_proposal_dossier(connection, lifecycle_id, locked_by="anna", now=NOW)
    opened = open_pre_vote(connection, lifecycle_id, opened_by="anna", now=NOW)
    assert opened["eligible_count"] == 4
    assert has_member_submitted_vote(connection, lifecycle_id, "pre", "anna") is False
    partial = submit_committee_vote(
        connection,
        lifecycle_id,
        "pre",
        "anna",
        decision="buy",
        proposed_weight_pct=8,
        confidence=5,
        rationale="Independent rationale.",
        strongest_objection="Valuation.",
        now=NOW,
    )
    assert partial["revealed"] is False
    assert has_member_submitted_vote(connection, lifecycle_id, "pre", "anna") is True
    assert partial["ballots"] == []
    assert partial["tally"] is None
    events = list_lifecycle_audit_events(connection, lifecycle_id)
    vote_event = events[-1]
    assert set(vote_event["payload"]) == {"member_id", "round", "ballot_hash"}
    with pytest.raises(ValueError, match="already submitted"):
        submit_committee_vote(
            connection,
            lifecycle_id,
            "pre",
            "anna",
            decision="watch",
            proposed_weight_pct=None,
            confidence=3,
            rationale="Changed after seeing nothing.",
            strongest_objection="Valuation.",
        )
    with pytest.raises(ValueError, match="absent, conflicted"):
        submit_committee_vote(
            connection,
            lifecycle_id,
            "pre",
            "recused",
            decision="reject",
            proposed_weight_pct=None,
            confidence=5,
            rationale="Conflict should prevent this.",
            strongest_objection="Conflict.",
        )

    for member, decision, weight in [
        ("jakub", "buy", 7),
        ("challenger", "watch", None),
        ("lukas", "watch", None),
    ]:
        revealed = submit_committee_vote(
            connection,
            lifecycle_id,
            "pre",
            member,
            decision=decision,
            proposed_weight_pct=weight,
            confidence=4,
            rationale=f"Independent rationale from {member}.",
            strongest_objection="Valuation.",
            dimensions={"clarity": 4} if member == "lukas" else {},
        )
    assert revealed["revealed"] is True
    assert len(revealed["ballots"]) == 4
    assert sum(revealed["tally"].values()) == 3  # advisory vote does not drive decision
    assert revealed["outcome"] == "buy"
    assert revealed["dissent"] == [
        {
            "member_id": "challenger",
            "decision": "watch",
            "strongest_objection": "Valuation.",
        }
    ]


def test_full_lifecycle_links_approval_sizing_execution_position_review_and_exit():
    connection = _connection()
    lifecycle_id = _advance_to_rule_check(connection)
    checked = record_rule_check(
        connection,
        lifecycle_id,
        rulebook_version=3,
        mandate_version=2,
        checks=[
            {"rule_id": "max_position", "passed": True, "limit_pct": 10, "actual_pct": 7},
            {"rule_id": "client_goal_fit", "passed": True, "goal_bucket": "growth"},
        ],
        evaluated_by="risk-owner",
        now=NOW,
    )
    assert checked["state"] == "final_approval"
    first_signoff = submit_final_approval(
        connection,
        lifecycle_id,
        "anna",
        decision="approve",
        comment="Approve subject to recorded size.",
        now=NOW,
    )
    assert first_signoff["state"] == "final_approval"
    with pytest.raises(ValueError, match="designated final approver"):
        submit_final_approval(
            connection,
            lifecycle_id,
            "challenger",
            decision="approve",
            comment="Should not be accepted.",
        )
    second_signoff = submit_final_approval(
        connection,
        lifecycle_id,
        "jakub",
        decision="approve",
        comment="Independent final approval.",
        now=NOW,
    )
    assert second_signoff["state"] == "sizing"
    assert second_signoff["final_approval_complete"] is True
    with pytest.raises(ValueError, match="exceeds the checked maximum"):
        record_position_sizing(
            connection,
            lifecycle_id,
            {
                "target_weight_pct": 12,
                "rationale": "This must not bypass the rule check.",
                "starter_position": False,
            },
            sized_by="portfolio-owner",
        )

    sized = record_position_sizing(
        connection,
        lifecycle_id,
        {
            "target_weight_pct": 7,
            "rationale": "Inside member proposals and sector risk budget.",
            "starter_position": False,
        },
        sized_by="portfolio-owner",
        now=NOW,
    )
    assert sized["state"] == "wins_execution"
    with pytest.raises(ValueError, match="Execution currency"):
        record_wins_execution(
            connection,
            lifecycle_id,
            {
                "wins_transaction_id": "WINS-INCOMPLETE",
                "side": "buy",
                "quantity": 10,
                "average_price": 225.50,
                "executed_at": "2026-08-15T14:30:00Z",
            },
            recorded_by="team-leader",
        )
    assert get_investment_lifecycle(connection, lifecycle_id)["state"] == "wins_execution"
    executed = record_wins_execution(
        connection,
        lifecycle_id,
        {
            "wins_transaction_id": "WINS-ORDER-1001",
            "side": "buy",
            "quantity": 10,
            "average_price": 225.50,
            "executed_at": "2026-08-15T14:30:00Z",
            "currency": "USD",
        },
        recorded_by="team-leader",
        now=NOW,
    )
    assert executed["state"] == "reconciliation"
    exception = record_wins_reconciliation(
        connection,
        lifecycle_id,
        {
            "status": "open_exceptions",
            "wins_snapshot_id": "WINS-SNAPSHOT-1",
            "exceptions": [{"field": "quantity", "wins": 10, "tracker": 9}],
        },
        recorded_by="operations",
        now=NOW,
    )
    assert exception["state"] == "reconciliation"
    assert exception["latest_reconciliation"]["status"] == "open_exceptions"

    _init_competition_positions(connection)
    tracker_position_id = _insert_tracker_position(
        connection,
        lifecycle_id=lifecycle_id,
        quantity=10,
        status="pending_reconciliation",
    )
    with pytest.raises(ValueError, match="persisted canonical portfolio pipeline"):
        record_wins_reconciliation(
            connection,
            lifecycle_id,
            {
                "status": "clean",
                "wins_snapshot_id": "FORGED-SNAPSHOT",
                "canonical_reconciliation_id": "FORGED-RECONCILIATION",
                "canonical_source": "portfolio_pipeline/competition",
                "position_id": str(tracker_position_id),
                "exceptions": [],
            },
            recorded_by="operations",
            now=NOW + timedelta(minutes=10),
        )
    assert get_investment_lifecycle(connection, lifecycle_id)["state"] == "reconciliation"
    assert connection.execute(
        "SELECT status FROM competition_positions WHERE id = ?", (tracker_position_id,)
    ).fetchone()["status"] == "pending_reconciliation"

    wins, ledger = _persist_clean_pipeline(
        connection,
        _tracker_reconciliation_rows(10),
        wins_quantity=10,
    )
    with pytest.raises(ValueError, match="bindings do not match"):
        record_wins_reconciliation(
            connection,
            lifecycle_id,
            {
                "status": "clean",
                "wins_snapshot_id": wins["snapshot_id"],
                "canonical_reconciliation_id": (
                    f"{latest_reconciliation(ledger)['reconciliation_id']}-forged"
                ),
                "canonical_source": "portfolio_pipeline/competition",
                "position_id": str(tracker_position_id),
                "exceptions": [],
            },
            recorded_by="operations",
            now=NOW + timedelta(minutes=12),
        )
    active = record_wins_reconciliation(
        connection,
        lifecycle_id,
        recorded_by="operations",
        now=NOW + timedelta(minutes=15),
    )
    assert active["state"] == "active"
    assert active["current_position_id"] == str(tracker_position_id)
    assert len(active["reconciliation_history"]) == 2
    assert active["latest_reconciliation"]["reconciliation"]["canonical_source"] == (
        "portfolio_pipeline/competition"
    )
    assert connection.execute(
        "SELECT status FROM competition_positions WHERE id = ?", (tracker_position_id,)
    ).fetchone()["status"] == "open"

    reviewed = append_position_review(
        connection,
        lifecycle_id,
        {"kpi_status": "on_track", "next_action": "retain"},
        outcome="confirmed",
        reviewed_by="research-owner",
        now=NOW + timedelta(days=30),
    )
    assert reviewed["position_reviews"][0]["outcome"] == "confirmed"
    exited = record_position_exit(
        connection,
        lifecycle_id,
        {
            "wins_transaction_id": "WINS-EXIT-1001",
            "executed_at": "2027-02-01T15:00:00Z",
            "reason": "Dossier invalidation condition was reached.",
        },
        recorded_by="team-leader",
        now=NOW + timedelta(days=170),
    )
    assert exited["state"] == "exited"
    assert exited["exit"]["wins_transaction_id"] == "WINS-EXIT-1001"
    assert exited["audit"]["valid"] is True
    assert exited["audit"]["checked_events"] > 20
    json.dumps(exited, allow_nan=False)


def test_clean_activation_supports_existing_same_ticker_lots_without_delta_confusion():
    connection = _connection()
    lifecycle_id = _approved_lifecycle_ready_for_execution(connection)
    _record_standard_execution(connection, lifecycle_id)
    _init_competition_positions(connection)
    existing_position_id = _insert_tracker_position(
        connection,
        lifecycle_id=None,
        quantity=5,
        status="open",
    )
    pending_position_id = _insert_tracker_position(
        connection,
        lifecycle_id=lifecycle_id,
        quantity=10,
        status="pending_reconciliation",
    )
    _persist_clean_pipeline(
        connection,
        _tracker_reconciliation_rows(5, 10),
        wins_quantity=15,
    )

    active = record_wins_reconciliation(
        connection,
        lifecycle_id,
        recorded_by="operations",
        now=NOW + timedelta(minutes=15),
    )

    assert active["state"] == "active"
    assert active["current_position_id"] == str(pending_position_id)
    statuses = dict(
        connection.execute("SELECT id, status FROM competition_positions ORDER BY id").fetchall()
    )
    assert statuses == {existing_position_id: "open", pending_position_id: "open"}


def test_uncommitted_execution_can_be_rolled_back_before_tracker_staging():
    connection = _connection()
    lifecycle_id = _approved_lifecycle_ready_for_execution(connection)

    executed = _record_standard_execution(connection, lifecycle_id, commit=False)
    assert executed["state"] == "reconciliation"
    connection.rollback()

    restored = get_investment_lifecycle(connection, lifecycle_id)
    assert restored["state"] == "wins_execution"
    assert restored["wins_execution"] is None


def test_rule_check_detects_superseded_universe_and_requires_named_override():
    connection = _connection()
    lifecycle_id = _advance_to_rule_check(connection)
    active = get_active_authoritative_universe(connection)
    publish_authoritative_universe(
        connection,
        [{"ticker": "AAPL", "eligibility": "eligible", "provenance_status": "official"}],
        source_name="New official universe",
        provenance_status="official",
        as_of_date="2026-08-16",
        published_by="Anna",
        expected_active_snapshot_id=active["id"],
        now=NOW + timedelta(days=1),
    )
    with pytest.raises(ValueError, match="missing failed rules: authoritative_universe_current"):
        record_rule_check(
            connection,
            lifecycle_id,
            rulebook_version=3,
            mandate_version=2,
            checks=[{"rule_id": "max_position", "passed": True}],
            override={
                "reason": "An override cannot omit an automatic failed rule.",
                "authorized_by": "anna",
                "scope": ["max_position"],
            },
            evaluated_by="risk-owner",
            now=NOW + timedelta(days=1),
        )
    assert get_investment_lifecycle(connection, lifecycle_id)["state"] == "rule_check"
    overridden = record_rule_check(
        connection,
        lifecycle_id,
        rulebook_version=3,
        mandate_version=2,
        checks=[{"rule_id": "max_position", "passed": True}],
        override={
            "reason": "New snapshot preserves AAPL eligibility; proposal retains original evidence.",
            "authorized_by": "anna",
            "scope": ["authoritative_universe_current"],
        },
        evaluated_by="risk-owner",
        now=NOW + timedelta(days=1),
    )
    assert overridden["state"] == "final_approval"
    assert overridden["latest_rule_check"]["passed"] is False
    assert overridden["latest_rule_check"]["effective_pass"] is True
    assert overridden["latest_rule_check"]["failed_rules"][0]["rule_id"] == (
        "authoritative_universe_current"
    )


def test_audit_chain_detects_direct_database_tampering():
    connection = _connection()
    _, _, _, lifecycle = _create_proposal(connection)
    lifecycle_id = lifecycle["id"]
    assert verify_lifecycle_audit_chain(connection, lifecycle_id)["valid"] is True
    row = connection.execute(
        """
        SELECT id, payload_json FROM canonical_investment_audit_events
        WHERE lifecycle_id = ? ORDER BY sequence LIMIT 1
        """,
        (lifecycle_id,),
    ).fetchone()
    payload = json.loads(row["payload_json"])
    payload["ticker"] = "TAMPERED"
    connection.execute(
        "UPDATE canonical_investment_audit_events SET payload_json = ? WHERE id = ?",
        (json.dumps(payload), row["id"]),
    )
    connection.commit()

    verification = verify_lifecycle_audit_chain(connection, lifecycle_id)
    assert verification["valid"] is False
    assert verification["error_sequence"] == 1
    assert "content hash" in verification["reason"]


def test_dossier_and_committee_invariants_block_incomplete_or_conflicted_workflows():
    connection = _connection()
    universe = _publish_universe(connection)
    draft = create_security_dossier(
        connection,
        "MSFT",
        {"thesis": "Incomplete draft"},
        created_by="Anna",
        now=NOW,
    )
    with pytest.raises(ValueError, match="missing"):
        freeze_dossier(connection, draft["id"], frozen_by="Anna")

    _, _, _, lifecycle = _create_proposal(connection)
    with pytest.raises(ValueError, match="state"):
        submit_committee_vote(
            connection,
            lifecycle["id"],
            "pre",
            "anna",
            decision="buy",
            proposed_weight_pct=7,
            confidence=4,
            rationale="Too early.",
            strongest_objection="None yet.",
        )
    update_committee_member_status(
        connection,
        lifecycle["id"],
        "challenger",
        conflicted=True,
        conflict_reason="Newly disclosed conflict.",
        updated_by="secretary",
    )
    with pytest.raises(ValueError, match="Required committee member challenger"):
        lock_proposal_dossier(connection, lifecycle["id"], locked_by="anna")
    assert universe["id"] > 0
    assert list_kpi_definitions(connection, draft["id"]) == []


def test_plain_tuple_rows_and_online_sync_connections_are_supported():
    connection = _SyncingTupleConnection()
    universe = _publish_universe(connection)
    dossier, frozen = _create_frozen_dossier(connection)
    lifecycle = create_investment_proposal(
        connection,
        security_ticker="AAPL",
        dossier_id=dossier["id"],
        dossier_version=frozen["version"],
        universe_snapshot_id=universe["id"],
        proposal={"action": "buy", "rationale": "Tuple-row compatibility."},
        committee_members=_committee(),
        owner_id="anna",
        challenger_id="challenger",
        required_approvers=["anna", "jakub"],
        quorum=3,
        created_by="anna",
        now=NOW,
    )

    assert lifecycle["ticker"] == "AAPL"
    assert lifecycle["audit"]["valid"] is True
    # Universe, dossier, KPI, freeze and proposal each commit/sync once.
    assert connection.commit_calls == 5
    assert connection.sync_calls == 5
