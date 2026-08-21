from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import sqlite3

from src.portfolio_tracker.authoritative_universe_store import (
    publish_authoritative_universe,
)
from src.portfolio_tracker.investment_lifecycle_store import (
    close_vote_round,
    create_investment_proposal,
    get_investment_lifecycle,
    lock_proposal_dossier,
    open_pre_vote,
    record_committee_discussion,
    record_position_sizing,
    record_rule_check,
    record_wins_execution,
    record_wins_reconciliation,
    submit_committee_vote,
    submit_final_approval,
    verify_lifecycle_audit_chain,
)
from src.portfolio_tracker.operating_system_store import save_record
from src.portfolio_tracker.portfolio_pipeline import (
    build_live_portfolio_pipeline,
    create_portfolio_snapshot,
    materialize_consumer_input,
)
from src.portfolio_tracker.qa_rehearsal import (
    add_qa_question,
    complete_mock_round,
    create_mock_round,
    create_qa_rehearsal_workspace,
    killer_question_status,
    record_qa_response,
)
from src.portfolio_tracker.reconciliation_ledger import (
    append_reconciliation,
    latest_reconciliation,
    new_reconciliation_ledger,
    sign_off_reconciliation,
)
from src.portfolio_tracker.security_dossier_store import (
    append_kpi_observation,
    create_security_dossier,
    freeze_dossier,
    get_kpi_monitor,
    upsert_kpi_definition,
)
from src.reporting.evidence_studio import (
    add_decision_case_study,
    add_report_claim,
    build_export_ready_report,
    create_report_workspace,
    finalise_report,
    freeze_report,
    record_report_approval,
    register_report_evidence,
    register_report_figure,
    set_performance_attribution,
    set_report_portfolio_snapshot,
    set_report_section_content,
    validate_report_workspace,
)


START = datetime(2026, 8, 15, 9, 0, tzinfo=timezone.utc)


def _at(minutes: int) -> datetime:
    return START + timedelta(minutes=minutes)


def _submit_blind_round(
    connection: sqlite3.Connection,
    lifecycle_id: int,
    vote_round: str,
    *,
    start_minute: int,
) -> dict[str, object]:
    ballots = (
        ("anna", "buy", 5.0, 5),
        ("jakub", "buy", 5.0, 4),
        (
            "martin",
            "watch" if vote_round == "pre" else "buy",
            None if vote_round == "pre" else 4.5,
            4,
        ),
        ("lukas", "buy", 4.0, 4),
    )
    result: dict[str, object] = {}
    for index, (member, decision, weight, confidence) in enumerate(ballots):
        result = submit_committee_vote(
            connection,
            lifecycle_id,
            vote_round,
            member,
            decision=decision,
            proposed_weight_pct=weight,
            confidence=confidence,
            rationale=f"Independent {vote_round}-vote rationale from {member}.",
            strongest_objection="Valuation compression could weaken the risk/reward.",
            now=_at(start_minute + index),
        )
        if index < len(ballots) - 1:
            assert result["revealed"] is False
            assert result["ballots"] == []
            assert result["tally"] is None
    assert result["revealed"] is True
    assert len(result["ballots"]) == 4
    return result


def test_full_investment_operating_system_flow_uses_one_audit_trail() -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row

    dossier = create_security_dossier(
        connection,
        "AAPL",
        {
            "thesis": "Services growth and installed-base monetisation support durable FCF.",
            "catalysts": ["Services acceleration", "Capital returns"],
            "invalidation_condition": "Services growth breaches the predefined floor.",
            "portfolio_role": "Quality compounder for the client's long-horizon goal.",
            "sell_discipline": "Exit on thesis invalidation or a risk-budget breach.",
        },
        candidate={"screen_id": "quality-2026-08", "security_type": "equity"},
        created_by="Anna",
        now=_at(0),
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
        owner="Anna",
        payload={"evidence_id": "apple-q3-filing"},
        updated_by="Anna",
        now=_at(1),
    )
    observation = append_kpi_observation(
        connection,
        dossier["id"],
        "services_growth",
        0.13,
        observed_at=_at(2),
        source_ref="Apple Q3 filing, Services note",
        payload={"evidence_id": "apple-q3-filing"},
        recorded_by="Anna",
        now=_at(2),
    )
    frozen_dossier = freeze_dossier(
        connection,
        dossier["id"],
        frozen_by="Anna",
        now=_at(3),
    )
    monitor = get_kpi_monitor(connection, dossier["id"], as_of=_at(4))
    assert observation["health_status"] == "on_track"
    assert monitor["summary"]["on_track"] == 1
    assert (
        frozen_dossier["kpi_snapshot"][0]["definition_version_id"] == kpi["definition_version_id"]
    )

    universe = publish_authoritative_universe(
        connection,
        [
            {
                "ticker": "AAPL",
                "eligibility": "eligible",
                "provenance_status": "official",
                "security_type": "common_stock",
                "payload": {"exchange": "NASDAQ"},
            }
        ],
        source_name="WInS official eligible-security export",
        source_url="https://example.test/wins-universe.csv",
        provenance_status="official",
        as_of_date="2026-08-15",
        payload={"source_sha256": "official-universe-hash"},
        published_by="Jakub",
        now=_at(4),
    )
    lifecycle = create_investment_proposal(
        connection,
        security_ticker="AAPL",
        dossier_id=dossier["id"],
        dossier_version=frozen_dossier["version"],
        universe_snapshot_id=universe["id"],
        proposal={
            "action": "buy",
            "rationale": "The frozen dossier supports client-aligned quality exposure.",
            "client_goal_id": "long-term-growth",
        },
        committee_members=[
            {"member_id": "anna", "name": "Anna"},
            {"member_id": "jakub", "name": "Jakub"},
            {"member_id": "martin", "name": "Martin"},
            {"member_id": "lukas", "name": "Lukáš"},
        ],
        owner_id="anna",
        challenger_id="martin",
        required_approvers=["anna", "jakub"],
        quorum=4,
        created_by="Anna",
        now=_at(5),
    )
    lifecycle_id = lifecycle["id"]
    lock_proposal_dossier(
        connection,
        lifecycle_id,
        locked_by="Anna",
        now=_at(6),
    )
    pre_open = open_pre_vote(
        connection,
        lifecycle_id,
        opened_by="Anna",
        now=_at(7),
    )
    assert pre_open["eligible_count"] == 4
    pre_vote = _submit_blind_round(
        connection,
        lifecycle_id,
        "pre",
        start_minute=8,
    )
    assert pre_vote["outcome"] == "buy"
    assert pre_vote["dissent"] == [
        {
            "member_id": "martin",
            "decision": "watch",
            "strongest_objection": "Valuation compression could weaken the risk/reward.",
        }
    ]
    close_vote_round(
        connection,
        lifecycle_id,
        "pre",
        closed_by="Anna",
        now=_at(12),
    )
    record_committee_discussion(
        connection,
        lifecycle_id,
        bull_case="The dossier documents client fit, KPI evidence and upside conditions.",
        bear_case="The challenger documented valuation and concentration failure modes.",
        q_and_a=[
            {
                "question": "What would invalidate the investment?",
                "answer": "The frozen KPI breach threshold triggers a formal review.",
                "evidence_ids": ["apple-q3-filing"],
            }
        ],
        recorded_by="Lukáš",
        now=_at(13),
    )
    post_vote = _submit_blind_round(
        connection,
        lifecycle_id,
        "post",
        start_minute=14,
    )
    assert post_vote["outcome"] == "buy"
    assert post_vote["dissent"] == []
    close_vote_round(
        connection,
        lifecycle_id,
        "post",
        closed_by="Anna",
        now=_at(18),
    )
    checked = record_rule_check(
        connection,
        lifecycle_id,
        rulebook_version=4,
        mandate_version=2,
        checks=[
            {"rule_id": "max_position", "passed": True, "limit_pct": 8.0},
            {"rule_id": "client_goal_fit", "passed": True},
            {"rule_id": "sector_budget", "passed": True},
        ],
        evaluated_by="Jakub",
        now=_at(19),
    )
    assert checked["state"] == "final_approval"
    submit_final_approval(
        connection,
        lifecycle_id,
        "anna",
        decision="approve",
        comment="Approved within the frozen dossier and rulebook limits.",
        now=_at(20),
    )
    approved = submit_final_approval(
        connection,
        lifecycle_id,
        "jakub",
        decision="approve",
        comment="Independent final approval.",
        now=_at(21),
    )
    assert approved["final_approval_complete"] is True
    sized = record_position_sizing(
        connection,
        lifecycle_id,
        {
            "target_weight_pct": 5.0,
            "rationale": "Within the checked cap and all four post-vote indications.",
            "starter_position": False,
        },
        sized_by="Anna",
        now=_at(22),
    )
    assert sized["state"] == "wins_execution"
    executed = record_wins_execution(
        connection,
        lifecycle_id,
        {
            "wins_transaction_id": "WINS-AAPL-20260815-01",
            "side": "buy",
            "quantity": 10,
            "average_price": 100.0,
            "executed_at": _at(23).isoformat(),
            "currency": "USD",
        },
        recorded_by="Anna",
        now=_at(23),
        commit=False,
    )
    assert executed["state"] == "reconciliation"

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
    tracker_cursor = connection.execute(
        """
        INSERT INTO competition_positions (
            ticker, security_type, quantity, entry_price, entry_date,
            last_price, status, currency, lifecycle_id
        ) VALUES ('AAPL', 'Stock', 10, 100, '2026-08-15', 100,
                  'pending_reconciliation', 'USD', ?)
        """,
        (lifecycle_id,),
    )
    tracker_position_id = int(tracker_cursor.lastrowid)
    connection.commit()

    tracked_positions = [
        {
            "ticker": "AAPL",
            "quantity": 10,
            "entry_price": 100.0,
            "last_price": 100.0,
            "security_type": "Stock",
            "currency": "USD",
        }
    ]
    wins_snapshot = create_portfolio_snapshot(
        tracked_positions,
        provider="WInS",
        method="manual_import",
        imported_by="Lukáš",
        observed_at=_at(24),
        received_at=_at(25),
        source_reference="WInS portfolio export after WINS-AAPL-20260815-01",
        cash_value=499_000.0,
        expected_tickers=("AAPL",),
    )
    tracker_snapshot = create_portfolio_snapshot(
        tracked_positions,
        provider="Portfolio Tracker",
        method="cache",
        imported_by="Anna",
        observed_at=_at(24),
        received_at=_at(25),
        source_reference="competition_positions",
        cash_value=499_000.0,
        expected_tickers=("AAPL",),
    )
    ledger = append_reconciliation(
        new_reconciliation_ledger(),
        wins_snapshot,
        tracker_snapshot,
        owner="Lukáš",
        performed_at=_at(26),
    )
    reconciliation = latest_reconciliation(ledger)
    assert reconciliation is not None
    assert reconciliation["base_is_clean"] is True
    ledger = sign_off_reconciliation(
        ledger,
        reconciliation["reconciliation_id"],
        decision="approved",
        signed_off_by="Jakub",
        note="The WInS export matches the canonical tracker projection.",
        signed_off_at=_at(27),
    )
    save_record(
        connection,
        "portfolio_pipeline",
        "competition",
        {
            "snapshots": [tracker_snapshot, wins_snapshot],
            "ledger": ledger,
            "expected_return_assumptions": {
                "status": "active",
                "values": {"AAPL": 0.08},
            },
        },
        actor="Anna",
        status="active",
        now=_at(27),
    )
    active = record_wins_reconciliation(
        connection,
        lifecycle_id,
        recorded_by="Lukáš",
        now=_at(28),
    )
    assert active["state"] == "active"
    assert active["current_position_id"] == str(tracker_position_id)
    assert connection.execute(
        "SELECT status FROM competition_positions WHERE id = ?",
        (tracker_position_id,),
    ).fetchone()[0] == "open"

    pipeline = build_live_portfolio_pipeline(
        [tracker_snapshot, wins_snapshot],
        ledger,
        mandate={"mandate_id": "mandate-2", "status": "active"},
        rulebook={"rulebook_id": "rulebook-4", "status": "active", "max_position": 0.08},
        expected_return_assumptions={
            "assumption_set_id": "expected-returns-3",
            "status": "active",
            "values": {"AAPL": 0.08},
        },
        now=_at(29),
    )
    quant_input = materialize_consumer_input(pipeline, "quant")
    assert pipeline["status"] == "ready"
    assert pipeline["authority"] == "wins_reconciled"
    assert quant_input["allowed"] is True
    assert quant_input["portfolio_snapshot_id"] == wins_snapshot["snapshot_id"]
    assert (
        quant_input["reconciliation_gate"]["latest_reconciliation_id"]
        == reconciliation["reconciliation_id"]
    )
    assert quant_input["portfolio"]["positions"][0]["ticker"] == "AAPL"

    dossier_evidence_id = f"dossier-{dossier['id']}-v{frozen_dossier['version']}"
    lifecycle_evidence_id = f"lifecycle-{lifecycle_id}"
    report = create_report_workspace(
        "final-e2e",
        "final",
        "Investment operating system smoke report",
        created_by="Anna",
        required_approvers=["Anna", "Jakub"],
        page_budget=2,
        section_schema=[
            {
                "id": "decision",
                "title": "Client-linked decision",
                "page_budget": 2,
                "owner": "Anna",
                "reviewer": "Jakub",
            }
        ],
        now=_at(30),
    )
    report = register_report_evidence(
        report,
        dossier_evidence_id,
        title="Frozen AAPL dossier and KPI evidence",
        citation="QuantSim canonical security dossier",
        source_locator=f"dossier://{dossier['id']}/version/{frozen_dossier['version']}",
        source_type="canonical_dossier",
        verified_by="Anna",
        accessed_at=_at(30),
        now=_at(30),
    )
    report = register_report_evidence(
        report,
        lifecycle_evidence_id,
        title="Approved AAPL Investment Committee lifecycle",
        citation="QuantSim append-only lifecycle audit trail",
        source_locator=f"lifecycle://{lifecycle_id}",
        source_type="decision_audit",
        verified_by="Jakub",
        accessed_at=_at(30),
        now=_at(30),
    )
    report = add_report_claim(
        report,
        "claim-client-fit",
        section_id="decision",
        statement="AAPL entered the portfolio only after the frozen client-linked process.",
        evidence_ids=[dossier_evidence_id, lifecycle_evidence_id],
        created_by="Anna",
        now=_at(31),
    )
    report = register_report_figure(
        report,
        "figure-canonical-allocation",
        section_id="decision",
        title="Reconciled canonical allocation",
        caption="The same signed WInS snapshot feeds Quant and the final report.",
        artifact_locator=f"snapshot://{wins_snapshot['snapshot_id']}/allocation.svg",
        evidence_ids=[lifecycle_evidence_id],
        data_as_of=wins_snapshot["observed_at"],
        owner="Anna",
        now=_at(31),
    )
    report = add_decision_case_study(
        report,
        f"case-lifecycle-{lifecycle_id}",
        section_id="decision",
        decision_id=str(lifecycle_id),
        ticker="AAPL",
        title="AAPL initiation",
        process_summary="Four blind votes, rule checks and two approvals preceded execution.",
        outcome_summary="The signed WInS snapshot became the canonical portfolio input.",
        lesson="One identifier and one snapshot prevent parallel investment histories.",
        evidence_ids=[dossier_evidence_id, lifecycle_evidence_id],
        now=_at(31),
    )
    report = set_report_portfolio_snapshot(
        report,
        wins_snapshot["snapshot_id"],
        as_of=wins_snapshot["observed_at"],
        source="WInS reconciled canonical pipeline",
        positions=quant_input["portfolio"]["positions"],
        reconciled=True,
        reconciliation_id=reconciliation["reconciliation_id"],
        now=_at(31),
    )
    report = set_performance_attribution(
        report,
        as_of=_at(31),
        benchmark="SPY",
        portfolio_return=0.01,
        benchmark_return=0.008,
        contributions=[{"id": "AAPL", "label": "AAPL", "contribution": 0.01}],
        methodology="Arithmetic holding-period contribution from the canonical snapshot.",
        now=_at(31),
    )
    report = set_report_section_content(
        report,
        "decision",
        content="The evidence graph links dossier, decision, WInS reconciliation and Quant.",
        estimated_pages=1.5,
        ready_for_freeze=True,
        now=_at(31),
    )
    assert validate_report_workspace(report)["is_ready"] is True
    report = freeze_report(report, frozen_by="Anna", now=_at(32))
    report = record_report_approval(report, approver="Anna", now=_at(33))
    report = record_report_approval(report, approver="Jakub", now=_at(34))
    report = finalise_report(report, finalised_by="Anna", now=_at(35))
    export = build_export_ready_report(report)
    assert export["export_ready"] is True
    assert export["portfolio_snapshot"]["snapshot_id"] == quant_input["portfolio_snapshot_id"]
    assert export["portfolio_snapshot"]["reconciliation_id"] == reconciliation["reconciliation_id"]
    assert export["decision_case_studies"][0]["decision_id"] == str(lifecycle_id)

    qa = create_qa_rehearsal_workspace(
        "finale-e2e",
        team_members=["Anna", "Jakub", "Martin", "Lukáš"],
        created_by="Anna",
        now=_at(36),
    )
    qa = add_qa_question(
        qa,
        "why-aapl",
        prompt="Who approved AAPL and which evidence supports its position size?",
        model_answer="Four members voted; Anna and Jakub approved the rule-checked 5% size.",
        evidence_ids=[dossier_evidence_id, lifecycle_evidence_id],
        primary_responder="Anna",
        backup_responder="Jakub",
        time_limit_seconds=60,
        category="governance",
        follow_ups=["How do you prove that the position is in WInS?"],
        killer_question=True,
        created_by="Martin",
        now=_at(36),
    )
    qa = create_mock_round(
        qa,
        "mock-e2e",
        started_by="Martin",
        question_ids=["why-aapl"],
        now=_at(37),
    )
    qa = record_qa_response(
        qa,
        "mock-e2e",
        "why-aapl",
        responder="Anna",
        answer=(
            "Lifecycle "
            f"{lifecycle_id} records all votes, both approvals and the 5% sizing; "
            f"reconciliation {reconciliation['reconciliation_id']} signs the WInS snapshot."
        ),
        duration_seconds=42,
        scores={"clarity": 5, "evidence": 5, "client_fit": 4},
        evaluator="Martin",
        follow_up_answers={
            "How do you prove that the position is in WInS?": (
                f"The report and Quant input share snapshot {wins_snapshot['snapshot_id']}."
            )
        },
        now=_at(38),
    )
    qa = complete_mock_round(qa, "mock-e2e", completed_by="Martin", now=_at(39))
    assert qa["rounds"]["mock-e2e"]["summary"]["pass_rate_pct"] == 100.0
    assert killer_question_status(qa)["unresolved_count"] == 0

    final_lifecycle = get_investment_lifecycle(connection, lifecycle_id)
    assert final_lifecycle is not None
    assert final_lifecycle["state"] == "active"
    assert final_lifecycle["dossier_id"] == dossier["id"]
    assert final_lifecycle["universe_snapshot_id"] == universe["id"]
    assert final_lifecycle["wins_execution"]["execution"]["wins_transaction_id"] == (
        "WINS-AAPL-20260815-01"
    )
    assert (
        final_lifecycle["latest_reconciliation"]["reconciliation"][
            "canonical_reconciliation_id"
        ]
        == reconciliation["reconciliation_id"]
    )
    assert verify_lifecycle_audit_chain(connection, lifecycle_id)["valid"] is True
    json.dumps(
        {
            "lifecycle": final_lifecycle,
            "ledger": ledger,
            "quant_input": quant_input,
            "report": export,
            "qa": qa,
        },
        allow_nan=False,
    )
