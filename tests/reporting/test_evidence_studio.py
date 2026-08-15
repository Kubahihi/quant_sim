from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json

import pytest

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


NOW = datetime(2026, 8, 15, 10, 0, tzinfo=timezone.utc)


def _complete_final_report():
    workspace = create_report_workspace(
        "final-2027",
        "final",
        "Final competition report",
        created_by="Anna",
        required_approvers=["Anna", "Boris"],
        page_budget=2,
        section_schema=[
            {
                "id": "strategy",
                "title": "Strategy",
                "page_budget": 1,
                "owner": "Anna",
                "reviewer": "Boris",
            },
            {
                "id": "decisions",
                "title": "Decisions",
                "page_budget": 1,
                "owner": "Boris",
                "reviewer": "Anna",
            },
        ],
        now=NOW,
    )
    workspace = register_report_evidence(
        workspace,
        "case-fact-1",
        title="Client goal",
        citation="Wharton case study, p. 2",
        source_locator="case-study.pdf#page=2",
        source_type="official_case",
        verified_by="Anna",
        accessed_at=NOW,
        now=NOW,
    )
    workspace = add_report_claim(
        workspace,
        "claim-client-fit",
        section_id="strategy",
        statement="The allocation prioritises the client's first goal.",
        evidence_ids=["case-fact-1"],
        created_by="Anna",
        now=NOW,
    )
    workspace = add_report_claim(
        workspace,
        "claim-learning",
        section_id="decisions",
        statement="The team reduced the position when its KPI breached.",
        evidence_ids=["case-fact-1"],
        created_by="Boris",
        now=NOW,
    )
    workspace = register_report_figure(
        workspace,
        "fig-allocation",
        section_id="strategy",
        title="Goal-bucket allocation",
        caption="Reconciled allocation as of the report date.",
        artifact_locator="figures/allocation.svg",
        evidence_ids=["case-fact-1"],
        data_as_of=NOW,
        owner="Anna",
        now=NOW,
    )
    workspace = add_decision_case_study(
        workspace,
        "case-aapl",
        section_id="decisions",
        decision_id="decision-aapl-1",
        ticker="aapl",
        title="AAPL sizing review",
        process_summary="The dossier was frozen before independent voting.",
        outcome_summary="The KPI breach led to a controlled reduction.",
        lesson="Predefined thresholds made the review faster.",
        evidence_ids=["case-fact-1"],
        now=NOW,
    )
    workspace = set_report_portfolio_snapshot(
        workspace,
        "wins-2026-08-15",
        as_of=NOW,
        source="WInS reconciled export",
        positions=[
            {"ticker": "AAPL", "weight": 0.6, "market_value": 60_000, "currency": "USD"},
            {"ticker": "CASH", "weight": 0.4, "market_value": 40_000, "currency": "USD"},
        ],
        reconciled=True,
        reconciliation_id="recon-44",
        now=NOW,
    )
    workspace = set_performance_attribution(
        workspace,
        as_of=NOW,
        benchmark="SPY",
        portfolio_return=0.08,
        benchmark_return=0.05,
        contributions=[
            {"id": "aapl", "label": "AAPL", "contribution": 0.06},
            {"id": "cash", "label": "Cash", "contribution": 0.01},
        ],
        methodology="Arithmetic holding-period contribution; residual reported separately.",
        now=NOW,
    )
    workspace = set_report_section_content(
        workspace,
        "strategy",
        content="Client-linked strategy narrative.",
        estimated_pages=0.9,
        ready_for_freeze=True,
        now=NOW,
    )
    workspace = set_report_section_content(
        workspace,
        "decisions",
        content="Decision case studies and lessons.",
        estimated_pages=0.8,
        ready_for_freeze=True,
        now=NOW,
    )
    return workspace


def test_final_report_freeze_approval_and_export_resolve_evidence_graph():
    draft = _complete_final_report()
    original = deepcopy(draft)

    validation = validate_report_workspace(draft)
    frozen = freeze_report(draft, frozen_by="Anna", now=NOW)
    approved_once = record_report_approval(frozen, approver="Anna", now=NOW)
    approved = record_report_approval(approved_once, approver="Boris", now=NOW)
    final = finalise_report(approved, finalised_by="Anna", now=NOW)
    exported = build_export_ready_report(final)

    assert validation["is_ready"] is True
    assert draft == original
    assert frozen["status"] == "frozen"
    assert len(frozen["freeze"]["content_hash"]) == 64
    assert final["status"] == "final"
    assert exported["export_ready"] is True
    assert exported["metadata"]["page_budget"] == 2
    assert exported["sections"][0]["claims"][0]["evidence"][0]["citation"]
    assert exported["portfolio_snapshot"]["reconciliation_id"] == "recon-44"
    assert exported["performance_attribution"]["residual"] == pytest.approx(0.01)
    assert len(exported["figure_register"]) == 1
    assert len(exported["decision_case_studies"]) == 1
    json.dumps(exported, allow_nan=False)


def test_report_cannot_freeze_without_complete_assignments_claims_and_reconciliation():
    workspace = create_report_workspace(
        "mid-1",
        "mid-project",
        "Mid-project report",
        created_by="Anna",
        section_schema=[{"id": "status", "title": "Status", "page_budget": 1}],
        page_budget=1,
        now=NOW,
    )
    workspace = set_report_portfolio_snapshot(
        workspace,
        "tracker-only",
        as_of=NOW,
        source="Manual tracker",
        positions=[],
        reconciled=False,
        now=NOW,
    )

    result = validate_report_workspace(workspace)

    assert result["is_ready"] is False
    assert {"section_assignment", "section_not_ready", "section_without_claim", "portfolio_reconciliation"} <= set(
        result["issue_codes"]
    )
    with pytest.raises(ValueError, match="not ready"):
        freeze_report(workspace, frozen_by="Anna", now=NOW)


def test_frozen_report_rejects_content_edits_and_change_request_returns_to_draft():
    frozen = freeze_report(_complete_final_report(), frozen_by="Anna", now=NOW)

    with pytest.raises(ValueError, match="draft"):
        set_report_section_content(
            frozen,
            "strategy",
            content="Retroactive rewrite",
            estimated_pages=0.5,
            ready_for_freeze=True,
            now=NOW,
        )

    reopened = record_report_approval(
        frozen,
        approver="Boris",
        decision="changes requested",
        notes="Clarify the benchmark choice.",
        now=NOW,
    )

    assert reopened["status"] == "draft"
    assert reopened["freeze"] is None
    assert reopened["approval_history"][-1]["decision"] == "changes_requested"


def test_page_budget_and_reference_invariants_are_enforced():
    with pytest.raises(ValueError, match="exceed"):
        create_report_workspace(
            "bad-budget",
            "final",
            "Bad budget",
            created_by="Anna",
            page_budget=1,
            section_schema=[{"id": "body", "title": "Body", "page_budget": 2}],
            now=NOW,
        )

    workspace = create_report_workspace(
        "mid-2",
        "mid_project",
        "Mid",
        created_by="Anna",
        section_schema=[{"id": "body", "title": "Body", "page_budget": 1}],
        page_budget=1,
        now=NOW,
    )
    with pytest.raises(ValueError, match="Unknown evidence"):
        add_report_claim(
            workspace,
            "claim-1",
            section_id="body",
            statement="Unsupported claim",
            evidence_ids=["missing"],
            created_by="Anna",
            now=NOW,
        )

