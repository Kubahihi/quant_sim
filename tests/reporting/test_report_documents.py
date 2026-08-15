from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import re
from xml.etree import ElementTree
import zipfile

import pytest

from src.reporting.evidence_studio import (
    add_decision_case_study,
    add_report_claim,
    create_report_workspace,
    finalise_report,
    freeze_report,
    record_report_approval,
    register_report_evidence,
    set_performance_attribution,
    set_report_portfolio_snapshot,
    set_report_section_content,
)
from src.reporting.report_documents import (
    generate_evidence_report_documents,
    generate_evidence_report_docx,
    generate_evidence_report_pdf,
    inspect_docx_bytes,
    inspect_pdf_bytes,
)


NOW = datetime(2026, 8, 15, 10, 0, tzinfo=timezone.utc)
WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _final_workspace():
    workspace = create_report_workspace(
        "final-export-1",
        "final",
        "Client Goal Investment Report",
        created_by="Anna",
        required_approvers=["Anna", "Boris"],
        page_budget=2,
        section_schema=[
            {
                "id": "strategy",
                "title": "Client strategy and decisions",
                "page_budget": 2,
                "owner": "Anna",
                "reviewer": "Boris",
            }
        ],
        now=NOW,
    )
    workspace = register_report_evidence(
        workspace,
        "case-goal",
        title="Client liquidity goal",
        citation="Official case study, page 2",
        source_locator="case-study.pdf#page=2",
        source_type="official_case",
        verified_by="Boris",
        accessed_at=NOW,
        now=NOW,
    )
    workspace = add_report_claim(
        workspace,
        "claim-fit",
        section_id="strategy",
        statement="The portfolio preserves the client's first liquidity goal.",
        evidence_ids=["case-goal"],
        created_by="Anna",
        now=NOW,
    )
    workspace = add_decision_case_study(
        workspace,
        "case-aapl",
        section_id="strategy",
        decision_id="decision-aapl-1",
        ticker="AAPL",
        title="Sizing with a predefined cap",
        process_summary="The dossier was frozen before independent voting.",
        outcome_summary="The approved starter position stayed inside the cap.",
        lesson="Explicit rules made sizing auditable.",
        evidence_ids=["case-goal"],
        now=NOW,
    )
    workspace = set_report_portfolio_snapshot(
        workspace,
        "wins-snapshot-44",
        as_of=NOW,
        source="Reconciled WInS export",
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
            {"id": "aapl", "label": "AAPL", "contribution": 0.07},
        ],
        methodology="Arithmetic holding-period contribution.",
        now=NOW,
    )
    workspace = set_report_section_content(
        workspace,
        "strategy",
        content="The strategy connects every position to a documented client goal.",
        estimated_pages=1.4,
        ready_for_freeze=True,
        now=NOW,
    )
    workspace = freeze_report(workspace, frozen_by="Anna", now=NOW)
    workspace = record_report_approval(workspace, approver="Anna", now=NOW)
    workspace = record_report_approval(workspace, approver="Boris", now=NOW)
    return finalise_report(workspace, finalised_by="Anna", now=NOW)


def test_docx_is_valid_ooxml_and_contains_report_evidence_and_audit_text():
    data = generate_evidence_report_docx(_final_workspace())
    diagnostics = inspect_docx_bytes(data)

    assert data.startswith(b"PK")
    assert diagnostics["is_valid"] is True
    assert diagnostics["table_count"] >= 3
    assert "Client Goal Investment Report" in diagnostics["text"]
    assert "Official case study, page 2" in diagnostics["text"]
    assert "decision-aapl-1" in diagnostics["text"]
    assert "Export integrity hash" in diagnostics["text"]

    with zipfile.ZipFile(BytesIO(data)) as package:
        assert package.testzip() is None
        document = ElementTree.fromstring(package.read("word/document.xml"))
        styles = ElementTree.fromstring(package.read("word/styles.xml"))
        assert document.tag == f"{{{WORD_NS}}}document"
        style_ids = {
            node.attrib[f"{{{WORD_NS}}}styleId"]
            for node in styles.findall(f"{{{WORD_NS}}}style")
        }
        assert {"Normal", "Heading1", "Heading2", "Heading3", "Citation"} <= style_ids


def test_pdf_has_valid_signature_pages_metadata_and_extractable_text_when_available():
    data = generate_evidence_report_pdf(_final_workspace())
    diagnostics = inspect_pdf_bytes(data)

    assert data.startswith(b"%PDF-")
    assert data.rstrip().endswith(b"%%EOF")
    assert diagnostics["is_valid"] is True
    assert diagnostics["page_count"] >= 2
    assert len(data) > 10_000
    assert re.search(rb"/Title\s*\(Client Goal Investment Report\)", data)

    try:
        from pypdf import PdfReader
    except ImportError:
        PdfReader = None
    if PdfReader is not None:
        extracted = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(data)).pages)
        assert "Client Goal Investment Report" in extracted
        assert "Official case study" in extracted


def test_bundle_returns_both_download_ready_formats():
    documents = generate_evidence_report_documents(
        _final_workspace(), include_appendices=False
    )

    assert set(documents) == {"docx", "pdf"}
    assert inspect_docx_bytes(documents["docx"])["is_valid"] is True
    assert inspect_pdf_bytes(documents["pdf"])["is_valid"] is True


def test_export_refuses_unfrozen_or_unapproved_workspace():
    draft = create_report_workspace(
        "draft",
        "mid_project",
        "Draft report",
        created_by="Anna",
        section_schema=[{"id": "body", "title": "Body", "page_budget": 1}],
        page_budget=1,
        now=NOW,
    )

    with pytest.raises(ValueError, match="final"):
        generate_evidence_report_docx(draft)
    with pytest.raises(ValueError, match="final"):
        generate_evidence_report_pdf(draft)


def test_byte_inspectors_reject_fake_files():
    with pytest.raises(ValueError, match="DOCX"):
        inspect_docx_bytes(b"PK-not-a-zip")
    with pytest.raises(ValueError, match="PDF"):
        inspect_pdf_bytes(b"%PDF-1.7\nno pages")

