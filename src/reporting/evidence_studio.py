"""Pure report-workspace model for evidence-backed competition deliverables.

The public functions in this module use copy-on-write semantics: callers pass a
JSON-compatible mapping and receive a new dictionary.  That makes the model
safe to persist in SQLite, a document store, or an ordinary JSON file without
giving the UI permission to silently rewrite a frozen report.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import date, datetime, timezone
import hashlib
import json
import math
from typing import Any


REPORT_TYPES = frozenset({"mid_project", "final"})
REPORT_STATUSES = frozenset({"draft", "frozen", "final"})

DEFAULT_REPORT_SCHEMAS: dict[str, dict[str, Any]] = {
    "mid_project": {
        "schema_version": "flexible-2026.1",
        "page_budget": 7.0,
        "sections": (
            ("client_and_strategy", "Client mandate and strategy", 2.0),
            ("portfolio_snapshot", "Portfolio snapshot", 2.0),
            ("decisions_and_learning", "Decisions and learning", 2.0),
            ("evidence_and_next_steps", "Evidence and next steps", 1.0),
        ),
    },
    "final": {
        "schema_version": "flexible-2026.1",
        "page_budget": 12.0,
        "sections": (
            ("executive_summary", "Executive summary", 1.0),
            ("client_mandate_strategy", "Client mandate and strategy", 2.0),
            ("portfolio_attribution", "Portfolio and attribution", 3.0),
            ("decision_case_studies", "Decision case studies", 2.0),
            ("risk_scenarios", "Risk and scenarios", 2.0),
            ("learning_conclusion", "Learning and conclusion", 2.0),
        ),
    },
}


def _text(value: Any, name: str, *, required: bool = False, limit: int = 20_000) -> str:
    result = " ".join(str(value or "").strip().split())
    if required and not result:
        raise ValueError(f"{name} must not be empty.")
    if len(result) > limit:
        raise ValueError(f"{name} must be at most {limit} characters.")
    return result


def _identifier(value: Any, name: str) -> str:
    result = _text(value, name, required=True, limit=160)
    if any(character.isspace() for character in result):
        raise ValueError(f"{name} must not contain whitespace.")
    return result


def _number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}.")
    return result


def _timestamp(value: date | datetime | str | None = None, *, name: str = "Timestamp") -> str:
    if value is None:
        parsed = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        return value.isoformat()
    else:
        raw = _text(value, name, required=True, limit=80)
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError as exc:
            try:
                return date.fromisoformat(raw).isoformat()
            except ValueError:
                raise ValueError(f"{name} must be an ISO date or timestamp.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _json_copy(value: Any, name: str = "Value") -> Any:
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON serialisable.") from exc
    return json.loads(encoded)


def _hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_draft(workspace: Mapping[str, Any]) -> dict[str, Any]:
    copied = _json_copy(workspace, "Workspace")
    if copied.get("status") != "draft":
        raise ValueError("Only a draft report can be edited.")
    return copied


def _section(workspace: Mapping[str, Any], section_id: str) -> dict[str, Any]:
    identifier = _identifier(section_id, "Section id")
    try:
        return workspace["sections"][identifier]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unknown section id: {identifier}.") from exc


def _normalise_section_schema(
    report_type: str,
    section_schema: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    raw_sections: Sequence[Any]
    if section_schema is None:
        raw_sections = [
            {"id": item[0], "title": item[1], "page_budget": item[2]}
            for item in DEFAULT_REPORT_SCHEMAS[report_type]["sections"]
        ]
    else:
        if isinstance(section_schema, (str, bytes, bytearray)):
            raise ValueError("Section schema must be a sequence of mappings.")
        raw_sections = section_schema

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_sections:
        if not isinstance(raw, Mapping):
            raise ValueError("Every section schema entry must be a mapping.")
        section_id = _identifier(raw.get("id"), "Section id")
        if section_id in seen:
            raise ValueError(f"Duplicate section id: {section_id}.")
        seen.add(section_id)
        result.append(
            {
                "id": section_id,
                "title": _text(raw.get("title"), "Section title", required=True, limit=300),
                "page_budget": _number(
                    raw.get("page_budget"), "Section page budget", minimum=0.1
                ),
                "owner": _text(raw.get("owner"), "Section owner", limit=200),
                "reviewer": _text(raw.get("reviewer"), "Section reviewer", limit=200),
            }
        )
    if not result:
        raise ValueError("At least one report section is required.")
    return result


def create_report_workspace(
    report_id: str,
    report_type: str,
    title: str,
    *,
    created_by: str,
    page_budget: float | None = None,
    section_schema: Sequence[Mapping[str, Any]] | None = None,
    schema_version: str | None = None,
    required_approvers: Sequence[str] | None = None,
    now: date | datetime | str | None = None,
) -> dict[str, Any]:
    """Create a flexible mid-project or final-report workspace."""
    kind = str(report_type or "").strip().lower().replace("-", "_").replace(" ", "_")
    if kind not in REPORT_TYPES:
        raise ValueError("Report type must be mid_project or final.")
    creator = _text(created_by, "Created by", required=True, limit=200)
    sections = _normalise_section_schema(kind, section_schema)
    budget = _number(
        page_budget
        if page_budget is not None
        else DEFAULT_REPORT_SCHEMAS[kind]["page_budget"],
        "Page budget",
        minimum=0.1,
    )
    allocated = sum(section["page_budget"] for section in sections)
    if allocated > budget + 1e-9:
        raise ValueError("Section page budgets exceed the report page budget.")

    approver_values = list(required_approvers or [creator])
    approvers: list[str] = []
    for value in approver_values:
        member = _text(value, "Required approver", required=True, limit=200)
        if member not in approvers:
            approvers.append(member)
    if not approvers:
        raise ValueError("At least one final approver is required.")

    timestamp = _timestamp(now)
    return {
        "report_id": _identifier(report_id, "Report id"),
        "report_type": kind,
        "title": _text(title, "Title", required=True, limit=500),
        "schema_version": _text(
            schema_version or DEFAULT_REPORT_SCHEMAS[kind]["schema_version"],
            "Schema version",
            required=True,
            limit=100,
        ),
        "page_budget": budget,
        "status": "draft",
        "created_by": creator,
        "created_at": timestamp,
        "updated_at": timestamp,
        "required_approvers": approvers,
        "sections": {
            item["id"]: {
                **item,
                "status": "draft",
                "estimated_pages": 0.0,
                "content": "",
                "claim_ids": [],
                "figure_ids": [],
                "case_study_ids": [],
            }
            for item in sections
        },
        "section_order": [item["id"] for item in sections],
        "evidence": {},
        "claims": {},
        "figures": {},
        "case_studies": {},
        "portfolio_snapshot": None,
        "performance_attribution": None,
        "freeze": None,
        "approvals": {},
        "approval_history": [],
        "finalised_at": None,
        "finalised_by": None,
    }


def assign_report_section(
    workspace: Mapping[str, Any],
    section_id: str,
    *,
    owner: str,
    reviewer: str,
    page_budget: float | None = None,
    now: date | datetime | str | None = None,
) -> dict[str, Any]:
    result = _require_draft(workspace)
    section = _section(result, section_id)
    section["owner"] = _text(owner, "Section owner", required=True, limit=200)
    section["reviewer"] = _text(reviewer, "Section reviewer", required=True, limit=200)
    if section["owner"] == section["reviewer"]:
        raise ValueError("Section owner and reviewer must be different people.")
    if page_budget is not None:
        section["page_budget"] = _number(
            page_budget, "Section page budget", minimum=0.1
        )
    if sum(item["page_budget"] for item in result["sections"].values()) > (
        result["page_budget"] + 1e-9
    ):
        raise ValueError("Section page budgets exceed the report page budget.")
    result["updated_at"] = _timestamp(now)
    return result


def set_report_section_content(
    workspace: Mapping[str, Any],
    section_id: str,
    *,
    content: str,
    estimated_pages: float,
    ready_for_freeze: bool = False,
    now: date | datetime | str | None = None,
) -> dict[str, Any]:
    result = _require_draft(workspace)
    section = _section(result, section_id)
    body = _text(content, "Section content", required=ready_for_freeze, limit=200_000)
    pages = _number(estimated_pages, "Estimated pages", minimum=0.0)
    if pages > section["page_budget"] + 1e-9:
        raise ValueError("Estimated pages exceed the section page budget.")
    section["content"] = body
    section["estimated_pages"] = pages
    section["status"] = "ready" if ready_for_freeze else "draft"
    result["updated_at"] = _timestamp(now)
    return result


def register_report_evidence(
    workspace: Mapping[str, Any],
    evidence_id: str,
    *,
    title: str,
    citation: str,
    source_locator: str,
    source_type: str = "other",
    verified_by: str,
    published_at: date | datetime | str | None = None,
    accessed_at: date | datetime | str | None = None,
    notes: str = "",
    now: date | datetime | str | None = None,
) -> dict[str, Any]:
    result = _require_draft(workspace)
    identifier = _identifier(evidence_id, "Evidence id")
    if identifier in result["evidence"]:
        raise ValueError(f"Evidence id already exists: {identifier}.")
    source_kind = _text(source_type, "Source type", required=True, limit=100).lower()
    record = {
        "evidence_id": identifier,
        "title": _text(title, "Evidence title", required=True, limit=500),
        "citation": _text(citation, "Citation", required=True, limit=4_000),
        "source_locator": _text(
            source_locator, "Source locator", required=True, limit=4_000
        ),
        "source_type": source_kind,
        "verified_by": _text(verified_by, "Verified by", required=True, limit=200),
        "published_at": _timestamp(published_at, name="Published at")
        if published_at is not None
        else None,
        "accessed_at": _timestamp(accessed_at, name="Accessed at")
        if accessed_at is not None
        else None,
        "notes": _text(notes, "Evidence notes", limit=20_000),
    }
    record["record_hash"] = _hash(record)
    result["evidence"][identifier] = record
    result["updated_at"] = _timestamp(now)
    return result


def add_report_claim(
    workspace: Mapping[str, Any],
    claim_id: str,
    *,
    section_id: str,
    statement: str,
    evidence_ids: Sequence[str],
    created_by: str,
    now: date | datetime | str | None = None,
) -> dict[str, Any]:
    result = _require_draft(workspace)
    identifier = _identifier(claim_id, "Claim id")
    if identifier in result["claims"]:
        raise ValueError(f"Claim id already exists: {identifier}.")
    section = _section(result, section_id)
    if isinstance(evidence_ids, (str, bytes, bytearray)):
        raise ValueError("Evidence ids must be a sequence.")
    links = [_identifier(value, "Evidence id") for value in evidence_ids]
    links = list(dict.fromkeys(links))
    if not links:
        raise ValueError("A report claim must cite at least one evidence record.")
    unknown = [value for value in links if value not in result["evidence"]]
    if unknown:
        raise ValueError(f"Unknown evidence ids: {', '.join(unknown)}.")
    result["claims"][identifier] = {
        "claim_id": identifier,
        "section_id": section["id"],
        "statement": _text(statement, "Claim statement", required=True, limit=20_000),
        "evidence_ids": links,
        "created_by": _text(created_by, "Created by", required=True, limit=200),
        "created_at": _timestamp(now),
    }
    section["claim_ids"].append(identifier)
    result["updated_at"] = _timestamp(now)
    return result


def register_report_figure(
    workspace: Mapping[str, Any],
    figure_id: str,
    *,
    section_id: str,
    title: str,
    caption: str,
    artifact_locator: str,
    evidence_ids: Sequence[str],
    data_as_of: date | datetime | str,
    owner: str,
    now: date | datetime | str | None = None,
) -> dict[str, Any]:
    result = _require_draft(workspace)
    identifier = _identifier(figure_id, "Figure id")
    if identifier in result["figures"]:
        raise ValueError(f"Figure id already exists: {identifier}.")
    section = _section(result, section_id)
    if isinstance(evidence_ids, (str, bytes, bytearray)):
        raise ValueError("Evidence ids must be a sequence.")
    links = list(dict.fromkeys(_identifier(value, "Evidence id") for value in evidence_ids))
    if not links or any(value not in result["evidence"] for value in links):
        raise ValueError("Every figure must link to at least one known evidence record.")
    result["figures"][identifier] = {
        "figure_id": identifier,
        "section_id": section["id"],
        "title": _text(title, "Figure title", required=True, limit=500),
        "caption": _text(caption, "Figure caption", required=True, limit=4_000),
        "artifact_locator": _text(
            artifact_locator, "Artifact locator", required=True, limit=4_000
        ),
        "evidence_ids": links,
        "data_as_of": _timestamp(data_as_of, name="Data as of"),
        "owner": _text(owner, "Figure owner", required=True, limit=200),
    }
    section["figure_ids"].append(identifier)
    result["updated_at"] = _timestamp(now)
    return result


def add_decision_case_study(
    workspace: Mapping[str, Any],
    case_study_id: str,
    *,
    section_id: str,
    decision_id: str,
    ticker: str,
    title: str,
    process_summary: str,
    outcome_summary: str,
    lesson: str,
    evidence_ids: Sequence[str],
    now: date | datetime | str | None = None,
) -> dict[str, Any]:
    result = _require_draft(workspace)
    identifier = _identifier(case_study_id, "Case study id")
    if identifier in result["case_studies"]:
        raise ValueError(f"Case study id already exists: {identifier}.")
    section = _section(result, section_id)
    if isinstance(evidence_ids, (str, bytes, bytearray)):
        raise ValueError("Evidence ids must be a sequence.")
    links = list(dict.fromkeys(_identifier(value, "Evidence id") for value in evidence_ids))
    if not links or any(value not in result["evidence"] for value in links):
        raise ValueError("Every case study must link to known evidence.")
    result["case_studies"][identifier] = {
        "case_study_id": identifier,
        "section_id": section["id"],
        "decision_id": _identifier(decision_id, "Decision id"),
        "ticker": _text(ticker, "Ticker", required=True, limit=32).upper(),
        "title": _text(title, "Case study title", required=True, limit=500),
        "process_summary": _text(
            process_summary, "Process summary", required=True, limit=20_000
        ),
        "outcome_summary": _text(
            outcome_summary, "Outcome summary", required=True, limit=20_000
        ),
        "lesson": _text(lesson, "Lesson", required=True, limit=20_000),
        "evidence_ids": links,
    }
    section["case_study_ids"].append(identifier)
    result["updated_at"] = _timestamp(now)
    return result


def set_report_portfolio_snapshot(
    workspace: Mapping[str, Any],
    snapshot_id: str,
    *,
    as_of: date | datetime | str,
    source: str,
    positions: Sequence[Mapping[str, Any]],
    reconciled: bool,
    reconciliation_id: str | None = None,
    now: date | datetime | str | None = None,
) -> dict[str, Any]:
    """Attach the exact, immutable portfolio state used by the report."""
    result = _require_draft(workspace)
    if isinstance(positions, (str, bytes, bytearray)):
        raise ValueError("Positions must be a sequence of mappings.")
    normalised: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in positions:
        if not isinstance(raw, Mapping):
            raise ValueError("Every position must be a mapping.")
        security_id = _identifier(raw.get("security_id") or raw.get("ticker"), "Security id")
        if security_id in seen:
            raise ValueError(f"Duplicate portfolio security: {security_id}.")
        seen.add(security_id)
        normalised.append(
            {
                "security_id": security_id,
                "ticker": _text(raw.get("ticker") or security_id, "Ticker", required=True, limit=32).upper(),
                "weight": _number(raw.get("weight"), "Position weight", minimum=0.0, maximum=1.0),
                "market_value": _number(raw.get("market_value", 0.0), "Market value", minimum=0.0),
                "currency": _text(raw.get("currency", "USD"), "Currency", required=True, limit=12).upper(),
            }
        )
    if sum(item["weight"] for item in normalised) > 1.000001:
        raise ValueError("Portfolio position weights must not exceed 100%.")
    if reconciled and not reconciliation_id:
        raise ValueError("A reconciled snapshot requires a reconciliation id.")
    snapshot = {
        "snapshot_id": _identifier(snapshot_id, "Snapshot id"),
        "as_of": _timestamp(as_of, name="Portfolio as of"),
        "source": _text(source, "Portfolio source", required=True, limit=500),
        "reconciled": bool(reconciled),
        "reconciliation_id": _identifier(reconciliation_id, "Reconciliation id")
        if reconciliation_id
        else None,
        "positions": sorted(normalised, key=lambda item: item["security_id"]),
    }
    snapshot["snapshot_hash"] = _hash(snapshot)
    result["portfolio_snapshot"] = snapshot
    result["updated_at"] = _timestamp(now)
    return result


def set_performance_attribution(
    workspace: Mapping[str, Any],
    *,
    as_of: date | datetime | str,
    benchmark: str,
    portfolio_return: float,
    benchmark_return: float,
    contributions: Sequence[Mapping[str, Any]],
    methodology: str,
    now: date | datetime | str | None = None,
) -> dict[str, Any]:
    result = _require_draft(workspace)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in contributions:
        if not isinstance(raw, Mapping):
            raise ValueError("Every attribution contribution must be a mapping.")
        key = _identifier(raw.get("id") or raw.get("label"), "Contribution id")
        if key in seen:
            raise ValueError(f"Duplicate contribution id: {key}.")
        seen.add(key)
        rows.append(
            {
                "id": key,
                "label": _text(raw.get("label") or key, "Contribution label", required=True, limit=300),
                "contribution": _number(raw.get("contribution"), "Contribution"),
            }
        )
    if not rows:
        raise ValueError("At least one attribution contribution is required.")
    portfolio_value = _number(portfolio_return, "Portfolio return")
    benchmark_value = _number(benchmark_return, "Benchmark return")
    attributed = sum(item["contribution"] for item in rows)
    result["performance_attribution"] = {
        "as_of": _timestamp(as_of, name="Attribution as of"),
        "benchmark": _text(benchmark, "Benchmark", required=True, limit=100).upper(),
        "portfolio_return": portfolio_value,
        "benchmark_return": benchmark_value,
        "active_return": portfolio_value - benchmark_value,
        "contributions": rows,
        "attributed_return": attributed,
        "residual": portfolio_value - attributed,
        "methodology": _text(methodology, "Methodology", required=True, limit=4_000),
    }
    result["updated_at"] = _timestamp(now)
    return result


def validate_report_workspace(workspace: Mapping[str, Any]) -> dict[str, Any]:
    """Return deterministic blocking issues and readiness information."""
    value = _json_copy(workspace, "Workspace")
    issues: list[dict[str, str]] = []

    def issue(code: str, message: str) -> None:
        issues.append({"code": code, "message": message})

    if value.get("status") not in REPORT_STATUSES:
        issue("invalid_status", "Report status is invalid.")
    sections = value.get("sections") if isinstance(value.get("sections"), dict) else {}
    section_order = value.get("section_order") if isinstance(value.get("section_order"), list) else []
    if len(section_order) != len(sections) or set(section_order) != set(sections):
        issue("section_order", "Section order must contain every section exactly once.")
    allocated = sum(float(item.get("page_budget", 0.0)) for item in sections.values())
    if allocated > float(value.get("page_budget", 0.0)) + 1e-9:
        issue("page_budget", "Section page budgets exceed the report page budget.")
    estimated = 0.0
    for section_id in section_order:
        section = sections.get(section_id, {})
        if not section.get("owner") or not section.get("reviewer"):
            issue("section_assignment", f"Section {section_id} needs an owner and reviewer.")
        elif section["owner"] == section["reviewer"]:
            issue("review_independence", f"Section {section_id} owner and reviewer must differ.")
        if section.get("status") != "ready" or not section.get("content"):
            issue("section_not_ready", f"Section {section_id} is not ready for freeze.")
        pages = float(section.get("estimated_pages", 0.0))
        estimated += pages
        if pages > float(section.get("page_budget", 0.0)) + 1e-9:
            issue("section_over_budget", f"Section {section_id} exceeds its page budget.")
        if not section.get("claim_ids"):
            issue("section_without_claim", f"Section {section_id} has no evidence-backed claim.")
        for collection_key, graph_key, code in (
            ("claim_ids", "claims", "section_claim_link"),
            ("figure_ids", "figures", "section_figure_link"),
            ("case_study_ids", "case_studies", "section_case_study_link"),
        ):
            for linked_id in section.get(collection_key, []):
                record = (value.get(graph_key) or {}).get(linked_id)
                if record is None or record.get("section_id") != section_id:
                    issue(code, f"Section {section_id} has an invalid {linked_id} link.")
    if estimated > float(value.get("page_budget", 0.0)) + 1e-9:
        issue("estimated_page_budget", "Estimated pages exceed the report page budget.")

    evidence = value.get("evidence") if isinstance(value.get("evidence"), dict) else {}
    for evidence_id, record in evidence.items():
        expected = dict(record)
        stored_hash = expected.pop("record_hash", None)
        if stored_hash != _hash(expected):
            issue("evidence_hash", f"Evidence {evidence_id} failed its integrity check.")
        if not record.get("citation") or not record.get("verified_by"):
            issue("evidence_incomplete", f"Evidence {evidence_id} lacks citation or verification.")
    claims = value.get("claims") if isinstance(value.get("claims"), dict) else {}
    for claim_id, claim in claims.items():
        links = claim.get("evidence_ids", [])
        if not links or any(link not in evidence for link in links):
            issue("claim_evidence", f"Claim {claim_id} is not fully supported by evidence.")
        section = sections.get(claim.get("section_id"), {})
        if claim_id not in section.get("claim_ids", []):
            issue("orphaned_claim", f"Claim {claim_id} is not registered in its section.")
    for figure_id, figure in (value.get("figures") or {}).items():
        figure_links = figure.get("evidence_ids", [])
        if not figure.get("caption") or not figure_links or any(
            link not in evidence for link in figure_links
        ):
            issue("figure_evidence", f"Figure {figure_id} lacks a caption or evidence.")
        section = sections.get(figure.get("section_id"), {})
        if figure_id not in section.get("figure_ids", []):
            issue("orphaned_figure", f"Figure {figure_id} is not registered in its section.")
    for case_id, case_study in (value.get("case_studies") or {}).items():
        case_links = case_study.get("evidence_ids", [])
        if not case_links or any(link not in evidence for link in case_links):
            issue("case_study_evidence", f"Case study {case_id} lacks valid evidence.")
        section = sections.get(case_study.get("section_id"), {})
        if case_id not in section.get("case_study_ids", []):
            issue("orphaned_case_study", f"Case study {case_id} is not registered in its section.")

    snapshot = value.get("portfolio_snapshot")
    if not isinstance(snapshot, dict):
        issue("portfolio_snapshot", "An as-of portfolio snapshot is required.")
    else:
        expected = dict(snapshot)
        stored_hash = expected.pop("snapshot_hash", None)
        if stored_hash != _hash(expected):
            issue("portfolio_snapshot_hash", "Portfolio snapshot failed its integrity check.")
        if not snapshot.get("reconciled"):
            issue("portfolio_reconciliation", "The report snapshot must be reconciled.")
    if value.get("report_type") == "final":
        if not value.get("performance_attribution"):
            issue("performance_attribution", "A final report requires performance attribution.")
        if not value.get("case_studies"):
            issue("case_study", "A final report requires at least one decision case study.")

    codes = [item["code"] for item in issues]
    return {
        "is_ready": not issues,
        "issue_count": len(issues),
        "issues": issues,
        "issue_codes": codes,
        "page_budget": float(value.get("page_budget", 0.0)),
        "allocated_pages": allocated,
        "estimated_pages": estimated,
        "claim_count": len(claims),
        "evidence_count": len(evidence),
        "figure_count": len(value.get("figures") or {}),
        "case_study_count": len(value.get("case_studies") or {}),
    }


def _freeze_content(workspace: Mapping[str, Any]) -> dict[str, Any]:
    value = _json_copy(workspace)
    for key in (
        "status",
        "updated_at",
        "freeze",
        "approvals",
        "approval_history",
        "finalised_at",
        "finalised_by",
    ):
        value.pop(key, None)
    return value


def freeze_report(
    workspace: Mapping[str, Any],
    *,
    frozen_by: str,
    now: date | datetime | str | None = None,
) -> dict[str, Any]:
    result = _require_draft(workspace)
    validation = validate_report_workspace(result)
    if not validation["is_ready"]:
        raise ValueError(
            "Report is not ready to freeze: " + ", ".join(validation["issue_codes"])
        )
    timestamp = _timestamp(now)
    result["status"] = "frozen"
    result["freeze"] = {
        "frozen_by": _text(frozen_by, "Frozen by", required=True, limit=200),
        "frozen_at": timestamp,
        "content_hash": _hash(_freeze_content(result)),
    }
    result["approvals"] = {}
    result["updated_at"] = timestamp
    return result


def record_report_approval(
    workspace: Mapping[str, Any],
    *,
    approver: str,
    decision: str = "approved",
    notes: str = "",
    now: date | datetime | str | None = None,
) -> dict[str, Any]:
    result = _json_copy(workspace, "Workspace")
    if result.get("status") != "frozen" or not result.get("freeze"):
        raise ValueError("Only a frozen report can be approved.")
    actor = _text(approver, "Approver", required=True, limit=200)
    if actor not in result.get("required_approvers", []):
        raise ValueError("Approver is not in the report's required approver list.")
    outcome = str(decision or "").strip().lower().replace("-", "_").replace(" ", "_")
    if outcome not in {"approved", "changes_requested"}:
        raise ValueError("Approval decision must be approved or changes_requested.")
    if result["freeze"]["content_hash"] != _hash(_freeze_content(result)):
        raise ValueError("Frozen report content has changed since freeze.")
    timestamp = _timestamp(now)
    record = {
        "approver": actor,
        "decision": outcome,
        "notes": _text(notes, "Approval notes", limit=10_000),
        "decided_at": timestamp,
        "content_hash": result["freeze"]["content_hash"],
    }
    result["approval_history"].append(record)
    if outcome == "changes_requested":
        result["status"] = "draft"
        result["freeze"] = None
        result["approvals"] = {}
    else:
        result["approvals"][actor] = record
    result["updated_at"] = timestamp
    return result


def finalise_report(
    workspace: Mapping[str, Any],
    *,
    finalised_by: str,
    now: date | datetime | str | None = None,
) -> dict[str, Any]:
    result = _json_copy(workspace, "Workspace")
    if result.get("status") != "frozen" or not result.get("freeze"):
        raise ValueError("Only a frozen report can be finalised.")
    if result["freeze"]["content_hash"] != _hash(_freeze_content(result)):
        raise ValueError("Frozen report content has changed since freeze.")
    missing = [
        actor
        for actor in result.get("required_approvers", [])
        if result.get("approvals", {}).get(actor, {}).get("decision") != "approved"
    ]
    if missing:
        raise ValueError("Missing required approvals: " + ", ".join(missing) + ".")
    timestamp = _timestamp(now)
    result["status"] = "final"
    result["finalised_by"] = _text(
        finalised_by, "Finalised by", required=True, limit=200
    )
    result["finalised_at"] = timestamp
    result["updated_at"] = timestamp
    return result


def build_export_ready_report(workspace: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve graph links into a deterministic renderer-neutral export model."""
    value = _json_copy(workspace, "Workspace")
    if value.get("status") != "final":
        raise ValueError("Only a final report can be exported.")
    if not value.get("freeze") or value["freeze"].get("content_hash") != _hash(
        _freeze_content(value)
    ):
        raise ValueError("Final report content failed its frozen integrity check.")
    sections: list[dict[str, Any]] = []
    for section_id in value["section_order"]:
        section = deepcopy(value["sections"][section_id])
        section["claims"] = []
        for claim_id in section.pop("claim_ids", []):
            claim = deepcopy(value["claims"][claim_id])
            claim["evidence"] = [
                deepcopy(value["evidence"][item]) for item in claim["evidence_ids"]
            ]
            section["claims"].append(claim)
        section["figures"] = [
            deepcopy(value["figures"][item]) for item in section.pop("figure_ids", [])
        ]
        section["case_studies"] = [
            deepcopy(value["case_studies"][item])
            for item in section.pop("case_study_ids", [])
        ]
        sections.append(section)
    model = {
        "export_ready": True,
        "metadata": {
            "report_id": value["report_id"],
            "report_type": value["report_type"],
            "title": value["title"],
            "schema_version": value["schema_version"],
            "page_budget": value["page_budget"],
            "finalised_at": value["finalised_at"],
            "finalised_by": value["finalised_by"],
        },
        "portfolio_snapshot": value["portfolio_snapshot"],
        "performance_attribution": value["performance_attribution"],
        "sections": sections,
        "figure_register": [value["figures"][key] for key in sorted(value["figures"])],
        "decision_case_studies": [
            value["case_studies"][key] for key in sorted(value["case_studies"])
        ],
        "audit": {
            "freeze": value["freeze"],
            "approvals": [value["approvals"][key] for key in sorted(value["approvals"])],
        },
    }
    model["export_hash"] = _hash(model)
    return _json_copy(model)


__all__ = [
    "DEFAULT_REPORT_SCHEMAS",
    "REPORT_STATUSES",
    "REPORT_TYPES",
    "add_decision_case_study",
    "add_report_claim",
    "assign_report_section",
    "build_export_ready_report",
    "create_report_workspace",
    "finalise_report",
    "freeze_report",
    "record_report_approval",
    "register_report_evidence",
    "register_report_figure",
    "set_performance_attribution",
    "set_report_portfolio_snapshot",
    "set_report_section_content",
    "validate_report_workspace",
]
