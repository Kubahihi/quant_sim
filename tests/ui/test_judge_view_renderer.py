from __future__ import annotations

import ast
from inspect import getsource
from pathlib import Path

from ui import judge_view


MODULE_PATH = Path(judge_view.__file__)


def _tree() -> ast.Module:
    return ast.parse(MODULE_PATH.read_text(encoding="utf-8"))


def test_judge_view_exposes_only_read_only_controls() -> None:
    forbidden_calls = {
        "button",
        "data_editor",
        "download_button",
        "file_uploader",
        "form",
        "form_submit_button",
        "tabs",
        "text_input",
        "text_area",
        "number_input",
        "selectbox",
        "multiselect",
        "radio",
        "checkbox",
        "toggle",
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(_tree())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert forbidden_calls.isdisjoint(called_attributes)


def test_judge_view_does_not_import_data_stores_or_mutation_services() -> None:
    imports: set[str] = set()
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")

    assert imports <= {
        "__future__",
        "collections.abc",
        "html",
        "typing",
        "pandas",
        "streamlit",
    }
    source = MODULE_PATH.read_text(encoding="utf-8").casefold()
    assert "insert into" not in source
    assert "update " not in source
    assert "delete from" not in source
    assert "commit(" not in source
    assert "rerun(" not in source


def test_dynamic_unsafe_html_is_routed_through_escaping_helper() -> None:
    unsafe_calls = []
    for node in ast.walk(_tree()):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "markdown":
            continue
        unsafe_keyword = next(
            (keyword for keyword in node.keywords if keyword.arg == "unsafe_allow_html"),
            None,
        )
        if not unsafe_keyword or not isinstance(unsafe_keyword.value, ast.Constant):
            continue
        if unsafe_keyword.value.value is True:
            unsafe_calls.append(node)

    assert unsafe_calls
    for call in unsafe_calls:
        first_argument = call.args[0]
        is_static = isinstance(first_argument, ast.Constant)
        is_escaped = (
            isinstance(first_argument, ast.Call)
            and isinstance(first_argument.func, ast.Name)
            and first_argument.func.id == "_safe_markup"
        )
        assert is_static or is_escaped


def test_html_escaping_protects_model_values() -> None:
    payload = '<script>alert("judge")</script>'

    rendered = judge_view._safe_markup("<strong>{value}</strong>", value=payload)

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "&quot;judge&quot;" in rendered


def test_renderer_contains_the_complete_judge_review_path() -> None:
    source = getsource(judge_view)

    assert "Judge View · Read only" in source
    assert "Portfolio Journey" in source
    assert "Client & Strategy" in source
    assert "Investment Cases" in source
    assert "Risk & Scenarios" in source
    assert "Decision Trail" in source
    assert "Security Lineage" in source
    assert "Portfolio" in source
    assert "Report & Evidence" in source
    assert "Rules & Integrity" in source
    assert "Complete App Record" in source
    assert "Open a module" in source
    assert "st.expander" in source
    assert "Questions for the Team" in source
    assert 'width="stretch"' in source


def test_curated_judge_story_is_rendered_in_chronological_order() -> None:
    source = getsource(judge_view.render_judge_view)
    calls = (
        "_render_portfolio_journey",
        "_render_client_and_strategy",
        "_render_investment_cases",
        "_render_risk_and_scenarios",
        "_render_decisions",
        "_render_security_lineage",
        "_render_portfolio",
        "_render_report_and_evidence",
        "_render_rules_and_integrity",
        "_render_questions",
        "_render_complete_app_record",
    )

    offsets = [source.index(f"{call}(view_model)") for call in calls]
    assert offsets == sorted(offsets)


def test_journey_and_lineage_connect_research_decision_and_capital() -> None:
    model = {
        "mandate": {"available": True, "client_name": "Case Family"},
        "strategy": {"available": True, "name": "Client-first"},
        "readiness": {"available": True, "constitution_score": 80},
        "dossiers": [
            {
                "ticker": "AAA",
                "dossier_status": "frozen",
                "eligibility": "eligible",
                "review_date": "2026-09-15",
            }
        ],
        "decisions": {
            "count": 1,
            "fully_voted_count": 1,
            "items": [
                {"ticker": "AAA", "state": "active", "proposed_weight_pct": 8.0}
            ],
        },
        "portfolio": {
            "available": True,
            "reconciled": True,
            "snapshot_id": "wins-9",
            "positions": [
                {
                    "ticker": "AAA",
                    "weight_pct": 7.6,
                    "return_pct": 4.2,
                    "pnl": 3200,
                    "lifecycle_state": "active",
                    "client_goal": "Education",
                }
            ],
        },
        "evidence": {"source_count": 3, "thesis_review_count": 1},
        "report": {"available": False},
        "compliance": {"check_count": 0},
        "integrity": {"reconciliation": {"status": "reconciled"}},
        "app_record": {"modules": []},
    }

    stages = judge_view._journey_stages(model)
    lineage = judge_view._security_lineage_rows(model)

    assert [stage["label"] for stage in stages] == [
        "Client mandate",
        "Strategy",
        "Research",
        "Risk tests",
        "Committee",
        "Execute & reconcile",
        "Monitor & learn",
        "Report & defend",
        "Controls",
    ]
    assert stages[4]["status"] == "Fully voted"
    assert stages[5]["status"] == "Reconciled"
    assert lineage[0]["Security"] == "AAA"
    assert "8.0% proposed" in lineage[0]["Committee"]
    assert lineage[0]["Capital today"] == "7.6% held"
    assert "Education" in lineage[0]["Monitoring link"]
