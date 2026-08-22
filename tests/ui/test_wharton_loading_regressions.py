from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ui.pages import wharton_dash


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_wharton_import_keeps_quant_stack_deferred() -> None:
    """A login-page import must not pull in optional numerical dependencies."""
    deferred_prefixes = (
        "src.analytics",
        "src.optimization",
        "src.simulation",
        "src.auth.database",
        "src.auth.manager",
        "src.auth.migrations",
        "cvxpy",
        "scipy",
        "statsmodels",
    )
    probe = (
        "import json, sys; "
        "import ui.pages.wharton_dash; "
        f"prefixes={deferred_prefixes!r}; "
        "loaded=sorted(name for name in sys.modules "
        "if any(name == prefix or name.startswith(prefix + '.') "
        "for prefix in prefixes)); "
        "print(json.dumps(loaded))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert json.loads(completed.stdout) == []


def test_cockpit_renders_only_active_panel_and_preserves_user_state(monkeypatch) -> None:
    draft = {"initial_fcff_growth": 17.0, "notes": "keep my edits"}
    cached_result = {"run_id": "existing-run"}
    session_state = {
        wharton_dash.QUANT_RESULT_KEY: cached_result,
        "company_dcf_draft": draft,
    }
    original_state = dict(session_state)
    rendered: list[str] = []

    monkeypatch.setattr(
        wharton_dash,
        "st",
        SimpleNamespace(session_state=session_state),
    )
    monkeypatch.setattr(wharton_dash, "_inject_cockpit_styles", lambda: None)
    monkeypatch.setattr(
        wharton_dash,
        "_get_current_profile",
        lambda: {"username": "Jakub"},
    )
    monkeypatch.setattr(wharton_dash, "init_db", lambda: None)
    monkeypatch.setattr(wharton_dash, "_render_header", lambda profile: None)
    monkeypatch.setattr(
        wharton_dash,
        "_render_cockpit_navigation",
        lambda profile, panels: "Research Workspace",
    )
    monkeypatch.setattr(
        wharton_dash,
        "_render_research_workspace",
        lambda profile: rendered.append("Research Workspace"),
    )

    inactive_renderers = (
        "_render_overview_action_center",
        "_render_competition_readiness",
        "_render_strategy_workspace",
        "_render_ios_security_dossiers",
        "_render_ios_investment_committee",
        "_render_competition_portfolio",
        "_render_ios_live_pipeline",
        "_render_risk_scenarios_workspace",
        "_render_report_pitch_workspace",
        "_render_rules_compliance_workspace",
    )

    def fail_if_rendered(*args, **kwargs):
        pytest.fail("An inactive cockpit panel performed work during this rerun.")

    for renderer_name in inactive_renderers:
        monkeypatch.setattr(wharton_dash, renderer_name, fail_if_rendered)

    wharton_dash.render_wharton_cockpit()

    assert rendered == ["Research Workspace"]
    assert session_state == original_state
    assert session_state["company_dcf_draft"] is draft
    assert session_state[wharton_dash.QUANT_RESULT_KEY] is cached_result
