from __future__ import annotations

from inspect import getsource

import numpy as np
import pandas as pd

from ui import investment_os
from ui.pages import wharton_dash


def test_primary_navigation_follows_one_competition_workflow() -> None:
    assert list(wharton_dash.COCKPIT_AREAS) == [
        "Home",
        "Client & Policy",
        "Research",
        "Decisions",
        "Portfolio",
        "Deliverables",
    ]
    visible_panels = {
        panel
        for panels in wharton_dash.COCKPIT_AREAS.values()
        for panel in panels
    }
    assert {
        "Thesis Monitor",
        "Approved Universe",
        "Decision Journal",
        "Advanced Monte Carlo",
        "Sub-Projects",
    }.isdisjoint(visible_panels)
    assert "Security Dossiers" in visible_panels
    assert "Investment Committee" in visible_panels


def test_strategy_workspace_has_no_parallel_security_or_decision_writers() -> None:
    source = getsource(wharton_dash._render_strategy_workspace)

    assert "Client Mandate" in source
    assert "Strategy Rulebook" in source
    assert "_render_thesis_monitor" not in source
    assert "_render_approved_universe" not in source
    assert "_render_decision_log" not in source
    assert "_render_review_learning" not in source


def test_only_committee_surface_contains_investment_vote_submission() -> None:
    dossier_source = getsource(investment_os.render_security_dossiers)
    committee_source = getsource(investment_os.render_investment_committee)

    assert "submit_committee_vote" not in dossier_source
    assert "submit_committee_vote" in committee_source
    assert "Voting happens only in Investment Committee" in dossier_source


def test_quant_engine_keeps_all_analytical_modules() -> None:
    assert wharton_dash.QUANT_MODULES == [
        "Mandate-Aware Optimizer",
        "Benchmark Analytics",
        "Cost-Aware Rebalance",
        "Historical Contribution & Risk Budget",
        "Simulation",
        "Methodology & Validation",
        "Models & Signals",
        "News Sentiment",
        "Robustness Check",
        "Backtest",
        "Run History",
    ]


def test_quant_sandbox_is_not_blocked_by_missing_competition_portfolio() -> None:
    canonical_input = {
        "allowed": False,
        "blockers": ["No reconciled WInS snapshot"],
        "portfolio_snapshot_id": None,
    }

    sandbox = wharton_dash._resolve_quant_run_source(
        "Standalone sandbox",
        canonical_input,
    )
    competition = wharton_dash._resolve_quant_run_source(
        "Competition portfolio",
        canonical_input,
    )

    assert sandbox["allowed"] is True
    assert sandbox["competition_mode"] is False
    assert sandbox["portfolio_snapshot_id"] is None
    assert competition["allowed"] is False
    assert competition["blockers"] == ["No reconciled WInS snapshot"]


def test_goal_return_history_uses_complete_calendar_years() -> None:
    complete = pd.bdate_range("2022-01-03", "2023-12-29")
    partial = pd.bdate_range("2024-10-01", "2024-12-31")
    index = complete.append(partial)
    daily = pd.DataFrame(
        {"AAA": np.full(len(index), 0.001), "BBB": np.full(len(index), 0.0005)},
        index=index,
    )

    annual = wharton_dash._annual_goal_return_history(daily)

    assert list(annual.index) == [2022, 2023]
    assert list(annual.columns) == ["AAA", "BBB"]
    assert (annual > 0).all().all()


def test_goal_candidates_are_aligned_and_normalized() -> None:
    result = {
        "tickers": ["BBB", "AAA"],
        "weights": np.array([0.25, 0.75]),
        "max_sharpe": {
            "symbols": ["AAA", "BBB"],
            "weights": np.array([0.6, 0.4]),
        },
        "mandate_aware": {
            "success": True,
            "symbols": ["AAA", "BBB", "CASH"],
            "weights": np.array([0.4, 0.4, 0.2]),
        },
    }

    candidates = wharton_dash._goal_candidate_weights(result, ["AAA", "BBB"])

    assert candidates["Current strategy"].tolist() == [0.75, 0.25]
    assert candidates["Max Sharpe"].tolist() == [0.6, 0.4]
    assert not any(name.startswith("Mandate-aware") for name in candidates)
    assert all(np.isclose(weights.sum(), 1.0) for weights in candidates.values())
