from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timezone
import json

import pytest

from src.portfolio_tracker.work_items import (
    add_work_item,
    create_work_item_registry,
    transition_work_item,
    validate_work_item_registry,
    work_item_dashboard,
    work_items_for_context,
)


NOW = datetime(2026, 8, 15, 10, 0, tzinfo=timezone.utc)


def _registry():
    return create_work_item_registry(
        "team-work",
        team_members=["Anna", "Boris"],
        client_goal_ids=["goal-liquidity"],
        dossier_ids=["dossier-aapl"],
        decision_ids=["decision-aapl-1"],
        created_by="Anna",
        now=NOW,
    )


def test_task_inherits_concrete_investment_links_from_subproject():
    registry = _registry()
    original = deepcopy(registry)
    registry = add_work_item(
        registry,
        "project-aapl-review",
        title="AAPL thesis review",
        kind="subproject",
        owner="Anna",
        due_at=date(2026, 8, 30),
        client_goal_id="goal-liquidity",
        dossier_id="dossier-aapl",
        decision_id="decision-aapl-1",
        created_by="Anna",
        now=NOW,
    )
    registry = add_work_item(
        registry,
        "task-refresh-kpi",
        title="Refresh margin KPI",
        kind="task",
        owner="Boris",
        due_at=date(2026, 8, 20),
        parent_id="project-aapl-review",
        created_by="Anna",
        now=NOW,
    )

    child = registry["items"]["task-refresh-kpi"]

    assert _registry() == original
    assert child["links"] == {
        "client_goal_id": "goal-liquidity",
        "dossier_id": "dossier-aapl",
        "decision_id": "decision-aapl-1",
    }
    assert child["explicit_links"] == []
    assert work_items_for_context(registry, decision_id="decision-aapl-1") == [
        registry["items"]["project-aapl-review"],
        child,
    ]
    assert validate_work_item_registry(registry)["is_valid"] is True
    json.dumps(registry, allow_nan=False)


def test_status_workflow_requires_notes_and_closed_children_before_subproject_completion():
    registry = add_work_item(
        _registry(),
        "project",
        title="Review",
        kind="subproject",
        owner="Anna",
        due_at=date(2026, 8, 30),
        client_goal_id="goal-liquidity",
        dossier_id="dossier-aapl",
        decision_id="decision-aapl-1",
        created_by="Anna",
        now=NOW,
    )
    registry = add_work_item(
        registry,
        "task",
        title="Evidence refresh",
        kind="task",
        owner="Boris",
        due_at=date(2026, 8, 20),
        parent_id="project",
        created_by="Boris",
        now=NOW,
    )
    registry = transition_work_item(registry, "project", "in progress", actor="Anna", now=NOW)
    registry = transition_work_item(registry, "task", "in_progress", actor="Boris", now=NOW)

    with pytest.raises(ValueError, match="open children"):
        transition_work_item(
            registry, "project", "done", actor="Anna", note="Complete.", now=NOW
        )
    with pytest.raises(ValueError, match="note"):
        transition_work_item(registry, "task", "blocked", actor="Boris", now=NOW)

    registry = transition_work_item(
        registry, "task", "done", actor="Boris", note="KPI evidence attached.", now=NOW
    )
    registry = transition_work_item(
        registry, "project", "done", actor="Anna", note="Review signed off.", now=NOW
    )
    assert registry["items"]["project"]["status"] == "done"
    assert registry["items"]["task"]["completion_note"] == "KPI evidence attached."


def test_missing_or_unknown_context_link_is_rejected_and_dashboard_flags_overdue():
    registry = _registry()
    with pytest.raises(ValueError, match="missing required links"):
        add_work_item(
            registry,
            "orphan",
            title="Orphan task",
            kind="task",
            owner="Anna",
            due_at=date(2026, 8, 10),
            created_by="Anna",
            now=NOW,
        )
    with pytest.raises(ValueError, match="Unknown dossier"):
        add_work_item(
            registry,
            "bad-link",
            title="Bad link",
            kind="task",
            owner="Anna",
            due_at=date(2026, 8, 10),
            client_goal_id="goal-liquidity",
            dossier_id="dossier-missing",
            decision_id="decision-aapl-1",
            created_by="Anna",
            now=NOW,
        )

    registry = add_work_item(
        registry,
        "overdue",
        title="Overdue linked task",
        kind="task",
        owner="Anna",
        due_at=date(2026, 8, 10),
        client_goal_id="goal-liquidity",
        dossier_id="dossier-aapl",
        decision_id="decision-aapl-1",
        created_by="Anna",
        now=NOW,
    )
    dashboard = work_item_dashboard(registry, as_of=NOW)
    assert dashboard["overdue_count"] == 1
    assert dashboard["overdue_items"][0]["work_item_id"] == "overdue"

