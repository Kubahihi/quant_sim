from __future__ import annotations

import inspect
import sqlite3

import pytest

from ui import investment_os
from ui.pages import wharton_dash


def test_decision_journal_is_a_read_only_canonical_projection() -> None:
    source = inspect.getsource(wharton_dash._render_decision_log)

    assert "canonical_investment_lifecycles" in source
    assert "Legacy records — read-only" in source
    assert "INSERT INTO decision_log" not in source
    assert "UPDATE decision_log" not in source
    assert "form_submit_button" not in source
    assert "pre_vote" not in source
    assert "post_vote" not in source


def test_portfolio_tracker_cannot_render_the_legacy_wins_uploader() -> None:
    source = inspect.getsource(wharton_dash._render_competition_portfolio)
    module_source = inspect.getsource(wharton_dash)

    assert "_render_wins_reconciliation" not in source
    assert not hasattr(wharton_dash, "_render_wins_reconciliation")
    assert "wins_reconciliation_upload" not in module_source
    assert '"WInS positions snapshot"' not in module_source
    assert "Live Portfolio & Data Reliability" in source


def test_live_pipeline_persists_legacy_ledger_migration() -> None:
    source = inspect.getsource(investment_os.render_live_portfolio_pipeline)

    assert "migrate_reconciliation_ledger" in source
    assert "automatic-ledger-migration" in source
    assert "_save_pipeline_workspace" in source


def test_pending_tracker_projection_is_not_counted_as_live_portfolio() -> None:
    positions = [
        {"id": 1, "ticker": "OPEN", "status": "open"},
        {"id": 2, "ticker": "PENDING", "status": "pending_reconciliation"},
        {"id": 3, "ticker": "CLOSED", "status": "closed"},
    ]

    settled = wharton_dash._settled_competition_positions(positions)

    assert [item["ticker"] for item in settled] == ["OPEN", "CLOSED"]
    assert wharton_dash._competition_position_status_label(positions[1]) == (
        "Pending WInS reconciliation"
    )
    portfolio_source = inspect.getsource(wharton_dash._render_competition_portfolio)
    rules_source = inspect.getsource(wharton_dash._render_competition_rules)
    alignment_source = inspect.getsource(wharton_dash._render_strategy_alignment)
    pretrade_source = inspect.getsource(wharton_dash._render_pretrade_lab)
    bond_source = inspect.getsource(wharton_dash._render_bond_analysis)
    assert "calculate_portfolio_performance(settled_positions" in portfolio_source
    assert "_competition_position_status_label(row)" in portfolio_source
    assert "_settled_competition_positions" in rules_source
    assert "_settled_competition_positions" in alignment_source
    assert "_settled_competition_positions" in pretrade_source
    assert "_settled_competition_positions" in bond_source


def test_legacy_position_reference_has_no_direct_write_path() -> None:
    source = inspect.getsource(wharton_dash._render_competition_portfolio)

    assert "Legacy position form (read-only migration reference)" in source
    assert "competition_add_position" not in source
    assert "INSERT INTO competition_positions" not in source
    assert "Direct position entry disabled" not in source


def test_tracker_mutations_fail_closed_for_canonical_positions() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE competition_positions "
        "(id INTEGER PRIMARY KEY, lifecycle_id INTEGER)"
    )
    conn.executemany(
        "INSERT INTO competition_positions (id, lifecycle_id) VALUES (?, ?)",
        [(1, None), (2, 42)],
    )

    wharton_dash._require_legacy_position_mutation(conn, 1)
    with pytest.raises(PermissionError, match="lifecycle #42"):
        wharton_dash._require_legacy_position_mutation(conn, 2)
    with pytest.raises(LookupError, match="no longer exists"):
        wharton_dash._require_legacy_position_mutation(conn, 999)

    source = inspect.getsource(wharton_dash._render_competition_portfolio)
    assert source.count("_require_legacy_position_mutation") == 3
    assert source.count("AND lifecycle_id IS NULL") >= 4


def test_canonical_projection_uniqueness_is_enforced_in_database() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE competition_positions "
        "(id INTEGER PRIMARY KEY, lifecycle_id INTEGER)"
    )

    assert wharton_dash._ensure_canonical_position_uniqueness(conn) == []
    index_names = {
        str(row["name"])
        for row in conn.execute("PRAGMA index_list(competition_positions)").fetchall()
    }
    assert "ux_competition_positions_lifecycle_id" in index_names

    conn.execute(
        "INSERT INTO competition_positions (id, lifecycle_id) VALUES (?, ?)",
        (1, 7),
    )
    with pytest.raises(sqlite3.IntegrityError, match="one tracker projection"):
        conn.execute(
            "INSERT INTO competition_positions (id, lifecycle_id) VALUES (?, ?)",
            (2, 7),
        )


def test_existing_duplicate_projections_are_preserved_and_quarantined() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE competition_positions "
        "(id INTEGER PRIMARY KEY, lifecycle_id INTEGER)"
    )
    conn.executemany(
        "INSERT INTO competition_positions (id, lifecycle_id) VALUES (?, ?)",
        [(1, 9), (2, 9)],
    )

    assert wharton_dash._ensure_canonical_position_uniqueness(conn) == [9]
    assert conn.execute(
        "SELECT COUNT(*) FROM competition_positions WHERE lifecycle_id = 9"
    ).fetchone()[0] == 2
    with pytest.raises(sqlite3.IntegrityError, match="one tracker projection"):
        conn.execute(
            "INSERT INTO competition_positions (id, lifecycle_id) VALUES (?, ?)",
            (3, 9),
        )

    assert wharton_dash._duplicate_canonical_projection_ids(
        [
            {"id": 1, "lifecycle_id": 9},
            {"id": 2, "lifecycle_id": 9},
            {"id": 3, "lifecycle_id": None},
        ]
    ) == [9]
