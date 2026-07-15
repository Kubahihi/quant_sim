from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from src.portfolio_tracker.strategy_store import (
    append_strategy_version,
    get_active_strategy_version,
    get_approved_security,
    get_holding_thesis,
    get_strategy_version,
    init_strategy_tables,
    list_approved_securities,
    list_company_research,
    list_holding_theses,
    list_strategy_versions,
    load_client_mandate,
    load_company_research,
    replace_approved_securities,
    save_client_mandate,
    save_company_research,
    set_active_strategy_version,
    upsert_approved_security,
    upsert_holding_thesis,
)


NOW = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)
LATER = NOW + timedelta(hours=2)


def test_client_mandate_is_singleton_with_automatic_versions_and_timestamps():
    connection = sqlite3.connect(":memory:")

    first = save_client_mandate(
        connection,
        {"risk_tolerance": "moderate", "horizon_years": 10},
        updated_by="Anna",
        now=NOW,
    )
    second = save_client_mandate(
        connection,
        {"risk_tolerance": "high", "horizon_years": 12},
        updated_by="Matej",
        now=LATER,
    )

    assert first["version"] == 1
    assert second == load_client_mandate(connection)
    assert second["version"] == 2
    assert second["payload"]["risk_tolerance"] == "high"
    assert second["created_at"] == NOW.isoformat()
    assert second["updated_at"] == LATER.isoformat()
    assert second["updated_by"] == "Matej"
    assert connection.execute("SELECT COUNT(*) FROM analytical_client_mandate").fetchone()[0] == 1


def test_strategy_versions_are_append_only_and_exactly_one_can_be_active():
    connection = sqlite3.connect(":memory:")
    first = append_strategy_version(
        connection,
        {"name": "Quality Compounders", "max_position_pct": 12},
        created_by="Anna",
        now=NOW,
    )
    second = append_strategy_version(
        connection,
        {"name": "Quality Compounders", "max_position_pct": 10},
        created_by="Matej",
        now=LATER,
    )

    assert first["version"] == 1
    assert second["version"] == 2
    assert get_active_strategy_version(connection) == second
    assert get_strategy_version(connection, 1)["payload"]["max_position_pct"] == 12
    assert [item["version"] for item in list_strategy_versions(connection)] == [2, 1]

    assert set_active_strategy_version(connection, 1)
    assert get_active_strategy_version(connection)["version"] == 1
    assert not set_active_strategy_version(connection, 999)
    assert get_strategy_version(connection, 2)["payload"]["max_position_pct"] == 10


def test_inactive_strategy_draft_does_not_replace_the_active_version():
    connection = sqlite3.connect(":memory:")
    active = append_strategy_version(connection, {"name": "Active"}, now=NOW)
    draft = append_strategy_version(connection, {"name": "Draft"}, activate=False, now=LATER)

    assert draft["is_active"] is False
    assert get_active_strategy_version(connection)["version"] == active["version"]


def test_holding_thesis_upsert_preserves_created_at_and_supports_status_filter():
    connection = sqlite3.connect(":memory:")
    upsert_holding_thesis(
        connection,
        " msft ",
        {"thesis": "Cloud scale", "invalidation": "Azure share loss"},
        status="watchlist",
        conviction=72.5,
        strategy_version=1,
        next_review_at="2026-08-01",
        updated_by="Anna",
        now=NOW,
    )
    updated = upsert_holding_thesis(
        connection,
        "MSFT",
        {"thesis": "Cloud and AI scale", "catalysts": ["Copilot adoption"]},
        status="holding",
        conviction=81,
        strategy_version=2,
        updated_by="Matej",
        now=LATER,
    )

    assert updated == get_holding_thesis(connection, "msft")
    assert updated["ticker"] == "MSFT"
    assert updated["conviction"] == pytest.approx(81)
    assert updated["created_at"] == NOW.isoformat()
    assert updated["updated_at"] == LATER.isoformat()
    assert updated["next_review_at"] is None
    assert list_holding_theses(connection, status="watchlist") == []
    assert [item["ticker"] for item in list_holding_theses(connection, status="holding")] == ["MSFT"]


def test_approved_security_list_can_show_approved_rejected_or_all():
    connection = sqlite3.connect(":memory:")
    upsert_approved_security(
        connection,
        "aapl",
        {"name": "Apple", "sector": "Technology"},
        updated_by="Anna",
        now=NOW,
    )
    upsert_approved_security(
        connection,
        "XYZ",
        {"reason": "Outside liquidity rule"},
        approved=False,
        now=NOW,
    )

    assert [item["ticker"] for item in list_approved_securities(connection)] == ["AAPL"]
    assert [item["ticker"] for item in list_approved_securities(connection, approved_only=False)] == ["XYZ"]
    assert [item["ticker"] for item in list_approved_securities(connection, approved_only=None)] == ["AAPL", "XYZ"]
    assert get_approved_security(connection, "AAPL")["payload"]["sector"] == "Technology"


def test_replace_approved_securities_replaces_the_whole_universe_and_normalises_rows():
    connection = sqlite3.connect(":memory:")
    upsert_approved_security(connection, "OLD", {"source": "stale"}, now=NOW)

    records = replace_approved_securities(
        connection,
        [
            {"ticker": " msft ", "payload": {"source": "official"}},
            {
                "ticker": "aapl",
                "approved": False,
                "payload": {"reason": "outside current list"},
            },
        ],
        updated_by="Anna",
        now=LATER,
    )

    assert [record["ticker"] for record in records] == ["AAPL", "MSFT"]
    assert get_approved_security(connection, "OLD") is None
    assert get_approved_security(connection, "MSFT") == {
        "ticker": "MSFT",
        "approved": True,
        "payload": {"source": "official"},
        "updated_by": "Anna",
        "created_at": LATER.isoformat(),
        "updated_at": LATER.isoformat(),
    }
    assert get_approved_security(connection, "AAPL")["approved"] is False


@pytest.mark.parametrize(
    "invalid_records",
    [
        [{"ticker": "MSFT", "payload": {}}, {"ticker": " ", "payload": {}}],
        [{"ticker": "MSFT", "payload": {}}, {"ticker": "msft", "payload": {}}],
        [{"ticker": "MSFT", "payload": {}}, {"ticker": "AAPL", "payload": []}],
        [{"ticker": "MSFT", "payload": {}}, {"ticker": "AAPL", "payload": {"bad": float("nan")}}],
        [{"ticker": "MSFT", "payload": {}}, {"ticker": "AAPL", "payload": {}, "approved": 1}],
    ],
)
def test_replace_approved_securities_validates_every_row_before_changing_data(invalid_records):
    connection = sqlite3.connect(":memory:")
    upsert_approved_security(connection, "OLD", {"source": "keep"}, now=NOW)

    with pytest.raises(ValueError):
        replace_approved_securities(connection, invalid_records, now=LATER)

    assert [record["ticker"] for record in list_approved_securities(connection)] == ["OLD"]
    assert get_approved_security(connection, "OLD")["payload"] == {"source": "keep"}


def test_replace_approved_securities_empty_input_clears_the_universe():
    connection = sqlite3.connect(":memory:")
    upsert_approved_security(connection, "OLD", {}, now=NOW)

    assert replace_approved_securities(connection, [], now=LATER) == []
    assert list_approved_securities(connection, approved_only=None) == []


def test_company_research_round_trip_updates_manual_swot_porter_and_peers():
    connection = sqlite3.connect(":memory:")
    first = save_company_research(
        connection,
        "nvda",
        {
            "swot": {"strengths": ["CUDA ecosystem"]},
            "porter": {"supplier_power": "medium"},
            "peer_tickers": ["AMD", "AVGO"],
        },
        updated_by="Anna",
        now=NOW,
    )
    second = save_company_research(
        connection,
        "NVDA",
        {
            "swot": {"strengths": ["CUDA ecosystem", "Scale"]},
            "peer_tickers": ["AMD", "AVGO", "INTC"],
        },
        updated_by="Matej",
        now=LATER,
    )

    assert first["created_at"] == NOW.isoformat()
    assert second == load_company_research(connection, "nvda")
    assert second["created_at"] == NOW.isoformat()
    assert second["updated_at"] == LATER.isoformat()
    assert second["payload"]["peer_tickers"] == ["AMD", "AVGO", "INTC"]
    assert [item["ticker"] for item in list_company_research(connection)] == ["NVDA"]


def test_plain_tuples_and_sqlite_rows_are_both_supported():
    tuple_connection = sqlite3.connect(":memory:")
    save_client_mandate(tuple_connection, {"goal": "capital growth"}, now=NOW)
    assert load_client_mandate(tuple_connection)["payload"]["goal"] == "capital growth"

    row_connection = sqlite3.connect(":memory:")
    row_connection.row_factory = sqlite3.Row
    save_company_research(row_connection, "TSM", {"peers": ["INTC"]}, now=NOW)
    assert load_company_research(row_connection, "tsm")["ticker"] == "TSM"


@pytest.mark.parametrize(
    "operation",
    [
        lambda connection: save_client_mandate(connection, {"bad": float("nan")}),
        lambda connection: append_strategy_version(connection, {"bad": object()}),
        lambda connection: save_company_research(connection, "AAPL", ["not", "an", "object"]),
        lambda connection: upsert_holding_thesis(connection, "AAPL", {}, conviction=float("inf")),
    ],
)
def test_invalid_json_and_non_finite_fields_are_rejected(operation):
    connection = sqlite3.connect(":memory:")
    with pytest.raises(ValueError):
        operation(connection)


def test_corrupt_payloads_are_not_returned_or_listed():
    connection = sqlite3.connect(":memory:")
    save_company_research(connection, "AAPL", {"swot": {}}, now=NOW)
    connection.execute(
        "UPDATE analytical_company_research SET payload_json = '{broken json' WHERE ticker = 'AAPL'"
    )
    connection.commit()

    assert load_company_research(connection, "AAPL") is None
    assert list_company_research(connection) == []


class _SyncingConnection:
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


def test_schema_initialization_and_each_mutation_sync_online_connections():
    connection = _SyncingConnection()

    init_strategy_tables(connection)
    save_client_mandate(connection, {"goal": "growth"}, now=NOW)
    append_strategy_version(connection, {"name": "Quality"}, now=NOW)
    save_company_research(connection, "AAPL", {"swot": {}}, now=NOW)

    assert connection.sync_calls == 4


def test_replace_approved_securities_commits_and_syncs_exactly_once():
    connection = _SyncingConnection()

    records = replace_approved_securities(
        connection,
        [
            {"ticker": "AAPL", "payload": {"source": "official"}},
            {"ticker": "MSFT", "payload": {"source": "official"}},
        ],
        now=NOW,
    )

    assert [record["ticker"] for record in records] == ["AAPL", "MSFT"]
    assert connection.commit_calls == 1
    assert connection.sync_calls == 1
