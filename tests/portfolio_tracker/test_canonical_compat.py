from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sqlite3

from src.portfolio_tracker.authoritative_universe_store import (
    publish_authoritative_universe,
)
from src.portfolio_tracker.canonical_compat import (
    list_canonical_first_approved_securities,
    list_canonical_first_holding_theses,
)
from src.portfolio_tracker.security_dossier_store import (
    append_dossier_version,
    create_security_dossier,
    upsert_kpi_definition,
)
from src.portfolio_tracker.strategy_store import (
    upsert_approved_security,
    upsert_holding_thesis,
)


NOW = datetime(2026, 8, 20, 9, 0, tzinfo=timezone.utc)
LATER = NOW + timedelta(hours=1)


def _connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    return connection


class _SelectOnlyConnection:
    """Fail the test immediately if a compatibility read attempts a write."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.statements: list[str] = []

    def execute(self, statement: str, parameters=()):
        normalised = " ".join(statement.split()).upper()
        assert normalised.startswith("SELECT "), statement
        self.statements.append(normalised)
        return self.connection.execute(statement, parameters)


def test_canonical_dossier_wins_and_legacy_only_thesis_remains_available():
    connection = _connection()
    upsert_holding_thesis(
        connection,
        "AAPL",
        {"investment_thesis": "Stale legacy thesis", "sector": "Legacy sector"},
        status="holding",
        conviction=1,
        strategy_version=1,
        next_review_at="2026-08-21",
        updated_by="Legacy analyst",
        now=NOW,
    )
    upsert_holding_thesis(
        connection,
        "MSFT",
        {"investment_thesis": "Legacy-only cloud thesis", "sector": "Technology"},
        status="holding",
        conviction=4,
        strategy_version=2,
        next_review_at="2026-09-01",
        updated_by="Legacy analyst",
        now=NOW,
    )
    dossier = create_security_dossier(
        connection,
        "AAPL",
        {
            "thesis": "Canonical installed-base thesis",
            "invalidation_condition": "Services growth misses the threshold twice",
            "client_goal": "Long-term growth",
            "conviction": 4,
            "review_date": "2026-09-15",
            "status": "watch",
            "strategy_version": 7,
            "fair_value_bear": 90,
            "fair_value_base": 130,
            "fair_value_bull": 170,
            "margin_of_safety_pct": 18.5,
        },
        created_by="Canonical analyst",
        now=NOW,
    )
    append_dossier_version(
        connection,
        dossier["id"],
        {
            **dossier["current_version"]["payload"],
            "thesis": "Canonical revised installed-base thesis",
            "conviction": 5,
        },
        created_by="Canonical reviewer",
        now=LATER,
    )
    upsert_kpi_definition(
        connection,
        dossier["id"],
        "services_growth",
        name="Services growth",
        baseline=0.10,
        expected_min=0.08,
        breach_below=0.05,
        unit="percent",
        source="10-Q",
        frequency="quarterly",
        owner="Canonical analyst",
        updated_by="Canonical analyst",
        now=LATER,
    )

    records = list_canonical_first_holding_theses(_SelectOnlyConnection(connection))

    assert [record["ticker"] for record in records] == ["AAPL", "MSFT"]
    aapl, msft = records
    assert aapl["source_system"] == "canonical_security_dossier"
    assert aapl["canonical_dossier_version"] == 2
    assert aapl["status"] == "watch"
    assert aapl["conviction"] == 5
    assert aapl["strategy_version"] == 7
    assert aapl["next_review_at"] == "2026-09-15"
    assert aapl["payload"]["investment_thesis"] == "Canonical revised installed-base thesis"
    assert aapl["payload"]["invalidation"] == "Services growth misses the threshold twice"
    assert aapl["payload"]["primary_goal"] == "Long-term growth"
    assert aapl["payload"]["goals"] == ["Long-term growth"]
    assert aapl["payload"]["fair_value_scenarios"] == {
        "bear": 90,
        "base": 130,
        "bull": 170,
    }
    assert aapl["payload"]["margin_of_safety"] == 18.5
    assert aapl["payload"]["monitoring_kpis"][0]["kpi_key"] == "services_growth"
    assert msft["source_system"] == "legacy_analytical_holding_thesis"
    assert msft["payload"]["investment_thesis"] == "Legacy-only cloud thesis"


def test_thesis_status_filter_runs_after_canonical_precedence():
    connection = _connection()
    upsert_holding_thesis(
        connection,
        "AAPL",
        {"investment_thesis": "Legacy"},
        status="holding",
        now=NOW,
    )
    upsert_holding_thesis(
        connection,
        "MSFT",
        {"investment_thesis": "Legacy-only"},
        status="holding",
        now=NOW,
    )
    create_security_dossier(
        connection,
        "AAPL",
        {"thesis": "Canonical", "status": "watch"},
        created_by="Analyst",
        now=LATER,
    )

    records = list_canonical_first_holding_theses(connection, status="holding")

    assert [record["ticker"] for record in records] == ["MSFT"]


def test_active_authoritative_universe_wins_and_legacy_only_names_fall_back():
    connection = _connection()
    upsert_approved_security(
        connection,
        "AAPL",
        {"security_type": "Legacy stock", "source_name": "Stale list"},
        approved=True,
        updated_by="Legacy analyst",
        now=NOW,
    )
    upsert_approved_security(
        connection,
        "MSFT",
        {"security_type": "Common stock", "source_name": "Legacy-only list"},
        approved=True,
        updated_by="Legacy analyst",
        now=NOW,
    )
    snapshot = publish_authoritative_universe(
        connection,
        [
            {
                "ticker": "AAPL",
                "eligibility": "ineligible",
                "security_type": "Common stock",
                "payload": {"reason": "Outside the active competition list"},
            },
            {
                "ticker": "NVDA",
                "eligibility": "unknown",
                "security_type": "Common stock",
            },
            {
                "ticker": "XYZ",
                "eligibility": "eligible",
                "security_type": "ETF",
                "payload": {"exchange": "NYSE"},
            },
        ],
        source_name="Official WInS universe",
        source_url="https://example.test/wins.csv",
        provenance_status="official",
        as_of_date="2026-08-20",
        published_by="Captain",
        now=LATER,
    )

    approved = list_canonical_first_approved_securities(connection)
    all_records = list_canonical_first_approved_securities(
        _SelectOnlyConnection(connection), approved_only=None
    )
    by_ticker = {record["ticker"]: record for record in all_records}

    assert [record["ticker"] for record in approved] == ["MSFT", "XYZ"]
    assert sorted(by_ticker) == ["AAPL", "MSFT", "NVDA", "XYZ"]
    assert by_ticker["AAPL"]["approved"] is False
    assert by_ticker["AAPL"]["status"] == "ineligible"
    assert by_ticker["AAPL"]["security_type"] == "Common stock"
    assert by_ticker["AAPL"]["source_system"] == "active_authoritative_universe"
    assert by_ticker["AAPL"]["authoritative_snapshot_id"] == snapshot["id"]
    assert by_ticker["AAPL"]["payload"]["source_name"] == "Official WInS universe"
    assert by_ticker["AAPL"]["payload"]["source_as_of"] == "2026-08-20"
    assert by_ticker["MSFT"]["source_system"] == "legacy_analytical_approved_security"
    assert by_ticker["NVDA"]["status"] == "unknown"
    assert by_ticker["XYZ"]["approved"] is True


def test_empty_database_reads_do_not_create_tables_or_change_data():
    connection = _connection()
    proxy = _SelectOnlyConnection(connection)
    before_changes = connection.total_changes

    assert list_canonical_first_holding_theses(proxy) == []
    assert list_canonical_first_approved_securities(proxy, approved_only=None) == []

    tables = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    ).fetchall()
    assert tables == []
    assert connection.total_changes == before_changes
    assert proxy.statements
