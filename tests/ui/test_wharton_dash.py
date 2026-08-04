from __future__ import annotations

import sqlite3
from pathlib import Path
import bcrypt
import math
import pytest
from types import SimpleNamespace

from ui.pages import wharton_dash


def test_strategy_form_number_helpers_reject_nan_and_preserve_zero():
    assert wharton_dash._finite_form_number(float("nan"), 7.0) == 7.0
    assert wharton_dash._finite_form_number(float("inf"), 8.0) == 8.0
    assert wharton_dash._finite_form_number("2.5", 0.0) == 2.5
    assert wharton_dash._saved_number({"limit": 0.0}, "limit", 0.15) == 0.0
    assert math.isclose(wharton_dash._saved_number({}, "limit", 0.15), 0.15)
    assert "Bond Analysis" in wharton_dash.COCKPIT_AREAS["Research"]
    assert "Bond Analysis" in wharton_dash.COCKPIT_PANEL_DESCRIPTIONS
    assert "Commodity Analysis" in wharton_dash.COCKPIT_AREAS["Research"]
    assert "Commodity Analysis" in wharton_dash.COCKPIT_PANEL_DESCRIPTIONS
    assert "Currency Risk & Hedging" in wharton_dash.COCKPIT_AREAS["Risk & Quant"]
    assert "Currency Risk & Hedging" in wharton_dash.COCKPIT_PANEL_DESCRIPTIONS
    assert wharton_dash._parse_tickers("", allow_empty=True) == []
    assert "Mandate-Aware Optimizer" in wharton_dash.QUANT_MODULES


def test_black_litterman_view_parser_and_optimizer_metadata_adapter():
    assert wharton_dash._parse_black_litterman_views(
        "msft=10%; NVDA=12.5"
    ) == {"MSFT": pytest.approx(0.10), "NVDA": pytest.approx(0.125)}
    with pytest.raises(ValueError, match="TICKER=annual return"):
        wharton_dash._parse_black_litterman_views("MSFT 10")
    with pytest.raises(ValueError, match="unique"):
        wharton_dash._parse_black_litterman_views("MSFT=10,MSFT=11")

    metadata = wharton_dash._build_optimizer_asset_metadata({
        "approved_securities": [
            {"ticker": "MSFT", "approved": True, "payload": {"source": "committee"}}
        ],
        "theses": [
            {
                "ticker": "MSFT",
                "payload": {
                    "sector": "Technology",
                    "beta": 1.05,
                    "tags": ["liquid"],
                },
            }
        ],
    })
    assert metadata["MSFT"] == {
        "source": "committee",
        "approved": True,
        "sector": "Technology",
        "beta": 1.05,
        "tags": ["liquid"],
    }


def test_mandate_optimizer_renderer_reaches_metrics_and_audit(monkeypatch):
    rendered = {"metrics": 0, "frames": 0}

    class FakeColumn:
        def metric(self, *args, **kwargs):
            rendered["metrics"] += 1

    fake_streamlit = SimpleNamespace(
        markdown=lambda *args, **kwargs: None,
        caption=lambda *args, **kwargs: None,
        columns=lambda count: [FakeColumn() for _ in range(count)],
        dataframe=lambda *args, **kwargs: rendered.__setitem__(
            "frames", rendered["frames"] + 1
        ),
        bar_chart=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(wharton_dash, "st", fake_streamlit)
    result = {
        "inputs": {"strategy_rulebook_applied": True},
        "mandate_aware": {
            "success": True,
            "objective": "maximum_utility",
            "symbols": ["AAA", "BBB"],
            "weights": [0.55, 0.45],
            "current_weights": [0.50, 0.50],
            "expected_return": 0.08,
            "volatility": 0.12,
            "sharpe_ratio": 0.42,
            "historical_cvar_daily": 0.02,
            "turnover": 0.10,
            "transaction_cost_drag": 0.0001,
            "constraint_report": [
                {
                    "name": "asset:AAA",
                    "actual": 0.55,
                    "minimum": 0.0,
                    "maximum": 0.60,
                    "binding": False,
                    "passed": True,
                }
            ],
            "warnings": [],
        },
    }

    wharton_dash._render_mandate_aware_optimizer(result, advanced=False)

    assert rendered == {"metrics": 6, "frames": 2}


def _configure_temp_wharton(monkeypatch, tmp_path: Path, password: str = "new-team-pass") -> Path:
    monkeypatch.setenv("QUANT_SIM_ENV", "development")
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "wharton.db"
    upload_dir = tmp_path / "data" / "wharton_uploads"
    monkeypatch.setattr(wharton_dash, "DB_PATH", db_path)
    monkeypatch.setattr(wharton_dash, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(wharton_dash, "DEFAULT_PASSWORD", password)
    return db_path


def test_init_db_uses_configured_paths_when_cwd_changes(monkeypatch, tmp_path):
    db_path = _configure_temp_wharton(monkeypatch, tmp_path)
    runner_cwd = tmp_path / "runner"
    runner_cwd.mkdir()
    monkeypatch.chdir(runner_cwd)

    wharton_dash.init_db()

    assert db_path.exists()
    assert Path(wharton_dash.UPLOAD_DIR).exists()
    assert not (runner_cwd / "data" / "wharton_production.db").exists()
    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    assert "competition_compliance" in tables
    assert "competition_positions" in tables
    assert "analytical_client_mandate" in tables
    assert "analytical_strategy_versions" in tables
    assert "analytical_holding_theses" in tables
    assert "analytical_approved_securities" in tables
    assert "analytical_company_research" in tables
    assert "analytical_research_sources" in tables
    assert "analytical_catalyst_events" in tables
    assert "analytical_thesis_reviews" in tables
    assert "analytical_decision_reviews" in tables
    with sqlite3.connect(db_path) as connection:
        decision_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(decision_log)")
        }
        position_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(competition_positions)")
        }
    assert {
        "horizon_days", "benchmark_ticker", "expected_return_min",
        "expected_return_max", "decision_confidence", "target_condition",
        "invalidation_condition", "planned_weight",
    }.issubset(decision_columns)
    assert {
        "bond_instrument_type", "bond_category", "isin", "issuer", "currency",
        "face_value", "coupon_rate", "maturity_date", "coupon_frequency",
        "next_coupon_date", "entry_accrued_interest", "accrued_interest",
        "entry_fx_rate_to_usd", "fx_rate_to_usd", "exit_accrued_interest",
        "exit_fx_rate_to_usd", "coupon_income", "yield_to_maturity",
        "modified_duration", "convexity", "credit_rating", "seniority",
        "valuation_source", "source_url", "price_observed_at",
        "callable", "call_date", "call_price", "benchmark_name",
        "benchmark_yield", "income_yield", "default_probability", "recovery_rate",
        "competition_eligibility_status", "eligibility_source", "eligibility_checked_at",
    }.issubset(position_columns)


def test_init_db_syncs_seeded_users_to_current_password(monkeypatch, tmp_path):
    """
    Verify that calling init_db() a second time updates stored password hashes
    when they no longer match the currently configured password.

    bcrypt always generates a unique salt per hash, so we cannot assert that all
    hash strings are equal — we assert instead that:
      - every stored hash verifies against the current configured password, and
      - none of the stored hashes still verify against the stale password.
    """
    db_path = _configure_temp_wharton(monkeypatch, tmp_path)

    # Fix the 'current' password so it is deterministic in the test environment
    current_password = "test-current-pass"
    monkeypatch.setattr(wharton_dash, "DEFAULT_PASSWORD", current_password)
    # Prevent init_db from trying to read st.secrets per-user passwords in production path
    monkeypatch.setattr(wharton_dash, "_is_development_mode", lambda: True)

    wharton_dash.init_db()

    old_password = "old-team-pass"
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        old_hash = bcrypt.hashpw(old_password.encode(), bcrypt.gensalt()).decode()
        connection.execute("UPDATE wharton_users SET password_hash = ?", (old_hash,))

    # Second call must detect hash mismatch and re-hash to the current password
    wharton_dash.init_db()

    with sqlite3.connect(db_path) as connection:
        hashes = [
            row[0]
            for row in connection.execute(
                "SELECT password_hash FROM wharton_users ORDER BY id"
            ).fetchall()
        ]

    assert len(hashes) > 0, "No users were seeded"

    # Every hash must validate against the current password
    for h in hashes:
        assert bcrypt.checkpw(
            current_password.encode(), h.encode()
        ), f"Hash {h!r} does not match current password"

    # None must still validate against the old stale password
    for h in hashes:
        assert not bcrypt.checkpw(
            old_password.encode(), h.encode()
        ), f"Hash {h!r} still matches old password — password was NOT re-synced"


def test_captains_have_matching_roles_and_matej_is_first_at_login(monkeypatch, tmp_path):
    _configure_temp_wharton(monkeypatch, tmp_path)
    wharton_dash.init_db()

    users = wharton_dash._fetch_users()

    assert [user["username"] for user in users[:2]] == ["Matěj", "Jakub"]
    assert {user["username"]: user["role"] for user in users[:2]} == {
        "Matěj": "Co-Captain",
        "Jakub": "Co-Captain",
    }
