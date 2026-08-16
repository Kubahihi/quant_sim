from __future__ import annotations

import importlib
import os
import sqlite3
from pathlib import Path
import subprocess
import sys

import bcrypt
import math
import pytest
from streamlit.testing.v1 import AppTest
from types import SimpleNamespace

from src.auth.wharton_credentials import WhartonCredentialConfigError
from ui.pages import wharton_dash


def test_wharton_build_fingerprint_is_visible_and_auditable(monkeypatch):
    captions = []
    monkeypatch.setattr(wharton_dash.st, "caption", captions.append)

    wharton_dash._render_build_fingerprint()

    assert captions == [wharton_dash._build_fingerprint_label()]
    assert "QuantSim v0.2.0" in captions[0]
    assert "build 2026-08-16" in captions[0]
    assert wharton_dash.WHARTON_BUILD_IDENTITY.commit in captions[0]
    assert wharton_dash.WHARTON_BUILD_IDENTITY.branch in captions[0]


def test_wharton_page_refreshes_stale_credential_module(monkeypatch):
    credential_module = importlib.import_module("src.auth.wharton_credentials")
    existing_config_error = credential_module.WhartonCredentialConfigError
    monkeypatch.delattr(credential_module, "resolve_wharton_credentials")

    reloaded_page = importlib.reload(wharton_dash)

    assert reloaded_page.resolve_wharton_credentials is (
        credential_module.resolve_wharton_credentials
    )
    assert reloaded_page.WhartonCredentialConfigError is existing_config_error


def test_wharton_page_can_start_directly_outside_repository(tmp_path):
    page_path = Path(wharton_dash.__file__).resolve()

    completed = subprocess.run(
        [sys.executable, str(page_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_wharton_page_replaces_preloaded_non_local_src_package(tmp_path):
    page_path = Path(wharton_dash.__file__).resolve()
    project_root = page_path.parents[2]
    fake_src = tmp_path / "src"
    fake_src.mkdir()
    (fake_src / "__init__.py").write_text(
        '"""Unrelated package that happens to be named src."""\n',
        encoding="utf-8",
    )
    environment = os.environ.copy()
    python_path = [str(tmp_path), str(project_root)]
    if existing_python_path := environment.get("PYTHONPATH"):
        python_path.append(existing_python_path)
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    command = (
        "import runpy, src; "
        f"assert src.__file__.startswith({str(fake_src)!r}); "
        f"runpy.run_path({str(page_path)!r}, run_name='__main__')"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_wharton_page_refreshes_stale_local_src_metadata(tmp_path):
    page_path = Path(wharton_dash.__file__).resolve()
    project_root = page_path.parents[2]
    environment = os.environ.copy()
    python_path = [str(project_root)]
    if existing_python_path := environment.get("PYTHONPATH"):
        python_path.append(existing_python_path)
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    command = (
        "import runpy, src; "
        "del src.__build_date__; "
        f"runpy.run_path({str(page_path)!r}, run_name='__main__')"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_public_login_renders_build_fingerprint_before_authentication():
    page_path = Path(wharton_dash.__file__).resolve()

    app = AppTest.from_file(str(page_path)).run(timeout=45)

    assert not app.exception
    captions = [str(item.value) for item in app.caption]
    assert any("QuantSim v0.2.0" in item for item in captions)
    assert any("build 2026-08-16" in item for item in captions)
    assert any(wharton_dash.WHARTON_BUILD_IDENTITY.commit in item for item in captions)


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


def test_anonymous_wharton_login_does_not_initialize_database(monkeypatch):
    rendered = []

    monkeypatch.setattr(wharton_dash, "_inject_cockpit_styles", lambda: None)
    monkeypatch.setattr(wharton_dash, "_get_current_profile", lambda: None)
    monkeypatch.setattr(wharton_dash, "_render_login", lambda: rendered.append("login"))
    monkeypatch.setattr(
        wharton_dash,
        "init_db",
        lambda: pytest.fail("anonymous Wharton route initialized the database"),
    )

    wharton_dash.render_wharton_cockpit()

    assert rendered == ["login"]


def test_login_reports_invalid_team_credential_configuration(monkeypatch):
    errors = []

    def fail_initialization():
        raise WhartonCredentialConfigError("Wharton credentials are incomplete.")

    monkeypatch.setattr(wharton_dash, "init_db", fail_initialization)
    monkeypatch.setattr(wharton_dash.st, "error", errors.append)

    assert wharton_dash._initialize_database_for_login() is False
    assert len(errors) == 1
    assert "Streamlit Cloud secrets" in errors[0]


def test_local_shared_password_overrides_stale_per_user_secrets():
    credentials = wharton_dash.resolve_wharton_credentials(
        {
            "WHARTON_PASSWORD": "local-fallback-pass",
            "wharton_users": {
                "Jakub": "jakub-local-pass",
                "Martin": "martin-local-pass",
            },
        },
        production=False,
    )

    assert set(credentials.values()) == {"local-fallback-pass"}


def test_local_credentials_support_legacy_nested_shared_secret():
    credentials = wharton_dash.resolve_wharton_credentials(
        {"wharton_users": {"WHARTON_PASSWORD": "nested-local-pass"}},
        production=False,
    )

    assert set(credentials.values()) == {"nested-local-pass"}


def test_production_shared_secret_applies_to_every_team_member():
    credentials = wharton_dash.resolve_wharton_credentials(
        {"WHARTON_SHARED_PASSWORD": "wharton123"},
        production=True,
    )

    assert credentials == dict.fromkeys(
        wharton_dash.REQUIRED_WHARTON_USERS,
        "wharton123",
    )


def test_production_supports_existing_top_level_shared_password():
    credentials = wharton_dash.resolve_wharton_credentials(
        {
            "WHARTON_PASSWORD": "wharton123",
            "wharton_users": {
                "Jakub": "stale-jakub-password",
                "Martin": "stale-martin-password",
            },
        },
        production=True,
    )

    assert credentials == dict.fromkeys(
        wharton_dash.REQUIRED_WHARTON_USERS,
        "wharton123",
    )


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
    monkeypatch.setattr(
        wharton_dash,
        "resolve_wharton_credentials",
        lambda *args, **kwargs: dict.fromkeys(
            wharton_dash.REQUIRED_WHARTON_USERS,
            password,
        ),
    )
    return db_path


def test_init_db_materializes_executemany_parameters_for_libsql(monkeypatch, tmp_path):
    db_path = _configure_temp_wharton(monkeypatch, tmp_path)

    class StrictBatchConnection:
        """Model libSQL's requirement that executemany receives a sequence."""

        def __init__(self, path: Path):
            self._connection = sqlite3.connect(path)
            self._connection.row_factory = sqlite3.Row

        def __getattr__(self, name):
            return getattr(self._connection, name)

        def __enter__(self):
            self._connection.__enter__()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return self._connection.__exit__(exc_type, exc_value, traceback)

        def executemany(self, sql, parameters):
            if not isinstance(parameters, (list, tuple)):
                raise TypeError("parameters must be a sequence")
            return self._connection.executemany(sql, parameters)

    monkeypatch.setattr(
        wharton_dash,
        "get_connection",
        lambda: StrictBatchConnection(db_path),
    )

    wharton_dash.init_db()

    with sqlite3.connect(db_path) as connection:
        usernames = {
            row[0]
            for row in connection.execute("SELECT username FROM wharton_users")
        }
    assert usernames == {user["username"] for user in wharton_dash.DEFAULT_USERS}


def test_init_db_removes_departed_team_member(monkeypatch, tmp_path):
    db_path = _configure_temp_wharton(monkeypatch, tmp_path)
    wharton_dash.init_db()

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "INSERT INTO wharton_users (username, password_hash, role, primary_module) "
            "VALUES (?, ?, ?, ?)",
            ("Alexandra", "retired-hash", "Team Member", "Teamspace"),
        )

    wharton_dash.init_db()

    with sqlite3.connect(db_path) as connection:
        departed_user = connection.execute(
            "SELECT 1 FROM wharton_users WHERE username = ?",
            ("Alexandra",),
        ).fetchone()
    assert departed_user is None


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
    monkeypatch.setattr(
        wharton_dash,
        "resolve_wharton_credentials",
        lambda *args, **kwargs: dict.fromkeys(
            wharton_dash.REQUIRED_WHARTON_USERS,
            current_password,
        ),
    )
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


def test_production_initialization_accepts_shared_legacy_password(monkeypatch, tmp_path):
    db_path = tmp_path / "data" / "wharton.db"
    upload_dir = tmp_path / "data" / "wharton_uploads"
    monkeypatch.setenv("QUANT_SIM_ENV", "production")
    monkeypatch.setattr(wharton_dash, "DB_PATH", db_path)
    monkeypatch.setattr(wharton_dash, "UPLOAD_DIR", upload_dir)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    def get_test_connection():
        connection = sqlite3.connect(db_path)
        connection.row_factory = sqlite3.Row
        return connection

    monkeypatch.setattr(wharton_dash, "get_connection", get_test_connection)
    monkeypatch.setattr(
        wharton_dash.st,
        "secrets",
        {"WHARTON_PASSWORD": "wharton123"},
    )

    wharton_dash._initialize_database()

    for username in wharton_dash.REQUIRED_WHARTON_USERS:
        profile = wharton_dash.authenticate_user(username, "wharton123")
        assert profile is not None
        assert profile["username"] == username


def test_team_roster_is_alphabetical_and_captains_keep_their_roles(monkeypatch, tmp_path):
    _configure_temp_wharton(monkeypatch, tmp_path)
    wharton_dash.init_db()

    expected_names = ["Jakub", "Lukáš", "Martin", "Matěj"]
    assert [user["username"] for user in wharton_dash.DEFAULT_USERS] == expected_names

    users = wharton_dash._fetch_users()

    assert [user["username"] for user in users] == expected_names
    roles = {user["username"]: user["role"] for user in users}
    assert roles["Matěj"] == "Co-Captain"
    assert roles["Jakub"] == "Co-Captain"
