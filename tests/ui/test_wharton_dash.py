from __future__ import annotations

import sqlite3
from pathlib import Path
import bcrypt

from ui.pages import wharton_dash


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

    # bcrypt intentionally gives each user a different salted hash. Verify the
    # configured password against every hash instead of comparing hash strings.
    assert len(password_hashes) == len(wharton_dash.DEFAULT_USERS)
    assert all(
        bcrypt.checkpw("new-team-pass".encode(), password_hash.encode())
        for password_hash in password_hashes
    )
    # None must still validate against the old stale password
    for h in hashes:
        assert not bcrypt.checkpw(
            old_password.encode(), h.encode()
        ), f"Hash {h!r} still matches old password — password was NOT re-synced"
