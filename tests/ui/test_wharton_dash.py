from __future__ import annotations

import sqlite3
from pathlib import Path

from ui.pages import wharton_dash


def _configure_temp_wharton(monkeypatch, tmp_path: Path, password: str = "new-team-pass") -> Path:
    db_path = tmp_path / "data" / "wharton.db"
    upload_dir = tmp_path / "data" / "wharton_uploads"
    monkeypatch.setattr(wharton_dash, "DB_PATH", db_path)
    monkeypatch.setattr(wharton_dash, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(wharton_dash, "DEFAULT_PASSWORD", password)
    monkeypatch.setattr(wharton_dash, "_hash_password", lambda raw: f"hash:{raw}")
    monkeypatch.setattr(wharton_dash, "_password_matches", lambda raw, hashed: hashed == f"hash:{raw}")
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
    db_path = _configure_temp_wharton(monkeypatch, tmp_path)
    wharton_dash.init_db()

    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE users SET password_hash = ?", ("hash:old-team-pass",))

    wharton_dash.init_db()

    with sqlite3.connect(db_path) as connection:
        password_hashes = {
            row[0]
            for row in connection.execute("SELECT password_hash FROM users ORDER BY id").fetchall()
        }

    assert password_hashes == {"hash:new-team-pass"}
