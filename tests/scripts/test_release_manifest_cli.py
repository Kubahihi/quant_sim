from __future__ import annotations

import json
from pathlib import Path
import platform

from scripts import release_manifest as release_cli
from src.operations.release_manifest import DEPLOYABLE_DIRECTORIES, DEPLOYABLE_FILES


COMMIT = "c" * 40


def _release_root(tmp_path: Path) -> Path:
    for directory in DEPLOYABLE_DIRECTORIES:
        path = tmp_path / directory
        path.mkdir(parents=True)
        (path / "runtime.txt").write_text(f"{directory}\n", encoding="utf-8")
    for relative in DEPLOYABLE_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        value = platform.python_version() if relative == ".python-version" else relative
        path.write_text(value + "\n", encoding="utf-8")
    return tmp_path


def test_cli_creates_then_verifies_manifest(tmp_path, monkeypatch, capsys):
    project_root = _release_root(tmp_path)
    destination = project_root / "build/release-manifest.json"
    monkeypatch.setattr(release_cli, "PROJECT_ROOT", project_root)

    def git_output(*args: str) -> str:
        if args[0] == "status":
            return ""
        if "--abbrev-ref" in args:
            return "main"
        return COMMIT

    monkeypatch.setattr(release_cli, "_git_output", git_output)

    assert release_cli.main(["create", "--output", str(destination)]) == 0
    created = json.loads(capsys.readouterr().out)
    assert created["created"] is True
    assert created["commit_sha"] == COMMIT

    assert release_cli.main([
        "verify",
        "--manifest",
        str(destination),
        "--expected-commit",
        COMMIT,
    ]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["valid"] is True
    assert verified["commit_sha"] == COMMIT


def test_cli_refuses_dirty_checkout(tmp_path, monkeypatch, capsys):
    project_root = _release_root(tmp_path)
    monkeypatch.setattr(release_cli, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(release_cli, "_git_output", lambda *args: "dirty")

    exit_code = release_cli.main([
        "create",
        "--output",
        str(project_root / "build/release-manifest.json"),
        "--commit",
        COMMIT,
        "--ref",
        "main",
    ])

    assert exit_code == 1
    assert json.loads(capsys.readouterr().out) == {
        "valid": False,
        "reason": "release_manifest_failed",
    }
