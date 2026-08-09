from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import pytest

from src.operations.release_manifest import (
    DEPLOYABLE_DIRECTORIES,
    DEPLOYABLE_FILES,
    ReleaseManifestError,
    collect_deployable_files,
    create_release_manifest,
    load_release_manifest,
    verify_release_manifest,
    write_release_manifest,
)


COMMIT = "a" * 40
OTHER_COMMIT = "b" * 40
CREATED_AT = datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def release_root(tmp_path: Path) -> Path:
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


def _create(root: Path) -> dict[str, object]:
    return create_release_manifest(
        root,
        commit_sha=COMMIT,
        ref="main",
        working_tree_clean=True,
        created_at=CREATED_AT,
    )


def test_create_and_verify_release_manifest(release_root: Path) -> None:
    manifest = _create(release_root)

    result = verify_release_manifest(
        release_root,
        manifest,
        expected_commit=COMMIT,
    )

    assert result == {
        "valid": True,
        "commit_sha": COMMIT,
        "source_tree_sha256": manifest["source_tree_sha256"],
        "file_count": len(manifest["files"]),
    }
    assert manifest["created_at"] == "2026-08-09T12:00:00Z"
    assert ".streamlit/secrets.toml" not in manifest["files"]


def test_manifest_is_deterministic(release_root: Path) -> None:
    assert _create(release_root) == _create(release_root)


def test_tampered_file_is_rejected(release_root: Path) -> None:
    manifest = _create(release_root)
    (release_root / "src/runtime.txt").write_text("tampered\n", encoding="utf-8")

    result = verify_release_manifest(release_root, manifest)

    assert result == {"valid": False, "reason": "file_hash_mismatch"}


def test_unrecorded_runtime_file_is_rejected(release_root: Path) -> None:
    manifest = _create(release_root)
    (release_root / "ui/new_runtime.py").write_text("VALUE = 1\n", encoding="utf-8")

    result = verify_release_manifest(release_root, manifest)

    assert result == {"valid": False, "reason": "inventory_mismatch"}


def test_expected_commit_mismatch_is_rejected(release_root: Path) -> None:
    result = verify_release_manifest(
        release_root,
        _create(release_root),
        expected_commit=OTHER_COMMIT,
    )

    assert result == {"valid": False, "reason": "commit_mismatch"}


def test_unsafe_manifest_path_is_rejected(release_root: Path) -> None:
    manifest = deepcopy(_create(release_root))
    manifest["files"]["../secrets.toml"] = "0" * 64

    result = verify_release_manifest(release_root, manifest)

    assert result == {"valid": False, "reason": "unsafe_path"}


def test_tampered_tree_hash_is_rejected(release_root: Path) -> None:
    manifest = _create(release_root)
    manifest["source_tree_sha256"] = "0" * 64

    result = verify_release_manifest(release_root, manifest)

    assert result == {"valid": False, "reason": "source_tree_hash_mismatch"}


def test_dirty_working_tree_cannot_create_release(release_root: Path) -> None:
    with pytest.raises(ReleaseManifestError, match="working_tree_dirty"):
        create_release_manifest(
            release_root,
            commit_sha=COMMIT,
            ref="main",
            working_tree_clean=False,
        )


def test_manifest_round_trip(release_root: Path, tmp_path: Path) -> None:
    destination = tmp_path / "build/release-manifest.json"
    manifest = _create(release_root)

    write_release_manifest(destination, manifest)

    assert load_release_manifest(destination) == manifest
    assert json.loads(destination.read_text(encoding="utf-8")) == manifest


def test_non_object_manifest_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ReleaseManifestError, match="manifest_invalid"):
        load_release_manifest(path)


def test_real_inventory_includes_release_verifier_but_never_secrets() -> None:
    project_root = Path(__file__).resolve().parents[2]

    inventory = collect_deployable_files(project_root)

    assert "scripts/release_manifest.py" in inventory
    assert "src/operations/release_manifest.py" in inventory
    assert ".streamlit/secrets.toml" not in inventory
    assert all("__pycache__" not in path for path in inventory)
