"""Create and verify a deterministic manifest for a deployable source tree."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import hmac
import json
from pathlib import Path, PurePosixPath
import platform
import re
from typing import Any


SCHEMA_VERSION = 1
APPLICATION_NAME = "quant-sim"
DEPLOYABLE_DIRECTORIES = ("src", "ui", "scripts", "config")
DEPLOYABLE_FILES = (
    ".dependency-cutoff",
    ".python-version",
    ".streamlit/config.toml",
    ".streamlit/packages.txt",
    "api_server.py",
    "main.py",
    "pyproject.toml",
    "requirements-dev.lock",
    "requirements-dev.txt",
    "requirements.in",
    "requirements.txt",
    "setup.py",
)
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40,64}", re.IGNORECASE)
_PYTHON_PATTERN = re.compile(r"\d+\.\d+\.\d+")
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}", re.IGNORECASE)


class ReleaseManifestError(ValueError):
    """Raised when a release manifest cannot be created safely."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_digest(files: Mapping[str, str]) -> str:
    canonical = json.dumps(
        dict(files), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _safe_relative_path(value: str) -> PurePosixPath | None:
    if not value or "\\" in value:
        return None
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or ".." in candidate.parts or "." in candidate.parts:
        return None
    return candidate


def collect_deployable_files(project_root: str | Path) -> dict[str, Path]:
    """Return the exact deployable inventory, rejecting links and missing inputs."""
    root = Path(project_root).resolve(strict=True)
    inventory: dict[str, Path] = {}

    def add_file(path: Path) -> None:
        if path.is_symlink():
            raise ReleaseManifestError("deployable_symlink")
        if not path.is_file():
            raise ReleaseManifestError("deployable_file_missing")
        relative = path.relative_to(root).as_posix()
        inventory[relative] = path

    for relative in DEPLOYABLE_FILES:
        add_file(root / relative)

    for relative in DEPLOYABLE_DIRECTORIES:
        directory = root / relative
        if directory.is_symlink():
            raise ReleaseManifestError("deployable_symlink")
        if not directory.is_dir():
            raise ReleaseManifestError("deployable_directory_missing")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ReleaseManifestError("deployable_symlink")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                add_file(path)

    return dict(sorted(inventory.items()))


def create_release_manifest(
    project_root: str | Path,
    *,
    commit_sha: str,
    ref: str,
    working_tree_clean: bool,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Build a release manifest for the current deployable source tree."""
    if not _COMMIT_PATTERN.fullmatch(commit_sha):
        raise ReleaseManifestError("commit_invalid")
    if not working_tree_clean:
        raise ReleaseManifestError("working_tree_dirty")
    if not isinstance(ref, str) or not ref.strip():
        raise ReleaseManifestError("ref_invalid")

    root = Path(project_root).resolve(strict=True)
    python_version = (root / ".python-version").read_text(encoding="utf-8").strip()
    if not _PYTHON_PATTERN.fullmatch(python_version):
        raise ReleaseManifestError("python_version_invalid")
    builder_python = platform.python_version()
    if builder_python != python_version:
        raise ReleaseManifestError("builder_python_mismatch")

    inventory = collect_deployable_files(root)
    hashes = {relative: _sha256_file(path) for relative, path in inventory.items()}
    timestamp = created_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ReleaseManifestError("created_at_not_utc")
    timestamp = timestamp.astimezone(timezone.utc)

    return {
        "application": APPLICATION_NAME,
        "builder_python_version": builder_python,
        "commit_sha": commit_sha.lower(),
        "created_at": timestamp.isoformat().replace("+00:00", "Z"),
        "files": hashes,
        "python_version": python_version,
        "ref": ref.strip(),
        "schema_version": SCHEMA_VERSION,
        "source_tree_sha256": _tree_digest(hashes),
        "working_tree_clean": True,
    }


def _invalid(reason: str) -> dict[str, Any]:
    return {"valid": False, "reason": reason}


def verify_release_manifest(
    project_root: str | Path,
    manifest: Mapping[str, Any],
    *,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    """Verify manifest structure, inventory, commit identity, and every file hash."""
    try:
        if manifest.get("schema_version") != SCHEMA_VERSION:
            return _invalid("schema_version_unsupported")
        if manifest.get("application") != APPLICATION_NAME:
            return _invalid("application_mismatch")
        if manifest.get("working_tree_clean") is not True:
            return _invalid("dirty_manifest")

        commit_sha = manifest.get("commit_sha")
        if not isinstance(commit_sha, str) or not _COMMIT_PATTERN.fullmatch(commit_sha):
            return _invalid("commit_invalid")
        if expected_commit is not None:
            if not _COMMIT_PATTERN.fullmatch(expected_commit):
                return _invalid("expected_commit_invalid")
            if not hmac.compare_digest(commit_sha.lower(), expected_commit.lower()):
                return _invalid("commit_mismatch")

        ref = manifest.get("ref")
        if not isinstance(ref, str) or not ref.strip():
            return _invalid("ref_invalid")
        created_at = manifest.get("created_at")
        if not isinstance(created_at, str) or not created_at.endswith("Z"):
            return _invalid("created_at_invalid")
        datetime.fromisoformat(created_at.removesuffix("Z") + "+00:00")

        root = Path(project_root).resolve(strict=True)
        python_version = (root / ".python-version").read_text(encoding="utf-8").strip()
        if manifest.get("python_version") != python_version:
            return _invalid("python_version_mismatch")
        if manifest.get("builder_python_version") != python_version:
            return _invalid("builder_python_mismatch")

        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            return _invalid("files_invalid")
        for relative, expected_hash in files.items():
            if not isinstance(relative, str) or _safe_relative_path(relative) is None:
                return _invalid("unsafe_path")
            if not isinstance(expected_hash, str) or not _HASH_PATTERN.fullmatch(expected_hash):
                return _invalid("file_hash_invalid")

        inventory = collect_deployable_files(root)
        if set(files) != set(inventory):
            return _invalid("inventory_mismatch")
        for relative, path in inventory.items():
            if not hmac.compare_digest(_sha256_file(path), files[relative].lower()):
                return _invalid("file_hash_mismatch")

        tree_hash = manifest.get("source_tree_sha256")
        if not isinstance(tree_hash, str) or not _HASH_PATTERN.fullmatch(tree_hash):
            return _invalid("source_tree_hash_invalid")
        if not hmac.compare_digest(_tree_digest(files), tree_hash.lower()):
            return _invalid("source_tree_hash_mismatch")
    except (OSError, TypeError, ValueError, ReleaseManifestError):
        return _invalid("manifest_invalid")

    return {
        "valid": True,
        "commit_sha": commit_sha.lower(),
        "source_tree_sha256": tree_hash.lower(),
        "file_count": len(files),
    }


def load_release_manifest(path: str | Path) -> dict[str, Any]:
    """Load a manifest without leaking parsing details to callers."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseManifestError("manifest_unreadable") from exc
    if not isinstance(value, dict):
        raise ReleaseManifestError("manifest_invalid")
    return value


def write_release_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    """Write a manifest atomically enough for CI artifact collection."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
