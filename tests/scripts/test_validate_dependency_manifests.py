"""Tests for dependency deployment parity validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.validate_dependency_manifests import (
    ManifestValidationError,
    _validate_cutoff,
    validate_manifests,
)


def _write(path: Path, contents: str) -> Path:
    path.write_text(contents.strip() + "\n", encoding="utf-8")
    return path


def test_validate_manifests_accepts_exact_prod_dev_parity(tmp_path):
    counts = validate_manifests(
        _write(tmp_path / "requirements.in", "Flask>=3"),
        _write(tmp_path / "requirements.txt", "Flask==3.1.3\nWerkzeug==3.1.8"),
        _write(
            tmp_path / "requirements-dev.lock",
            "Flask==3.1.3\nWerkzeug==3.1.8\npytest==9.1.1",
        ),
    )

    assert counts == {"direct": 1, "production": 2, "development": 3}


def test_validate_manifests_rejects_unpinned_production_requirement(tmp_path):
    with pytest.raises(ManifestValidationError, match="non-exact.*Flask"):
        validate_manifests(
            _write(tmp_path / "requirements.in", "Flask>=3"),
            _write(tmp_path / "requirements.txt", "Flask>=3"),
            _write(tmp_path / "requirements-dev.lock", "Flask==3.1.3"),
        )


def test_validate_manifests_rejects_missing_direct_requirement(tmp_path):
    with pytest.raises(ManifestValidationError, match="missing direct.*flask"):
        validate_manifests(
            _write(tmp_path / "requirements.in", "Flask>=3"),
            _write(tmp_path / "requirements.txt", "Werkzeug==3.1.8"),
            _write(tmp_path / "requirements-dev.lock", "Werkzeug==3.1.8"),
        )


def test_validate_manifests_rejects_prod_dev_version_drift(tmp_path):
    with pytest.raises(ManifestValidationError, match="disagree.*flask"):
        validate_manifests(
            _write(tmp_path / "requirements.in", "Flask>=3"),
            _write(tmp_path / "requirements.txt", "Flask==3.1.3"),
            _write(tmp_path / "requirements-dev.lock", "Flask==3.1.4"),
        )


def test_validate_cutoff_accepts_past_utc_timestamp(tmp_path):
    cutoff = _write(tmp_path / ".dependency-cutoff", "2025-01-02T03:04:05Z")

    assert _validate_cutoff(cutoff) == "2025-01-02T03:04:05Z"


@pytest.mark.parametrize(
    "value",
    ["not-a-date", "2025-01-02T03:04:05", "2999-01-02T03:04:05Z"],
)
def test_validate_cutoff_rejects_invalid_or_future_value(tmp_path, value):
    cutoff = _write(tmp_path / ".dependency-cutoff", value)

    with pytest.raises(ManifestValidationError):
        _validate_cutoff(cutoff)
