from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.validate_sbom import SBOMValidationError, validate_sbom


def _write_requirements(tmp_path: Path) -> Path:
    path = tmp_path / "requirements.txt"
    path.write_text("Alpha_Package==1.2.3\nbeta==4.5.6\n", encoding="utf-8")
    return path


def _sbom() -> dict:
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.4",
        "version": 1,
        "components": [
            {
                "type": "library",
                "name": "alpha-package",
                "version": "1.2.3",
                "bom-ref": "alpha-package==1.2.3",
            },
            {
                "type": "library",
                "name": "beta",
                "version": "4.5.6",
                "bom-ref": "beta==4.5.6",
            },
        ],
        "vulnerabilities": [],
    }


def _write_sbom(tmp_path: Path, value: object) -> Path:
    path = tmp_path / "sbom.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_validate_sbom_accepts_exact_lock_inventory(tmp_path: Path) -> None:
    result = validate_sbom(_write_sbom(tmp_path, _sbom()), _write_requirements(tmp_path))

    assert result == {
        "format": "CycloneDX",
        "spec_version": "1.4",
        "components": 2,
        "vulnerabilities": 0,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["components"].pop(), "missing packages"),
        (
            lambda value: value["components"].append(
                {"type": "library", "name": "extra", "version": "1.0"}
            ),
            "unexpected packages",
        ),
        (
            lambda value: value["components"][0].update(version="9.9.9"),
            "versions differ",
        ),
        (
            lambda value: value.update(
                vulnerabilities=[{"id": "CVE-2026-0001"}]
            ),
            "known vulnerabilities",
        ),
    ],
)
def test_validate_sbom_rejects_inventory_or_security_drift(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    sbom = deepcopy(_sbom())
    mutation(sbom)

    with pytest.raises(SBOMValidationError, match=message):
        validate_sbom(_write_sbom(tmp_path, sbom), _write_requirements(tmp_path))


def test_validate_sbom_rejects_non_cyclonedx_document(tmp_path: Path) -> None:
    with pytest.raises(SBOMValidationError, match="not CycloneDX"):
        validate_sbom(
            _write_sbom(tmp_path, {"bomFormat": "SPDX", "components": []}),
            _write_requirements(tmp_path),
        )
