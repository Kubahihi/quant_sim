from __future__ import annotations

from email.message import Message
from pathlib import Path

import pytest

from scripts import generate_license_inventory as inventory_module


class _Distribution:
    def __init__(self, version: str, license_expression: str | None) -> None:
        self.version = version
        self.files = []
        self.metadata = Message()
        if license_expression is not None:
            self.metadata["License-Expression"] = license_expression


def test_inventory_requires_exact_installed_versions_and_license_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("Alpha_Package==1.2.3\nbeta==4.5.6\n", encoding="utf-8")
    distributions = {
        "Alpha_Package": _Distribution("1.2.3", "MIT"),
        "beta": _Distribution("4.5.6", "Apache-2.0"),
    }
    monkeypatch.setattr(
        inventory_module.metadata,
        "distribution",
        lambda name: distributions[name],
    )

    result = inventory_module.generate_license_inventory(requirements)

    assert result["package_count"] == 2
    assert result["packages"] == [
        {
            "license": "MIT",
            "license_evidence": "license-expression",
            "name": "alpha-package",
            "version": "1.2.3",
        },
        {
            "license": "Apache-2.0",
            "license_evidence": "license-expression",
            "name": "beta",
            "version": "4.5.6",
        },
    ]


def test_inventory_rejects_installed_version_drift(tmp_path: Path, monkeypatch) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("alpha==1.2.3\n", encoding="utf-8")
    monkeypatch.setattr(
        inventory_module.metadata,
        "distribution",
        lambda _name: _Distribution("9.9.9", "MIT"),
    )

    with pytest.raises(inventory_module.LicenseInventoryError, match="version differs"):
        inventory_module.generate_license_inventory(requirements)


def test_inventory_rejects_missing_license_evidence(tmp_path: Path, monkeypatch) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("alpha==1.2.3\n", encoding="utf-8")
    monkeypatch.setattr(
        inventory_module.metadata,
        "distribution",
        lambda _name: _Distribution("1.2.3", None),
    )

    with pytest.raises(inventory_module.LicenseInventoryError, match="license evidence"):
        inventory_module.generate_license_inventory(requirements)
