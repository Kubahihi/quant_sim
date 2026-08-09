"""Validate that a CycloneDX SBOM exactly represents the production lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SBOMValidationError(RuntimeError):
    """Raised when an SBOM is malformed or differs from the production lock."""


def _locked_requirements(path: Path) -> dict[str, str]:
    locked: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            requirement = Requirement(line)
        except InvalidRequirement as error:
            raise SBOMValidationError(
                f"{path.name}:{line_number} is not a valid locked requirement."
            ) from error
        specifiers = list(requirement.specifier)
        if len(specifiers) != 1 or specifiers[0].operator != "==":
            raise SBOMValidationError(
                f"{path.name}:{line_number} is not exactly pinned."
            )
        name = canonicalize_name(requirement.name)
        if name in locked:
            raise SBOMValidationError(f"Duplicate locked package: {name}.")
        locked[name] = specifiers[0].version
    if not locked:
        raise SBOMValidationError("Production lock is empty.")
    return locked


def _sbom_components(sbom: dict[str, Any]) -> dict[str, str]:
    if sbom.get("bomFormat") != "CycloneDX":
        raise SBOMValidationError("SBOM format is not CycloneDX.")
    spec_version = sbom.get("specVersion")
    if not isinstance(spec_version, str) or not spec_version:
        raise SBOMValidationError("CycloneDX specVersion is missing.")
    if sbom.get("version") != 1:
        raise SBOMValidationError("CycloneDX document version must be 1.")
    if sbom.get("vulnerabilities") not in (None, []):
        raise SBOMValidationError("SBOM contains known vulnerabilities.")

    raw_components = sbom.get("components")
    if not isinstance(raw_components, list) or not raw_components:
        raise SBOMValidationError("SBOM component inventory is empty.")
    components: dict[str, str] = {}
    for component in raw_components:
        if not isinstance(component, dict) or component.get("type") != "library":
            raise SBOMValidationError("SBOM contains an invalid component.")
        raw_name = component.get("name")
        version = component.get("version")
        if not isinstance(raw_name, str) or not isinstance(version, str):
            raise SBOMValidationError("SBOM component identity is incomplete.")
        name = canonicalize_name(raw_name)
        if name in components:
            raise SBOMValidationError(f"Duplicate SBOM component: {name}.")
        components[name] = version
    return components


def validate_sbom(sbom_path: Path, requirements_path: Path) -> dict[str, int | str]:
    """Require exact package/version parity and a vulnerability-free SBOM."""
    try:
        sbom = json.loads(sbom_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SBOMValidationError("SBOM cannot be read as JSON.") from error
    if not isinstance(sbom, dict):
        raise SBOMValidationError("SBOM root must be an object.")

    locked = _locked_requirements(requirements_path)
    components = _sbom_components(sbom)
    missing = sorted(set(locked) - set(components))
    unexpected = sorted(set(components) - set(locked))
    drifted = sorted(
        name for name in set(locked) & set(components) if locked[name] != components[name]
    )
    if missing:
        raise SBOMValidationError("SBOM is missing packages: " + ", ".join(missing))
    if unexpected:
        raise SBOMValidationError(
            "SBOM contains unexpected packages: " + ", ".join(unexpected)
        )
    if drifted:
        raise SBOMValidationError(
            "SBOM package versions differ from the lock: " + ", ".join(drifted)
        )
    return {
        "format": "CycloneDX",
        "spec_version": str(sbom["specVersion"]),
        "components": len(components),
        "vulnerabilities": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sbom",
        type=Path,
        default=PROJECT_ROOT / "build/release-sbom.cdx.json",
    )
    parser.add_argument(
        "--requirements",
        type=Path,
        default=PROJECT_ROOT / "requirements.txt",
    )
    args = parser.parse_args(argv)
    try:
        result = validate_sbom(args.sbom, args.requirements)
    except SBOMValidationError as error:
        print(f"SBOM validation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
