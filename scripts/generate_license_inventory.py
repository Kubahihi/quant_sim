"""Generate a complete production license inventory from installed metadata."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import sys
from typing import Any

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LicenseInventoryError(RuntimeError):
    """Raised when installed package/license evidence is incomplete."""


def _license_from_file(distribution: metadata.Distribution) -> str | None:
    license_paths = [
        path
        for path in distribution.files or []
        if Path(str(path)).name.lower().startswith(("license", "copying"))
    ]
    text = "\n".join(
        distribution.locate_file(path).read_text(encoding="utf-8", errors="ignore")
        for path in license_paths
    )
    if "Permission is hereby granted, free of charge" in text:
        return "MIT"
    if "Apache License" in text and "Version 2.0" in text:
        return "Apache-2.0"
    if "Redistribution and use in source and binary forms" in text:
        return "BSD"
    return None


def _license_evidence(distribution: metadata.Distribution) -> tuple[str, str]:
    package_metadata = distribution.metadata
    expression = str(package_metadata.get("License-Expression") or "").strip()
    if expression:
        return expression, "license-expression"
    declared = str(package_metadata.get("License") or "").strip()
    if declared and len(declared) <= 160 and declared.upper() not in {"UNKNOWN", "N/A"}:
        return declared, "license-field"
    classifiers = [
        value.removeprefix("License :: ").strip()
        for value in package_metadata.get_all("Classifier", [])
        if value.startswith("License :: ")
    ]
    if classifiers:
        return " | ".join(classifiers), "classifier"
    detected = _license_from_file(distribution)
    if detected:
        return detected, "license-file"
    raise LicenseInventoryError("Package license evidence is missing.")


def generate_license_inventory(requirements_path: Path) -> dict[str, Any]:
    """Require every exact production pin to match an installed licensed package."""
    packages: list[dict[str, str]] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        requirements_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        specifiers = list(requirement.specifier)
        if len(specifiers) != 1 or specifiers[0].operator != "==":
            raise LicenseInventoryError(
                f"{requirements_path.name}:{line_number} is not exactly pinned."
            )
        name = canonicalize_name(requirement.name)
        if name in seen:
            raise LicenseInventoryError(f"Duplicate production package: {name}.")
        seen.add(name)
        try:
            distribution = metadata.distribution(requirement.name)
        except metadata.PackageNotFoundError as error:
            raise LicenseInventoryError(f"Production package is not installed: {name}.") from error
        installed_version = distribution.version
        if installed_version != specifiers[0].version:
            raise LicenseInventoryError(f"Installed production package version differs: {name}.")
        license_value, evidence = _license_evidence(distribution)
        packages.append(
            {
                "license": license_value,
                "license_evidence": evidence,
                "name": name,
                "version": installed_version,
            }
        )
    if not packages:
        raise LicenseInventoryError("Production license inventory is empty.")
    return {
        "application": "quant-sim",
        "package_count": len(packages),
        "packages": sorted(packages, key=lambda item: item["name"]),
        "schema_version": 1,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requirements",
        type=Path,
        default=PROJECT_ROOT / "requirements.txt",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        inventory = generate_license_inventory(args.requirements)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(inventory, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (LicenseInventoryError, OSError, ValueError) as error:
        print(f"License inventory failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"packages": inventory["package_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
