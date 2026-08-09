"""Validate production/development dependency lock parity."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ManifestValidationError(RuntimeError):
    """Raised when dependency manifests are incomplete or non-deterministic."""


def _load_requirements(path: Path) -> dict[str, Requirement]:
    requirements: dict[str, Requirement] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            requirement = Requirement(line)
        except InvalidRequirement as error:
            raise ManifestValidationError(
                f"{path.name}:{line_number} is not a valid requirement: {line!r}"
            ) from error

        name = canonicalize_name(requirement.name)
        if name in requirements:
            raise ManifestValidationError(
                f"{path.name} contains duplicate requirement {requirement.name!r}."
            )
        requirements[name] = requirement
    return requirements


def _is_exact_pin(requirement: Requirement) -> bool:
    specifiers = list(requirement.specifier)
    return (
        requirement.url is None
        and len(specifiers) == 1
        and specifiers[0].operator == "=="
        and "*" not in specifiers[0].version
    )


def _locked_identity(requirement: Requirement) -> tuple[str, str | None]:
    marker = str(requirement.marker) if requirement.marker is not None else None
    return str(requirement.specifier), marker


def validate_manifests(
    direct_path: Path,
    production_path: Path,
    development_path: Path,
) -> dict[str, int]:
    """Validate direct dependency coverage and exact prod/dev lock parity."""
    direct = _load_requirements(direct_path)
    production = _load_requirements(production_path)
    development = _load_requirements(development_path)

    unpinned = sorted(
        requirement.name
        for requirement in production.values()
        if not _is_exact_pin(requirement)
    )
    if unpinned:
        raise ManifestValidationError(
            "Production manifest contains non-exact requirements: " + ", ".join(unpinned)
        )

    missing_direct = sorted(set(direct) - set(production))
    if missing_direct:
        raise ManifestValidationError(
            "Production manifest is missing direct requirements: "
            + ", ".join(missing_direct)
        )

    missing_from_development = sorted(set(production) - set(development))
    if missing_from_development:
        raise ManifestValidationError(
            "Development lock is missing production requirements: "
            + ", ".join(missing_from_development)
        )

    drifted = sorted(
        name
        for name, requirement in production.items()
        if _locked_identity(requirement) != _locked_identity(development[name])
    )
    if drifted:
        raise ManifestValidationError(
            "Production and development locks disagree for: " + ", ".join(drifted)
        )

    return {
        "direct": len(direct),
        "production": len(production),
        "development": len(development),
    }


def _validate_runtime(version_file: Path) -> str:
    expected = version_file.read_text(encoding="utf-8").strip()
    actual = ".".join(str(part) for part in sys.version_info[:3])
    if actual != expected:
        raise ManifestValidationError(
            f"Runtime mismatch: expected Python {expected}, running Python {actual}."
        )
    return actual


def _validate_cutoff(cutoff_file: Path) -> str:
    cutoff = cutoff_file.read_text(encoding="utf-8").strip()
    try:
        parsed = datetime.fromisoformat(cutoff.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise ManifestValidationError(
            ".dependency-cutoff must be an ISO-8601 UTC timestamp."
        ) from error
    if not cutoff.endswith("Z") or parsed.tzinfo != timezone.utc:
        raise ManifestValidationError(
            ".dependency-cutoff must be an ISO-8601 UTC timestamp."
        )
    if parsed > datetime.now(timezone.utc):
        raise ManifestValidationError(".dependency-cutoff cannot be in the future.")
    return cutoff


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-runtime",
        action="store_true",
        help="Also require the interpreter to match .python-version exactly.",
    )
    args = parser.parse_args(argv)

    try:
        counts = validate_manifests(
            PROJECT_ROOT / "requirements.in",
            PROJECT_ROOT / "requirements.txt",
            PROJECT_ROOT / "requirements-dev.lock",
        )
        cutoff = _validate_cutoff(PROJECT_ROOT / ".dependency-cutoff")
        runtime = (
            _validate_runtime(PROJECT_ROOT / ".python-version")
            if args.check_runtime
            else None
        )
    except (ManifestValidationError, OSError) as error:
        print(f"Dependency manifest validation failed: {error}", file=sys.stderr)
        return 1

    summary = (
        f"Dependency manifests valid: {counts['direct']} direct, "
        f"{counts['production']} production, {counts['development']} development, "
        f"cutoff {cutoff}"
    )
    if runtime is not None:
        summary += f", Python {runtime}"
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
