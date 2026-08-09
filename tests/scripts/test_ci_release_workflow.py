from __future__ import annotations

from pathlib import Path
import re

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _workflow() -> dict:
    value = yaml.safe_load(
        (PROJECT_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    )
    assert isinstance(value, dict)
    return value


def test_ci_retains_attested_release_manifest() -> None:
    workflow = _workflow()
    assert workflow["permissions"] == {"contents": "read"}
    quality = workflow["jobs"]["quality"]
    assert quality["needs"] == "security"
    assert quality["permissions"] == {
        "attestations": "write",
        "contents": "read",
        "id-token": "write",
    }
    steps = quality["steps"]
    by_name = {step["name"]: step for step in steps}

    build = by_name["Build and verify release manifest"]
    assert "release_manifest.py create" in build["run"]
    assert "release_manifest.py verify" in build["run"]
    assert '--expected-commit "$GITHUB_SHA"' in build["run"]

    attestation = by_name["Attest release manifest provenance"]
    assert attestation["uses"] == "actions/attest@v4"
    assert attestation["with"]["subject-path"] == "build/release-manifest.json"
    assert attestation["if"] == "github.event_name != 'pull_request'"

    upload = by_name["Retain verified release manifest"]
    assert upload["uses"] == "actions/upload-artifact@v7"
    assert upload["if"] == "github.event_name != 'pull_request'"
    assert upload["with"]["retention-days"] == 90
    assert upload["with"]["if-no-files-found"] == "error"

    names = [step["name"] for step in steps]
    assert names.index("Run test suite with coverage gate") < names.index(
        "Build and verify release manifest"
    )
    assert names.index("Build and verify release manifest") < names.index(
        "Attest release manifest provenance"
    )
    assert names.index("Attest release manifest provenance") < names.index(
        "Retain verified release manifest"
    )


def test_ci_security_gate_has_least_privilege_and_blocks_quality() -> None:
    workflow = _workflow()
    security = workflow["jobs"]["security"]
    assert security["permissions"] == {"contents": "read"}
    by_name = {step["name"]: step for step in security["steps"]}

    checkout = by_name["Check out complete repository history"]
    assert checkout["uses"] == "actions/checkout@v6"
    assert checkout["with"]["fetch-depth"] == 0

    secrets = by_name["Scan repository history for secrets"]
    assert secrets["uses"] == "gitleaks/gitleaks-action@v2"
    assert secrets["env"]["GITLEAKS_ENABLE_COMMENTS"] == "false"
    assert secrets["env"]["GITLEAKS_ENABLE_UPLOAD_ARTIFACT"] == "false"

    review = by_name["Review dependency changes"]
    assert review["uses"] == "actions/dependency-review-action@v5"
    assert review["if"] == "github.event_name == 'pull_request'"
    assert review["with"]["fail-on-severity"] == "low"

    quality_steps = {
        step["name"]: step for step in workflow["jobs"]["quality"]["steps"]
    }
    audit = quality_steps["Audit installed dependencies for known vulnerabilities"]
    assert "--local --strict" in audit["run"]
    lock_validation = quality_steps["Validate lock files"]["run"]
    assert lock_validation.count('--exclude-newer "$(cat .dependency-cutoff)"') == 2


def test_secret_allowlist_contains_only_specific_fingerprints() -> None:
    entries = [
        line.strip()
        for line in (PROJECT_ROOT / ".gitleaksignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    fingerprint = re.compile(r"[0-9a-f]{40}:[^:*?]+:[^:*?]+:\d+")

    assert entries
    assert all(fingerprint.fullmatch(entry) for entry in entries)
