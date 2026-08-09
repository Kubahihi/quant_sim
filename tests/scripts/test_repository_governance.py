from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_ci_exposes_stable_required_check_names() -> None:
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())

    assert workflow["jobs"]["security"]["name"] == "Security gate"
    assert workflow["jobs"]["quality"]["name"] == "Release quality gate"
    assert workflow["jobs"]["quality"]["needs"] == "security"


def test_codeowners_covers_release_sensitive_paths() -> None:
    codeowners = (ROOT / ".github" / "CODEOWNERS").read_text()

    for pattern in (
        "* @Kubahihi",
        "/.github/ @Kubahihi",
        "/.streamlit/ @Kubahihi",
        "/requirements*.lock @Kubahihi",
        "/src/api/ @Kubahihi",
        "/src/auth/ @Kubahihi",
        "/src/storage/ @Kubahihi",
    ):
        assert pattern in codeowners


def test_review_material_covers_security_and_rollback() -> None:
    template = (ROOT / ".github" / "pull_request_template.md").read_text().lower()
    ruleset = (ROOT / "docs" / "GITHUB_RULESET.md").read_text()

    for topic in ("credentials", "dependencies", "rollback", "sbom"):
        assert topic in template
    for required_check in ("CI / Security gate", "CI / Release quality gate"):
        assert required_check in ruleset
    assert "Code Owners" in ruleset
    assert "cannot prove" in ruleset
