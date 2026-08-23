"""Runtime environment resolution tests."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.utils import environment


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_environment_import_keeps_optional_utility_dependencies_lazy():
    probe = (
        "import json, sys; import src.utils.environment; "
        "print(json.dumps({name: name in sys.modules for name in ('yaml', 'loguru')}))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert json.loads(completed.stdout) == {"yaml": False, "loguru": False}


def test_utility_public_exports_still_resolve_lazily():
    import src.utils as utils

    assert utils.resolve_environment is environment.resolve_environment
    assert callable(utils.load_config)
    assert callable(utils.setup_logger)


def test_environment_variable_has_priority_over_streamlit_secret(monkeypatch):
    monkeypatch.setenv("QUANT_SIM_ENV", "test")

    assert environment.resolve_environment(
        streamlit_secrets={"QUANT_SIM_ENV": "production"}
    ) == "test"


def test_streamlit_secret_selects_production_when_environment_is_absent(monkeypatch):
    monkeypatch.delenv("QUANT_SIM_ENV", raising=False)

    assert environment.resolve_environment(
        streamlit_secrets={"QUANT_SIM_ENV": "production"}
    ) == "production"


def test_ambiguous_streamlit_server_fails_closed(monkeypatch):
    monkeypatch.delenv("QUANT_SIM_ENV", raising=False)
    monkeypatch.setenv("STREAMLIT_SERVER_PORT", "8501")
    monkeypatch.setattr(environment, "_streamlit_environment", lambda: None)

    assert environment.is_production_environment(fail_closed_streamlit=True) is True


def test_explicit_development_overrides_streamlit_server_fallback(monkeypatch):
    monkeypatch.setenv("QUANT_SIM_ENV", "development")
    monkeypatch.setenv("STREAMLIT_SERVER_PORT", "8501")

    assert environment.is_production_environment(fail_closed_streamlit=True) is False


def test_invalid_explicit_environment_is_rejected(monkeypatch):
    monkeypatch.setenv("QUANT_SIM_ENV", "staging-ish")

    with pytest.raises(environment.EnvironmentConfigurationError, match="must be one of"):
        environment.resolve_environment()
