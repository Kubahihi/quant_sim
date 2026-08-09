"""Runtime environment resolution tests."""

from __future__ import annotations

import pytest

from src.utils import environment


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
