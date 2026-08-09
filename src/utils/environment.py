"""Single source of truth for runtime environment selection."""

from __future__ import annotations

import os
from typing import Any, Mapping


VALID_ENVIRONMENTS = frozenset({"development", "test", "production"})


class EnvironmentConfigurationError(ValueError):
    """Raised when an explicit runtime environment is unsupported."""


def _streamlit_environment() -> Any:
    """Read the top-level Streamlit secret without exposing other secrets."""
    try:
        import streamlit as st

        return st.secrets.get("QUANT_SIM_ENV")
    except Exception:
        return None


def resolve_environment(
    default: str | None = "development",
    *,
    streamlit_secrets: Mapping[str, Any] | None = None,
) -> str | None:
    """Resolve environment variables first, then Streamlit secrets."""
    raw_value: Any = os.environ.get("QUANT_SIM_ENV")
    if raw_value is None or not str(raw_value).strip():
        if streamlit_secrets is not None:
            raw_value = streamlit_secrets.get("QUANT_SIM_ENV")
        else:
            raw_value = _streamlit_environment()
    if raw_value is None or not str(raw_value).strip():
        raw_value = default
    if raw_value is None:
        return None

    environment = str(raw_value).strip().lower()
    if environment not in VALID_ENVIRONMENTS:
        allowed = ", ".join(sorted(VALID_ENVIRONMENTS))
        raise EnvironmentConfigurationError(
            f"QUANT_SIM_ENV must be one of: {allowed}."
        )
    return environment


def is_production_environment(*, fail_closed_streamlit: bool = False) -> bool:
    """Return production status, optionally treating ambiguous Streamlit as prod."""
    environment = resolve_environment(default=None)
    if environment is not None:
        return environment == "production"
    return fail_closed_streamlit and "STREAMLIT_SERVER_PORT" in os.environ
