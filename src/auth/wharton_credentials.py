"""Fail-closed validation for the fixed Wharton team credential set."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


REQUIRED_WHARTON_USERS = (
    "Alexandra",
    "Jakub",
    "Lukáš",
    "Martin",
    "Matěj",
)
_FORBIDDEN_PASSWORDS = {
    "change_me_in_secrets",
    "dev_only_insecure_default",
    "replace-with-a-strong-password",
    "strong-unique-password",
}
_MIN_PASSWORD_BYTES = 10
_MAX_PASSWORD_BYTES = 72


class WhartonCredentialConfigError(RuntimeError):
    """Raised without secret values when team credentials are unsafe."""


def validate_wharton_credentials(raw: Any) -> dict[str, str]:
    """Return validated credentials or fail without exposing their values."""
    if not isinstance(raw, Mapping):
        raise WhartonCredentialConfigError("Wharton credentials are incomplete.")

    credentials: dict[str, str] = {}
    for username in REQUIRED_WHARTON_USERS:
        value = raw.get(username)
        if not isinstance(value, str):
            raise WhartonCredentialConfigError("Wharton credentials are incomplete.")
        password = value.strip()
        byte_length = len(password.encode("utf-8"))
        if (
            byte_length < _MIN_PASSWORD_BYTES
            or byte_length > _MAX_PASSWORD_BYTES
            or password.casefold() in _FORBIDDEN_PASSWORDS
        ):
            raise WhartonCredentialConfigError("Wharton credentials are invalid.")
        credentials[username] = password

    return credentials
