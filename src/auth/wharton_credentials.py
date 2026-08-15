"""Fail-closed validation for the fixed Wharton team credential set."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


REQUIRED_WHARTON_USERS = (
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


def resolve_wharton_credentials(
    secret_values: Mapping[str, Any],
    *,
    production: bool,
) -> dict[str, str]:
    """Resolve the shared team secret, with per-user credentials as a fallback."""
    raw_users = secret_values.get("wharton_users")
    user_values = raw_users if isinstance(raw_users, Mapping) else {}

    if production:
        shared_password = secret_values.get("WHARTON_SHARED_PASSWORD")
        # Older deployments used WHARTON_PASSWORD at the top level. Keep that
        # value working online so a production restart can synchronize every
        # team account without requiring an immediate secrets migration.
        if shared_password is None:
            shared_password = secret_values.get("WHARTON_PASSWORD")
        if shared_password is None:
            shared_password = user_values.get("WHARTON_PASSWORD")
        if shared_password is not None:
            return validate_wharton_credentials(
                dict.fromkeys(REQUIRED_WHARTON_USERS, shared_password)
            )
        return validate_wharton_credentials(user_values)

    # A shared local password intentionally applies to all four accounts, even
    # when an older secrets file still contains stale per-user values.
    shared_password = secret_values.get("WHARTON_PASSWORD")
    if shared_password is None:
        shared_password = user_values.get("WHARTON_PASSWORD")
    if shared_password is not None:
        return validate_wharton_credentials(
            dict.fromkeys(REQUIRED_WHARTON_USERS, shared_password)
        )

    local_credentials = {
        username: user_values.get(username)
        for username in REQUIRED_WHARTON_USERS
    }
    return validate_wharton_credentials(local_credentials)
