import pytest

from src.auth.wharton_credentials import (
    REQUIRED_WHARTON_USERS,
    WhartonCredentialConfigError,
    resolve_wharton_credentials,
    validate_wharton_credentials,
)


def _valid_credentials() -> dict[str, str]:
    return {
        username: f"Unique-{index}-Password9"
        for index, username in enumerate(REQUIRED_WHARTON_USERS)
    }


def test_valid_credentials_are_returned_without_transformation() -> None:
    credentials = _valid_credentials()

    assert validate_wharton_credentials(credentials) == credentials


def test_shared_ten_character_password_is_allowed() -> None:
    credentials = dict.fromkeys(REQUIRED_WHARTON_USERS, "wharton123")

    assert validate_wharton_credentials(credentials) == credentials


def test_production_legacy_shared_password_overrides_per_user_values() -> None:
    credentials = resolve_wharton_credentials(
        {
            "WHARTON_PASSWORD": "wharton123",
            "wharton_users": {
                username: f"Stale-{index}-Password9"
                for index, username in enumerate(REQUIRED_WHARTON_USERS)
            },
        },
        production=True,
    )

    assert credentials == dict.fromkeys(REQUIRED_WHARTON_USERS, "wharton123")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda values: values.pop("Matěj"),
        lambda values: values.__setitem__("Matěj", "short9"),
        lambda values: values.__setitem__("Matěj", "a" * 73),
        lambda values: values.__setitem__("Matěj", "replace-with-a-strong-password"),
    ],
)
def test_incomplete_or_unsafe_credentials_fail_closed(mutation) -> None:
    credentials = _valid_credentials()
    mutation(credentials)

    with pytest.raises(WhartonCredentialConfigError):
        validate_wharton_credentials(credentials)
