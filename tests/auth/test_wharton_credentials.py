import pytest

from src.auth.wharton_credentials import (
    REQUIRED_WHARTON_USERS,
    WhartonCredentialConfigError,
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


@pytest.mark.parametrize(
    "mutation",
    [
        lambda values: values.pop("Matěj"),
        lambda values: values.__setitem__("Matěj", "short9"),
        lambda values: values.__setitem__("Matěj", "a" * 73),
        lambda values: values.__setitem__("Matěj", "replace-with-a-strong-password"),
        lambda values: values.__setitem__("Matěj", values["Martin"]),
    ],
)
def test_incomplete_or_unsafe_credentials_fail_closed(mutation) -> None:
    credentials = _valid_credentials()
    mutation(credentials)

    with pytest.raises(WhartonCredentialConfigError):
        validate_wharton_credentials(credentials)
