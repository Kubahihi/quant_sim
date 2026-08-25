import pytest

from src.auth.wharton_credentials import (
    REQUIRED_WHARTON_USERS,
    WHARTON_JUDGE_USERNAME,
    WhartonCredentialConfigError,
    resolve_wharton_credentials,
    resolve_wharton_judge_credential,
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
        lambda values: values.__setitem__("Matěj", "replace-with-the-shared-password"),
    ],
)
def test_incomplete_or_unsafe_credentials_fail_closed(mutation) -> None:
    credentials = _valid_credentials()
    mutation(credentials)

    with pytest.raises(WhartonCredentialConfigError):
        validate_wharton_credentials(credentials)


def test_judge_username_is_canonical_lowercase() -> None:
    assert WHARTON_JUDGE_USERNAME == "judge"
    assert WHARTON_JUDGE_USERNAME.islower()


def test_absent_judge_credential_disables_judge_account() -> None:
    assert (
        resolve_wharton_judge_credential(
            {},
            team_credentials=_valid_credentials(),
        )
        is None
    )


@pytest.mark.parametrize(
    ("secret_values", "expected"),
    [
        ({"WHARTON_JUDGE_PASSWORD": "Top-Level-Judge9"}, "Top-Level-Judge9"),
        ({"wharton_users": {"judge": "Nested-Judge-Pass9"}}, "Nested-Judge-Pass9"),
    ],
)
def test_judge_credential_supports_top_level_and_nested_secrets(
    secret_values,
    expected,
) -> None:
    assert resolve_wharton_judge_credential(
        secret_values,
        team_credentials=_valid_credentials(),
    ) == expected


def test_top_level_judge_credential_overrides_nested_value() -> None:
    assert resolve_wharton_judge_credential(
        {
            "WHARTON_JUDGE_PASSWORD": "Current-Judge-Pass9",
            "wharton_users": {"judge": "Stale-Nested-Pass9"},
        },
        team_credentials=_valid_credentials(),
    ) == "Current-Judge-Pass9"


@pytest.mark.parametrize(
    "invalid_password",
    [
        1234567890,
        "short9",
        "a" * 73,
        "replace-with-a-distinct-judge-password",
        "replace-with-a-strong-password",
    ],
)
def test_invalid_judge_credentials_fail_closed(invalid_password) -> None:
    with pytest.raises(WhartonCredentialConfigError):
        resolve_wharton_judge_credential(
            {"WHARTON_JUDGE_PASSWORD": invalid_password},
            team_credentials=_valid_credentials(),
        )


@pytest.mark.parametrize("team_username", REQUIRED_WHARTON_USERS)
def test_judge_credential_must_differ_from_every_team_password(
    team_username: str,
) -> None:
    team_credentials = _valid_credentials()

    with pytest.raises(WhartonCredentialConfigError):
        resolve_wharton_judge_credential(
            {"WHARTON_JUDGE_PASSWORD": team_credentials[team_username]},
            team_credentials=team_credentials,
        )
