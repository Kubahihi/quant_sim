from unittest.mock import Mock

from src.auth import database


def test_plaintext_session_tokens_are_migrated_to_digests() -> None:
    plaintext_token = "legacy-active-session-token"
    cursor = Mock()
    cursor.fetchall.return_value = [(plaintext_token,)]
    connection = Mock()
    connection.execute.return_value = cursor

    database._migrate_plaintext_session_tokens(connection)

    expected_digest = database._session_token_digest(plaintext_token)
    connection.execute.assert_any_call(
        "UPDATE sessions SET token = ? WHERE token = ?",
        (expected_digest, plaintext_token),
    )
    assert expected_digest.startswith("sha256:")
    assert plaintext_token not in expected_digest
