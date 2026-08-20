import uuid

import pytest

from app.core.security import (
    InvalidTokenError,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_hash_password_does_not_store_plaintext() -> None:
    hashed = hash_password("correct horse battery staple")
    assert hashed != "correct horse battery staple"


def test_verify_password_accepts_correct_and_rejects_wrong() -> None:
    hashed = hash_password("correct horse battery staple")

    assert verify_password("correct horse battery staple", hashed) is True
    assert verify_password("wrong password", hashed) is False


def test_access_token_round_trips_user_id() -> None:
    user_id = uuid.uuid4()
    token = create_access_token(user_id=user_id)

    assert decode_access_token(token) == user_id


def test_decode_access_token_rejects_garbage() -> None:
    with pytest.raises(InvalidTokenError):
        decode_access_token("not-a-real-token")


def test_decode_access_token_rejects_tampered_signature() -> None:
    # Flips a character in the middle of the signature, not the last one — the
    # last base64url character of a 32-byte HS256 signature carries 2 unused
    # padding bits, so tampering it can occasionally decode to the same bytes
    # and leave the "tampered" token still valid (real, observed flakiness,
    # not hypothetical). A middle character has no such boundary case.
    token = create_access_token(user_id=uuid.uuid4())
    header, payload, signature = token.split(".")
    mid = len(signature) // 2
    tampered_char = "A" if signature[mid] != "A" else "B"
    tampered_signature = signature[:mid] + tampered_char + signature[mid + 1 :]
    tampered = f"{header}.{payload}.{tampered_signature}"

    with pytest.raises(InvalidTokenError):
        decode_access_token(tampered)
