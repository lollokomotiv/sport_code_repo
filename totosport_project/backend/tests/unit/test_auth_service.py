"""Unit test del servizio auth — logica pura, nessun DB."""

from datetime import timedelta

import pytest
from jose import jwt

from app.config import settings
from app.services.auth import (
    ALGORITHM,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)


# ─── Password hashing ─────────────────────────────────────────────────────────


def test_hash_password_is_not_plaintext():
    hashed = hash_password("test1234")
    assert hashed != "test1234"
    assert hashed.startswith("$2")  # prefisso bcrypt


def test_verify_password_correct():
    hashed = hash_password("test1234")
    assert verify_password("test1234", hashed) is True


def test_verify_password_wrong():
    hashed = hash_password("test1234")
    assert verify_password("sbagliata", hashed) is False


def test_hash_is_salted_each_time():
    # Due hash della stessa password devono differire (salt diverso)
    assert hash_password("uguale") != hash_password("uguale")


# ─── Token creation & payload ──────────────────────────────────────────────────


def test_access_token_has_correct_type_and_sub():
    token = create_access_token({"sub": "user-123", "role": "admin"})
    payload = decode_token(token)
    assert payload["type"] == "access"
    assert payload["sub"] == "user-123"
    assert payload["role"] == "admin"
    assert "exp" in payload


def test_refresh_token_has_correct_type():
    token = create_refresh_token({"sub": "user-123"})
    payload = decode_token(token)
    assert payload["type"] == "refresh"
    assert payload["sub"] == "user-123"


def test_access_and_refresh_tokens_differ():
    data = {"sub": "user-123"}
    assert create_access_token(data) != create_refresh_token(data)


# ─── Token decoding ──────────────────────────────────────────────────────────


def test_decode_invalid_token_returns_empty():
    assert decode_token("non-un-token-valido") == {}


def test_decode_token_signed_with_wrong_key_returns_empty():
    # Firmato con una chiave diversa → deve fallire la verifica
    forged = jwt.encode({"sub": "x", "type": "access"}, "chiave-sbagliata", algorithm=ALGORITHM)
    assert decode_token(forged) == {}


def test_decode_expired_token_returns_empty():
    # Token già scaduto
    expired = jwt.encode(
        {"sub": "x", "type": "access", "exp": 1},  # epoch 1 = 1970
        settings.secret_key,
        algorithm=ALGORITHM,
    )
    assert decode_token(expired) == {}
