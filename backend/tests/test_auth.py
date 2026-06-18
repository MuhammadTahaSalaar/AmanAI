"""Unit tests for auth: admin gating, JWT decode path, token_use validation."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import jwt
import pytest
from fastapi import HTTPException

from backend.app.auth import _decode_token, require_admin, require_user


def test_require_admin_passes_for_admin_group():
    claims = {"sub": "user1", "cognito:groups": ["admin", "users"]}
    result = require_admin(claims)
    assert result == claims


def test_require_admin_rejects_non_admin():
    claims = {"sub": "user2", "cognito:groups": ["users"]}
    with pytest.raises(HTTPException) as exc_info:
        require_admin(claims)
    assert exc_info.value.status_code == 403


def test_require_admin_rejects_empty_groups():
    claims = {"sub": "user3", "cognito:groups": []}
    with pytest.raises(HTTPException) as exc_info:
        require_admin(claims)
    assert exc_info.value.status_code == 403


def test_require_admin_rejects_missing_groups_key():
    claims = {"sub": "user4"}
    with pytest.raises(HTTPException):
        require_admin(claims)


def test_require_admin_passes_with_admin_claims():
    admin_claims = {"sub": "user5", "cognito:groups": ["admin"]}
    assert require_admin(admin_claims) == admin_claims


def test_decode_token_rejects_id_token(monkeypatch):
    """token_use != 'access' must raise 401."""
    fake_claims = {
        "sub": "user1",
        "token_use": "id",  # id token, not access
        "aud": "test-client-id",
        "exp": 9999999999,
        "iss": "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_TESTPOOL",
    }

    fake_key = MagicMock()
    fake_jwks = {"keys": [{"kid": "testkey", "kty": "RSA"}]}

    monkeypatch.setattr("backend.app.auth._get_jwks", lambda: fake_jwks)
    monkeypatch.setattr(
        "jwt.algorithms.RSAAlgorithm.from_jwk", lambda data: fake_key
    )
    monkeypatch.setattr(
        "jwt.get_unverified_header", lambda token: {"kid": "testkey", "alg": "RS256"}
    )
    monkeypatch.setattr("jwt.decode", lambda *a, **kw: fake_claims)

    with pytest.raises(HTTPException) as exc_info:
        _decode_token("fake.token.here")
    assert exc_info.value.status_code == 401
    assert "access token" in exc_info.value.detail.lower()


def test_decode_token_accepts_access_token(monkeypatch):
    """token_use == 'access' must succeed."""
    fake_claims = {
        "sub": "user1",
        "token_use": "access",
        "aud": "test-client-id",
        "exp": 9999999999,
        "iss": "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_TESTPOOL",
    }

    fake_key = MagicMock()
    fake_jwks = {"keys": [{"kid": "testkey", "kty": "RSA"}]}

    monkeypatch.setattr("backend.app.auth._get_jwks", lambda: fake_jwks)
    monkeypatch.setattr(
        "jwt.algorithms.RSAAlgorithm.from_jwk", lambda data: fake_key
    )
    monkeypatch.setattr(
        "jwt.get_unverified_header", lambda token: {"kid": "testkey", "alg": "RS256"}
    )
    monkeypatch.setattr("jwt.decode", lambda *a, **kw: fake_claims)

    result = _decode_token("fake.token.here")
    assert result["token_use"] == "access"
