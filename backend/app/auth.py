"""Cognito JWT verification and FastAPI auth dependencies."""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

import jwt
import requests
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import get_settings

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=True)

_JWKS_CACHE: dict[str, Any] = {}
_JWKS_CACHE_TTL = 3600  # seconds
_jwks_fetched_at: float = 0.0
_jwks_lock = threading.Lock()


def _jwks_url() -> str:
    settings = get_settings()
    return (
        f"https://cognito-idp.{settings.COGNITO_REGION}.amazonaws.com"
        f"/{settings.COGNITO_USER_POOL_ID}/.well-known/jwks.json"
    )


def _get_jwks() -> dict[str, Any]:
    global _jwks_fetched_at, _JWKS_CACHE
    now = time.time()
    with _jwks_lock:
        if _JWKS_CACHE and (now - _jwks_fetched_at) < _JWKS_CACHE_TTL:
            return _JWKS_CACHE

        url = _jwks_url()
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            _JWKS_CACHE = resp.json()
            _jwks_fetched_at = now
            return _JWKS_CACHE
        except Exception as exc:
            logger.error("Failed to fetch JWKS from %s: %s", url, exc)
            if _JWKS_CACHE:
                return _JWKS_CACHE  # serve stale on error
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Auth service unavailable",
            )


def _decode_token(token: str) -> dict[str, Any]:
    settings = get_settings()

    # Decode header to get kid
    try:
        header = jwt.get_unverified_header(token)
    except jwt.exceptions.DecodeError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc

    kid = header.get("kid")
    jwks = _get_jwks()

    # Find matching key
    matching_key: Optional[Any] = None
    for key_data in jwks.get("keys", []):
        if key_data.get("kid") == kid:
            matching_key = jwt.algorithms.RSAAlgorithm.from_jwk(key_data)
            break

    if matching_key is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token signing key not found")

    issuer = (
        f"https://cognito-idp.{settings.COGNITO_REGION}.amazonaws.com"
        f"/{settings.COGNITO_USER_POOL_ID}"
    )

    try:
        claims = jwt.decode(
            token,
            key=matching_key,
            algorithms=["RS256"],
            audience=settings.COGNITO_CLIENT_ID,
            issuer=issuer,
            options={"require": ["exp", "iss", "sub", "aud", "token_use"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc

    if claims.get("token_use") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type: access token required",
        )

    return claims


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

def require_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> dict[str, Any]:
    """Verify JWT; return claims. Any authenticated user passes."""
    return _decode_token(credentials.credentials)


def require_admin(
    claims: dict[str, Any] = Depends(require_user),
) -> dict[str, Any]:
    """Verify JWT and that user belongs to the 'admin' Cognito group."""
    groups: list[str] = claims.get("cognito:groups", [])
    if "admin" not in groups:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin group membership required",
        )
    return claims
