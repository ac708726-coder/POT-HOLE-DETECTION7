"""Signed, expiring login tokens. No password or signing key enters the token."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Any

SESSION_TTL_SECONDS = 7 * 24 * 60 * 60
COOKIE_NAME = "divot_session"


def _key_bytes(key: str) -> bytes:
    value = key.encode("utf-8")
    if len(value) < 32:
        raise ValueError("DIVOT_COOKIE_SIGNING_KEY must contain at least 32 bytes.")
    return value


def create_token(
    user_id: int, key: str, *, ttl: int = SESSION_TTL_SECONDS, now: int | None = None
) -> str:
    if type(user_id) is not int or user_id <= 0 or ttl <= 0:
        raise ValueError("A positive account id and token lifetime are required.")
    issued = int(time.time()) if now is None else now
    claims = {
        "v": 1,
        "uid": user_id,
        "iat": issued,
        "exp": issued + ttl,
        "sid": secrets.token_urlsafe(24),
    }
    payload = (
        base64.urlsafe_b64encode(json.dumps(claims, separators=(",", ":")).encode())
        .decode()
        .rstrip("=")
    )
    signature = hmac.new(_key_bytes(key), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{signature}"


def validate_token(
    token: str, key: str, *, now: int | None = None
) -> dict[str, Any] | None:
    """Return verified claims, or None for tampered, malformed or expired input."""
    key_bytes = _key_bytes(key)
    if not isinstance(token, str) or len(token) > 2048:
        return None
    try:
        payload, signature = token.split(".")
        expected = hmac.new(key_bytes, payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return None
        claims = json.loads(
            base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
        )
        current = int(time.time()) if now is None else now
        if (
            not isinstance(claims, dict)
            or claims.get("v") != 1
            or type(claims.get("uid")) is not int
            or claims["uid"] <= 0
            or type(claims.get("iat")) is not int
            or type(claims.get("exp")) is not int
            or not claims["iat"] <= current < claims["exp"]
            or not isinstance(claims.get("sid"), str)
            or len(claims["sid"]) != 32
        ):
            return None
        return claims
    except (ValueError, TypeError, UnicodeError):
        return None
