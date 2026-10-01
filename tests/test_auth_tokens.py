from __future__ import annotations

import base64
import hashlib
import hmac
import json

import pytest

from utils.auth import (
    change_password,
    create_user,
    login_session_user,
    register_login_session,
    revoke_login_session,
)
from utils.auth_tokens import create_token, validate_token

KEY = "test-signing-key-" * 4


def test_token_round_trip_and_uniqueness():
    first = create_token(7, KEY, now=100, ttl=60)
    second = create_token(7, KEY, now=100, ttl=60)
    assert first != second
    claims = validate_token(first, KEY, now=101)
    assert claims["uid"] == 7
    assert claims["exp"] == 160
    assert "password" not in claims


def test_token_expiration_boundary_and_future_issue():
    token = create_token(7, KEY, now=100, ttl=60)
    assert validate_token(token, KEY, now=159)
    assert validate_token(token, KEY, now=160) is None
    assert validate_token(token, KEY, now=99) is None


def test_token_rejects_tampering_and_wrong_key():
    token = create_token(7, KEY, now=100)
    payload, signature = token.split(".")
    assert validate_token(payload + "." + "0" * 64, KEY, now=101) is None
    assert validate_token("A" + payload[1:] + "." + signature, KEY, now=101) is None
    assert validate_token(token, "different-secret-key-" * 4, now=101) is None


@pytest.mark.parametrize("value", [None, "", "abc", "a.b.c", "x" * 3000, "☃.abc"])
def test_malformed_token_is_refused(value):
    assert validate_token(value, KEY) is None


def test_valid_signature_does_not_bypass_claim_checks():
    claims = {"v": 1, "uid": True, "iat": 100, "exp": 200, "sid": "x" * 32}
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    signature = hmac.new(KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()
    assert validate_token(payload + "." + signature, KEY, now=101) is None


@pytest.mark.parametrize("key", ["", "too-short"])
def test_weak_signing_key_is_not_accepted(key):
    with pytest.raises(ValueError, match="32 bytes"):
        create_token(1, key)


def test_signout_revokes_token_and_password_change_invalidates_sessions(tmp_path):
    db = tmp_path / "accounts.db"
    user_id = create_user("driver", "long-test-password", db)
    claims = validate_token(create_token(user_id, KEY), KEY)
    register_login_session(claims, db)
    assert login_session_user(claims, db) == "driver"
    revoke_login_session(claims["sid"], db)
    assert login_session_user(claims, db) is None
    new_claims = validate_token(create_token(user_id, KEY), KEY)
    register_login_session(new_claims, db)
    change_password(user_id, "long-test-password", "other-long-password", db)
    assert login_session_user(new_claims, db) is None
