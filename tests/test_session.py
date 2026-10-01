from types import SimpleNamespace

import pytest

from utils import auth, session
from utils.auth_tokens import COOKIE_NAME, create_token, validate_token

KEY = "session-test-signing-key-" * 3


@pytest.fixture()
def signed_session(tmp_path, monkeypatch):
    db = tmp_path / "accounts.db"
    user_id = auth.create_user("driver", "long-test-password", db)
    token = create_token(user_id, KEY)
    auth.register_login_session(validate_token(token, KEY), db)
    fake_st = SimpleNamespace(
        session_state={},
        secrets={},
        context=SimpleNamespace(
            cookies={COOKIE_NAME: token}, url="https://divot.example/image_detection"
        ),
    )
    monkeypatch.setattr(session, "st", fake_st)
    monkeypatch.setenv("DIVOT_COOKIE_SIGNING_KEY", KEY)
    monkeypatch.setattr(
        session,
        "login_session_user",
        lambda claims: auth.login_session_user(claims, db),
    )
    monkeypatch.setattr(
        session, "revoke_login_session", lambda sid: auth.revoke_login_session(sid, db)
    )
    return fake_st, user_id, db


def test_fresh_connection_restores_cookie_without_component_hydration(signed_session):
    st, user_id, _ = signed_session
    assert session.restore_session() == user_id
    assert st.session_state["username"] == "driver"


def test_signout_clears_results_and_revokes_old_cookie(signed_session):
    st, _, _ = signed_session
    old_cookie = st.context.cookies[COOKIE_NAME]
    session.restore_session()
    st.session_state["image_detection_results"] = {"old": "data"}
    session.end_session()
    assert st.session_state["_cookie_action"]["ttl"] == 0
    assert "image_detection_results" not in st.session_state
    assert session.restore_session() is None
    st.session_state.clear()  # new browser connection still sends the old cookie
    st.context.cookies = {COOKIE_NAME: old_cookie}
    assert session.restore_session() is None


def test_missing_account_invalidates_cookie(signed_session):
    import sqlite3

    _, _, db = signed_session
    with sqlite3.connect(db) as connection:
        connection.execute("DELETE FROM users")
    assert session.restore_session() is None


def test_cookie_expiry_is_enforced_in_an_existing_tab(signed_session, monkeypatch):
    st, _, _ = signed_session
    session.restore_session()
    monkeypatch.setattr(session, "validate_token", lambda token, key: None)
    assert session.restore_session() is None
    assert "user_id" not in st.session_state


def test_key_can_come_from_streamlit_secrets(signed_session, monkeypatch):
    st, user_id, _ = signed_session
    monkeypatch.delenv("DIVOT_COOKIE_SIGNING_KEY")
    st.secrets = {"DIVOT_COOKIE_SIGNING_KEY": KEY}
    assert session.restore_session() == user_id
