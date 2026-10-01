"""Per-page sign-in guard.

Streamlit only runs app.py for pages routed through st.navigation. When the
main script stops early, Streamlit falls back to its own pages/ directory
navigation, and those links execute a page script on their own. So the gate in
app.py is not enough: every page has to assert the session for itself.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone

import streamlit as st

from config import DATABASE_PATH
from utils.auth import (
    initialize_users,
    login_session_user,
    register_login_session,
    revoke_login_session,
)
from utils.auth_tokens import COOKIE_NAME, create_token, validate_token


def cookie_signing_key() -> str | None:
    key = os.getenv("DIVOT_COOKIE_SIGNING_KEY")
    if not key:
        try:
            key = st.secrets.get("DIVOT_COOKIE_SIGNING_KEY")
        except FileNotFoundError:
            key = None
    return str(key) if key and len(str(key).encode()) >= 32 else None


def flush_cookie_action() -> None:
    """Render writes on the run AFTER a login/logout rerun, so they reach the browser.

    Keep the identical write mounted during this connection. Cookie reads on a
    fresh connection use st.context.cookies, avoiding asynchronous hydration races.
    """
    action = st.session_state.get("_cookie_action")
    if not action:
        return
    from streamlit_cookies_controller import CookieController

    controller = CookieController(key="divot_cookie_controller")
    controller.set(
        COOKIE_NAME,
        action["value"],
        path="/",
        expires=datetime.fromtimestamp(action["expires"], timezone.utc),
        max_age=action["ttl"],
        same_site="lax",
        secure=st.context.url.startswith("https://"),
    )


def begin_session(user_id: int, username: str) -> None:
    key = cookie_signing_key()
    st.session_state["_signed_out"] = False
    if key:
        token = create_token(user_id, key)
        claims = validate_token(token, key)
        register_login_session(claims)
        st.session_state["_login_token"] = token
        st.session_state["_cookie_action"] = {
            "value": token,
            "expires": claims["exp"],
            "ttl": claims["exp"] - claims["iat"],
        }
    else:
        # Existing session-only sign-in remains usable during initial secret setup.
        st.session_state["_session_only"] = True
    st.session_state["user_id"] = user_id
    st.session_state["username"] = username


def restore_session() -> int | None:
    """Validate expiry, signature, revocation and account existence on every load."""
    if st.session_state.get("_signed_out"):
        return None
    key = cookie_signing_key()
    token = st.session_state.get("_login_token") or st.context.cookies.get(COOKIE_NAME)
    if key and token:
        claims = validate_token(token, key)
        username = login_session_user(claims) if claims else None
        if username:
            st.session_state.update(
                user_id=claims["uid"], username=username, _login_token=token
            )
            return claims["uid"]
    elif st.session_state.get("_session_only"):
        user_id = st.session_state.get("user_id")
        if user_id and _account_exists(int(user_id)):
            return int(user_id)
    for name in ("user_id", "username", "_login_token", "_session_only"):
        st.session_state.pop(name, None)
    return None


def end_session() -> None:
    key = cookie_signing_key()
    token = st.session_state.get("_login_token") or st.context.cookies.get(COOKIE_NAME)
    claims = validate_token(token, key) if token and key else None
    if claims:
        revoke_login_session(claims["sid"])
    st.session_state.clear()
    st.session_state["_signed_out"] = True
    st.session_state["_cookie_action"] = {"value": "", "expires": 0, "ttl": 0}


def _account_exists(user_id: int) -> bool:
    initialize_users()
    with sqlite3.connect(DATABASE_PATH) as connection:
        row = connection.execute(
            "SELECT 1 FROM users WHERE id = ?", (int(user_id),)
        ).fetchone()
    return row is not None


def require_user() -> int:
    """Return the signed-in account id, or stop the page.

    Also re-checks that the account still exists, so a session left open after
    an account is removed cannot keep reading or writing history.
    """

    user_id = restore_session()
    if user_id:
        return user_id

    for key in ("user_id", "username"):
        st.session_state.pop(key, None)

    st.warning("Sign in to use this page.", icon=":material/lock:")
    # A plain link, not st.page_link: this page can run outside st.navigation,
    # where the page registry has no entry to resolve and page_link raises.
    st.link_button("Go to sign in", "/", icon=":material/login:")
    st.stop()
    raise AssertionError("unreachable")  # pragma: no cover
