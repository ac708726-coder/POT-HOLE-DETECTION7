"""Per-page sign-in guard.

Streamlit only runs app.py for pages routed through st.navigation. When the
main script stops early, Streamlit falls back to its own pages/ directory
navigation, and those links execute a page script on their own. So the gate in
app.py is not enough: every page has to assert the session for itself.
"""

from __future__ import annotations

import sqlite3

import streamlit as st

from config import DATABASE_PATH
from utils.auth import initialize_users


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

    user_id = st.session_state.get("user_id")
    if user_id and _account_exists(int(user_id)):
        return int(user_id)

    for key in ("user_id", "username"):
        st.session_state.pop(key, None)

    st.warning("Sign in to use this page.", icon=":material/lock:")
    # A plain link, not st.page_link: this page can run outside st.navigation,
    # where the page registry has no entry to resolve and page_link raises.
    st.link_button("Go to sign in", "/", icon=":material/login:")
    st.stop()
    raise AssertionError("unreachable")  # pragma: no cover
