"""Streamlit entry point and navigation for the pothole detection project."""

from __future__ import annotations

import streamlit as st

from config import MODEL_PATH, ensure_runtime_directories
from utils.auth import AuthError, create_user, user_count, verify_user
from utils.observability import configure_logging, log_auth_event
from utils.session import (
    begin_session,
    end_session,
    flush_cookie_action,
    restore_session,
)
from utils.ui import (
    account_badge,
    apply_app_style,
    brand,
    hero,
    hero_brief,
    section_heading,
    signin_shell,
)

st.set_page_config(
    page_title="Divot — Road intelligence",
    page_icon=":material/add_road:",
    layout="wide",
    initial_sidebar_state="expanded",
)
ensure_runtime_directories()
configure_logging()
apply_app_style()


def home() -> None:
    """Render the product overview page."""

    model_ready = MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 0
    hero(
        "Road intelligence / redefined",
        "Every impact leaves a signature.",
        "Divot turns road images and video into visible, reviewable pothole candidates—without hiding uncertainty.",
        ready=model_ready,
    )
    hero_brief()

    section_heading(
        "Begin an inspection",
        "Choose the source that matches the road evidence you have.",
    )
    image_col, video_col = st.columns(2)
    with image_col, st.container(key="image_action", border=False):
        st.html('<span class="action-type">Still image</span>')
        st.markdown("### Inspect a single frame")
        st.write("Precision review with fast, balanced, and thorough scan modes.")
        st.page_link(
            "pages/1_image_detection.py",
            label="Open image scanner",
            icon=":material/arrow_forward:",
        )
    with video_col, st.container(key="video_action", border=False):
        st.html('<span class="action-type">Moving image</span>')
        st.markdown("### Follow the whole road")
        st.write("Frame-by-frame analysis with approximate pothole tracking.")
        st.page_link(
            "pages/2_video_detection.py",
            label="Open video scanner",
            icon=":material/arrow_forward:",
        )

    st.caption(
        "Advisory computer vision—not a civil-engineering measurement. Wet roads, shadows, repairs, and camera blur can affect results."
    )


def _sign_in_gate() -> None:
    """Restore a valid signed cookie, or show the existing account forms."""

    if restore_session():
        return

    signin_shell(
        "Sign in",
        "Detection history is saved per account. Nobody else can see yours.",
    )

    with st.container(key="auth_panel", border=False):
        sign_in_tab, register_tab = st.tabs(["Sign in", "Create account"])

        with sign_in_tab:
            with st.form("sign_in", border=False):
                username = st.text_input("Username", key="signin_username")
                password = st.text_input(
                    "Password", type="password", key="signin_password"
                )
                submitted = st.form_submit_button(
                    "Sign in", type="primary", width="stretch"
                )
            if submitted:
                try:
                    begin_session(
                        verify_user(username, password), username.strip().lower()
                    )
                except AuthError as exc:
                    log_auth_event("sign-in-refused", username, reason=exc)
                    st.error(str(exc), icon=":material/lock:")
                else:
                    log_auth_event("sign-in", username)
                    st.rerun()

        with register_tab:
            with st.form("register", border=False):
                new_username = st.text_input("Username", key="register_username")
                new_password = st.text_input(
                    "Password",
                    type="password",
                    key="register_password",
                    help="At least 10 characters.",
                )
                confirm = st.text_input(
                    "Confirm password", type="password", key="register_confirm"
                )
                created = st.form_submit_button("Create account", width="stretch")
            if created:
                if new_password != confirm:
                    st.error("Passwords do not match.", icon=":material/error:")
                else:
                    try:
                        begin_session(
                            create_user(new_username, new_password),
                            new_username.strip().lower(),
                        )
                    except AuthError as exc:
                        log_auth_event("register-refused", new_username, reason=exc)
                        st.error(str(exc), icon=":material/error:")
                    else:
                        log_auth_event("register", new_username)
                        st.rerun()

    if user_count() == 0:
        st.caption("No accounts yet. Create the first one to begin.")
    st.caption(
        "Passwords are stored as salted scrypt hashes, never in plain text. "
        "Serve this app over HTTPS so they are not readable in transit."
    )
    st.stop()


def _sign_out() -> None:
    log_auth_event("sign-out", str(st.session_state.get("username", "")))
    end_session()
    st.rerun()


flush_cookie_action()
_sign_in_gate()

with st.sidebar:
    brand()
    account_badge(str(st.session_state.get("username", "")))
    if st.button("Sign out", icon=":material/logout:", width="stretch"):
        _sign_out()

navigation = st.navigation(
    [
        st.Page(home, title="Overview", icon=":material/home:", default=True),
        st.Page(
            "pages/1_image_detection.py",
            title="Image detection",
            icon=":material/image_search:",
            url_path="image_detection",
        ),
        st.Page(
            "pages/2_video_detection.py",
            title="Video detection",
            icon=":material/videocam:",
            url_path="video_detection",
        ),
        st.Page(
            "pages/3_detection_history.py",
            title="Detection history",
            icon=":material/history:",
            url_path="detection_history",
        ),
    ],
    position="sidebar",
)
navigation.run()
