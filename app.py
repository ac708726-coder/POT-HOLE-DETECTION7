"""Streamlit entry point and navigation for the pothole detection project."""

from __future__ import annotations

import streamlit as st

from config import MODEL_PATH, ensure_runtime_directories
from utils.ui import apply_app_style, brand, hero, hero_brief, section_heading

st.set_page_config(
    page_title="Surface/01 — Road intelligence",
    page_icon=":material/add_road:",
    layout="wide",
    initial_sidebar_state="expanded",
)
ensure_runtime_directories()
apply_app_style()


def home() -> None:
    """Render the product overview page."""

    model_ready = MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 0
    hero(
        "Road intelligence / redefined",
        "Every impact leaves a signature.",
        "Surface/01 turns road images and video into visible, reviewable pothole candidates—without hiding uncertainty.",
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


with st.sidebar:
    brand()

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
