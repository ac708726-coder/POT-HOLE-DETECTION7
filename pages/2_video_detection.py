"""Streamlit page for incremental video detection and approximate tracking."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st

from config import (
    DEFAULT_CONFIDENCE,
    MAX_VIDEO_SIZE_MB,
    MIN_SCAN_CONFIDENCE,
    MODEL_PATH,
)
from utils.detector import (
    DetectorError,
    ModelNotFoundError,
    inference_profile_details,
    warm_up_model,
)
from utils.observability import report_error
from utils.scan_state import scan_key, store_scan_result, sync_scan_inputs
from utils.session import require_user
from utils.storage import create_detection_record
from utils.ui import media_stamp, page_intro, scanning_banner, section_heading
from utils.validators import (
    ValidationError,
    safe_generated_filename,
    validate_video_upload,
)
from utils.video_processor import VideoProcessingError, process_video

# Streamlit can run this page without app.py, so the gate is asserted here.
USER_ID = require_user()

model_stat = MODEL_PATH.stat() if MODEL_PATH.is_file() else None
model_ready = model_stat is not None and model_stat.st_size > 0
page_intro(
    "02",
    "motion inspection",
    "Follow the road, frame by frame.",
    "Process moving footage, track recurring candidates, and export an annotated route review.",
)

section_heading(
    "Load road footage",
    "MP4, MOV, or AVI. Nothing is saved to history until you choose to save it.",
)
with st.container(border=True):
    uploaded_file = st.file_uploader(
        "Upload a road video",
        type=["mp4", "mov", "avi"],
        help=f"Up to {MAX_VIDEO_SIZE_MB} MB and 5 minutes.",
    )

if uploaded_file is None:
    sync_scan_inputs(st.session_state, "video_detection", None)
    st.caption(
        "MP4, MOV, or AVI up to 200 MB and five minutes. Video is not added to history automatically."
    )
    st.stop()

video_bytes = uploaded_file.getvalue()
try:
    input_extension = validate_video_upload(uploaded_file.name, video_bytes)
except ValidationError as exc:
    st.error(str(exc), icon=":material/error:")
    st.stop()

media_stamp(
    uploaded_file.name,
    f"{len(video_bytes) / (1024 * 1024):.1f} MB",
    input_extension.removeprefix(".").upper(),
)

section_heading(
    "Preview and tune the route",
    "Balance frame coverage, confidence, and processing time for this recording.",
)
preview_col, controls_col = st.columns([1.18, 0.82], vertical_alignment="top")
with preview_col:
    st.video(video_bytes)
with controls_col, st.container(key="video_scan_controls", border=True):
    mode = st.segmented_control(
        "Processing mode",
        options=["fast", "balanced"],
        default="fast",
        required=True,
        format_func=lambda value: inference_profile_details(value)["label"],
        help="Balanced is more robust but can be several times slower on video.",
        width="stretch",
    )
    st.caption(inference_profile_details(str(mode))["description"])
    confidence = st.slider(
        "Minimum confidence",
        min_value=MIN_SCAN_CONFIDENCE,
        max_value=0.95,
        value=DEFAULT_CONFIDENCE,
        step=0.05,
    )
    frame_skip = st.segmented_control(
        "Analyze every",
        options=[1, 2, 3],
        default=2,
        required=True,
        format_func=lambda value: "Frame" if value == 1 else f"{value} frames",
        help="Analyzing fewer frames improves speed but can miss brief appearances.",
        width="stretch",
    )
    process_clicked = st.button(
        "Run route scan",
        type="primary",
        icon=":material/play_arrow:",
        disabled=not model_ready,
        width="stretch",
        key="primary_action",
    )

model_signature = (
    f"{model_stat.st_mtime_ns}:{model_stat.st_size}" if model_stat else "missing"
)
result_key = scan_key(video_bytes, str(mode), confidence, int(frame_skip))
settings_changed = sync_scan_inputs(
    st.session_state, "video_detection", result_key, model_signature
)
if settings_changed and not process_clicked:
    st.info("Settings changed — rescan.", icon=":material/refresh:")

if not model_ready:
    st.warning(
        "Video processing is unavailable because `models/best.pt` is missing.",
        icon=":material/model_training:",
    )

# The controls above have rendered and the user is still choosing settings, so this is
# the cheapest moment to absorb model start-up cost before the first real scan.
if model_ready:
    warm_up_model()

if process_clicked:
    progress = st.progress(0, text="Preparing video…")
    scanning_banner("Tracking road damage frame by frame…")
    try:
        with TemporaryDirectory(prefix="pothole_video_") as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / safe_generated_filename(
                uploaded_file.name, input_extension
            )
            output_path = temporary_path / safe_generated_filename("result.mp4", ".mp4")
            input_path.write_bytes(video_bytes)

            details = process_video(
                input_path,
                output_path,
                confidence=confidence,
                frame_skip=int(frame_skip),
                inference_profile=str(mode),
                progress_callback=lambda value: progress.progress(
                    int(value * 100), text=f"Processing video… {int(value * 100)}%"
                ),
            )
            processed_bytes = output_path.read_bytes()
        store_scan_result(
            st.session_state,
            "video_detection",
            result_key,
            {
                "filename": uploaded_file.name,
                "details": details,
                "video_bytes": processed_bytes,
            },
        )
        progress.empty()
        st.toast("Video processing complete.", icon=":material/check_circle:")
    except (
        ValidationError,
        ModelNotFoundError,
        DetectorError,
        VideoProcessingError,
        OSError,
    ) as exc:
        progress.empty()
        reference = report_error("video detection failed", exc)
        st.error(f"{exc} (reference {reference})", icon=":material/error:")

saved = st.session_state.get("video_detection_results", {}).get(result_key)
if not saved:
    st.stop()

details = saved["details"]
section_heading(
    "Review the processed footage",
    "Inspect the annotated route and the approximate recurring-candidate count.",
)
metric_a, metric_b, metric_c, metric_d = st.columns(4)
metric_a.metric("Unique potholes (estimate)", details["unique_count_estimate"])
metric_b.metric("Processing time", f"{details['processing_seconds']:.1f} s")
metric_c.metric("Processing speed", f"{details['processing_fps']:.1f} FPS")
metric_d.metric("Mode", inference_profile_details(str(mode))["label"])

with st.container(border=True):
    st.video(saved["video_bytes"], format="video/mp4")
st.caption(
    "Camera movement, occlusion, or scene changes can create a new tracking ID. "
    "Treat the unique count as an estimate."
)

download_col, save_col = st.columns(2)
with download_col:
    st.download_button(
        "Download annotated video",
        data=saved["video_bytes"],
        file_name="annotated_potholes.mp4",
        mime="video/mp4",
        icon=":material/download:",
        width="stretch",
    )
with save_col:
    if st.button("Save summary to history", icon=":material/save:", width="stretch"):
        record_id = create_detection_record(
            user_id=USER_ID,
            input_type="video",
            input_name=uploaded_file.name,
            detection_count=details["unique_count_estimate"],
            processing_ms=details["processing_seconds"] * 1000,
        )
        st.toast(f"Saved history record {record_id}.", icon=":material/save:")
