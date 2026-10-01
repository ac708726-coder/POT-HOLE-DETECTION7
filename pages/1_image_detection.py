"""Primary Streamlit page for image detection."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU_THRESHOLD,
    MAX_IMAGE_SIZE_MB,
    MIN_SCAN_CONFIDENCE,
    MODEL_PATH,
)
from utils.detection_details import no_detection_hint
from utils.detector import (
    DetectorError,
    ModelNotFoundError,
    inference_profile_details,
    predict_image,
    warm_up_model,
)
from utils.image_processor import decode_image, encode_image, image_metadata
from utils.observability import report_error
from utils.scan_state import scan_key, store_scan_result, sync_scan_inputs
from utils.session import require_user
from utils.severity import estimate_apparent_severity, summarize_severity
from utils.storage import create_detection_record
from utils.ui import (
    media_stamp,
    page_intro,
    result_reveal,
    scanning_banner,
    section_heading,
)
from utils.validators import ValidationError, validate_image_upload

# Streamlit can run this page without app.py, so the gate is asserted here.
USER_ID = require_user()

model_stat = MODEL_PATH.stat() if MODEL_PATH.is_file() else None
model_ready = model_stat is not None and model_stat.st_size > 0
page_intro(
    "01",
    "still-image inspection",
    "Interrogate every pixel.",
    "Load one road image, tune the detector, and review the evidence at full scale.",
)

section_heading(
    "Load a road image",
    "JPG, JPEG, or PNG. Nothing is saved to history until you choose to save it.",
)
with st.container(border=True):
    uploaded_file = st.file_uploader(
        "Upload a road image",
        type=["jpg", "jpeg", "png"],
        help=f"JPG, JPEG, or PNG up to {MAX_IMAGE_SIZE_MB} MB.",
    )

if uploaded_file is None:
    sync_scan_inputs(st.session_state, "image_detection", None)
    st.caption(
        "JPG, JPEG, or PNG. Your upload is processed for this session and is not added to history automatically."
    )
    st.stop()

file_bytes = uploaded_file.getvalue()
try:
    validate_image_upload(uploaded_file.name, file_bytes)
    original_image = decode_image(file_bytes)
except (ValidationError, RuntimeError) as exc:
    st.error(str(exc), icon=":material/error:")
    st.stop()

metadata = image_metadata(original_image)
media_stamp(
    uploaded_file.name,
    f"{metadata['width']} × {metadata['height']} px",
    f"{metadata['megapixels']:.2f} MP",
)

section_heading(
    "Frame and tune the scan",
    "Preview the source, select an inspection profile, and set the confidence floor.",
)
preview_col, controls_col = st.columns([1.18, 0.82], vertical_alignment="top")
with preview_col:
    st.image(original_image, caption="Source frame", width="stretch")
with controls_col, st.container(key="image_scan_controls", border=True):
    mode = st.segmented_control(
        "Scan mode",
        options=["fast", "balanced", "thorough"],
        default="balanced",
        required=True,
        format_func=lambda value: inference_profile_details(value)["label"],
        help="Balanced improves robustness. Thorough runs two augmented scales.",
        width="stretch",
    )
    selected_profile = inference_profile_details(str(mode))
    st.caption(selected_profile["description"])
    confidence = st.slider(
        "Minimum confidence",
        min_value=MIN_SCAN_CONFIDENCE,
        max_value=0.95,
        value=DEFAULT_CONFIDENCE,
        step=0.05,
        help="Lower values find more candidates but can add false detections.",
    )
    st.caption(
        "Lower confidence finds more candidates; higher confidence is more selective."
    )
    detect_clicked = st.button(
        "Run surface scan",
        type="primary",
        icon=":material/radar:",
        disabled=not model_ready,
        width="stretch",
        key="primary_action",
    )

model_signature = (
    f"{model_stat.st_mtime_ns}:{model_stat.st_size}" if model_stat else "missing"
)
result_key = scan_key(file_bytes, str(mode), confidence)
settings_changed = sync_scan_inputs(
    st.session_state, "image_detection", result_key, model_signature
)
if settings_changed and not detect_clicked:
    st.info("Settings changed — rescan.", icon=":material/refresh:")

if not model_ready:
    st.warning(
        "Detection is unavailable because `models/best.pt` is missing. "
        "See `models/README.md` for setup instructions.",
        icon=":material/model_training:",
    )

# The controls above have rendered and the user is still choosing settings, so this is
# the cheapest moment to absorb model start-up cost before the first real scan.
if model_ready:
    warm_up_model()

if detect_clicked:
    try:
        scanning_banner("Analyzing road texture and verifying candidate boxes…")
        with st.status("Scanning image…", expanded=True) as status:
            st.write(f"Using {selected_profile['label']} mode")
            pass_count = selected_profile["pass_count"]
            st.write(f"Running {pass_count} {'pass' if pass_count == 1 else 'passes'}")
            result = predict_image(
                original_image,
                confidence=confidence,
                iou_threshold=DEFAULT_IOU_THRESHOLD,
                inference_profile=str(mode),
            )
            status.update(label="Road scan complete", state="complete", expanded=False)
        store_scan_result(
            st.session_state,
            "image_detection",
            result_key,
            {
                "filename": uploaded_file.name,
                "result": result,
                "image_size": original_image.size,
            },
        )
        plural = "s" if result["count"] != 1 else ""
        st.toast(
            f"Found {result['count']} pothole candidate{plural}.",
            icon=":material/check_circle:",
        )
    except (ModelNotFoundError, DetectorError, ValidationError) as exc:
        reference = report_error("image detection failed", exc)
        st.error(f"{exc} (reference {reference})", icon=":material/error:")

saved = st.session_state.get("image_detection_results", {}).get(result_key)
if not saved:
    st.stop()

result = saved["result"]
result_reveal()
section_heading(
    "Review the evidence",
    "Switch between the source and annotated frame before exporting the result.",
)
result_view = st.segmented_control(
    "Result view",
    options=["Detection", "Original"],
    default="Detection",
    label_visibility="collapsed",
    key="image_result_view",
)
if result_view == "Original":
    st.image(original_image, caption="Original road image", width="stretch")
else:
    st.image(
        result["annotated_image"],
        caption="Detected pothole candidates",
        width="stretch",
    )

confidences = [item["confidence"] for item in result["detections"]]
average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
metric_a, metric_b, metric_c, metric_d = st.columns(4)
metric_a.metric("Potholes", result["count"])
metric_b.metric("Average confidence", f"{average_confidence:.0%}")
metric_c.metric("Scan time", f"{result['inference_ms']:.0f} ms")
metric_d.metric("Scan mode", selected_profile["label"])

if result["count"] == 0:
    st.warning(
        no_detection_hint(str(mode), confidence),
        icon=":material/search_off:",
    )
else:
    st.success(
        "Detection completed. Review every box before using the result.",
        icon=":material/task_alt:",
    )
    rows = [
        {
            "Detection": index,
            "Confidence": round(detection["confidence"], 3),
            "Apparent severity": estimate_apparent_severity(
                detection["box"], original_image.size
            ),
        }
        for index, detection in enumerate(result["detections"], start=1)
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "Apparent severity uses relative box area only. It does not measure depth or "
        "real-world dimensions."
    )

annotated_bytes = encode_image(result["annotated_image"], "JPEG")
download_col, save_col = st.columns(2)
with download_col:
    st.download_button(
        "Download annotated image",
        data=annotated_bytes,
        file_name="annotated_potholes.jpg",
        mime="image/jpeg",
        icon=":material/download:",
        width="stretch",
    )
with save_col:
    if st.button("Save summary to history", icon=":material/save:", width="stretch"):
        record_id = create_detection_record(
            user_id=USER_ID,
            input_type="image",
            input_name=uploaded_file.name,
            detection_count=result["count"],
            average_confidence=average_confidence if confidences else None,
            apparent_severity=summarize_severity(
                result["detections"], original_image.size
            ),
            processing_ms=result["inference_ms"],
        )
        st.toast(f"Saved history record {record_id}.", icon=":material/save:")
