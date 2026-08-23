"""Incremental OpenCV video processing with approximate object tracking."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

from config import DEFAULT_CONFIDENCE, DEFAULT_IOU_THRESHOLD, MAX_VIDEO_DURATION_SECONDS
from utils.detector import inference_runtime_details, predict_image, predict_images
from utils.image_processor import annotate_image
from utils.tracker import PotholeTracker


class VideoProcessingError(RuntimeError):
    """Raised when a video cannot be read, processed, or written."""


def _opencv() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise VideoProcessingError(
            "OpenCV is not installed. Install the packages from requirements.txt."
        ) from exc
    return cv2


def read_video_info(path: str | Path) -> dict[str, Any]:
    cv2 = _opencv()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoProcessingError("The uploaded file is not a readable video.")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0.0
        if width <= 0 or height <= 0 or fps <= 0:
            raise VideoProcessingError("The video metadata is invalid or unsupported.")
        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_seconds": duration,
        }
    finally:
        capture.release()


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    confidence: float = DEFAULT_CONFIDENCE,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    frame_skip: int = 1,
    progress_callback: Callable[[float], None] | None = None,
    prediction_function: Callable[..., dict[str, Any]] = predict_image,
    inference_profile: str = "fast",
) -> dict[str, Any]:
    cv2 = _opencv()
    info = read_video_info(input_path)
    if info["duration_seconds"] > MAX_VIDEO_DURATION_SECONDS:
        raise VideoProcessingError(
            f"Videos must be {MAX_VIDEO_DURATION_SECONDS // 60} minutes or shorter."
        )
    if frame_skip < 1:
        raise VideoProcessingError("Frame skip must be at least 1.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(input_path))
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        info["fps"],
        (info["width"], info["height"]),
    )
    if not capture.isOpened() or not writer.isOpened():
        capture.release()
        writer.release()
        raise VideoProcessingError(
            "The video input or output codec could not be opened."
        )

    tracker = PotholeTracker()
    processed_frames = 0
    frames_read = 0
    inference_frames = 0
    last_tracked: list[dict[str, Any]] = []
    runtime = inference_runtime_details()
    device_batch_size = int(runtime["video_batch_size"])
    if prediction_function is not predict_image:
        # A caller-supplied prediction function has no batched counterpart.
        batch_size = 1
    elif inference_profile == "fast":
        batch_size = device_batch_size
    else:
        # Augmented profiles run test-time augmentation internally, which multiplies
        # activation memory per image. Halve the batch so these modes still benefit
        # from batching without risking out-of-memory on small GPUs.
        batch_size = max(1, device_batch_size // 2)
    pending_frames: list[tuple[int, np.ndarray]] = []
    pending_inference_frames = 0
    started = perf_counter()

    def flush_pending_frames() -> None:
        nonlocal processed_frames, inference_frames, last_tracked
        nonlocal pending_inference_frames
        if not pending_frames:
            return

        analysis_images = [
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for frame_index, frame in pending_frames
            if frame_index % frame_skip == 0
        ]
        if prediction_function is predict_image and len(analysis_images) > 1:
            predictions = predict_images(
                analysis_images,
                confidence=confidence,
                iou_threshold=iou_threshold,
                inference_profile=inference_profile,
                annotate=False,
            )
        else:
            predictions = [
                prediction_function(
                    image,
                    confidence=confidence,
                    iou_threshold=iou_threshold,
                    inference_profile=inference_profile,
                )
                for image in analysis_images
            ]

        prediction_index = 0
        for frame_index, frame in pending_frames:
            if frame_index % frame_skip == 0:
                last_tracked = tracker.update(
                    predictions[prediction_index]["detections"]
                )
                prediction_index += 1
                inference_frames += 1

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated = annotate_image(Image.fromarray(rgb_frame), last_tracked)
            writer.write(cv2.cvtColor(np.asarray(annotated), cv2.COLOR_RGB2BGR))
            processed_frames += 1
            if progress_callback and info["frame_count"] > 0:
                progress_callback(min(1.0, processed_frames / info["frame_count"]))

        pending_frames.clear()
        pending_inference_frames = 0

    try:
        while True:
            readable, frame = capture.read()
            if not readable:
                break

            pending_frames.append((frames_read, frame))
            if frames_read % frame_skip == 0:
                pending_inference_frames += 1
            frames_read += 1
            if pending_inference_frames >= batch_size:
                flush_pending_frames()
        flush_pending_frames()
    except Exception as exc:
        if isinstance(exc, VideoProcessingError):
            raise
        raise VideoProcessingError(f"Video processing failed: {exc}") from exc
    finally:
        capture.release()
        writer.release()

    if processed_frames == 0:
        raise VideoProcessingError("The video did not contain any readable frames.")
    if progress_callback:
        progress_callback(1.0)
    elapsed = perf_counter() - started
    return {
        **info,
        "processed_frames": processed_frames,
        "inference_frames": inference_frames,
        "unique_count_estimate": tracker.total_tracks_created,
        "processing_seconds": elapsed,
        "processing_fps": processed_frames / elapsed if elapsed > 0 else 0.0,
        "inference_profile": inference_profile,
        "video_batch_size": batch_size,
        "runtime": runtime,
        "output_path": str(output),
    }
