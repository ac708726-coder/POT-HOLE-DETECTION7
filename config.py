"""Central configuration for the pothole detection application."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.getenv("POTHOLE_MODEL_PATH", str(BASE_DIR / "models" / "best.pt"))
).expanduser()
DATABASE_PATH = Path(
    os.getenv("POTHOLE_DATABASE_PATH", str(BASE_DIR / "database" / "potholes.db"))
).expanduser()

OUTPUTS_DIR = BASE_DIR / "outputs"
IMAGE_OUTPUT_DIR = OUTPUTS_DIR / "images"
VIDEO_OUTPUT_DIR = OUTPUTS_DIR / "videos"
REPORT_OUTPUT_DIR = OUTPUTS_DIR / "reports"
METRICS_OUTPUT_DIR = OUTPUTS_DIR / "metrics"

DEFAULT_CONFIDENCE = 0.35
DEFAULT_IOU_THRESHOLD = 0.50
MIN_SCAN_CONFIDENCE = 0.15
# Highest-resolution Thorough pass; tune in multiples of the YOLO stride (32).
THOROUGH_IMAGE_SIZE = 1280
MAX_IMAGE_SIZE_MB = 10
MAX_VIDEO_SIZE_MB = 200
MAX_VIDEO_DURATION_SECONDS = 300
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}


def ensure_runtime_directories() -> None:
    """Create directories used for generated output and local history."""

    for directory in (
        IMAGE_OUTPUT_DIR,
        VIDEO_OUTPUT_DIR,
        REPORT_OUTPUT_DIR,
        METRICS_OUTPUT_DIR,
        DATABASE_PATH.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
