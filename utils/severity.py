"""Approximate visual severity based only on bounding-box area."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def estimate_apparent_severity(box: list[float], image_size: tuple[int, int]) -> str:
    """Return low, medium, or high apparent severity; this is not depth."""

    x1, y1, x2, y2 = box
    image_width, image_height = image_size
    image_area = max(1.0, float(image_width * image_height))
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    relative_area = box_area / image_area
    if relative_area < 0.02:
        return "low"
    if relative_area < 0.08:
        return "medium"
    return "high"


def summarize_severity(
    detections: Iterable[dict[str, Any]], image_size: tuple[int, int]
) -> str | None:
    ranking = {"low": 1, "medium": 2, "high": 3}
    severities = [
        estimate_apparent_severity(detection["box"], image_size)
        for detection in detections
    ]
    return max(severities, key=ranking.get) if severities else None
