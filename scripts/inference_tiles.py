"""Benchmark-only crop experiment; not enabled in the application's scan modes."""

from __future__ import annotations

from typing import Any


def tile_windows(
    size: tuple[int, int], fraction: float = 0.65, minimum_edge: int = 256
) -> list[tuple[int, int, int, int]]:
    """Cover the image with at most four overlapping views; skip tiny images."""
    if not 0.5 < fraction < 1 or minimum_edge <= 0:
        raise ValueError("Use a tile fraction in (0.5, 1) and a positive minimum edge.")
    width, height = size
    if min(size) < minimum_edge * 2:
        return []
    crop_width = max(minimum_edge, round(width * fraction))
    crop_height = max(minimum_edge, round(height * fraction))
    return [
        (left, top, left + crop_width, top + crop_height)
        for top in (0, height - crop_height)
        for left in (0, width - crop_width)
    ]


def remap_tile_detections(
    detections: list[dict[str, Any]],
    window: tuple[int, int, int, int],
    image_size: tuple[int, int],
    edge_margin: float = 2.0,
) -> list[dict[str, Any]]:
    """Translate crop-local boxes, rejecting fragments cut at internal seams.

    Full-image predictions remain available for large seam-spanning potholes.
    Original image edges are not rejected. Do not change confidence scores.
    """
    left, top, right, bottom = window
    width, height = image_size
    crop_width, crop_height = right - left, bottom - top
    mapped = []
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        if (
            (left > 0 and x1 <= edge_margin)
            or (top > 0 and y1 <= edge_margin)
            or (right < width and x2 >= crop_width - 1 - edge_margin)
            or (bottom < height and y2 >= crop_height - 1 - edge_margin)
        ):
            continue
        mapped.append({**detection, "box": [x1 + left, y1 + top, x2 + left, y2 + top]})
    return mapped
