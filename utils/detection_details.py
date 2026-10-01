"""Plain-language hints and measurements for reviewing detection results."""

from __future__ import annotations

from config import MIN_SCAN_CONFIDENCE


def detection_rows(detections: list[dict], image_size: tuple[int, int]) -> list[dict]:
    """Box dimensions refer to the original image, not the resized YOLO input."""
    image_area = image_size[0] * image_size[1]
    rows = []
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection["box"]
        width, height = max(0.0, x2 - x1), max(0.0, y2 - y1)
        rows.append(
            {
                "ID": index,
                "Confidence": float(detection["confidence"]),
                "Width (px)": round(width, 1),
                "Height (px)": round(height, 1),
                "Image area (%)": (
                    round(100 * width * height / image_area, 2) if image_area else 0.0
                ),
            }
        )
    return rows


def no_detection_hint(mode: str, confidence: float) -> str:
    prefix = "No candidate met this threshold. This does not prove the road has no potholes. "
    if mode == "fast":
        return prefix + "Try Balanced or Thorough mode for a more detailed scan."
    if mode == "balanced":
        return prefix + "Try Thorough mode for a more detailed scan."
    if mode == "thorough":
        if confidence > MIN_SCAN_CONFIDENCE:
            return (
                prefix + "Try lowering confidence, or use a closer, better-lit photo."
            )
        return (
            prefix
            + "Use a closer, better-lit photo; confidence is already at its minimum."
        )
    raise ValueError(f"Unknown scan mode: {mode}")
