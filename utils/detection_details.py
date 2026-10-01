"""Plain-language hints and measurements for reviewing detection results."""

from __future__ import annotations

from config import MIN_SCAN_CONFIDENCE


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
