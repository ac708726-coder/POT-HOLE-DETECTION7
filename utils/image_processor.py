"""Image decoding, annotation, encoding, and metadata helpers."""

from __future__ import annotations

from collections.abc import Iterable
from io import BytesIO
from typing import Any

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError


class ImageProcessingError(RuntimeError):
    """Raised when an image cannot be decoded or encoded."""


def decode_image(data: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(data)) as source:
            oriented = ImageOps.exif_transpose(source)
            return oriented.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageProcessingError("The uploaded image could not be decoded.") from exc


def image_metadata(image: Image.Image) -> dict[str, Any]:
    return {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "megapixels": round((image.width * image.height) / 1_000_000, 2),
    }


def annotate_image(
    image: Image.Image,
    detections: Iterable[dict[str, Any]],
) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    line_width = max(2, round(min(annotated.size) / 250))

    for detection in detections:
        box = [float(value) for value in detection["box"]]
        x1, y1, x2, y2 = box
        class_name = str(detection.get("class_name", "pothole"))
        confidence = float(detection.get("confidence", 0.0))
        track_id = detection.get("track_id")
        id_text = f" #{track_id}" if track_id is not None else ""
        label = f"{class_name}{id_text} {confidence:.2f}"

        draw.rectangle((x1, y1, x2, y2), outline="#dc2626", width=line_width)
        text_box = draw.textbbox((0, 0), label)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_top = max(0, y1 - text_height - 8)
        draw.rectangle(
            (x1, label_top, x1 + text_width + 8, label_top + text_height + 8),
            fill="#dc2626",
        )
        draw.text((x1 + 4, label_top + 4), label, fill="white")

    return annotated


def encode_image(image: Image.Image, image_format: str = "JPEG") -> bytes:
    normalized_format = image_format.upper()
    if normalized_format not in {"JPEG", "PNG"}:
        raise ImageProcessingError("Only JPEG and PNG output formats are supported.")

    buffer = BytesIO()
    try:
        save_options = (
            {"quality": 92, "optimize": True} if normalized_format == "JPEG" else {}
        )
        image.convert("RGB").save(buffer, format=normalized_format, **save_options)
    except OSError as exc:
        raise ImageProcessingError("The annotated image could not be encoded.") from exc
    return buffer.getvalue()
