"""Validation helpers for uploaded media and inference settings."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from uuid import uuid4

from PIL import Image, UnidentifiedImageError

from config import (
    MAX_IMAGE_SIZE_MB,
    MAX_VIDEO_SIZE_MB,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_VIDEO_EXTENSIONS,
)


class ValidationError(ValueError):
    """Raised when user-provided input fails validation."""


def validate_extension(filename: str, supported_extensions: set[str]) -> str:
    if not filename or not Path(filename).suffix:
        raise ValidationError("The uploaded file must have a filename extension.")

    extension = Path(filename).suffix.lower()
    if extension not in supported_extensions:
        allowed = ", ".join(sorted(supported_extensions))
        raise ValidationError(f"Unsupported file type. Allowed extensions: {allowed}.")
    return extension


def validate_file_size(size_bytes: int, maximum_mb: int) -> None:
    if size_bytes <= 0:
        raise ValidationError("The uploaded file is empty.")
    if size_bytes > maximum_mb * 1024 * 1024:
        raise ValidationError(f"The uploaded file must be {maximum_mb} MB or smaller.")


def validate_image_upload(filename: str, data: bytes) -> str:
    extension = validate_extension(filename, SUPPORTED_IMAGE_EXTENSIONS)
    validate_file_size(len(data), MAX_IMAGE_SIZE_MB)

    allowed_formats = {"JPEG", "PNG"}
    try:
        with Image.open(BytesIO(data)) as image:
            detected_format = (image.format or "").upper()
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValidationError("The uploaded file is not a readable image.") from exc

    if detected_format not in allowed_formats:
        raise ValidationError("The image content must be JPEG or PNG.")
    if extension in {".jpg", ".jpeg"} and detected_format != "JPEG":
        raise ValidationError(
            "The filename extension does not match the image content."
        )
    if extension == ".png" and detected_format != "PNG":
        raise ValidationError(
            "The filename extension does not match the image content."
        )
    return extension


def validate_video_upload(filename: str, data: bytes) -> str:
    extension = validate_extension(filename, SUPPORTED_VIDEO_EXTENSIONS)
    validate_file_size(len(data), MAX_VIDEO_SIZE_MB)
    return extension


def validate_confidence(value: float) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Confidence must be a number between 0 and 1.") from exc
    if not 0.0 <= confidence <= 1.0:
        raise ValidationError("Confidence must be between 0 and 1.")
    return confidence


def safe_generated_filename(original_name: str, suffix: str | None = None) -> str:
    """Return an untrusted-name-independent filename with a safe extension."""

    original_extension = Path(original_name).suffix.lower()
    extension = suffix if suffix is not None else original_extension
    if extension and not extension.startswith("."):
        extension = f".{extension}"
    return f"{uuid4().hex}{extension}"
