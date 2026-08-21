from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from utils.validators import (
    ValidationError,
    safe_generated_filename,
    validate_confidence,
    validate_image_upload,
)


def make_image_bytes(image_format: str = "PNG") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (16, 12), "gray").save(buffer, format=image_format)
    return buffer.getvalue()


def test_accepts_valid_png() -> None:
    assert validate_image_upload("road.PNG", make_image_bytes("PNG")) == ".png"


def test_rejects_unsupported_extension() -> None:
    with pytest.raises(ValidationError, match="Unsupported"):
        validate_image_upload("road.gif", make_image_bytes("PNG"))


def test_rejects_extension_content_mismatch() -> None:
    with pytest.raises(ValidationError, match="does not match"):
        validate_image_upload("road.jpg", make_image_bytes("PNG"))


def test_rejects_corrupted_image() -> None:
    with pytest.raises(ValidationError, match="readable image"):
        validate_image_upload("road.png", b"not an image")


@pytest.mark.parametrize("value", [0.0, 0.4, 1.0])
def test_valid_confidence(value: float) -> None:
    assert validate_confidence(value) == value


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_invalid_confidence(value: float) -> None:
    with pytest.raises(ValidationError):
        validate_confidence(value)


def test_generated_filename_does_not_reuse_untrusted_stem() -> None:
    generated = safe_generated_filename("../../private road.JPG")
    assert generated.endswith(".jpg")
    assert "private" not in generated
    assert "/" not in generated and "\\" not in generated
