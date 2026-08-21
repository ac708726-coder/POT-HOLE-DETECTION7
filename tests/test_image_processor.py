from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from utils.image_processor import (
    ImageProcessingError,
    annotate_image,
    decode_image,
    encode_image,
    image_metadata,
)


def test_decode_converts_to_rgb_and_keeps_dimensions() -> None:
    buffer = BytesIO()
    Image.new("L", (24, 18), 128).save(buffer, format="PNG")
    decoded = decode_image(buffer.getvalue())
    assert decoded.mode == "RGB"
    assert decoded.size == (24, 18)


def test_annotation_keeps_dimensions_and_changes_pixels() -> None:
    image = Image.new("RGB", (100, 80), "white")
    annotated = annotate_image(
        image,
        [{"box": [10, 15, 60, 55], "class_name": "pothole", "confidence": 0.91}],
    )
    assert annotated.size == image.size
    assert annotated.tobytes() != image.tobytes()


def test_encoded_jpeg_can_be_reopened() -> None:
    data = encode_image(Image.new("RGB", (20, 10), "black"), "JPEG")
    with Image.open(BytesIO(data)) as decoded:
        assert decoded.format == "JPEG"
        assert decoded.size == (20, 10)


def test_unsupported_output_format_is_rejected() -> None:
    with pytest.raises(ImageProcessingError):
        encode_image(Image.new("RGB", (2, 2)), "GIF")


def test_metadata() -> None:
    assert image_metadata(Image.new("RGB", (1000, 500)))["megapixels"] == 0.5
