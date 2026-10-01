"""Integration smoke test: use the actual shipped pothole checkpoint."""

import pytest
from PIL import Image

from utils.detector import predict_image


@pytest.mark.parametrize("mode", ["fast", "balanced", "thorough"])
def test_blank_gray_image_has_no_pothole_detections(mode):
    result = predict_image(
        Image.new("RGB", (960, 1280), (128, 128, 128)),
        confidence=0.15,
        inference_profile=mode,
    )
    assert result["count"] == 0
    assert result["detections"] == []
