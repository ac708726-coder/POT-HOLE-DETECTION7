from types import SimpleNamespace

import numpy as np
from PIL import Image

from utils.detector import predict_image, predict_images
from utils.image_processor import image_to_bgr


def empty_result():
    return SimpleNamespace(
        boxes=SimpleNamespace(xyxy=np.empty((0, 4)), conf=np.empty(0), cls=np.empty(0)),
        names={0: "pothole"},
    )


def test_yolo_conversion_is_bgr_contiguous_uint8_and_does_not_modify_source():
    image = Image.new("RGB", (10, 8), (240, 30, 10))
    source = image_to_bgr(image)
    assert source[0, 0].tolist() == [10, 30, 240]
    assert source.dtype == np.uint8
    assert source.flags.c_contiguous
    assert image.getpixel((0, 0)) == (240, 30, 10)


def test_all_single_image_modes_receive_bgr_not_rgb():
    class CheckingModel:
        def predict(self, **kwargs):
            assert kwargs["source"][0, 0].tolist() == [10, 30, 240]
            return [empty_result()]

    image = Image.new("RGB", (10, 8), (240, 30, 10))
    for mode in ("fast", "balanced", "thorough"):
        result = predict_image(image, model=CheckingModel(), inference_profile=mode)
        assert result["annotated_image"].getpixel((0, 0)) == (240, 30, 10)


def test_batched_video_path_receives_bgr_but_keeps_rgb_annotations():
    class CheckingModel:
        def predict(self, **kwargs):
            assert kwargs["source"][0][0, 0].tolist() == [10, 30, 240]
            assert kwargs["source"][1][0, 0].tolist() == [240, 30, 10]
            return [empty_result(), empty_result()]

    images = [
        Image.new("RGB", (10, 8), colour) for colour in ((240, 30, 10), (10, 30, 240))
    ]
    results = predict_images(images, model=CheckingModel())
    assert results[0]["annotated_image"].getpixel((0, 0)) == (240, 30, 10)
    assert results[1]["annotated_image"].getpixel((0, 0)) == (10, 30, 240)


def test_rgb_numpy_application_input_is_converted_once():
    class CheckingModel:
        def predict(self, **kwargs):
            assert kwargs["source"][0, 0].tolist() == [10, 30, 240]
            return [empty_result()]

    image = np.full((8, 10, 3), [240, 30, 10], dtype=np.uint8)
    predict_image(image, model=CheckingModel())
