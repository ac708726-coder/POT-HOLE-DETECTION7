from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from scripts.benchmark_detection_recall import predict_tiled_candidate
from scripts.inference_tiles import remap_tile_detections, tile_windows
from utils import detector


def detection(box, confidence=0.8):
    return {
        "class_id": 0,
        "class_name": "pothole",
        "confidence": confidence,
        "box": box,
    }


def test_tiles_cover_all_corners_with_overlap_and_bounded_count():
    windows = tile_windows((1000, 800))
    assert len(windows) == 4
    assert windows == [
        (0, 0, 650, 520),
        (350, 0, 1000, 520),
        (0, 280, 650, 800),
        (350, 280, 1000, 800),
    ]
    assert tile_windows((400, 800)) == []
    for fraction in (0.4, 1.0):
        with pytest.raises(ValueError):
            tile_windows((1000, 800), fraction)


def test_remap_preserves_confidence_and_original_data():
    original = detection([20, 30, 100, 120])
    mapped = remap_tile_detections([original], (350, 280, 1000, 800), (1000, 800))
    assert mapped[0]["box"] == [370, 310, 450, 400]
    assert mapped[0]["confidence"] == 0.8
    assert original["box"] == [20, 30, 100, 120]


def test_cut_fragments_are_rejected_but_original_image_edges_survive():
    boxes = [detection([10, 20, 648, 100]), detection([0, 0, 50, 40])]
    mapped = remap_tile_detections(boxes, (0, 0, 650, 520), (1000, 800))
    assert [item["box"] for item in mapped] == [[0, 0, 50, 40]]
    assert (
        remap_tile_detections(
            [detection([0, 10, 80, 70])], (350, 0, 1000, 520), (1000, 800)
        )
        == []
    )


def test_tile_prediction_recovers_an_object_missed_in_full_frame_and_deduplicates(
    monkeypatch,
):
    class CropModel:
        def predict(self, **kwargs):
            sources = kwargs["source"]
            if not isinstance(sources, list):
                sources = [sources]
            results = []
            for source in sources:
                # Use a visible white marker to derive crop-local coordinates.
                yy, xx = np.where(source[:, :, 0] > 0)
                coordinates = (
                    [[xx.min(), yy.min(), xx.max(), yy.max()]]
                    if source.shape[:2] == (650, 650)
                    else []
                )
                count = len(coordinates)
                results.append(
                    SimpleNamespace(
                        boxes=SimpleNamespace(
                            xyxy=np.array(coordinates).reshape(-1, 4),
                            conf=np.full(count, 0.8),
                            cls=np.zeros(count),
                        ),
                        names={0: "pothole"},
                    )
                )
            return results

    monkeypatch.setattr(
        detector, "inference_runtime_details", lambda: {"video_batch_size": 1}
    )
    source = np.zeros((1000, 1000, 3), dtype=np.uint8)
    source[450:551, 450:551] = 255
    result = predict_tiled_candidate(Image.fromarray(source), model=CropModel())
    assert result["inference_tiles"] == 4
    assert result["count"] == 1
    assert result["detections"][0]["box"] == [450, 450, 550, 550]


def test_bad_tile_backend_result_is_not_silently_ignored():
    class ShortModel:
        def predict(self, **kwargs):
            return []

    with pytest.raises(detector.DetectorError, match="tile results"):
        predict_tiled_candidate(Image.new("RGB", (1000, 1000)), model=ShortModel())


def test_fast_explicitly_resets_resolution_and_augmentation():
    assert detector.INFERENCE_PROFILES["fast"]["passes"] == (
        {"imgsz": 640, "augment": False},
    )


def test_invalid_or_degenerate_boxes_are_removed():
    result = SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=np.array(
                [[0, 0, 0, 1], [np.nan, 0, 2, 3], [0, 0, 5, 5], [2, 2, 5, 5]]
            ),
            conf=np.array([0.9, 0.9, np.nan, 0.8]),
            cls=np.zeros(4),
        ),
        names={0: "pothole"},
    )
    assert detector._extract_detections(result, 0.15, (100, 100)) == [
        detection([2, 2, 5, 5])
    ]
