from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from PIL import Image

from utils.detector import (
    DetectorError,
    ModelNotFoundError,
    _merge_detections,
    inference_profile_details,
    load_model,
    predict_image,
    predict_images,
)
from utils.validators import ValidationError


class FakeBoxes:
    def __init__(self) -> None:
        self.xyxy = np.array([[5, 6, 50, 60], [10, 12, 20, 22]], dtype=float)
        self.conf = np.array([0.91, 0.20], dtype=float)
        self.cls = np.array([0, 0], dtype=float)


class FakeResult:
    boxes = FakeBoxes()
    names: ClassVar[dict[int, str]] = {0: "pothole"}


class FakeModel:
    def predict(self, **kwargs):
        assert kwargs["conf"] == 0.40
        assert kwargs["iou"] == 0.45
        assert kwargs["source"].shape == (80, 100, 3)
        return [FakeResult()]


class FailingModel:
    def predict(self, **kwargs):
        raise RuntimeError("backend unavailable")


def test_prediction_filters_and_counts_results() -> None:
    result = predict_image(
        Image.new("RGB", (100, 80)),
        confidence=0.40,
        iou_threshold=0.45,
        model=FakeModel(),
    )
    assert result["count"] == 1
    assert result["detections"][0]["class_name"] == "pothole"
    assert result["detections"][0]["confidence"] == pytest.approx(0.91)
    assert result["annotated_image"].size == (100, 80)
    assert result["inference_ms"] >= 0


def test_prediction_error_is_actionable() -> None:
    with pytest.raises(DetectorError, match="backend unavailable"):
        predict_image(Image.new("RGB", (10, 10)), model=FailingModel())


def test_balanced_profile_enables_augmented_inference() -> None:
    class ProfileModel:
        def predict(self, **kwargs):
            assert kwargs["imgsz"] == 640
            assert kwargs["augment"] is True
            return [FakeResult()]

    result = predict_image(
        Image.new("RGB", (100, 80)),
        confidence=0.40,
        model=ProfileModel(),
        inference_profile="balanced",
    )
    assert result["inference_profile"] == "balanced"
    assert result["inference_passes"] == 1


def test_multi_pass_merge_removes_overlapping_duplicate() -> None:
    detections = [
        {"class_id": 0, "confidence": 0.9, "box": [0, 0, 20, 20]},
        {"class_id": 0, "confidence": 0.8, "box": [1, 1, 20, 20]},
        {"class_id": 0, "confidence": 0.7, "box": [30, 30, 40, 40]},
    ]
    merged = _merge_detections(detections, iou_threshold=0.45)
    assert [item["confidence"] for item in merged] == [0.9, 0.7]


def test_merge_fuses_cluster_into_confidence_weighted_box() -> None:
    detections = [
        {"class_id": 0, "confidence": 0.75, "box": [0.0, 0.0, 10.0, 10.0]},
        {"class_id": 0, "confidence": 0.25, "box": [2.0, 0.0, 12.0, 10.0]},
    ]
    merged = _merge_detections(detections, iou_threshold=0.45)

    assert len(merged) == 1
    assert merged[0]["confidence"] == pytest.approx(0.75)
    assert merged[0]["box"][0] == pytest.approx(0.5)
    assert merged[0]["box"][2] == pytest.approx(10.5)


def test_merge_keeps_distinct_classes_separate() -> None:
    detections = [
        {"class_id": 0, "confidence": 0.9, "box": [0.0, 0.0, 10.0, 10.0]},
        {"class_id": 1, "confidence": 0.8, "box": [0.0, 0.0, 10.0, 10.0]},
    ]
    merged = _merge_detections(detections, iou_threshold=0.45)

    assert sorted(item["class_id"] for item in merged) == [0, 1]


def test_unknown_inference_profile_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Inference profile"):
        inference_profile_details("unknown")


def test_batch_prediction_preserves_result_order() -> None:
    class BatchModel:
        def predict(self, **kwargs):
            assert len(kwargs["source"]) == 2
            return [FakeResult(), FakeResult()]

    results = predict_images(
        [Image.new("RGB", (100, 80)), Image.new("RGB", (100, 80))],
        confidence=0.40,
        model=BatchModel(),
    )
    assert len(results) == 2
    assert [result["count"] for result in results] == [1, 1]
    assert all(result["annotated_image"].size == (100, 80) for result in results)


def test_warm_up_runs_once_and_is_reported() -> None:
    from utils import detector

    class CountingModel:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, **kwargs):
            self.calls += 1
            assert kwargs["source"].shape == (640, 640, 3)
            return [FakeResult()]

    monkeypatched = CountingModel()
    detector._warmed_up = False
    try:
        assert detector.warm_up_model(monkeypatched) is True
        assert detector.warm_up_model(monkeypatched) is True
        assert monkeypatched.calls == 1
    finally:
        detector._warmed_up = False


def test_warm_up_failure_is_swallowed_and_retryable() -> None:
    from utils import detector

    class WarmableModel:
        def predict(self, **kwargs):
            return [FakeResult()]

    detector._warmed_up = False
    try:
        # A broken checkpoint must not raise out of warm-up...
        assert detector.warm_up_model(FailingModel()) is False
        # ...and must not be cached, so a later working model still warms up.
        assert detector.warm_up_model(WarmableModel()) is True
    finally:
        detector._warmed_up = False


def test_missing_model_has_clear_error(tmp_path: Path) -> None:
    load_model.cache_clear()
    with pytest.raises(ModelNotFoundError, match="best.pt"):
        load_model(tmp_path / "best.pt")


def test_model_reloads_when_checkpoint_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created_models: list[object] = []

    def fake_yolo(path: str) -> object:
        del path
        model = object()
        created_models.append(model)
        return model

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=fake_yolo))
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"first")
    load_model.cache_clear()

    first = load_model(weights)
    cached = load_model(weights)
    weights.write_bytes(b"replacement checkpoint")
    replacement = load_model(weights)

    assert first is cached
    assert replacement is not first
    assert len(created_models) == 2


def test_loaded_model_fuses_supported_layers_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FusibleModel:
        def __init__(self) -> None:
            self.fuse_calls = 0

        def fuse(self) -> FusibleModel:
            self.fuse_calls += 1
            return self

    model = FusibleModel()
    monkeypatch.setitem(
        sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda path: model)
    )
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"checkpoint")
    load_model.cache_clear()

    assert load_model(weights) is model
    assert load_model(weights) is model
    assert model.fuse_calls == 1
