import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from training.evaluate import save_failure_cases, unmatched_boxes


def test_failure_matching_counts_duplicate_as_false_positive():
    missed, false = unmatched_boxes(
        [[0, 0, 20, 20]], [[0, 0, 20, 20], [1, 1, 20, 20]], [0.9, 0.7], 0.5
    )
    assert missed == []
    assert false == [1]


def test_failure_matching_catches_misses_and_negatives():
    assert unmatched_boxes([[0, 0, 20, 20]], [], [], 0.5) == ([0], [])
    assert unmatched_boxes([], [[0, 0, 20, 20]], [0.9], 0.5) == ([], [0])


def test_failure_export_saves_missed_boxes_and_negative_false_positives(
    tmp_path, monkeypatch
):
    from ultralytics.data import utils

    images, labels = tmp_path / "images/test", tmp_path / "labels/test"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for name, label in [("positive", "0 0.5 0.5 0.5 0.5"), ("negative", "")]:
        Image.new("RGB", (100, 100), (240, 30, 10)).save(images / f"{name}.png")
        (labels / f"{name}.txt").write_text(label)
    monkeypatch.setattr(
        utils, "check_det_dataset", lambda *args, **kwargs: {"test": str(images)}
    )

    class Model:
        def predict(self, **kwargs):
            assert kwargs["source"][0, 0].tolist() == [10, 30, 240]
            # Same prediction on both images: FP on the negative, miss on the positive.
            return [
                SimpleNamespace(
                    boxes=SimpleNamespace(
                        xyxy=SimpleNamespace(cpu=lambda: np.array([[0, 0, 10, 10]])),
                        conf=SimpleNamespace(cpu=lambda: np.array([0.8])),
                    )
                )
            ]

    args = SimpleNamespace(
        data=tmp_path / "data.yaml",
        output=tmp_path / "metrics.json",
        imgsz=640,
        failure_confidence=0.15,
        failure_iou=0.5,
        max_failures=10,
    )
    destination = save_failure_cases(Model(), args, "cpu")
    report = json.loads((destination / "failures.json").read_text())
    assert report["images_checked"] == 2
    assert report["saved"] == 2
    assert len(list(destination.glob("*.jpg"))) == 2
    positive = next(
        case for case in report["cases"] if "positive.png" in case["source"]
    )
    assert positive["missed_truth_ids"] == [0]
    (labels / "positive.txt").unlink()
    with pytest.raises(ValueError, match="Missing held-out label"):
        save_failure_cases(Model(), args, "cpu")
