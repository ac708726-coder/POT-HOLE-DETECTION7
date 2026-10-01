import pytest

from scripts.benchmark_detection_recall import (
    ground_truth,
    operating_metrics,
    validation_sample,
)


def test_score_duplicate_predictions_as_false_positives_and_count_misses():
    rows = [
        {
            "truth": [[0, 0, 10, 10]],
            "detections": [
                {"confidence": 0.9, "box": [0, 0, 10, 10]},
                {"confidence": 0.8, "box": [0, 0, 10, 10]},
            ],
        },
        {"truth": [[0, 0, 10, 10]], "detections": []},
        {"truth": [], "detections": [{"confidence": 0.7, "box": [20, 20, 30, 30]}]},
    ]
    result = operating_metrics(rows, 0.35)
    assert result["true_positives"] == 1
    assert result["false_positives"] == 2
    assert result["false_negatives"] == 1
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["recall"] == 0.5
    assert operating_metrics(rows, 0.85)["false_positives"] == 0


def test_sample_is_reproducible_and_keeps_negatives(tmp_path):
    images = tmp_path / "images" / "val"
    labels = tmp_path / "labels" / "val"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for name, label in (("positive", "0 .5 .5 .2 .2"), ("negative", "")):
        (images / f"{name}.jpg").touch()
        (labels / f"{name}.txt").write_text(label, encoding="utf-8")
    assert len(validation_sample(tmp_path, 1, 1, 42)) == 2
    assert validation_sample(tmp_path, 1, 1, 42) == validation_sample(
        tmp_path, 1, 1, 42
    )
    with pytest.raises(ValueError, match="available"):
        validation_sample(tmp_path, 2, 1, 42)


def test_invalid_labels_fail_rather_than_inflate_metrics(tmp_path):
    label = tmp_path / "invalid.txt"
    for content in ("1 .5 .5 .2 .2", "0 nan .5 .2 .2", "0 .5 .5 0 .2"):
        label.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError):
            ground_truth(label, (100, 100))
