from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from utils.tracker import PotholeTracker, intersection_over_union


def test_intersection_over_union() -> None:
    assert intersection_over_union([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert intersection_over_union([0, 0, 5, 5], [6, 6, 10, 10]) == 0.0


def test_tracker_reuses_identifier_for_overlapping_detection() -> None:
    tracker = PotholeTracker(iou_threshold=0.2)
    first = tracker.update([{"box": [10, 10, 30, 30], "confidence": 0.8}])
    second = tracker.update([{"box": [12, 11, 32, 31], "confidence": 0.9}])
    assert first[0]["track_id"] == second[0]["track_id"]
    assert tracker.total_tracks_created == 1


@pytest.mark.skipif(
    importlib.util.find_spec("cv2") is None, reason="OpenCV not installed"
)
def test_processes_short_video_with_fake_predictions(tmp_path: Path) -> None:
    import cv2

    from utils.video_processor import process_video, read_video_info

    input_path = tmp_path / "input.avi"
    output_path = tmp_path / "output.mp4"
    writer = cv2.VideoWriter(
        str(input_path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (64, 48)
    )
    if not writer.isOpened():
        pytest.skip("Test video codec unavailable")
    for _ in range(5):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()

    def fake_predict(image, confidence, iou_threshold, inference_profile):
        assert inference_profile == "fast"
        return {
            "detections": [
                {
                    "box": [10, 10, 35, 30],
                    "confidence": 0.9,
                    "class_name": "pothole",
                }
            ]
        }

    progress_values: list[float] = []
    result = process_video(
        input_path,
        output_path,
        prediction_function=fake_predict,
        progress_callback=progress_values.append,
    )
    assert result["processed_frames"] == 5
    assert result["unique_count_estimate"] == 1
    assert output_path.is_file() and output_path.stat().st_size > 0
    assert read_video_info(output_path)["frame_count"] == 5
    assert progress_values[-1] == 1.0


@pytest.mark.skipif(
    importlib.util.find_spec("cv2") is None, reason="OpenCV not installed"
)
def test_default_fast_video_path_batches_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cv2

    from utils import video_processor

    input_path = tmp_path / "batch_input.avi"
    output_path = tmp_path / "batch_output.mp4"
    writer = cv2.VideoWriter(
        str(input_path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (64, 48)
    )
    if not writer.isOpened():
        pytest.skip("Test video codec unavailable")
    for _ in range(8):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()

    batch_lengths: list[int] = []

    def fake_batch(images, **kwargs):
        batch_lengths.append(len(images))
        return [
            {
                "detections": [
                    {
                        "box": [10, 10, 35, 30],
                        "confidence": 0.9,
                        "class_name": "pothole",
                    }
                ]
            }
            for _ in images
        ]

    monkeypatch.setattr(video_processor, "predict_images", fake_batch)
    monkeypatch.setattr(
        video_processor,
        "inference_runtime_details",
        lambda: {
            "device": "cuda",
            "runtime_label": "GPU accelerated",
            "precision": "FP16",
            "video_batch_size": 4,
        },
    )

    result = video_processor.process_video(input_path, output_path)
    assert batch_lengths == [4, 4]
    assert result["video_batch_size"] == 4
    assert result["processed_frames"] == 8


@pytest.mark.skipif(
    importlib.util.find_spec("cv2") is None, reason="OpenCV not installed"
)
def test_balanced_video_path_batches_at_reduced_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Augmented profiles must still batch, at a halved size for memory headroom."""

    import cv2

    from utils import video_processor

    input_path = tmp_path / "balanced_input.avi"
    output_path = tmp_path / "balanced_output.mp4"
    writer = cv2.VideoWriter(
        str(input_path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (64, 48)
    )
    if not writer.isOpened():
        pytest.skip("Test video codec unavailable")
    for _ in range(8):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()

    batch_lengths: list[int] = []

    def fake_batch(images, **kwargs):
        assert kwargs["inference_profile"] == "balanced"
        batch_lengths.append(len(images))
        return [{"detections": []} for _ in images]

    monkeypatch.setattr(video_processor, "predict_images", fake_batch)
    monkeypatch.setattr(
        video_processor,
        "inference_runtime_details",
        lambda: {
            "device": "cuda",
            "runtime_label": "GPU accelerated",
            "precision": "FP16",
            "video_batch_size": 4,
        },
    )

    result = video_processor.process_video(
        input_path, output_path, inference_profile="balanced"
    )
    assert result["video_batch_size"] == 2
    assert batch_lengths == [2, 2, 2, 2]
    assert result["processed_frames"] == 8
