"""YOLO model loading and stable application-level inference results."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

from config import DEFAULT_CONFIDENCE, DEFAULT_IOU_THRESHOLD, MODEL_PATH
from utils.image_processor import annotate_image
from utils.validators import ValidationError, validate_confidence


class DetectorError(RuntimeError):
    """Base error for model loading or prediction failures."""


class ModelNotFoundError(DetectorError):
    """Raised when the configured trained weights do not exist."""


INFERENCE_PROFILES: dict[str, dict[str, Any]] = {
    "fast": {
        "label": "Fast",
        "description": "One standard pass. Best for video and quick checks.",
        "passes": ({},),
    },
    "balanced": {
        "label": "Balanced",
        "description": "Augmented inference for better recovery in difficult lighting.",
        "passes": ({"imgsz": 640, "augment": True},),
    },
    "thorough": {
        "label": "Thorough",
        "description": "Two augmented scales, merged to reduce duplicate boxes.",
        "passes": (
            {"imgsz": 640, "augment": True},
            {"imgsz": 960, "augment": True},
        ),
    },
}


@lru_cache(maxsize=1)
def _runtime_predict_options() -> dict[str, Any]:
    """Select safe PyTorch inference settings for the available hardware."""

    try:
        import torch
    except ImportError:
        return {"device": "cpu"}

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        return {"device": 0, "quantize": 16}
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return {"device": "mps"}
    return {"device": "cpu"}


def inference_runtime_details() -> dict[str, Any]:
    """Describe the selected runtime without exposing backend implementation objects."""

    options = _runtime_predict_options()
    device = options["device"]
    if device == 0:
        return {
            "device": "cuda",
            "runtime_label": "GPU accelerated",
            "precision": "FP16",
            "video_batch_size": 4,
        }
    if device == "mps":
        return {
            "device": "mps",
            "runtime_label": "GPU accelerated",
            "precision": "FP32",
            "video_batch_size": 2,
        }
    return {
        "device": "cpu",
        "runtime_label": "CPU mode",
        "precision": "FP32",
        "video_batch_size": 1,
    }


@lru_cache(maxsize=1)
def _load_model_cached(
    resolved_weights_path: str, file_signature: tuple[int, int]
) -> Any:
    """Load a checkpoint identified by its path and filesystem signature."""

    del file_signature
    path = Path(resolved_weights_path)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise DetectorError(
            "Ultralytics is not installed. Install the packages from requirements.txt."
        ) from exc

    try:
        model = YOLO(str(path))
        fuse = getattr(model, "fuse", None)
        if callable(fuse):
            fused_model = fuse()
            if fused_model is not None:
                model = fused_model
        return model
    except Exception as exc:  # Ultralytics raises several backend-specific error types.
        raise DetectorError(
            f"The model at '{path}' could not be loaded: {exc}"
        ) from exc


def load_model(weights_path: str | Path = MODEL_PATH) -> Any:
    """Load and cache a model, reloading automatically when its checkpoint changes."""

    path = Path(weights_path).expanduser().resolve()
    if not path.is_file():
        raise ModelNotFoundError(
            f"Trained weights were not found at '{path}'. Place best.pt there or set "
            "POTHOLE_MODEL_PATH."
        )
    stat = path.stat()
    if stat.st_size == 0:
        raise ModelNotFoundError(
            f"Trained weights were not found at '{path}'. Place best.pt there or set "
            "POTHOLE_MODEL_PATH."
        )
    return _load_model_cached(str(path), (stat.st_mtime_ns, stat.st_size))


load_model.cache_clear = _load_model_cached.cache_clear  # type: ignore[attr-defined]


def _as_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, "pothole"))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return "pothole"


def _extract_detections(
    result: Any, confidence: float, image_size: tuple[int, int]
) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    coordinates = _as_numpy(getattr(boxes, "xyxy", None)).reshape(-1, 4)
    confidences = _as_numpy(getattr(boxes, "conf", None)).reshape(-1)
    classes = _as_numpy(getattr(boxes, "cls", None)).reshape(-1)
    names = getattr(result, "names", {0: "pothole"})
    width, height = image_size

    detections: list[dict[str, Any]] = []
    for coordinates_row, score, class_value in zip(coordinates, confidences, classes):
        score_value = float(score)
        if score_value < confidence:
            continue
        x1, y1, x2, y2 = (float(value) for value in coordinates_row)
        clipped_box = [
            max(0.0, min(x1, width - 1)),
            max(0.0, min(y1, height - 1)),
            max(0.0, min(x2, width - 1)),
            max(0.0, min(y2, height - 1)),
        ]
        class_id = int(class_value)
        detections.append(
            {
                "class_id": class_id,
                "class_name": _class_name(names, class_id),
                "confidence": score_value,
                "box": clipped_box,
            }
        )
    return detections


def _box_iou(first: list[float], second: list[float]) -> float:
    """Calculate intersection over union for two xyxy boxes."""

    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _merge_detections(
    detections: list[dict[str, Any]], iou_threshold: float
) -> list[dict[str, Any]]:
    """Merge duplicate detections from multiple inference passes with NMS."""

    ordered = sorted(detections, key=lambda item: item["confidence"], reverse=True)
    kept: list[dict[str, Any]] = []
    for candidate in ordered:
        duplicate = any(
            candidate["class_id"] == existing["class_id"]
            and _box_iou(candidate["box"], existing["box"]) >= iou_threshold
            for existing in kept
        )
        if not duplicate:
            kept.append(candidate)
    return kept


def inference_profile_details(profile: str) -> dict[str, Any]:
    """Return validated public details for an inference profile."""

    normalized = str(profile).strip().lower()
    if normalized not in INFERENCE_PROFILES:
        choices = ", ".join(INFERENCE_PROFILES)
        raise ValidationError(f"Inference profile must be one of: {choices}.")
    details = INFERENCE_PROFILES[normalized]
    return {
        "name": normalized,
        "label": details["label"],
        "description": details["description"],
        "pass_count": len(details["passes"]),
    }


def predict_image(
    image: Image.Image | np.ndarray,
    confidence: float = DEFAULT_CONFIDENCE,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    model: Any | None = None,
    inference_profile: str = "fast",
) -> dict[str, Any]:
    """Run inference and return model-independent detection data."""

    confidence = validate_confidence(confidence)
    iou_threshold = validate_confidence(iou_threshold)
    profile_details = inference_profile_details(inference_profile)
    profile = INFERENCE_PROFILES[profile_details["name"]]
    pil_image = (
        image.convert("RGB")
        if isinstance(image, Image.Image)
        else Image.fromarray(image).convert("RGB")
    )
    active_model = model if model is not None else load_model()

    started = perf_counter()
    try:
        detections: list[dict[str, Any]] = []
        for pass_options in profile["passes"]:
            results = active_model.predict(
                source=np.asarray(pil_image),
                conf=confidence,
                iou=iou_threshold,
                verbose=False,
                **_runtime_predict_options(),
                **pass_options,
            )
            first_result = results[0] if results else None
            if first_result is not None:
                detections.extend(
                    _extract_detections(first_result, confidence, pil_image.size)
                )
    except Exception as exc:
        raise DetectorError(f"Pothole detection failed: {exc}") from exc
    elapsed_ms = (perf_counter() - started) * 1000

    detections = _merge_detections(detections, iou_threshold)
    return {
        "count": len(detections),
        "detections": detections,
        "annotated_image": annotate_image(pil_image, detections),
        "inference_ms": elapsed_ms,
        "inference_profile": profile_details["name"],
        "inference_passes": profile_details["pass_count"],
        "runtime": inference_runtime_details(),
    }


def predict_images(
    images: list[Image.Image | np.ndarray],
    confidence: float = DEFAULT_CONFIDENCE,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    model: Any | None = None,
    inference_profile: str = "fast",
    annotate: bool = True,
) -> list[dict[str, Any]]:
    """Run batched inference for video frames and return one result per image."""

    if not images:
        return []
    confidence = validate_confidence(confidence)
    iou_threshold = validate_confidence(iou_threshold)
    profile_details = inference_profile_details(inference_profile)
    profile = INFERENCE_PROFILES[profile_details["name"]]
    pil_images = [
        (
            image.convert("RGB")
            if isinstance(image, Image.Image)
            else Image.fromarray(image).convert("RGB")
        )
        for image in images
    ]
    active_model = model if model is not None else load_model()
    detections_by_image: list[list[dict[str, Any]]] = [
        [] for _ in range(len(pil_images))
    ]

    started = perf_counter()
    try:
        for pass_options in profile["passes"]:
            results = active_model.predict(
                source=[np.asarray(image) for image in pil_images],
                conf=confidence,
                iou=iou_threshold,
                verbose=False,
                **_runtime_predict_options(),
                **pass_options,
            )
            if len(results) != len(pil_images):
                raise DetectorError(
                    "The model returned an unexpected number of batch results."
                )
            for index, (result, pil_image) in enumerate(zip(results, pil_images)):
                detections_by_image[index].extend(
                    _extract_detections(result, confidence, pil_image.size)
                )
    except DetectorError:
        raise
    except Exception as exc:
        raise DetectorError(f"Batched pothole detection failed: {exc}") from exc
    elapsed_ms = (perf_counter() - started) * 1000
    per_image_ms = elapsed_ms / len(pil_images)
    runtime = inference_runtime_details()

    batch_results: list[dict[str, Any]] = []
    for pil_image, detections in zip(pil_images, detections_by_image):
        merged = _merge_detections(detections, iou_threshold)
        batch_results.append(
            {
                "count": len(merged),
                "detections": merged,
                "annotated_image": (
                    annotate_image(pil_image, merged) if annotate else None
                ),
                "inference_ms": per_image_ms,
                "inference_profile": profile_details["name"],
                "inference_passes": profile_details["pass_count"],
                "runtime": runtime,
            }
        )
    return batch_results
