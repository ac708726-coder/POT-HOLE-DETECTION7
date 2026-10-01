"""Compare real application inference on a reproducible labelled validation sample.

No training, downloads, test-set tuning, or checkpoint replacement. Precision and
recall use one-to-one IoU=0.5 matching at the exact UI confidence thresholds.
The validation sample is stratified; its precision is not a full-dataset score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.inference_tiles import remap_tile_detections, tile_windows
from training.evaluate import unmatched_boxes
from utils import detector
from utils.image_processor import annotate_image, image_to_bgr


def predict_tiled_candidate(image, model, profile="fast", confidence=0.15):
    """Experimental crop comparison, isolated from the application's scan path."""
    result = detector.predict_image(
        image, model=model, inference_profile=profile, confidence=confidence
    )
    started = perf_counter()
    detections = list(result["detections"])
    windows = tile_windows(image.size)
    batch_size = detector.inference_runtime_details()["video_batch_size"]
    for start in range(0, len(windows), batch_size):
        views = windows[start : start + batch_size]
        crops = [image.crop(window) for window in views]
        predictions = model.predict(
            source=[image_to_bgr(crop) for crop in crops],
            imgsz=640,
            augment=False,
            conf=confidence,
            iou=0.5,
            agnostic_nms=True,
            verbose=False,
            **detector._runtime_predict_options(),
        )
        if len(predictions) != len(crops):
            raise detector.DetectorError("Unexpected number of tile results.")
        for prediction, crop, window in zip(predictions, crops, views):
            local = detector._extract_detections(prediction, confidence, crop.size)
            detections.extend(remap_tile_detections(local, window, image.size))
    result["detections"] = detector._merge_detections(detections, 0.5)
    result["count"] = len(result["detections"])
    result["inference_tiles"] = len(windows)
    result["inference_ms"] += (perf_counter() - started) * 1000
    result["annotated_image"] = annotate_image(image, result["detections"])
    return result


class LegacyColorModel:
    """Reproduce the RGB-as-BGR bug ONLY inside this benchmark process."""

    def __init__(self, model):
        self.model = model

    def predict(self, **kwargs):
        source = kwargs["source"]
        kwargs["source"] = (
            [np.ascontiguousarray(image[..., ::-1]) for image in source]
            if isinstance(source, list)
            else np.ascontiguousarray(source[..., ::-1])
        )
        return self.model.predict(**kwargs)


def validation_sample(root: Path, positives: int, negatives: int, seed: int):
    groups: dict[bool, list[Path]] = {True: [], False: []}
    for label in (root / "labels" / "val").glob("*.txt"):
        image = root / "images" / "val" / f"{label.stem}.jpg"
        if not image.is_file():
            raise ValueError(f"Missing validation image: {image}")
        groups[bool(label.read_text(encoding="utf-8").strip())].append(image)
    selected = []
    for positive, count in ((True, positives), (False, negatives)):
        ordered = sorted(
            groups[positive],
            key=lambda path: hashlib.sha256(f"{seed}:{path.name}".encode()).digest(),
        )
        if count < 0 or count > len(ordered):
            raise ValueError(
                f"Requested {count} examples; only {len(ordered)} available."
            )
        selected.extend(ordered[:count])
    return sorted(selected, key=lambda path: path.name)


def ground_truth(path: Path, size: tuple[int, int]):
    width, height = size
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if len(values) != 5 or values[0] != 0 or not all(np.isfinite(values)):
            raise ValueError(f"Invalid single-class YOLO annotation: {path}")
        _, cx, cy, bw, bh = values
        if not all(0 <= value <= 1 for value in values[1:]) or bw <= 0 or bh <= 0:
            raise ValueError(f"Invalid box: {path}")
        boxes.append(
            [
                (cx - bw / 2) * width,
                (cy - bh / 2) * height,
                (cx + bw / 2) * width,
                (cy + bh / 2) * height,
            ]
        )
    return boxes


def operating_metrics(rows: list[dict], confidence: float):
    tp = fp = fn = 0
    false_positive_images = 0
    for row in rows:
        predictions = [d for d in row["detections"] if d["confidence"] >= confidence]
        misses, extras = unmatched_boxes(
            row["truth"],
            [d["box"] for d in predictions],
            [d["confidence"] for d in predictions],
            0.5,
        )
        tp += len(row["truth"]) - len(misses)
        fp += len(extras)
        fn += len(misses)
        false_positive_images += bool(extras)
    precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {
        "confidence": confidence,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "false_positive_images": false_positive_images,
    }


def save_recovery_examples(root: Path, output: Path, mode: str, limit: int = 3):
    """Illustrative recoveries, not a substitute for scoring the complete sample."""
    old_file = output.with_name(f"{output.stem}_legacy_{mode}_predictions.json")
    new_file = output.with_name(f"{output.stem}_{mode}_predictions.json")
    if not old_file.is_file() or not new_file.is_file():
        return
    before = json.loads(old_file.read_text(encoding="utf-8"))
    after = json.loads(new_file.read_text(encoding="utf-8"))
    if len(before) != len(after):
        raise ValueError("Cannot compare prediction files with different image counts.")
    improved = []
    for old, new in zip(before, after):
        if old["filename"] != new["filename"] or old["truth"] != new["truth"]:
            raise ValueError(
                "Cannot compare predictions from different validation samples."
            )
        old_score = operating_metrics([old], 0.35)
        new_score = operating_metrics([new], 0.35)
        gain = new_score["true_positives"] - old_score["true_positives"]
        if gain > 0:
            improved.append((gain, old, new, old_score, new_score))
    folder = output.parent / f"{output.stem}_recoveries" / mode
    folder.mkdir(parents=True, exist_ok=True)
    for _, old, new, old_score, new_score in sorted(
        improved, key=lambda entry: -entry[0]
    )[:limit]:
        with Image.open(root / "images" / "val" / new["filename"]) as source:
            image = source.convert("RGB")
        panels = []
        captions = [
            "Labelled ground truth",
            f"Before: {old_score['true_positives']} matched",
            f"After: {new_score['true_positives']} matched",
        ]
        truth = image.copy()
        drawing = ImageDraw.Draw(truth)
        for box in new["truth"]:
            drawing.rectangle(box, outline="lime", width=max(2, image.width // 200))
        panels.append(truth)
        for row in (old, new):
            panels.append(
                annotate_image(
                    image, [d for d in row["detections"] if d["confidence"] >= 0.35]
                )
            )
        target_height = round(image.height * min(1, 600 / image.width))
        resized = [panel.resize((600, target_height)) for panel in panels]
        canvas = Image.new("RGB", (1800, target_height + 70), "#101820")
        draw = ImageDraw.Draw(canvas)
        for index, (panel, caption) in enumerate(zip(resized, captions)):
            canvas.paste(panel, (600 * index, 70))
            draw.text(
                (600 * index + 15, 12),
                caption,
                fill="white",
                font=ImageFont.load_default(size=22),
            )
        draw.text(
            (15, 44),
            f"{new['filename']} | {mode} | confidence 0.35 | same weights",
            fill="white",
        )
        canvas.save(folder / new["filename"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=PROJECT_ROOT / "data" / "processed"
    )
    parser.add_argument("--positives", type=int, default=100)
    parser.add_argument("--negatives", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", type=Path, default=detector.MODEL_PATH)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "legacy_fast",
            "fast",
            "legacy_balanced",
            "balanced",
            "legacy_thorough",
            "thorough",
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "metrics" / "recall_benchmark.json",
    )
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error("The requested checkpoint does not exist.")
    detector.MODEL_PATH = args.weights.resolve()
    if any(
        name.removeprefix("legacy_")
        not in {
            "fast",
            "balanced",
            "thorough",
            "multiscale",
            "tiled",
            "tiled_multiscale",
            "wide_scale",
            "zoomout",
            "balanced_union",
        }
        for name in args.variants
    ):
        parser.error("Unknown inference variant.")
    detector.INFERENCE_PROFILES.update(
        {
            "multiscale": {
                "label": "Candidate",
                "description": "Unaugmented scales",
                "passes": (
                    {"imgsz": 640, "augment": False},
                    {"imgsz": 1280, "augment": False},
                ),
            },
            "tiled": {
                "label": "Candidate",
                "description": "Four overlapping crops",
                "passes": ({"imgsz": 640, "augment": False},),
                "tiles": True,
            },
            "tiled_multiscale": {
                "label": "Candidate",
                "description": "Crops plus scales",
                "passes": (
                    {"imgsz": 640, "augment": False},
                    {"imgsz": 1280, "augment": False},
                ),
                "tiles": True,
            },
            "zoomout": {
                "label": "Candidate",
                "description": "Include a wider field of view",
                "passes": (
                    {"imgsz": 320, "augment": False},
                    {"imgsz": 640, "augment": False},
                ),
            },
            "wide_scale": {
                "label": "Candidate",
                "description": "Large-to-small object scales",
                "passes": (
                    {"imgsz": 320, "augment": False},
                    {"imgsz": 640, "augment": True},
                    {"imgsz": 1280, "augment": False},
                ),
            },
            "balanced_union": {
                "label": "Candidate",
                "description": "Preserve original and augmented boxes",
                "passes": (
                    {"imgsz": 640, "augment": False},
                    {"imgsz": 640, "augment": True},
                ),
            },
        }
    )
    paths = validation_sample(args.root, args.positives, args.negatives, args.seed)
    if not paths:
        parser.error("At least one validation image is required.")
    model = detector.load_model()
    detector.warm_up_model(model)
    # Diverse aspect ratios can make cuDNN autotuning consume substantial memory.
    # Disable it for this offline benchmark, not in the application.
    import torch

    torch.backends.cudnn.benchmark = False
    summary = {
        "split": "val",
        "seed": args.seed,
        "images": len(paths),
        "positive_images": args.positives,
        "negative_images": args.negatives,
        "model_sha256": hashlib.sha256(detector.MODEL_PATH.read_bytes()).hexdigest(),
        "sample_sha256": hashlib.sha256(
            "\n".join(p.name for p in paths).encode()
        ).hexdigest(),
        "matching_iou": 0.5,
        "runtime": detector.inference_runtime_details(),
        "note": "Fixed validation subset, not test mAP or deployment accuracy.",
        "variants": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        profile = variant.removeprefix("legacy_")
        active_model = (
            LegacyColorModel(model) if variant.startswith("legacy_") else model
        )
        rows = []
        started = perf_counter()
        for index, path in enumerate(paths, start=1):
            with Image.open(path) as source:
                image = source.convert("RGB")
            truth = ground_truth(
                args.root / "labels" / "val" / f"{path.stem}.txt", image.size
            )
            if detector.INFERENCE_PROFILES[profile].get("tiles"):
                result = predict_tiled_candidate(image, active_model, profile)
            else:
                result = detector.predict_image(
                    image,
                    confidence=0.15,
                    model=active_model,
                    inference_profile=profile,
                )
            rows.append(
                {
                    "filename": path.name,
                    "truth": truth,
                    "detections": result["detections"],
                    "inference_ms": result["inference_ms"],
                }
            )
            if index % 50 == 0:
                print(
                    f"{variant}: {index}/{len(paths)} ({perf_counter() - started:.1f}s)",
                    flush=True,
                )
        metrics = [
            operating_metrics(rows, confidence) for confidence in (0.15, 0.25, 0.35)
        ]
        summary["variants"][variant] = {
            "metrics": metrics,
            "mean_ms": float(np.mean([row["inference_ms"] for row in rows])),
            "median_ms": float(np.median([row["inference_ms"] for row in rows])),
            "ground_truth_boxes": sum(len(row["truth"]) for row in rows),
        }
        raw_path = args.output.with_name(
            f"{args.output.stem}_{variant}_predictions.json"
        )
        raw_path.write_text(json.dumps(rows), encoding="utf-8")
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(variant, json.dumps(summary["variants"][variant]), flush=True)
        # A few labelled positive examples for manual inspection; no cherry-picked metrics.
        evidence = args.output.parent / f"{args.output.stem}_examples" / variant
        evidence.mkdir(parents=True, exist_ok=True)
        for row in [row for row in rows if row["truth"]][:3]:
            with Image.open(args.root / "images" / "val" / row["filename"]) as source:
                image = source.convert("RGB")
            detections = [d for d in row["detections"] if d["confidence"] >= 0.35]
            annotate_image(image, detections).save(evidence / row["filename"])
    print(f"Saved comparison: {args.output}", flush=True)
    for mode in ("fast", "balanced", "thorough"):
        save_recovery_examples(args.root, args.output, mode)


if __name__ == "__main__":
    main()
