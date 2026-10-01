"""Evaluate a trained pothole model on the held-out YOLO test split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", type=Path, default=PROJECT_ROOT / "models" / "best.pt"
    )
    parser.add_argument(
        "--data", type=Path, default=PROJECT_ROOT / "data" / "potholes.yaml"
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="test_evaluation")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "metrics" / "test_metrics.json",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--quantize", type=int, choices=[16, 32], default=32)
    parser.add_argument("--failure-confidence", type=float, default=0.15)
    parser.add_argument("--failure-iou", type=float, default=0.50)
    parser.add_argument(
        "--max-failures",
        type=int,
        default=100,
        help="Maximum failure images to save; 0 disables collection.",
    )
    return parser.parse_args()


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def unmatched_boxes(
    truth: list[list[float]],
    predictions: list[list[float]],
    scores: list[float],
    iou_threshold: float,
) -> tuple[list[int], list[int]]:
    """One-to-one IoU matching: duplicate predictions count as false positives."""
    unmatched = set(range(len(truth)))
    false_positives = []
    for index in sorted(range(len(predictions)), key=lambda i: scores[i], reverse=True):
        box = predictions[index]
        best_iou, best_index = 0.0, None
        for target in sorted(unmatched):
            gt = truth[target]
            intersection = max(0, min(box[2], gt[2]) - max(box[0], gt[0])) * max(
                0, min(box[3], gt[3]) - max(box[1], gt[1])
            )
            union = (
                (box[2] - box[0]) * (box[3] - box[1])
                + (gt[2] - gt[0]) * (gt[3] - gt[1])
                - intersection
            )
            iou = intersection / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_index = iou, target
        if best_index is not None and best_iou >= iou_threshold:
            unmatched.remove(best_index)
        else:
            false_positives.append(index)
    return sorted(unmatched), false_positives


def save_failure_cases(model: Any, args: argparse.Namespace, device: str) -> Path:
    from ultralytics.data.utils import IMG_FORMATS, check_det_dataset, img2label_paths

    dataset = check_det_dataset(str(args.data), autodownload=False)
    split = getattr(args, "split", "test")
    sources = dataset.get(split)
    if not sources:
        raise ValueError(
            f"A labelled {split} split is required for failure-case collection."
        )
    files = []
    for source in sources if isinstance(sources, list) else [sources]:
        path = Path(source)
        if path.is_dir():
            files.extend(
                p.resolve()
                for p in path.rglob("*")
                if p.suffix.lower().lstrip(".") in IMG_FORMATS
            )
        elif path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = Path(line.strip())
                    files.append(
                        (path.parent / entry).resolve()
                        if not entry.is_absolute()
                        else entry
                    )
        else:
            raise ValueError(f"Unsupported test source: {path}")
    if not files:
        raise ValueError("No held-out images found for failure-case collection.")
    destination = args.output.parent / f"{args.output.stem}_failures"
    destination.mkdir(parents=True, exist_ok=True)
    report = []
    checked = 0
    for path in sorted(set(files)):
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        label_path = Path(img2label_paths([str(path)])[0])
        if not label_path.is_file():
            raise ValueError(
                f"Missing held-out label: {label_path}; use an empty file for a true negative."
            )
        truth = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            values = [float(value) for value in line.split()]
            if len(values) != 5 or values[0] != 0 or not all(np.isfinite(values)):
                raise ValueError(f"Expected single-class YOLO boxes in {label_path}")
            _, cx, cy, width, height = values
            if (
                not all(0 <= value <= 1 for value in values[1:])
                or width <= 0
                or height <= 0
            ):
                raise ValueError(f"Invalid box in {label_path}")
            truth.append(
                [
                    (cx - width / 2) * image.width,
                    (cy - height / 2) * image.height,
                    (cx + width / 2) * image.width,
                    (cy + height / 2) * image.height,
                ]
            )
        result = model.predict(
            # NumPy sources use BGR; saved Pillow failure images remain RGB.
            source=np.ascontiguousarray(np.asarray(image)[..., ::-1]),
            imgsz=args.imgsz,
            conf=args.failure_confidence,
            iou=0.5,
            agnostic_nms=True,
            device=device,
            verbose=False,
        )[0]
        boxes = result.boxes
        predictions = boxes.xyxy.cpu().tolist()
        scores = boxes.conf.cpu().tolist()
        missed, false_positives = unmatched_boxes(
            truth, predictions, scores, args.failure_iou
        )
        checked += 1
        if not missed and not false_positives:
            continue
        draw = ImageDraw.Draw(image)
        for box in truth:
            draw.rectangle(box, outline="lime", width=3)
        for index, (box, score) in enumerate(zip(predictions, scores), start=1):
            draw.rectangle(box, outline="red", width=3)
            draw.text(
                (box[0], max(0, box[1] - 12)), f"#{index} {score:.0%}", fill="red"
            )
        identity = hashlib.sha256(str(path).encode()).hexdigest()[:12]
        output = destination / f"{identity}_{path.stem}.jpg"
        image.save(output)
        report.append(
            {
                "source": str(path),
                "annotated": output.name,
                "missed_truth_ids": missed,
                "false_positive_ids": false_positives,
                "truth": truth,
                "predictions": predictions,
                "confidences": scores,
            }
        )
        if len(report) >= args.max_failures:
            break
    (destination / "failures.json").write_text(
        json.dumps(
            {
                "confidence": args.failure_confidence,
                "matching_iou": args.failure_iou,
                "images_checked": checked,
                "saved": len(report),
                "cases": report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return destination


def main() -> None:
    args = parse_args()
    if (
        not 0 < args.failure_iou <= 1
        or not 0 < args.failure_confidence <= 1
        or args.max_failures < 0
    ):
        raise SystemExit("Use confidence/IoU in (0, 1] and nonnegative --max-failures.")
    for path in (args.weights, args.data):
        if not path.is_file():
            raise SystemExit(f"Required file not found: {path}")

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install requirements.txt before evaluation.") from exc

    torch.set_num_threads(4)

    options: dict[str, Any] = {
        "data": str(args.data),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(PROJECT_ROOT / "outputs" / "metrics"),
        "name": args.name,
        "plots": True,
        "workers": 0,
        "cache": False,
        "quantize": args.quantize,
    }
    selected_device = args.device
    if selected_device is None:
        selected_device = "0" if torch.cuda.is_available() else "cpu"
    options["device"] = selected_device
    print(f"PyTorch {torch.__version__} evaluation device: {selected_device}")

    model = YOLO(str(args.weights))
    metrics = model.val(**options)
    box = metrics.box
    summary = {
        "weights": str(args.weights),
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "split": args.split,
        "imgsz": args.imgsz,
        "data": str(args.data.resolve()),
        "data_config_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "quantize": args.quantize,
        "batch": args.batch,
        "precision": _number(getattr(box, "mp", None)),
        "recall": _number(getattr(box, "mr", None)),
        "map50": _number(getattr(box, "map50", None)),
        "map50_95": _number(getattr(box, "map", None)),
        "speed_ms_per_image": dict(getattr(metrics, "speed", {})),
    }
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved metrics to {output_path}")
    if args.max_failures:
        print(
            f"Saved failure cases to {save_failure_cases(model, args, selected_device)}"
        )


if __name__ == "__main__":
    main()
