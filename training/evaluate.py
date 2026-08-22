"""Evaluate a trained pothole model on the held-out YOLO test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
    return parser.parse_args()


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    args = parse_args()
    for path in (args.weights, args.data):
        if not path.is_file():
            raise SystemExit(f"Required file not found: {path}")

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install requirements.txt before evaluation.") from exc

    options: dict[str, Any] = {
        "data": str(args.data),
        "split": "test",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(PROJECT_ROOT / "outputs" / "metrics"),
        "name": args.name,
        "plots": True,
    }
    selected_device = args.device
    if selected_device is None:
        selected_device = "0" if torch.cuda.is_available() else "cpu"
    options["device"] = selected_device
    print(f"PyTorch {torch.__version__} evaluation device: {selected_device}")

    metrics = YOLO(str(args.weights)).val(**options)
    box = metrics.box
    summary = {
        "weights": str(args.weights),
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


if __name__ == "__main__":
    main()
