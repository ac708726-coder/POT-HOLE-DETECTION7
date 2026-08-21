"""Run the selected checkpoint on one image or a directory of sample images."""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--weights", type=Path, default=PROJECT_ROOT / "models" / "best.pt"
    )
    parser.add_argument("--confidence", type=float, default=0.40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        raise SystemExit(f"Model weights not found: {args.weights}")
    if not args.source.exists():
        raise SystemExit(f"Prediction source not found: {args.source}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install requirements.txt before prediction.") from exc

    YOLO(str(args.weights)).predict(
        source=str(args.source),
        conf=args.confidence,
        save=True,
        project=str(PROJECT_ROOT / "outputs" / "images"),
        name="sample_predictions",
    )


if __name__ == "__main__":
    main()
