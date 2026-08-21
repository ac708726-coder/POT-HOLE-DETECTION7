"""Train a pothole detector by fine-tuning an Ultralytics YOLO checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="yolo11n.pt", help="Pretrained detection checkpoint"
    )
    parser.add_argument(
        "--data", type=Path, default=PROJECT_ROOT / "data" / "potholes.yaml"
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--device", default=None, help="For example: cpu, 0, or 0,1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise SystemExit(f"Dataset configuration not found: {args.data}")

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install requirements.txt before training.") from exc

    selected_device = args.device
    if selected_device is None:
        selected_device = "0" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch {torch.__version__} training device: {selected_device}")

    model = YOLO(args.model)
    options = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(PROJECT_ROOT / "runs" / "pothole"),
        "name": args.name,
        "seed": 42,
        "deterministic": True,
    }
    options["device"] = selected_device
    model.train(**options)


if __name__ == "__main__":
    main()
