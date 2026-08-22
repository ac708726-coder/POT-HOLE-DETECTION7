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
    parser.add_argument(
        "--accuracy-preset",
        action="store_true",
        help=(
            "Use a conservative low-learning-rate continuation preset for an "
            "existing pothole checkpoint"
        ),
    )
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
    if args.accuracy_preset:
        options.update(
            {
                "optimizer": "AdamW",
                # Continue below the final learning rate of the 25-epoch base run;
                # raising it here erased useful localization features.
                "lr0": 0.0001,
                "lrf": 0.1,
                "warmup_epochs": 0.0,
                "warmup_bias_lr": 0.0,
                "warmup_momentum": 0.9,
                "weight_decay": 0.0005,
                "patience": min(12, max(6, args.epochs)),
                "close_mosaic": 0,
                "mosaic": 0.0,
                "degrees": 2.0,
                "translate": 0.05,
                "scale": 0.2,
                "perspective": 0.0001,
                "fliplr": 0.5,
                "amp": True,
                "plots": True,
                "save_period": 5,
            }
        )
    options["device"] = selected_device
    model.train(**options)


if __name__ == "__main__":
    main()
