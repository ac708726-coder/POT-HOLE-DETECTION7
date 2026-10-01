"""Train a pothole detector by fine-tuning an Ultralytics YOLO checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def attach_memory_safety(model) -> None:
    """Release unused caches between stages on memory-constrained laptops.

    Model/optimizer tensors and gradient accumulation are not modified. Standard
    Ultralytics cleanup is VRAM-threshold-based, but Windows host commit can be
    exhausted even when allocated VRAM is low.
    """
    import gc

    import torch

    batches = 0

    def trim(_trainer):
        gc.collect()
        if getattr(getattr(_trainer, "device", None), "type", None) == "cuda":
            torch.cuda.empty_cache()

    def batch_end(trainer):
        nonlocal batches
        batches += 1
        if batches % 50 == 0:
            trim(trainer)

    def prepare(trainer):
        trainer.args.plots = False
        trim(trainer)

    model.add_callback("on_pretrain_routine_start", prepare)
    model.add_callback("on_train_epoch_start", trim)
    model.add_callback("on_fit_epoch_end", trim)
    model.add_callback("on_train_batch_end", batch_end)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="yolo11s.pt",
        help=(
            "Pretrained detection checkpoint. Defaults to yolo11s.pt to match the "
            "lineage of the shipped models/best.pt (see docs/model_card.md); pass "
            "yolo11n.pt for a smaller/faster model."
        ),
    )
    parser.add_argument(
        "--data", type=Path, default=PROJECT_ROOT / "data" / "potholes.yaml"
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help=(
            "Images per batch. Use -1 for Ultralytics AutoBatch, which sizes the "
            "batch to roughly 60%% of free VRAM; recommended when raising --imgsz "
            "or moving to a larger backbone on a laptop GPU."
        ),
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--validation-subset",
        type=int,
        default=None,
        help="Budget checkpoint selection on a fixed stratified val subset; evaluate full val afterward.",
    )
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--device", default=None, help="For example: cpu, 0, or 0,1")
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help=(
            "Stop after this many epochs without validation-fitness improvement. "
            "Ultralytics defaults to 100, which never triggers on shorter runs."
        ),
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=None,
        help=(
            "Write a checkpoint every N epochs so an interrupted laptop run can be "
            "recovered rather than restarted."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume the run named by --name from its last.pt checkpoint. Other "
            "training options are restored from that run and cannot be changed."
        ),
    )
    parser.add_argument(
        "--accuracy-preset",
        action="store_true",
        help=(
            "Use a conservative low-learning-rate continuation preset for an "
            "existing pothole checkpoint"
        ),
    )
    parser.add_argument(
        "--hard-example-preset",
        action="store_true",
        help="Low-LR replay fine-tune with moderate zoom/lighting/viewpoint augmentation.",
    )
    parser.add_argument(
        "--multi-scale",
        action="store_true",
        help=(
            "Vary input resolution across batches during training. Helps recall on "
            "small or distant potholes, at extra training cost."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.accuracy_preset and args.hard_example_preset:
        raise SystemExit("Choose only one fine-tuning preset.")
    if not args.data.is_file():
        raise SystemExit(f"Dataset configuration not found: {args.data}")

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install requirements.txt before training.") from exc

    if args.hard_example_preset:
        torch.set_num_threads(4)

    selected_device = args.device
    if selected_device is None:
        selected_device = "0" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch {torch.__version__} training device: {selected_device}")
    if selected_device != "cpu" and not torch.cuda.is_available():
        raise SystemExit(
            f"Device '{selected_device}' was requested but CUDA is unavailable to "
            f"PyTorch {torch.__version__}. A '+cpu' build cannot train on the GPU; "
            "reinstall torch from the CUDA index URL in README.md."
        )

    run_directory = PROJECT_ROOT / "runs" / "pothole" / args.name
    if args.resume:
        last_checkpoint = run_directory / "weights" / "last.pt"
        if not last_checkpoint.is_file():
            raise SystemExit(
                f"Cannot resume: no checkpoint at {last_checkpoint}. Start the run "
                "without --resume, or correct --name."
            )
        print(f"Resuming from {last_checkpoint}")
        resumed = YOLO(str(last_checkpoint))
        if args.hard_example_preset:
            attach_memory_safety(resumed)
        resumed.train(resume=True)
        return

    if run_directory.exists():
        raise SystemExit(f"Run already exists: {run_directory}; choose a new --name.")

    if args.validation_subset is not None:
        # This helper imports only evaluation utilities, not the Streamlit app.
        sys.path.insert(0, str(PROJECT_ROOT))
        from training.hard_examples import prepare_validation_subset

        args.data = prepare_validation_subset(args.data, args.validation_subset)

    model = YOLO(args.model)
    if args.hard_example_preset:
        attach_memory_safety(model)
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
        "multi_scale": args.multi_scale,
        "patience": 15,
        "save_period": 5,
        "mosaic": 0.5,
        "close_mosaic": min(10, args.epochs),
        "scale": 0.4,
        "perspective": 0.0002,
        "degrees": 5.0,
        "translate": 0.1,
        "hsv_h": 0.015,
        "hsv_s": 0.5,
        "hsv_v": 0.35,
        "fliplr": 0.5,
        "amp": True,
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
    if args.hard_example_preset:
        options.update(
            {
                "optimizer": "AdamW",
                "lr0": 0.00005,
                "lrf": 0.2,
                "warmup_epochs": 0.0,
                "warmup_bias_lr": 0.0,
                "weight_decay": 0.0005,
                "patience": 4,
                "mosaic": 0.0,
                "close_mosaic": 0,
                "scale": 0.35,
                "degrees": 10.0,
                "translate": 0.1,
                "perspective": 0.0002,
                "hsv_h": 0.015,
                "hsv_s": 0.4,
                "hsv_v": 0.3,
                "cache": False,
                "save_period": 1,
                "plots": False,
            }
        )
    # Explicit flags win over the accuracy preset's own patience/save_period values.
    if args.patience is not None:
        options["patience"] = args.patience
    if args.save_period is not None:
        options["save_period"] = args.save_period
    options["device"] = selected_device
    model.train(**options)


if __name__ == "__main__":
    main()
