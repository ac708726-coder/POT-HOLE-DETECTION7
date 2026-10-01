"""Mine existing TRAIN labels for missed potholes; never train on held-out photos.

Creates an auditable, bounded replay manifest, without copying or editing images
or labels. Hard positives receive one extra appearance; negatives prevent a
recall-only fine-tune from learning to call every road defect a pothole.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.evaluate import unmatched_boxes


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def label_boxes(label: Path, size: tuple[int, int]) -> list[list[float]]:
    """Missing labels are errors, not presumed negative examples."""
    width, height = size
    boxes = []
    for line in label.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if (
            len(values) != 5
            or values[0] != 0
            or not all(math.isfinite(value) for value in values)
            or not all(0 <= value <= 1 for value in values[1:])
            or values[3] <= 0
            or values[4] <= 0
        ):
            raise ValueError(f"Invalid single-class YOLO label: {label}")
        _, cx, cy, bw, bh = values
        boxes.append(
            [
                (cx - bw / 2) * width,
                (cy - bh / 2) * height,
                (cx + bw / 2) * width,
                (cy + bh / 2) * height,
            ]
        )
    return boxes


def images_in(folder: Path) -> list[Path]:
    files = sorted(
        p.resolve()
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not files:
        raise ValueError(f"No images found: {folder}")
    return files


def clean_train_files(root: Path) -> tuple[list[Path], list[dict]]:
    """Exclude exact held-out duplicates and deduplicate the training pool."""
    held_out = images_in(root / "images/val") + images_in(root / "images/test")
    held_names = {path.name.casefold() for path in held_out}
    held_hashes = {digest(path) for path in held_out}
    selected, excluded, seen = [], [], set()
    for path in images_in(root / "images/train"):
        label = root / "labels/train" / f"{path.stem}.txt"
        # Validate even files later excluded, before launching a costly scan.
        label_boxes(label, (1, 1))
        identity = digest(path)
        reason = (
            "held-out duplicate"
            if path.name.casefold() in held_names or identity in held_hashes
            else "train duplicate" if identity in seen else None
        )
        if reason:
            excluded.append({"filename": path.name, "reason": reason})
        else:
            selected.append(path)
            seen.add(identity)
    return selected, excluded


def diverse_take(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Deterministic round-robin across source countries/viewpoints."""
    groups = defaultdict(list)
    for row in rows:
        groups[Path(row["filename"]).stem.rsplit("_", 1)[0]].append(row)
    for group in groups.values():
        group.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['filename']}".encode()
            ).digest()
        )
    result = []
    while groups and len(result) < count:
        for key in sorted(groups):
            result.append(groups[key].pop())
            if not groups[key]:
                del groups[key]
            if len(result) == count:
                break
    return result


def replay_rows(rows: list[dict], seed: int = 42) -> tuple[list[dict], list[dict]]:
    hard = diverse_take([r for r in rows if r["missed"]], 600, seed)
    hard_names = {r["filename"] for r in hard}
    positives = diverse_take(
        [r for r in rows if r["truth"] and r["filename"] not in hard_names],
        1200 - len(hard),
        seed,
    )
    hard_negatives = diverse_take(
        [r for r in rows if not r["truth"] and r["false_positives"]], 300, seed
    )
    hard_negative_names = {r["filename"] for r in hard_negatives}
    negatives = diverse_take(
        [
            r
            for r in rows
            if not r["truth"] and r["filename"] not in hard_negative_names
        ],
        1200 - len(hard_negatives),
        seed,
    )
    unique = hard + positives + hard_negatives + negatives
    return unique + hard, unique


def prepare_validation_subset(data: Path, count: int) -> Path:
    """Budget checkpoint selection only; final comparisons still use full val/test."""
    import yaml

    if count < 100 or count % 10:
        raise ValueError("Validation subset size must be >= 100 and a multiple of 10.")
    config = yaml.safe_load(data.read_text(encoding="utf-8"))
    root = Path(config["path"])
    if Path(config["val"]).resolve() != (root / "images/val").resolve():
        raise ValueError("Expected the original validation directory.")
    rows = []
    for path in images_in(root / "images/val"):
        truth = label_boxes(root / "labels/val" / f"{path.stem}.txt", (1, 1))
        rows.append({"filename": path.name, "truth": truth})
    positives = diverse_take([r for r in rows if r["truth"]], count // 10, 73)
    negatives = diverse_take([r for r in rows if not r["truth"]], count * 9 // 10, 73)
    if len(positives) + len(negatives) != count:
        raise ValueError("Insufficient validation examples for the requested subset.")
    manifest = data.parent / f"validation_{count}.txt"
    selected = sorted(
        (root / "images/val" / row["filename"]).resolve()
        for row in positives + negatives
    )
    train_files = {
        Path(line).resolve()
        for line in Path(config["train"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if train_files.intersection(selected):
        raise ValueError("Validation subset overlaps training.")
    output = data.parent / f"data_validation_{count}.yaml"
    if manifest.exists() or output.exists():
        raise ValueError("Validation subset already exists; reuse its dataset config.")
    manifest.write_text(
        "\n".join(p.as_posix() for p in selected) + "\n", encoding="utf-8"
    )
    config["val"] = manifest.resolve().as_posix()
    output.write_text(yaml.safe_dump(config), encoding="utf-8")
    print(
        f"Checkpoint-selection val: {len(positives)} positive/{len(negatives)} negative images; SHA256 {digest(manifest)}",
        flush=True,
    )
    return output


def write_manifest(root: Path, output: Path, rows: list[dict]) -> dict:
    import yaml

    repeated, unique = replay_rows(rows)
    if not any(row["truth"] for row in unique):
        raise ValueError("No labelled potholes in the selected training pool.")
    train_folder = (root / "images/train").resolve()
    files = []
    for row in repeated:
        path = (train_folder / row["filename"]).resolve()
        if path.parent != train_folder or not path.is_file():
            raise ValueError("A manifest entry must reference an existing TRAIN image.")
        files.append(path.as_posix())
    manifest = output / "train.txt"
    manifest.write_text("\n".join(files) + "\n", encoding="utf-8")
    config = {
        "path": root.resolve().as_posix(),
        "train": manifest.resolve().as_posix(),
        "val": (root / "images/val").resolve().as_posix(),
        "test": (root / "images/test").resolve().as_posix(),
        "names": {0: "pothole"},
    }
    (output / "data.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return {
        "unique_images": len(unique),
        "training_appearances": len(repeated),
        "positives": sum(bool(row["truth"]) for row in unique),
        "negatives": sum(not row["truth"] for row in unique),
        "extra_hard_positive_appearances": len(repeated) - len(unique),
        "hard_negatives": sum(
            not row["truth"] and bool(row["false_positives"]) for row in unique
        ),
        "sources": {
            group: sum(
                Path(r["filename"]).stem.rsplit("_", 1)[0] == group for r in unique
            )
            for group in sorted(
                {Path(r["filename"]).stem.rsplit("_", 1)[0] for r in unique}
            )
        },
        "manifest_sha256": digest(manifest),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--weights", type=Path, default=ROOT / "models/best.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    if not args.weights.is_file() or not 0 < args.confidence <= 1 or args.batch < 1:
        raise SystemExit("Supply an existing checkpoint, valid confidence and batch.")
    # Never overwrite an earlier mining run or modify the existing dataset.
    args.output.mkdir(parents=True, exist_ok=False)
    print("Checking TRAIN identities against held-out image hashes...", flush=True)
    files, excluded = clean_train_files(args.root)
    if not files:
        raise SystemExit("No leakage-free training images.")
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; refusing to start a slow CPU mining run.")
    torch.set_num_threads(4)
    model = YOLO(str(args.weights.resolve()))
    rows = []
    for start in range(0, len(files), args.batch):
        paths = files[start : start + args.batch]
        results = model.predict(
            source=[str(p) for p in paths],
            imgsz=640,
            rect=False,
            augment=False,
            conf=args.confidence,
            iou=0.5,
            agnostic_nms=True,
            device=0,
            quantize=16,
            verbose=False,
            save=False,
        )
        # Fixed shapes and one GPU task at a time bound laptop memory usage.
        torch.backends.cudnn.benchmark = False
        if len(results) != len(paths):
            raise ValueError("Predictor returned an unexpected result count.")
        for path, result in zip(paths, results):
            height, width = result.orig_shape
            truth = label_boxes(
                args.root / "labels/train" / f"{path.stem}.txt", (width, height)
            )
            boxes = result.boxes.xyxy.cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()
            missed, false = unmatched_boxes(truth, boxes, scores, 0.5)
            rows.append(
                {
                    "filename": path.name,
                    "truth": truth,
                    "missed": missed,
                    "false_positives": false,
                    "predictions": boxes,
                    "scores": scores,
                }
            )
        if len(rows) % 256 == 0 or len(rows) == len(files):
            print(f"Mined {len(rows)}/{len(files)} TRAIN images", flush=True)
    selection = write_manifest(args.root, args.output, rows)
    report = {
        "baseline_sha256": digest(args.weights),
        "confidence": args.confidence,
        "matching_iou": 0.5,
        "mining_profile": "single pass, 640, correct file-path BGR decoding",
        "excluded": excluded,
        "scanned": len(rows),
        "missed_positive_images": sum(bool(row["missed"]) for row in rows),
        "selection": selection,
        "rows": rows,
    }
    (args.output / "mining.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
