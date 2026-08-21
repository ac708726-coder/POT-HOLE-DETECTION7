"""Check raw or processed YOLO images and labels for common dataset errors."""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
BOUNDARY_TOLERANCE = 1e-7


def validate_label(path: Path) -> list[str]:
    errors: list[str] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            errors.append(f"{path}: line {line_number} must have 5 fields")
            continue
        try:
            class_id = int(fields[0])
            x_center, y_center, width, height = map(float, fields[1:])
        except ValueError:
            errors.append(f"{path}: line {line_number} contains non-numeric values")
            continue
        if class_id != 0:
            errors.append(
                f"{path}: line {line_number} uses class {class_id}; expected 0"
            )
        if not all(
            0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)
        ):
            errors.append(f"{path}: line {line_number} has values outside 0..1")
        if width <= 0 or height <= 0:
            errors.append(f"{path}: line {line_number} has a zero-size box")
        if (
            x_center - width / 2 < -BOUNDARY_TOLERANCE
            or x_center + width / 2 > 1 + BOUNDARY_TOLERANCE
        ):
            errors.append(f"{path}: line {line_number} extends beyond image width")
        if (
            y_center - height / 2 < -BOUNDARY_TOLERANCE
            or y_center + height / 2 > 1 + BOUNDARY_TOLERANCE
        ):
            errors.append(f"{path}: line {line_number} extends beyond image height")
    return errors


def check_dataset(root: Path) -> list[str]:
    images_dir = root / "images"
    labels_dir = root / "labels"
    errors: list[str] = []
    if not images_dir.is_dir() or not labels_dir.is_dir():
        return [f"Expected both {images_dir} and {labels_dir} directories"]

    images = [
        path
        for path in images_dir.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    labels = list(labels_dir.rglob("*.txt"))
    image_stems = {path.relative_to(images_dir).with_suffix("") for path in images}
    label_stems = {path.relative_to(labels_dir).with_suffix("") for path in labels}

    for missing in sorted(image_stems - label_stems):
        errors.append(f"Missing label for image: {missing}")
    for orphan in sorted(label_stems - image_stems):
        errors.append(f"Label has no matching image: {orphan}")

    hashes: dict[str, list[Path]] = defaultdict(list)
    for image_path in images:
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError):
            errors.append(f"Unreadable image: {image_path}")
            continue
        hashes[hashlib.sha256(image_path.read_bytes()).hexdigest()].append(image_path)

    for duplicate_paths in hashes.values():
        if len(duplicate_paths) > 1:
            errors.append("Duplicate images: " + ", ".join(map(str, duplicate_paths)))
    for label_path in labels:
        errors.extend(validate_label(label_path))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    errors = check_dataset(args.root)
    if errors:
        print("Dataset check failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print(f"Dataset check passed: {args.root}")


if __name__ == "__main__":
    main()
