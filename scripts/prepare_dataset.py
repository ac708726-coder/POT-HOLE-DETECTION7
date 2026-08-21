"""Convert RDD2022 Pascal VOC data into pothole-only YOLO dataset splits."""

from __future__ import annotations

import argparse
import random
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
POTHOLE_CLASS = "D40"


@dataclass(frozen=True)
class Sample:
    country: str
    image_path: Path
    annotation_path: Path
    group: str


def locate_dataset_root(path: Path) -> Path:
    """Accept either the RDD2022 directory or its extraction parent."""

    candidates = (path, path / "RDD2022")
    for candidate in candidates:
        if any(
            (candidate / country).is_dir() for country in ("India", "Japan", "Czech")
        ):
            return candidate
    raise FileNotFoundError(
        f"RDD2022 country folders were not found under '{path}'. Expected, for example, "
        f"'{path / 'India' / 'train' / 'images'}'."
    )


def sequence_group(country: str, stem: str, group_size: int) -> str:
    """Keep nearby numbered images together to reduce sequence leakage."""

    match = re.search(r"(\d+)$", stem)
    if not match or group_size <= 1:
        return f"{country}:{stem}"
    sequence_number = int(match.group(1))
    return f"{country}:block-{sequence_number // group_size:06d}"


def collect_samples(root: Path, countries: list[str], group_size: int) -> list[Sample]:
    samples: list[Sample] = []
    for country in countries:
        country_root = root / country
        image_directory = country_root / "train" / "images"
        annotation_directory = country_root / "train" / "annotations" / "xmls"
        if not annotation_directory.is_dir():
            annotation_directory = country_root / "train" / "annotations"
        if not image_directory.is_dir() or not annotation_directory.is_dir():
            raise FileNotFoundError(
                f"Missing RDD2022 training images or annotations for {country}: {country_root}"
            )

        annotations = {path.stem: path for path in annotation_directory.rglob("*.xml")}
        images = sorted(
            path
            for path in image_directory.rglob("*")
            if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise FileNotFoundError(f"No training images found for {country}.")

        missing_annotations = [
            image.name for image in images if image.stem not in annotations
        ]
        if missing_annotations:
            examples = ", ".join(missing_annotations[:5])
            raise ValueError(
                f"{len(missing_annotations)} {country} training images have no XML annotation "
                f"(examples: {examples}). Unlabelled official test images must not be used for "
                "model evaluation."
            )

        samples.extend(
            Sample(
                country=country,
                image_path=image,
                annotation_path=annotations[image.stem],
                group=sequence_group(country, image.stem, group_size),
            )
            for image in images
        )
    return samples


def convert_annotation(annotation_path: Path, image_path: Path) -> list[str]:
    """Return YOLO class-0 lines for valid D40 boxes only."""

    try:
        root = ET.parse(annotation_path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"Unreadable XML annotation: {annotation_path}") from exc

    size = root.find("size")
    try:
        width = int(float(size.findtext("width", "0"))) if size is not None else 0
        height = int(float(size.findtext("height", "0"))) if size is not None else 0
    except ValueError:
        width = height = 0
    if width <= 0 or height <= 0:
        with Image.open(image_path) as image:
            width, height = image.size

    labels: list[str] = []
    for object_node in root.findall("object"):
        if object_node.findtext("name", "").strip() != POTHOLE_CLASS:
            continue
        box = object_node.find("bndbox")
        if box is None:
            continue
        try:
            xmin = max(0.0, min(float(box.findtext("xmin", "0")), float(width)))
            ymin = max(0.0, min(float(box.findtext("ymin", "0")), float(height)))
            xmax = max(0.0, min(float(box.findtext("xmax", "0")), float(width)))
            ymax = max(0.0, min(float(box.findtext("ymax", "0")), float(height)))
        except ValueError:
            continue
        box_width = xmax - xmin
        box_height = ymax - ymin
        if box_width <= 0 or box_height <= 0:
            continue
        x_center = (xmin + xmax) / 2 / width
        y_center = (ymin + ymax) / 2 / height
        normalized_width = box_width / width
        normalized_height = box_height / height
        labels.append(
            "0 "
            f"{x_center:.8f} {y_center:.8f} "
            f"{normalized_width:.8f} {normalized_height:.8f}"
        )
    return labels


def allocate_groups(groups: list[str], seed: int) -> dict[str, str]:
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    if total < 3:
        raise ValueError("At least three independent sequence groups are required.")

    train_end = max(1, round(total * 0.70))
    validation_end = train_end + max(1, round(total * 0.20))
    train_end = min(train_end, total - 2)
    validation_end = min(max(train_end + 1, validation_end), total - 1)
    return {
        group: (
            "train"
            if index < train_end
            else "val" if index < validation_end else "test"
        )
        for index, group in enumerate(shuffled)
    }


def require_empty_output(path: Path, clean: bool) -> None:
    generated_directories = [
        path / kind / split
        for kind in ("images", "labels")
        for split in ("train", "val", "test")
    ]
    has_generated_files = any(
        directory.is_dir() and any(directory.iterdir())
        for directory in generated_directories
    )
    if has_generated_files and not clean:
        raise FileExistsError(
            f"Processed data already exists under '{path}'. Re-run with --clean to replace "
            "only the generated train/val/test folders."
        )
    if clean:
        for directory in generated_directories:
            resolved = directory.resolve()
            if path.resolve() not in resolved.parents:
                raise RuntimeError(f"Refusing to clean unexpected path: {resolved}")
            if directory.exists():
                shutil.rmtree(directory)


def prepare(
    source: Path,
    output: Path,
    countries: list[str],
    seed: int,
    group_size: int,
    max_train_negative_ratio: float,
    clean: bool,
) -> dict[str, int]:
    root = locate_dataset_root(source)
    require_empty_output(output, clean)
    samples = collect_samples(root, countries, group_size)
    allocation = allocate_groups(sorted({sample.group for sample in samples}), seed)
    labels_by_sample = {
        sample: convert_annotation(sample.annotation_path, sample.image_path)
        for sample in samples
    }
    train_positive_samples = [
        sample
        for sample in samples
        if allocation[sample.group] == "train" and labels_by_sample[sample]
    ]
    train_negative_samples = [
        sample
        for sample in samples
        if allocation[sample.group] == "train" and not labels_by_sample[sample]
    ]
    selected_train_negatives = set(train_negative_samples)
    if max_train_negative_ratio >= 0:
        maximum_negatives = round(
            len(train_positive_samples) * max_train_negative_ratio
        )
        shuffled_negatives = list(train_negative_samples)
        random.Random(seed + 1).shuffle(shuffled_negatives)
        selected_train_negatives = set(shuffled_negatives[:maximum_negatives])

    split_rows: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    counts = {
        "images": 0,
        "potholes": 0,
        "negative_images": 0,
        "skipped_train_negative_images": 0,
    }

    for sample in samples:
        split = allocation[sample.group]
        labels = labels_by_sample[sample]
        if split == "train" and not labels and sample not in selected_train_negatives:
            counts["skipped_train_negative_images"] += 1
            continue
        image_output = output / "images" / split
        label_output = output / "labels" / split
        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)

        country_prefix = f"{sample.country}_"
        output_stem = (
            sample.image_path.stem
            if sample.image_path.stem.startswith(country_prefix)
            else f"{country_prefix}{sample.image_path.stem}"
        )
        output_image = image_output / f"{output_stem}{sample.image_path.suffix.lower()}"
        output_label = label_output / f"{output_stem}.txt"
        shutil.copy2(sample.image_path, output_image)
        output_label.write_text(
            "\n".join(labels) + ("\n" if labels else ""), encoding="utf-8"
        )

        try:
            recorded_path = output_image.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            recorded_path = output_image.resolve().as_posix()
        split_rows[split].append(recorded_path)
        counts["images"] += 1
        counts["potholes"] += len(labels)
        counts["negative_images"] += int(not labels)

    splits_directory = PROJECT_ROOT / "data" / "splits"
    splits_directory.mkdir(parents=True, exist_ok=True)
    for split, rows in split_rows.items():
        (splits_directory / f"{split}.txt").write_text(
            "\n".join(sorted(rows)) + "\n", encoding="utf-8"
        )
        counts[f"{split}_images"] = len(rows)

    dataset_yaml = PROJECT_ROOT / "data" / "potholes.yaml"
    dataset_yaml.write_text(
        "# Generated by scripts/prepare_dataset.py\n"
        f'path: "{output.resolve().as_posix()}"\n'
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: pothole\n",
        encoding="utf-8",
    )
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "RDD2022",
        help="Extracted RDD2022 root",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "data" / "processed"
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        default=["India"],
        help="Extracted country folders to include; India is the recommended baseline",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sequence-group-size",
        type=int,
        default=50,
        help="Keep blocks of nearby numbered images in the same split",
    )
    parser.add_argument(
        "--max-train-negative-ratio",
        type=float,
        default=1.5,
        help=(
            "Maximum negative images per positive training image; use -1 to keep all "
            "training negatives"
        ),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Replace existing generated train/val/test files under data/processed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        counts = prepare(
            args.source,
            args.output,
            args.countries,
            args.seed,
            args.sequence_group_size,
            args.max_train_negative_ratio,
            args.clean,
        )
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc

    print("RDD2022 pothole dataset prepared:")
    for name, count in counts.items():
        print(f"- {name.replace('_', ' ')}: {count}")


if __name__ == "__main__":
    main()
