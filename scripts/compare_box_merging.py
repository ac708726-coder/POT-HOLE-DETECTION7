"""Compare box-merging strategies for multi-pass inference on the held-out test split.

The application merges overlapping boxes from multiple inference passes with plain NMS:
keep the highest-confidence box in each cluster, drop the rest. Weighted Boxes Fusion is
the obvious alternative — replace each cluster with a confidence-weighted average box.
Fusion only moves coordinates, so the kept set and its confidences are identical and any
effect should show at strict IoU thresholds (mAP50-95) more than at mAP50.

This runs the application's own inference path (predict_image) once per strategy over the
same images and scores both with the same AP implementation, so the comparison is
internally consistent. Absolute values here are NOT comparable to training/evaluate.py,
which uses Ultralytics' own metrics and different defaults; only the delta is meaningful.

Measured 2026-08-23 on the thorough profile over all 3,925 test images: fusion came out
slightly behind (mAP50 0.2348 -> 0.2327, mAP50-95 0.1120 -> 0.1115), so the application
kept plain NMS. Re-run this before revisiting that decision.

    python scripts/compare_box_merging.py --profile thorough
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from utils import detector
from utils.detector import _box_iou, predict_image


def plain_nms(detections, iou_threshold):
    """What the application ships: keep the max-confidence box, drop overlaps."""

    ordered = sorted(detections, key=lambda item: item["confidence"], reverse=True)
    kept = []
    for candidate in ordered:
        duplicate = any(
            candidate["class_id"] == existing["class_id"]
            and _box_iou(candidate["box"], existing["box"]) >= iou_threshold
            for existing in kept
        )
        if not duplicate:
            kept.append(candidate)
    return kept


def wbf_merge(detections, iou_threshold):
    """Weighted Boxes Fusion: average each cluster's boxes by confidence."""

    ordered = sorted(detections, key=lambda item: item["confidence"], reverse=True)
    kept: list[dict] = []
    clusters: list[list[dict]] = []
    for candidate in ordered:
        match_index = next(
            (
                index
                for index, existing in enumerate(kept)
                if candidate["class_id"] == existing["class_id"]
                and _box_iou(candidate["box"], existing["box"]) >= iou_threshold
            ),
            None,
        )
        if match_index is None:
            kept.append(candidate)
            clusters.append([candidate])
        else:
            clusters[match_index].append(candidate)

    fused = []
    for representative, cluster in zip(kept, clusters):
        total_weight = sum(member["confidence"] for member in cluster)
        merged = dict(representative)
        merged["box"] = [
            sum(member["box"][i] * member["confidence"] for member in cluster)
            / total_weight
            for i in range(4)
        ]
        fused.append(merged)
    return fused


def load_ground_truth(label_path: Path, width: int, height: int):
    """Read YOLO-format labels and return absolute xyxy boxes."""

    if not label_path.is_file():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, xc, yc, w, h = (float(v) for v in parts)
        boxes.append(
            [
                (xc - w / 2) * width,
                (yc - h / 2) * height,
                (xc + w / 2) * width,
                (yc + h / 2) * height,
            ]
        )
    return boxes


def average_precision(records, total_gt, iou_threshold):
    """Standard 101-point interpolated AP for a single class."""

    if total_gt == 0:
        return 0.0
    # records: (confidence, image_index, box)
    records = sorted(records, key=lambda r: r[0], reverse=True)
    matched: dict[int, set[int]] = {}
    tp = np.zeros(len(records))
    fp = np.zeros(len(records))

    for i, (_conf, image_index, box, gt_boxes) in enumerate(records):
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gt_boxes):
            if j in matched.get(image_index, set()):
                continue
            score = _box_iou(box, gt)
            if score > best_iou:
                best_iou, best_j = score, j
        if best_j >= 0 and best_iou >= iou_threshold:
            matched.setdefault(image_index, set()).add(best_j)
            tp[i] = 1
        else:
            fp[i] = 1

    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    recall = tp_cum / total_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = precision[recall >= t].max() if np.any(recall >= t) else 0.0
        ap += p / 101
    return float(ap)


def score(all_detections, total_gt):
    ap50 = average_precision(all_detections, total_gt, 0.50)
    aps = [
        average_precision(all_detections, total_gt, t)
        for t in np.arange(0.50, 1.00, 0.05)
    ]
    return ap50, float(np.mean(aps))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="thorough")
    parser.add_argument("--limit", type=int, default=0, help="0 = all test images")
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()

    images_dir = PROJECT / "data" / "processed" / "images" / "test"
    labels_dir = PROJECT / "data" / "processed" / "labels" / "test"
    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    # Positives carry every ground-truth box; negatives only contribute false
    # positives. Both matter for AP, so keep the split intact unless --limit is set.
    if args.limit:
        image_paths = image_paths[: args.limit]

    print(f"profile={args.profile}  images={len(image_paths)}")

    results = {}
    for label, merge_fn in (
        ("wbf_fusion", wbf_merge),
        ("plain_nms", plain_nms),
    ):
        original = detector._merge_detections
        detector._merge_detections = merge_fn
        records, total_gt = [], 0
        started = time.perf_counter()
        try:
            for index, path in enumerate(image_paths):
                image = Image.open(path).convert("RGB")
                gt = load_ground_truth(labels_dir / f"{path.stem}.txt", *image.size)
                total_gt += len(gt)
                out = predict_image(
                    image,
                    confidence=args.confidence,
                    inference_profile=args.profile,
                )
                for d in out["detections"]:
                    records.append((d["confidence"], index, d["box"], gt))
                if (index + 1) % 250 == 0:
                    rate = (time.perf_counter() - started) / (index + 1)
                    print(
                        f"  {label}: {index + 1}/{len(image_paths)} "
                        f"({rate:.3f}s/img)",
                        flush=True,
                    )
        finally:
            detector._merge_detections = original

        ap50, ap5095 = score(records, total_gt)
        elapsed = time.perf_counter() - started
        results[label] = {
            "mAP50": ap50,
            "mAP50_95": ap5095,
            "detections": len(records),
            "ground_truth_boxes": total_gt,
            "seconds": elapsed,
        }
        print(
            f"{label}: mAP50={ap50:.4f}  mAP50-95={ap5095:.4f}  "
            f"dets={len(records)}  gt={total_gt}  ({elapsed:.0f}s)"
        )

    w, n = results["wbf_fusion"], results["plain_nms"]
    print("\n=== WBF vs plain NMS ===")
    print(
        f"mAP50    : {n['mAP50']:.4f} -> {w['mAP50']:.4f}  ({w['mAP50']-n['mAP50']:+.4f})"
    )
    print(
        f"mAP50-95 : {n['mAP50_95']:.4f} -> {w['mAP50_95']:.4f}  "
        f"({w['mAP50_95']-n['mAP50_95']:+.4f})"
    )
    # A --limit run is a smoke test, not a result. Keep it out of the full-run file so a
    # quick check cannot silently overwrite a measurement that took several minutes.
    suffix = f"_limit{args.limit}" if args.limit else ""
    out_path = (
        PROJECT / "outputs" / "metrics" / f"wbf_vs_nms_{args.profile}{suffix}.json"
    )
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
