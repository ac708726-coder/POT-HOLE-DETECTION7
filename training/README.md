# Divot: prepare the next pothole model

The reported misses on phone close-ups and top-down photos suggest a viewpoint
coverage gap. Treat that as a hypothesis: label and evaluate those examples before
claiming a cause or an accuracy improvement. The existing checkpoint stays in place.
This folder prepares training; running the app never triggers training.

## Candidate public data

| Source | Role and preparation |
| --- | --- |
| [RDD2022 official repository](https://github.com/sekilab/RoadDamageDetector) / [Figshare](https://doi.org/10.6084/m9.figshare.21431547) | Road-scene baseline, including China_Drone. Convert Pascal VOC XML using `scripts/prepare_dataset.py`; keep D40 only as class 0. Official test images are unlabelled, so use a separate labelled holdout. |
| [Roboflow public pothole dataset](https://public.roboflow.com/object-detection/pothole) | Candidate additional road/pothole photographs. Review viewpoints, annotations and source licence before using; export YOLO detection format. |
| [Roboflow Universe pothole projects](https://universe.roboflow.com/search?q=pothole) | Look specifically for phone close-ups, top-down views, wet surfaces and small distant potholes. Project licences and quality vary; record the exact project/version and attribution. Avoid repackaged RDD2022 duplicates. |

Public availability does not automatically grant training or redistribution rights.
Record the original source, licence, version, viewpoint and split in a manifest.
Add consented phone photos where the public candidates leave gaps. Include hard
negatives: patches, manholes, puddles, cracks, shadows, gravel and undamaged roads.

## Merge into one YOLO dataset

1. Keep source downloads unchanged in `data/raw/<source>/`. Inspect annotations and
   merge only bounding-box datasets; segmentation polygons need explicit conversion.
2. Remap every pothole label to class **0**. Discard other class boxes, retaining their
   images as negatives only after checking that no visible pothole is unlabelled.
3. Each image has a same-stem `.txt`: `0 x_center y_center width height`, normalized
   by the original image dimensions to [0, 1]. True negatives use an empty label file.
   VOC conversion: center=(xmin+xmax)/2, (ymin+ymax)/2; size=xmax-xmin, ymax-ymin.
4. Deduplicate by image content hash and review near duplicates/perceptual hashes.
   Group all frames from the same road/video/site together. Split original images
   **before** augmentation; never move a transformed copy across splits.
5. Preserve existing validation/test identities where possible, plus a new untouched
   phone/top-down holdout. Allocate new groups roughly 70/15/15, stratified by
   viewpoint and source. Tune only against validation; use test for final reporting.
6. Prefix file stems with source + content hash to avoid filename collisions and copy
   image/label pairs into the layout below. Validate counts, missing labels,
   zero-area/out-of-range boxes, duplicate splits and the positive/negative mix.

```text
data/mixed/
  images/train/   images/val/   images/test/
  labels/train/   labels/val/   labels/test/
  manifest.csv   # file, source, group, viewpoint, split, sha256, licence
```

Copy `training/data.yaml` to a working config and set `path` to the **absolute**
merged dataset folder. Run `python scripts/check_dataset.py --root data/mixed`.
YOLO format reference: [Ultralytics detection datasets](https://docs.ultralytics.com/datasets/detect/).

## Training recipe (prepare only until explicitly started)

Use PyTorch/Ultralytics from the project venv with a compatible CUDA install.
The existing `train.py` defaults to YOLO11s, 60 epochs, 960px, batch 8, patience 15,
seed 42 and checkpoint saves every 5 epochs. A larger input helps small potholes;
reduce batch size if VRAM runs out. Mosaic 0.5 (closed for the last 10 epochs),
scale 0.4, mild perspective 0.0002 and HSV changes broaden texture/lighting coverage.
They do not replace actual close-up training examples. See [training settings](https://docs.ultralytics.com/modes/train/).

```powershell
# Commands for a FUTURE run; nothing is launched automatically.
python training/train.py --data training/data.yaml --name mixed_domain --device 0
# Conservative continuation from a trained checkpoint, with mosaic disabled:
python training/train.py --data training/data.yaml --model models/best.pt --accuracy-preset --epochs 15 --name mixed_refine
```

## Held-out evaluation and failure review

```powershell
python training/evaluate.py --data training/data.yaml --weights runs/pothole/mixed_domain/weights/best.pt --imgsz 960 --output outputs/metrics/mixed_test.json
```

The script prints/saves precision, recall, mAP50 and mAP50-95 on the labelled test
split. It also saves up to 100 failure-case images and a JSON report beside the
metrics, with green ground-truth boxes and red predictions. Failure matching is
one-to-one class-0 IoU at 0.50 using confidence 0.15; override with
`--failure-iou`, `--failure-confidence`, or `--max-failures` (0 disables collection).
The case report records false positives and missed boxes, including negatives with
spurious predictions. Missing or invalid labels stop collection rather than
silently treating an unlabelled image as a negative.

Report metrics separately for phone/top-down, dashcam, small/distant and wet/low-light
subsets as well as overall. Compare the shipped and candidate checkpoints on the
same untouched sets; reject a gain that materially harms recall or runtime.
Do not replace `models/best.pt` automatically. Record results in `experiments.md`.
