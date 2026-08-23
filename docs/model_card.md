# Pothole detector model card

## Model details

- Task: single-class object detection
- Class: `pothole`
- Framework: Ultralytics YOLO / PyTorch
- Deep-learning runtime: PyTorch only
- Application checkpoint: `models/best.pt`
- Model family: YOLO11s, fine-tuned from pretrained PyTorch weights
- Input image size: 640 × 640
- Dataset: all seven RDD2022 country subsets, D40 converted to class `pothole`
- Training: 60 base epochs, then 5 low-rate AdamW refinement epochs; batch 16, seed 42,
  CUDA AMP, early-stopping patience 20
- Prepared training split: 6,395 images and 4,611 pothole boxes
- Selected checkpoint: best validation fitness from the `s_640_long_refine` stage
  (refinement epoch 2), with optimizer state stripped

## Intended use

The detector is intended for reviewing visible road potholes in uploaded images and
videos. It supports inspection, research, and demonstrations. It is not a safety system
and must not be the sole basis for maintenance or driving decisions.

## Evaluation

The source is RDD2022 (CC BY 4.0 on its Figshare record). Only Pascal VOC
objects named `D40` are converted to the application class `pothole`. Images containing
only other road-damage classes are retained as negative examples.

| Metric | Installed | 60-epoch stage | Earlier checkpoint |
|---|---:|---:|---:|
| Precision | **0.5520** | 0.5300 | 0.4820 |
| Recall | 0.4136 | 0.4212 | 0.4015 |
| mAP@0.50 | **0.4334** | 0.4260 | 0.4011 |
| mAP@0.50–0.95 | **0.1992** | 0.1960 | 0.1769 |
| GPU inference time | 5.7 ms/image | 5.7 ms/image | 5.5 ms/image |

Metrics were produced by `training/evaluate.py` on the untouched 3,925-image
multinational test split and are saved in `outputs/metrics/s_640_long_refine_test.json`,
`s_640_long_test.json`, and `baseline_recheck.json`. All three come from the same script
and split; the earlier checkpoint was re-evaluated on the same day and reproduced its
recorded figures exactly, so the columns are directly comparable.

Two changes produced the gain, in order. First, training length: the earlier checkpoint's
schedule ended while the model was still improving, and rerunning the same recipe for 60
epochs lifted mAP@0.50 from 0.4011 to 0.4260. Second, a low-rate AdamW refinement stage
on top of that added a further +0.0074. Recall is the one metric that did not improve in
the refinement stage, falling 0.4212 → 0.4136 while precision rose 0.5300 → 0.5520; the
stage traded a little recall for more precision.

These values remain below the PRD's aspirational quality targets.
Confidence changes the precision/recall tradeoff and is not equivalent to overall
accuracy. The test split is 90% negative images, so precision here is measured against
3,544 chances to produce a false positive and is a stricter figure than a
positives-only benchmark would give.

## Known limitations

Shadows, repaired patches, drains, debris, water, unusual surfaces, poor lighting, small
or distant damage, and camera motion can cause false positives or missed detections.
Apparent severity uses bounding-box area and does not measure physical depth. Video
tracking can count one pothole more than once after occlusion or rapid camera movement.
