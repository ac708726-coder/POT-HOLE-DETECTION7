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
- Training: 25 base epochs plus 8 low-rate refinement epochs, batch 16, seed 42, CUDA AMP
- Prepared training split: 6,395 images and 4,611 pothole boxes
- Selected checkpoint: refinement epoch 8 by validation fitness

## Intended use

The detector is intended for reviewing visible road potholes in uploaded images and
videos. It supports inspection, research, and demonstrations. It is not a safety system
and must not be the sole basis for maintenance or driving decisions.

## Evaluation

The source is RDD2022 (CC BY 4.0 on its Figshare record). Only Pascal VOC
objects named `D40` are converted to the application class `pothole`. Images containing
only other road-damage classes are retained as negative examples.

| Metric | Held-out test result |
|---|---:|
| Precision | 0.4820 |
| Recall | 0.4015 |
| mAP@0.50 | 0.4011 |
| mAP@0.50–0.95 | 0.1769 |
| GPU inference time | 5.7 ms/image |

Metrics were produced by `training/evaluate.py` on the untouched 3,925-image
multinational test split and are saved in `outputs/metrics/test_metrics.json`. These
values remain below the PRD's aspirational quality targets. Confidence changes the
precision/recall tradeoff and is not equivalent to overall accuracy.

## Known limitations

Shadows, repaired patches, drains, debris, water, unusual surfaces, poor lighting, small
or distant damage, and camera motion can cause false positives or missed detections.
Apparent severity uses bounding-box area and does not measure physical depth. Video
tracking can count one pothole more than once after occlusion or rapid camera movement.
