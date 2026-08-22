# Training experiments

Record every material experiment before comparing checkpoints.

| Date | Name | Dataset version | Base model | Epochs | Image size | Batch | Precision | Recall | mAP50 | mAP50–95 | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-08-21 | baseline_cuda | RDD2022 India / D40 only | YOLO11n | 25 | 640 | 16 | 0.3582 | 0.2729 | 0.2519 | 0.0993 | PyTorch 2.11.0+cu130; selected epoch 24; held-out test metrics |
| 2026-08-21 | balanced_finetune | RDD2022 India / D40 only / 1.5:1 train negatives | YOLO11n | 30 | 640 | 16 | 0.3927 | 0.4038 | 0.3790 | 0.1393 | Fine-tuned from baseline; held-out 745-image India test metrics |
| 2026-08-21 | full_s_25 | RDD2022 all seven countries / D40 only / 1.5:1 train negatives | YOLO11s | 25 | 640 | 16 | 0.5027 | 0.3712 | 0.3816 | 0.1639 | Previous checkpoint; held-out 3,925-image multinational test metrics |
| 2026-08-22 | accuracy_finetune_640 | RDD2022 all seven countries / D40 only / 1.5:1 train negatives | full_s_25 checkpoint | 8 | 640 | 16 | 0.4820 | 0.4015 | 0.4011 | 0.1769 | Installed checkpoint; low-rate AdamW refinement without mosaic; held-out 3,925-image multinational test metrics |

Keep related frames and images from the same road location in one split. Change one major
factor per experiment so improvements can be attributed honestly.
