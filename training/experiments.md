# Training experiments

Record every material experiment before comparing checkpoints.

| Date | Name | Dataset version | Base model | Epochs | Image size | Batch | Precision | Recall | mAP50 | mAP50–95 | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-08-21 | baseline_cuda | RDD2022 India / D40 only | YOLO11n | 25 | 640 | 16 | 0.3582 | 0.2729 | 0.2519 | 0.0993 | PyTorch 2.11.0+cu130; selected epoch 24; held-out test metrics |
| 2026-08-21 | balanced_finetune | RDD2022 India / D40 only / 1.5:1 train negatives | YOLO11n | 30 | 640 | 16 | 0.3927 | 0.4038 | 0.3790 | 0.1393 | Fine-tuned from baseline; held-out 745-image India test metrics |
| 2026-08-21 | full_s_25 | RDD2022 all seven countries / D40 only / 1.5:1 train negatives | YOLO11s | 25 | 640 | 16 | 0.5027 | 0.3712 | 0.3816 | 0.1639 | Previous checkpoint; held-out 3,925-image multinational test metrics |
| 2026-08-22 | accuracy_finetune_640 | RDD2022 all seven countries / D40 only / 1.5:1 train negatives | full_s_25 checkpoint | 8 | 640 | 16 | 0.4820 | 0.4015 | 0.4011 | 0.1769 | Previous checkpoint; low-rate AdamW refinement without mosaic; held-out 3,925-image multinational test metrics |
| 2026-08-23 | s_640_long | RDD2022 all seven countries / D40 only / 1.5:1 train negatives | YOLO11s | 60 | 640 | 16 | 0.5300 | 0.4212 | 0.4260 | 0.1960 | Installed checkpoint; length-only change vs full_s_25 (60 epochs, patience 20); PyTorch 2.11.0+cu130; held-out 3,925-image multinational test metrics |

Keep related frames and images from the same road location in one split. Change one major
factor per experiment so improvements can be attributed honestly.

## Planned runs

Queued to raise mAP above the `accuracy_finetune_640` baseline. Each run changes one
major factor so the gain can be attributed. Fill the table above with held-out test
metrics from `training/evaluate.py` as each finishes, then delete its row here.

| Order | Name | Changed factor | Command |
|---:|---|---|---|
| ~~1~~ | ~~`s_640_long`~~ | **DONE — see results table above. Won on all four metrics.** | — |
| 2 | `s_960_ms` | Resolution + multi-scale, from run 1's winner | `python training/train.py --model yolo11s.pt --epochs 60 --imgsz 960 --batch -1 --multi-scale --workers 8 --patience 20 --save-period 5 --name s_960_ms` |
| 3 | `m_640_long` | Backbone capacity only (YOLO11m vs YOLO11s) | `python training/train.py --model yolo11m.pt --epochs 60 --imgsz 640 --batch -1 --workers 8 --patience 20 --save-period 5 --name m_640_long` |
| 4 | `s_640_long_refine` | Low-rate refinement on the run 1 winner | `python training/train.py --model models/best.pt --epochs 10 --imgsz 640 --batch 16 --workers 8 --accuracy-preset --name s_640_long_refine` |

Run 1 answered its question: the previous checkpoints **were** undertrained. `full_s_25`
peaked at its own final epoch (val mAP50 0.4029) with its cosine schedule fully decayed,
and simply giving the same recipe a 60-epoch schedule lifted held-out test mAP50 from
0.4011 to 0.4260. Runs 2 and 3 are still independent of each other; if only one fits the
available GPU time, prefer run 2, since the model card names small and distant damage as
a known failure mode.

Run 4 is the natural follow-up rather than "more epochs" on run 1: run 1's schedule ran
to completion (final lr 0.000053) and its last ten epochs gained only +0.0083 val mAP50
against +0.1009 for epochs 20→30, so extending that schedule has nothing left to give. A
refinement stage needs a fresh low learning rate, which is what `--accuracy-preset`
provides — the same step that produced `accuracy_finetune_640` from `full_s_25`.

Use `--workers 8` on this 16-core machine. Run 1 used the default of 2 and sat at roughly
85% GPU utilization, so a modest throughput gain is available for free.

Combining runs 2 and 3 into a single `m_960_ms` run is tempting but leaves the result
uninterpretable, and at roughly 8x the activation cost of the 640 YOLO11s baseline it is
the most likely configuration to exhaust 8 GB of VRAM. Only try it after 2 and 3 report.

### Split composition

Measured from `data/processed` on 2026-08-23:

| Split | Images | Positives | Negatives | Boxes |
|---|---:|---:|---:|---:|
| train | 6,395 | 2,558 | 3,837 (60%) | 4,611 |
| val | 7,718 | 735 | 6,983 (90%) | 1,273 |
| test | 3,925 | 381 | 3,544 (90%) | 660 |

`prepare_dataset.py` caps negatives only in train (`--max-train-negative-ratio`), so val
and test keep every negative and val ends up larger than train. Two consequences worth
remembering when reading metrics:

- Reported precision is measured against 3,544 negative test images, so it is a strict,
  realistic number rather than a flattering one. Do not compare it to figures from
  positives-only benchmarks.
- Every epoch validates 7,718 images to score 735 positives, which is a large share of
  epoch time. Capping val negatives would speed up training, but it changes both
  best-checkpoint selection and val-fitness comparability, so it must be its own
  experiment rather than a silent change inside another run.

Notes for an 8 GB laptop GPU (RTX 4060):

- `--batch -1` uses Ultralytics AutoBatch, which profiles the model and targets roughly
  60% of free VRAM. Prefer it over guessing whenever `--imgsz` or the backbone changes.
- If a run still exits with a CUDA out-of-memory error, set an explicit small batch
  (`--batch 4`) rather than lowering `--imgsz`, so the changed factor stays isolated.
- `--save-period 5` plus `--resume` lets an interrupted overnight run continue instead of
  restarting: `python training/train.py --name <run-name> --resume`.
