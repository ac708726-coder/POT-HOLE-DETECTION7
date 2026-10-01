# Training experiments

Record every material experiment before comparing checkpoints.

## 2026-10-01 targeted-replay experiment (completed; rejected)

User explicitly requested additional training on missed potholes. Start from the
installed checkpoint (SHA256
`09b78ce3bf3643e840e32626a94f4e218f1feee501ba2879b48fdf37082fe2a0`). Mine
misses from TRAIN only, using correct file-path decoding at 640px/confidence 0.35.
Keep all original validation/test photos held out, checking exact-content overlap
before building the new manifest. Fine-tune a separate 960px candidate for up to
10 epochs with `--hard-example-preset`, batch 8 and workers 0 to bound laptop memory.
This changes sampling, resolution and augmentation together, so an improvement
cannot be attributed to a single factor.

Predeclared validation gate at 640px: mAP50 gain >= 0.005, recall gain > 0,
precision loss <= 0.02. Only the frozen validation winner reaches a final
same-settings, full-test comparison and application operating-point check. Do not
install a candidate that fails the gate or worsens held-out detection results.
The current `models/best.pt` stays untouched during mining/training/evaluation.
No original failed phone-photo uploads are available: this experiment addresses
analogous labelled RDD2022 training failures, not proven phone-domain coverage.

Comparison uses batch 16, FP16 (`--quantize 16`) and workers 0 for both checkpoints.
The first slow FP32/batch-4 baseline diagnostic was interrupted before producing
metrics; no comparison is made against that unfinished run or historic scores.

To bound epoch time, checkpoint selection uses `--validation-subset 2000`
(200 positives/1,800 negatives, seed 73). This is a separately recorded manifest
within the existing validation split, never training data. The original full
validation split is used for the promotion gate; the test split remains untouched.

The first batch-8 attempt failed before epoch 1: CUDA OOM, followed by host-memory
allocation failure during Ultralytics' automatic batch-4 retry. No candidate
checkpoint was produced. Restart cold at batch **2**, workers 0, same 960px inputs
and pre-generated validation manifest, under name `hard_examples_960_b2_20261001`.
Keep the failed run folder for diagnostics; no checkpoint was installed.

The batch-2 960px run also failed during epoch 1 with OpenCV host-memory allocation
failure, before saving a checkpoint. A 640px/batch-2 cold restart
(`hard_examples_640_b2_20261001`) progressed successfully beyond the initial
optimizer steps. It retains the same mined/replay images, low-LR preset and fixed
2,000-image checkpoint-selection validation manifest. 640px matches Fast/Balanced
application inference. Neither failed 960px run altered `models/best.pt`.

The 640px run saved epoch 1, then hit host-memory allocation failure during epoch 2.
Its full validation evaluation is a diagnostic, not evidence of completed training.
Before resuming, add explicit unused-cache cleanup at epoch boundaries/every 50
batches and disable training plots. Preserve all model tensors, gradients and
optimizer state; do not change data, resolution, learning-rate schedule or epochs.

### Final outcome

The resumed run completed **7 epochs total**, then stopped automatically (patience
4; resume initializes the stopper independently of its pre-interruption history).
The best checkpoint remained epoch 1. A planned stop request arrived after the
process had already finished, so it did not terminate the run. Explicit cache
cleanup and disabled plots allowed the resumed stages to complete without another
memory failure. Normal completion stripped the candidate's optimizer state.

Full original validation split: 7,718 images / 1,273 labelled pothole boxes.
Both checkpoints used 640px, batch 16, FP16, workers 0 and the same evaluator/data
config. The candidate's post-completion evaluation reproduced its epoch-1 metrics
exactly; the later epochs did not yield a better checkpoint.

| Metric | Current model | Candidate | Absolute change |
|---|---:|---:|---:|
| Precision | 0.527945 | 0.509514 | -0.018431 |
| Recall | 0.439906 | 0.412090 | -0.027815 |
| mAP50 | 0.445983 | 0.399606 | -0.046378 |
| mAP50-95 | 0.199378 | 0.165362 | -0.034016 |

**Rejected.** `models/best.pt` remains byte-for-byte unchanged (SHA256 above).
The candidate failed the validation gate, so test-set and application-benchmark
comparisons were intentionally not run. Do not present these validation results
as test accuracy, or assume that hard-example replay/extra epochs help this model.

Artifacts (Git-ignored):
- `data/splits/hard_examples_20261001/mining.json`: all 6,395 TRAIN images mined;
  1,521 positive images had at least one miss; no exact held-out overlap found.
- Training selection: 2,400 unique photos, 1,200 positives/1,200 negatives;
  600 hard positives repeated once (3,000 training appearances), 84 hard negatives,
  210 China_Drone photos. Manifest SHA256
  `54cd2a931900d2fb0179ce00e8c28f057b13793f6670dbdc2602517d3d1943ff`.
- Checkpoint-selection validation: 200 positives/1,800 negatives, manifest SHA256
  `ddbe1208c3ed00f007065994e35f48ec90a41a0ec8850500f6291f1c67ac01ec`.
- `runs/pothole/hard_examples_640_b2_20261001/weights/best.pt`: rejected candidate,
  SHA256 `7667c9f632af3804aff602b8ae208e74a30d7adaf2dd4c10843c8d36c9fc0672`.
- `outputs/metrics/hard_examples_resume_20261001_result.json`: final comparison.

Next useful experiment: review annotations/viewpoint coverage and add independently
labelled phone/close-up examples with a separate untouched holdout. More epochs of
this same replay recipe are not supported by the measured result. The 665-image
[Roboflow public pothole dataset](https://public.roboflow.com/object-detection/pothole)
is a candidate for review, listed under ODbL v1.0 by its source; it was not downloaded
or used in this run. No Git commit, GitHub push, deployment or theme change was made.

| Date | Name | Dataset version | Base model | Epochs | Image size | Batch | Precision | Recall | mAP50 | mAP50–95 | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-08-21 | baseline_cuda | RDD2022 India / D40 only | YOLO11n | 25 | 640 | 16 | 0.3582 | 0.2729 | 0.2519 | 0.0993 | PyTorch 2.11.0+cu130; selected epoch 24; held-out test metrics |
| 2026-08-21 | balanced_finetune | RDD2022 India / D40 only / 1.5:1 train negatives | YOLO11n | 30 | 640 | 16 | 0.3927 | 0.4038 | 0.3790 | 0.1393 | Fine-tuned from baseline; held-out 745-image India test metrics |
| 2026-08-21 | full_s_25 | RDD2022 all seven countries / D40 only / 1.5:1 train negatives | YOLO11s | 25 | 640 | 16 | 0.5027 | 0.3712 | 0.3816 | 0.1639 | Previous checkpoint; held-out 3,925-image multinational test metrics |
| 2026-08-22 | accuracy_finetune_640 | RDD2022 all seven countries / D40 only / 1.5:1 train negatives | full_s_25 checkpoint | 8 | 640 | 16 | 0.4820 | 0.4015 | 0.4011 | 0.1769 | Previous checkpoint; low-rate AdamW refinement without mosaic; held-out 3,925-image multinational test metrics |
| 2026-08-23 | s_640_long | RDD2022 all seven countries / D40 only / 1.5:1 train negatives | YOLO11s | 60 | 640 | 16 | 0.5300 | 0.4212 | 0.4260 | 0.1960 | Length-only change vs full_s_25 (60 epochs, patience 20); PyTorch 2.11.0+cu130; held-out 3,925-image multinational test metrics |
| 2026-08-23 | s_640_long_refine | RDD2022 all seven countries / D40 only / 1.5:1 train negatives | s_640_long checkpoint | 5 of 10 | 640 | 16 | 0.5520 | 0.4136 | 0.4334 | 0.1992 | Installed checkpoint; low-rate AdamW refinement without mosaic; stopped early at operator request, best epoch 2 by validation fitness; optimizer state stripped after the interrupted run; held-out 3,925-image multinational test metrics |

Keep related frames and images from the same road location in one split. Change one major
factor per experiment so improvements can be attributed honestly.

## Planned runs

Queued to raise mAP above the `accuracy_finetune_640` baseline. Each run changes one
major factor so the gain can be attributed. Fill the table above with held-out test
metrics from `training/evaluate.py` as each finishes, then delete its row here.

| Order | Name | Changed factor | Command |
|---:|---|---|---|
| ~~1~~ | ~~`s_640_long`~~ | **DONE — won on all four metrics.** | — |
| ~~4~~ | ~~`s_640_long_refine`~~ | **DONE — installed. Stopped at 5 of 10 epochs; still improved mAP.** | — |
| 2 | `s_960_ms` | Resolution + multi-scale, from the installed checkpoint | `python training/train.py --model yolo11s.pt --epochs 60 --imgsz 960 --batch -1 --multi-scale --workers 2 --patience 20 --save-period 5 --name s_960_ms` |
| 3 | `m_640_long` | Backbone capacity only (YOLO11m vs YOLO11s) | `python training/train.py --model yolo11m.pt --epochs 60 --imgsz 640 --batch -1 --workers 2 --patience 20 --save-period 5 --name m_640_long` |

Run 1 answered its question: the previous checkpoints **were** undertrained. `full_s_25`
peaked at its own final epoch (val mAP50 0.4029) with its cosine schedule fully decayed,
and simply giving the same recipe a 60-epoch schedule lifted held-out test mAP50 from
0.4011 to 0.4260. Runs 2 and 3 are still independent of each other; if only one fits the
available GPU time, prefer run 2, since the model card names small and distant damage as
a known failure mode.

Run 4 was the natural follow-up rather than "more epochs" on run 1: run 1's schedule ran
to completion (final lr 0.000053) and its last ten epochs gained only +0.0083 val mAP50
against +0.1009 for epochs 20→30, so extending that schedule had nothing left to give. A
refinement stage needs a fresh low learning rate, which is what `--accuracy-preset`
provides — the same step that produced `accuracy_finetune_640` from `full_s_25`. It
worked: five refinement epochs added +0.0074 test mAP50 on top of run 1.

Two operational notes from these runs:

- A first attempt at run 4 with `--workers 8` died with `CUDA error: unknown error`
  inside the AdamW step at epoch 0. The identical command with the default `--workers 2`
  ran cleanly, so the queue above stays on 2. The root cause was never confirmed and the
  GPU tested healthy immediately afterwards, so treat this as suspected rather than
  established.
- Ultralytics only strips optimizer state from `best.pt` when a run finishes normally. A
  run stopped early leaves a checkpoint roughly 4x larger (76 MB here) that still carries
  AdamW state. Strip it before installing, or the extra weight lands in git history:
  `python -c "from ultralytics.utils.torch_utils import strip_optimizer; strip_optimizer('models/best.pt')"`.
  Stripping left every test metric bit-identical.

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
