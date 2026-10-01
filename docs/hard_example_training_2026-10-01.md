# Targeted fine-tuning: measured result

The additional training experiment **did not improve the model**. The application's
existing checkpoint is unchanged. Publishing the training tools does not install
the rejected candidate or run training on Streamlit Cloud. No theme change was
made. See `training/experiments.md` for the full recipe, hashes and diagnostics.

## What actually ran

- Mined all 6,395 existing training images using reviewed YOLO labels, not guessed
  pseudo-labels. Found 1,521 positive photos containing at least one missed box.
- Checked training images against validation/test by exact content hash. No overlap
  was found; existing images, labels and raw downloads were not edited.
- Built a 2,400-photo source-diverse replay set with 600 missed-positive photos
  repeated once, plus regular positives and negatives (including 84 hard negatives).
  This includes 210 China_Drone images but not the user's original failed phone photos.
- Attempted 960px training, which failed on laptop GPU/host-memory limits before
  completing an epoch. Retried at 640px/batch 2 with PyTorch CUDA AMP.
- Saved epoch 1, then recovered from another host-memory failure by resuming its
  optimizer/checkpoint, disabling training plots and releasing unused caches.
- Completed **7 epochs total**, with early stopping. The best checkpoint was epoch 1.

## Full validation comparison

7,718 images / 1,273 ground-truth potholes, same 640px/batch-16/FP16 evaluator.
These are **validation detection metrics**, not overall accuracy or test metrics.

| Metric | Original | Best fine-tuned candidate |
|---|---:|---:|
| Precision | 52.79% | 50.95% |
| Recall | 43.99% | 41.21% |
| mAP50 | 44.60% | 39.96% |
| mAP50-95 | 19.94% | 16.54% |

The candidate failed the predeclared validation gate and was **not installed**.
Test-set/application comparisons were skipped once that gate failed. Saved results:
`outputs/metrics/hard_examples_resume_20261001_result.json`.

## What remains useful

The project now has reusable train-only failure mining, exact held-out overlap
checks, auditable replay manifests, a low-memory fine-tuning preset, and a serial
comparison runner that refuses mismatched evaluation settings. Unit tests cover
the overlap checks, label validation, replay weighting, isolated validation subset,
cache-cleanup callbacks and promotion gate.

To address the specific phone-close-up misses, the next experiment needs new,
reviewed close-up training examples and an independent close-up holdout. The current
result does not prove a cause, but does rule out claiming that this replay recipe
or simply more epochs improved the shipped checkpoint.
