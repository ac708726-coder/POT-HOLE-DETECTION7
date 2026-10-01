# Detection preprocessing correction

## What changed

The application decoded images as RGB and passed `np.asarray(image)` to YOLO.
Ultralytics assumes NumPy images are **BGR**, then reverses them to RGB model
tensors. This fed the model swapped red/blue channels. See the
[official input-source contract](https://docs.ultralytics.com/modes/predict/#inference-sources)
and the pinned installation's `ultralytics/engine/predictor.py` preprocessing.

`utils/image_processor.py:image_to_bgr()` now returns contiguous BGR buffers.
Single-image inference, batched video inference and failure-case evaluation use
the correct colour order. UI images remain RGB. Fast mode explicitly resets to
640 px without augmentation instead of inheriting a previous predictor's settings.
Invalid/degenerate boxes are discarded; the existing NMS and thresholds stay intact.
The inference revision invalidates old results in open image/video sessions.

This preprocessing experiment did not retrain, download data, replace the
checkpoint or change the theme. Subsequent Git publication does not change the
original model; the model SHA-256 before and after is:

`09b78ce3bf3643e840e32626a94f4e218f1feee501ba2879b48fdf37082fe2a0`

## Measured confirmation

Run on 2026-10-01 using the **validation** split, not the held-out test split.
The reproducible sample contains 1,000 images: 100 positives, 900 negatives,
172 labelled pothole boxes. File selection uses seed 99 and SHA-256 ordering.
Sample filename SHA-256:

`203a4a2caa397d574beae0e2e6da1f5cf6f454d01b114255f00c9a738c1c96a0`

Both paths use the same images, checkpoint, CUDA/FP16 runtime, confidence **0.35**,
class-agnostic NMS IoU 0.50 and one-to-one ground-truth matching IoU 0.50.
The benchmark's `legacy_*` adapter reproduces the colour bug without restoring it
in the application. cuDNN autotuning is disabled only in this offline benchmark
to bound memory for diverse aspect ratios. These GPU numbers are not Cloud CPU
benchmarks or full-dataset test mAP/accuracy.

| Mode | Correct boxes before → after | False boxes before → after | Precision before → after | Recall before → after |
|---|---:|---:|---:|---:|
| Fast | 46 → 69 | 37 → 42 | 55.42% → 62.16% | 26.74% → 40.12% |
| Balanced | 52 → 79 | 56 → 57 | 48.15% → 58.09% | 30.23% → 45.93% |
| Thorough | 65 → 92 | 148 → 164 | 30.52% → 35.94% | 37.79% → 53.49% |

Balanced recovered **51.9% more correctly matched boxes**, while false boxes
increased by one across all 1,000 images. This is a relative count gain, not
"52% more accuracy". Thorough maximizes recall at a substantial precision cost;
Balanced at 0.35 remains the cleaner starting point. Lowering confidence still
adds false positives: corrected Thorough at 0.15 scored 117 true boxes and 567
false boxes on this sample. Do not mistake more drawn boxes for more correct boxes.

The initial 500-image pilot also showed gains with the colour fix. Overlapping
crop and additional-scale trials introduced too much noise before the fix, so
they were not enabled in the application. A colour-corrected crop trial was
interrupted by memory exhaustion while GPU workloads overlapped; it provides no
completed evidence for promotion. Crop code remains benchmark-only. The final
1,000-image comparison completed with one GPU workload at a time.

## Reproduce and inspect

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_detection_recall.py --positives 100 --negatives 900 --seed 99 --output outputs/metrics/color_fix_confirmation.json
```

The ignored local outputs include the summary JSON, per-image predictions for
each mode, and before/after examples in
`outputs/metrics/color_fix_confirmation_recoveries/`. Green boxes are dataset
ground truth; red boxes are predictions. Recovery examples are illustrative
selections, not the basis of the aggregate scores.

Regression tests check coloured inputs (not only gray/black inputs), RGB UI
preservation, batched video colour order, malformed boxes, benchmark scoring and
the existing real-checkpoint blank-image smoke tests in all three modes.

## Remaining limitations

This fixes inference preprocessing, not the RDD-heavy training distribution.
Phone/top-down close-ups, water-filled holes and unfamiliar surfaces can still be
missed. The user's original failed uploads were unavailable for direct retesting.
The mixed-viewpoint plan in `training/README.md` remains the next training step;
no new overall accuracy or held-out mAP is claimed from this change.
