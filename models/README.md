# Model weights

The trained single-class YOLO11s checkpoint is stored here as:

```text
models/best.pt
```

The model uses class `0` for `pothole`. The application intentionally shows a clear
missing-model message instead of downloading unrelated generic weights or returning fake
detections.

It is a YOLO11s checkpoint fine-tuned on all seven RDD2022 country subsets for 60 epochs
(run `s_640_long`), followed by a 5-epoch low-rate AdamW refinement stage
(`s_640_long_refine`), selected by best validation fitness. Held-out test metrics are
recorded in `docs/model_card.md`. To use a checkpoint elsewhere, set
`POTHOLE_MODEL_PATH` to its full path before starting Streamlit.

If you install a checkpoint from a run you stopped early, strip its optimizer state
first — Ultralytics only does that automatically when a run finishes normally, and the
difference here was 76 MB versus 19 MB:

```powershell
python -c "from ultralytics.utils.torch_utils import strip_optimizer; strip_optimizer('models/best.pt')"
```
