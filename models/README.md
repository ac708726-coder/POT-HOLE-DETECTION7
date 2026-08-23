# Model weights

The trained single-class YOLO11s checkpoint is stored here as:

```text
models/best.pt
```

The model uses class `0` for `pothole`. The application intentionally shows a clear
missing-model message instead of downloading unrelated generic weights or returning fake
detections.

It is a YOLO11s checkpoint fine-tuned on all seven RDD2022 country subsets for 60 epochs
(run `s_640_long`), selected by best validation fitness. Held-out test metrics are
recorded in `docs/model_card.md`. To use a checkpoint elsewhere, set
`POTHOLE_MODEL_PATH` to its full path before starting Streamlit.
