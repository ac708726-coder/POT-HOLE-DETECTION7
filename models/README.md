# Model weights

The trained single-class YOLO11n checkpoint is stored here as:

```text
models/best.pt
```

The model uses class `0` for `pothole`. The application intentionally shows a clear
missing-model message instead of downloading unrelated generic weights or returning fake
detections.

It was trained for 25 epochs on RDD2022 India and selected at epoch 24. Held-out test
metrics are recorded in `docs/model_card.md`. To use a checkpoint elsewhere, set
`POTHOLE_MODEL_PATH` to its full path before starting Streamlit.
