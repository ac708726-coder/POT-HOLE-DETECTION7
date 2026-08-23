# Road Pothole Detection

A responsive Streamlit application for detecting visible potholes in uploaded road
images and videos with a trained single-class Ultralytics YOLO model.

PyTorch is the only deep-learning runtime. Ultralytics supplies the YOLO model API, while
training, inference, checkpoints, tensors, and CUDA execution are handled by PyTorch.
TensorFlow, Keras, and ONNX Runtime are not used.

## Current status

The application, utilities, RDD2022 data pipeline, tests, documentation, and trained
checkpoint are implemented. `models/best.pt` is a YOLO11s checkpoint fine-tuned on the
seven RDD2022 country subsets for 25 base epochs plus an 8-epoch low-rate refinement
stage, trained with PyTorch 2.11.0 on CUDA 13.0 and an RTX 4060 Laptop GPU. It improves
geographic coverage and strict localization, but it is not a production-quality
road-safety model.

Run every command below through the project virtual environment at `.venv`, which is
where the pinned CUDA build of PyTorch lives. A bare `python` on PATH may resolve to a
different interpreter with a CPU-only torch, which silently disables GPU inference.

## Setup on Windows PowerShell

```powershell
Set-Location -LiteralPath 'D:\POT HOLE DETECTION'
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local address printed by Streamlit. The app uses native Streamlit navigation
for Image detection, Video detection, and Detection history.

## Model placement

The trained single-class checkpoint is already stored at `models/best.pt`, where class
`0` represents `pothole`. To use another checkpoint, set an absolute path for the current
shell:

```powershell
$env:POTHOLE_MODEL_PATH = 'D:\models\pothole-best.pt'
streamlit run app.py
```

The project does not silently download generic weights because a general YOLO model is
not trained for the required pothole class.

## RDD2022 dataset and preparation

Use the public RDD2022 Road Damage Dataset. Start with the India subset from the
[official repository](https://github.com/sekilab/RoadDamageDetector#dataset), or use the
[official Figshare record](https://doi.org/10.6084/m9.figshare.21431547). Extract it so
the project contains:

```text
data/raw/RDD2022/India/train/images/
data/raw/RDD2022/India/train/annotations/xmls/
data/raw/RDD2022/India/test/images/
```

The official training annotations are Pascal VOC XML. `scripts/prepare_dataset.py`
keeps only `D40` pothole boxes, converts them to YOLO class `0`, and retains images with
other damage types as useful negative pothole examples. Official test images have no
annotations, so they are not used to report model accuracy.

```powershell
python scripts/prepare_dataset.py
python scripts/check_dataset.py --root data/processed
```

Preparation copies data into `data/processed`; it does not modify RDD2022. The default
split is approximately 70% train, 20% validation, and 10% test. Blocks of 50 nearby
numbered images stay together to reduce leakage from related road sequences. Re-running
against existing processed files requires the explicit `--clean` flag.

The installed checkpoint used all seven extracted subsets. To reproduce that prepared
dataset, run:

```powershell
python scripts/prepare_dataset.py --clean --countries China_Drone China_MotorBike Czech India Japan Norway United_States --max-train-negative-ratio 1.5
```

## Train, evaluate, and inspect

```powershell
python training/train.py --model yolo11s.pt --epochs 25 --imgsz 640 --batch 16 --name full_s_25
Copy-Item -LiteralPath 'runs\pothole\full_s_25\weights\best.pt' -Destination 'models\best.pt'
python training/evaluate.py
python training/predict_sample.py assets/sample_images
```

Training downloads the selected pretrained checkpoint if Ultralytics does not already
have it locally. Record every run in `training/experiments.md`. Evaluation writes
held-out metrics and plots under `outputs/metrics`.

`training/experiments.md` carries a **Planned runs** queue with the exact commands for
the next accuracy attempts. Before starting one, confirm all three prerequisites:

```powershell
python -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available())"
Test-Path 'data\processed\images\train'
python scripts\check_dataset.py --root data\processed
```

Training needs the prepared dataset, so run the `prepare_dataset.py` command above first
if `data/processed` does not exist. Long runs benefit from `--patience` (early stopping)
and `--save-period` (periodic checkpoints); an interrupted run continues with:

```powershell
python training/train.py --name <run-name> --resume
```

After a run beats the current checkpoint on the held-out split, promote it and re-record
the metrics in `docs/model_card.md` and this file:

```powershell
Copy-Item -LiteralPath 'runs\pothole\<run-name>\weights\best.pt' -Destination 'models\best.pt'
python training/evaluate.py
```

## Tests and code checks

```powershell
python -m pytest
ruff check .
black --check .
```

Tests use fake model outputs and do not require `best.pt`. The video integration test is
skipped when OpenCV is unavailable.

## Configuration

Central settings live in `config.py`:

- Image types: JPG, JPEG, PNG; maximum 10 MB.
- Video types: MP4, MOV, AVI; maximum 200 MB and five minutes.
- Default confidence: 0.35, a selective starting point that can be lowered when recall matters more.
- Default IoU threshold: 0.45.
- Database: `database/potholes.db`, created only when history is used.

Image inference includes three PyTorch modes: Fast uses one standard 640-pixel pass,
Balanced uses test-time augmentation, and Thorough merges augmented 640- and 960-pixel
passes. Overlapping boxes from multiple passes are merged with NMS. Video defaults to
Fast mode to preserve throughput.
CUDA laptops automatically use FP16 inference and four-frame video batching. Balanced
video mode also batches, at half that size, because test-time augmentation multiplies
activation memory per frame. CPU-only laptops stay on FP32 with batch size one, so the
same code remains portable.

Both detection pages warm the model up once while the upload controls are on screen, so
the first real scan does not pay CUDA context and autotuning cost. Warm-up is best
effort: if the checkpoint is missing or broken, the page still renders and the error is
reported by the scan itself.

Verify the GPU path is actually active before benchmarking, because installing
`requirements.txt` without the CUDA index URL above silently yields a CPU-only build:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

A `+cpu` version string or `False` means inference runs on CPU regardless of the
available hardware.

## Limitations

Detection quality depends on the final dataset and checkpoint. Shadows, patches,
drains, debris, water, lighting, and camera angle can cause errors. Apparent severity is
based only on relative box area and does not measure depth or real dimensions. Video
tracking is an estimate. The tool does not guarantee road safety or make maintenance
decisions.

## Held-out test results

The selected checkpoint was evaluated once on the untouched 3,925-image multinational
held-out split containing 660 pothole boxes.

| Metric | Installed | 60-epoch stage | Earlier checkpoint |
|---|---:|---:|---:|
| Precision | **0.5520** | 0.5300 | 0.4820 |
| Recall | 0.4136 | 0.4212 | 0.4015 |
| mAP@0.50 | **0.4334** | 0.4260 | 0.4011 |
| mAP@0.50–0.95 | **0.1992** | 0.1960 | 0.1769 |
| GPU inference time | 5.7 ms/image | 5.7 ms/image | 5.5 ms/image |

All columns come from the same script and the same split, evaluated on the same day, so
they are directly comparable. Two changes produced the gain: training length (the earlier
checkpoint's schedule ended while the model was still improving), then a low-rate AdamW
refinement stage on top. Recall is the one metric the refinement stage did not improve —
it traded a little recall for more precision.

These results do not meet the aspirational PRD targets. Improving them requires more
training and data balancing/curation; they must not be presented as production accuracy.
The confidence control changes the precision/recall tradeoff; a confidence percentage
must not be presented as overall model accuracy.

See `docs/user_guide.md`, `docs/model_card.md`, and `docs/PRD.md` for further details.
