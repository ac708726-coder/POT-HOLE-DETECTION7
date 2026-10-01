# Divot

Pothole detection for road photos and videos.

Divot lets you upload a road image or a short video, mark the potholes the model
finds, and review the results before saving or downloading them. It brings the
detector, scan controls, and inspection history into one Streamlit app instead of
requiring a separate script for every scan.

The detector is a trained **YOLO11s model running on PyTorch**. The weights are
included in this repository: you do not need the training dataset or a separate
YOLO program to use the app. It runs on a CPU, with GPU acceleration available on
compatible NVIDIA systems.

## What you can do

- **Overview:** choose an image or video inspection.
- **Image detection:** compare the original and annotated photo, review per-box
  confidence and measurements, and download the annotated image.
- **Video detection:** process selected frames and download an annotated video.
  Approximate tracking associates detections across frames, but does not guarantee
  an exact count of unique potholes.
- **Detection history:** save scan summaries to your account, filter previous
  inspections, and export the visible records as CSV.

Image results show the pothole count, average and maximum confidence, inference
time, and individual detections. Changing the upload, mode, or confidence clears
the old result so it is not mistaken for a fresh scan.

## Choosing a scan mode

| Mode | What it runs | When to use it |
|---|---|---|
| Fast | One 640px pass | Quick checks and video processing |
| Balanced | A 640px pass with test-time augmentation | A starting point for still images |
| Thorough | Augmented 640px and 1280px passes | A slower second look at missed or small potholes |

Overlapping boxes are filtered with class-agnostic non-maximum suppression at
IoU 0.50. Confidence defaults to **0.35**, with a minimum of **0.15**. Lowering it
can recover missed potholes, but also produces more false positives. Thorough
is not always more accurate than Balanced, and zero detections does not mean
the road is undamaged.

The highest Thorough resolution is `THOROUGH_IMAGE_SIZE` in [config.py](config.py).
Increasing it costs more time and memory, especially on CPU-only hosting.

## Run it locally

Use **Python 3.12**. Run these commands where you want the project. If you already
have a copy, skip cloning and open that folder instead.

### Windows PowerShell

```powershell
git clone https://github.com/ac708726-coder/POT-HOLE-DETECTION7.git
cd POT-HOLE-DETECTION7
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### Linux

```bash
git clone https://github.com/ac708726-coder/POT-HOLE-DETECTION7.git
cd POT-HOLE-DETECTION7
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`.
Create an account, sign in, and choose a scan. In VS Code, select the project's
`.venv` interpreter rather than a different system Python installation.

Supported uploads are JPG, JPEG, and PNG images up to 10 MB, and MP4, MOV, and AVI
videos up to 200 MB and five minutes. Longer videos and Thorough scans can be
slow on a laptop or a free cloud instance.

### Optional: use an NVIDIA GPU

`requirements.txt` installs CPU builds of PyTorch and torchvision for Streamlit
Cloud and CPU-only machines. After installing the requirements, replace those
two packages with the matching CUDA builds if your driver supports CUDA 13.0.
For Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade torch==2.11.0+cu130 torchvision==0.26.0+cu130 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

On Linux, use `.venv/bin/python` for those commands. A `+cu130` version and
`CUDA available: True` confirm that the environment can use the GPU.
Reinstalling `requirements.txt` restores its CPU pins.

## Accounts and saved history

Passwords are stored as salted scrypt hashes, not plain text. Each account can
read its own saved history. Accounts, login sessions, and scan summaries live
in `database/potholes.db`; generated media goes under `outputs/`.

On a deployed app, those files are on the **server**, not on every visitor's
laptop. Downloading a result saves a copy to your device.

To stay signed in after refreshing, set `DIVOT_COOKIE_SIGNING_KEY` to a stable,
random secret of at least 32 bytes. Use Cloud's **App settings → Secrets**, an
environment variable, or a gitignored local `.streamlit/secrets.toml`. Generate
your value privately:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Login cookies expire after seven days. Sign out revokes the session; rotating
the signing key invalidates existing cookies. Without a valid key, sign-in
lasts only for the current session. Cookies contain no passwords and are signed,
not encrypted. The JavaScript cookie controller cannot set HttpOnly; the app
should not render untrusted JavaScript.

**Streamlit Community Cloud does not provide durable local storage.** A rebuild,
restart, or replacement container can remove the database and generated media.
Cookies cannot preserve a deleted account. Use persistent storage and backups
if history must survive redeploys. Never commit a signing key, real database,
or `.streamlit/secrets.toml` to GitHub.

## The model and its results

The current checkpoint was fine-tuned on the seven country subsets of
[RDD2022](https://github.com/sekilab/RoadDamageDetector). The preparation script
converts Pascal VOC XML annotations into YOLO labels and keeps **D40: pothole**
as class 0. Roads without a labelled pothole are included as negative examples.
The prepared training split contains 6,395 images and 4,611 pothole boxes.

Training used 60 base epochs followed by five low-learning-rate refinement
epochs. The installed checkpoint was selected by validation fitness. Its
recorded evaluation on a separate, labelled **3,925-image test split** was:

| Metric | Result |
|---|---:|
| Precision | 55.20% |
| Recall | 41.36% |
| mAP at IoU 0.50 | 43.34% |
| mAP at IoU 0.50–0.95 | 19.92% |

These are object-detection metrics, not a single overall accuracy percentage.
They describe checkpoint evaluation, not a promise about every photo or scan
mode. The split includes negative road images, not just pothole photos.
The [model card](docs/model_card.md) records the evaluation and earlier comparisons.

A later fix corrected the RGB/BGR colour order passed to YOLO. On a separate
1,000-image validation sample at confidence 0.35, Balanced found **79 correctly
matched boxes instead of 52**, with 57 false boxes instead of 56. This improved
the inference path without replacing the model. It is not a new held-out test
score. See the [detection report](docs/detection_improvement_2026-10-01.md).

The latest hard-example fine-tuning experiment made the full-validation results
worse, so that candidate was rejected and **the original application model was
kept**. The [experiment report](docs/hard_example_training_2026-10-01.md) records
what was tried and why it was not promoted.

## Where it still struggles

Phone close-ups, top-down views, water-filled holes, poor lighting, and small or
distant potholes can be missed. Shadows, repaired patches, drains, and puddles
can produce false detections. More epochs alone have not solved those cases.
The next useful experiment needs reviewed examples of the missing viewpoints
and a separate holdout to check whether they actually help.

The severity label uses the box's share of the image. It does **not** measure
pothole depth, real-world size, or structural danger. Divot is a review tool,
not a road-safety certification or a replacement for an on-site inspection.

## Training and development

You only need the dataset if you want to train or evaluate models. Download it
from the [official RDD2022 repository](https://github.com/sekilab/RoadDamageDetector)
or [Figshare record](https://doi.org/10.6084/m9.figshare.21431547). The unlabelled
official test images are not the labelled holdout used for the scores above.

The [training guide](training/README.md) covers preparation, additional close-up
data, augmentations, evaluation, and hard-example mining. Scripts never install
a candidate into the app automatically, and starting Streamlit does not launch
training. Keep the existing model until a candidate passes a like-for-like
evaluation.

The project uses Streamlit for the interface, PyTorch and Ultralytics for the
detector, Pillow and OpenCV for media, and SQLite for accounts and history.
No TensorFlow or ONNX runtime is used.

- `app.py` and `pages/`: sign-in, navigation, and inspection screens.
- `utils/`: inference, image/video processing, tracking, auth, and storage.
- `models/best.pt`: the checkpoint used by the app.
- `scripts/`: dataset preparation, benchmarks, and database backups.
- `training/`: training recipes, evaluation, and experiment records.
- `tests/`: regression tests, including real-model blank-image smoke tests.

From the project environment, run:

```bash
python -m pytest -q
python -m ruff check .
python -m black --check .
```

The latest verified suite passed **127 tests**. The real-model smoke tests use
the included checkpoint. Training data, experiment checkpoints, generated
media, and secrets are excluded from Git.

## Deploying on Streamlit Cloud

Select this repository, the `main` branch, `app.py`, and Python 3.12. Add the
private signing key through Cloud Secrets for refresh-persistent login. Cloud
uses the CPU dependencies in `requirements.txt` and system libraries in
`packages.txt`.

Cloud follows its configured branch: an app still tracking
`deploy/streamlit-cloud` will not pick up a merge into `main`. Check build logs
after a release, and remember the storage limitations above before relying on
saved history. The [deployment guide](docs/deployment.md) covers the full setup,
backups, and rollback.
