# Road Pothole Detection and Monitoring System — implementation scope

This repository implements the supplied Product Requirements Document dated 21 August
2026. The canonical source provided for this build is
`D:/PotholeDetection/pothole_detection_prd.md`.

## Required MVP

- Accept readable JPG, JPEG, and PNG road images up to the configured limit.
- Load a trained single-class YOLO model from `models/best.pt`.
- Display bounding boxes, confidence scores, detection count, and processing time.
- Let the user change the minimum confidence threshold.
- Provide the annotated image as a download.
- Handle invalid input, missing weights, inference failure, and no-detection results.
- Keep validation, inference, processing, storage, and interface code separate.

## Included extensions

The project also includes uploaded-video processing, progress, simple IoU tracking,
apparent visual severity, and opt-in SQLite summary history. Video unique counts and
severity are explicitly labelled as estimates.

## Non-goals and constraints

The application does not measure true depth or dimensions, guarantee road safety,
automatically report to authorities, perform continuous city-scale monitoring, or claim
validated accuracy before a held-out test evaluation is completed.

## Acceptance status

Application code, data tools, tests, and documentation are provided. Final model-quality
acceptance remains pending until a labelled dataset is prepared, `best.pt` is trained,
and precision, recall, mAP@0.50, and mAP@0.50–0.95 are measured on unseen test data.

