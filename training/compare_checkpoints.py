"""Apply the predeclared validation gate; report evidence, never install weights."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

PROVENANCE = (
    "split",
    "imgsz",
    "data_config_sha256",
    "torch_version",
    "quantize",
    "batch",
)
METRICS = ("precision", "recall", "map50", "map50_95")


def compare(baseline: dict, candidate: dict) -> dict:
    for key in PROVENANCE:
        if (
            key not in baseline
            or key not in candidate
            or baseline[key] != candidate[key]
        ):
            raise ValueError(f"Evaluation settings differ or are missing: {key}")
    for report in (baseline, candidate):
        if not report.get("weights_sha256"):
            raise ValueError("Checkpoint hash is required.")
        for metric in METRICS:
            value = report.get(metric)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"Invalid metric: {metric}")
    delta = {key: candidate[key] - baseline[key] for key in METRICS}
    gates = {
        "map50_gain_at_least_0.005": delta["map50"] >= 0.005,
        "recall_improved": delta["recall"] > 0,
        "precision_loss_at_most_0.02": delta["precision"] >= -0.02,
    }
    return {
        "split": baseline["split"],
        "baseline_sha256": baseline["weights_sha256"],
        "candidate_sha256": candidate["weights_sha256"],
        "delta": delta,
        "gates": gates,
        "passes_gate": all(gates.values()),
        "note": "Validation gate only. This script never promotes or changes model weights.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    if baseline.get("split") != "val":
        raise SystemExit("Select candidates using validation, not test.")
    report = compare(baseline, candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
