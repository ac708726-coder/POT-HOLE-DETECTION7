"""Wait for one training run, evaluate serially, and report; never install weights.

This enables a long laptop run to finish safely without concurrent GPU jobs. It
requires the current baseline's full-val report and earlier app benchmark to match
its exact checkpoint hash. It never pushes, commits, trains, or rewrites raw data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.compare_checkpoints import compare


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main() -> None:
    import psutil

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-pid", type=int, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prefix", default="hard_examples_20261001")
    args = parser.parse_args()
    run = args.run.resolve()
    if run.parent != (ROOT / "runs/pothole").resolve():
        raise SystemExit("Run must be directly inside this project's runs/pothole.")
    if not args.prefix.replace("_", "").isalnum():
        raise SystemExit("Use a simple alphanumeric/underscore output prefix.")
    output = ROOT / "outputs/metrics"
    baseline = ROOT / "models/best.pt"
    baseline_val = output / "hard_baseline_val_20261001.json"
    baseline_app = output / "color_fix_confirmation.json"
    baseline_hash = sha(baseline)
    if read(baseline_val)["weights_sha256"] != baseline_hash:
        raise SystemExit("Baseline changed: re-evaluate it before comparing.")
    if read(baseline_app)["model_sha256"] != baseline_hash:
        raise SystemExit("App baseline benchmark belongs to a different checkpoint.")
    try:
        trainer = psutil.Process(args.trainer_pid)
        command = " ".join(trainer.cmdline())
        if (
            "training/train.py" not in command.replace("\\", "/")
            or run.name not in command
        ):
            raise SystemExit("PID does not identify this named project training run.")
        started = trainer.create_time()
        print(f"Waiting for trainer PID {args.trainer_pid} ({run.name})...", flush=True)
        while trainer.is_running() and trainer.create_time() == started:
            time.sleep(2)
    except psutil.NoSuchProcess:
        pass
    if sha(baseline) != baseline_hash:
        raise SystemExit("App checkpoint changed during training; comparison aborted.")
    weights = run / "weights/best.pt"
    results = run / "results.csv"
    if not weights.is_file() or not results.is_file():
        raise SystemExit("Training did not produce a completed epoch/checkpoint.")
    with results.open(encoding="utf-8", newline="") as source:
        epochs = list(csv.DictReader(source))
    if not epochs:
        raise SystemExit("No completed training epochs recorded.")

    def execute(name: str, script: str, options: list[str]) -> None:
        log = output / f"{args.prefix}_{name}.log"
        print(f"Running {name}; log: {log}", flush=True)
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        with log.open("w", encoding="utf-8") as destination:
            subprocess.run(
                [sys.executable, "-u", script, *options],
                cwd=ROOT,
                env=env,
                stdout=destination,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def evaluate(name: str, checkpoint: Path, split: str) -> dict:
        report = output / f"{args.prefix}_{name}.json"
        execute(
            name,
            "training/evaluate.py",
            [
                "--weights",
                str(checkpoint),
                "--split",
                split,
                "--imgsz",
                "640",
                "--batch",
                "16",
                "--quantize",
                "16",
                "--max-failures",
                "0",
                "--device",
                "0",
                "--name",
                f"{args.prefix}_{name}",
                "--output",
                str(report),
            ],
        )
        return read(report)

    candidate_val = evaluate("candidate_val", weights, "val")
    gate = compare(read(baseline_val), candidate_val)
    report = {
        "run": str(run),
        "completed_epochs": len(epochs),
        "baseline_val": read(baseline_val),
        "candidate_val": candidate_val,
        "validation_gate": gate,
        "eligible_for_manual_promotion": False,
        "model_installed": False,
    }
    if gate["passes_gate"]:
        before_test = evaluate("baseline_test", baseline, "test")
        after_test = evaluate("candidate_test", weights, "test")
        test_comparison = compare(before_test, after_test)
        app_path = output / f"{args.prefix}_candidate_app.json"
        execute(
            "candidate_app",
            "scripts/benchmark_detection_recall.py",
            [
                "--weights",
                str(weights),
                "--positives",
                "100",
                "--negatives",
                "900",
                "--seed",
                "99",
                "--variants",
                "balanced",
                "--output",
                str(app_path),
            ],
        )
        before_app, after_app = read(baseline_app), read(app_path)
        if before_app["sample_sha256"] != after_app["sample_sha256"]:
            raise ValueError("Application benchmark samples differ.")
        before = next(
            m
            for m in before_app["variants"]["balanced"]["metrics"]
            if m["confidence"] == 0.35
        )
        after = next(
            m
            for m in after_app["variants"]["balanced"]["metrics"]
            if m["confidence"] == 0.35
        )
        app_passes = (
            after["true_positives"] > before["true_positives"]
            and after["precision"] >= before["precision"] - 0.02
        )
        report.update(
            {
                "baseline_test": before_test,
                "candidate_test": after_test,
                "test_comparison": test_comparison,
                "baseline_app": before,
                "candidate_app": after,
                "app_gate_passed": app_passes,
                "eligible_for_manual_promotion": test_comparison["passes_gate"]
                and app_passes,
            }
        )
    if sha(baseline) != baseline_hash:
        raise ValueError("Baseline changed before evaluation finished.")
    destination = output / f"{args.prefix}_result.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(
        f"Saved experiment result: {destination}. App weights have not changed.",
        flush=True,
    )


if __name__ == "__main__":
    main()
