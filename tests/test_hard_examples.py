import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

from training.hard_examples import (
    clean_train_files,
    diverse_take,
    label_boxes,
    prepare_validation_subset,
    replay_rows,
    write_manifest,
)
from training.train import attach_memory_safety


def test_miner_excludes_held_out_content_and_train_duplicates(tmp_path):
    for split in ("train", "val", "test"):
        (tmp_path / f"images/{split}").mkdir(parents=True)
        (tmp_path / f"labels/{split}").mkdir(parents=True)
    for split, name, color in [
        ("val", "held_out", "red"),
        ("test", "test", "green"),
        ("train", "renamed_held_out", "red"),
        ("train", "train_1", "blue"),
        ("train", "train_2", "blue"),
    ]:
        Image.new("RGB", (20, 20), color).save(tmp_path / f"images/{split}/{name}.png")
        (tmp_path / f"labels/{split}/{name}.txt").write_text("")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    files, excluded = clean_train_files(tmp_path)
    assert [p.name for p in files] == ["train_1.png"]
    assert {r["reason"] for r in excluded} == {
        "held-out duplicate",
        "train duplicate",
    }
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("label", ["1 .5 .5 .1 .1", "0 nan .5 .1 .1", "0 .5 .5 0 .1"])
def test_invalid_mining_label_is_rejected(tmp_path, label):
    path = tmp_path / "invalid.txt"
    path.write_text(label)
    with pytest.raises(ValueError, match="Invalid"):
        label_boxes(path, (100, 100))
    with pytest.raises(FileNotFoundError):
        label_boxes(tmp_path / "missing.txt", (100, 100))


def test_replay_repeats_only_hard_positives_and_keeps_negatives(tmp_path):
    rows = [
        {
            "filename": "India_1.jpg",
            "truth": [[0, 0, 1, 1]],
            "missed": [0],
            "false_positives": [],
        },
        {
            "filename": "China_Drone_1.jpg",
            "truth": [[0, 0, 1, 1]],
            "missed": [],
            "false_positives": [],
        },
        {"filename": "India_2.jpg", "truth": [], "missed": [], "false_positives": [0]},
        {"filename": "Norway_1.jpg", "truth": [], "missed": [], "false_positives": []},
    ]
    repeated, unique = replay_rows(rows)
    assert len(unique) == 4
    assert len(repeated) == 5
    assert [r["filename"] for r in repeated].count("India_1.jpg") == 2
    assert diverse_take(rows, 3, 42) == diverse_take(rows, 3, 42)
    root = tmp_path / "dataset"
    folder = root / "images/train"
    folder.mkdir(parents=True)
    for row in rows:
        Image.new("RGB", (20, 20)).save(folder / row["filename"])
    output = tmp_path / "manifest"
    output.mkdir()
    summary = write_manifest(root, output, rows)
    assert summary["positives"] == summary["negatives"] == 2
    config = yaml.safe_load((output / "data.yaml").read_text())
    files = Path(config["train"]).read_text().splitlines()
    assert len(files) == 5
    assert all(Path(p).parent == folder.resolve() for p in files)
    assert config["val"] == (root / "images/val").resolve().as_posix()
    # The report remains JSON serializable for experiment auditing.
    json.dumps(summary)
    rows[0]["filename"] = "../val/held_out.jpg"
    with pytest.raises(ValueError, match="TRAIN"):
        write_manifest(root, output, rows)


def test_budget_validation_is_separate_and_does_not_move_images(tmp_path):
    root = tmp_path / "dataset"
    for split in ("train", "val"):
        (root / f"images/{split}").mkdir(parents=True)
        (root / f"labels/{split}").mkdir(parents=True)
    train_image = root / "images/train/India_train.jpg"
    Image.new("RGB", (10, 10)).save(train_image)
    for index in range(100):
        name = f"India_{index}"
        Image.new("RGB", (10, 10)).save(root / f"images/val/{name}.jpg")
        (root / f"labels/val/{name}.txt").write_text(
            "0 .5 .5 .2 .2" if index < 10 else ""
        )
    training = tmp_path / "train.txt"
    training.write_text(train_image.resolve().as_posix() + "\n")
    config = tmp_path / "data.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "path": root.resolve().as_posix(),
                "train": training.resolve().as_posix(),
                "val": (root / "images/val").resolve().as_posix(),
                "test": (root / "images/test").resolve().as_posix(),
                "names": {0: "pothole"},
            }
        )
    )
    original = config.read_bytes()
    selected_config = prepare_validation_subset(config, 100)
    selected = yaml.safe_load(selected_config.read_text())
    paths = Path(selected["val"]).read_text().splitlines()
    assert len(paths) == 100
    assert str(train_image.resolve().as_posix()) not in paths
    assert selected["train"] == training.resolve().as_posix()
    assert selected["test"] == (root / "images/test").resolve().as_posix()
    assert config.read_bytes() == original
    assert len(list((root / "images/val").glob("*.jpg"))) == 100
    with pytest.raises(ValueError, match="already exists"):
        prepare_validation_subset(config, 100)


def test_memory_callbacks_clear_only_unused_cache_and_disable_plots(monkeypatch):
    calls = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: calls.append("cuda cache"),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr("gc.collect", lambda: calls.append("python cache"))
    callbacks = {}
    model = SimpleNamespace(
        add_callback=lambda name, function: callbacks.update({name: function})
    )
    attach_memory_safety(model)
    sentinel = object()
    trainer = SimpleNamespace(
        args=SimpleNamespace(plots=True),
        model=sentinel,
        optimizer=sentinel,
        device=SimpleNamespace(type="cuda"),
    )
    callbacks["on_pretrain_routine_start"](trainer)
    assert not trainer.args.plots
    for _ in range(49):
        callbacks["on_train_batch_end"](trainer)
    assert calls.count("cuda cache") == 1
    callbacks["on_train_batch_end"](trainer)
    assert calls.count("cuda cache") == 2
    callbacks["on_fit_epoch_end"](trainer)
    callbacks["on_train_epoch_start"](trainer)
    assert calls.count("cuda cache") == 4
    assert trainer.model is sentinel and trainer.optimizer is sentinel

    trainer.device.type = "cpu"
    callbacks["on_train_epoch_start"](trainer)
    assert calls.count("cuda cache") == 4
