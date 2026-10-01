import pytest

from training.compare_checkpoints import compare


def metrics(**overrides):
    return {
        "split": "val",
        "imgsz": 640,
        "data_config_sha256": "data",
        "torch_version": "test",
        "quantize": 16,
        "batch": 16,
        "weights_sha256": "checkpoint",
        "precision": 0.55,
        "recall": 0.44,
        "map50": 0.446,
        "map50_95": 0.2,
        **overrides,
    }


def test_only_meaningful_detection_gain_passes_gate():
    assert compare(metrics(), metrics(map50=0.46, recall=0.46))["passes_gate"]
    assert not compare(metrics(), metrics(map50=0.45, recall=0.46))["passes_gate"]
    assert not compare(metrics(), metrics(map50=0.46, recall=0.43))["passes_gate"]
    assert not compare(metrics(), metrics(map50=0.46, recall=0.46, precision=0.52))[
        "passes_gate"
    ]


@pytest.mark.parametrize(
    "key,value",
    [
        ("split", "test"),
        ("imgsz", 960),
        ("data_config_sha256", "different"),
        ("quantize", 32),
    ],
)
def test_different_evaluation_protocol_is_not_comparable(key, value):
    with pytest.raises(ValueError, match="settings"):
        compare(metrics(), metrics(**{key: value}))


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -0.1])
def test_invalid_results_cannot_pass(value):
    with pytest.raises(ValueError, match="Invalid metric"):
        compare(metrics(), metrics(map50=value))
