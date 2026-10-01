import pytest

from utils.detection_details import detection_rows, no_detection_hint


def test_fast_hint_suggests_both_more_detailed_modes():
    text = no_detection_hint("fast", 0.35)
    assert "Balanced or Thorough" in text


def test_balanced_hint_suggests_only_thorough():
    text = no_detection_hint("balanced", 0.35)
    assert "Thorough" in text
    assert "Balanced" not in text


def test_thorough_hint_does_not_suggest_the_same_mode():
    text = no_detection_hint("thorough", 0.35)
    assert "lowering confidence" in text
    assert "closer, better-lit" in text
    assert "Balanced" not in text and "Thorough" not in text


def test_thorough_at_slider_floor_does_not_suggest_unavailable_lower_value():
    text = no_detection_hint("thorough", 0.15)
    assert "already at its minimum" in text
    assert "lowering" not in text


def test_unknown_hint_mode_fails_explicitly():
    with pytest.raises(ValueError):
        no_detection_hint("typo", 0.35)


def test_box_dimensions_and_area_use_original_pixels():
    rows = detection_rows([{"box": [10, 20, 50, 60], "confidence": 0.75}], (100, 80))
    assert rows == [
        {
            "ID": 1,
            "Confidence": 0.75,
            "Width (px)": 40.0,
            "Height (px)": 40.0,
            "Image area (%)": 20.0,
        }
    ]
