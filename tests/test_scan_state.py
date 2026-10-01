import pytest

from utils.scan_state import scan_key, store_scan_result, sync_scan_inputs


@pytest.mark.parametrize(
    "new",
    [
        (b"new file", "fast", 0.35),
        (b"file", "thorough", 0.35),
        (b"file", "fast", 0.15),
    ],
)
def test_changing_any_scan_input_clears_old_result(new):
    state = {}
    key = scan_key(b"file", "fast", 0.35)
    sync_scan_inputs(state, "image", key)
    store_scan_result(state, "image", key, {"count": 1})
    assert sync_scan_inputs(state, "image", scan_key(*new))
    assert state["image_results"] == {}
    sync_scan_inputs(state, "image", key)
    assert state["image_results"] == {}  # reverting does not resurrect old output


def test_removing_upload_or_changing_model_invalidates_result():
    state = {}
    key = scan_key(b"file", "fast", 0.35)
    sync_scan_inputs(state, "image", key, "model1")
    store_scan_result(state, "image", key, {"count": 1})
    assert sync_scan_inputs(state, "image", key, "model2")
    store_scan_result(state, "image", key, {"count": 2})
    assert sync_scan_inputs(state, "image", None)
    assert not state["image_results"]


def test_new_scan_clears_banner_and_unchanged_settings_keep_result():
    state = {}
    key = scan_key(b"file", "balanced", 0.35)
    sync_scan_inputs(state, "image", key)
    store_scan_result(state, "image", key, {"count": 1})
    assert not sync_scan_inputs(state, "image", key)
    assert state["image_results"][key]["count"] == 1
