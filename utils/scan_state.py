"""Bounded, per-session scan results tied to the inputs that produced them."""

from __future__ import annotations

import hashlib
from collections.abc import MutableMapping
from typing import Any


def scan_key(data: bytes, mode: str, confidence: float, *options: Any) -> tuple:
    return (
        hashlib.sha256(data).hexdigest(),
        mode,
        round(float(confidence), 4),
        *options,
    )


def sync_scan_inputs(
    state: MutableMapping, namespace: str, key: tuple | None, model_signature: str = ""
) -> bool:
    """Invalidate immediately; never resurrect an old result by reverting controls."""
    inputs = (key, model_signature)
    input_name, results_name = f"{namespace}_inputs", f"{namespace}_results"
    changed_name = f"{namespace}_changed"
    if state.get(input_name) != inputs:
        had_results = bool(state.get(results_name))
        state[results_name] = {}
        state[input_name] = inputs
        state[changed_name] = had_results or state.get(changed_name, False)
    return bool(state.get(changed_name))


def store_scan_result(
    state: MutableMapping, namespace: str, key: tuple, result: dict
) -> None:
    # Only the current result remains in memory, including video bytes.
    state[f"{namespace}_results"] = {key: result}
    state[f"{namespace}_changed"] = False
