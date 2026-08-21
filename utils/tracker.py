"""Small IoU tracker used to estimate unique potholes in videos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def intersection_over_union(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


@dataclass
class _Track:
    track_id: int
    box: list[float]
    missed_frames: int = 0


class PotholeTracker:
    """Associate detections between nearby frames using greedy IoU matching."""

    def __init__(self, iou_threshold: float = 0.25, max_missed_frames: int = 8) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self.total_tracks_created = 0

    def update(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for track in self._tracks.values():
            track.missed_frames += 1

        available_track_ids = set(self._tracks)
        tracked_detections: list[dict[str, Any]] = []
        for detection in detections:
            best_id = None
            best_iou = 0.0
            for track_id in available_track_ids:
                score = intersection_over_union(
                    detection["box"], self._tracks[track_id].box
                )
                if score > best_iou:
                    best_iou = score
                    best_id = track_id

            if best_id is not None and best_iou >= self.iou_threshold:
                track = self._tracks[best_id]
                track.box = list(detection["box"])
                track.missed_frames = 0
                available_track_ids.remove(best_id)
                track_id = best_id
            else:
                track_id = self._next_id
                self._next_id += 1
                self.total_tracks_created += 1
                self._tracks[track_id] = _Track(track_id, list(detection["box"]))

            tracked = dict(detection)
            tracked["track_id"] = track_id
            tracked_detections.append(tracked)

        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if track.missed_frames > self.max_missed_frames
        ]
        for track_id in expired:
            del self._tracks[track_id]
        return tracked_detections
