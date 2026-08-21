"""SQLite boundary for optional detection-history records."""

from __future__ import annotations

import csv
import sqlite3
from io import StringIO
from pathlib import Path
from typing import Any

from config import DATABASE_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    input_type TEXT NOT NULL CHECK (input_type IN ('image', 'video')),
    input_name TEXT,
    detection_count INTEGER NOT NULL CHECK (detection_count >= 0),
    average_confidence REAL,
    apparent_severity TEXT,
    input_media_path TEXT,
    output_media_path TEXT,
    latitude REAL,
    longitude REAL,
    processing_ms REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def initialize_database(db_path: str | Path = DATABASE_PATH) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.executescript(SCHEMA)


def create_detection_record(
    *,
    input_type: str,
    detection_count: int,
    input_name: str | None = None,
    average_confidence: float | None = None,
    apparent_severity: str | None = None,
    input_media_path: str | None = None,
    output_media_path: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    processing_ms: float | None = None,
    db_path: str | Path = DATABASE_PATH,
) -> int:
    if input_type not in {"image", "video"}:
        raise ValueError("input_type must be 'image' or 'video'.")
    if detection_count < 0:
        raise ValueError("detection_count cannot be negative.")

    initialize_database(db_path)
    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO detections (
                input_type, input_name, detection_count, average_confidence,
                apparent_severity, input_media_path, output_media_path,
                latitude, longitude, processing_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                input_type,
                input_name,
                detection_count,
                average_confidence,
                apparent_severity,
                input_media_path,
                output_media_path,
                latitude,
                longitude,
                processing_ms,
            ),
        )
        return int(cursor.lastrowid)


def list_detection_records(
    limit: int = 1000, db_path: str | Path = DATABASE_PATH
) -> list[dict[str, Any]]:
    initialize_database(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM detections ORDER BY created_at DESC, id DESC LIMIT ?",
            (max(1, int(limit)),),
        ).fetchall()
    return [dict(row) for row in rows]


def records_to_csv(records: list[dict[str, Any]]) -> bytes:
    if not records:
        return b""
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue().encode("utf-8")
