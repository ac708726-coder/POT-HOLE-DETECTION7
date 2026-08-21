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

