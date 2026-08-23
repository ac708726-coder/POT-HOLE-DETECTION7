"""Take a consistent backup of the SQLite database.

Uses SQLite's online backup API rather than copying the file, so a backup
taken while the app is running cannot capture a half-written page. Old
backups are pruned so the directory does not grow without bound.

    python scripts/backup_database.py
    python scripts/backup_database.py --output-dir /var/backups/divot --keep 30

Schedule it (cron, Task Scheduler, or the platform's job runner) and verify a
restore periodically: an unverified backup is not a backup.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATABASE_PATH

DEFAULT_KEEP = 14
BACKUP_PREFIX = "potholes-"
BACKUP_SUFFIX = ".db"


def backup_database(source: Path, destination: Path) -> Path:
    """Copy `source` to `destination` through the online backup API."""

    if not source.is_file():
        raise FileNotFoundError(f"No database at {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")

    source_connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    target = sqlite3.connect(partial)
    try:
        source_connection.backup(target)
    finally:
        # Both handles must be closed before the rename below: Windows refuses
        # to move a file that is still open.
        target.close()
        source_connection.close()

    # Only name it a backup once it is complete, so a crashed run never leaves
    # a truncated file that looks restorable.
    partial.replace(destination)
    return destination


def prune_backups(directory: Path, keep: int) -> list[Path]:
    """Delete all but the newest `keep` backups. Returns what was removed."""

    if keep < 1:
        return []
    backups = sorted(
        directory.glob(f"{BACKUP_PREFIX}*{BACKUP_SUFFIX}"),
        key=lambda path: path.name,
        reverse=True,
    )
    removed = []
    for stale in backups[keep:]:
        stale.unlink()
        removed.append(stale)
    return removed


def verify_backup(path: Path) -> None:
    """Raise if the backup does not open and pass an integrity check."""

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        result = connection.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        connection.close()
    if result != "ok":
        raise RuntimeError(f"Integrity check failed for {path}: {result}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=DATABASE_PATH,
        help="Database to back up (default: the app's configured database).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DATABASE_PATH).parent / "backups",
        help="Where backups are written.",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=DEFAULT_KEEP,
        help=f"How many backups to retain (default: {DEFAULT_KEEP}).",
    )
    arguments = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = arguments.output_dir / f"{BACKUP_PREFIX}{stamp}{BACKUP_SUFFIX}"

    try:
        written = backup_database(arguments.database, destination)
        verify_backup(written)
    except (FileNotFoundError, RuntimeError, sqlite3.Error) as error:
        print(f"Backup failed: {error}", file=sys.stderr)
        return 1

    size_kb = written.stat().st_size / 1024
    print(f"Wrote {written} ({size_kb:.1f} KiB), integrity check ok")
    for removed in prune_backups(arguments.output_dir, arguments.keep):
        print(f"Pruned {removed.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
