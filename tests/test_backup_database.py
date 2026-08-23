from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from backup_database import (
    backup_database,
    prune_backups,
    verify_backup,
)


def make_database(path: Path, rows: int = 3) -> Path:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT)")
        connection.executemany(
            "INSERT INTO notes (body) VALUES (?)", [(f"row {n}",) for n in range(rows)]
        )
    return path


def test_backup_copies_every_row(tmp_path: Path) -> None:
    source = make_database(tmp_path / "source.db", rows=5)
    destination = backup_database(source, tmp_path / "backups" / "potholes-a.db")

    with sqlite3.connect(destination) as connection:
        count = connection.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    assert count == 5
    verify_backup(destination)


def test_backup_leaves_no_partial_file(tmp_path: Path) -> None:
    source = make_database(tmp_path / "source.db")
    destination = backup_database(source, tmp_path / "backups" / "potholes-a.db")

    leftovers = list(destination.parent.glob("*.partial"))
    assert leftovers == []


def test_backup_rejects_missing_database(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        backup_database(tmp_path / "absent.db", tmp_path / "out.db")


def test_prune_keeps_the_newest_backups(tmp_path: Path) -> None:
    for stamp in ("20240101T000000Z", "20240102T000000Z", "20240103T000000Z"):
        (tmp_path / f"potholes-{stamp}.db").write_bytes(b"")

    removed = prune_backups(tmp_path, keep=2)

    remaining = sorted(path.name for path in tmp_path.glob("potholes-*.db"))
    assert remaining == ["potholes-20240102T000000Z.db", "potholes-20240103T000000Z.db"]
    assert [path.name for path in removed] == ["potholes-20240101T000000Z.db"]


def test_prune_ignores_unrelated_files(tmp_path: Path) -> None:
    (tmp_path / "potholes-20240101T000000Z.db").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("keep me", encoding="utf-8")

    prune_backups(tmp_path, keep=0)

    assert (tmp_path / "notes.txt").is_file()
    assert (tmp_path / "potholes-20240101T000000Z.db").is_file()


def test_verify_backup_rejects_a_corrupt_file(tmp_path: Path) -> None:
    corrupt = tmp_path / "potholes-bad.db"
    corrupt.write_bytes(b"not a database")

    with pytest.raises(sqlite3.DatabaseError):
        verify_backup(corrupt)
