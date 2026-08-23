"""Account rules and the history isolation they exist to provide."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from utils import auth, storage

GOOD_PASSWORD = "correct-horse-battery"


@pytest.fixture()
def db(tmp_path):
    return tmp_path / "test.db"


def test_password_is_never_stored_in_the_clear(db):
    auth.create_user("driver", GOOD_PASSWORD, db_path=db)
    with sqlite3.connect(db) as connection:
        row = connection.execute(
            "SELECT password_salt, password_hash FROM users"
        ).fetchone()
    assert GOOD_PASSWORD not in row[0]
    assert GOOD_PASSWORD not in row[1]
    assert len(row[1]) == 64  # 32-byte scrypt digest, hex encoded


def test_same_password_gets_a_different_hash_per_account(db):
    auth.create_user("one", GOOD_PASSWORD, db_path=db)
    auth.create_user("two", GOOD_PASSWORD, db_path=db)
    with sqlite3.connect(db) as connection:
        hashes = [r[0] for r in connection.execute("SELECT password_hash FROM users")]
    assert hashes[0] != hashes[1], "per-user salt should make the digests differ"


def test_sign_in_round_trip(db):
    user_id = auth.create_user("driver", GOOD_PASSWORD, db_path=db)
    assert auth.verify_user("driver", GOOD_PASSWORD, db_path=db) == user_id


def test_username_is_case_and_space_insensitive(db):
    user_id = auth.create_user("  Driver  ", GOOD_PASSWORD, db_path=db)
    assert auth.verify_user("DRIVER", GOOD_PASSWORD, db_path=db) == user_id


def test_duplicate_username_is_refused(db):
    auth.create_user("driver", GOOD_PASSWORD, db_path=db)
    with pytest.raises(auth.AuthError):
        auth.create_user("Driver", GOOD_PASSWORD, db_path=db)


def test_short_password_is_refused(db):
    with pytest.raises(auth.AuthError):
        auth.create_user("driver", "short", db_path=db)


def test_password_equal_to_username_is_refused(db):
    with pytest.raises(auth.AuthError):
        auth.create_user("driverdriver", "driverdriver", db_path=db)


def test_wrong_password_and_unknown_user_look_the_same(db):
    auth.create_user("driver", GOOD_PASSWORD, db_path=db)
    with pytest.raises(auth.AuthError) as wrong:
        auth.verify_user("driver", "not-the-password", db_path=db)
    with pytest.raises(auth.AuthError) as missing:
        auth.verify_user("ghost", "not-the-password", db_path=db)
    assert str(wrong.value) == str(missing.value) == auth.INVALID_CREDENTIALS


def test_account_locks_after_repeated_failures(db):
    auth.create_user("driver", GOOD_PASSWORD, db_path=db)
    for _ in range(auth.MAX_FAILED_ATTEMPTS - 1):
        with pytest.raises(auth.AuthError):
            auth.verify_user("driver", "wrong", db_path=db)
    with pytest.raises(auth.AuthError, match="Locked"):
        auth.verify_user("driver", "wrong", db_path=db)
    # the right password is refused too while the lock stands
    with pytest.raises(auth.AuthError, match="Try again"):
        auth.verify_user("driver", GOOD_PASSWORD, db_path=db)
    assert auth.lockout_remaining("driver", db_path=db) > 0


def test_lock_expires(db):
    auth.create_user("driver", GOOD_PASSWORD, db_path=db)
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE users SET locked_until = ?", (past,))
    assert auth.lockout_remaining("driver", db_path=db) == 0
    assert auth.verify_user("driver", GOOD_PASSWORD, db_path=db) > 0


def test_successful_sign_in_clears_earlier_failures(db):
    auth.create_user("driver", GOOD_PASSWORD, db_path=db)
    with pytest.raises(auth.AuthError):
        auth.verify_user("driver", "wrong", db_path=db)
    auth.verify_user("driver", GOOD_PASSWORD, db_path=db)
    with sqlite3.connect(db) as connection:
        attempts = connection.execute("SELECT failed_attempts FROM users").fetchone()[0]
    assert attempts == 0


def test_change_password(db):
    user_id = auth.create_user("driver", GOOD_PASSWORD, db_path=db)
    auth.change_password(user_id, GOOD_PASSWORD, "a-brand-new-secret", db_path=db)
    with pytest.raises(auth.AuthError):
        auth.verify_user("driver", GOOD_PASSWORD, db_path=db)
    assert auth.verify_user("driver", "a-brand-new-secret", db_path=db) == user_id


def test_change_password_needs_the_current_one(db):
    user_id = auth.create_user("driver", GOOD_PASSWORD, db_path=db)
    with pytest.raises(auth.AuthError):
        auth.change_password(user_id, "guess", "a-brand-new-secret", db_path=db)


# --------------------------------------------------------------- isolation


def test_history_is_private_to_each_account(db):
    alice = auth.create_user("alice", GOOD_PASSWORD, db_path=db)
    bob = auth.create_user("bob", GOOD_PASSWORD, db_path=db)

    storage.create_detection_record(
        user_id=alice,
        input_type="image",
        input_name="alice.jpg",
        detection_count=2,
        db_path=db,
    )
    storage.create_detection_record(
        user_id=bob,
        input_type="video",
        input_name="bob.mp4",
        detection_count=5,
        db_path=db,
    )

    alice_rows = storage.list_detection_records(alice, db_path=db)
    bob_rows = storage.list_detection_records(bob, db_path=db)

    assert [r["input_name"] for r in alice_rows] == ["alice.jpg"]
    assert [r["input_name"] for r in bob_rows] == ["bob.mp4"]


def test_saving_without_an_account_is_refused(db):
    with pytest.raises(ValueError):
        storage.create_detection_record(
            user_id=0, input_type="image", detection_count=1, db_path=db
        )


def test_listing_without_an_account_is_refused(db):
    with pytest.raises(ValueError):
        storage.list_detection_records(0, db_path=db)


def test_one_account_cannot_delete_anothers_record(db):
    alice = auth.create_user("alice", GOOD_PASSWORD, db_path=db)
    bob = auth.create_user("bob", GOOD_PASSWORD, db_path=db)
    record_id = storage.create_detection_record(
        user_id=alice, input_type="image", detection_count=1, db_path=db
    )
    assert storage.delete_detection_record(record_id, bob, db_path=db) is False
    assert len(storage.list_detection_records(alice, db_path=db)) == 1
    assert storage.delete_detection_record(record_id, alice, db_path=db) is True


def test_rows_written_before_accounts_existed_belong_to_nobody(db):
    """The pre-auth database must not hand its rows to the first account."""

    with sqlite3.connect(db) as connection:
        connection.executescript("""
            CREATE TABLE detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_type TEXT NOT NULL,
                input_name TEXT,
                detection_count INTEGER NOT NULL,
                average_confidence REAL,
                apparent_severity TEXT,
                input_media_path TEXT,
                output_media_path TEXT,
                latitude REAL,
                longitude REAL,
                processing_ms REAL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            INSERT INTO detections (input_type, input_name, detection_count)
            VALUES ('image', 'legacy.jpg', 3);
            """)

    storage.initialize_database(db)  # runs the migration
    first = auth.create_user("first", GOOD_PASSWORD, db_path=db)
    assert storage.list_detection_records(first, db_path=db) == []

    with sqlite3.connect(db) as connection:
        owner = connection.execute(
            "SELECT user_id FROM detections WHERE input_name = 'legacy.jpg'"
        ).fetchone()[0]
    assert owner is None
