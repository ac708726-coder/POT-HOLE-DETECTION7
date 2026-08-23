"""Local account store: password hashing, sign-in checks, and lockout.

Deliberately free of Streamlit so the rules can be unit tested. Passwords are
never stored or logged; only a per-user salt and an scrypt digest are kept.
"""

from __future__ import annotations

import hmac
import re
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from hashlib import scrypt
from pathlib import Path

from config import DATABASE_PATH

# scrypt cost. n is the memory/CPU factor; 2**14 keeps a single check near a
# tenth of a second on a laptop, which is slow enough to make offline guessing
# expensive and fast enough for a login form.
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_DK_LEN = 32
_SALT_BYTES = 16

MIN_PASSWORD_LENGTH = 10
MAX_PASSWORD_LENGTH = 256
_USERNAME_PATTERN = re.compile(r"^[a-z0-9._-]{3,32}$")

MAX_FAILED_ATTEMPTS = 5
LOCKOUT_MINUTES = 15

# One message for "no such user" and for "wrong password", so the form cannot
# be used to discover which accounts exist.
INVALID_CREDENTIALS = "Username or password is incorrect."


class AuthError(Exception):
    """Raised when a sign-up or sign-in request cannot be honoured."""


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_salt TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_users(db_path: str | Path = DATABASE_PATH) -> None:
    """Create the users table if it is not there yet."""

    with _connect(db_path) as connection:
        connection.executescript(SCHEMA)


def normalize_username(username: str) -> str:
    """Fold a username to its stored form, or raise if it is not usable."""

    candidate = (username or "").strip().lower()
    if not _USERNAME_PATTERN.fullmatch(candidate):
        raise AuthError(
            "Username must be 3-32 characters, using letters, numbers, dot, "
            "underscore or hyphen."
        )
    return candidate


def check_password_strength(password: str, username: str = "") -> None:
    """Raise AuthError if the password is too weak to accept."""

    if len(password) < MIN_PASSWORD_LENGTH:
        raise AuthError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise AuthError(f"Password must be at most {MAX_PASSWORD_LENGTH} characters.")
    if username and password.lower() == username.strip().lower():
        raise AuthError("Password cannot be the same as the username.")
    if password.strip() == "":
        raise AuthError("Password cannot be blank.")


def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    """Return (salt_hex, hash_hex) for a password."""

    if salt is None:
        salt = secrets.token_bytes(_SALT_BYTES)
    digest = scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_DK_LEN,
    )
    return salt.hex(), digest.hex()


def _password_matches(password: str, salt_hex: str, expected_hash_hex: str) -> bool:
    _, candidate = hash_password(password, bytes.fromhex(salt_hex))
    return hmac.compare_digest(candidate, expected_hash_hex)


def create_user(
    username: str, password: str, db_path: str | Path = DATABASE_PATH
) -> int:
    """Register an account and return its id."""

    name = normalize_username(username)
    check_password_strength(password, name)
    salt_hex, hash_hex = hash_password(password)

    initialize_users(db_path)
    try:
        with _connect(db_path) as connection:
            cursor = connection.execute(
                "INSERT INTO users (username, password_salt, password_hash) "
                "VALUES (?, ?, ?)",
                (name, salt_hex, hash_hex),
            )
            return int(cursor.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise AuthError("That username is already taken.") from exc


def lockout_remaining(username: str, db_path: str | Path = DATABASE_PATH) -> int:
    """Seconds left on an account lockout, or 0 when the account is open."""

    initialize_users(db_path)
    try:
        name = normalize_username(username)
    except AuthError:
        return 0
    with _connect(db_path) as connection:
        row = connection.execute(
            "SELECT locked_until FROM users WHERE username = ?", (name,)
        ).fetchone()
    if row is None or not row["locked_until"]:
        return 0
    until = datetime.fromisoformat(row["locked_until"])
    remaining = (until - _now()).total_seconds()
    return int(remaining) if remaining > 0 else 0


def verify_user(
    username: str, password: str, db_path: str | Path = DATABASE_PATH
) -> int:
    """Return the user id for valid credentials, else raise AuthError.

    Repeated failures lock the account for a while, which blunts online
    guessing without needing anything outside the database.
    """

    initialize_users(db_path)
    try:
        name = normalize_username(username)
    except AuthError:
        # An unusable username cannot match a stored account, but it must not
        # produce a different message from a wrong password.
        raise AuthError(INVALID_CREDENTIALS) from None

    # Deliberately not inside `with _connect(...)`: that context manager rolls
    # back when the block raises, which would silently undo the failed-attempt
    # counter and the lockout every time a sign-in was refused.
    connection = _connect(db_path)
    try:
        row = connection.execute(
            "SELECT id, password_salt, password_hash, failed_attempts, locked_until "
            "FROM users WHERE username = ?",
            (name,),
        ).fetchone()

        if row is None:
            # Spend comparable time so a missing account is not obvious from
            # how quickly the form comes back.
            hash_password(password)
            error = AuthError(INVALID_CREDENTIALS)
        elif (
            row["locked_until"] and datetime.fromisoformat(row["locked_until"]) > _now()
        ):
            until = datetime.fromisoformat(row["locked_until"])
            minutes = max(1, int((until - _now()).total_seconds() // 60) + 1)
            error = AuthError(
                f"Too many failed attempts. Try again in {minutes} minute(s)."
            )
        elif not _password_matches(
            password, row["password_salt"], row["password_hash"]
        ):
            attempts = int(row["failed_attempts"]) + 1
            locked_until = None
            if attempts >= MAX_FAILED_ATTEMPTS:
                locked_until = (_now() + timedelta(minutes=LOCKOUT_MINUTES)).isoformat()
                attempts = 0
            connection.execute(
                "UPDATE users SET failed_attempts = ?, locked_until = ? WHERE id = ?",
                (attempts, locked_until, row["id"]),
            )
            connection.commit()
            error = AuthError(
                f"Too many failed attempts. Locked for {LOCKOUT_MINUTES} minutes."
                if locked_until
                else INVALID_CREDENTIALS
            )
        else:
            connection.execute(
                "UPDATE users SET failed_attempts = 0, locked_until = NULL "
                "WHERE id = ?",
                (row["id"],),
            )
            connection.commit()
            return int(row["id"])
    finally:
        connection.close()
    raise error


def change_password(
    user_id: int,
    current_password: str,
    new_password: str,
    db_path: str | Path = DATABASE_PATH,
) -> None:
    """Replace a password after confirming the current one."""

    initialize_users(db_path)
    with _connect(db_path) as connection:
        row = connection.execute(
            "SELECT username, password_salt, password_hash FROM users WHERE id = ?",
            (int(user_id),),
        ).fetchone()
        if row is None:
            raise AuthError("Account not found.")
        if not _password_matches(
            current_password, row["password_salt"], row["password_hash"]
        ):
            raise AuthError("Current password is incorrect.")
        check_password_strength(new_password, row["username"])
        salt_hex, hash_hex = hash_password(new_password)
        connection.execute(
            "UPDATE users SET password_salt = ?, password_hash = ? WHERE id = ?",
            (salt_hex, hash_hex, int(user_id)),
        )


def user_count(db_path: str | Path = DATABASE_PATH) -> int:
    """How many accounts exist, used to word the first-run screen."""

    initialize_users(db_path)
    with _connect(db_path) as connection:
        return int(connection.execute("SELECT COUNT(*) FROM users").fetchone()[0])
